import json
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np

# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
# Set to True to include the Nominal Ideal curves and the interactive widget.
# Set to False to strictly show only Experimental Points + Adjusted Ideal lines.
SHOW_NOMINAL_IDEAL = False

GAIN_FACTOR = 5.0  # Internal gain scaling factor

# Physical Constants
R = 8.314462618  # Gas constant in J/(mol·K)
F = 96485.3321  # Faraday constant in C/mol

# Base Reference Temperature (Kelvin) for T_mean calculation
T_REF_K = 298.15  # 25 °C

# ------------------------------------------------------------------------------
# MANUAL USER CONSTANTS (UNITS: mV/K)
# ------------------------------------------------------------------------------
# Standard electrode potential temperature coefficient (dE°/dT) in mV/K
E_STD_TEMP_COEFF = -0.00065 * 1000  # Default: -0.65 mV/K

# Pure solution/junction Seebeck coefficient term constant in mV
SEEBECK_COEFF = 0.0015 * 1000  # Default: +11.23 mV (evaluated as 11.23/T_mean)
# ------------------------------------------------------------------------------

# Theoretical Nernstian slope per Kelvin for decadal log10(a): [mV/K/dec]
nernst_slope_per_K = (R / F) * np.log(10) * 1000

# ==============================================================================
# DATA LOADING & PREPARATION
# ==============================================================================
with open("titration_steps.json", "r") as f:
    json_data = json.load(f)

category_mapping = [
    {
        "json_key": "Small (~1°C)",
        "dT_nominal": 1,
        "label": "1°C PID Setpoint",
        "color": "#FDB863",
        "marker": "o",
    },
    {
        "json_key": "Medium (~5°C)",
        "dT_nominal": 5,
        "label": "5°C PID Setpoint",
        "color": "#E66101",
        "marker": "s",
    },
    {
        "json_key": "Large (~10°C)",
        "dT_nominal": 10,
        "label": "10°C PID Setpoint",
        "color": "#B33B00",
        "marker": "^",
    },
]

# Extract activities & log activity values
activities = np.array([step["activity"] for step in json_data])
x_data = np.log10(activities)  # log10(a_Cl-)
ln_a = np.log(activities)  # ln(a_Cl-)
x_fit = np.linspace(min(x_data), max(x_data), 100)

# Lists to store offsets across categories for comparison plotting
dt_fits = []
exp_offsets = []
theo_offsets = []

# ==============================================================================
# FIGURE 1: MAIN & RESIDUAL PLOTS
# ==============================================================================
fig1, (ax_main, ax_res) = plt.subplots(
    2,
    1,
    figsize=(11, 8.5),
    dpi=100,
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

if SHOW_NOMINAL_IDEAL:
    plt.subplots_adjust(left=0.20, right=0.92)
else:
    plt.subplots_adjust(left=0.10, right=0.92)

nominal_ideal_lines = []

bar_width = 0.035
num_categories = len(category_mapping)
offsets = np.linspace(
    -(num_categories - 1) * bar_width / 2,
    (num_categories - 1) * bar_width / 2,
    num_categories,
)

for idx, cat in enumerate(category_mapping):
    key = cat["json_key"]
    dT_nom = cat["dT_nominal"]
    color = cat["color"]
    marker = cat["marker"]
    label_str = cat["label"]
    offset = offsets[idx]

    # --- A. Adjusted Experimental Data (-CP / 5) ---
    y_exp_raw = np.array(
        [
            step["pulse_analysis"]["categories"][key]["delta_CP_mean_mV"]
            for step in json_data
        ]
    )
    y_err_raw = np.array(
        [
            step["pulse_analysis"]["categories"][key]["delta_CP_std_mV"]
            for step in json_data
        ]
    )

    y_exp = -1.0 * (y_exp_raw / GAIN_FACTOR)
    y_err = y_err_raw / GAIN_FACTOR

    # Get exp slope/intercept
    m_exp, c_exp = np.polyfit(x_data, y_exp, 1)

    # Compute Coefficient of Determination (R^2)
    y_pred = m_exp * x_data + c_exp
    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    # Slope-matched delta T
    dT_fit = m_exp / nernst_slope_per_K

    # Calculate Theoretical Offset using user-defined constants
    T_mean = T_REF_K + (dT_fit / 2.0)
    b_theo = (E_STD_TEMP_COEFF + (SEEBECK_COEFF / T_mean)) * dT_fit

    # Collect offset values for Figure 2
    dt_fits.append(dT_fit)
    exp_offsets.append(c_exp)
    theo_offsets.append(b_theo)

    # Plot Experimental points
    ax_main.errorbar(
        x_data,
        y_exp,
        yerr=y_err,
        fmt=marker,
        color=color,
        ecolor=color,
        capsize=3,
        capthick=1,
        markersize=6,
        linestyle="None",
        label=f"Exp Data ({label_str})",
        zorder=4,
    )

    # --- B. Nominal Ideal Series ---
    delta_E_ideal_mV = dT_nom * (R / F) * ln_a * 1000
    m_ideal, c_ideal = np.polyfit(x_data, delta_E_ideal_mV, 1)

    label_ideal = (
        rf"Ideal {dT_nom} °C: $y = {m_ideal:.3f}x + {c_ideal:.3f}\text{{ mV}}$"
        if SHOW_NOMINAL_IDEAL
        else "_nolegend_"
    )

    (line_nom,) = ax_main.plot(
        x_fit,
        m_ideal * x_fit + c_ideal,
        color=color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=label_ideal,
        visible=SHOW_NOMINAL_IDEAL,
        zorder=2,
    )
    line_nom._ideal_label = (
        rf"Ideal {dT_nom} °C: $y = {m_ideal:.3f}x + {c_ideal:.3f}\text{{ mV}}$"
    )
    nominal_ideal_lines.append(line_nom)

    # --- C. Adjusted Ideal Series ---
    y_fit_line = m_exp * x_fit + c_exp
    label_fit_dT = rf"Adjusted Ideal ($\Delta T = {dT_fit:.2f}\text{{ K}}$, $R^2 = {r_squared:.4f}$): $y = {m_exp:.3f}x + {c_exp:.3f}\text{{ mV}}$"

    ax_main.plot(
        x_fit,
        y_fit_line,
        color=color,
        linestyle=":",
        linewidth=2.0,
        label=label_fit_dT,
        zorder=3,
    )

    # --- D. Midpoint Annotation Above Line ---
    mid_idx = len(x_fit) // 2
    x_mid = x_fit[mid_idx]
    y_mid = y_fit_line[mid_idx]

    ax_main.annotate(
        rf"$\Delta T = {dT_fit:.2f}\text{{ °C}}$",
        xy=(x_mid, y_mid),
        xytext=(-80, 40),
        textcoords="offset points",
        color=color,
        fontsize=12,
        weight="bold",
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
        ),
        zorder=5,
    )

    # --- E. Residuals Bar Chart ---
    y_adj_ideal_at_pts = m_exp * x_data + c_exp
    residuals = y_exp - y_adj_ideal_at_pts
    x_bars = x_data + offset

    ax_res.bar(
        x_bars,
        residuals,
        width=bar_width,
        color=color,
        alpha=0.7,
        edgecolor=color,
        linewidth=1,
        yerr=y_err,
        capsize=2,
        label=label_str,
    )

ax_main.invert_yaxis()
ax_main.set_ylabel(r"$\Delta E$ / Adjusted $\Delta CP$ (mV)", fontsize=11)
ax_main.set_title(
    r"Experimental Data Points vs. Ideal Nernstian Curves",
    fontsize=12,
    pad=10,
)
ax_main.grid(True, linestyle=":", alpha=0.6)
ax_main.legend(fontsize=8, loc="upper right", framealpha=0.9)

ax_res.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.8)
ax_res.set_xlabel(r"Decadal Activity $\log_{10}(a_{\text{Cl}^-})$", fontsize=11)
ax_res.set_ylabel(r"Residuals (mV)", fontsize=10)
ax_res.grid(True, linestyle=":", alpha=0.6)

if SHOW_NOMINAL_IDEAL:
    rax = plt.axes([0.02, 0.5, 0.14, 0.12])
    check = CheckButtons(rax, ["Show Nominal Ideal"], [SHOW_NOMINAL_IDEAL])

    def toggle_nominal(label):
        visible = check.get_status()[0]
        for line in nominal_ideal_lines:
            line.set_visible(visible)
            line.set_label(line._ideal_label if visible else "_nolegend_")
        ax_main.legend(fontsize=8, loc="upper right", framealpha=0.9)
        fig1.canvas.draw_idle()

    check.on_clicked(toggle_nominal)

# ==============================================================================
# FIGURE 2: OFFSET COMPARISON & EXP LINEAR FIT
# ==============================================================================
fig2, ax_offset = plt.subplots(figsize=(8.5, 5.5), dpi=100)

dt_fits = np.array(dt_fits)
exp_offsets = np.array(exp_offsets)
theo_offsets = np.array(theo_offsets)

# --- 1. Linear Fit for Experimental Offsets vs. dT ---
m_exp_offset, c_exp_offset = np.polyfit(dt_fits, exp_offsets, 1)

# R^2 for Experimental Offset Fit
y_exp_offset_pred = m_exp_offset * dt_fits + c_exp_offset
ss_res_off = np.sum((exp_offsets - y_exp_offset_pred) ** 2)
ss_tot_off = np.sum((exp_offsets - np.mean(exp_offsets)) ** 2)
r2_exp_offset = 1 - (ss_res_off / ss_tot_off) if ss_tot_off != 0 else 1.0

# --- 2. Theoretical Continuous Curve ---
dt_continuous = np.linspace(0, max(dt_fits) * 1.15, 200)
T_mean_cont = T_REF_K + (dt_continuous / 2.0)
b_theo_continuous = (
    E_STD_TEMP_COEFF + (SEEBECK_COEFF / T_mean_cont)
) * dt_continuous

# Plot Theoretical Model
ax_offset.plot(
    dt_continuous,
    b_theo_continuous,
    color="navy",
    linestyle="--",
    linewidth=2,
    label=rf"Theoretical Model ($E_{{std}} = {E_STD_TEMP_COEFF:.2f}\text{{ mV/K}}, S_{{sol}} = \frac{{{SEEBECK_COEFF:.2f}}}{{T_{{mean}}}}\text{{ mV/K}}$)",
)

ax_offset.scatter(
    dt_fits,
    theo_offsets,
    color="navy",
    s=50,
    facecolors="none",
    edgecolors="navy",
    linewidth=1.5,
    zorder=3,
    label="Theoretical Points @ Exp $\Delta T$",
)

# Plot Experimental Fit Trendline
exp_fit_line = m_exp_offset * dt_continuous + c_exp_offset
ax_offset.plot(
    dt_continuous,
    exp_fit_line,
    color="crimson",
    linestyle="-",
    linewidth=2,
    label=rf"Exp Offset Fit: $y = {m_exp_offset:.3f}\Delta T + {c_exp_offset:.3f}\text{{ mV}}$ ($R^2 = {r2_exp_offset:.4f}$)",
    zorder=2,
)

# Plot Experimental Data Points
ax_offset.scatter(
    dt_fits,
    exp_offsets,
    color="crimson",
    s=70,
    marker="D",
    zorder=4,
    label="Experimental Fits Intercept ($c_{exp}$)",
)

# Annotate Data Points
for dt_i, exp_i, theo_i in zip(dt_fits, exp_offsets, theo_offsets):
    ax_offset.annotate(
        f"Exp: {exp_i:.2f} mV\nTheo: {theo_i:.2f} mV",
        xy=(dt_i, exp_i),
        xytext=(10, -15),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="lightyellow",
            edgecolor="gray",
            alpha=0.8,
        ),
    )

ax_offset.set_xlabel(r"Temperature Difference $\Delta T$ (K or °C)", fontsize=11)
ax_offset.set_ylabel(r"Linear Fit Offset $b$ (mV)", fontsize=11)
ax_offset.set_title(
    r"Linear Fit Intercept (Offset) vs. $\Delta T$: Experimental Fit vs. Model",
    fontsize=12,
    pad=10,
)
ax_offset.grid(True, linestyle=":", alpha=0.6)
ax_offset.legend(fontsize=8.5, loc="best", framealpha=0.9)

plt.tight_layout()
plt.show()