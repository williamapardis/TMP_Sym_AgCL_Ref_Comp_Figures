import json
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
json_path = "titration_steps.json"

# Toggle X-Axis: Set to 'concentration' or 'activity'
x_axis_mode = "activity"  # Options: 'concentration' | 'activity'

# X-Axis scale toggle: True for logarithmic fit & display, False for linear fit
use_log_scale = True

# Step Selection Toggle:
# - Range tuple/list [start, end]: e.g., (0, 7) or [0, 7] (inclusive)
# - Specific list: e.g., [0, 1, 3, 5, 7]
# - None: include all steps
STEPS_TO_INCLUDE = (0, 7)

categories_to_plot = [
    "Small (~1°C)",
    "Medium (~5°C)",
    "Large (~10°C)",
]

# Style map using shades of Orange/Coral
cat_styles = {
    "Small (~1°C)": {
        "color": "#FDB863",  # Light Orange
        "marker": "o",
        "label": "Small (~1°C)",
    },
    "Medium (~5°C)": {
        "color": "#E66101",  # Medium Orange
        "marker": "s",
        "label": "Medium (~5°C)",
    },
    "Large (~10°C)": {
        "color": "#B33B00",  # Dark Rust / Deep Orange
        "marker": "^",
        "label": "Large (~10°C)",
    },
}

x_key_map = {
    "concentration": "concentration_M",
    "activity": "activity",
}

# ==========================================
# DATA LOADING & PARSING
# ==========================================
with open(json_path, "r") as f:
    data = json.load(f)

plot_data = {
    cat: {"x": [], "y": [], "yerr": []} for cat in categories_to_plot
}
lj_data = {"x": [], "y": [], "yerr": []}

json_x_key = x_key_map.get(x_axis_mode, "concentration_M")

# Parse active step indices based on configuration
for step_idx, entry in enumerate(data):
    # Step filtering logic
    if STEPS_TO_INCLUDE is not None:
        if (
            isinstance(STEPS_TO_INCLUDE, (tuple, list))
            and len(STEPS_TO_INCLUDE) == 2
            and not isinstance(STEPS_TO_INCLUDE[0], list)
        ):
            start_step, end_step = STEPS_TO_INCLUDE
            if not (start_step <= step_idx <= end_step):
                continue
        elif isinstance(STEPS_TO_INCLUDE, (list, tuple, set)):
            if step_idx not in STEPS_TO_INCLUDE:
                continue

    x_val = entry.get(json_x_key)
    if x_val is None:
        continue

    # Extract L.J. CP data
    lj_mean = entry.get("lj_cp_mean_mV")
    lj_std = entry.get("lj_cp_std_mV", 0.0)
    if lj_mean is not None:
        lj_data["x"].append(x_val)
        lj_data["y"].append(lj_mean)
        lj_data["yerr"].append(lj_std if lj_std is not None else 0.0)

    # Extract Pulse categories
    pulse_analysis = entry.get("pulse_analysis", {})
    categories_data = pulse_analysis.get("categories", {})

    for cat_name in categories_to_plot:
        if cat_name in categories_data:
            cat_stats = categories_data[cat_name]
            y_mean = cat_stats.get("delta_CP_mean_mV")
            y_std = cat_stats.get("delta_CP_std_mV")

            if y_mean is not None:
                plot_data[cat_name]["x"].append(x_val)
                plot_data[cat_name]["y"].append(y_mean)
                plot_data[cat_name]["yerr"].append(
                    y_std if y_std is not None else 0.0
                )

# ==========================================
# PLOTTING WITH SECONDARY Y-AXES
# ==========================================
fig, (ax_main, ax_res) = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

# Secondary Y-Axes for L.J. CP
ax_main_lj = ax_main.twinx()
ax_res_lj = ax_res.twinx()

x_label_str = (
    "Concentration [M]" if x_axis_mode == "concentration" else "Activity"
)

processed_results = {}
main_handles, main_labels = [], []

# Track max residual magnitude to align y=0
max_pulse_res_val = 0.0
max_lj_res_val = 0.0

# ------------------------------------------
# 1. FIT & PLOT PULSE CATEGORIES (LEFT Y-AXIS)
# ------------------------------------------
for cat_name, pdata in plot_data.items():
    if not pdata["x"]:
        continue

    sorted_indices = np.argsort(pdata["x"])
    x_data = np.array(pdata["x"])[sorted_indices]
    y_data = np.array(pdata["y"])[sorted_indices]
    yerr_data = np.array(pdata["yerr"])[sorted_indices]

    style = cat_styles[cat_name]

    # Fit calculation
    x_fit_transform = np.log10(x_data) if use_log_scale else x_data
    slope, intercept = np.polyfit(x_fit_transform, y_data, 1)
    y_pred = slope * x_fit_transform + intercept
    residuals = y_data - y_pred

    # Update max magnitude for left axis scaling
    current_max = np.max(np.abs(residuals) + yerr_data)
    if current_max > max_pulse_res_val:
        max_pulse_res_val = current_max

    # R^2 calculation
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    processed_results[cat_name] = {
        "x": x_data,
        "residuals": residuals,
        "yerr": yerr_data,
        "style": style,
    }

    x_var_symbol = r"\log_{10}(X)" if use_log_scale else "X"
    equation_str = (
        f"{style['label']}\n"
        f"  $y = {slope:.2f} \cdot {x_var_symbol} + {intercept:.2f}$ | $R^2 = {r_squared:.4f}$"
    )

    eb = ax_main.errorbar(
        x_data,
        y_data,
        yerr=yerr_data,
        fmt=style["marker"],
        color=style["color"],
        ecolor=style["color"],
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        alpha=0.9,
        label=equation_str,
    )

    x_dense = np.linspace(x_data.min(), x_data.max(), 100)
    x_dense_transform = np.log10(x_dense) if use_log_scale else x_dense
    y_dense_fit = slope * x_dense_transform + intercept

    ax_main.plot(
        x_dense,
        y_dense_fit,
        linestyle="--",
        color=style["color"],
        linewidth=1.5,
        alpha=0.8,
    )

    main_handles.append(eb)
    main_labels.append(equation_str)

# ------------------------------------------
# 2. FIT & PLOT L.J. CP (RIGHT Y-AXIS)
# ------------------------------------------
if lj_data["x"]:
    sorted_indices_lj = np.argsort(lj_data["x"])
    x_lj = np.array(lj_data["x"])[sorted_indices_lj]
    y_lj = np.array(lj_data["y"])[sorted_indices_lj]
    yerr_lj = np.array(lj_data["yerr"])[sorted_indices_lj]

    x_fit_lj = np.log10(x_lj) if use_log_scale else x_lj
    slope_lj, intercept_lj = np.polyfit(x_fit_lj, y_lj, 1)
    y_pred_lj = slope_lj * x_fit_lj + intercept_lj
    residuals_lj = y_lj - y_pred_lj

    max_lj_res_val = np.max(np.abs(residuals_lj) + yerr_lj)

    ss_res_lj = np.sum(residuals_lj**2)
    ss_tot_lj = np.sum((y_lj - np.mean(y_lj)) ** 2)
    r2_lj = 1 - (ss_res_lj / ss_tot_lj) if ss_tot_lj != 0 else 0.0

    lj_style = {"color": "navy", "marker": "D", "label": "L.J. CP"}
    processed_results["L.J. CP"] = {
        "x": x_lj,
        "residuals": residuals_lj,
        "yerr": yerr_lj,
        "style": lj_style,
    }

    eq_lj = (
        f"L.J. CP\n"
        f"  $y = {slope_lj:.2f} \cdot {x_var_symbol} + {intercept_lj:.2f}$ | $R^2 = {r2_lj:.4f}$"
    )

    eb_lj = ax_main_lj.errorbar(
        x_lj,
        y_lj,
        yerr=yerr_lj,
        fmt="D",
        color="navy",
        ecolor="navy",
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        alpha=0.85,
    )

    x_dense_lj = np.linspace(x_lj.min(), x_lj.max(), 100)
    x_dense_transform_lj = (
        np.log10(x_dense_lj) if use_log_scale else x_dense_lj
    )
    y_dense_fit_lj = slope_lj * x_dense_transform_lj + intercept_lj

    ax_main_lj.plot(
        x_dense_lj,
        y_dense_fit_lj,
        linestyle=":",
        color="navy",
        linewidth=1.8,
        alpha=0.85,
    )

    main_handles.append(eb_lj)
    main_labels.append(eq_lj)

# ------------------------------------------
# 3. RESIDUALS BAR SUBPLOT (BOTH Y-AXES)
# ------------------------------------------
num_cats = len(processed_results)
bar_offsets = np.linspace(-0.03, 0.03, num_cats)

for idx, (cat_name, res_data) in enumerate(processed_results.items()):
    x_vals = res_data["x"]
    res_vals = res_data["residuals"]
    err_vals = res_data["yerr"]
    style = res_data["style"]

    if use_log_scale:
        bar_x_positions = x_vals * (10 ** bar_offsets[idx])
        bar_widths = x_vals * 0.035
    else:
        bar_x_positions = x_vals + (idx - num_cats / 2) * 0.008
        bar_widths = 0.006

    target_ax = ax_res_lj if cat_name == "L.J. CP" else ax_res

    target_ax.bar(
        bar_x_positions,
        res_vals,
        width=bar_widths,
        yerr=err_vals,
        color=style["color"],
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
        alpha=0.8,
        label=style["label"],
    )

# Reference zero line for residuals
ax_res.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

# ------------------------------------------
# SYMMETRIC Y-LIMITS FOR ALIGNED ZERO
# ------------------------------------------
if max_pulse_res_val > 0:
    ax_res.set_ylim(-max_pulse_res_val * 1.15, max_pulse_res_val * 1.15)
if max_lj_res_val > 0:
    ax_res_lj.set_ylim(-max_lj_res_val * 1.15, max_lj_res_val * 1.15)

# ------------------------------------------
# AXIS & FORMATTING
# ------------------------------------------
if use_log_scale:
    ax_main.set_xscale("log")
    ax_res.set_xscale("log")

# Left Y-Axis labels
ax_main.set_ylabel(
    r"$\Delta$CP Mean ($W_2 - W_1$) [mV]", fontsize=11, color="#B33B00"
)
ax_res.set_ylabel("Pulse Res. [mV]", fontsize=10, color="#B33B00")
ax_main.tick_params(axis="y", labelcolor="#B33B00")
ax_res.tick_params(axis="y", labelcolor="#B33B00")

# Right Y-Axis labels
ax_main_lj.set_ylabel("L.J. CP Mean [mV]", fontsize=11, color="navy")
ax_res_lj.set_ylabel("L.J. Res. [mV]", fontsize=10, color="navy")
ax_main_lj.tick_params(axis="y", labelcolor="navy")
ax_res_lj.tick_params(axis="y", labelcolor="navy")

title_step_suffix = (
    f" (Steps: {STEPS_TO_INCLUDE})" if STEPS_TO_INCLUDE is not None else ""
)
ax_main.set_title(
    f"Pulse Response ($\Delta$CP) & L.J. CP vs {x_label_str}{title_step_suffix}",
    fontsize=13,
    fontweight="bold",
)
ax_main.grid(True, which="both", linestyle="--", alpha=0.5)
ax_res.grid(True, which="both", linestyle="--", alpha=0.5)

# Unified Legend inside main plot axes
ax_main.legend(
    main_handles,
    main_labels,
    title="Fit Equations & $R^2$",
    loc="upper right",
    fontsize=8,
    title_fontsize=9,
    framealpha=0.85,
    facecolor="white",
    edgecolor="gray",
)

ax_res.set_xlabel(x_label_str, fontsize=11)

plt.tight_layout()
plt.show()