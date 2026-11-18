import json
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ==========================================
# Configuration
# ==========================================
METRICS_JSON_PATH = "all_pulse_metrics.json"
OUTPUT_SENSITIVITY_PNG = "thermoelectric_sensitivity_vs_log_activity.png"
OUTPUT_DELTA_E_PNG = "delta_e_vs_log_activity.png"

BG_COLOR = "#F4F5F7"

# Physical constants (CODATA) & baseline ambient temperature [deg C]
R_GAS = 8.314462618      # J / (mol * K)
F_FARADAY = 96485.33212  # C / mol
T_AMBIENT_C = 25.0       # Reference ambient temperature for Bates & Bower E^0(T)

SERIES_ORDER = ["1C", "5C", "10C"]

SERIES_STYLES: Dict[str, dict] = {
    "1C": {
        "label": r"$\Delta T \approx 1\,^\circ\mathrm{C}$",
        "point_color": "#111827",  # Near-Black
        "fit_color": "#C27803",    # Darker Amber/Gold
        "marker": "o",
        "linestyle": "-",
    },
    "5C": {
        "label": r"$\Delta T \approx 5\,^\circ\mathrm{C}$",
        "point_color": "#111827",  # Near-Black
        "fit_color": "#F5A623",    # Base Golden-Amber (matched to Temperature trace)
        "marker": "s",
        "linestyle": "--",
    },
    "10C": {
        "label": r"$\Delta T \approx 10\,^\circ\mathrm{C}$",
        "point_color": "#111827",  # Near-Black
        "fit_color": "#FCD34D",    # Lighter Golden-Amber
        "marker": "^",
        "linestyle": "-.",
    },
}


def bates_bower_e0_v(temp_c: np.ndarray) -> np.ndarray:
    """
    Standard potential E^0 [V] of the Ag|AgCl electrode from 0 to 90 deg C
    (Bates & Bower, J. Res. NBS 53, 283, 1954, Eq. 4).
    """
    return (
        0.23659
        - 4.8564e-4 * temp_c
        - 3.4205e-6 * (temp_c**2)
        + 5.869e-9 * (temp_c**3)
    )


def bates_bower_delta_e0_mv(delta_t: np.ndarray, t_amb_c: float = T_AMBIENT_C) -> np.ndarray:
    """
    Theoretical thermal shift in standard potential Delta E^0(Delta T) [mV]
    relative to ambient temperature t_amb_c, using Bates & Bower (1954) Eq. 4.
    """
    e0_hot = bates_bower_e0_v(t_amb_c + delta_t)
    e0_amb = bates_bower_e0_v(np.asarray(t_amb_c, dtype=float))
    return (e0_hot - e0_amb) * 1000.0


def compute_series_arrays(
    step_summaries: dict,
    mode: str = "sensitivity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts log10(activity), Y values, Y standard deviations, Delta T means,
    and Delta T standard deviations sorted by step index.
    """
    sorted_keys = sorted(step_summaries.keys(), key=lambda k: int(k))

    x_list = []
    y_list = []
    y_err_list = []
    dt_mean_list = []
    dt_std_list = []

    for k in sorted_keys:
        entry = step_summaries[k]
        log_a = float(entry["log_activity"])
        de_mean = float(entry["delta_e_mean_mv"])
        de_std = float(entry["delta_e_std_mv"])
        dt_mean = float(entry["delta_t_mean_c"])
        dt_std = float(entry["delta_t_std_c"])

        if mode == "sensitivity":
            if dt_mean == 0.0:
                continue
            val = de_mean / dt_mean
            rel_err_e = (de_std / de_mean) if de_mean != 0.0 else 0.0
            rel_err_t = (dt_std / dt_mean) if dt_mean != 0.0 else 0.0
            val_std = abs(val) * np.sqrt(rel_err_e**2 + rel_err_t**2)
        elif mode == "delta_e":
            val = de_mean
            val_std = de_std
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        x_list.append(log_a)
        y_list.append(val)
        y_err_list.append(val_std)
        dt_mean_list.append(dt_mean)
        dt_std_list.append(dt_std)

    return (
        np.array(x_list, dtype=float),
        np.array(y_list, dtype=float),
        np.array(y_err_list, dtype=float),
        np.array(dt_mean_list, dtype=float),
        np.array(dt_std_list, dtype=float),
    )


def linear_fit_with_r2(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Performs an ordinary least-squares linear fit y = m*x + b.
    Returns (slope, slope_se, intercept, intercept_se, r_squared, y_pred, residuals).
    """
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    residuals = y - y_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 1.0

    # Standard errors of OLS slope and intercept
    n = len(x)
    sxx = np.sum((x - np.mean(x)) ** 2)
    if n > 2 and sxx > 0.0:
        s2 = ss_res / (n - 2)
        slope_se = float(np.sqrt(s2 / sxx))
        intercept_se = float(np.sqrt(s2 * (1.0 / n + (np.mean(x) ** 2) / sxx)))
    else:
        slope_se = 0.0
        intercept_se = 0.0

    return (
        float(slope),
        slope_se,
        float(intercept),
        intercept_se,
        float(r_squared),
        y_pred,
        residuals,
    )


def generate_fit_and_residual_figures(
    all_metrics: dict,
    output_path: str,
    mode: str,
    title: str,
    y_label: str,
    res_y_label: str,
    int_y_label: str,
    y_fmt: str = "%.3f",
    hollow_markers: bool = False,
    jitter_step: float = 0.0,
) -> None:
    """
    Generates three standalone figures:
    1. Main Fit Figure: Solid black markers + wide STDEV error bars
       + OLS linear fit lines (shades of golden-amber) + upper-right text labels.
    2. Residuals Figure: Standalone grouped bar plot of linear-fit residuals
       with STDEV error bars.
    3. Intercepts vs. Theoretical Temperature Figure: Plots the 3 linear-fit intercepts
       against theoretical Delta T = |S| * F / (ln(10) * R), with a dashed black OLS linear fit,
       the negated Bates & Bower (1954) -Delta E^0 curve, and a 0.6 mV/deg C linear reference model.
    """
    plt.rcdefaults()

    base_name, ext = os.path.splitext(output_path)
    residual_output_path = f"{base_name}_residuals{ext}"
    intercept_output_path = f"{base_name}_intercepts{ext}"

    fig_main, ax_main = plt.subplots(figsize=(10.0, 6.2), facecolor=BG_COLOR)
    ax_main.set_facecolor(BG_COLOR)

    fig_res, ax_res = plt.subplots(figsize=(10.0, 4.5), facecolor=BG_COLOR)
    ax_res.set_facecolor(BG_COLOR)

    fig_int, ax_int = plt.subplots(figsize=(8.8, 5.6), facecolor=BG_COLOR)
    ax_int.set_facecolor(BG_COLOR)

    x_min_global, x_max_global = np.inf, -np.inf
    parsed_series = {}

    for series_key in SERIES_ORDER:
        if series_key not in all_metrics:
            print(f"Warning: '{series_key}' not found in metrics JSON, skipping.")
            continue
        x, y, y_err, dt_means, dt_stds = compute_series_arrays(
            all_metrics[series_key]["step_summaries"], mode=mode
        )
        parsed_series[series_key] = (x, y, y_err, dt_means, dt_stds)
        x_min_global = min(x_min_global, float(np.min(x)))
        x_max_global = max(x_max_global, float(np.max(x)))

    x_pad = 0.08 * (x_max_global - x_min_global) if x_max_global > x_min_global else 0.1
    x_fit_grid = np.linspace(x_min_global - x_pad, x_max_global + x_pad, 200)

    active_keys = [k for k in SERIES_ORDER if k in parsed_series]
    n_series = len(active_keys)
    sample_x = parsed_series[active_keys[0]][0]
    min_dx = float(np.min(np.diff(np.sort(sample_x)))) if len(sample_x) > 1 else 0.1
    bar_width = (min_dx * 0.72) / max(n_series, 1)

    fit_annotations = []
    ax_res.axhline(0.0, color="#111827", linestyle="-", linewidth=1.0, alpha=0.8, zorder=2)

    # Storage for Figure 3 (Intercepts vs. Theoretical Delta T from slope)
    theo_temp_vals = []
    theo_temp_errs = []
    intercept_vals = []
    intercept_errs = []
    series_markers = []

    # Conversion factor from slope [mV/decade] to Delta T [K or deg C]:
    # S [V/decade] = (S_mV * 1e-3), Delta T = |S_V| * F / (ln(10) * R)
    nernst_inv_factor = (1e-3 * F_FARADAY) / (np.log(10.0) * R_GAS)

    for idx, series_key in enumerate(active_keys):
        x, y, y_err, dt_means, dt_stds = parsed_series[series_key]
        style = SERIES_STYLES[series_key]
        point_color = style["point_color"]
        fit_color = style["fit_color"]
        marker = style["marker"]
        linestyle = style["linestyle"]
        base_label = style["label"]

        slope, slope_se, intercept, intercept_se, r2, _, residuals = linear_fit_with_r2(x, y)
        y_fit_grid = slope * x_fit_grid + intercept

        # Theoretical Delta T calculated from the linear fit slope S
        delta_t_theo = abs(slope) * nernst_inv_factor
        delta_t_theo_se = slope_se * nernst_inv_factor

        theo_temp_vals.append(delta_t_theo)
        theo_temp_errs.append(delta_t_theo_se)
        intercept_vals.append(intercept)
        intercept_errs.append(intercept_se)
        series_markers.append(marker)

        sign_str = "+" if intercept >= 0 else "-"
        eq_str = (
            f"{base_label}: $\Delta\ E = {slope:.2f}" + r"\ \log_{10}(a_{\mathrm{Cl}^-})" + f"{sign_str} {abs(intercept):.2f}$  "
            f"($R^2 = {r2:.2f}$)"
        )
        fit_annotations.append((fit_color, eq_str))

        x_plot = x + (idx - (n_series - 1) / 2.0) * jitter_step

        # Figure 1: Linear Fit Line
        ax_main.plot(
            x_fit_grid,
            y_fit_grid,
            color=fit_color,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=0.95,
            zorder=3,
        )

        # Figure 1: Solid black markers + wider error bar caps
        mfc = BG_COLOR if hollow_markers else point_color
        ax_main.errorbar(
            x_plot,
            y,
            yerr=y_err,
            fmt=marker,
            color=point_color,
            ecolor=point_color,
            markerfacecolor=mfc,
            markeredgecolor=point_color,
            markeredgewidth=1.0,
            markersize=4.5,
            elinewidth=1.6,
            capsize=6.5,
            capthick=1.6,
            linestyle="none",
            zorder=5,
        )

        # Figure 2: Grouped Bar Plot of Residuals
        bar_offset = (idx - (n_series - 1) / 2.0) * bar_width
        ax_res.bar(
            x + bar_offset,
            residuals,
            width=bar_width * 0.92,
            color=fit_color,
            edgecolor="#111827",
            linewidth=0.6,
            alpha=0.9,
            yerr=y_err,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 1.3,
                "capsize": 3.5,
                "capthick": 1.3,
            },
            label=base_label,
            zorder=3,
        )

    # Display Linear Fit Equations & R^2 Labels in Upper Right of Main Figure
    box_x, box_y_start = 0.97, 0.96
    line_spacing = 0.062
    ax_main.text(
        box_x,
        box_y_start + 0.01,
        r"$\Delta\ E = -\Delta T\ \frac{2.303\ R}{F}\ \log_{10}(a_{\mathrm{Cl}^-}) + E^{\circ}(\Delta\ T)$:",
        transform=ax_main.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
        va="top",
        ha="right",
    )
    for idx, (fit_color, eq_text) in enumerate(fit_annotations):
        ax_main.text(
            box_x,
            box_y_start - (idx + 1) * line_spacing,
            eq_text,
            transform=ax_main.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=fit_color,
            va="top",
            ha="right",
        )

    # ==========================================
    # Formatting - Figure 1 (Main Linear Fit)
    # ==========================================
    ax_main.set_xlabel(
        r"$\log_{10}(a_{\mathrm{Cl}^-})$",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )
    ax_main.set_ylabel(y_label, fontsize=13, fontweight="bold", color="#111827")
    ax_main.set_title(title, fontsize=14, fontweight="bold", color="#111827", pad=12)
    ax_main.grid(True, linestyle="--", color="#cbd5e1", alpha=0.75)
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))
    ax_main.set_xlim(x_min_global - x_pad, x_max_global + x_pad)

    fig_main.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig_main.get_facecolor(),
    )
    print(f"Saved main figure to {output_path}")

    # ==========================================
    # Formatting - Figure 2 (Standalone Residuals)
    # ==========================================
    ax_res.set_xlabel(
        r"$\log_{10}(a_{\mathrm{Cl}^-})$",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )
    ax_res.set_ylabel(res_y_label, fontsize=12.5, fontweight="bold", color="#111827")
    ax_res.set_title(
        f"Linear Fit Residuals — {title}",
        fontsize=13.5,
        fontweight="bold",
        color="#111827",
        pad=12,
    )
    ax_res.grid(True, linestyle="--", color="#cbd5e1", alpha=0.75, zorder=0)
    leg_res = ax_res.legend(loc="best", frameon=True, fontsize=9.5, ncol=n_series)
    leg_res.get_frame().set_facecolor("#ffffff")
    leg_res.get_frame().set_alpha(0.88)
    ax_res.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))
    ax_res.set_xlim(x_min_global - x_pad, x_max_global + x_pad)

    fig_res.savefig(
        residual_output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig_res.get_facecolor(),
    )
    print(f"Saved residuals figure to {residual_output_path}")

    # ==========================================
    # Figure 3: Intercepts vs. Theoretical Delta T (from Slope S)
    # ==========================================
    t_arr = np.array(theo_temp_vals, dtype=float)
    t_err_arr = np.array(theo_temp_errs, dtype=float)
    b_arr = np.array(intercept_vals, dtype=float)
    b_err_arr = np.array(intercept_errs, dtype=float)

    t_slope, _, t_intercept, _, t_r2, _, _ = linear_fit_with_r2(t_arr, b_arr)

    t_min, t_max = float(np.min(t_arr)), float(np.max(t_arr))
    t_pad = 0.12 * (t_max - t_min) if t_max > t_min else 0.5
    t_fit_grid = np.linspace(max(0.0, t_min - t_pad), t_max + t_pad, 200)
    b_fit_grid = t_slope * t_fit_grid + t_intercept

    # Negated Bates & Bower (1954) E^0 thermal shift curve: -Delta E^0 [mV]
    neg_e0_theo_grid = -bates_bower_delta_e0_mv(t_fit_grid, t_amb_c=T_AMBIENT_C)

    # Simple 0.6 mV/deg C linear model
    linear_06_grid = 0.6 * t_fit_grid

    # Plot OLS fit line in dashed black
    ax_int.plot(
        t_fit_grid,
        b_fit_grid,
        color="#111827",
        linestyle="--",
        linewidth=2.2,
        alpha=0.95,
        label="Experimental Linear Fit",
        zorder=3,
    )

    # Plot -Delta E^0 Bates & Bower curve in warm golden-amber
    ax_int.plot(
        t_fit_grid,
        neg_e0_theo_grid,
        color="#F5A623",
        linestyle="-",
        linewidth=2.2,
        alpha=0.95,
        label=rf"Bates & Bower (1954) $-\Delta E^\circ$ ($T_0={T_AMBIENT_C:.0f}\,^\circ\mathrm{{C}}$)",
        zorder=4,
    )

    # Plot simple 0.6 mV/deg C linear model in darker amber dash-dot
    ax_int.plot(
        t_fit_grid,
        linear_06_grid,
        color="#C27803",
        linestyle="-.",
        linewidth=2.2,
        alpha=0.95,
        label=r"Simple Linear Model ($0.6\ \mathrm{mV/^\circ C}$)",
        zorder=4,
    )

    # Plot each series' intercept point with matching marker shape & error bars
    mfc = BG_COLOR if hollow_markers else "#111827"
    for t_val, t_err, b_val, b_err, mkr in zip(
        t_arr, t_err_arr, b_arr, b_err_arr, series_markers
    ):
        ax_int.errorbar(
            t_val,
            b_val,
            xerr=t_err,
            yerr=b_err,
            fmt=mkr,
            color="#111827",
            ecolor="#111827",
            markerfacecolor=mfc,
            markeredgecolor="#111827",
            markeredgewidth=1.0,
            markersize=5.2,
            elinewidth=1.6,
            capsize=6.0,
            capthick=1.6,
            linestyle="none",
            zorder=5,
        )

    # Display Linear Fit Equation & R^2 in Upper Left (or Upper Right if negative slope)
    t_sign_str = "+" if t_intercept >= 0 else "-"
    t_eq_str = (
        f"$b = {t_slope:.4f}\\,\\Delta T_{{S}} {t_sign_str} {abs(t_intercept):.4f}$\n"
        f"$R^2 = {t_r2:.4f}$"
    )
    text_x, text_ha = (0.04, "left") if t_slope > 0 else (0.96, "right")
    ax_int.text(
        text_x,
        0.95,
        t_eq_str,
        transform=ax_int.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#111827",
        va="top",
        ha=text_ha,
    )

    ax_int.set_xlabel(
        r"Theoretical Temperature Difference, $\Delta T = |S|\frac{F}{\ln(10)\,R}\ [^\circ\mathrm{C}]$",
        fontsize=12.5,
        fontweight="bold",
        color="#111827",
    )
    ax_int.set_ylabel(int_y_label, fontsize=13, fontweight="bold", color="#111827")
    ax_int.set_title(
        f"Linear Fit Intercepts vs. Theoretical $\\Delta T$ — {title}",
        fontsize=13.5,
        fontweight="bold",
        color="#111827",
        pad=12,
    )
    ax_int.grid(True, linestyle="--", color="#cbd5e1", alpha=0.75)
    leg_int = ax_int.legend(loc="lower right", frameon=True, fontsize=9.5)
    leg_int.get_frame().set_facecolor("#ffffff")
    leg_int.get_frame().set_alpha(0.88)
    ax_int.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))
    ax_int.set_xlim(max(0.0, t_min - t_pad), t_max + t_pad)

    fig_int.savefig(
        intercept_output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig_int.get_facecolor(),
    )
    print(f"Saved intercepts figure to {intercept_output_path}")

    plt.show()
    plt.close(fig_main)
    plt.close(fig_res)
    plt.close(fig_int)


if __name__ == "__main__":
    if not os.path.exists(METRICS_JSON_PATH):
        raise FileNotFoundError(
            f"Could not find '{METRICS_JSON_PATH}'. Run pulseCalc.py first to generate it."
        )

    with open(METRICS_JSON_PATH, "r") as f:
        metrics_data = json.load(f)

    # # Figure 1: Delta E / Delta T vs. log10(activity)
    # generate_fit_and_residual_figures(
    #     all_metrics=metrics_data,
    #     output_path=OUTPUT_SENSITIVITY_PNG,
    #     mode="sensitivity",
    #     title=r"Thermoelectric Sensitivity ($\Delta E / \Delta T$) vs. $\log_{10}(a_{\mathrm{Cl}^-})$",
    #     y_label=r"$\Delta E / \Delta T\ [\mathrm{mV \cdot K^{-1}}]$",
    #     res_y_label=r"Residuals $[\mathrm{mV \cdot K^{-1}}]$",
    #     int_y_label=r"Intercept, $b\ [\mathrm{mV \cdot K^{-1}}]$",
    #     y_fmt="%.3f",
    #     hollow_markers=False,
    #     jitter_step=0.012,
    # )

    # Figure 2: Delta E vs. log10(activity)
    generate_fit_and_residual_figures(
        all_metrics=metrics_data,
        output_path=OUTPUT_DELTA_E_PNG,
        mode="delta_e",
        title=r"Thermoelectric Potential ($\Delta E$) vs. $\log_{10}(a_{\mathrm{Cl}^-})$",
        y_label=r"$\Delta E\ [\mathrm{mV}]$",
        res_y_label=r"Residuals $[\mathrm{mV}]$",
        int_y_label=r"Intercept, $b\ [\mathrm{mV}]$",
        y_fmt="%.2f",
        hollow_markers=False,
        jitter_step=0.0,
    )