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

SERIES_ORDER = ["1C", "5C", "10C"]

SERIES_STYLES: Dict[str, dict] = {
    "1C": {
        "label": r"$\Delta T \approx 1\,^\circ\mathrm{C}$",
        "color": "#2563eb",   # Royal Blue
        "marker": "o",
        "linestyle": "-",
    },
    "5C": {
        "label": r"$\Delta T \approx 5\,^\circ\mathrm{C}$",
        "color": "#f97316",   # Vivid Orange
        "marker": "s",
        "linestyle": "--",
    },
    "10C": {
        "label": r"$\Delta T \approx 10\,^\circ\mathrm{C}$",
        "color": "#dc2626",   # Crimson Red
        "marker": "^",
        "linestyle": "-.",
    },
}


def compute_series_arrays(
    step_summaries: dict,
    mode: str = "sensitivity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts log10(activity), Y values, and standard deviations sorted by step index.

    Parameters
    ----------
    step_summaries : dict
        Per-step dictionary from all_pulse_metrics.json.
    mode : str
        - "sensitivity": Y = Delta E / Delta T [mV/K], propagated STDEV.
        - "delta_e":     Y = Delta E [mV], direct delta_e_std_mv.
    """
    sorted_keys = sorted(step_summaries.keys(), key=lambda k: int(k))

    x_list = []
    y_list = []
    y_err_list = []

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

    return (
        np.array(x_list, dtype=float),
        np.array(y_list, dtype=float),
        np.array(y_err_list, dtype=float),
    )


def linear_fit_with_r2(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Performs an ordinary least-squares linear fit y = m*x + b.
    Returns (slope, intercept, r_squared, y_pred, residuals).
    """
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    residuals = y - y_pred

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 1.0

    return float(slope), float(intercept), float(r_squared), y_pred, residuals


def generate_fit_and_residual_figure(
    all_metrics: dict,
    output_path: str,
    mode: str,
    title: str,
    y_label: str,
    res_y_label: str,
    y_fmt: str = "%.3f",
    legend_loc: str = "lower left",
) -> None:
    """
    Generates a two-panel figure:
      - Top: Scatter points with STDEV error bars + OLS linear fit + equations & R^2.
      - Bottom: Grouped bar plot of linear-fit residuals with STDEV error bars.
    """
    plt.rcdefaults()
    fig, (ax_main, ax_res) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10.5, 8.2),
        gridspec_kw={"height_ratios": [3.0, 1.35], "hspace": 0.14},
    )

    x_min_global, x_max_global = np.inf, -np.inf
    parsed_series = {}

    for series_key in SERIES_ORDER:
        if series_key not in all_metrics:
            print(f"Warning: '{series_key}' not found in metrics JSON, skipping.")
            continue
        x, y, y_err = compute_series_arrays(
            all_metrics[series_key]["step_summaries"], mode=mode
        )
        parsed_series[series_key] = (x, y, y_err)
        x_min_global = min(x_min_global, float(np.min(x)))
        x_max_global = max(x_max_global, float(np.max(x)))

    x_pad = 0.08 * (x_max_global - x_min_global) if x_max_global > x_min_global else 0.1
    x_fit_grid = np.linspace(x_min_global - x_pad, x_max_global + x_pad, 200)

    # Calculate grouped bar width based on minimum spacing between adjacent x points
    active_keys = [k for k in SERIES_ORDER if k in parsed_series]
    n_series = len(active_keys)
    sample_x = parsed_series[active_keys[0]][0]
    min_dx = float(np.min(np.diff(np.sort(sample_x)))) if len(sample_x) > 1 else 0.1
    bar_width = (min_dx * 0.72) / max(n_series, 1)

    fit_annotations = []
    ax_res.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.8, zorder=2)

    for idx, series_key in enumerate(active_keys):
        x, y, y_err = parsed_series[series_key]
        style = SERIES_STYLES[series_key]
        color = style["color"]
        marker = style["marker"]
        linestyle = style["linestyle"]
        base_label = style["label"]

        slope, intercept, r2, _, residuals = linear_fit_with_r2(x, y)
        y_fit_grid = slope * x_fit_grid + intercept

        sign_str = "+" if intercept >= 0 else "-"
        eq_str = (
            f"{series_key}: $y = {slope:.4f}x {sign_str} {abs(intercept):.4f}$  "
            f"($R^2 = {r2:.4f}$)"
        )
        fit_annotations.append((color, eq_str))

        # Top Subplot: Data points with STDEV error bars
        ax_main.errorbar(
            x,
            y,
            yerr=y_err,
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=6.5,
            elinewidth=1.3,
            capsize=3.5,
            capthick=1.2,
            linestyle="none",
            label=f"{base_label} Data",
            zorder=4,
        )

        # Top Subplot: Linear Fit Line
        ax_main.plot(
            x_fit_grid,
            y_fit_grid,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            alpha=0.9,
            label=f"{base_label} Fit ($R^2={r2:.3f}$)",
            zorder=3,
        )

        # Bottom Subplot: Grouped Bar Plot of Residuals with STDEV error bars
        bar_offset = (idx - (n_series - 1) / 2.0) * bar_width
        ax_res.bar(
            x + bar_offset,
            residuals,
            width=bar_width * 0.92,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.82,
            yerr=y_err,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 0.9,
                "capsize": 2.0,
                "capthick": 0.9,
            },
            label=base_label,
            zorder=3,
        )

    # Display Linear Fit Equations & R^2 Box in Top Right of Top Subplot
    box_x, box_y_start = 0.97, 0.96
    line_spacing = 0.068
    ax_main.text(
        box_x,
        box_y_start + 0.01,
        r"Linear Fits ($y = m\log_{10}(a_{\mathrm{Cl}^-}) + b$):",
        transform=ax_main.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color="#111827",
        va="top",
        ha="right",
    )
    for idx, (color, eq_text) in enumerate(fit_annotations):
        ax_main.text(
            box_x,
            box_y_start - (idx + 1) * line_spacing,
            eq_text,
            transform=ax_main.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color=color,
            va="top",
            ha="right",
        )

    # Axis Formatting - Top Subplot
    ax_main.set_ylabel(y_label, fontsize=13, fontweight="bold", color="#111827")
    ax_main.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax_main.grid(True, linestyle="--", alpha=0.35)
    ax_main.legend(loc=legend_loc, frameon=True, fontsize=9.5, ncol=2)
    ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))

    # Axis Formatting - Bottom Subplot (Residuals Bar Plot)
    ax_res.set_ylabel(res_y_label, fontsize=11.5, fontweight="bold", color="#111827")
    ax_res.set_xlabel(
        r"$\log_{10}(a_{\mathrm{Cl}^-})$",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )
    ax_res.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax_res.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))

    ax_main.set_xlim(x_min_global - x_pad, x_max_global + x_pad)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    if not os.path.exists(METRICS_JSON_PATH):
        raise FileNotFoundError(
            f"Could not find '{METRICS_JSON_PATH}'. Run pulseCalc.py first to generate it."
        )

    with open(METRICS_JSON_PATH, "r") as f:
        metrics_data = json.load(f)

    # Figure 1: Delta E / Delta T vs. log10(activity)
    generate_fit_and_residual_figure(
        all_metrics=metrics_data,
        output_path=OUTPUT_SENSITIVITY_PNG,
        mode="sensitivity",
        title=r"Thermoelectric Sensitivity ($\Delta E / \Delta T$) vs. $\log_{10}(a_{\mathrm{Cl}^-})$",
        y_label=r"$\Delta E / \Delta T\ [\mathrm{mV \cdot K^{-1}}]$",
        res_y_label=r"Residuals" "\n" r"$[\mathrm{mV \cdot K^{-1}}]$",
        y_fmt="%.3f",
        legend_loc="lower left",
    )

    # Figure 2: Delta E vs. log10(activity)
    generate_fit_and_residual_figure(
        all_metrics=metrics_data,
        output_path=OUTPUT_DELTA_E_PNG,
        mode="delta_e",
        title=r"Thermoelectric Potential ($\Delta E$) vs. $\log_{10}(a_{\mathrm{Cl}^-})$",
        y_label=r"$\Delta E\ [\mathrm{mV}]$",
        res_y_label=r"Residuals" "\n" r"$[\mathrm{mV}]$",
        y_fmt="%.2f",
        legend_loc="lower left",
    )