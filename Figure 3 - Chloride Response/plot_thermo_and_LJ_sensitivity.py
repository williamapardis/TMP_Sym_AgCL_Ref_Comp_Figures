import json
import os
from typing import Dict, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats


# ==========================================
# Configuration
# ==========================================
TITRATION_TXT_PATH = "closed_loop_titration.txt"
METRICS_JSON_PATH = "all_pulse_metrics.json"
OUTPUT_DELTA_E_PNG = "delta_e_and_lj_cp_vs_log_activity.png"

# Invert L.J. CP signal sign to correct for experimental setup polarity
INVERT_LJ_CP_SIGN = False

# Step Selection Toggle:
# - Range tuple (start, end): e.g., (0, 7) (inclusive)
# - Specific list: e.g., [0, 1, 3, 5, 7]
# - None: include all steps
STEPS_TO_INCLUDE: Optional[Union[Tuple[int, int], Sequence[int]]] = (0, 7)

# Toggle x-axis metric for L.J. CP subplot: "activity" or "concentration"
X_METRIC = "activity"

# Default time window for L.J. CP averaging within each step (in minutes)
AVG_WINDOW_START = 45
AVG_WINDOW_END = 85

# Per-step averaging window overrides: {step_id: (start_min, end_min)}
STEP_AVG_WINDOW_OVERRIDES: Dict[int, Tuple[float, float]] = {
    0: (70, 110),
}

# Axis limits for L.J. CP Nernst subplot (Set to None for automatic scaling)
LJ_CP_X_LIMITS = None  # e.g., (-2.2, -0.2)
LJ_CP_Y_LIMITS = None #(-180, -70)  # e.g., (-250, -50)

# Axis limits for continuous L.J. CP Time-Series inset (Set to None for auto)
LJ_CP_TS_X_LIMITS = None  # Continuous titration elapsed time [hr]
LJ_CP_TS_Y_LIMITS = None

# Y-axis tick label decimal precision for Figure 1 subplots
LJ_CP_Y_FMT = "%.0f"   # Left subplot (L.J. CP [mV])
DELTA_E_Y_FMT = "%.1f" # Right subplot (Thermoelectric Delta E [mV])

# Toggle Full Titration Time-Series Inset on Figure 1 L.J. CP Subplot
SHOW_TS_INSET = True

BG_COLOR = "#F4F5F7"

# Physical constants (CODATA) & baseline ambient temperature [deg C]
R_GAS = 8.314462618      # J / (mol * K)
F_FARADAY = 96485.33212  # C / mol
T_AMBIENT_C = 25.0       # Reference ambient temperature for Bates & Bower E^0(T)

# Theoretical Nernstian Slope at 25 °C for Cl- (monovalent anion: z = -1)
# Slope = (2.3026 * R * T) / (z * F) = -59.16 mV / log10(a_Cl-)
NERNST_SLOPE_25C = 59.16

SERIES_ORDER = ["1C", "5C", "10C"]

# Blue marker color matched to the AgCl|AgCl potential traces in the reference figure
SERIES_MARKER_BLUE = "#1D63B8"

SERIES_STYLES: Dict[str, dict] = {
    "1C": {
        "label": r"$\Delta T \approx 1\,^\circ\mathrm{C}$",
        "point_color": SERIES_MARKER_BLUE,
        "fit_color": "#C27803",    # Darker Amber/Gold
        "marker": "o",
        "linestyle": "-",
    },
    "5C": {
        "label": r"$\Delta T \approx 5\,^\circ\mathrm{C}$",
        "point_color": SERIES_MARKER_BLUE,
        "fit_color": "#F5A623",    # Base Golden-Amber (matched to Temperature trace)
        "marker": "s",
        "linestyle": "--",
    },
    "10C": {
        "label": r"$\Delta T \approx 10\,^\circ\mathrm{C}$",
        "point_color": SERIES_MARKER_BLUE,
        "fit_color": "#FCD34D",    # Lighter Golden-Amber
        "marker": "^",
        "linestyle": "-.",
    },
}


def get_step_avg_window(step_id: int) -> Tuple[float, float]:
    """
    Returns the (start_min, end_min) averaging window for `step_id`,
    checking STEP_AVG_WINDOW_OVERRIDES before falling back to
    (AVG_WINDOW_START, AVG_WINDOW_END).
    """
    return STEP_AVG_WINDOW_OVERRIDES.get(
        int(step_id), (float(AVG_WINDOW_START), float(AVG_WINDOW_END))
    )


def align_dual_axis_zeros(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """
    Adjusts the y-limits of `ax_left` and `ax_right` so that y = 0 lies at the
    exact same vertical fraction on both axes without clipping any plotted data.
    """
    y1_min, y1_max = ax_left.get_ylim()
    y2_min, y2_max = ax_right.get_ylim()

    y1_min, y1_max = min(y1_min, 0.0), max(y1_max, 0.0)
    y2_min, y2_max = min(y2_min, 0.0), max(y2_max, 0.0)

    r1 = y1_max - y1_min
    r2 = y2_max - y2_min
    if r1 == 0.0 or r2 == 0.0:
        return

    f1 = -y1_min / r1
    f2 = -y2_min / r2

    if f1 > f2:
        new_y2_min = -(f1 / (1.0 - f1)) * y2_max if f1 < 1.0 else y2_min
        ax_left.set_ylim(y1_min, y1_max)
        ax_right.set_ylim(new_y2_min, y2_max)
    elif f2 > f1:
        new_y1_min = -(f2 / (1.0 - f2)) * y1_max if f2 < 1.0 else y1_min
        ax_left.set_ylim(new_y1_min, y1_max)
        ax_right.set_ylim(y2_min, y2_max)
    else:
        ax_left.set_ylim(y1_min, y1_max)
        ax_right.set_ylim(y2_min, y2_max)


def is_step_included(
    step_idx: int,
    steps_to_include: Optional[Union[Tuple[int, int], Sequence[int]]],
) -> bool:
    """
    Determines whether a given integer step index should be included:
    - None: includes all steps
    - tuple (start, end): inclusive range [start, end]
    - list/sequence: explicit list of step indices
    """
    if steps_to_include is None:
        return True
    if isinstance(steps_to_include, tuple):
        if len(steps_to_include) != 2:
            raise ValueError(
                f"Range tuple must have length 2 (start, end), got {steps_to_include}"
            )
        start, end = steps_to_include
        return start <= step_idx <= end
    return step_idx in steps_to_include


def get_step_log_x_from_metrics(
    all_metrics: dict,
    step_id: int,
    metric: str = "activity",
) -> Optional[float]:
    """
    Retrieves log10(activity) or log10(concentration) for a given step_id
    directly from all_pulse_metrics.json.
    """
    step_key = str(step_id)
    for series_key in SERIES_ORDER:
        step_summaries = all_metrics.get(series_key, {}).get("step_summaries", {})
        if step_key in step_summaries:
            entry = step_summaries[step_key]
            if metric.lower() == "activity":
                if "log_activity" in entry:
                    return float(entry["log_activity"])
                if "activity" in entry:
                    return float(np.log10(float(entry["activity"])))
            else:
                if "log_concentration" in entry:
                    return float(entry["log_concentration"])
                for conc_key in ("concentration", "concentration_m", "concentration_M"):
                    if conc_key in entry:
                        return float(np.log10(float(entry[conc_key])))
    return None


def parse_and_update_lj_cp_metrics(
    txt_path: str,
    metrics_json_path: str,
    steps_to_include: Optional[Union[Tuple[int, int], Sequence[int]]] = STEPS_TO_INCLUDE,
) -> Tuple[dict, pd.DataFrame, Dict[int, float], Dict[int, float]]:
    """
    Parses closed_loop_titration.txt, computes both continuous titration time
    and within-step averaging windows (respecting STEP_AVG_WINDOW_OVERRIDES),
    updates all_pulse_metrics.json with 'lj_cp_mean_mV' and 'lj_cp_std_mV',
    and returns the updated metrics dict, processed DataFrame, and step averages/stds.
    """
    if not os.path.exists(metrics_json_path):
        raise FileNotFoundError(
            f"Could not find '{metrics_json_path}'. Run pulseCalc.py first to generate it."
        )
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Could not find '{txt_path}'.")

    with open(metrics_json_path, "r") as f:
        all_metrics = json.load(f)

    columns = [
        "timestamp",
        "step",
        "volume dispensed",
        "temperature setpoint",
        "avg min temp",
        "PID setpoint",
        "avg max temp",
        "temperature",
        "PID enable",
        "current",
        "time to next",
        "time on",
        "time off",
        "AgCl CP",
        "L.J. CP",
        "avg min CP",
        "avg max CP",
        "TRA",
        "averages",
    ]

    df = pd.read_csv(txt_path, names=columns, header=0, skipinitialspace=True)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["L.J. CP"] = pd.to_numeric(df["L.J. CP"], errors="coerce")

    df["L.J. CP"] = df["L.J. CP"]-1193/5

    if INVERT_LJ_CP_SIGN:
        df["L.J. CP"] = -1.0 * df["L.J. CP"]

    df = df.dropna(subset=["timestamp", "step", "L.J. CP"])
    df["timestamp"] = (
        df["timestamp"]
        .astype(str)
        .str.extract(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")[0]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["step"] = df["step"].astype(int)

    df = df[df["step"].apply(lambda s: is_step_included(int(s), steps_to_include))].copy()

    t0 = df["timestamp"].min()
    df["elapsed_total_min"] = (df["timestamp"] - t0).dt.total_seconds() / 60.0
    df["elapsed_total_hr"] = df["elapsed_total_min"] / 60.0

    df["elapsed_time_step_min"] = df.groupby("step")["timestamp"].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 60.0
    )

    step_averages: Dict[int, float] = {}
    step_std_devs: Dict[int, float] = {}

    for step_id, group in df.groupby("step"):
        win_start, win_end = get_step_avg_window(int(step_id))
        window_mask = (group["elapsed_time_step_min"] >= win_start) & (
            group["elapsed_time_step_min"] <= win_end
        )
        window_data = group[window_mask]

        if not window_data.empty:
            mean_val = float(window_data["L.J. CP"].mean())
            std_val = float(window_data["L.J. CP"].std())
            step_averages[int(step_id)] = mean_val
            step_std_devs[int(step_id)] = std_val if not np.isnan(std_val) else 0.0

    lj_cp_summary = all_metrics.setdefault("lj_cp_summaries", {})
    for step_id in sorted(step_averages.keys()):
        step_key = str(step_id)
        mean_val = step_averages[step_id]
        std_val = step_std_devs[step_id]

        lj_cp_summary[step_key] = {
            "lj_cp_mean_mV": mean_val,
            "lj_cp_std_mV": std_val,
        }

        for series_key in SERIES_ORDER:
            if (
                series_key in all_metrics
                and "step_summaries" in all_metrics[series_key]
                and step_key in all_metrics[series_key]["step_summaries"]
            ):
                all_metrics[series_key]["step_summaries"][step_key]["lj_cp_mean_mV"] = mean_val
                all_metrics[series_key]["step_summaries"][step_key]["lj_cp_std_mV"] = std_val

    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"Successfully updated L.J. CP statistics in '{metrics_json_path}'.")
    return all_metrics, df, step_averages, step_std_devs


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
    mode: str = "delta_e",
    steps_to_include: Optional[Union[Tuple[int, int], Sequence[int]]] = STEPS_TO_INCLUDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts log10(activity), Y values, Y standard deviations, Delta T means,
    and Delta T standard deviations sorted by step index, filtered by steps_to_include.
    """
    sorted_keys = sorted(
        (k for k in step_summaries.keys() if is_step_included(int(k), steps_to_include)),
        key=lambda k: int(k),
    )

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


def plot_lj_cp_subplot(
    ax: plt.Axes,
    all_metrics: dict,
    df_titration: pd.DataFrame,
    step_averages: Dict[int, float],
    step_std_devs: Dict[int, float],
    hollow_markers: bool = False,
    y_fmt: str = LJ_CP_Y_FMT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    """
    Renders Average L.J. CP vs. log10(Metric) on `ax` with bold fit text annotations
    in the bottom-left corner, and places the continuous full-titration L.J. CP time
    series (in hours) with averaged step markers inside an inset window in the top-right corner.
    Returns (log_x, std_cp, residuals_fit, residuals_ideal, lj_fit_slope, x_label_str) for Figure 2.
    """
    steps_sorted = sorted(step_averages.keys())

    if X_METRIC.lower() == "activity":
        x_label_str = r"$\log_{10}(a_{\mathrm{Cl}^-})$"
    else:
        x_label_str = r"$\log_{10}([\mathrm{Cl}^-])$"

    log_x = []
    avg_cp = []
    std_cp = []

    for step_id in steps_sorted:
        x_val = get_step_log_x_from_metrics(all_metrics, step_id, metric=X_METRIC)
        if x_val is None:
            print(
                f"Warning: '{X_METRIC}' value for Step {step_id} not found in "
                f"'{METRICS_JSON_PATH}', skipping step in L.J. CP subplot."
            )
            continue
        log_x.append(x_val)
        avg_cp.append(step_averages[step_id])
        std_cp.append(step_std_devs[step_id])

    log_x = np.array(log_x, dtype=float)
    avg_cp = np.array(avg_cp, dtype=float)
    std_cp = np.array(std_cp, dtype=float)

    residuals_fit = np.zeros_like(avg_cp)
    residuals_ideal = np.zeros_like(avg_cp)
    slope = NERNST_SLOPE_25C

    mfc = BG_COLOR if hollow_markers else "#111827"

    if len(log_x) > 2:
        res = stats.linregress(log_x, avg_cp)
        slope = float(res.slope)
        intercept = float(res.intercept)
        r_squared = float(res.rvalue ** 2)

        y_pred_fit = slope * log_x + intercept
        residuals_fit = avg_cp - y_pred_fit

        x_mean = np.mean(log_x)
        y_mean = np.mean(avg_cp)
        ideal_intercept = y_mean - (NERNST_SLOPE_25C * x_mean)
        y_pred_ideal = NERNST_SLOPE_25C * log_x + ideal_intercept
        residuals_ideal = avg_cp - y_pred_ideal

        ax.errorbar(
            log_x,
            avg_cp,
            yerr=std_cp,
            fmt="o",
            color="#000000",
            ecolor="#000000",
            markerfacecolor="#000000",
            markeredgecolor="#000000",
            markeredgewidth=1.0,
            elinewidth=1.5,
            capsize=5.5,
            capthick=1.5,
            markersize=4.8,
            zorder=5,
        )

        x_pad = 0.08 * (log_x.max() - log_x.min()) if log_x.max() > log_x.min() else 0.1
        x_fit = np.linspace(log_x.min() - x_pad, log_x.max() + x_pad, 200)
        y_fit = slope * x_fit + intercept
        y_ideal = NERNST_SLOPE_25C * x_fit + ideal_intercept

        fit_color = "#A0A0A0"
        ideal_color = "#525252"

        ax.plot(
            x_fit,
            y_fit,
            color=fit_color,
            linestyle="--",
            linewidth=2.2,
            alpha=0.95,
            zorder=4,
        )

        ax.plot(
            x_fit,
            y_ideal,
            color=ideal_color,
            linestyle="-",
            linewidth=1.8,
            alpha=0.9,
            zorder=3,
        )

        # Bottom-left bold text annotations
        box_x, box_y_bottom = 0.01, 0.78 #0.46, 0.58 upper right
        line_spacing = 0.062
        sign_str = "+" if intercept >= 0 else "-"
        ideal_sign_str = "+" if ideal_intercept >= 0 else "-"

        lj_annotations = [
            (
                fit_color,
                f"OLS Fit (dashed): $E_{{\\mathrm{{LJ}}}} = {slope:.2f}\\,{x_label_str.strip('$')} {sign_str} {abs(intercept):.0f}$  "
                f"($R^2 = {r_squared:.3f}$)",
            ),
            (
                ideal_color,
                f"Ideal Nernst 25°C (solid): $E_{{\\mathrm{{LJ}}}} = {NERNST_SLOPE_25C:.2f}\\,{x_label_str.strip('$')} {ideal_sign_str} {abs(ideal_intercept):.2f}$",
            ),
        ]

        # ax.text(
        #     box_x,
        #     box_y_bottom + len(lj_annotations) * line_spacing + 0.01,
        #     rf"$E_{{\mathrm{{LJ}}}} = S\,{x_label_str.strip('$')} + E^{{\circ}}_{{\mathrm{{eff}}}}$:",
        #     transform=ax.transAxes,
        #     fontsize=10.5,
        #     fontweight="bold",
        #     color="#111827",
        #     va="bottom",
        #     ha="left",
        # )

        for idx, (txt_color, eq_text) in enumerate(lj_annotations):
            y_pos = box_y_bottom + (len(lj_annotations) - 1 - idx) * line_spacing
            ax.text(
                box_x,
                y_pos,
                eq_text,
                transform=ax.transAxes,
                fontsize=10.5,
                fontweight="bold",
                color=txt_color,
                va="bottom",
                ha="left",
            )

        ax.set_xlim(log_x.min() - x_pad, log_x.max() + x_pad)

    if LJ_CP_X_LIMITS is not None:
        ax.set_xlim(LJ_CP_X_LIMITS)
    if LJ_CP_Y_LIMITS is not None:
        ax.set_ylim(LJ_CP_Y_LIMITS)

    # ==========================================
    # Inset Window: Full Continuous Titration L.J. CP Time Series [hr]
    # ==========================================
    if SHOW_TS_INSET and not df_titration.empty:
        ax_inset = ax.inset_axes([0.46, 0.09, 0.5, 0.35])
        ax_inset.set_facecolor("#ffffff")

        avg_times_hr = []
        avg_vals_mv = []
        avg_stds_mv = []

        for idx_s, (step_id, group) in enumerate(df_titration.groupby("step")):
            step_int = int(step_id)
            win_start, win_end = get_step_avg_window(step_int)
            window_mask = (group["elapsed_time_step_min"] >= win_start) & (
                group["elapsed_time_step_min"] <= win_end
            )
            win_grp = group[window_mask]
            if not win_grp.empty:
                t_min_hr = float(win_grp["elapsed_total_hr"].min())
                t_max_hr = float(win_grp["elapsed_total_hr"].max())
                ax_inset.axvspan(
                    t_min_hr,
                    t_max_hr,
                    color="#94A3B8",
                    alpha=0.25,
                    label="Avg Window" if idx_s == 0 else None,
                    zorder=1,
                )
                if step_int in step_averages:
                    avg_times_hr.append(float(win_grp["elapsed_total_hr"].mean()))
                    avg_vals_mv.append(step_averages[step_int])
                    avg_stds_mv.append(step_std_devs.get(step_int, 0.0))

        ax_inset.plot(
            df_titration["elapsed_total_hr"],
            df_titration["L.J. CP"],
            color="#000000",
            linestyle="-",
            linewidth=1.1,
            alpha=0.9,
            zorder=2,
        )

        # Averaged points (black, matching the main plot)
        if avg_times_hr:
            inset_mfc = "#ffffff" if hollow_markers else "#EC1F1F"
            ax_inset.errorbar(
                avg_times_hr,
                avg_vals_mv,
                yerr=avg_stds_mv,
                fmt="o",
                color="#EC1F1F",
                ecolor="#EC1F1F",
                markerfacecolor=inset_mfc,
                markeredgecolor="#EC1F1F",
                markeredgewidth=0.9,
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                markersize=3.4,
                zorder=4,
            )

        if LJ_CP_TS_X_LIMITS is not None:
            ax_inset.set_xlim(LJ_CP_TS_X_LIMITS)
        else:
            ax_inset.set_xlim(
                float(df_titration["elapsed_total_hr"].min()),
                float(df_titration["elapsed_total_hr"].max()),
            )
        if LJ_CP_TS_Y_LIMITS is not None:
            ax_inset.set_ylim(LJ_CP_TS_Y_LIMITS)

        ax_inset.set_title(
            "Titration Time Series",
            fontsize=8,
            fontweight="bold",
            color="#111827",
            pad=3,
        )
        ax_inset.set_xlabel("Elapsed Time [hr]", fontsize=7, color="#111827")
        ax_inset.set_ylabel("$E_{Ag/AgCl}$ [mV]", fontsize=7, color="#111827")
        ax_inset.tick_params(labelsize=6.5)
        ax_inset.grid(True, linestyle=":", color="#cbd5e1", alpha=0.75)

    ax.set_title(
        "Open Circuit Potential Against L.J. Reference",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        pad=12,
    )
    ax.set_xlabel(x_label_str, fontsize=13, fontweight="bold", color="#111827")
    ax.set_ylabel(r"$E_{Ag/AgCl}$ [mV]", fontsize=13, fontweight="bold", color="#111827")
    ax.grid(True, linestyle="--", color="#cbd5e1", alpha=0.75)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))

    return log_x, std_cp, residuals_fit, residuals_ideal, slope, x_label_str


def generate_merged_figures(
    all_metrics: dict,
    df_titration: pd.DataFrame,
    step_averages: Dict[int, float],
    step_std_devs: Dict[int, float],
    output_path: str,
    mode: str,
    title: str,
    y_label: str,
    res_y_label: str,
    int_y_label: str,
    y_fmt: str = DELTA_E_Y_FMT,
    lj_y_fmt: str = LJ_CP_Y_FMT,
    hollow_markers: bool = False,
    jitter_step: float = 0.0,
    steps_to_include: Optional[Union[Tuple[int, int], Sequence[int]]] = STEPS_TO_INCLUDE,
) -> None:
    """
    Generates three figures:
    1. Merged 1x2 Main Figure:
       - Left (ax_ljcp): Average L.J. CP vs. log10(activity) with Top-Right Time-Series Inset [hr]
       - Right (ax_main): Thermoelectric Potential (Delta E) vs. log10(activity)
    2. Vertically Stacked 2x1 Residuals Figure (with aligned y = 0 across both axes):
       - Top Subplot (ax_res): Raw Residuals [mV] (Thermo Left Axis, L.J. CP Right Axis)
       - Bottom Subplot (ax_res_norm): Sensitivity-Normalized Residuals [decades] (Residual / |Slope|)
    3. Intercepts vs. Theoretical Temperature Figure.
    """
    plt.rcdefaults()

    base_name, ext = os.path.splitext(output_path)
    residual_output_path = f"{base_name}_residuals{ext}"
    intercept_output_path = f"{base_name}_intercepts{ext}"

    # Merged 1x2 Main Figure (Left: ax_ljcp, Right: ax_main)
    fig_main, (ax_ljcp, ax_main) = plt.subplots(
        1, 2, figsize=(19.5, 6.4), facecolor=BG_COLOR
    )
    ax_ljcp.set_facecolor(BG_COLOR)
    ax_main.set_facecolor(BG_COLOR)

    # Vertically Stacked 2x1 Residuals Figure (Top: Raw mV, Bottom: Normalized by |Slope|)
    fig_res, (ax_res, ax_res_norm) = plt.subplots(
        2, 1, figsize=(11.5, 8.8), sharex=True, facecolor=BG_COLOR
    )
    ax_res.set_facecolor(BG_COLOR)
    ax_res_norm.set_facecolor(BG_COLOR)

    fig_int, ax_int = plt.subplots(figsize=(8.8, 5.6), facecolor=BG_COLOR)
    ax_int.set_facecolor(BG_COLOR)

    x_min_global, x_max_global = np.inf, -np.inf
    parsed_series = {}

    for series_key in SERIES_ORDER:
        if series_key not in all_metrics:
            print(f"Warning: '{series_key}' not found in metrics JSON, skipping.")
            continue
        x, y, y_err, dt_means, dt_stds = compute_series_arrays(
            all_metrics[series_key]["step_summaries"],
            mode=mode,
            steps_to_include=steps_to_include,
        )
        if len(x) == 0:
            print(
                f"Warning: No matching steps found for '{series_key}' with filter {steps_to_include}, skipping."
            )
            continue
        parsed_series[series_key] = (x, y, y_err, dt_means, dt_stds)
        x_min_global = min(x_min_global, float(np.min(x)))
        x_max_global = max(x_max_global, float(np.max(x)))

    if not parsed_series:
        raise ValueError(
            f"No data points remained after applying STEPS_TO_INCLUDE={steps_to_include}."
        )

    x_pad = 0.08 * (x_max_global - x_min_global) if x_max_global > x_min_global else 0.1
    x_fit_grid = np.linspace(x_min_global - x_pad, x_max_global + x_pad, 200)

    active_keys = [k for k in SERIES_ORDER if k in parsed_series]
    n_series = len(active_keys)

    total_res_groups = n_series + 2
    sample_x = parsed_series[active_keys[0]][0]
    min_dx = float(np.min(np.diff(np.sort(sample_x)))) if len(sample_x) > 1 else 0.1
    bar_width = (min_dx * 0.78) / max(total_res_groups, 1)

    fit_annotations = []
    ax_res.axhline(0.0, color="#111827", linestyle="-", linewidth=1.1, alpha=0.85, zorder=2)
    ax_res_norm.axhline(0.0, color="#111827", linestyle="-", linewidth=1.1, alpha=0.85, zorder=2)

    theo_temp_vals = []
    theo_temp_errs = []
    intercept_vals = []
    intercept_errs = []
    series_markers = []

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

        delta_t_theo = abs(slope) * nernst_inv_factor
        delta_t_theo_se = slope_se * nernst_inv_factor

        theo_temp_vals.append(delta_t_theo)
        theo_temp_errs.append(delta_t_theo_se)
        intercept_vals.append(intercept)
        intercept_errs.append(intercept_se)
        series_markers.append(marker)

        sign_str = "+" if intercept >= 0 else "-"
        eq_str = (
            f"{base_label}: $\\Delta\\ E = {slope:.2f}"
            + r"\ \log_{10}(a_{\mathrm{Cl}^-})"
            + f"{sign_str} {abs(intercept):.2f}$  "
            f"($R^2 = {r2:.3f}$)"
        )
        fit_annotations.append((fit_color, eq_str))

        x_plot = x + (idx - (n_series - 1) / 2.0) * jitter_step

        # Right Subplot of Fig 1: Linear Fit Line
        ax_main.plot(
            x_fit_grid,
            y_fit_grid,
            color=fit_color,
            linestyle=linestyle,
            linewidth=2.2,
            alpha=0.95,
            zorder=3,
        )

        # Right Subplot of Fig 1: Blue markers + error bars matching AgCl|AgCl trace palette
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

        # Figure 2 Top Subplot (Left Y-Axis): Raw Thermoelectric Residuals
        bar_offset = (idx - (total_res_groups - 1) / 2.0) * bar_width
        ax_res.bar(
            x + bar_offset,
            residuals,
            width=bar_width * 0.90,
            color=fit_color,
            edgecolor="#111827",
            linewidth=0.6,
            alpha=0.9,
            yerr=y_err,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 1.2,
                "capsize": 3.0,
                "capthick": 1.2,
            },
            label=f"{base_label} (Thermo)",
            zorder=3,
        )

        # Figure 2 Bottom Subplot (Left Y-Axis): Sensitivity-Normalized Residuals (Res / |Slope|)
        abs_slope = abs(slope) if slope != 0.0 else 1.0
        norm_residuals = residuals / abs_slope
        norm_y_err = y_err / abs_slope

        ax_res_norm.bar(
            x + bar_offset,
            norm_residuals,
            width=bar_width * 0.90,
            color=fit_color,
            edgecolor="#111827",
            linewidth=0.6,
            alpha=0.9,
            yerr=norm_y_err,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 1.2,
                "capsize": 3.0,
                "capthick": 1.2,
            },
            label=f"{base_label} (Thermo)",
            zorder=3,
        )

    # Display Linear Fit Equations & R^2 Labels in Upper Right of Right Subplot
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
    # Formatting - Merged Figure 1 (Left: L.J. CP | Right: Thermoelectric Delta E)
    # ==========================================
    lj_log_x, lj_std_cp, lj_res_fit, lj_res_ideal, lj_slope, lj_x_label = plot_lj_cp_subplot(
        ax=ax_ljcp,
        all_metrics=all_metrics,
        df_titration=df_titration,
        step_averages=step_averages,
        step_std_devs=step_std_devs,
        hollow_markers=hollow_markers,
        y_fmt=lj_y_fmt,
    )

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

    fig_main.tight_layout(pad=2.0)
    fig_main.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig_main.get_facecolor(),
    )
    print(f"Saved merged main figure to {output_path}")

    # ==========================================
    # Formatting - Figure 2 (Top: Raw Residuals | Bottom: Slope-Normalized Residuals)
    # ==========================================
    ax_res_ljcp = ax_res.twinx()
    ax_res_ljcp_norm = ax_res_norm.twinx()
    lj_res_color = "#2563EB"  # Slate-blue accent matching L.J. CP time-series styling

    if len(lj_log_x) > 0:
        offset_lj_fit = (n_series - (total_res_groups - 1) / 2.0) * bar_width
        offset_lj_ideal = (n_series + 1 - (total_res_groups - 1) / 2.0) * bar_width

        # Top Subplot (Right Y-Axis): Raw L.J. CP Residuals [mV]
        ax_res_ljcp.bar(
            lj_log_x + offset_lj_fit,
            lj_res_fit,
            width=bar_width * 0.90,
            yerr=lj_std_cp,
            color=lj_res_color,
            edgecolor="#111827",
            linewidth=0.6,
            alpha=0.88,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 1.2,
                "capsize": 3.0,
                "capthick": 1.2,
            },
            label="L.J. CP vs. OLS Fit (Right Axis)",
            zorder=3,
        )
        ax_res_ljcp.bar(
            lj_log_x + offset_lj_ideal,
            lj_res_ideal,
            width=bar_width * 0.90,
            yerr=lj_std_cp,
            color="#60A5FA",
            edgecolor="#111827",
            linewidth=0.6,
            hatch="//",
            alpha=0.85,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 1.2,
                "capsize": 3.0,
                "capthick": 1.2,
            },
            label="L.J. CP vs. Ideal Nernst (Right Axis)",
            zorder=3,
        )

        # Bottom Subplot (Right Y-Axis): Normalized L.J. CP Residuals [decades]
        abs_lj_slope = abs(lj_slope) if lj_slope != 0.0 else 1.0
        abs_nernst_slope = abs(NERNST_SLOPE_25C)

        ax_res_ljcp_norm.bar(
            lj_log_x + offset_lj_fit,
            lj_res_fit / abs_lj_slope,
            width=bar_width * 0.90,
            yerr=lj_std_cp / abs_lj_slope,
            color=lj_res_color,
            edgecolor="#111827",
            linewidth=0.6,
            alpha=0.88,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 1.2,
                "capsize": 3.0,
                "capthick": 1.2,
            },
            label="L.J. CP vs. OLS Fit (Right Axis)",
            zorder=3,
        )
        ax_res_ljcp_norm.bar(
            lj_log_x + offset_lj_ideal,
            lj_res_ideal / abs_nernst_slope,
            width=bar_width * 0.90,
            yerr=lj_std_cp / abs_nernst_slope,
            color="#60A5FA",
            edgecolor="#111827",
            linewidth=0.6,
            hatch="//",
            alpha=0.85,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 1.2,
                "capsize": 3.0,
                "capthick": 1.2,
            },
            label="L.J. CP vs. Ideal Nernst (Right Axis)",
            zorder=3,
        )

    ax_res.set_ylabel(res_y_label, fontsize=12, fontweight="bold", color="#111827")
    ax_res.set_title(
        "Linear Fit Residuals Comparison",
        fontsize=13.5,
        fontweight="bold",
        color="#111827",
        pad=10,
    )
    ax_res.grid(True, linestyle="--", color="#cbd5e1", alpha=0.75, zorder=0)
    ax_res.set_xlim(x_min_global - x_pad, x_max_global + x_pad)

    ax_res_ljcp.set_ylabel(
        r"L.J. CP Residual [mV]", fontsize=12, fontweight="bold", color=lj_res_color
    )
    ax_res_ljcp.tick_params(axis="y", colors=lj_res_color)
    align_dual_axis_zeros(ax_res, ax_res_ljcp)

    handles_left, labels_left = ax_res.get_legend_handles_labels()
    handles_right, labels_right = ax_res_ljcp.get_legend_handles_labels()
    ax_res.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=9.5,
    )

    ax_res_norm.set_xlabel(lj_x_label, fontsize=13, fontweight="bold", color="#111827")
    ax_res_norm.set_ylabel(
        r"Normalized Residual [decades]", fontsize=12, fontweight="bold", color="#111827"
    )
    ax_res_norm.grid(True, linestyle="--", color="#cbd5e1", alpha=0.75, zorder=0)

    ax_res_ljcp_norm.set_ylabel(
        r"L.J. CP Norm. Residual [decades]",
        fontsize=12,
        fontweight="bold",
        color=lj_res_color,
    )
    ax_res_ljcp_norm.tick_params(axis="y", colors=lj_res_color)
    align_dual_axis_zeros(ax_res_norm, ax_res_ljcp_norm)

    fig_res.tight_layout(pad=2.0)
    fig_res.savefig(
        residual_output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig_res.get_facecolor(),
    )
    print(f"Saved residuals figure to {residual_output_path}")

    # ==========================================
    # Formatting - Figure 3 (Intercepts vs. Theoretical Delta T)
    # ==========================================
    theo_temp_arr = np.array(theo_temp_vals, dtype=float)
    theo_temp_err_arr = np.array(theo_temp_errs, dtype=float)
    intercept_arr = np.array(intercept_vals, dtype=float)
    intercept_err_arr = np.array(intercept_errs, dtype=float)

    if len(theo_temp_arr) > 0:
        dt_max_plot = max(float(np.max(theo_temp_arr + theo_temp_err_arr)) * 1.15, 11.0)
        dt_curve = np.linspace(0.0, dt_max_plot, 200)
        e0_curve = bates_bower_delta_e0_mv(dt_curve, t_amb_c=T_AMBIENT_C)

        ax_int.plot(
            dt_curve,
            e0_curve,
            color="#2563EB",
            linestyle="-",
            linewidth=2.0,
            label=rf"Bates & Bower $\Delta E^\circ(\Delta T)$ ($T_0 = {T_AMBIENT_C:.0f}\,^\circ\mathrm{{C}}$)",
            zorder=2,
        )

        for idx, series_key in enumerate(active_keys):
            style = SERIES_STYLES[series_key]
            ax_int.errorbar(
                theo_temp_arr[idx],
                intercept_arr[idx],
                xerr=theo_temp_err_arr[idx],
                yerr=intercept_err_arr[idx],
                fmt=style["marker"],
                color=style["point_color"],
                ecolor=style["point_color"],
                markerfacecolor=style["fit_color"],
                markeredgecolor=style["point_color"],
                markeredgewidth=1.1,
                markersize=6.5,
                elinewidth=1.5,
                capsize=5.0,
                capthick=1.5,
                label=style["label"],
                zorder=4,
            )

    ax_int.axhline(0.0, color="#64748B", linestyle=":", linewidth=1.1, zorder=1)
    ax_int.set_xlabel(
        r"Slope-Derived $\Delta T_{\mathrm{theo}}$ [$^\circ\mathrm{C}$]",
        fontsize=12.5,
        fontweight="bold",
        color="#111827",
    )
    ax_int.set_ylabel(int_y_label, fontsize=12.5, fontweight="bold", color="#111827")
    ax_int.set_title(
        r"Extrapolated $E^{\circ}(\Delta T)$ vs. Slope-Derived $\Delta T$",
        fontsize=13.5,
        fontweight="bold",
        color="#111827",
        pad=12,
    )
    ax_int.grid(True, linestyle="--", color="#cbd5e1", alpha=0.75)
    ax_int.legend(
        loc="best",
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        fontsize=10,
    )

    fig_int.tight_layout(pad=1.8)
    fig_int.savefig(
        intercept_output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=fig_int.get_facecolor(),
    )
    print(f"Saved intercepts figure to {intercept_output_path}")


if __name__ == "__main__":
    all_metrics, df_titration, step_averages, step_std_devs = parse_and_update_lj_cp_metrics(
        txt_path=TITRATION_TXT_PATH,
        metrics_json_path=METRICS_JSON_PATH,
        steps_to_include=STEPS_TO_INCLUDE,
    )

    generate_merged_figures(
        all_metrics=all_metrics,
        df_titration=df_titration,
        step_averages=step_averages,
        step_std_devs=step_std_devs,
        output_path=OUTPUT_DELTA_E_PNG,
        mode="delta_e",
        title="Thermoelectric Potential vs. Chloride Activity",
        y_label=r"$\Delta E$ [mV]",
        res_y_label=r"Thermo Residual [mV]",
        int_y_label=r"Intercept $E^{\circ}(\Delta T)$ [mV]",
        y_fmt=DELTA_E_Y_FMT,
        lj_y_fmt=LJ_CP_Y_FMT,
        hollow_markers=False,
        jitter_step=0.0,
        steps_to_include=STEPS_TO_INCLUDE,
    )