from dataclasses import dataclass
import json
import math
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ==========================================
# 1. Data Structure Definitions
# ==========================================
@dataclass
class Pulse:
    pulse_index: int
    loop_type: str  # 'closed-loop'
    time_on_s: float
    time_off_s: float
    ambient_temp_setpoint_c: float
    max_current_a: float
    step: int  # Step index from CSV data
    data: pd.DataFrame
    calculated_ambient_temp_c: Optional[float] = None
    delta_temp_target_c: Optional[float] = None

    # Endpoint values extracted at end of ON and OFF phases
    temp_last_on: Optional[float] = None
    temp_last_off: Optional[float] = None
    cp_last_on: Optional[float] = None
    cp_last_off: Optional[float] = None

    def __post_init__(self):
        """Ensures numeric metadata fields are strictly typed regardless of input source."""
        self.time_on_s = float(self.time_on_s)
        self.time_off_s = float(self.time_off_s)
        self.ambient_temp_setpoint_c = (
            float(self.ambient_temp_setpoint_c)
            if self.ambient_temp_setpoint_c is not None
            else None
        )
        self.max_current_a = float(self.max_current_a)
        self.step = int(self.step)
        if self.calculated_ambient_temp_c is not None:
            self.calculated_ambient_temp_c = float(self.calculated_ambient_temp_c)
        if self.delta_temp_target_c is not None:
            self.delta_temp_target_c = float(self.delta_temp_target_c)

    def extract_end_points(self) -> None:
        """Extracts temperature and potential values at the end of ON and OFF periods."""
        df = self.data
        if df.empty:
            return

        rel_time = (
            pd.to_datetime(df["datetime"]) - pd.to_datetime(df["datetime"].iloc[0])
        ).dt.total_seconds()

        idx_on = int((rel_time - self.time_on_s).abs().idxmin())
        idx_off = len(df) - 1

        self.temp_last_on = float(df["temperature"].iloc[idx_on])
        self.temp_last_off = float(df["temperature"].iloc[idx_off])

        cp_col = "cp_zeroed" if "cp_zeroed" in df.columns else "AgCl CP"
        self.cp_last_on = float(df[cp_col].iloc[idx_on])
        self.cp_last_off = float(df[cp_col].iloc[idx_off])

    def to_dict(self) -> dict:
        """Converts metadata and DataFrame time series to a JSON-serializable dictionary."""
        return {
            "pulse_index": self.pulse_index,
            "loop_type": self.loop_type,
            "metadata": {
                "time_on_s": self.time_on_s,
                "time_off_s": self.time_off_s,
                "ambient_temp_setpoint_c": self.ambient_temp_setpoint_c,
                "calculated_ambient_temp_c": self.calculated_ambient_temp_c,
                "max_current_a": self.max_current_a,
                "step": self.step,
                "delta_temp_target_c": self.delta_temp_target_c,
                "temp_last_on": self.temp_last_on,
                "temp_last_off": self.temp_last_off,
                "cp_last_on": self.cp_last_on,
                "cp_last_off": self.cp_last_off,
            },
            "time_series": self.data.to_dict(orient="list"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pulse":
        """Reconstructs a Pulse object from a JSON dictionary."""
        meta = data.get("metadata", {})
        ts_data = data.get("time_series", {})

        df = pd.DataFrame(ts_data)
        if "datetime" in df.columns and not pd.api.types.is_numeric_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])

        return cls(
            pulse_index=data.get("pulse_index", 0),
            loop_type=data.get("loop_type", "closed-loop"),
            time_on_s=meta.get("time_on_s", 0.0),
            time_off_s=meta.get("time_off_s", 0.0),
            ambient_temp_setpoint_c=meta.get("ambient_temp_setpoint_c"),
            max_current_a=meta.get("max_current_a", 0.0),
            step=meta.get("step", 0),
            calculated_ambient_temp_c=meta.get("calculated_ambient_temp_c"),
            delta_temp_target_c=meta.get("delta_temp_target_c"),
            temp_last_on=meta.get("temp_last_on"),
            temp_last_off=meta.get("temp_last_off"),
            cp_last_on=meta.get("cp_last_on"),
            cp_last_off=meta.get("cp_last_off"),
            data=df,
        )


# ==========================================
# 2. I/O Helpers, Filtering & Endpoint/Delta Engine
# ==========================================
def _to_naive_timestamps(series: pd.Series) -> pd.Series:
    """Parses datetime values and strips timezone info if present for safe comparison."""
    ts = pd.to_datetime(series)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    return ts


def _to_naive_timestamp(dt: pd.Timestamp) -> pd.Timestamp:
    """Converts a single Timestamp to timezone-naive if it carries tzinfo."""
    ts = pd.to_datetime(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def load_pulses_from_json(filepath: str) -> List[Pulse]:
    """Loads all Pulse objects from a JSON file."""
    with open(filepath, "r") as f:
        raw_data = json.load(f)
    return [Pulse.from_dict(p) for p in raw_data]


def save_pulses_to_json(pulses: List[Pulse], filepath: str) -> None:
    """Saves a list of Pulse objects to a JSON file matching the exact schema."""
    json_data = [p.to_dict() for p in pulses]
    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=4, default=str)
    print(f"Saved {len(pulses)} pulses to '{filepath}'.")


def compute_pulse_metrics(pulses: List[Pulse], window_label: str) -> dict:
    """Applies 10s baseline CP zeroing, extracts ON/OFF endpoints for each pulse,
    and computes mean/std ΔT and ΔE grouped by (ambient_temp_setpoint_c, delta_temp_target_c).
    """
    for p in pulses:
        if p.data.empty:
            continue
        dt = _to_naive_timestamps(p.data["datetime"])
        rel_t = (dt - dt.iloc[0]).dt.total_seconds()
        max_t = rel_t.iloc[-1]

        # Baseline correction for AgCl CP based on final 10 seconds of pulse
        baseline_cp = float(np.mean(p.data["AgCl CP"][rel_t >= (max_t - 10.0)]))
        p.data["cp_zeroed"] = p.data["AgCl CP"] - baseline_cp
        p.extract_end_points()

    # Group pulses by (ambient_temp_setpoint_c, delta_temp_target_c)
    grouped: Dict[Tuple[Optional[float], Optional[float]], List[Pulse]] = {}
    for p in pulses:
        sp_key = round(p.ambient_temp_setpoint_c, 2) if p.ambient_temp_setpoint_c is not None else None
        dt_key = round(p.delta_temp_target_c, 2) if p.delta_temp_target_c is not None else None
        grouped.setdefault((sp_key, dt_key), []).append(p)

    condition_summaries = []
    for (sp_val, dt_val), g_pulses in sorted(
        grouped.items(), key=lambda item: (item[0][0] or 0.0, item[0][1] or 0.0)
    ):
        valid = [
            p for p in g_pulses
            if p.temp_last_on is not None and p.temp_last_off is not None
            and p.cp_last_on is not None and p.cp_last_off is not None
        ]
        if not valid:
            continue

        t_on_vals = [p.temp_last_on for p in valid]
        t_off_vals = [p.temp_last_off for p in valid]
        delta_t_vals = [p.temp_last_on - p.temp_last_off for p in valid]

        cp_on_vals = [p.cp_last_on for p in valid]
        cp_off_vals = [p.cp_last_off for p in valid]
        delta_e_vals = [p.cp_last_on - p.cp_last_off for p in valid]

        summary = {
            "ambient_temp_setpoint_c": sp_val,
            "delta_temp_target_c": dt_val,
            "pulse_count": len(valid),
            "delta_t_mean_c": float(np.mean(delta_t_vals)),
            "delta_t_std_c": float(np.std(delta_t_vals)),
            "t_hot_mean_c": float(np.mean(t_on_vals)),
            "t_hot_std_c": float(np.std(t_on_vals)),
            "t_amb_mean_c": float(np.mean(t_off_vals)),
            "t_amb_std_c": float(np.std(t_off_vals)),
            "delta_e_mean_mv": float(np.mean(delta_e_vals)),
            "delta_e_std_mv": float(np.std(delta_e_vals)),
            "e_on_mean_mv": float(np.mean(cp_on_vals)),
            "e_on_std_mv": float(np.std(cp_on_vals)),
            "e_off_mean_mv": float(np.mean(cp_off_vals)),
            "e_off_std_mv": float(np.std(cp_off_vals)),
        }
        condition_summaries.append(summary)

        print(
            f"  [Window {window_label} | T_amb={sp_val}°C, ΔT_target={dt_val}°C] (n={len(valid)}): "
            f"ΔT = {summary['delta_t_mean_c']:.2f} ± {summary['delta_t_std_c']:.2f} °C | "
            f"ΔE = {summary['delta_e_mean_mv']:.3f} ± {summary['delta_e_std_mv']:.3f} mV"
        )

    return {
        "window": window_label,
        "total_pulses": len(pulses),
        "condition_summaries": condition_summaries,
    }


def filter_pulses(
    json_filepath: str,
    target_steps: Optional[List[int]] = None,
    delta_temp_target_c: Optional[float] = None,
    ambient_temp_setpoint_c: Optional[float] = None,
    max_current_a: Optional[float] = None,
    time_on_s: Optional[float] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
    pulses: Optional[List[Pulse]] = None,
) -> Tuple[List[Pulse], List[Pulse], dict]:
    """Filters closed-loop pulses based on metadata criteria and time range."""
    if pulses is None:
        pulses = load_pulses_from_json(json_filepath)

    retained_pulses = []
    removed_pulses = []

    target_step_set = set(target_steps) if target_steps is not None else None
    s_bound = _to_naive_timestamp(start_time) if start_time is not None else None
    e_bound = _to_naive_timestamp(end_time) if end_time is not None else None

    for pulse in pulses:
        keep = True

        # 1. Target Steps Check (ignored when target_steps=None)
        if target_step_set is not None and pulse.step not in target_step_set:
            keep = False

        # 2. Delta Temp Target Check
        if keep and delta_temp_target_c is not None:
            if pulse.delta_temp_target_c is None or not math.isclose(
                pulse.delta_temp_target_c, delta_temp_target_c, rel_tol=1e-4
            ):
                keep = False

        # 3. Ambient Temp Setpoint Check
        if keep and ambient_temp_setpoint_c is not None:
            if pulse.ambient_temp_setpoint_c is None or not math.isclose(
                pulse.ambient_temp_setpoint_c,
                ambient_temp_setpoint_c,
                rel_tol=1e-4,
            ):
                keep = False

        # 4. Max Current Check
        if keep and max_current_a is not None:
            if pulse.max_current_a is None or not math.isclose(
                pulse.max_current_a, max_current_a, rel_tol=1e-4
            ):
                keep = False

        # 5. Time On Check
        if keep and time_on_s is not None:
            if pulse.time_on_s is None or not math.isclose(
                pulse.time_on_s, time_on_s, rel_tol=1e-4
            ):
                keep = False

        # 6. Time Window Filtering (timezone-normalized)
        if keep and (s_bound is not None or e_bound is not None):
            if pulse.data.empty or "datetime" not in pulse.data.columns:
                keep = False
            else:
                timestamps = _to_naive_timestamps(pulse.data["datetime"])
                pulse_start = timestamps.min()
                pulse_end = timestamps.max()

                if s_bound is not None and pulse_start < s_bound:
                    keep = False
                if e_bound is not None and pulse_end > e_bound:
                    keep = False

        if keep:
            retained_pulses.append(pulse)
        else:
            removed_pulses.append(pulse)

    active_params = {}
    if target_steps is not None:
        active_params["Steps"] = target_steps
    if delta_temp_target_c is not None:
        active_params["ΔT Target"] = f"{delta_temp_target_c}°C"
    if ambient_temp_setpoint_c is not None:
        active_params["T_amb Setpoint"] = f"{ambient_temp_setpoint_c}°C"
    if max_current_a is not None:
        active_params["Max Current"] = f"{max_current_a}A"
    if time_on_s is not None:
        active_params["Time On"] = f"{time_on_s}s"
    if s_bound is not None:
        active_params["Start"] = s_bound.strftime("%Y-%m-%d %H:%M")
    if e_bound is not None:
        active_params["End"] = e_bound.strftime("%Y-%m-%d %H:%M")

    return retained_pulses, removed_pulses, active_params


# ==========================================
# 3. Visualization Functions
# ==========================================
GROUP_COLORS = {
    "1°C": "#1f77b4",   # Blue
    "5°C": "#2ca02c",   # Green
    "10°C": "#ff7f0e",  # Orange
}

WINDOW_STYLES = {
    "1": {"linestyle": "-", "alpha": 0.85, "linewidth": 1.3},
    "2": {"linestyle": "--", "alpha": 0.75, "linewidth": 1.3},
    "3": {"linestyle": ":", "alpha": 0.85, "linewidth": 1.5},
}


def plot_pulses_overlay(
    pulse_groups: Dict[str, List[Pulse]],
    removed_pulses: Optional[List[Pulse]] = None,
    time_window_s: Optional[Tuple[float, float]] = None,
    smooth_window: Optional[int] = None,
    window_label: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Overlays Temperature, AgCl CP, and Current starting from relative time t = 0,
    coloring each ΔT group distinctly and showing removed pulses in dashed crimson.
    """
    total_retained = sum(len(plist) for plist in pulse_groups.values())
    if total_retained == 0 and not removed_pulses:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (group_label, pulses) in enumerate(pulse_groups.items()):
        color = GROUP_COLORS.get(group_label, fallback_colors[idx % len(fallback_colors)])
        labeled = False

        for p in pulses:
            df = p.data.copy()
            time_vals = _to_naive_timestamps(df["datetime"])
            rel_time = (time_vals - time_vals.iloc[0]).dt.total_seconds()

            label = f"ΔT = {group_label} (n={len(pulses)})" if not labeled else ""
            labeled = True

            ax1.plot(rel_time, df["temperature"], color=color, alpha=0.75, linewidth=1.2, label=label)
            ax2.plot(rel_time, df["AgCl CP"], color=color, alpha=0.75, linewidth=1.2)

            if "current" in df.columns:
                current_vals = (
                    df["current"].rolling(window=smooth_window, center=True, min_periods=1).mean()
                    if smooth_window
                    else df["current"]
                )
                ax3.plot(rel_time, current_vals, color=color, alpha=0.75, linewidth=1.2)

    if removed_pulses:
        removed_labeled = False
        for p in removed_pulses:
            df = p.data.copy()
            time_vals = _to_naive_timestamps(df["datetime"])
            rel_time = (time_vals - time_vals.iloc[0]).dt.total_seconds()

            label = f"Removed Pulse (n={len(removed_pulses)})" if not removed_labeled else ""
            removed_labeled = True

            ax1.plot(rel_time, df["temperature"], color="crimson", linestyle="--", alpha=0.8, linewidth=1.5, label=label)
            ax2.plot(rel_time, df["AgCl CP"], color="crimson", linestyle="--", alpha=0.8, linewidth=1.5)

            if "current" in df.columns:
                current_vals = (
                    df["current"].rolling(window=smooth_window, center=True, min_periods=1).mean()
                    if smooth_window
                    else df["current"]
                )
                ax3.plot(rel_time, current_vals, color="crimson", linestyle="--", alpha=0.8, linewidth=1.5)

    if time_window_s:
        ax1.axvspan(
            time_window_s[0],
            time_window_s[1],
            color="gold",
            alpha=0.25,
            label=f"Ambient Window ({time_window_s[0]:.0f}-{time_window_s[1]:.0f}s)",
        )
        ax2.axvspan(time_window_s[0], time_window_s[1], color="gold", alpha=0.25)
        ax3.axvspan(time_window_s[0], time_window_s[1], color="gold", alpha=0.25)

    title_prefix = f"[Window {window_label}] " if window_label else ""
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"{title_prefix}Closed-Loop Pulse Overlay — Retained: {total_retained}, Removed: {len(removed_pulses or [])}"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", frameon=True)

    ax2.set_ylabel("AgCl Potential (mV)")
    ax2.grid(True, alpha=0.3)

    curr_suffix = f" (Smoothed, w={smooth_window})" if smooth_window else ""
    ax3.set_ylabel(f"Current{curr_suffix}")
    ax3.set_xlabel("Relative Time from Pulse Start (s)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved combined overlay plot: {save_path}")
    plt.show(block=False)


def plot_closed_loop_timeseries(
    pulse_groups: Dict[str, List[Pulse]],
    removed_pulses: Optional[List[Pulse]] = None,
    filter_params: Optional[dict] = None,
    window_label: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plots a 2-panel time series (Temperature, AgCl CP) with broken x-axes between
    ambient temperature setpoints, coloring each ΔT group distinctly and highlighting
    removed pulses in crimson.
    """
    removed_pulses = removed_pulses or []
    retained_pulses = [p for plist in pulse_groups.values() for p in plist]

    if not retained_pulses and not removed_pulses:
        print(f"No closed-loop pulses available to plot for Window {window_label}.")
        return

    pulse_group_map = {
        p.pulse_index: group_label
        for group_label, plist in pulse_groups.items()
        for p in plist
    }
    removed_indices = {p.pulse_index for p in removed_pulses}

    all_pulses = sorted(retained_pulses + removed_pulses, key=lambda p: p.pulse_index)

    setpoint_groups: Dict[Optional[float], List[Pulse]] = {}
    for p in all_pulses:
        sp_key = (
            round(p.ambient_temp_setpoint_c, 2)
            if p.ambient_temp_setpoint_c is not None
            else None
        )
        setpoint_groups.setdefault(sp_key, []).append(p)

    ordered_setpoints = sorted(
        setpoint_groups.keys(),
        key=lambda sp: _to_naive_timestamps(setpoint_groups[sp][0].data["datetime"]).iloc[0],
    )
    n_cols = len(ordered_setpoints)

    width_ratios = []
    for sp in ordered_setpoints:
        sp_start = min(_to_naive_timestamps(p.data["datetime"]).iloc[0] for p in setpoint_groups[sp])
        sp_end = max(_to_naive_timestamps(p.data["datetime"]).iloc[-1] for p in setpoint_groups[sp])
        duration_s = max((sp_end - sp_start).total_seconds(), 60.0)
        width_ratios.append(duration_s)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(14, 6.8),
        sharey="row",
        sharex="col",
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05, "hspace": 0.08},
    )

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    legend_tracker = {g: False for g in pulse_groups}
    legend_tracker["removed"] = False

    for col_idx, sp_val in enumerate(ordered_setpoints):
        ax1 = axes[0, col_idx]
        ax2 = axes[1, col_idx]
        sp_pulses = setpoint_groups[sp_val]

        for p in sp_pulses:
            df = p.data.copy()
            time_vals = _to_naive_timestamps(df["datetime"])
            temp_vals = df["temperature"]
            cp_vals = df["AgCl CP"]

            is_removed = p.pulse_index in removed_indices
            if is_removed:
                color = "crimson"
                linestyle = "--"
                alpha = 0.85
                linewidth = 1.4
                label = "Removed Pulse" if not legend_tracker["removed"] else ""
                legend_tracker["removed"] = True
            else:
                g_label = pulse_group_map[p.pulse_index]
                color = GROUP_COLORS.get(g_label, "#1f77b4")
                linestyle = "-"
                alpha = 0.75
                linewidth = 1.1
                label = f"ΔT = {g_label}" if not legend_tracker[g_label] else ""
                legend_tracker[g_label] = True

            ax1.plot(time_vals, temp_vals, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, label=label)
            ax2.plot(time_vals, cp_vals, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)

            if is_removed:
                ax1.axvspan(time_vals.iloc[0], time_vals.iloc[-1], color="crimson", alpha=0.15, zorder=0)
                ax2.axvspan(time_vals.iloc[0], time_vals.iloc[-1], color="crimson", alpha=0.15, zorder=0)

        if sp_val is None:
            col_title = "T_amb = Unknown"
        else:
            sp_fmt = f"{int(sp_val)}" if float(sp_val).is_integer() else f"{sp_val}"
            col_title = f"T_amb = {sp_fmt}°C"

        ax1.set_title(col_title, fontsize=10.5, fontweight="bold", pad=6)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=4))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=9)

        if col_idx > 0:
            ax1.spines["left"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax1.tick_params(axis="y", left=False)
            ax2.tick_params(axis="y", left=False)

        if col_idx < n_cols - 1:
            ax1.spines["right"].set_visible(False)
            ax2.spines["right"].set_visible(False)

        d = 0.015
        kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False, linewidth=1.1)
        if col_idx < n_cols - 1:
            ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        if col_idx > 0:
            ax2.plot((-d, +d), (-d, +d), **kwargs)

    axes[0, 0].set_ylabel("Temperature (°C)", fontsize=11)
    axes[1, 0].set_ylabel("AgCl Potential (mV)", fontsize=11)
    axes[0, -1].legend(loc="upper right", frameon=True)
    fig.supxlabel("Datetime (Broken by Ambient Setpoint)", fontsize=11, y=0.02)

    title_prefix = f"[Window {window_label}] " if window_label else ""
    fig.suptitle(
        f"{title_prefix}Closed-Loop Time Series — Total Pulses: {len(all_pulses)} "
        f"(Retained: {len(retained_pulses)}, Removed: {len(removed_pulses)})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    if filter_params:
        param_str = "  |  ".join(f"{k}: {v}" for k, v in filter_params.items())
        fig.text(
            0.5,
            0.935,
            f"Filter Parameters — {param_str}",
            ha="center",
            va="top",
            fontsize=9.5,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#f4f6f8",
                edgecolor="#cccccc",
            ),
        )
        fig.subplots_adjust(top=0.84, bottom=0.16, left=0.07, right=0.98)
    else:
        fig.subplots_adjust(top=0.89, bottom=0.16, left=0.07, right=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved broken time-series plot: {save_path}")

    plt.show(block=False)


def plot_windows_timeseries_overlay(
    window_pulse_groups: Dict[str, Dict[str, List[Pulse]]],
    setpoints: List[float],
    use_zeroed_cp: bool = True,
    save_path: Optional[str] = None,
):
    """Overlays filtered pulse time series across all windows (1, 2, 3) on a broken
    x-axis grouped by ambient temperature setpoint, aligning each setpoint block
    to elapsed minutes from its start (t = 0 min).
    """
    ordered_setpoints = [round(sp, 2) for sp in setpoints]
    n_cols = len(ordered_setpoints)
    if n_cols == 0:
        return

    # Organize pulses: setpoint -> window_label -> list of (dt_label, Pulse)
    sp_window_pulses: Dict[float, Dict[str, List[Tuple[str, Pulse]]]] = {
        sp: {w: [] for w in window_pulse_groups} for sp in ordered_setpoints
    }

    has_any_data = False
    for win_label, dt_groups in window_pulse_groups.items():
        for dt_label, plist in dt_groups.items():
            for p in plist:
                if p.ambient_temp_setpoint_c is None or p.data.empty:
                    continue
                sp_key = round(p.ambient_temp_setpoint_c, 2)
                if sp_key in sp_window_pulses:
                    sp_window_pulses[sp_key][win_label].append((dt_label, p))
                    has_any_data = True

    if not has_any_data:
        print("No retained pulses available across windows for time-series overlay.")
        return

    # Calculate relative time (minutes from start of each window's setpoint block)
    # and determine proportional column widths
    width_ratios = []
    for sp in ordered_setpoints:
        max_dur_min = 1.0
        for win_label, items in sp_window_pulses[sp].items():
            if not items:
                continue
            block_start = min(_to_naive_timestamps(p.data["datetime"]).iloc[0] for _, p in items)
            block_end = max(_to_naive_timestamps(p.data["datetime"]).iloc[-1] for _, p in items)
            dur_min = (block_end - block_start).total_seconds() / 60.0
            max_dur_min = max(max_dur_min, dur_min)
        width_ratios.append(max_dur_min)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(14.5, 7.0),
        sharey="row",
        sharex="col",
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.06, "hspace": 0.08},
    )

    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fallback_styles = [
        {"linestyle": "-", "alpha": 0.85, "linewidth": 1.3},
        {"linestyle": "--", "alpha": 0.75, "linewidth": 1.3},
        {"linestyle": ":", "alpha": 0.85, "linewidth": 1.5},
        {"linestyle": "-.", "alpha": 0.8, "linewidth": 1.3},
    ]

    for col_idx, sp_val in enumerate(ordered_setpoints):
        ax1 = axes[0, col_idx]
        ax2 = axes[1, col_idx]

        for w_idx, (win_label, items) in enumerate(sp_window_pulses[sp_val].items()):
            if not items:
                continue

            style = WINDOW_STYLES.get(win_label, fallback_styles[w_idx % len(fallback_styles)])
            block_start = min(_to_naive_timestamps(p.data["datetime"]).iloc[0] for _, p in items)

            for dt_label, p in items:
                df = p.data
                time_vals = _to_naive_timestamps(df["datetime"])
                rel_min = (time_vals - block_start).dt.total_seconds() / 60.0

                color = GROUP_COLORS.get(dt_label, "#1f77b4")
                cp_col = "cp_zeroed" if (use_zeroed_cp and "cp_zeroed" in df.columns) else "AgCl CP"

                ax1.plot(
                    rel_min,
                    df["temperature"],
                    color=color,
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    linewidth=style["linewidth"],
                )
                ax2.plot(
                    rel_min,
                    df[cp_col],
                    color=color,
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    linewidth=style["linewidth"],
                )

        sp_fmt = f"{int(sp_val)}" if float(sp_val).is_integer() else f"{sp_val}"
        ax1.set_title(f"T_amb = {sp_fmt}°C", fontsize=11, fontweight="bold", pad=6)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax2.grid(True, linestyle="--", alpha=0.5)

        if col_idx > 0:
            ax1.spines["left"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax1.tick_params(axis="y", left=False)
            ax2.tick_params(axis="y", left=False)

        if col_idx < n_cols - 1:
            ax1.spines["right"].set_visible(False)
            ax2.spines["right"].set_visible(False)

        d = 0.015
        kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False, linewidth=1.1)
        if col_idx < n_cols - 1:
            ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        if col_idx > 0:
            ax2.plot((-d, +d), (-d, +d), **kwargs)

    # Build combined legend for ΔT Colors + Window Line Styles
    dt_handles = [
        Line2D([0], [0], color=color, lw=2.0, linestyle="-", label=f"ΔT = {dt_lbl}")
        for dt_lbl, color in GROUP_COLORS.items()
    ]
    win_handles = []
    for w_idx, win_label in enumerate(window_pulse_groups.keys()):
        st = WINDOW_STYLES.get(win_label, fallback_styles[w_idx % len(fallback_styles)])
        win_handles.append(
            Line2D([0], [0], color="black", lw=st["linewidth"], linestyle=st["linestyle"], label=f"Window {win_label}")
        )

    axes[0, -1].legend(
        handles=dt_handles + win_handles,
        loc="upper right",
        frameon=True,
        ncol=2,
        fontsize=8.5,
    )

    axes[0, 0].set_ylabel("Temperature (°C)", fontsize=11)
    cp_ylabel = "Zeroed AgCl Potential (mV)" if use_zeroed_cp else "AgCl Potential (mV)"
    axes[1, 0].set_ylabel(cp_ylabel, fontsize=11)
    fig.supxlabel("Elapsed Time in Setpoint Block (min)", fontsize=11, y=0.02)

    fig.suptitle(
        "Overlay of Filtered Closed-Loop Time Series Across Windows (1, 2, 3)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved multi-window time-series overlay plot: {save_path}")

    plt.show(block=False)


# ==========================================
# 4. Execution Pipeline
# ==========================================
if __name__ == "__main__":
    JSON_FILE = "pulses.json"
    METRICS_OUTPUT_FILE = "all_window_pulse_metrics.json"
    SETPOINTS = [15.0, 25.0, 35.0]
    DELTA_TEMPS = [1.0, 5.0, 10.0]

    # Define the 3 time windows labeled "1", "2", and "3" (adjust timestamps as needed)
    TIME_WINDOWS = {
        "1": (
            pd.to_datetime("2026-09-13 15:00:00"),
            pd.to_datetime("2026-09-13 21:00:00"),
        ),
        "2": (
            pd.to_datetime("2026-09-14 10:00:00"),
            pd.to_datetime("2026-09-14 16:00:00"),
        ),
        "3": (
            pd.to_datetime("2026-09-15 05:00:00"),
            pd.to_datetime("2026-09-15 11:00:00"),
        ),
    }

    all_pulses = load_pulses_from_json(JSON_FILE)
    sp_labels = [f"{int(sp)}°C" if sp.is_integer() else f"{sp}°C" for sp in SETPOINTS]
    all_window_metrics = {}
    window_retained_groups: Dict[str, Dict[str, List[Pulse]]] = {}

    for win_label, (start_time, end_time) in TIME_WINDOWS.items():
        print(f"\n=== Processing Time Window {win_label} ({start_time} to {end_time}) ===")

        retained_groups: Dict[str, List[Pulse]] = {}
        all_window_retained: List[Pulse] = []
        retained_indices = set()
        shared_filter_params = {}

        for dt_target in DELTA_TEMPS:
            dt_label = f"{int(dt_target)}°C" if dt_target.is_integer() else f"{dt_target}°C"
            dt_group_pulses: List[Pulse] = []

            for sp in SETPOINTS:
                retained, _, active_filters = filter_pulses(
                    json_filepath=JSON_FILE,
                    ambient_temp_setpoint_c=sp,
                    delta_temp_target_c=dt_target,
                    start_time=start_time,
                    end_time=end_time,
                    pulses=all_pulses,
                )
                dt_group_pulses.extend(retained)
                shared_filter_params = active_filters

            dt_group_pulses.sort(key=lambda p: p.pulse_index)
            retained_groups[dt_label] = dt_group_pulses
            all_window_retained.extend(dt_group_pulses)
            retained_indices.update(p.pulse_index for p in dt_group_pulses)

        # Sort combined pulses chronologically, extract endpoints, and compute ΔT & ΔE metrics
        all_window_retained.sort(key=lambda p: p.pulse_index)
        all_window_metrics[win_label] = compute_pulse_metrics(all_window_retained, win_label)
        window_retained_groups[win_label] = retained_groups

        # Save combined JSON (with populated endpoint metadata & cp_zeroed series)
        save_pulses_to_json(
            all_window_retained, f"window_{win_label}_filtered_pulses.json"
        )

        # Identify pulses inside this time window that were excluded by setpoint or ΔT
        s_bound = _to_naive_timestamp(start_time)
        e_bound = _to_naive_timestamp(end_time)

        entirely_removed = []
        for p in all_pulses:
            if p.pulse_index in retained_indices or p.data.empty or "datetime" not in p.data.columns:
                continue
            ts = _to_naive_timestamps(p.data["datetime"])
            if ts.min() >= s_bound and ts.max() <= e_bound:
                entirely_removed.append(p)

        shared_filter_params["Window"] = win_label
        shared_filter_params["T_amb Setpoint"] = ", ".join(sp_labels)
        shared_filter_params["ΔT Target"] = ", ".join(retained_groups.keys())

        # 1. Per-Window 3-Panel Relative Overlay (Temperature, AgCl CP, Current)
        plot_pulses_overlay(
            pulse_groups=retained_groups,
            removed_pulses=entirely_removed,
            time_window_s=(220.0, 240.0),
            smooth_window=3,
            window_label=win_label,
            save_path=f"window_{win_label}_filtered_pulses_overlay.png",
        )

        # 2. Per-Window 2-Panel Broken-Axis Time Series Plot (Broken by Ambient Setpoint)
        plot_closed_loop_timeseries(
            pulse_groups=retained_groups,
            removed_pulses=entirely_removed,
            filter_params=shared_filter_params,
            window_label=win_label,
            save_path=f"window_{win_label}_filtered_pulses_timeseries.png",
        )

    # 3. Combined Multi-Window Broken-Axis Time Series Overlay (Windows 1, 2, 3 aligned at t = 0 min per setpoint)
    plot_windows_timeseries_overlay(
        window_pulse_groups=window_retained_groups,
        setpoints=SETPOINTS,
        use_zeroed_cp=True,
        save_path="all_windows_timeseries_overlay.png",
    )

    # Save aggregated ΔT and ΔE metrics across all 3 windows
    with open(METRICS_OUTPUT_FILE, "w") as mf:
        json.dump(all_window_metrics, mf, indent=2)
    print(f"\nSaved combined endpoint, ΔT, and ΔE metrics to '{METRICS_OUTPUT_FILE}'.")

    input("Press Enter to close all plots...")