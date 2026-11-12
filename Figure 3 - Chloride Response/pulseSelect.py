from dataclasses import dataclass
import json
import math
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    calculated_ambient_temp_c: Optional[float] = None  # Saved to JSON metadata
    delta_temp_target_c: Optional[float] = None  # Added target temperature delta metadata

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
            self.calculated_ambient_temp_c = float(
                self.calculated_ambient_temp_c
            )
        if self.delta_temp_target_c is not None:
            self.delta_temp_target_c = float(self.delta_temp_target_c)

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
            data=df,
        )


# ==========================================
# 2. I/O Helpers & Filtering Engine
# ==========================================
def save_pulses_to_json(pulses: List[Pulse], filepath: str) -> None:
    """Saves a list of Pulse objects to a JSON file matching the exact schema."""
    json_data = [p.to_dict() for p in pulses]
    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=4, default=str)
    print(f"Saved {len(pulses)} pulses to '{filepath}'.")


def filter_pulses(
    json_filepath: str,
    target_steps: Optional[List[int]] = None,
    delta_temp_target_c: Optional[float] = None,
    ambient_temp_setpoint_c: Optional[float] = None,
    max_current_a: Optional[float] = None,
    time_on_s: Optional[float] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> Tuple[List[Pulse], List[Pulse], dict]:
    """Filters closed-loop pulses based on metadata criteria, target steps, and time range.

    Returns (retained_pulses, removed_pulses, active_filter_params).
    """
    with open(json_filepath, "r") as f:
        raw_data = json.load(f)

    pulses = [Pulse.from_dict(p) for p in raw_data]
    retained_pulses = []
    removed_pulses = []

    target_step_set = set(target_steps) if target_steps is not None else None

    for pulse in pulses:
        keep = True

        # 1. Target Steps Check
        if target_step_set is not None and pulse.step not in target_step_set:
            keep = False

        # 2. Delta Temp Target Check (with float tolerance)
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

        # 6. Time Window Filtering
        if keep and (start_time is not None or end_time is not None):
            if not pulse.data.empty and "datetime" in pulse.data.columns:
                timestamps = pd.to_datetime(pulse.data["datetime"])
                pulse_start = timestamps.min()
                pulse_end = timestamps.max()

                if start_time is not None and pulse_start < start_time:
                    keep = False
                if end_time is not None and pulse_end > end_time:
                    keep = False

        if keep:
            retained_pulses.append(pulse)
        else:
            removed_pulses.append(pulse)

    # Record only active filter parameters for plot annotations
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
    if start_time is not None:
        active_params["Start"] = start_time.strftime("%Y-%m-%d %H:%M")
    if end_time is not None:
        active_params["End"] = end_time.strftime("%Y-%m-%d %H:%M")

    print(
        f"Filtering '{json_filepath}': Retained={len(retained_pulses)}, Removed={len(removed_pulses)}"
    )
    return retained_pulses, removed_pulses, active_params


# ==========================================
# 3. Visualization Functions
# ==========================================
def plot_pulses_overlay(
    pulses: List[Pulse],
    removed_pulses: List[Pulse] = None,
    time_window_s: Optional[Tuple[float, float]] = None,
    smooth_window: Optional[int] = None,
    save_path: str = None,
):
    """Overlays Temperature, AgCl CP, and Current starting from relative time t = 0."""
    if not pulses and not removed_pulses:
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    legend_tracker = {"retained": False, "removed": False}

    # Plot retained pulses
    for p in pulses:
        df = p.data.copy()
        time_vals = df["datetime"]
        rel_time = (
            pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
        ).dt.total_seconds()

        label = ""
        if not legend_tracker["retained"]:
            label = "Retained Pulse"
            legend_tracker["retained"] = True

        ax1.plot(
            rel_time,
            df["temperature"],
            alpha=0.7,
            linewidth=1.2,
            label=label,
        )
        ax2.plot(rel_time, df["AgCl CP"], alpha=0.7, linewidth=1.2)

        if "current" in df.columns:
            current_vals = (
                df["current"].rolling(window=smooth_window, center=True, min_periods=1).mean()
                if smooth_window
                else df["current"]
            )
            ax3.plot(rel_time, current_vals, alpha=0.7, linewidth=1.2)

    # Plot removed pulses (if any)
    if removed_pulses:
        for p in removed_pulses:
            df = p.data.copy()
            time_vals = df["datetime"]
            rel_time = (
                pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
            ).dt.total_seconds()

            label = ""
            if not legend_tracker["removed"]:
                label = "Removed Pulse"
                legend_tracker["removed"] = True

            ax1.plot(
                rel_time,
                df["temperature"],
                color="crimson",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
                label=label,
            )
            ax2.plot(
                rel_time,
                df["AgCl CP"],
                color="crimson",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
            )

            if "current" in df.columns:
                current_vals = (
                    df["current"].rolling(window=smooth_window, center=True, min_periods=1).mean()
                    if smooth_window
                    else df["current"]
                )
                ax3.plot(
                    rel_time,
                    current_vals,
                    color="crimson",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                )

    # Highlight ambient measurement window across all 3 subplots
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

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"Closed-Loop Pulse Overlay — Retained: {len(pulses or [])}, Removed: {len(removed_pulses or [])}"
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


def plot_current_vs_cp_overlay(
    pulses: List[Pulse],
    smooth_window: Optional[int] = None,
    save_path: str = None,
):
    """Overlays Current and Closed-Loop AgCl CP on dual twin y-axes against relative time."""
    valid_pulses = [p for p in pulses if "current" in p.data.columns and "AgCl CP" in p.data.columns]
    if not valid_pulses:
        print("Warning: No pulses with both 'current' and 'AgCl CP' available to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    current_color = "tab:red"
    cp_color = "tab:orange"

    curr_labeled = False
    cp_labeled = False

    for p in valid_pulses:
        df = p.data.copy()
        time_vals = df["datetime"]
        rel_time = (
            pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
        ).dt.total_seconds()

        current_vals = (
            df["current"].rolling(window=smooth_window, center=True, min_periods=1).mean()
            if smooth_window
            else df["current"]
        )

        ax1.plot(
            rel_time,
            current_vals,
            color=current_color,
            alpha=0.6,
            linewidth=1.2,
            label="Current" if not curr_labeled else "",
        )
        curr_labeled = True

        ax2.plot(
            rel_time,
            df["AgCl CP"],
            color=cp_color,
            alpha=0.6,
            linewidth=1.2,
            label="AgCl CP" if not cp_labeled else "",
        )
        cp_labeled = True

    ax1.set_xlabel("Relative Time from Pulse Start (s)")
    ax1.set_ylabel("Current", color=current_color)
    ax1.tick_params(axis="y", labelcolor=current_color)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Closed Loop AgCl CP", color=cp_color)
    ax2.tick_params(axis="y", labelcolor=cp_color)

    title_smooth = f" (Smoothed window={smooth_window})" if smooth_window else ""
    plt.title(f"Closed-Loop Overlay: Current vs. AgCl CP{title_smooth} (Pulses: {len(valid_pulses)})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Current vs. CP overlay plot: {save_path}")
    plt.show(block=False)


def plot_closed_loop_timeseries(
    retained_pulses: List[Pulse],
    removed_pulses: List[Pulse] = None,
    filter_params: Optional[dict] = None,
    save_path: str = None,
):
    """Plots a 2-panel time series (Temperature, AgCl CP) with broken x-axes between steps."""
    if not retained_pulses and not removed_pulses:
        print("No closed-loop pulses available to plot.")
        return

    removed_pulses = removed_pulses or []
    removed_indices = {p.pulse_index for p in removed_pulses}
    all_pulses = sorted(
        retained_pulses + removed_pulses, key=lambda p: p.pulse_index
    )

    # Group pulses by step in chronological order
    step_groups = {}
    for p in all_pulses:
        step_groups.setdefault(p.step, []).append(p)

    ordered_steps = sorted(
        step_groups.keys(),
        key=lambda s: pd.to_datetime(step_groups[s][0].data["datetime"].iloc[0]),
    )
    n_steps = len(ordered_steps)

    # Scale each column width proportionally to the step's time duration
    width_ratios = []
    for s in ordered_steps:
        s_start = min(pd.to_datetime(p.data["datetime"].iloc[0]) for p in step_groups[s])
        s_end = max(pd.to_datetime(p.data["datetime"].iloc[-1]) for p in step_groups[s])
        duration_s = max((s_end - s_start).total_seconds(), 60.0)
        width_ratios.append(duration_s)

    fig, axes = plt.subplots(
        2,
        n_steps,
        figsize=(14, 6.8),
        sharey="row",
        sharex="col",
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.05, "hspace": 0.08},
    )

    # Normalize axes shape to 2D array (2, n_steps) even if n_steps == 1
    if n_steps == 1:
        axes = axes.reshape(2, 1)

    legend_tracker = {"retained": False, "removed": False}

    for col_idx, step_val in enumerate(ordered_steps):
        ax1 = axes[0, col_idx]
        ax2 = axes[1, col_idx]
        step_pulses = step_groups[step_val]

        for p in step_pulses:
            df = p.data.copy()
            time_vals = pd.to_datetime(df["datetime"])
            temp_vals = df["temperature"]
            cp_vals = df["AgCl CP"]

            is_removed = p.pulse_index in removed_indices
            color = "crimson" if is_removed else "#1f77b4"
            alpha = 0.85 if is_removed else 0.7
            linewidth = 1.4 if is_removed else 1.0

            label = ""
            if is_removed and not legend_tracker["removed"]:
                label = "Removed Pulse"
                legend_tracker["removed"] = True
            elif not is_removed and not legend_tracker["retained"]:
                label = "Retained Pulse"
                legend_tracker["retained"] = True

            # Row 0: Temperature
            ax1.plot(
                time_vals,
                temp_vals,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                label=label,
            )

            # Row 1: AgCl Potential
            ax2.plot(
                time_vals,
                cp_vals,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

            if is_removed:
                ax1.axvspan(
                    time_vals.iloc[0],
                    time_vals.iloc[-1],
                    color="crimson",
                    alpha=0.15,
                    zorder=0,
                )
                ax2.axvspan(
                    time_vals.iloc[0],
                    time_vals.iloc[-1],
                    color="crimson",
                    alpha=0.15,
                    zorder=0,
                )

        # Column Header & Grid Styling
        ax1.set_title(f"Step {step_val}", fontsize=10.5, fontweight="bold", pad=6)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Format bottom x-axis timestamps cleanly per step block
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=4))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=9)

        # Style broken axis spines & diagonal break slash markers
        if col_idx > 0:
            ax1.spines["left"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax1.tick_params(axis="y", left=False)
            ax2.tick_params(axis="y", left=False)

        if col_idx < n_steps - 1:
            ax1.spines["right"].set_visible(False)
            ax2.spines["right"].set_visible(False)

        d = 0.015  # Size of diagonal break marks in axes coordinates
        kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False, linewidth=1.1)
        if col_idx < n_steps - 1:
            ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        if col_idx > 0:
            ax2.plot((-d, +d), (-d, +d), **kwargs)

    # Shared Y-axis labels on left column & legend on right column
    axes[0, 0].set_ylabel("Temperature (°C)", fontsize=11)
    axes[1, 0].set_ylabel("AgCl Potential (mV)", fontsize=11)
    axes[0, -1].legend(loc="upper right", frameon=True)
    fig.supxlabel("Datetime (Broken by Step)", fontsize=11, y=0.02)

    # Figure Title & Filtered Parameters Annotation
    fig.suptitle(
        f"Closed-Loop Time Series — Total Pulses: {len(all_pulses)} "
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


# ==========================================
# 4. Execution Pipeline
# ==========================================
if __name__ == "__main__":

    print("--- Processing Closed-Loop Data ---")
    closed_retained, closed_removed, active_filters = filter_pulses(
        json_filepath="pulses.json",
        target_steps=[0, 4, 9],
        delta_temp_target_c=5.0,
        ambient_temp_setpoint_c=25.0,
        start_time=pd.to_datetime("2026-09-12 19:00:00"),
        end_time=pd.to_datetime("2026-09-14 03:30:00"),
    )

    # Save filtered subset to JSON
    save_pulses_to_json(closed_retained, "filtered_pulses.json")

    # Combined 3-Panel Relative Overlay (Temperature, AgCl CP, Current)
    plot_pulses_overlay(
        pulses=closed_retained,
        removed_pulses=closed_removed,
        time_window_s=(220.0, 240.0),
        smooth_window=3,
        save_path="filtered_pulses.png",
    )

    # 2-Panel Broken-Axis Time Series Plot with Filter Parameters
    plot_closed_loop_timeseries(
        retained_pulses=closed_retained,
        removed_pulses=closed_removed,
        filter_params=active_filters,
        save_path="filtered_pulse_timeseries.png",
    )

    input("Press Enter to close all plots...")