from dataclasses import dataclass
import json
import math
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
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
) -> Tuple[List[Pulse], List[Pulse]]:
    """Filters closed-loop pulses based on metadata criteria, target steps, and time range.

    Returns (retained_pulses, removed_pulses).
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

    print(
        f"Filtering '{json_filepath}': Retained={len(retained_pulses)}, Removed={len(removed_pulses)}"
    )
    return retained_pulses, removed_pulses


# ==========================================
# 3. Visualization Functions
# ==========================================
def plot_pulses_overlay(
    pulses: List[Pulse],
    removed_pulses: List[Pulse] = None,
    time_window_s: Optional[Tuple[float, float]] = None,
    save_path: str = None,
):
    """Overlays closed-loop pulses starting from relative time t = 0 and highlights ambient calculation window."""
    if not pulses and not removed_pulses:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot retained pulses
    for p in pulses:
        df = p.data.copy()
        time_vals = df["datetime"]
        rel_time = (
            pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
        ).dt.total_seconds()

        ax1.plot(
            rel_time,
            df["temperature"],
            alpha=0.7,
            linewidth=1.2,
            label=f"Pulse #{p.pulse_index}",
        )
        ax2.plot(rel_time, df["AgCl CP"], alpha=0.7, linewidth=1.2)

    # Plot removed pulses (if any)
    if removed_pulses:
        for p in removed_pulses:
            df = p.data.copy()
            time_vals = df["datetime"]
            rel_time = (
                pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
            ).dt.total_seconds()

            ax1.plot(
                rel_time,
                df["temperature"],
                color="crimson",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
            )
            ax2.plot(
                rel_time,
                df["AgCl CP"],
                color="crimson",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
            )

    # Highlight ambient measurement window
    if time_window_s:
        ax1.axvspan(
            time_window_s[0],
            time_window_s[1],
            color="gold",
            alpha=0.25,
            label=f"Ambient Window ({time_window_s[0]:.0f}-{time_window_s[1]:.0f}s)",
        )

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"Closed-Loop Pulse Overlay — Retained: {len(pulses or [])}, Removed: {len(removed_pulses or [])}"
    )
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Relative Time from Pulse Start (s)")
    ax2.set_ylabel("AgCl Potential (mV)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved overlay plot: {save_path}")
    plt.show(block=False)


def plot_current_pulses_overlay(
    pulses: List[Pulse],
    removed_pulses: List[Pulse] = None,
    time_window_s: Optional[Tuple[float, float]] = None,
    smooth_window: Optional[int] = None,
    save_path: str = None,
):
    """Overlays current time series with optional rolling mean smoothing."""
    if not pulses and not removed_pulses:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot retained pulses
    for p in pulses:
        if "current" not in p.data.columns:
            continue
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

        ax.plot(
            rel_time,
            current_vals,
            alpha=0.7,
            linewidth=1.2,
            label=f"Pulse #{p.pulse_index}",
        )

    # Plot removed pulses (if any)
    if removed_pulses:
        for p in removed_pulses:
            if "current" not in p.data.columns:
                continue
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

            ax.plot(
                rel_time,
                current_vals,
                color="crimson",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
            )

    if time_window_s:
        ax.axvspan(
            time_window_s[0],
            time_window_s[1],
            color="gold",
            alpha=0.25,
            label=f"Ambient Window ({time_window_s[0]:.0f}-{time_window_s[1]:.0f}s)",
        )

    ax.set_xlabel("Relative Time from Pulse Start (s)")
    title_suffix = f" (Smoothed, window={smooth_window})" if smooth_window else ""
    ax.set_ylabel(f"Current{title_suffix}")
    ax.set_title(
        f"Closed-Loop Current Overlay{title_suffix} — Retained: {len(pulses or [])}, Removed: {len(removed_pulses or [])}"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved current overlay plot: {save_path}")
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
    save_path: str = None,
):
    """Plots a 3-panel continuous time series (Temperature, AgCl CP, Current) of closed-loop pulses."""
    if not retained_pulses and not removed_pulses:
        print("No closed-loop pulses available to plot.")
        return

    removed_pulses = removed_pulses or []
    removed_indices = {p.pulse_index for p in removed_pulses}
    all_pulses = sorted(
        retained_pulses + removed_pulses, key=lambda p: p.pulse_index
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]}
    )

    legend_tracker = {"retained": False, "removed": False}

    for p in all_pulses:
        df = p.data.copy()
        time_vals = pd.to_datetime(df["datetime"])
        temp_vals = df["temperature"]
        cp_vals = df["AgCl CP"]
        curr_vals = df["current"] if "current" in df.columns else None

        is_removed = p.pulse_index in removed_indices
        color = "crimson" if is_removed else "#1f77b4"
        alpha = 0.85 if is_removed else 0.7
        linewidth = 1.4 if is_removed else 1.0

        # Subplot 1: Temperature
        ax1.plot(
            time_vals,
            temp_vals,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            label="Removed Pulse"
            if is_removed and not legend_tracker["removed"]
            else (
                "Retained Pulse"
                if not is_removed and not legend_tracker["retained"]
                else ""
            ),
        )

        # Highlight removed pulse regions
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
            ax3.axvspan(
                time_vals.iloc[0],
                time_vals.iloc[-1],
                color="crimson",
                alpha=0.15,
                zorder=0,
            )

        # Subplot 2: Potential (AgCl CP)
        ax2.plot(
            time_vals,
            cp_vals,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

        # Subplot 3: Current
        if curr_vals is not None:
            ax3.plot(
                time_vals,
                curr_vals,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )

        if is_removed:
            legend_tracker["removed"] = True
        else:
            legend_tracker["retained"] = True

    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.set_title(
        f"Closed-Loop Time Series — Total Pulses: {len(all_pulses)} "
        f"(Retained: {len(retained_pulses)}, Removed: {len(removed_pulses)})",
        fontsize=12,
        pad=10,
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    ax2.set_ylabel("AgCl Potential (mV)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.set_ylabel("Current", fontsize=11)
    ax3.set_xlabel("Datetime", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.xticks(rotation=25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved continuous time-series plot: {save_path}")

    plt.show(block=False)


# ==========================================
# 4. Execution Pipeline
# ==========================================
if __name__ == "__main__":

    print("--- Processing Closed-Loop Data ---")
    closed_retained, closed_removed = filter_pulses(
        json_filepath="closed_loop_pulses_filtered.json",
        target_steps=[0, 4, 9],
        delta_temp_target_c=5.0,
        ambient_temp_setpoint_c=25.0,
        start_time=pd.to_datetime("2026-09-12 19:00:00"),
        end_time=pd.to_datetime("2026-09-14 03:30:00"),
    )

    # Save filtered subset to JSON
    save_pulses_to_json(closed_retained, "closed_loop_pulses_retained.json")

    # Relative Overlays
    plot_pulses_overlay(
        pulses=closed_retained,
        removed_pulses=closed_removed,
        time_window_s=(220.0, 240.0),
        save_path="closed_loop_overlay.png",
    )
    
    # Smoothed Current Overlay
    plot_current_pulses_overlay(
        pulses=closed_retained,
        removed_pulses=closed_removed,
        time_window_s=(220.0, 240.0),
        smooth_window=10,
        save_path="closed_loop_current_overlay_smoothed.png",
    )
    
    # Dual Y-Axis Overlay: Current vs AgCl CP
    plot_current_vs_cp_overlay(
        pulses=closed_retained,
        smooth_window=8,
        save_path="closed_loop_current_vs_cp_overlay.png",
    )

    # 3-Panel Continuous Time Series Plot
    plot_closed_loop_timeseries(
        retained_pulses=closed_retained,
        removed_pulses=closed_removed,
        save_path="closed_loop_timeseries_highlighted.png",
    )

    input("Press Enter to close all plots...")