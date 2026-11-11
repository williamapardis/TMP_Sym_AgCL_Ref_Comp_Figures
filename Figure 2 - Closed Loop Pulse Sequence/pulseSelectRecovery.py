import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================
# 1. Data Structure Definitions
# ==========================================
@dataclass
class Pulse:
    pulse_index: int
    loop_type: str  # 'open-loop' or 'closed-loop'
    time_on_s: float
    time_off_s: float
    ambient_temp_setpoint_c: float
    max_current_a: float
    data: pd.DataFrame
    calculated_ambient_temp_c: Optional[float] = None  # Saved to JSON metadata
    delta_temp_target_c: Optional[float] = None  # Added to metadata

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
        time_col = "Timestamp" if data.get("loop_type") == "open-loop" else "datetime"
        if time_col in df.columns and not pd.api.types.is_numeric_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        return cls(
            pulse_index=data.get("pulse_index", 0),
            loop_type=data.get("loop_type", ""),
            time_on_s=meta.get("time_on_s"),
            time_off_s=meta.get("time_off_s"),
            ambient_temp_setpoint_c=meta.get("ambient_temp_setpoint_c"),
            max_current_a=meta.get("max_current_a"),
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
    delta_temp_target_c: Optional[float] = None,
    ambient_temp_setpoint_c: Optional[float] = None,
    max_current_a: Optional[float] = None,
    time_on_s: Optional[float] = None,
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> Tuple[List[Pulse], List[Pulse]]:
    """Filters pulses based on metadata criteria and time range.

    Returns (retained_pulses, removed_pulses).
    """
    with open(json_filepath, "r") as f:
        raw_data = json.load(f)

    pulses = [Pulse.from_dict(p) for p in raw_data]
    retained_pulses = []
    removed_pulses = []

    for pulse in pulses:
        keep = True

        # 1. Delta Temp Target Check (with float tolerance)
        if delta_temp_target_c is not None:
            if pulse.delta_temp_target_c is None or not math.isclose(
                pulse.delta_temp_target_c, delta_temp_target_c, rel_tol=1e-4
            ):
                keep = False

        # 2. Ambient Temp Setpoint Check
        if keep and ambient_temp_setpoint_c is not None:
            if pulse.ambient_temp_setpoint_c is None or not math.isclose(
                pulse.ambient_temp_setpoint_c,
                ambient_temp_setpoint_c,
                rel_tol=1e-4,
            ):
                keep = False

        # 3. Max Current Check
        if keep and max_current_a is not None:
            if pulse.max_current_a is None or not math.isclose(
                pulse.max_current_a, max_current_a, rel_tol=1e-4
            ):
                keep = False

        # 4. Time On Check
        if keep and time_on_s is not None:
            if pulse.time_on_s is None or not math.isclose(
                pulse.time_on_s, time_on_s, rel_tol=1e-4
            ):
                keep = False

        # 5. Time Window Filtering
        if keep and (start_time is not None or end_time is not None):
            time_col = "Timestamp" if pulse.loop_type == "open-loop" else "datetime"
            if not pulse.data.empty and time_col in pulse.data.columns:
                timestamps = pd.to_datetime(pulse.data[time_col])
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
    pulses: list[Pulse],
    removed_pulses: list[Pulse] = None,
    time_window_s=None,
    save_path: str = None,
):
    """Overlays pulses starting from relative time t = 0 and highlights ambient calculation window(s)."""
    if not pulses and not removed_pulses:
        return

    loop_type = pulses[0].loop_type if pulses else removed_pulses[0].loop_type
    temp_col = "Temperature_C" if loop_type == "open-loop" else "temperature"
    cp_col = "E_mV" if loop_type == "open-loop" else "AgCl CP"
    time_col = "Timestamp" if loop_type == "open-loop" else "datetime"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot retained pulses
    for p in pulses:
        df = p.data.copy()
        time_vals = df[time_col]

        if pd.api.types.is_numeric_dtype(time_vals):
            rel_time = time_vals - time_vals.iloc[0]
        else:
            rel_time = (
                pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
            ).dt.total_seconds()

        ax1.plot(
            rel_time,
            df[temp_col],
            alpha=0.7,
            linewidth=1.2,
            label=f"Pulse #{p.pulse_index}",
        )
        ax2.plot(rel_time, df[cp_col], alpha=0.7, linewidth=1.2)

    # Plot removed pulses (if any)
    if removed_pulses:
        for p in removed_pulses:
            df = p.data.copy()
            time_vals = df[time_col]

            if pd.api.types.is_numeric_dtype(time_vals):
                rel_time = time_vals - time_vals.iloc[0]
            else:
                rel_time = (
                    pd.to_datetime(time_vals)
                    - pd.to_datetime(time_vals.iloc[0])
                ).dt.total_seconds()

            ax1.plot(
                rel_time,
                df[temp_col],
                color="crimson",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
            )
            ax2.plot(
                rel_time,
                df[cp_col],
                color="crimson",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
            )

    # Highlight ambient measurement window(s) on temperature subplot
    if time_window_s:
        if isinstance(time_window_s, dict):
            for off_dur, (w_start, w_end) in time_window_s.items():
                ax1.axvspan(
                    w_start,
                    w_end,
                    color="gold",
                    alpha=0.2,
                    label=f"Window ({off_dur:g}s OFF: {w_start:.0f}-{w_end:.0f}s)",
                )
        else:
            ax1.axvspan(
                time_window_s[0],
                time_window_s[1],
                color="gold",
                alpha=0.25,
                label=f"Ambient Window ({time_window_s[0]:.0f}-{time_window_s[1]:.0f}s)",
            )

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"Pulse Overlay ({loop_type.title()} Mode) — Retained: {len(pulses)}, Removed: {len(removed_pulses or [])}"
    )
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Relative Time from Pulse Start (s)")
    ax2.set_ylabel("Potential / CP")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved overlay plot: {save_path}")
    plt.show(block=False)


def plot_current_pulses_overlay(
    pulses: list[Pulse],
    removed_pulses: list[Pulse] = None,
    time_window_s=None,
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
        time_col = "Timestamp" if p.loop_type == "open-loop" else "datetime"
        time_vals = df[time_col]

        if pd.api.types.is_numeric_dtype(time_vals):
            rel_time = time_vals - time_vals.iloc[0]
        else:
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
            time_col = "Timestamp" if p.loop_type == "open-loop" else "datetime"
            time_vals = df[time_col]

            if pd.api.types.is_numeric_dtype(time_vals):
                rel_time = time_vals - time_vals.iloc[0]
            else:
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

    # Highlight ambient measurement window(s)
    if time_window_s:
        if isinstance(time_window_s, dict):
            for off_dur, (w_start, w_end) in time_window_s.items():
                ax.axvspan(
                    w_start,
                    w_end,
                    color="gold",
                    alpha=0.2,
                    label=f"Window ({off_dur:g}s OFF: {w_start:.0f}-{w_end:.0f}s)",
                )
        else:
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
    loop_type = pulses[0].loop_type if pulses else removed_pulses[0].loop_type
    ax.set_title(
        f"Current Overlay ({loop_type.title()} Mode){title_suffix} — Retained: {len(pulses or [])}, Removed: {len(removed_pulses or [])}"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved current overlay plot: {save_path}")
    plt.show(block=False)


def plot_current_vs_cp_overlay(
    pulses: list[Pulse],
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

        if pd.api.types.is_numeric_dtype(time_vals):
            rel_time = time_vals - time_vals.iloc[0]
        else:
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


def plot_pulses_timeseries(
    pulses: list[Pulse],
    removed_pulses: list[Pulse] = None,
    save_path: str = None,
):
    """Plots pulses continuously on their absolute timeline (datetime or total timestamp)."""
    if not pulses and not removed_pulses:
        return

    loop_type = pulses[0].loop_type if pulses else removed_pulses[0].loop_type
    temp_col = "Temperature_C" if loop_type == "open-loop" else "temperature"
    cp_col = "E_mV" if loop_type == "open-loop" else "AgCl CP"
    time_col = "Timestamp" if loop_type == "open-loop" else "datetime"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot retained pulses in continuous time
    if pulses:
        for p in pulses:
            df = p.data
            ax1.plot(
                df[time_col],
                df[temp_col],
                label=f"Pulse #{p.pulse_index}",
                alpha=0.85,
            )
            ax2.plot(df[time_col], df[cp_col], alpha=0.85)

    # Plot removed pulses in continuous time
    if removed_pulses:
        for p in removed_pulses:
            df = p.data
            ax1.plot(
                df[time_col],
                df[temp_col],
                color="crimson",
                linestyle="--",
                alpha=0.5,
                label=f"Pulse #{p.pulse_index} (Removed)",
            )
            ax2.plot(
                df[time_col],
                df[cp_col],
                color="crimson",
                linestyle="--",
                alpha=0.5,
            )

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"Continuous Time Series ({loop_type.title()} Mode) — Retained: {len(pulses or [])}, Removed: {len(removed_pulses or [])}"
    )
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time / Datetime")
    ax2.set_ylabel("Potential / CP")
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved time-series plot: {save_path}")
    plt.show(block=False)


def plot_combined_overlay(
    open_pulses: list[Pulse],
    closed_pulses: list[Pulse],
    save_path: str = None,
):
    """Overlays open and closed loop pulses on relative time scale t = 0.

    Top subplot: Temperature (°C) on shared y-axis.
    Bottom subplot: Potential / CP on dual twin y-axes (left: Open Loop E_mV, right: Closed Loop AgCl CP).
    """
    if not open_pulses and not closed_pulses:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax2_twin = ax2.twinx()  # Dedicated right y-axis for closed-loop CP

    open_color = "tab:blue"
    closed_color = "tab:orange"

    open_temp_labeled = False
    closed_temp_labeled = False
    open_cp_labeled = False
    closed_cp_labeled = False

    for p in open_pulses:
        df = p.data.copy()
        time_vals = df["Timestamp"]

        if pd.api.types.is_numeric_dtype(time_vals):
            rel_time = time_vals - time_vals.iloc[0]
        else:
            rel_time = (
                pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
            ).dt.total_seconds()

        ax1.plot(
            rel_time,
            df["Temperature_C"],
            color=open_color,
            alpha=0.5,
            linewidth=1.2,
            label="Open Loop" if not open_temp_labeled else "",
        )
        open_temp_labeled = True

        ax2.plot(
            rel_time,
            df["E_mV"],
            color=open_color,
            alpha=0.5,
            linewidth=1.2,
            label="Open Loop (E_mV)" if not open_cp_labeled else "",
        )
        open_cp_labeled = True

    for p in closed_pulses:
        df = p.data.copy()
        time_vals = df["datetime"]

        if pd.api.types.is_numeric_dtype(time_vals):
            rel_time = time_vals - time_vals.iloc[0]
        else:
            rel_time = (
                pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
            ).dt.total_seconds()

        ax1.plot(
            rel_time,
            df["temperature"],
            color=closed_color,
            alpha=0.5,
            linewidth=1.2,
            label="Closed Loop" if not closed_temp_labeled else "",
        )
        closed_temp_labeled = True

        ax2_twin.plot(
            rel_time,
            df["AgCl CP"],
            color=closed_color,
            alpha=0.5,
            linewidth=1.2,
            label="Closed Loop (AgCl CP)" if not closed_cp_labeled else "",
        )
        closed_cp_labeled = True

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"Combined Open vs. Closed Loop Overlay (Open: {len(open_pulses)}, Closed: {len(closed_pulses)})"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.set_xlabel("Relative Time from Pulse Start (s)")
    ax2.set_ylabel("Open Loop Potential (E_mV)", color=open_color)
    ax2.tick_params(axis="y", labelcolor=open_color)
    ax2.grid(True, alpha=0.3)

    ax2_twin.set_ylabel("Closed Loop AgCl CP", color=closed_color)
    ax2_twin.tick_params(axis="y", labelcolor=closed_color)

    lines_left, labels_left = ax2.get_legend_handles_labels()
    lines_right, labels_right = ax2_twin.get_legend_handles_labels()
    ax2.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper right",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved combined overlay plot: {save_path}")
    plt.show(block=False)


# ==========================================
# 4. Execution Pipeline
# ==========================================
if __name__ == "__main__":

    # --- A. CLOSED LOOP RUN ---
    print("--- Processing Closed-Loop Data (ambient_temp_setpoint_c == 25) ---")
    closed_retained, closed_removed = filter_pulses(
        json_filepath="closed_loop_pulses_filtered.json",
        delta_temp_target_c=10.0,
        ambient_temp_setpoint_c=25,
        start_time=pd.to_datetime("2026-09-13 21:20:00"),
        end_time=pd.to_datetime("2026-09-14 03:30:00"),
    )

    save_pulses_to_json(closed_retained, "closed_loop_pulses_retained.json")

    plot_pulses_overlay(
        pulses=closed_retained,
        removed_pulses=closed_removed,
        save_path="closed_loop_overlay.png",
    )
    
    # Smoothed Current Overlay (adjust smooth_window to change moving average window size)
    plot_current_pulses_overlay(
        pulses=closed_retained,
        smooth_window=10,
        save_path="closed_loop_current_overlay_smoothed.png",
    )
    
    # Dual Y-Axis Overlay: Current vs AgCl CP
    plot_current_vs_cp_overlay(
        pulses=closed_retained,
        smooth_window=8,
        save_path="closed_loop_current_vs_cp_overlay.png",
    )

    plot_pulses_timeseries(pulses=closed_retained)

    # --- B. OPEN LOOP RUN ---
    print("--- Processing Open-Loop Data (ambient_temp_setpoint_c == 25) ---")
    open_retained, open_removed = filter_pulses(
        json_filepath="open_loop_pulses_filtered.json",
        max_current_a=0.654,
        ambient_temp_setpoint_c=25,
        time_on_s=60,
        start_time=pd.to_datetime("2026-09-03 18:00:00"),
        end_time=pd.to_datetime("2026-09-04 14:00:00"),
    )

    plot_pulses_overlay(pulses=open_retained)
    plot_pulses_timeseries(pulses=open_retained)

    # --- C. COMBINED OVERLAY RUN ---
    print("--- Generating Combined Open vs. Closed Loop Overlay ---")
    plot_combined_overlay(
        open_pulses=open_retained,
        closed_pulses=closed_retained,
        save_path="combined_open_closed_overlay.png",
    )

    input("Press Enter to close all plots...")