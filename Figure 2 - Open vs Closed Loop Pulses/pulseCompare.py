from dataclasses import dataclass
import json
import math
import matplotlib.pyplot as plt
import pandas as pd


# ==========================================
# 1. Pulse Data Structure & Serialization
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
    calculated_ambient_temp_c: float = None  # Saved to JSON metadata

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
            },
            "time_series": self.data.to_dict(orient="list"),
        }


def save_pulses_to_json(pulses: list[Pulse], json_filepath: str):
    """Exports a list of Pulse objects to a JSON file."""
    pulses_data = [pulse.to_dict() for pulse in pulses]
    with open(json_filepath, "w") as f:
        json.dump(pulses_data, f, indent=4, default=str)
    print(f"Successfully exported {len(pulses)} pulses to '{json_filepath}'.")


def load_pulses_from_json(json_filepath: str) -> list[Pulse]:
    """Loads a JSON file back into a list of Pulse objects."""
    with open(json_filepath, "r") as f:
        data = json.load(f)

    pulses = []
    for item in data:
        pulse = Pulse(
            pulse_index=item["pulse_index"],
            loop_type=item["loop_type"],
            time_on_s=item["metadata"]["time_on_s"],
            time_off_s=item["metadata"]["time_off_s"],
            ambient_temp_setpoint_c=item["metadata"]["ambient_temp_setpoint_c"],
            max_current_a=item["metadata"]["max_current_a"],
            calculated_ambient_temp_c=item["metadata"].get(
                "calculated_ambient_temp_c"
            ),
            data=pd.DataFrame(item["time_series"]),
        )
        pulses.append(pulse)
    return pulses


# ==========================================
# 2. Filtering Helper Function
# ==========================================
def filter_pulses(
    json_filepath: str,
    start_time: pd.Timestamp = None,
    end_time: pd.Timestamp = None,
    tolerance: float = 1e-5,
    **metadata_filters,
) -> tuple[list[Pulse], list[Pulse]]:
    """Loads pulses from JSON and filters them based on datetime windows and Pulse metadata attributes."""
    all_pulses = load_pulses_from_json(json_filepath)
    retained_pulses = []
    removed_pulses = []

    for pulse_obj in all_pulses:
        keep = True

        # 1. Filter by datetime window (if start_time or end_time provided)
        if start_time is not None or end_time is not None:
            df_pulse = pulse_obj.data
            time_col = (
                "Timestamp"
                if pulse_obj.loop_type == "open-loop"
                else "datetime"
            )
            df_pulse[time_col] = pd.to_datetime(df_pulse[time_col])

            mask = pd.Series(True, index=df_pulse.index)
            if start_time is not None:
                mask &= df_pulse[time_col] >= start_time
            if end_time is not None:
                mask &= df_pulse[time_col] <= end_time

            if not mask.any():
                keep = False

        # 2. Filter by metadata/property attributes (with float conversion safety)
        if keep:
            for attr_name, target_val in metadata_filters.items():
                if not hasattr(pulse_obj, attr_name):
                    raise AttributeError(
                        f"Pulse object has no attribute '{attr_name}'"
                    )

                val = getattr(pulse_obj, attr_name)

                try:
                    val_num = float(val) if val is not None else None
                    target_num = (
                        float(target_val) if target_val is not None else None
                    )

                    if val_num is not None and target_num is not None:
                        if not math.isclose(
                            val_num, target_num, abs_tol=tolerance
                        ):
                            keep = False
                            break
                    elif val != target_val:
                        keep = False
                        break
                except (ValueError, TypeError):
                    if str(val) != str(target_val):
                        keep = False
                        break

        if keep:
            retained_pulses.append(pulse_obj)
        else:
            removed_pulses.append(pulse_obj)

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

    # Color palettes for distinguishability
    open_color = "tab:blue"
    closed_color = "tab:orange"

    open_temp_labeled = False
    closed_temp_labeled = False
    open_cp_labeled = False
    closed_cp_labeled = False

    # 1. Plot Open Loop Pulses
    for p in open_pulses:
        df = p.data.copy()
        time_vals = df["Timestamp"]

        if pd.api.types.is_numeric_dtype(time_vals):
            rel_time = time_vals - time_vals.iloc[0]
        else:
            rel_time = (
                pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
            ).dt.total_seconds()

        # Top Subplot: Temperature
        ax1.plot(
            rel_time,
            df["Temperature_C"],
            color=open_color,
            alpha=0.5,
            linewidth=1.2,
            label="Open Loop" if not open_temp_labeled else "",
        )
        open_temp_labeled = True

        # Bottom Subplot: Open Loop Potential (Left Y-Axis)
        ax2.plot(
            rel_time,
            df["E_mV"],
            color=open_color,
            alpha=0.5,
            linewidth=1.2,
            label="Open Loop (E_mV)" if not open_cp_labeled else "",
        )
        open_cp_labeled = True

    # 2. Plot Closed Loop Pulses
    for p in closed_pulses:
        df = p.data.copy()
        time_vals = df["datetime"]

        if pd.api.types.is_numeric_dtype(time_vals):
            rel_time = time_vals - time_vals.iloc[0]
        else:
            rel_time = (
                pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
            ).dt.total_seconds()

        # Top Subplot: Temperature
        ax1.plot(
            rel_time,
            df["temperature"],
            color=closed_color,
            alpha=0.5,
            linewidth=1.2,
            label="Closed Loop" if not closed_temp_labeled else "",
        )
        closed_temp_labeled = True

        # Bottom Subplot: Closed Loop CP (Right Twin Y-Axis)
        ax2_twin.plot(
            rel_time,
            df["AgCl CP"],
            color=closed_color,
            alpha=0.5,
            linewidth=1.2,
            label="Closed Loop (AgCl CP)" if not closed_cp_labeled else "",
        )
        closed_cp_labeled = True

    # Styling Top Subplot
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"Combined Open vs. Closed Loop Overlay (Open: {len(open_pulses)}, Closed: {len(closed_pulses)})"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Styling Bottom Subplot (Dual Y-Axes)
    ax2.set_xlabel("Relative Time from Pulse Start (s)")
    ax2.set_ylabel("Open Loop Potential (E_mV)", color=open_color)
    ax2.tick_params(axis="y", labelcolor=open_color)
    ax2.grid(True, alpha=0.3)

    ax2_twin.set_ylabel("Closed Loop AgCl CP", color=closed_color)
    ax2_twin.tick_params(axis="y", labelcolor=closed_color)

    # Combine legends from both bottom axes
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

# --- A. CLOSED LOOP RUN ---
print("--- Processing Closed-Loop Data (ambient_temp_setpoint_c == 25) ---")
closed_retained, closed_removed = filter_pulses(
    json_filepath="closed_loop_pulses_filtered.json",
    ambient_temp_setpoint_c=25,
    start_time=pd.to_datetime("2026-09-13 12:45:00"),
    end_time=pd.to_datetime("2026-09-13 22:45:00"),
)

plot_pulses_overlay(pulses=closed_retained)
plot_pulses_timeseries(pulses=closed_retained)

# --- B. OPEN LOOP RUN ---
print("--- Processing Open-Loop Data (ambient_temp_setpoint_c == 25) ---")
open_retained, open_removed = filter_pulses(
    json_filepath="open_loop_pulses_filtered.json",
    ambient_temp_setpoint_c=25,
    time_on_s=60,
    start_time=pd.to_datetime("2026-09-03 20:00:00"),
    end_time=pd.to_datetime("2026-09-04 00:00:00"),
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

input("Press Enter to close all plots...")  # Keeps the script alive