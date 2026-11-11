from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import numpy as np
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

    def __post_init__(self):
        """Ensures numeric metadata fields are strictly float types regardless of source data."""
        self.time_on_s = float(self.time_on_s)
        self.time_off_s = float(self.time_off_s)
        self.ambient_temp_setpoint_c = (
            float(self.ambient_temp_setpoint_c)
            if self.ambient_temp_setpoint_c is not None
            else None
        )
        self.max_current_a = float(self.max_current_a)
        if self.calculated_ambient_temp_c is not None:
            self.calculated_ambient_temp_c = float(
                self.calculated_ambient_temp_c
            )

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
    print(
        f"Successfully exported {len(pulses)} pulses to '{json_filepath}'."
    )


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
# 2. File Parser (Pulse Definition: ON -> OFF)
# ==========================================
def parse_pulses(file_path: str, loop_type: str) -> list[Pulse]:
    """Parses open-loop or closed-loop text files into structured Pulse objects.

    Pulse definition: One full ON cycle (1) followed by one full OFF cycle (0).
    """
    df = pd.read_csv(file_path, sep=r"\t|,", engine="python")
    df.columns = df.columns.str.strip()

    state_col = "heater_state" if loop_type == "open-loop" else "PID enable"

    if state_col not in df.columns:
        possible_matches = [
            c for c in df.columns if "PID" in c or "enable" in c or "heater" in c
        ]
        raise KeyError(
            f"Expected state column '{state_col}' not found in '{file_path}'.\n"
            f"Detected columns: {list(df.columns)}\n"
            f"Possible candidates: {possible_matches}"
        )

    df["state_change"] = df[state_col].diff().fillna(0) != 0
    df["cycle_id"] = df["state_change"].cumsum()

    pulses = []
    pulse_counter = 0

    cycles = [group for _, group in df.groupby("cycle_id")]

    i = 0
    while i < len(cycles) - 1:
        on_block = cycles[i]
        off_block = cycles[i + 1]

        if (
            on_block[state_col].iloc[0] == 1
            and off_block[state_col].iloc[0] == 0
        ):
            pulse_df = pd.concat([on_block, off_block]).drop(
                columns=["state_change", "cycle_id"]
            )

            if loop_type == "open-loop":
                ts_data = pulse_df[["Timestamp", "Temperature_C", "E_mV"]]
                time_on_s = float(pulse_df["on_period_ms"].iloc[0]) / 1000.0
                time_off_s = float(pulse_df["off_period_ms"].iloc[0]) / 1000.0
                temp_sp = float(pulse_df["Temperature_Setpoint_C"].iloc[0])
                max_curr_a = float(pulse_df["Current_mA"].iloc[0]) / 1000.0
            else:
                ts_data = pulse_df[["datetime", "temperature", "AgCl CP"]]
                time_on_s = float(pulse_df["time on"].iloc[0])
                time_off_s = float(pulse_df["time off"].iloc[0])
                temp_sp = float(pulse_df["temperature setpoint"].iloc[0])
                max_curr_a = 925.0 / 1000.0

            pulse = Pulse(
                pulse_index=pulse_counter,
                loop_type=loop_type,
                time_on_s=time_on_s,
                time_off_s=time_off_s,
                ambient_temp_setpoint_c=temp_sp,
                max_current_a=max_curr_a,
                data=ts_data.reset_index(drop=True),
            )
            pulses.append(pulse)
            pulse_counter += 1
            i += 2
        else:
            i += 1

    return pulses


# ==========================================
# 3. Ambient Temperature Calculation & Filtering
# ==========================================
def calculate_ambient_temperatures(
    pulses: list[Pulse], time_window_s
) -> list[Pulse]:
    """Calculates ambient temperature as mean temp within a relative window [t_start, t_end].

    time_window_s can be:
    - A single tuple (t_start, t_end)
    - A dict mapping time_off_s -> (t_start, t_end)
    """
    for p in pulses:
        df = p.data
        t_col = "Timestamp" if p.loop_type == "open-loop" else "datetime"
        temp_col = (
            "Temperature_C" if p.loop_type == "open-loop" else "temperature"
        )

        if isinstance(time_window_s, dict):
            off_key = min(
                time_window_s.keys(), key=lambda k: abs(k - p.time_off_s)
            )
            t_start, t_end = time_window_s[off_key]
        else:
            t_start, t_end = time_window_s

        time_vals = df[t_col]
        if pd.api.types.is_numeric_dtype(time_vals):
            rel_time = time_vals - time_vals.iloc[0]
        else:
            rel_time = (
                pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
            ).dt.total_seconds()

        window_mask = (rel_time >= t_start) & (rel_time <= t_end)
        window_temps = df.loc[window_mask, temp_col]

        if not window_temps.empty:
            p.calculated_ambient_temp_c = float(window_temps.mean())
        else:
            p.calculated_ambient_temp_c = float(df[temp_col].iloc[-1])
            print(
                f"[{p.loop_type.upper()}] Warning: Pulse #{p.pulse_index} window [{t_start:.1f}, {t_end:.1f}]s outside bounds. Used final sample."
            )

    return pulses


def filter_pulses_by_ambient_setpoint_delta(
    pulses: list[Pulse], max_delta_c: float
) -> tuple[list[Pulse], list[Pulse]]:
    """Filters out pulses where |calculated_ambient_temp_c - ambient_temp_setpoint_c| > max_delta_c."""
    retained = []
    removed = []

    for p in pulses:
        if p.calculated_ambient_temp_c is None:
            raise ValueError(
                f"Pulse #{p.pulse_index} has no calculated ambient temperature."
            )

        delta = abs(p.calculated_ambient_temp_c - p.ambient_temp_setpoint_c)

        if delta <= max_delta_c:
            retained.append(p)
        else:
            removed.append(p)
            print(
                f"[{p.loop_type.upper()}] Pulse #{p.pulse_index} REMOVED (Ambient Delta) — "
                f"Calc Ambient: {p.calculated_ambient_temp_c:.2f}°C | "
                f"Setpoint: {p.ambient_temp_setpoint_c:.2f}°C | "
                f"|Delta|: {delta:.2f}°C (> {max_delta_c:.2f}°C)"
            )

    return retained, removed


def filter_pulses_by_duration(
    pulses: list[Pulse], min_duration_ratio: float = 0.90
) -> tuple[list[Pulse], list[Pulse]]:
    """Filters out pulses with actual duration < min_duration_ratio * (time_on_s + time_off_s)."""
    retained_pulses = []
    removed_pulses = []

    for p in pulses:
        t_col = "Timestamp" if p.loop_type == "open-loop" else "datetime"
        time_vals = p.data[t_col]

        if pd.api.types.is_numeric_dtype(time_vals):
            actual_duration = time_vals.iloc[-1] - time_vals.iloc[0]
        else:
            actual_duration = (
                pd.to_datetime(time_vals.iloc[-1])
                - pd.to_datetime(time_vals.iloc[0])
            ).total_seconds()

        expected_duration = p.time_on_s + p.time_off_s
        min_threshold = min_duration_ratio * expected_duration

        if actual_duration >= min_threshold:
            retained_pulses.append(p)
        else:
            removed_pulses.append(p)
            print(
                f"[{p.loop_type.upper()}] Pulse #{p.pulse_index} REMOVED (Short Duration) — "
                f"Actual: {actual_duration:.2f}s | Expected: {expected_duration:.2f}s "
                f"(Cutoff: {min_threshold:.2f}s)"
            )

    return retained_pulses, removed_pulses


# ==========================================
# 4. Visualization Functions
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot retained pulses
    for p in pulses:
        df = p.data.copy()
        time_vals = (
            df["Timestamp"] if loop_type == "open-loop" else df["datetime"]
        )

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
            time_vals = (
                df["Timestamp"] if loop_type == "open-loop" else df["datetime"]
            )

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
    plt.show()


def plot_closed_loop_timeseries(
    retained_pulses: list[Pulse],
    removed_pulses: list[Pulse],
    save_path: str = None,
):
    """Plots a continuous time series of closed-loop pulses, highlighting retained vs. removed pulses."""
    if not retained_pulses and not removed_pulses:
        print("No closed-loop pulses available to plot.")
        return

    removed_indices = {p.pulse_index for p in removed_pulses}
    all_pulses = sorted(
        retained_pulses + removed_pulses, key=lambda p: p.pulse_index
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
    )

    legend_tracker = {"retained": False, "removed": False}

    for p in all_pulses:
        df = p.data.copy()
        time_vals = pd.to_datetime(df["datetime"])
        temp_vals = df["temperature"]
        cp_vals = df["AgCl CP"]

        is_removed = p.pulse_index in removed_indices
        color = "crimson" if is_removed else "#1f77b4"
        alpha = 0.85 if is_removed else 0.7
        linewidth = 1.4 if is_removed else 1.0

        # Temperature Subplot
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

        # Highlight removed pulse regions with light crimson background shading
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

        # Potential (AgCl CP) Subplot
        ax2.plot(
            time_vals,
            cp_vals,
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
    ax2.set_xlabel("Datetime", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.xticks(rotation=25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved continuous time-series plot: {save_path}")

    plt.show()


# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == "__main__":
    open_loop_file = "open_loop_elec_35_25_15C.txt"
    closed_loop_file = "closed_loop_elec_35_25_15C.txt"

    # MANUAL WINDOW SELECTION
    open_loop_window_map = {
        15.0: (90.0, 100.0),
        30.0: (140.0, 150.0),
        60.0: (190.0, 200.0),
    }
    closed_loop_window_s = (220.0, 240.0)

    ambient_delta_threshold_c = 0.25
    min_duration_ratio = 0.90

    # Step 1: Parse Files
    print("Parsing Open-Loop Data...")
    open_loop_pulses = parse_pulses(open_loop_file, loop_type="open-loop")

    print("Parsing Closed-Loop Data...")
    closed_loop_pulses = parse_pulses(
        closed_loop_file, loop_type="closed-loop"
    )

    plot_pulses_overlay(
        open_loop_pulses, save_path="open_loop_overlay_raw.png"
    )
    plot_pulses_overlay(
        closed_loop_pulses, save_path="closed_loop_overlay_raw.png"
    )

    # Step 2: Calculate Ambient Temperatures using configured windows
    open_loop_pulses = calculate_ambient_temperatures(
        open_loop_pulses, time_window_s=open_loop_window_map
    )
    closed_loop_pulses = calculate_ambient_temperatures(
        closed_loop_pulses, time_window_s=closed_loop_window_s
    )

    # Step 3: Dynamic Duration Filter (< 90% expected length)
    print("\n--- Filtering Open-Loop (Short Duration) ---")
    open_loop_dur_clean, open_loop_dur_removed = filter_pulses_by_duration(
        open_loop_pulses, min_duration_ratio=min_duration_ratio
    )

    print("\n--- Filtering Closed-Loop (Short Duration) ---")
    closed_loop_dur_clean, closed_loop_dur_removed = filter_pulses_by_duration(
        closed_loop_pulses, min_duration_ratio=min_duration_ratio
    )

    # Step 4: Ambient Setpoint Delta Filter
    print(
        f"\n--- Filtering Open-Loop (Threshold: |T_calc - T_sp| > {ambient_delta_threshold_c}°C) ---"
    )
    clean_open_pulses, removed_open_ambient = (
        filter_pulses_by_ambient_setpoint_delta(
            open_loop_dur_clean, max_delta_c=ambient_delta_threshold_c
        )
    )

    print(
        f"\n--- Filtering Closed-Loop (Threshold: |T_calc - T_sp| > {ambient_delta_threshold_c}°C) ---"
    )
    clean_closed_pulses, removed_closed_ambient = (
        filter_pulses_by_ambient_setpoint_delta(
            closed_loop_dur_clean, max_delta_c=ambient_delta_threshold_c
        )
    )

    all_removed_open = open_loop_dur_removed + removed_open_ambient
    all_removed_closed = closed_loop_dur_removed + removed_closed_ambient

    # Step 5: Plot & Save Relative Overlays
    plot_pulses_overlay(
        clean_open_pulses,
        removed_pulses=all_removed_open,
        time_window_s=open_loop_window_map,
        save_path="open_loop_overlay_removed.png",
    )

    plot_pulses_overlay(
        clean_closed_pulses,
        removed_pulses=all_removed_closed,
        time_window_s=closed_loop_window_s,
        save_path="closed_loop_overlay_removed.png",
    )

    # Step 6: Plot Continuous Closed-Loop Time Series
    plot_closed_loop_timeseries(
        retained_pulses=clean_closed_pulses,
        removed_pulses=all_removed_closed,
        save_path="closed_loop_timeseries_highlighted.png",
    )

    # Step 7: Save Filtered Pulses to JSON
    save_pulses_to_json(clean_open_pulses, "open_loop_pulses_filtered.json")
    save_pulses_to_json(
        clean_closed_pulses, "closed_loop_pulses_filtered.json"
    )

    plot_pulses_overlay(
        clean_open_pulses,
        save_path="open_loop_overlay_filtered.png",
    )
    plot_pulses_overlay(
        clean_closed_pulses,
        save_path="closed_loop_overlay_filtered.png",
    )