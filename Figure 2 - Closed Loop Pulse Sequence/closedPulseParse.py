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
    loop_type: str  # 'closed-loop'
    time_on_s: float
    time_off_s: float
    ambient_temp_setpoint_c: float
    max_current_a: float
    step: int  # Step index from CSV data
    data: pd.DataFrame
    calculated_ambient_temp_c: float = None  # Saved to JSON metadata
    delta_temp_target_c: float = None  # Added target temperature delta metadata

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
            step=item["metadata"].get("step", 0),
            calculated_ambient_temp_c=item["metadata"].get(
                "calculated_ambient_temp_c"
            ),
            delta_temp_target_c=item["metadata"].get("delta_temp_target_c"),
            data=pd.DataFrame(item["time_series"]),
        )
        pulses.append(pulse)
    return pulses


# ==========================================
# 2. File Parser (Pulse Definition: ON -> OFF)
# ==========================================
def parse_closed_loop_pulses(file_path: str) -> list[Pulse]:
    """Parses closed-loop text files into structured Pulse objects.

    Pulse definition: One full PID enable ON cycle (1) followed by one full OFF cycle (0).
    """
    df = pd.read_csv(file_path, sep=r"\t|,", engine="python")
    df.columns = df.columns.str.strip()

    state_col = "PID enable"

    if state_col not in df.columns:
        possible_matches = [
            c for c in df.columns if "PID" in c or "enable" in c
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

            ts_data = pulse_df[
                ["datetime", "temperature", "AgCl CP", "current", "step"]
            ]
            time_on_s = float(pulse_df["time on"].iloc[0])
            time_off_s = float(pulse_df["time off"].iloc[0])
            temp_sp = float(pulse_df["temperature setpoint"].iloc[0])
            max_curr_a = 925.0 / 1000.0
            step_val = int(pulse_df["step"].iloc[0])

            pulse = Pulse(
                pulse_index=pulse_counter,
                loop_type="closed-loop",
                time_on_s=time_on_s,
                time_off_s=time_off_s,
                ambient_temp_setpoint_c=temp_sp,
                max_current_a=max_curr_a,
                step=step_val,
                data=ts_data.reset_index(drop=True),
            )
            pulses.append(pulse)
            pulse_counter += 1
            i += 2
        else:
            i += 1

    return pulses


# ==========================================
# 3. Ambient & Target Delta Calculation Stage
# ==========================================
def calculate_ambient_temperatures(
    pulses: list[Pulse], time_window_s: tuple[float, float]
) -> list[Pulse]:
    """Calculates ambient temperature as mean temp within a relative window [t_start, t_end]."""
    t_start, t_end = time_window_s

    for p in pulses:
        df = p.data
        time_vals = df["datetime"]

        rel_time = (
            pd.to_datetime(time_vals) - pd.to_datetime(time_vals.iloc[0])
        ).dt.total_seconds()

        window_mask = (rel_time >= t_start) & (rel_time <= t_end)
        window_temps = df.loc[window_mask, "temperature"]

        if not window_temps.empty:
            p.calculated_ambient_temp_c = float(window_temps.mean())
        else:
            p.calculated_ambient_temp_c = float(df["temperature"].iloc[-1])
            print(
                f"[CLOSED-LOOP] Warning: Pulse #{p.pulse_index} window [{t_start:.1f}, {t_end:.1f}]s outside bounds. Used final sample."
            )

    return pulses


def assign_delta_temp_targets(
    pulses: list[Pulse], target_amplitudes: list[float]
) -> list[Pulse]:
    """Calculates observed max - min temperature delta for each pulse and assigns the closest target amplitude."""
    assigned_count = 0
    unassigned_count = 0

    for p in pulses:
        if "temperature" in p.data.columns and not p.data["temperature"].empty:
            temp_series = p.data["temperature"]
            min_temp = temp_series.min()
            max_temp = temp_series.max()
            delta_temp_observed = max_temp - min_temp

            target_amp = float(
                min(target_amplitudes, key=lambda t: abs(t - delta_temp_observed))
            )
            p.delta_temp_target_c = target_amp
            assigned_count += 1
        else:
            p.delta_temp_target_c = None
            unassigned_count += 1

    print(
        f"[CLOSED-LOOP] Target Delta Assigned: {assigned_count} | Unassigned: {unassigned_count}"
    )
    return pulses


# ==========================================
# 4. Filtering Functions
# ==========================================
def filter_pulses_by_target_steps(
    pulses: list[Pulse], target_steps: list[int]
) -> tuple[list[Pulse], list[Pulse]]:
    """Filters out pulses whose 'step' attribute is NOT in the target_steps list."""
    if target_steps is None:
        return pulses, []

    retained = []
    removed = []
    target_set = set(target_steps)

    for p in pulses:
        if p.step in target_set:
            retained.append(p)
        else:
            removed.append(p)
            print(
                f"[CLOSED-LOOP] Pulse #{p.pulse_index} REMOVED (Step Excluded) — "
                f"Step: {p.step} (Not in allowed steps: {target_steps})"
            )

    return retained, removed


def filter_pulses_by_step_offset(
    pulses: list[Pulse], step_start_window_s: float
) -> tuple[list[Pulse], list[Pulse]]:
    """Filters out pulses starting within step_start_window_s seconds of the start of a step."""
    if not pulses or step_start_window_s <= 0:
        return pulses, []

    retained = []
    removed = []

    step_start_times = {}
    for p in pulses:
        p_start = pd.to_datetime(p.data["datetime"].iloc[0])
        if p.step not in step_start_times:
            step_start_times[p.step] = p_start
        else:
            step_start_times[p.step] = min(step_start_times[p.step], p_start)

    for p in pulses:
        p_start = pd.to_datetime(p.data["datetime"].iloc[0])
        s_start = step_start_times[p.step]
        offset_s = (p_start - s_start).total_seconds()

        if offset_s >= step_start_window_s:
            retained.append(p)
        else:
            removed.append(p)
            print(
                f"[CLOSED-LOOP] Pulse #{p.pulse_index} REMOVED (Step Window) — "
                f"Step: {p.step} | Offset from step start: {offset_s:.1f}s (< {step_start_window_s:.1f}s)"
            )

    return retained, removed


def filter_pulses_by_last_in_step(
    pulses: list[Pulse],
) -> tuple[list[Pulse], list[Pulse]]:
    """Filters out the final pulse of every step group."""
    if not pulses:
        return [], []

    pulses_by_step = {}
    for p in pulses:
        pulses_by_step.setdefault(p.step, []).append(p)

    retained = []
    removed = []

    for step_val, step_pulses in pulses_by_step.items():
        retained.extend(step_pulses[:-1])
        last_pulse = step_pulses[-1]
        removed.append(last_pulse)
        print(
            f"[CLOSED-LOOP] Pulse #{last_pulse.pulse_index} REMOVED (Last Pulse in Step) — "
            f"Step: {step_val}"
        )

    return retained, removed


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
                f"[CLOSED-LOOP] Pulse #{p.pulse_index} REMOVED (Ambient Delta) — "
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
        time_vals = p.data["datetime"]
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
                f"[CLOSED-LOOP] Pulse #{p.pulse_index} REMOVED (Short Duration) — "
                f"Actual: {actual_duration:.2f}s | Expected: {expected_duration:.2f}s "
                f"(Cutoff: {min_threshold:.2f}s)"
            )

    return retained_pulses, removed_pulses


# ==========================================
# 5. Visualization Functions
# ==========================================
def plot_pulses_overlay(
    pulses: list[Pulse],
    removed_pulses: list[Pulse] = None,
    time_window_s: tuple[float, float] = None,
    save_path: str = None,
):
    """Overlays closed-loop pulses starting from relative time t = 0 and highlights ambient window."""
    if not pulses and not removed_pulses:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for p in pulses:
        df = p.data.copy()
        rel_time = (
            pd.to_datetime(df["datetime"]) - pd.to_datetime(df["datetime"].iloc[0])
        ).dt.total_seconds()

        ax1.plot(
            rel_time,
            df["temperature"],
            alpha=0.7,
            linewidth=1.2,
            label=f"Pulse #{p.pulse_index}",
        )
        ax2.plot(rel_time, df["AgCl CP"], alpha=0.7, linewidth=1.2)

    if removed_pulses:
        for p in removed_pulses:
            df = p.data.copy()
            rel_time = (
                pd.to_datetime(df["datetime"])
                - pd.to_datetime(df["datetime"].iloc[0])
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
        f"Closed-Loop Pulse Overlay — Retained: {len(pulses)}, Removed: {len(removed_pulses or [])}"
    )
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Relative Time from Pulse Start (s)")
    ax2.set_ylabel("AgCl Potential (mV)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved overlay plot: {save_path}")
    plt.show()


def plot_current_pulses_overlay(
    pulses: list[Pulse],
    removed_pulses: list[Pulse] = None,
    time_window_s: tuple[float, float] = None,
    save_path: str = None,
):
    """Overlays closed-loop current time series starting from relative time t = 0."""
    if not pulses and not removed_pulses:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for p in pulses:
        if "current" not in p.data.columns:
            continue
        df = p.data.copy()
        rel_time = (
            pd.to_datetime(df["datetime"]) - pd.to_datetime(df["datetime"].iloc[0])
        ).dt.total_seconds()

        ax.plot(
            rel_time,
            df["current"],
            alpha=0.7,
            linewidth=1.2,
            label=f"Pulse #{p.pulse_index}",
        )

    if removed_pulses:
        for p in removed_pulses:
            if "current" not in p.data.columns:
                continue
            df = p.data.copy()
            rel_time = (
                pd.to_datetime(df["datetime"])
                - pd.to_datetime(df["datetime"].iloc[0])
            ).dt.total_seconds()

            ax.plot(
                rel_time,
                df["current"],
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
    ax.set_ylabel("Current")
    ax.set_title(
        f"Current Overlay (Closed-Loop Mode) — Retained: {len(pulses)}, Removed: {len(removed_pulses or [])}"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved current overlay plot: {save_path}")
    plt.show()


def plot_closed_loop_timeseries(
    retained_pulses: list[Pulse],
    removed_pulses: list[Pulse],
    selection_params: dict = None,
    save_path: str = None,
):
    """Plots a 2-panel continuous time series (Temperature, AgCl CP) with selection parameters."""
    if not retained_pulses and not removed_pulses:
        print("No closed-loop pulses available to plot.")
        return

    removed_indices = {p.pulse_index for p in removed_pulses}
    all_pulses = sorted(
        retained_pulses + removed_pulses, key=lambda p: p.pulse_index
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
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

        # Subplot 2: Potential (AgCl CP)
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
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    ax2.set_ylabel("AgCl Potential (mV)", fontsize=11)
    ax2.set_xlabel("Datetime", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Figure Title & Selection Parameters Annotation
    fig.suptitle(
        f"Closed-Loop Time Series — Total Pulses: {len(all_pulses)} "
        f"(Retained: {len(retained_pulses)}, Removed: {len(removed_pulses)})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    if selection_params:
        param_str = "  |  ".join(f"{k}: {v}" for k, v in selection_params.items())
        fig.text(
            0.5,
            0.935,
            f"Selection Parameters — {param_str}",
            ha="center",
            va="top",
            fontsize=9.5,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#f4f6f8",
                edgecolor="#cccccc",
            ),
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.xticks(rotation=25)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved continuous time-series plot: {save_path}")

    plt.show()


# ==========================================
# 6. Main Execution Block
# ==========================================
if __name__ == "__main__":
    closed_loop_file = "closed_loop_titration.txt"

    closed_loop_window_s = (220.0, 240.0)
    ambient_delta_threshold_c = 0.25
    min_duration_ratio = 0.90
    step_start_window_s = 600.0  # Window at start of step (in seconds)

    # Target steps to retain
    target_steps = [0, 4, 9]

    # Target pulse temperature step amplitudes (°C)
    target_amplitudes = [1.0, 5.0, 10.0]

    # Step 1: Parse Closed-Loop Data
    print("Parsing Closed-Loop Data...")
    closed_loop_pulses = parse_closed_loop_pulses(closed_loop_file)

    plot_pulses_overlay(closed_loop_pulses, save_path="raw_pulses.png")

    # Step 2: Filter by Target Steps
    print(f"\n--- Filtering Closed-Loop (Target Steps Only: {target_steps}) ---")
    closed_loop_step_sel_clean, closed_loop_step_sel_removed = (
        filter_pulses_by_target_steps(
            closed_loop_pulses, target_steps=target_steps
        )
    )

    # Step 3: Filter Out Last Pulse of Every Step
    print("\n--- Filtering Closed-Loop (Last Pulse in Each Step) ---")
    closed_loop_last_pulse_clean, closed_loop_last_pulse_removed = (
        filter_pulses_by_last_in_step(closed_loop_step_sel_clean)
    )

    # Step 4: Calculate Ambient Temperatures & Assign Delta Temp Targets
    closed_loop_last_pulse_clean = calculate_ambient_temperatures(
        closed_loop_last_pulse_clean, time_window_s=closed_loop_window_s
    )
    closed_loop_last_pulse_clean = assign_delta_temp_targets(
        closed_loop_last_pulse_clean, target_amplitudes=target_amplitudes
    )

    # Step 5: Duration Filter (< 90% expected length)
    print("\n--- Filtering Closed-Loop (Short Duration) ---")
    closed_loop_dur_clean, closed_loop_dur_removed = filter_pulses_by_duration(
        closed_loop_last_pulse_clean, min_duration_ratio=min_duration_ratio
    )

    # Step 6: Step-Start Window Filter (Beginning of step)
    print(
        f"\n--- Filtering Closed-Loop (Step Start Window: < {step_start_window_s}s) ---"
    )
    closed_loop_offset_clean, closed_loop_offset_removed = filter_pulses_by_step_offset(
        closed_loop_dur_clean, step_start_window_s=step_start_window_s
    )

    # Step 7: Ambient Setpoint Delta Filter
    print(
        f"\n--- Filtering Closed-Loop (Threshold: |T_calc - T_sp| > {ambient_delta_threshold_c}°C) ---"
    )
    clean_closed_pulses, removed_closed_ambient = (
        filter_pulses_by_ambient_setpoint_delta(
            closed_loop_offset_clean, max_delta_c=ambient_delta_threshold_c
        )
    )

    # Combine all removed pulses across stages
    all_removed_closed = (
        closed_loop_step_sel_removed
        + closed_loop_last_pulse_removed
        + closed_loop_dur_removed
        + closed_loop_offset_removed
        + removed_closed_ambient
    )

    # Step 8: Plot Relative Overlays (With Ambient Window)
    plot_pulses_overlay(
        clean_closed_pulses,
        removed_pulses=all_removed_closed,
        time_window_s=closed_loop_window_s,
    )

    plot_current_pulses_overlay(
        clean_closed_pulses,
        removed_pulses=all_removed_closed,
        time_window_s=closed_loop_window_s,
    )

    # Step 9: Plot Continuous Closed-Loop Time Series (Temperature, AgCl CP)
    selection_summary = {
        "Steps": target_steps,
        "Last in Step": "Excluded",
        "Ambient Window": f"{closed_loop_window_s[0]:.0f}–{closed_loop_window_s[1]:.0f}s",
        "Max |ΔT_amb|": f"{ambient_delta_threshold_c}°C",
        "Min Duration": f"{min_duration_ratio:.0%}",
        "Step Start Offset": f"≥{step_start_window_s:.0f}s",
    }

    plot_closed_loop_timeseries(
        retained_pulses=clean_closed_pulses,
        removed_pulses=all_removed_closed,
        selection_params=selection_summary,
        save_path="parsed_pulse_timeseries.png",
    )

    # Step 10: Save Filtered Data (with assigned delta_temp_target_c metadata) to JSON
    save_pulses_to_json(
        clean_closed_pulses, "pulses.json"
    )

    plot_pulses_overlay(clean_closed_pulses, save_path="selected_pulses.png")
    plot_current_pulses_overlay(clean_closed_pulses)