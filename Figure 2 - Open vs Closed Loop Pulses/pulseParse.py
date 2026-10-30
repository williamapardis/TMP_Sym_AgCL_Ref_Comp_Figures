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

    def to_dict(self) -> dict:
        """Converts metadata and DataFrame time series to a JSON-serializable dictionary."""
        return {
            "pulse_index": self.pulse_index,
            "loop_type": self.loop_type,
            "metadata": {
                "time_on_s": self.time_on_s,
                "time_off_s": self.time_off_s,
                "ambient_temp_setpoint_c": self.ambient_temp_setpoint_c,
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
            data=pd.DataFrame(item["time_series"]),
        )
        pulses.append(pulse)
    return pulses


# ==========================================
# 2. File Parser (Swapped: ON -> OFF)
# ==========================================
def parse_pulses(file_path: str, loop_type: str) -> list[Pulse]:
    """Parses open-loop or closed-loop text files into structured Pulse objects.

    Pulse definition: One full ON cycle (1) followed by one full OFF cycle (0).
    """
    # Tab or comma separation preserves multi-word headers like 'PID enable'
    df = pd.read_csv(file_path, sep=r"\t|,", engine="python")

    # Strip extra whitespace from column names
    df.columns = df.columns.str.strip()

    # Determine state tracking column based on loop type
    state_col = "heater_state" if loop_type == "open-loop" else "PID enable"

    # Defensive check with diagnostics if missing
    if state_col not in df.columns:
        possible_matches = [
            c for c in df.columns if "PID" in c or "enable" in c or "heater" in c
        ]
        raise KeyError(
            f"Expected state column '{state_col}' not found in '{file_path}'.\n"
            f"Detected columns: {list(df.columns)}\n"
            f"Possible candidates: {possible_matches}"
        )

    # Identify state transitions (OFF -> ON or ON -> OFF)
    df["state_change"] = df[state_col].diff().fillna(0) != 0
    df["cycle_id"] = df["state_change"].cumsum()

    pulses = []
    pulse_counter = 0

    # Group by consecutive state blocks
    cycles = [group for _, group in df.groupby("cycle_id")]

    i = 0
    while i < len(cycles) - 1:
        on_block = cycles[i]
        off_block = cycles[i + 1]

        # SWAPPED: Ensure ON (1) followed by OFF (0)
        if (
            on_block[state_col].iloc[0] == 1
            and off_block[state_col].iloc[0] == 0
        ):
            pulse_df = pd.concat([on_block, off_block]).drop(
                columns=["state_change", "cycle_id"]
            )

            # Metadata and time-series extraction mapping table
            if loop_type == "open-loop":
                ts_data = pulse_df[["Timestamp", "Temperature_C", "E_mV"]]
                time_on_s = pulse_df["on_period_ms"].iloc[0] / 1000.0
                time_off_s = pulse_df["off_period_ms"].iloc[0] / 1000.0
                temp_sp = pulse_df["Temperature_Setpoint_C"].iloc[0]
                max_curr_a = pulse_df["Current_mA"].iloc[0] / 1000.0
            else:
                ts_data = pulse_df[["datetime", "temperature", "AgCl CP"]]
                time_on_s = pulse_df["time on"].iloc[0]
                time_off_s = pulse_df["time off"].iloc[0]
                temp_sp = pulse_df["temperature setpoint"].iloc[0]
                max_curr_a = 925 / 1000.0  # Constant 925 mA = 0.925 A

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
            i += 2  # Move to the next ON/OFF pair
        else:
            i += 1  # Shift by 1 if not starting on an ON cycle

    return pulses


# ==========================================
# 3. Visualization Functions
# ==========================================
def plot_single_pulse(pulse: Pulse, save_path: str = None):
    """Plots temperature and potential (E_mV / AgCl CP) for a single pulse."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    time_col = (
        "Timestamp" if pulse.loop_type == "open-loop" else "datetime"
    )
    temp_col = (
        "Temperature_C" if pulse.loop_type == "open-loop" else "temperature"
    )
    cp_col = "E_mV" if pulse.loop_type == "open-loop" else "AgCl CP"

    # Temperature Subplot
    ax1.plot(
        pulse.data[time_col],
        pulse.data[temp_col],
        color="crimson",
        linewidth=1.5,
        label="Temperature",
    )
    ax1.axhline(
        pulse.ambient_temp_setpoint_c,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Setpoint ({pulse.ambient_temp_setpoint_c}°C)",
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"{pulse.loop_type.title()} — Pulse #{pulse.pulse_index} (ON -> OFF)\n"
        f"Time On: {pulse.time_on_s}s | Time Off: {pulse.time_off_s}s | I_max: {pulse.max_current_a}A"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Potential / CP Subplot
    ax2.plot(
        pulse.data[time_col],
        pulse.data[cp_col],
        color="darkblue",
        linewidth=1.5,
        label=cp_col,
    )
    ax2.set_xlabel(f"Time ({time_col})")
    ax2.set_ylabel(
        "Potential (mV)" if pulse.loop_type == "open-loop" else "AgCl CP"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_pulses_overlay(pulses: list[Pulse], save_path: str = None):
    """Overlays all pulses starting from relative time t = 0."""
    if not pulses:
        return

    loop_type = pulses[0].loop_type
    temp_col = "Temperature_C" if loop_type == "open-loop" else "temperature"
    cp_col = "E_mV" if loop_type == "open-loop" else "AgCl CP"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

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

    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(
        f"Pulse-to-Pulse Overlay ({loop_type.title()} Mode, ON -> OFF)"
    )
    ax1.grid(True, alpha=0.3)
    #ax1.legend(loc="upper right", fontsize="small", ncol=2)

    ax2.set_xlabel("Relative Time from Pulse Start (s)")
    ax2.set_ylabel("Potential / CP")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    open_loop_file = "open_loop_elec_35_25_15C.txt"
    closed_loop_file = "closed_loop_elec_35_25_15C.txt"

    # Step 1: Parse Files
    print("Parsing Open-Loop Data...")
    open_loop_pulses = parse_pulses(open_loop_file, loop_type="open-loop")

    print("Parsing Closed-Loop Data...")
    closed_loop_pulses = parse_pulses(
        closed_loop_file, loop_type="closed-loop"
    )

    # Step 2: Save to JSON
    save_pulses_to_json(open_loop_pulses, "open_loop_pulses.json")
    save_pulses_to_json(closed_loop_pulses, "closed_loop_pulses.json")

    # Step 3: Load back from JSON (Validation)
    open_loop_loaded = load_pulses_from_json("open_loop_pulses.json")
    closed_loop_loaded = load_pulses_from_json("closed_loop_pulses.json")

    # Step 4: Plot
    if open_loop_loaded:
        #plot_single_pulse(
        #    open_loop_loaded[0], save_path="open_loop_pulse_0.png"
        #)
        plot_pulses_overlay(
            open_loop_loaded, save_path="open_loop_overlay.png"
        )

    if closed_loop_loaded:
        #plot_single_pulse(
        #    closed_loop_loaded[0], save_path="closed_loop_pulse_0.png"
        #)
        plot_pulses_overlay(
            closed_loop_loaded, save_path="closed_loop_overlay.png"
        )