import json
from dataclasses import dataclass
import pandas as pd

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
    delta_temp_target_c: float = None  # Added to metadata

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


def process_pulses_metadata(file_path: str):
    target_amplitudes = [1.0, 5.0, 10.0]

    with open(file_path, 'r') as f:
        pulses = json.load(f)

    assigned_count = 0
    unassigned_count = 0

    for pulse in pulses:
        temp_series = pulse.get("time_series", {}).get("temperature", [])
        
        # Ensure 'metadata' dictionary exists
        if "metadata" not in pulse or pulse["metadata"] is None:
            pulse["metadata"] = {}

        if temp_series:
            min_temp = min(temp_series)
            max_temp = max(temp_series)
            delta_temp_observed = max_temp - min_temp

            # Assign closest target amplitude
            target_amp = float(min(target_amplitudes, key=lambda t: abs(t - delta_temp_observed)))
            pulse["metadata"]["delta_temp_target_c"] = target_amp
            assigned_count += 1
        else:
            pulse["metadata"]["delta_temp_target_c"] = None
            unassigned_count += 1

    # Save updated JSON file back to disk
    with open(file_path, 'w') as f:
        json.dump(pulses, f, indent=4)

    # Output requested summary count
    print(f"Pulses assigned: {assigned_count}")
    print(f"Pulses that could not be assigned: {unassigned_count}")


if __name__ == "__main__":
    process_pulses_metadata("closed_loop_pulses_filtered.json")