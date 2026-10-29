import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt

# 1. SPECIFY YOUR FILE NAME HERE
input_file_path = "open_loop_elec_35_25_15C.txt"

columns = [
    'Timestamp', 'Temperature_Setpoint_C', 'Current_mA', 'Temperature_C', 
    'heater_state', 'on_period_ms', 'off_period_ms', 'E_mV', 'averages'
]

if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"Could not find the file: '{input_file_path}'. Please check the path.")

log_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(.*)$')

parsed_rows = []
with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line or line.startswith("Timestamp"):  
            continue
            
        match = log_pattern.match(line)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            csv_part = match.group(3)
            
            timestamp_str = f"{date_str} {time_str}"
            remaining_data = [item.strip() for item in csv_part.split(',') if item.strip()]
            
            row = [timestamp_str] + remaining_data
            if len(row) == len(columns):
                parsed_rows.append(row)

if not parsed_rows:
    raise ValueError(f"No valid data rows were parsed from '{input_file_path}'.")

# Create DataFrame & enforce data types
df = pd.DataFrame(parsed_rows, columns=columns)
numeric_cols = columns[1:]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['time_s'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()

# Track pulses sequentially
pulse_starts = (df['heater_state'] == 1) & (df['heater_state'].shift(1) == 0)
df['pulse_id'] = pulse_starts.cumsum() + 1
df['pulse_id'] = df['pulse_id'] - df['pulse_id'].min() + 1

# --- NEW: FILTER PULSES BASED ON TEMPERATURE SETPOINT DEVIATION ---
def keep_pulse(group):
    # Get the target setpoint (using mode to handle minor fluctuation noise)
    setpoint = group['Temperature_Setpoint_C'].mode().iloc[0]
    min_temp = group['Temperature_C'].min()
    # Absolute difference ensures we catch drops or spikes wider than 2°C
    return abs(min_temp - setpoint) <= 1

# Count pulses before filter for reporting
total_raw_pulses = df['pulse_id'].nunique()

# Apply the filter group-by-group
df = df.groupby('pulse_id').filter(keep_pulse)

if df.empty:
    raise ValueError("All pulses were filtered out! Check if your temperature threshold is too strict.")

print(f"Filtered out {total_raw_pulses - df['pulse_id'].nunique()} unstable pulses. Keeping {df['pulse_id'].nunique()} pulses.")
# -----------------------------------------------------------------

# Faster native JSON export
pulses_list = []
for p_id, group in df.groupby('pulse_id'):
    records = group.drop(columns=['pulse_id']).copy()
    records['Timestamp'] = records['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    pulses_list.append({
        "pulse_id": int(p_id),
        "data_points": records.to_dict(orient='records')
    })
    
file_json_payload = {
    "total_pulses_found": len(pulses_list),
    "pulses": pulses_list
}

output_json_filename = "structured_pulses.json"
with open(output_json_filename, "w") as f:
    json.dump({os.path.basename(input_file_path): file_json_payload}, f, indent=4)
print(f"Success! Exported clean dataset to {output_json_filename}")

# Setup plotting aesthetics
unique_currents = sorted(df['Current_mA'].unique())
if len(unique_currents) <= 10:
    cmap = plt.get_cmap('tab10')
    current_color_map = {curr: cmap(i) for i, curr in enumerate(unique_currents)}
else:
    cmap = plt.get_cmap('viridis')
    current_color_map = {curr: cmap(i / (len(unique_currents) - 1)) for i, curr in enumerate(unique_currents)}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
tracked_current_labels = set()

for p_id, group in df.groupby('pulse_id'):
    pulse_current = round(group['Current_mA'].mode().iloc[0], 2)
    color = current_color_map[pulse_current]
    
    relative_time = group['time_s'] - group['time_s'].iloc[0]
    line_label = f"{pulse_current} mA" if pulse_current not in tracked_current_labels else ""
    tracked_current_labels.add(pulse_current)
        
    ax1.plot(relative_time, group['E_mV'], color=color, alpha=0.7, label=line_label)
    ax2.plot(relative_time, group['Temperature_C'], color=color, alpha=0.7)

ax1.set_title(f"Segmented Pulses (Cell Potential): {os.path.basename(input_file_path)}", loc='left', fontsize=11, fontweight='semibold')
ax1.set_ylabel("Cell Potential [mV]")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(title="Current Setting", loc='upper right', bbox_to_anchor=(1.12, 1))

ax2.set_title(f"Segmented Pulses (Temperature Profiles)", loc='left', fontsize=11, fontweight='semibold')
ax2.set_ylabel("Temperature [°C]")
ax2.set_xlabel("Relative Pulse Duration [s]")
ax2.grid(True, linestyle=':', alpha=0.6)

fig.suptitle("Pulse Visualizer Summary (Filtered)", fontsize=13, fontweight='bold', y=0.98)
fig.tight_layout()
plt.show()
