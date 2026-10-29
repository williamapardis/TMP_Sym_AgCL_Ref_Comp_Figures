import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# File paths
closed_loop_file = "closed_loop_elec_35_25_15C.txt"
open_loop_file = "open_loop_elec_35_25_15C.txt"
output_image_path = "temperature_time_series.png"

# ==============================================================================
# AXIS LIMITS CONFIGURATION
# Set values as numbers/strings or None for auto-scaling.
# Example date string format for x-limits: "2026-09-13 10:00"
# ==============================================================================
AXIS_LIMITS = {
    "closed_loop": {
        "xlim": (None, None),  # e.g., ("2026-09-13 10:00", "2026-09-14 10:00")
        "ylim": (None, None),  # e.g., (10, 40)
    },
    "open_loop": {
        "xlim": (None, None),
        "ylim": (None, None),
    },
}


def find_setpoint_column(df, file_label):
    """Finds the temperature setpoint column while avoiding false positives like 'volume dispensed'."""
    cols = df.columns.tolist()

    # Priority 1: Exact matches or explicit "temperature setpoint"
    for col in cols:
        col_lower = col.lower()
        if "temp" in col_lower and "setpoint" in col_lower:
            return col

    # Priority 2: Standard setpoint terms
    for col in cols:
        col_lower = col.lower()
        if "setpoint" in col_lower or col_lower == "sp" or "target" in col_lower:
            return col

    raise KeyError(
        f"Could not find a valid setpoint column in {file_label}. Available columns: {cols}"
    )


# 1. Load Closed Loop Data
df_closed = pd.read_csv(closed_loop_file, skipinitialspace=True)
df_closed.columns = df_closed.columns.str.strip()
df_closed["datetime"] = pd.to_datetime(df_closed["datetime"], errors="coerce")
df_closed = df_closed.dropna(subset=["datetime"]).sort_values("datetime")

# 2. Load Open Loop Data
df_open = pd.read_csv(
    open_loop_file,
    sep=r"[\t,]+",
    engine="python",
    skipinitialspace=True,
    skip_blank_lines=True,
)
df_open.columns = df_open.columns.str.strip()
df_open["Timestamp"] = pd.to_datetime(df_open["Timestamp"], errors="coerce")
df_open = df_open.dropna(subset=["Timestamp"]).sort_values("Timestamp")

# Detect setpoint columns dynamically
closed_sp_col = find_setpoint_column(df_closed, "Closed Loop File")
open_sp_col = find_setpoint_column(df_open, "Open Loop File")

print(f"Closed Loop Setpoint Column: '{closed_sp_col}'")
print(f"Open Loop Setpoint Column:   '{open_sp_col}'")

# 3. Create Widescreen Figure
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(19.2, 10.8))

date_fmt = mdates.DateFormatter("%m/%d %H:%M")


# Helper function to add vertical 12-hour marker lines starting from t_0
def add_12hr_lines(ax, df, time_col):
    if df.empty:
        return
    start_time = df[time_col].min()
    end_time = df[time_col].max()

    marker_times = pd.date_range(start=start_time, end=end_time, freq="12h")

    for t in marker_times:
        ax.axvline(
            x=t,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="12-hr Interval" if t == marker_times[0] else "",
        )


# --- Top Subplot: Closed Loop ---
ax1.plot(
    df_closed["datetime"],
    df_closed["temperature"],
    color="tab:blue",
    linewidth=1.5,
    label="Temperature",
)
ax1.plot(
    df_closed["datetime"],
    df_closed["temperature setpoint"],
    color="tab:red",
    linewidth=1.5,
    linestyle="--",
    label="Setpoint",
)
add_12hr_lines(ax1, df_closed, "datetime")

ax1.set_title("Closed Loop Temperature", fontsize=14, fontweight="bold")
ax1.set_xlabel("Time", fontsize=12)
ax1.set_ylabel("Temperature (°C)", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.4)
ax1.xaxis.set_major_formatter(date_fmt)
ax1.legend(loc="upper left")
plt.setp(ax1.get_xticklabels(), visible=True, rotation=30, ha="right")

# Apply Closed Loop Axis Limits
if any(v is not None for v in AXIS_LIMITS["closed_loop"]["xlim"]):
    ax1.set_xlim(
        [
            pd.to_datetime(v) if v is not None else None
            for v in AXIS_LIMITS["closed_loop"]["xlim"]
        ]
    )
if any(v is not None for v in AXIS_LIMITS["closed_loop"]["ylim"]):
    ax1.set_ylim(AXIS_LIMITS["closed_loop"]["ylim"])


# --- Bottom Subplot: Open Loop ---
ax2.plot(
    df_open["Timestamp"],
    df_open["Temperature_C"],
    color="tab:orange",
    linewidth=1.5,
    label="Temperature",
)
ax2.plot(
    df_open["Timestamp"],
    df_open[open_sp_col],
    color="tab:red",
    linewidth=1.5,
    linestyle="--",
    label="Setpoint",
)
add_12hr_lines(ax2, df_open, "Timestamp")

ax2.set_title("Open Loop Temperature", fontsize=14, fontweight="bold")
ax2.set_xlabel("Time", fontsize=12)
ax2.set_ylabel("Temperature (°C)", fontsize=12)
ax2.grid(True, linestyle=":", alpha=0.4)
ax2.xaxis.set_major_formatter(date_fmt)
ax2.legend(loc="upper left")
plt.setp(ax2.get_xticklabels(), visible=True, rotation=30, ha="right")

# Apply Open Loop Axis Limits
if any(v is not None for v in AXIS_LIMITS["open_loop"]["xlim"]):
    ax2.set_xlim(
        [
            pd.to_datetime(v) if v is not None else None
            for v in AXIS_LIMITS["open_loop"]["xlim"]
        ]
    )
if any(v is not None for v in AXIS_LIMITS["open_loop"]["ylim"]):
    ax2.set_ylim(AXIS_LIMITS["open_loop"]["ylim"])


# Adjust padding across the canvas
plt.tight_layout(pad=3.0)

# Save figure in 16:9 widescreen format
plt.savefig(output_image_path, dpi=300)
print(f"Full-screen figure saved to: {output_image_path}")

# Trigger full screen display across GUI backends
fig_manager = plt.get_current_fig_manager()
try:
    fig_manager.full_screen_toggle()
except AttributeError:
    try:
        fig_manager.window.state("zoomed")
    except AttributeError:
        pass

plt.show()