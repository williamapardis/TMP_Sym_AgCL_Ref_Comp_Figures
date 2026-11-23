import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================

# 1. Define Multiple Selected Time Windows
WINDOWS = [
    ('2026-09-13 14:42:00', '2026-09-13 20:38:00'),
    ('2026-09-14 09:42:00', '2026-09-14 15:38:00'),
    ('2026-09-15 04:42:00', '2026-09-15 10:38:00'),
]

# 2. Temperature Threshold Selection for Subplots 3 & 4
TEMP_FILTER_BOUNDS = (14.0, 16.0)  # (min_temp, max_temp)

# Highlight shading colors for top subplot
WINDOW_SHADES = ['gold', 'mediumspringgreen', 'violet']

# Trace colors for each window: (Temp Color, AgCl Color, TRA Color)
WINDOW_TRACE_COLORS = [
    ('tab:red', 'tab:blue'),      # Window 1: Red / Blue
    ('darkorange', 'teal'),       # Window 2: Orange / Teal
    ('purple', 'forestgreen')     # Window 3: Purple / Green
]

# 3. Axis Limits (Set to None for automatic scaling)
TEMP_Y_LIMITS = (14.0, 36.0)  # (ymin, ymax) for Temperature Setpoint
CP_Y_LIMITS   = None          # (ymin, ymax) for AgCl CP Values
TRA_Y_LIMITS  = None          # (ymin, ymax) for TRA Values
X_TIME_LIMITS = None          # Optional manual limits for top plot

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_continuous_segments(df_win, time_col, mask_col):
    """Finds start and end times for continuous True blocks in mask_col."""
    if df_win.empty:
        return []
    
    mask = df_win[mask_col].values
    times = df_win[time_col].values
    
    if not np.any(mask):
        return []

    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask) - 1)
        
    segments = []
    for s, e in zip(starts, ends):
        segments.append((times[s], times[e]))
    return segments

# ==========================================
# DATA LOADING & CLEANING
# ==========================================

df = pd.read_csv('closed_loop_elec_35_25_15C.txt', skipinitialspace=True)

# Strip leading and trailing whitespace from column names to prevent KeyErrors
df.columns = df.columns.str.strip()

# Parse datetime safely
df['datetime'] = pd.to_datetime(df['datetime'])

# Ensure relevant numeric columns are floats
numeric_cols = ['temperature setpoint', 'temperature', 'AgCl CP', 'L.J. CP', 'TRA']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df_windows = []
for idx, (start, end) in enumerate(WINDOWS):
    t_start = pd.to_datetime(start)
    t_end   = pd.to_datetime(end)
    
    sub_df = df[(df['datetime'] >= t_start) & (df['datetime'] <= t_end)].copy()
    
    df_filtered = pd.DataFrame()
    time_segments = []
    
    if not sub_df.empty:
        sub_df['elapsed_hours'] = (sub_df['datetime'] - t_start).dt.total_seconds() / 3600.0
        
        t_min, t_max = TEMP_FILTER_BOUNDS
        sub_df['in_temp_range'] = (
            (sub_df['temperature setpoint'] >= t_min) & 
            (sub_df['temperature setpoint'] <= t_max)
        )
        
        # Insert NaNs where temperature is out of bounds so lines gap cleanly
        df_filtered = sub_df.copy()
        df_filtered.loc[~df_filtered['in_temp_range'], ['AgCl CP', 'TRA']] = np.nan
        
        time_segments = get_continuous_segments(sub_df, 'elapsed_hours', 'in_temp_range')

    df_windows.append({
        'start': t_start,
        'end': t_end,
        'df': sub_df,
        'df_filtered': df_filtered,
        'time_segments': time_segments,
        'label': f"Win {idx+1} ({t_start.strftime('%m/%d %H:%M')})",
        'temp_color': WINDOW_TRACE_COLORS[idx % len(WINDOW_TRACE_COLORS)][0],
        'cp_color':   WINDOW_TRACE_COLORS[idx % len(WINDOW_TRACE_COLORS)][1],
        'shade_color': WINDOW_SHADES[idx % len(WINDOW_SHADES)]
    })

# ==========================================
# PLOTTING (4 SUBPLOTS)
# ==========================================

fig, (ax_top, ax_mid, ax_bot, ax_tra) = plt.subplots(4, 1, figsize=(12, 16))

# ------------------------------------------
# 1. TOP SUBPLOT: Full Dataset Overview
# ------------------------------------------
ax_top_cp = ax_top.twinx()
ax_top.set_zorder(ax_top_cp.get_zorder() + 1)
ax_top.patch.set_visible(False)

ax_top.set_xlabel('Time', fontsize=10, fontweight='bold')
ax_top.set_ylabel('Temperature Setpoint (°C)', color='tab:red', fontsize=10, fontweight='bold')
line_temp_full = ax_top.plot(
    df['datetime'], df['temperature setpoint'], 
    color='tab:red', alpha=0.35, linewidth=1.5, zorder=3, label='Temp Setpoint (Full)'
)
ax_top.tick_params(axis='y', labelcolor='tab:red')
ax_top.grid(True, linestyle='--', alpha=0.5)

ax_top_cp.set_ylabel('AgCl CP', color='tab:blue', fontsize=10, fontweight='bold')
line_agcl_full = ax_top_cp.plot(
    df['datetime'], df['AgCl CP'], 
    color='tab:blue', alpha=0.35, linewidth=1.5, zorder=2, label='AgCl CP (Full)'
)
ax_top_cp.tick_params(axis='y', labelcolor='tab:blue')

span_handles = []
for win in df_windows:
    span = ax_top.axvspan(
        win['start'], win['end'], color=win['shade_color'], alpha=0.3, zorder=1, 
        label=win['label']
    )
    span_handles.append(span)

    if not win['df'].empty:
        ax_top.plot(
            win['df']['datetime'], win['df']['temperature setpoint'], 
            color=win['temp_color'], linewidth=2, zorder=5
        )
        ax_top_cp.plot(
            win['df']['datetime'], win['df']['AgCl CP'], 
            color=win['cp_color'], linewidth=2, zorder=4
        )

if TEMP_Y_LIMITS is not None:
    ax_top.set_ylim(TEMP_Y_LIMITS)
if CP_Y_LIMITS is not None:
    ax_top_cp.set_ylim(CP_Y_LIMITS)

if X_TIME_LIMITS is not None:
    ax_top.set_xlim(pd.to_datetime(X_TIME_LIMITS[0]), pd.to_datetime(X_TIME_LIMITS[1]))
else:
    ax_top.set_xlim(df['datetime'].min(), df['datetime'].max())

lines_top = line_temp_full + line_agcl_full + span_handles
labels_top = [l.get_label() for l in lines_top]
ax_top.legend(lines_top, labels_top, loc='upper left', framealpha=0.9, fontsize=8)
ax_top.set_title('Full Dataset with Selected Windows Overlay', fontsize=11, fontweight='bold')

# ------------------------------------------
# 2. MIDDLE SUBPLOT: Overlaid Elapsed Time View
# ------------------------------------------
ax_mid_cp = ax_mid.twinx()
ax_mid.set_zorder(ax_mid_cp.get_zorder() + 1)
ax_mid.patch.set_visible(False)

ax_mid.set_xlabel('Elapsed Time (Hours)', fontsize=10, fontweight='bold')
ax_mid.set_ylabel('Temperature Setpoint (°C)', color='tab:red', fontsize=10, fontweight='bold')
ax_mid_cp.set_ylabel('AgCl CP', color='tab:blue', fontsize=10, fontweight='bold')

mid_lines = []
highlight_added = False

for win in df_windows:
    if win['df'].empty:
        continue

    for idx_seg, (t_start_seg, t_end_seg) in enumerate(win['time_segments']):
        span_label = 'Selected Temp Time Range' if not highlight_added else "_nolegend_"
        shade_span = ax_mid.axvspan(
            t_start_seg, t_end_seg, color='gray', alpha=0.15, zorder=1, label=span_label
        )
        if not highlight_added:
            mid_lines.append(shade_span)
            highlight_added = True

    line_t = ax_mid.plot(
        win['df']['elapsed_hours'], win['df']['temperature setpoint'], 
        color=win['temp_color'], linewidth=2, zorder=5, 
        label=f"Temp - {win['label']}"
    )
    line_c = ax_mid_cp.plot(
        win['df']['elapsed_hours'], win['df']['AgCl CP'], 
        color=win['cp_color'], linewidth=2, linestyle='--', zorder=4, 
        label=f"AgCl CP - {win['label']}"
    )
    mid_lines.extend(line_t + line_c)

ref_line1 = ax_mid.axhline(
    TEMP_FILTER_BOUNDS[0], color='red', linestyle=':', linewidth=1.2, zorder=6, 
    label=f'Temp Bounds [{TEMP_FILTER_BOUNDS[0]}°C, {TEMP_FILTER_BOUNDS[1]}°C]'
)
ax_mid.axhline(TEMP_FILTER_BOUNDS[1], color='red', linestyle=':', linewidth=1.2, zorder=6)
mid_lines.append(ref_line1)

ax_mid.tick_params(axis='y', labelcolor='tab:red')
ax_mid_cp.tick_params(axis='y', labelcolor='tab:blue')
ax_mid.grid(True, linestyle='--', alpha=0.5)

if TEMP_Y_LIMITS is not None:
    ax_mid.set_ylim(TEMP_Y_LIMITS)
if CP_Y_LIMITS is not None:
    ax_mid_cp.set_ylim(CP_Y_LIMITS)

labels_mid = [l.get_label() for l in mid_lines]
ax_mid.legend(mid_lines, labels_mid, loc='upper left', framealpha=0.9, fontsize=8, ncol=2)
ax_mid.set_title(
    f'Overlaid Windows (Vertical Shading: Time Intervals matching Temp [{TEMP_FILTER_BOUNDS[0]}°C, {TEMP_FILTER_BOUNDS[1]}°C])', 
    fontsize=11, fontweight='bold'
)

# ------------------------------------------
# 3. THIRD SUBPLOT: Temperature-Filtered AgCl CP Only
# ------------------------------------------
ax_bot.set_xlabel('Elapsed Time (Hours)', fontsize=10, fontweight='bold')
ax_bot.set_ylabel('AgCl CP', fontsize=10, fontweight='bold')
ax_bot.grid(True, linestyle='--', alpha=0.5)

bot_lines = []
has_filtered_data = False

for win in df_windows:
    if not win['df_filtered'].dropna(subset=['AgCl CP']).empty:
        has_filtered_data = True
        line_cp = ax_bot.plot(
            win['df_filtered']['elapsed_hours'], win['df_filtered']['AgCl CP'], 
            color=win['cp_color'], linewidth=2, 
            label=f"AgCl CP ({win['label']})"
        )
        bot_lines.extend(line_cp)

if CP_Y_LIMITS is not None:
    ax_bot.set_ylim(CP_Y_LIMITS)

if has_filtered_data:
    labels_bot = [l.get_label() for l in bot_lines]
    ax_bot.legend(bot_lines, labels_bot, loc='upper left', framealpha=0.9, fontsize=8)
else:
    ax_bot.text(
        0.5, 0.5, 
        f"No data matching Temp Setpoint range [{TEMP_FILTER_BOUNDS[0]}°C, {TEMP_FILTER_BOUNDS[1]}°C]", 
        ha='center', va='center', transform=ax_bot.transAxes, fontsize=11
    )

ax_bot.set_title(
    f'Filtered Window: AgCl CP Only for Temp Setpoints in [{TEMP_FILTER_BOUNDS[0]}°C, {TEMP_FILTER_BOUNDS[1]}°C]', 
    fontsize=11, fontweight='bold'
)

# ------------------------------------------
# 4. FOURTH SUBPLOT: Temperature-Filtered TRA Only
# ------------------------------------------
ax_tra.set_xlabel('Elapsed Time (Hours)', fontsize=10, fontweight='bold')
ax_tra.set_ylabel('TRA', fontsize=10, fontweight='bold')
ax_tra.grid(True, linestyle='--', alpha=0.5)

tra_lines = []
has_tra_data = False

for win in df_windows:
    if not win['df_filtered'].dropna(subset=['TRA']).empty:
        has_tra_data = True
        line_tra = ax_tra.plot(
            win['df_filtered']['elapsed_hours'], win['df_filtered']['TRA'], 
            color=win['cp_color'], linewidth=2, linestyle='-', 
            label=f"TRA ({win['label']})"
        )
        tra_lines.extend(line_tra)

if TRA_Y_LIMITS is not None:
    ax_tra.set_ylim(TRA_Y_LIMITS)

if has_tra_data:
    labels_tra = [l.get_label() for l in tra_lines]
    ax_tra.legend(tra_lines, labels_tra, loc='upper left', framealpha=0.9, fontsize=8)
else:
    ax_tra.text(
        0.5, 0.5, 
        f"No data matching Temp Setpoint range [{TEMP_FILTER_BOUNDS[0]}°C, {TEMP_FILTER_BOUNDS[1]}°C]", 
        ha='center', va='center', transform=ax_tra.transAxes, fontsize=11
    )

ax_tra.set_title(
    f'Filtered Window: TRA Only for Temp Setpoints in [{TEMP_FILTER_BOUNDS[0]}°C, {TEMP_FILTER_BOUNDS[1]}°C]', 
    fontsize=11, fontweight='bold'
)

fig.autofmt_xdate()
fig.tight_layout()

plt.savefig('full_series_overlaid_and_temp_time_shaded.png', dpi=300)
plt.show()