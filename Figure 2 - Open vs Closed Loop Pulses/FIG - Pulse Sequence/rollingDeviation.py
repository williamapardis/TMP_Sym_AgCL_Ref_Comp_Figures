from dataclasses import dataclass
import json
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 1. Data Structure Definition
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
  calculated_ambient_temp_c: Optional[float] = None
  delta_temp_target_c: Optional[float] = None

  temp_last_on: Optional[float] = None
  temp_last_off: Optional[float] = None
  cp_last_on: Optional[float] = None
  cp_last_off: Optional[float] = None

  def extract_end_points(self):
    df = self.data
    rel_t = (
        (
            pd.to_datetime(df['datetime'])
            - pd.to_datetime(df['datetime'].iloc[0])
        ).dt.total_seconds()
        if 'datetime' in df.columns
        else (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
    )

    idx_on = (rel_t - self.time_on_s).abs().idxmin()
    idx_off = len(df) - 1

    self.temp_last_on = float(df['temperature'].iloc[idx_on])
    self.temp_last_off = float(df['temperature'].iloc[idx_off])

    cp_col = 'cp_zeroed' if 'cp_zeroed' in df.columns else 'AgCl CP'
    self.cp_last_on = float(df[cp_col].iloc[idx_on])
    self.cp_last_off = float(df[cp_col].iloc[idx_off])

  @classmethod
  def from_dict(cls, data: dict) -> 'Pulse':
    meta = data.get('metadata', {})
    ts_data = data.get('time_series', {})
    df = pd.DataFrame(ts_data)
    time_col = (
        'Timestamp' if data.get('loop_type') == 'open-loop' else 'datetime'
    )
    if time_col in df.columns and not pd.api.types.is_numeric_dtype(
        df[time_col]
    ):
      df[time_col] = pd.to_datetime(df[time_col])

    return cls(
        pulse_index=data.get('pulse_index', 0),
        loop_type=data.get('loop_type', ''),
        time_on_s=meta.get('time_on_s', 120.0),
        time_off_s=meta.get('time_off_s', 120.0),
        ambient_temp_setpoint_c=meta.get('ambient_temp_setpoint_c', 25.0),
        max_current_a=meta.get('max_current_a', 1.0),
        calculated_ambient_temp_c=meta.get('calculated_ambient_temp_c'),
        delta_temp_target_c=meta.get('delta_temp_target_c'),
        temp_last_on=meta.get('temp_last_on'),
        temp_last_off=meta.get('temp_last_off'),
        cp_last_on=meta.get('cp_last_on'),
        cp_last_off=meta.get('cp_last_off'),
        data=df,
    )


# ==========================================
# 2. Data Processing & Grid Alignment
# ==========================================
with open('closed_loop_pulses_25C.json', 'r') as f:
  pulses_json = json.load(f)

pulse_objects = []
interpolated_cp_list = []
common_grid = np.arange(0, 241, 1.0)

for p_dict in pulses_json:
  p = Pulse.from_dict(p_dict)
  dt = pd.to_datetime(p.data['datetime'])
  rel_t = (dt - dt.iloc[0]).dt.total_seconds().values

  # Zero CP against last 10s baseline
  max_t = rel_t[-1]
  baseline_cp = np.mean(p.data['AgCl CP'][rel_t >= (max_t - 10.0)])
  p.data['cp_zeroed'] = p.data['AgCl CP'] - baseline_cp

  p.extract_end_points()
  pulse_objects.append(p)

  # Interpolate onto common_grid
  cp_interp = np.interp(common_grid, rel_t, p.data['cp_zeroed'].values)
  interpolated_cp_list.append(cp_interp)

# Matrix & Rolling Standard Deviation
cp_df = pd.DataFrame(
    np.column_stack(interpolated_cp_list), index=common_grid
)
rolling_std_df = cp_df.rolling(window=5, min_periods=1, center=True).std()
mean_rolling_std = rolling_std_df.mean(axis=1)

# Transition End-Point Stats
t_on_mean, t_on_std = np.mean([p.temp_last_on for p in pulse_objects]), np.std(
    [p.temp_last_on for p in pulse_objects]
)
t_off_mean, t_off_std = np.mean(
    [p.temp_last_off for p in pulse_objects]
), np.std([p.temp_last_off for p in pulse_objects])
cp_on_mean, cp_on_std = np.mean([p.cp_last_on for p in pulse_objects]), np.std(
    [p.cp_last_on for p in pulse_objects]
)
cp_off_mean, cp_off_std = np.mean(
    [p.cp_last_off for p in pulse_objects]
), np.std([p.cp_last_off for p in pulse_objects])

# ==========================================
# 3. MANUAL TIME WINDOW DEFINITIONS & STATS
# ==========================================
win_max_bounds = (80, 115.0)  # Max Window Bounds (s)
win_min_bounds = (235.0, 240.0)  # Min Window Bounds (s)


def compute_window_stats(pulses, t_start, t_end):
  temp_vals, cp_vals = [], []
  for p in pulses:
    df = p.data
    rel_t = (
        (
            pd.to_datetime(df['datetime'])
            - pd.to_datetime(df['datetime'].iloc[0])
        ).dt.total_seconds()
        if 'datetime' in df.columns
        else (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
    )

    mask = (rel_t >= t_start) & (rel_t <= t_end)
    temp_vals.extend(df['temperature'][mask].values)
    cp_vals.extend(df['cp_zeroed'][mask].values)

  return (
      np.mean(temp_vals),
      np.std(temp_vals),
      np.mean(cp_vals),
      np.std(cp_vals),
  )


t_max_win_mean, t_max_win_std, cp_max_win_mean, cp_max_win_std = (
    compute_window_stats(pulse_objects, *win_max_bounds)
)
t_min_win_mean, t_min_win_std, cp_min_win_mean, cp_min_win_std = (
    compute_window_stats(pulse_objects, *win_min_bounds)
)

t_max_mid = (win_max_bounds[0] + win_max_bounds[1]) / 2.0
t_min_mid = (win_min_bounds[0] + win_min_bounds[1]) / 2.0

# ==========================================
# 4. Plot Generation
# ==========================================
plt.rcdefaults()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

pcb_orange = '#f99e1a'
red_std = '#dc2626'

for p in pulse_objects:
  rel_time = (
      pd.to_datetime(p.data['datetime']) - pd.to_datetime(p.data['datetime'].iloc[0])
  ).dt.total_seconds()
  ax1.plot(rel_time, p.data['temperature'], color=pcb_orange, alpha=0.55)
  ax2.plot(rel_time, p.data['cp_zeroed'], color='#0284c7', alpha=0.5)

# Secondary Y-axis for rolling standard deviation
ax2_twin = ax2.twinx()
ax2_twin.plot(
    common_grid,
    mean_rolling_std,
    color=red_std,
    linestyle='-',
    linewidth=1.5,
    alpha=0.85,
)
ax2_twin.set_ylabel(
    'Rolling Std Dev (mV)', color=red_std, fontweight='bold'
)
ax2_twin.tick_params(axis='y', labelcolor=red_std)

# Manual Axis Limits
ax1_ylim = (24.0, 31.0)
ax2_ylim = (-4.0, 18.0)
ax2_twin_ylim = (0.0, 0.10)

ax1.set_ylim(ax1_ylim)
ax2.set_ylim(ax2_ylim)
ax2_twin.set_ylim(ax2_twin_ylim)

# ------------------------------------------
# BACKGROUND SPANS & WINDOW OVERLAYS
# ------------------------------------------
# 1. Primary sequence phases
windows = [(0, 120, '#fee2e2'), (120, 180, '#e0f2fe'), (180, 240, '#bfdbfe')]

for start, end, bg_color in windows:
  for ax in (ax1, ax2):
    ax.axvspan(start, end, color=bg_color, alpha=0.4, zorder=0)
    ax.axvline(start, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.axvline(end, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)

# 2. Overlaid custom measurement windows
custom_windows = [
    (win_max_bounds[0], win_max_bounds[1], '#fde047'),  # Yellow for Max Window
    (win_min_bounds[0], win_min_bounds[1], '#c084fc'),  # Purple for Min Window
]

for start, end, highlight_color in custom_windows:
  for ax in (ax1, ax2):
    # Shaded region
    ax.axvspan(start, end, color=highlight_color, alpha=0.25, zorder=1)
    # Dotted boundary lines
    ax.axvline(
        start, color=highlight_color, linestyle=':', alpha=0.9, linewidth=1.2
    )
    ax.axvline(
        end, color=highlight_color, linestyle=':', alpha=0.9, linewidth=1.2
    )
# ------------------------------------------

# Points & Annotations
ax1.errorbar(
    120.0,
    t_on_mean,
    yerr=t_on_std,
    fmt='o',
    color='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)
ax1.errorbar(
    240.0,
    t_off_mean,
    yerr=t_off_std,
    fmt='o',
    color='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)
ax2.errorbar(
    120.0,
    cp_on_mean,
    yerr=cp_on_std,
    fmt='o',
    color='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)
ax2.errorbar(
    240.0,
    cp_off_mean,
    yerr=cp_off_std,
    fmt='o',
    color='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)

ax1.errorbar(
    t_max_mid,
    t_max_win_mean,
    yerr=t_max_win_std,
    fmt='o',
    color='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)
ax1.errorbar(
    t_min_mid,
    t_min_win_mean,
    yerr=t_min_win_std,
    fmt='o',
    color='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)
ax2.errorbar(
    t_max_mid,
    cp_max_win_mean,
    yerr=cp_max_win_std,
    fmt='o',
    color='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)
ax2.errorbar(
    t_min_mid,
    cp_min_win_mean,
    yerr=cp_min_win_std,
    fmt='o',
    color='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)

# Text Annotations (Transition Points)
ax1.annotate(
    f'{t_on_mean:.2f} \u00b1 {t_on_std:.3f} \u00b0C',
    (120.0, t_on_mean),
    textcoords='offset points',
    xytext=(-10, -15),
    ha='right',
    va='top',
    fontsize=9,
    fontweight='bold',
)
ax1.annotate(
    f'{t_off_mean:.2f} \u00b1 {t_off_std:.3f} \u00b0C',
    (240.0, t_off_mean),
    textcoords='offset points',
    xytext=(-10, 12),
    ha='right',
    va='bottom',
    fontsize=9,
    fontweight='bold',
)
ax2.annotate(
    f'{cp_on_mean:.2f} \u00b1 {cp_on_std:.3f} mV',
    (120.0, cp_on_mean),
    textcoords='offset points',
    xytext=(-10, -15),
    ha='right',
    va='top',
    fontsize=9,
    fontweight='bold',
)
ax2.annotate(
    f'{cp_off_mean:.2f} \u00b1 {cp_off_std:.3f} mV',
    (240.0, cp_off_mean),
    textcoords='offset points',
    xytext=(-10, 12),
    ha='right',
    va='bottom',
    fontsize=9,
    fontweight='bold',
)

# Text Annotations (Window Max/Min Points)
ax1.annotate(
    f'{t_max_win_mean:.2f} \u00b1 {t_max_win_std:.3f} \u00b0C',
    (t_max_mid, t_max_win_mean),
    textcoords='offset points',
    xytext=(0, +15),
    ha='center',
    va='top',
    fontsize=9,
    fontweight='bold',
)
ax1.annotate(
    f'{t_min_win_mean:.2f} \u00b1 {t_min_win_std:.3f} \u00b0C',
    (t_min_mid, t_min_win_mean),
    textcoords='offset points',
    xytext=(0, 12),
    ha='center',
    va='bottom',
    fontsize=9,
    fontweight='bold',
)

ax2.annotate(
    f'{cp_max_win_mean:.2f} \u00b1 {cp_max_win_std:.3f} mV',
    (t_max_mid, cp_max_win_mean),
    textcoords='offset points',
    xytext=(0, 12),
    ha='center',
    va='bottom',
    fontsize=9,
    fontweight='bold',
)
ax2.annotate(
    f'{cp_min_win_mean:.2f} \u00b1 {cp_min_win_std:.3f} mV',
    (t_min_mid, cp_min_win_mean),
    textcoords='offset points',
    xytext=(0, -15),
    ha='center',
    va='top',
    fontsize=9,
    fontweight='bold',
)

ax1.set_ylabel('Temperature (°C)')
ax2.set_ylabel(
    r'$\Delta$Cell Potential Ag/AgCl (mV)', color='#0284c7', fontweight='bold'
)
ax2.tick_params(axis='y', labelcolor='#0284c7')
ax2.set_xlabel('Relative Time (s)')
ax1.set_title('Closed-Loop Thermal & Potential Response Overlay')

plt.xlim(0, 240)
plt.tight_layout()
plt.show()