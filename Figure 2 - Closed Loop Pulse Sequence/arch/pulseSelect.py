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
  calculated_ambient_temp_c: Optional[float] = (
      None  # Saved to JSON metadata
  )
  delta_temp_target_c: Optional[float] = None  # Added to metadata

  # New properties for end-of-cycle values
  temp_last_on: Optional[float] = None
  temp_last_off: Optional[float] = None
  cp_last_on: Optional[float] = None
  cp_last_off: Optional[float] = None

  def extract_end_points(self):
    """Extracts the last point of the ON cycle (t ≈ time_on_s)

    and the last point of the OFF cycle (t ≈ end of pulse).
    """
    df = self.data
    if 'datetime' in df.columns:
      rel_time = (
          pd.to_datetime(df['datetime']) - pd.to_datetime(df['datetime'].iloc[0])
      ).dt.total_seconds()
    else:
      rel_time = (
          df['Timestamp'] - df['Timestamp'].iloc[0]
      ).dt.total_seconds()

    # Index for last point of ON phase (closest to time_on_s)
    idx_on = (rel_time - self.time_on_s).abs().idxmin()
    # Index for last point of OFF phase (final row of dataframe)
    idx_off = len(df) - 1

    self.temp_last_on = float(df['temperature'].iloc[idx_on])
    self.temp_last_off = float(df['temperature'].iloc[idx_off])

    # Extract CP (using zeroed potential if available, otherwise raw CP)
    cp_col = 'cp_zeroed' if 'cp_zeroed' in df.columns else 'AgCl CP'
    self.cp_last_on = float(df[cp_col].iloc[idx_on])
    self.cp_last_off = float(df[cp_col].iloc[idx_off])

  def to_dict(self) -> dict:
    """Converts metadata, extracted end points, and DataFrame time series to a JSON-serializable dictionary."""
    return {
        'pulse_index': self.pulse_index,
        'loop_type': self.loop_type,
        'metadata': {
            'time_on_s': self.time_on_s,
            'time_off_s': self.time_off_s,
            'ambient_temp_setpoint_c': self.ambient_temp_setpoint_c,
            'calculated_ambient_temp_c': self.calculated_ambient_temp_c,
            'max_current_a': self.max_current_a,
            'delta_temp_target_c': self.delta_temp_target_c,
            'temp_last_on': self.temp_last_on,
            'temp_last_off': self.temp_last_off,
            'cp_last_on': self.cp_last_on,
            'cp_last_off': self.cp_last_off,
        },
        'time_series': self.data.to_dict(orient='list'),
    }

  @classmethod
  def from_dict(cls, data: dict) -> 'Pulse':
    """Reconstructs a Pulse object from a JSON dictionary."""
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
# 2. Data Processing & Baseline Correction
# ==========================================
with open('closed_loop_pulses_25C.json', 'r') as f:
  pulses_json = json.load(f)

pulse_objects = []
for p_dict in pulses_json:
  p = Pulse.from_dict(p_dict)

  # Calculate zeroed potential baseline (last 10s avg)
  dt = pd.to_datetime(p.data['datetime'])
  rel_t = (dt - dt.iloc[0]).dt.total_seconds()
  max_t = rel_t.iloc[-1]
  baseline_cp = np.mean(p.data['AgCl CP'][rel_t >= (max_t - 10.0)])
  p.data['cp_zeroed'] = p.data['AgCl CP'] - baseline_cp

  # Extract end-point properties
  p.extract_end_points()
  pulse_objects.append(p)

# Calculate statistics across all pulses
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
# 3. Plot Generation
# ==========================================
plt.rcdefaults()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

pcb_orange = '#f99e1a'

for p in pulse_objects:
  rel_time = (
      pd.to_datetime(p.data['datetime']) - pd.to_datetime(p.data['datetime'].iloc[0])
  ).dt.total_seconds()
  ax1.plot(rel_time, p.data['temperature'], color=pcb_orange, alpha=0.55)
  ax2.plot(rel_time, p.data['cp_zeroed'], color='#0284c7', alpha=0.5)

windows = [(0, 120, '#fee2e2'), (120, 180, '#e0f2fe'), (180, 240, '#bfdbfe')]

for start, end, bg_color in windows:
  for ax in (ax1, ax2):
    ax.axvspan(start, end, color=bg_color, alpha=0.5, zorder=0)
    ax.axvline(start, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.axvline(end, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)

# Add error bars at t=120s (end of ON) and t=240s (end of OFF)
ax1.errorbar(
    120.0,
    t_on_mean,
    yerr=t_on_std,
    fmt='o',
    color='black',
    ecolor='black',
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
    ecolor='black',
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
    ecolor='black',
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
    ecolor='black',
    capsize=4,
    capthick=1.2,
    elinewidth=1.2,
    markersize=5,
    zorder=5,
)

# Text labels of form avg ± stdev without bounding boxes
ax1.annotate(
    f'{t_on_mean:.2f} \u00b1 {t_on_std:.2f} \u00b0C',
    (120.0, t_on_mean),
    textcoords='offset points',
    xytext=(-15, -15),
    ha='right',
    va='top',
    fontsize=9,
    fontweight='bold',
)

ax1.annotate(
    f'{t_off_mean:.2f} \u00b1 {t_off_std:.2f} \u00b0C',
    (240.0, t_off_mean),
    textcoords='offset points',
    xytext=(-15, 12),
    ha='right',
    va='bottom',
    fontsize=9,
    fontweight='bold',
)

ax2.annotate(
    f'{cp_on_mean:.2f} \u00b1 {cp_on_std:.2f} mV',
    (120.0, cp_on_mean),
    textcoords='offset points',
    xytext=(-15, -15),
    ha='right',
    va='top',
    fontsize=9,
    fontweight='bold',
)

ax2.annotate(
    f'{cp_off_mean:.2f} \u00b1 {cp_off_std:.2f} mV',
    (240.0, cp_off_mean),
    textcoords='offset points',
    xytext=(-15, 12),
    ha='right',
    va='bottom',
    fontsize=9,
    fontweight='bold',
)

ax1.set_ylabel('Temperature (°C)')
ax2.set_ylabel(r'$\Delta$Cell Potential Ag/AgCl (mV)')
ax2.set_xlabel('Relative Time (s)')
ax1.set_title('Closed-Loop Thermal & Potential Response Overlay')

plt.xlim(0, 240)
plt.tight_layout()
plt.savefig('closed_loop_pulses_merged.png', dpi=300)
plt.show()