from dataclasses import dataclass
import json
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
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
        if 'datetime' in df.columns:
            rel_time = (
                pd.to_datetime(df['datetime']) - pd.to_datetime(df['datetime'].iloc[0])
            ).dt.total_seconds()
        else:
            rel_time = (
                df['Timestamp'] - df['Timestamp'].iloc[0]
            ).dt.total_seconds()

        idx_on = (rel_time - self.time_on_s).abs().idxmin()
        idx_off = len(df) - 1

        self.temp_last_on = float(df['temperature'].iloc[idx_on])
        self.temp_last_off = float(df['temperature'].iloc[idx_off])

        cp_col = 'cp_zeroed' if 'cp_zeroed' in df.columns else 'AgCl CP'
        self.cp_last_on = float(df[cp_col].iloc[idx_on])
        self.cp_last_off = float(df[cp_col].iloc[idx_off])

    def to_dict(self) -> dict:
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

    dt = pd.to_datetime(p.data['datetime'])
    rel_t = (dt - dt.iloc[0]).dt.total_seconds()
    max_t = rel_t.iloc[-1]
    baseline_cp = np.mean(p.data['AgCl CP'][rel_t >= (max_t - 10.0)])
    p.data['cp_zeroed'] = p.data['AgCl CP'] - baseline_cp

    p.extract_end_points()
    pulse_objects.append(p)

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

delta_temp_list = [p.temp_last_on - p.temp_last_off for p in pulse_objects]
delta_temp_mean = np.mean(delta_temp_list)
delta_temp_std = np.std(delta_temp_list)

delta_cp_list = [p.cp_last_on - p.cp_last_off for p in pulse_objects]
delta_cp_mean = np.mean(delta_cp_list)
delta_cp_std = np.std(delta_cp_list)

# ==========================================
# 3. Plot Generation
# ==========================================
plt.rcdefaults()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

pcb_orange = '#f99e1a'

for p in pulse_objects:
    rel_time = (
        pd.to_datetime(p.data['datetime']) - pd.to_datetime(p.data['datetime'].iloc[0])
    ).dt.total_seconds()

    shifted_time = np.where(rel_time >= 180.0, rel_time - 240.0, rel_time)

    sort_idx = np.argsort(shifted_time)
    shifted_time = shifted_time[sort_idx]
    temp_sorted = p.data['temperature'].iloc[sort_idx]
    cp_sorted = p.data['cp_zeroed'].iloc[sort_idx]

    ax1.plot(shifted_time, temp_sorted, color=pcb_orange, alpha=0.55)
    ax2.plot(shifted_time, cp_sorted, color='#0284c7', alpha=0.5)

windows = [(-60, 0, '#bfdbfe'), (0, 120, '#fee2e2'), (120, 180, '#bfdbfe')]

for start, end, bg_color in windows:
    for ax in (ax1, ax2):
        ax.axvspan(start, end, color=bg_color, alpha=0.5, zorder=0)
        ax.axvline(start, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)
        ax.axvline(end, color='gray', linestyle='--', alpha=0.6, linewidth=0.8)

# Reference lines & errorbars
ax1.axhline(t_on_mean, color=pcb_orange, linestyle='--', alpha=0.6, linewidth=1, zorder=3)
ax1.axhline(t_off_mean, color=pcb_orange, linestyle='--', alpha=0.6, linewidth=1, zorder=3)
ax2.axhline(cp_on_mean, color='#0284c7', linestyle='--', alpha=0.6, linewidth=1, zorder=3)
ax2.axhline(cp_off_mean, color='#0284c7', linestyle='--', alpha=0.6, linewidth=1, zorder=3)

ax1.errorbar(120.0, t_on_mean, yerr=t_on_std, fmt='o', color='black', ecolor='black', capsize=4, capthick=1.2, elinewidth=1.2, markersize=5, zorder=5)
ax1.errorbar(0.0, t_off_mean, yerr=t_off_std, fmt='o', color='black', ecolor='black', capsize=4, capthick=1.2, elinewidth=1.2, markersize=5, zorder=5)
ax2.errorbar(120.0, cp_on_mean, yerr=cp_on_std, fmt='o', color='black', ecolor='black', capsize=4, capthick=1.2, elinewidth=1.2, markersize=5, zorder=5)
ax2.errorbar(0.0, cp_off_mean, yerr=cp_off_std, fmt='o', color='black', ecolor='black', capsize=4, capthick=1.2, elinewidth=1.2, markersize=5, zorder=5)

# Annotations
ax1.annotate(rf'$T_{{on}}$', (120.0, t_on_mean), textcoords='offset points', xytext=(-10, 10), ha='right', va='bottom', fontsize=14, fontweight='bold', color=pcb_orange)
ax1.annotate(rf'$T_{{off}}$', (0.0, t_off_mean), textcoords='offset points', xytext=(-10, 10), ha='right', va='bottom', fontsize=14, fontweight='bold', color=pcb_orange)
ax2.annotate(rf'$E_{{on}}$', (120.0, cp_on_mean), textcoords='offset points', xytext=(-10, 10), ha='right', va='bottom', fontsize=14, fontweight='bold', color='#0284c7')
ax2.annotate(rf'$E_{{off}}$', (0.0, cp_off_mean), textcoords='offset points', xytext=(-10, 10), ha='right', va='bottom', fontsize=14, fontweight='bold', color='#0284c7')

# Data Labels (Summary statistics enclosed in styled boxes with matching colors)
bbox_props_temp = dict(boxstyle="round,pad=0.3", fc="white", ec=pcb_orange, lw=1.5, alpha=0.9)
ax1.text(
    0.55, 0.5,
    rf'$\Delta T_{{on-off}} = {delta_temp_mean:.2f} \pm {delta_temp_std:.2f}\ ^\circ\mathrm{{C}}$',
    transform=ax1.transAxes, fontsize=16, fontweight='bold', color=pcb_orange,
    va='top', ha='center',
)

bbox_props_cp = dict(boxstyle="round,pad=0.3", fc="white", ec='#0284c7', lw=1.5, alpha=0.9)
ax2.text(
    0.55, 0.5,
    rf'$\Delta E_{{on-off}} = {delta_cp_mean:.2f} \pm {delta_cp_std:.2f}\ \mathrm{{mV}}$',
    transform=ax2.transAxes, fontsize=16, fontweight='bold', color='#0284c7',
    va='top', ha='center',
)

# Axis configuration & Colors
ax1.set_ylabel('Temperature [°C]', color=pcb_orange, fontweight='bold', fontsize=14)
ax1.tick_params(axis='y', colors=pcb_orange, labelcolor=pcb_orange)
ax1.spines['left'].set_color(pcb_orange)

ax2.set_ylabel('AgCl|AgCl Potential [mV]', color='#0284c7', fontweight='bold', fontsize=14)
ax2.tick_params(axis='y', colors='#0284c7', labelcolor='#0284c7')
ax2.spines['left'].set_color('#0284c7')

ax2.set_xlabel('Time [s]')

plt.xlim(-60, 180)

# Set 60-second X-ticks interval [-60, 0, 60, 120, 180]
ax2.xaxis.set_major_locator(ticker.MultipleLocator(60))

# Enable X-tick numerical labels on ax1 as well
ax1.tick_params(axis='x', labelbottom=True)

# Set Y-axis limits
ax1.set_ylim(24, 31)
ax2.set_ylim(-2.5, 16)

# ==============================================================================
# Custom Y-Ticks and Decimal Precision Settings
# ==============================================================================
# ax1 (Temperature): Manually defined ticks & formatted to 1 decimal place (e.g. 24.0, 26.0)
ax1.set_yticks([24.0, t_off_mean, t_on_mean, 31.0])
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# ax2 (Cell Potential): Manually defined ticks & formatted to 2 decimal places (e.g. 0.00, 5.00)
ax2.set_yticks([-2.5, cp_off_mean, cp_on_mean, 16.0])
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

# Phase Labels directly above ax1
trans_inter = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
phase_labels = [
    (-30.0, "1. zero equilibration", "#3890fc"),
    (60.0, "2. Heating - PID Enabled", "#fc5050"),
    (150.0, "3. flushing cell", '#3890fc'),
]

y_position = 1.03  # Positions labels just above the top edge of ax1 frame

for x_time, label_text, textColor in phase_labels:
    ax1.text(
        x_time,
        y_position,
        label_text,
        transform=trans_inter,
        ha='center',
        va='bottom',
        fontsize=16,
        fontweight='bold',
        color=textColor,
        clip_on=False,
    )

plt.subplots_adjust(top=0.88, hspace=0.3)
plt.savefig('closed_loop_pulses_merged.png', dpi=300)
plt.show()