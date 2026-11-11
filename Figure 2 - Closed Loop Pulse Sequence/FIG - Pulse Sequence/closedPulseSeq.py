from dataclasses import dataclass
import json
import math
from typing import List, Optional
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
    loop_type: str  # 'closed-loop'
    time_on_s: float
    time_off_s: float
    ambient_temp_setpoint_c: float
    max_current_a: float
    step: int  # Step index from CSV data
    data: pd.DataFrame
    calculated_ambient_temp_c: Optional[float] = None
    delta_temp_target_c: Optional[float] = None

    temp_last_on: Optional[float] = None
    temp_last_off: Optional[float] = None
    cp_last_on: Optional[float] = None
    cp_last_off: Optional[float] = None

    def __post_init__(self):
        """Ensures numeric metadata fields are strictly typed regardless of input source."""
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

    def extract_end_points(self):
        """Extracts temperature and potential values at the end of ON and OFF periods."""
        df = self.data
        rel_time = (
            pd.to_datetime(df["datetime"]) - pd.to_datetime(df["datetime"].iloc[0])
        ).dt.total_seconds()

        idx_on = (rel_time - self.time_on_s).abs().idxmin()
        idx_off = len(df) - 1

        self.temp_last_on = float(df["temperature"].iloc[idx_on])
        self.temp_last_off = float(df["temperature"].iloc[idx_off])

        cp_col = "cp_zeroed" if "cp_zeroed" in df.columns else "AgCl CP"
        self.cp_last_on = float(df[cp_col].iloc[idx_on])
        self.cp_last_off = float(df[cp_col].iloc[idx_off])

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
                "temp_last_on": self.temp_last_on,
                "temp_last_off": self.temp_last_off,
                "cp_last_on": self.cp_last_on,
                "cp_last_off": self.cp_last_off,
            },
            "time_series": self.data.to_dict(orient="list"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pulse":
        """Reconstructs a Pulse object from a JSON dictionary."""
        meta = data.get("metadata", {})
        ts_data = data.get("time_series", {})

        df = pd.DataFrame(ts_data)
        if "datetime" in df.columns and not pd.api.types.is_numeric_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])

        return cls(
            pulse_index=data.get("pulse_index", 0),
            loop_type=data.get("loop_type", "closed-loop"),
            time_on_s=meta.get("time_on_s", 120.0),
            time_off_s=meta.get("time_off_s", 120.0),
            ambient_temp_setpoint_c=meta.get("ambient_temp_setpoint_c", 25.0),
            max_current_a=meta.get("max_current_a", 1.0),
            step=meta.get("step", 0),
            calculated_ambient_temp_c=meta.get("calculated_ambient_temp_c"),
            delta_temp_target_c=meta.get("delta_temp_target_c"),
            temp_last_on=meta.get("temp_last_on"),
            temp_last_off=meta.get("temp_last_off"),
            cp_last_on=meta.get("cp_last_on"),
            cp_last_off=meta.get("cp_last_off"),
            data=df,
        )


# ==========================================
# 2. Data Processing & Per-Step Baseline Correction
# ==========================================
with open("closed_loop_pulses_retained_0_4_9.json", "r") as f:
    pulses_json = json.load(f)

pulse_objects = []
for p_dict in pulses_json:
    p = Pulse.from_dict(p_dict)

    dt = pd.to_datetime(p.data["datetime"])
    rel_t = (dt - dt.iloc[0]).dt.total_seconds()
    max_t = rel_t.iloc[-1]

    # Baseline correction for AgCl CP based on final 10 seconds of pulse
    baseline_cp = np.mean(p.data["AgCl CP"][rel_t >= (max_t - 10.0)])
    p.data["cp_zeroed"] = p.data["AgCl CP"] - baseline_cp

    p.extract_end_points()
    pulse_objects.append(p)

# Global Temperature Stats
t_on_mean, t_on_std = np.mean([p.temp_last_on for p in pulse_objects]), np.std(
    [p.temp_last_on for p in pulse_objects]
)
t_off_mean, t_off_std = np.mean(
    [p.temp_last_off for p in pulse_objects]
), np.std([p.temp_last_off for p in pulse_objects])

delta_temp_list = [p.temp_last_on - p.temp_last_off for p in pulse_objects]
delta_temp_mean = np.mean(delta_temp_list)
delta_temp_std = np.std(delta_temp_list)

# Group Pulses by Step for Cell Potential Stats
steps = sorted(list(set(p.step for p in pulse_objects)))
step_stats = {}

for s in steps:
    s_pulses = [p for p in pulse_objects if p.step == s]
    cp_on_vals = [p.cp_last_on for p in s_pulses]
    cp_off_vals = [p.cp_last_off for p in s_pulses]
    delta_cp_vals = [p.cp_last_on - p.cp_last_off for p in s_pulses]

    step_stats[s] = {
        "pulses": s_pulses,
        "cp_on_mean": np.mean(cp_on_vals),
        "cp_on_std": np.std(cp_on_vals),
        "cp_off_mean": np.mean(cp_off_vals),
        "cp_off_std": np.std(cp_off_vals),
        "delta_cp_mean": np.mean(delta_cp_vals),
        "delta_cp_std": np.std(delta_cp_vals),
    }

# Assign distinct shades of blue to each step
blue_shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(steps)))
step_color_map = {s: blue_shades[i] for i, s in enumerate(steps)}


# ==========================================
# 3. Interpolation onto Common Time Grid for Mean & STD
# ==========================================
R_OHMS = 2.0
common_time = np.linspace(-60, 180, 241)  # 1-second resolution

temp_matrix = []
pwr_matrix = []
step_cp_matrices = {s: [] for s in steps}

for p in pulse_objects:
    rel_time = (
        pd.to_datetime(p.data["datetime"]) - pd.to_datetime(p.data["datetime"].iloc[0])
    ).dt.total_seconds()

    shifted_time = np.where(rel_time >= 180.0, rel_time - 240.0, rel_time)
    sort_idx = np.argsort(shifted_time)
    t_sorted = shifted_time[sort_idx]

    temp_sorted = p.data["temperature"].iloc[sort_idx].values
    cp_sorted = p.data["cp_zeroed"].iloc[sort_idx].values

    # Interpolate Temperature & CP onto common_time
    interp_temp = np.interp(common_time, t_sorted, temp_sorted)
    interp_cp = np.interp(common_time, t_sorted, cp_sorted)

    temp_matrix.append(interp_temp)
    step_cp_matrices[p.step].append(interp_cp)

    if "current" in p.data.columns:
        current_sorted = p.data["current"].iloc[sort_idx].values
        pwr_sorted_w = np.square(current_sorted / 1000.0) * R_OHMS
        interp_pwr = np.interp(common_time, t_sorted, pwr_sorted_w)
        pwr_matrix.append(interp_pwr)

# Compute Mean & STD arrays
temp_mean = np.mean(temp_matrix, axis=0)
temp_std = np.std(temp_matrix, axis=0)

if pwr_matrix:
    pwr_mean = np.mean(pwr_matrix, axis=0)
    pwr_std = np.std(pwr_matrix, axis=0)


# ==========================================
# 4. Plot Generation
# ==========================================
plt.rcdefaults()

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(10, 10),
    gridspec_kw={"hspace": 0.55}  # Creates whitespace between ax1 and ax2
)

# Twin Y-axis for Power on Subplot 1
ax1_pwr = ax1.twinx()

# Overlay ax1_pwr directly on top of ax1's left spine (position = (0, 0))
ax1_pwr.spines["left"].set_position(("axes", 0))
ax1_pwr.spines["left"].set_visible(True)
ax1_pwr.spines["right"].set_visible(False)
ax1_pwr.yaxis.set_label_position("left")
ax1_pwr.yaxis.tick_left()

pcb_orange = "#f99e1a"
power_red = "#dc2626"

# ==============================================================================
# Legend Location Configurations (Normalized Axes Coordinates: 0.0 to 1.0)
# ==============================================================================
# Adjust (x, y) coordinates of the bbox_to_anchor to place legends precisely
ax1_legend_loc = "upper right"
ax1_legend_bbox = (0.96, 0.88)

ax2_legend_loc = "upper right"
ax2_legend_bbox = (0.93, 0.92)


# 1. Plot Temperature Mean & Error Band on ax1 (Primary Left Axis, Orange)
ax1.plot(common_time, temp_mean, color=pcb_orange, linewidth=2.0, label="Temperature [°C]")
ax1.fill_between(
    common_time,
    temp_mean - temp_std,
    temp_mean + temp_std,
    color=pcb_orange,
    alpha=0.25,
    zorder=2,
)

# 2. Plot Power Mean & Error Band on ax1_pwr (Overlaid Left Axis, Red)
if pwr_matrix:
    ax1_pwr.plot(common_time, pwr_mean, color=power_red, linewidth=1.5, label="Heater Power [W]")
    ax1_pwr.fill_between(
        common_time,
        pwr_mean - pwr_std,
        pwr_mean + pwr_std,
        color=power_red,
        alpha=0.20,
        zorder=2,
    )

# Max Power Horizontal Reference Line & Text Label at 1.71 W
ax1_pwr.axhline(1.71, color=power_red, linestyle="--", alpha=0.7, linewidth=1.2, zorder=3)
ax1_pwr.annotate(
    "Max Power",
    (0, 1.71),
    textcoords="offset points",
    xytext=(-10, 5),
    ha="left",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color=power_red,
)

# 3. Plot AgCl CP Mean & Error Band per Step on ax2 (Blue Shades by Step)
for s in steps:
    s_color = step_color_map[s]
    cp_mat = np.array(step_cp_matrices[s])
    cp_mean = np.mean(cp_mat, axis=0)
    cp_std = np.std(cp_mat, axis=0)

    ax2.plot(common_time, cp_mean, color=s_color, linewidth=1.8, label=f"Step {s}")
    ax2.fill_between(
        common_time,
        cp_mean - cp_std,
        cp_mean + cp_std,
        color=s_color,
        alpha=0.25,
        zorder=2,
    )

# Shaded Phase Windows
windows = [(-60, 0, "#bfdbfe"), (0, 120, "#fee2e2"), (120, 180, "#bfdbfe")]

for start, end, bg_color in windows:
    for ax in (ax1, ax2):
        ax.axvspan(start, end, color=bg_color, alpha=0.5, zorder=0)
        ax.axvline(start, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.axvline(end, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)

# Temperature Reference Lines & Error Bars (on ax1)
ax1.axhline(t_on_mean, color=pcb_orange, linestyle="--", alpha=0.6, linewidth=1, zorder=3)
ax1.axhline(t_off_mean, color=pcb_orange, linestyle="--", alpha=0.6, linewidth=1, zorder=3)

ax1.errorbar(120.0, t_on_mean, yerr=t_on_std, fmt="o", color="black", ecolor="black", capsize=4, capthick=1.2, elinewidth=1.2, markersize=5, zorder=5)
ax1.errorbar(0.0, t_off_mean, yerr=t_off_std, fmt="o", color="black", ecolor="black", capsize=4, capthick=1.2, elinewidth=1.2, markersize=5, zorder=5)

ax1.annotate(rf"$T_{{hot}}$", (120.0, t_on_mean), textcoords="offset points", xytext=(-10, 10), ha="right", va="bottom", fontsize=14, fontweight="bold", color=pcb_orange)
ax1.annotate(rf"$T_{{amb.}}$", (0.0, t_off_mean), textcoords="offset points", xytext=(-10, 10), ha="right", va="bottom", fontsize=14, fontweight="bold", color=pcb_orange)

# Per-Step AgCl CP Reference Lines & Error Bars
for i, s in enumerate(steps):
    st = step_stats[s]
    s_color = step_color_map[s]
    x_off = 0

    # Reference Lines
    ax2.axhline(st["cp_on_mean"], color=s_color, linestyle="--", alpha=0.5, linewidth=0.9, zorder=3)
    ax2.axhline(st["cp_off_mean"], color=s_color, linestyle="--", alpha=0.5, linewidth=0.9, zorder=3)

    # Individual Error Bars per Step
    ax2.errorbar(
        120.0 + x_off, st["cp_on_mean"], yerr=st["cp_on_std"],
        fmt="o", color="black", ecolor="black", capsize=4, capthick=1.2, elinewidth=1.2, markersize=5, zorder=5
    )
    ax2.errorbar(
        0.0 + x_off, st["cp_off_mean"], yerr=st["cp_off_std"],
        fmt="o", color="black", ecolor="black", capsize=4, capthick=1.2, elinewidth=1.2, markersize=5, zorder=5
    )

# Summary Box Annotations for Subplot 1
ax1.text(
    0.55, 0.3,
    rf"$\Delta T_{{hot-amb.}} = {delta_temp_mean:.2f} \pm {delta_temp_std:.2f}\ ^\circ\mathrm{{C}}$",
    transform=ax1.transAxes, fontsize=15, fontweight="bold", color=pcb_orange,
    va="top", ha="center",
)

# ==========================================
# Independent Text Box Placement for Subplot 2
# ==========================================
ax2_text_positions = {
    steps[0]: (0.50, 0.78),
    steps[1]: (0.55, 0.54),
    steps[2]: (0.65, 0.38),
}

for s, st in step_stats.items():
    s_color = step_color_map[s]
    x_pos, y_pos = ax2_text_positions.get(s, (0.55, 0.45))
    
    text_str = rf"$\Delta E = {st['delta_cp_mean']:.2f} \pm {st['delta_cp_std']:.2f}\ \mathrm{{mV}}$"
    
    ax2.text(
        x_pos, y_pos,
        text_str,
        transform=ax2.transAxes, fontsize=12, fontweight="bold", color=s_color,
        va="top", ha="center",
    )

# Subplot 1 Primary Left Y-Axis Configuration (Temperature)
ax1.set_ylabel("Temperature [°C]", color=pcb_orange, fontweight="bold", fontsize=12)
ax1.tick_params(axis="y", colors=pcb_orange, labelcolor=pcb_orange)
ax1.spines["left"].set_color(pcb_orange)

ax1.set_ylim(24, 36)
ax1.set_yticks([24.0, t_off_mean, t_on_mean])
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

# Subplot 1 Overlaid Left Y-Axis Configuration (Power in Watts - 2 Lines)
ax1_pwr.set_ylabel("Heater\nPower [W]", color=power_red, fontweight="bold", fontsize=12, labelpad=15)
ax1_pwr.tick_params(axis="y", colors=power_red, labelcolor=power_red)
ax1_pwr.spines["left"].set_color(power_red)

# Scaling power axis limits and ticks in Watts [W]
ax1_power_ylim = (-4.0, 2.5)
ax1_power_yticks = [0.0, 0.5, 1.0, 1.71]

ax1_pwr.set_ylim(ax1_power_ylim)
ax1_pwr.set_yticks(ax1_power_yticks)
ax1_pwr.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

# Subplot 1 Combined Legend with Bounding Box Positioning
lines_1, labels_1 = ax1_pwr.get_legend_handles_labels()
lines_2, labels_2 = ax1.get_legend_handles_labels()

ax1.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc=ax1_legend_loc,
    bbox_to_anchor=ax1_legend_bbox,
    frameon=True,
    fontsize=10,
)

# Subplot 2 Axis Configuration & Legend with Bounding Box Positioning
ax2.set_ylabel("AgCl|AgCl Potential [mV]", color="#0284c7", fontweight="bold", fontsize=14)
ax2.tick_params(axis="y", colors="#bfdbfe", labelcolor="#0284c7")
ax2.spines["left"].set_color("#0284c7")
ax2.set_xlabel("Time [s]", fontsize=12)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

lines_3, labels_3 = ax2.get_legend_handles_labels()
labels_3 = ["0.01 [M]", "0.10 [M]", "0.60 [M]"]
ax2.legend(
    lines_3,
    labels_3,
    loc=ax2_legend_loc,
    bbox_to_anchor=ax2_legend_bbox,
    frameon=True,
    fontsize=10,
    title="Concentration",  # <-- Adds the legend title
    title_fontsize=11,       # <-- Sets legend title font size (optional)
)


plt.xlim(-60, 180)

# Set 60-second X-ticks interval [-60, 0, 60, 120, 180]
ax2.xaxis.set_major_locator(ticker.MultipleLocator(60))
ax1.tick_params(axis="x", labelbottom=True)

# Set Y-axis Limits & Formats for Subplot 2 (Cell Potential)
ax2.set_ylim(-1, 8)
ax2.set_yticks([0] + [st["cp_on_mean"] for st in step_stats.values()] + [8])

# ==============================================================================
# Subplot 1 Y-Axis Label Vertical Positioning Configuration
# ==============================================================================
# Adjust y-coordinates in normalized axes units (0.5 is centered, 1.0 is top, 0.0 is bottom)
temp_label_x, temp_label_y = -0.05, 0.25  # Temperature axis label position
pwr_label_x, pwr_label_y = -0.045, 0.79   # Heater Power axis label position

ax1.yaxis.set_label_coords(temp_label_x, temp_label_y)
ax1_pwr.yaxis.set_label_coords(pwr_label_x, pwr_label_y)

# Phase Labels above ax1
trans_inter = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
phase_labels = [
    (-30.0, "1. Baseline", "#3890fc"),
    (60.0, "2. Heating", "#fc5050"),
    (150.0, "3. Flushing", "#3890fc"),
]

for x_time, label_text, textColor in phase_labels:
    ax1.text(
        x_time,
        1.03,
        label_text,
        transform=trans_inter,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color=textColor,
        clip_on=False,
    )

plt.subplots_adjust(top=0.90, bottom=0.08, left=0.14)
plt.savefig("closed_loop_pulses_merged_by_step.png", dpi=300)
plt.show()