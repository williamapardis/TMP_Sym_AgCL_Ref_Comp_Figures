import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ==========================================
# CONFIGURATION
# ==========================================
file_path = "cleaned.txt"
json_path = "titration_steps.json"

# Invert signal sign to correct for experimental setup polarity
INVERT_LJ_CP_SIGN = True

# Step Selection Toggle:
# - Range tuple/list [start, end]: e.g., (0, 7) or [0, 7] (inclusive)
# - Specific list: e.g., [0, 1, 3, 5, 7]
# - None: include all steps
STEPS_TO_INCLUDE = (0, 7)

# Toggle x-axis metric for Subplot 2: "activity" or "concentration"
X_METRIC = "activity"

# Concentration values [M] for Steps 0 through 10 (11 steps total)
CONCENTRATIONS = [
    0.0100,
    0.0168,
    0.0301,
    0.0498,
    0.1000,
    0.2025,
    0.3021,
    0.4015,
    0.4982,
    0.5992,
    0.6992,
]

# Chloride activity values corresponding to Steps 0 through 10
ACTIVITIES = [
    0.00902,
    0.01478,
    0.02559,
    0.04074,
    0.07780,
    0.14884,
    0.21449,
    0.27824,
    0.33927,
    0.40266,
    0.46497,
]

# Theoretical Nernstian Slope at 25 °C for Cl- (monovalent anion: z = -1)
# Slope = (2.3026 * R * T) / (z * F) = -59.16 mV / log10(a_Cl-)
NERNST_SLOPE_25C = -59.16

# Time window for averaging (in minutes)
AVG_WINDOW_START = 45
AVG_WINDOW_END = 85

# Axis limits for Subplot 1 (Set to None for automatic scaling)
X_LIMITS = (0, 90)
Y_LIMITS = None

# Axis limits for Subplot 2 (Set to None for automatic scaling)
SUBPLOT2_X_LIMITS = None  # e.g., (-2.2, -0.2)
SUBPLOT2_Y_LIMITS = None  # e.g., (-250, -50)

# Toggle Residuals Bar Chart Inset on Subplot 2
SHOW_RESIDUAL_INSET = True

# ==========================================
# DATA PROCESSING
# ==========================================
columns = [
    "timestamp",
    "step",
    "volume dispensed",
    "temperature setpoint",
    "avg min temp",
    "PID setpoint",
    "avg max temp",
    "temperature",
    "PID enable",
    "current",
    "time to next",
    "time on",
    "time off",
    "AgCl CP",
    "L.J. CP",
    "avg min CP",
    "avg max CP",
    "TRA",
    "averages",
]

df = pd.read_csv(file_path, names=columns, header=0, skipinitialspace=True)

df["step"] = pd.to_numeric(df["step"], errors="coerce")
df["L.J. CP"] = pd.to_numeric(df["L.J. CP"], errors="coerce")

# Invert L.J. CP sign if enabled
if INVERT_LJ_CP_SIGN:
    df["L.J. CP"] = -1.0 * df["L.J. CP"]

df = df.dropna(subset=["timestamp", "step", "L.J. CP"])

df["timestamp"] = (
    df["timestamp"]
    .astype(str)
    .str.extract(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")[0]
)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
df = df.sort_values("timestamp")
df["step"] = df["step"].astype(int)

# --- APPLY STEP FILTERING ---
if STEPS_TO_INCLUDE is not None:
    if isinstance(STEPS_TO_INCLUDE, (tuple, list)) and len(STEPS_TO_INCLUDE) == 2 and not isinstance(STEPS_TO_INCLUDE[0], list):
        start_step, end_step = STEPS_TO_INCLUDE
        df = df[(df["step"] >= start_step) & (df["step"] <= end_step)]
    elif isinstance(STEPS_TO_INCLUDE, (list, tuple, set)):
        df = df[df["step"].isin(STEPS_TO_INCLUDE)]

df["elapsed_time_step_min"] = df.groupby("step")["timestamp"].transform(
    lambda x: (x - x.min()).dt.total_seconds() / 60.0
)

# ==========================================
# PLOTTING (1 Row, 2 Columns)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- SUBPLOT 1: Raw Curves & Window Averages ---
ax1.axvspan(
    AVG_WINDOW_START,
    AVG_WINDOW_END,
    color="gray",
    alpha=0.15,
    label="Averaging Window",
)

step_averages = {}
step_std_devs = {}

for step_id, group in df.groupby("step"):
    lines = ax1.plot(
        group["elapsed_time_step_min"],
        group["L.J. CP"],
        marker="o",
        markersize=3,
        linestyle="-",
        label=f"Step {step_id}",
    )
    step_color = lines[0].get_color()

    window_mask = (group["elapsed_time_step_min"] >= AVG_WINDOW_START) & (
        group["elapsed_time_step_min"] <= AVG_WINDOW_END
    )
    window_data = group[window_mask]

    if not window_data.empty:
        mean_val = window_data["L.J. CP"].mean()
        std_val = window_data["L.J. CP"].std()

        step_averages[step_id] = mean_val
        step_std_devs[step_id] = std_val if not np.isnan(std_val) else 0.0

        # Plot dashed average line across window
        ax1.hlines(
            y=mean_val,
            xmin=AVG_WINDOW_START,
            xmax=AVG_WINDOW_END,
            colors=step_color,
            linestyles="--",
            linewidth=1.5,
        )
        
        # Plot crimson point with error bar at center of window
        window_center = (AVG_WINDOW_START + AVG_WINDOW_END) / 2.0
        ax1.errorbar(
            window_center,
            mean_val,
            yerr=std_val,
            fmt="o",
            color="crimson",
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            capthick=1.2,
            markersize=6,
            zorder=4,
        )

if X_LIMITS is not None:
    ax1.set_xlim(X_LIMITS)
if Y_LIMITS is not None:
    ax1.set_ylim(Y_LIMITS)

ax1.set_title("L.J. CP vs. Elapsed Time per Step", fontsize=12, fontweight="bold")
ax1.set_xlabel("Elapsed Time per Step (minutes)", fontsize=11)
ax1.set_ylabel("L.J. CP (mV)", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(title="Step", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

# --- SUBPLOT 2: Average vs. Log10(Metric) with Fit, Ideal Line & Residuals ---
steps_sorted = sorted(step_averages.keys())

# Select values based on the X_METRIC toggle
if X_METRIC.lower() == "activity":
    metric_values = ACTIVITIES
    x_label_str = r"$\log_{10}(a_{\text{Cl}^-})$"
    title_metric_str = "Activity"
else:
    metric_values = CONCENTRATIONS
    x_label_str = r"$\log_{10}([\text{Cl}^-])$"
    title_metric_str = "Concentration"

log_x = []
avg_cp = []
std_cp = []

for step_id in steps_sorted:
    idx = step_id  # 0-indexed step mapping
    if 0 <= idx < len(metric_values):
        val = metric_values[idx]
        log_x.append(np.log10(val))
        avg_cp.append(step_averages[step_id])
        std_cp.append(step_std_devs[step_id])

log_x = np.array(log_x)
avg_cp = np.array(avg_cp)
std_cp = np.array(std_cp)

if len(log_x) > 2:
    n = len(log_x)
    # Perform linear regression using scipy
    res = stats.linregress(log_x, avg_cp)
    slope = res.slope
    intercept = res.intercept
    r_squared = res.rvalue ** 2

    # Linear fit predictions & residuals
    y_pred_fit = slope * log_x + intercept
    residuals_fit = avg_cp - y_pred_fit
    s_err = np.sqrt(np.sum(residuals_fit**2) / (n - 2))

    # Ideal Nernst response predictions & residuals (anchored to centroid of active subset)
    x_mean = np.mean(log_x)
    y_mean = np.mean(avg_cp)
    ideal_intercept = y_mean - (NERNST_SLOPE_25C * x_mean)
    y_pred_ideal = NERNST_SLOPE_25C * log_x + ideal_intercept
    residuals_ideal = avg_cp - y_pred_ideal

    # Plot data points with measurement error bars
    ax2.errorbar(
        log_x,
        avg_cp,
        yerr=std_cp,
        fmt="o",
        color="crimson",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        markersize=6,
        label="Step Averages (±1 SD)",
        zorder=3,
    )

    # Line domain range
    x_fit = np.linspace(log_x.min(), log_x.max(), 100)
    y_fit = slope * x_fit + intercept
    y_ideal = NERNST_SLOPE_25C * x_fit + ideal_intercept

    # 1. Plot Linear Fit Line (Crimson with reduced opacity)
    ax2.plot(
        x_fit,
        y_fit,
        color="crimson",
        alpha=0.6,
        linestyle="--",
        linewidth=1.3,
        label=f"Fit: y = {slope:.2f}x + {intercept:.2f}\n$R^2 = {r_squared:.4f}$, RMSE = {s_err:.3f}",
        zorder=2,
    )

    # 2. Plot Ideal Nernst Response Line (-59.16 mV/dec at 25 °C anchored to data centroid)
    ax2.plot(
        x_fit,
        y_ideal,
        color="black",
        linestyle="-",
        linewidth=1.8,
        label=f"Ideal Nernst 25°C ({NERNST_SLOPE_25C:.2f} mV/dec)",
        zorder=2,
    )

    # Legend set to Upper Right
    ax2.legend(loc="upper right", fontsize="small")

    # Inset Plot for Residuals set to Lower Left ([left, bottom, width, height])
    if SHOW_RESIDUAL_INSET:
        ax_inset = ax2.inset_axes([0.08, 0.12, 0.38, 0.28])
        ax_inset.axhline(0, color="black", linestyle="-", linewidth=0.8, zorder=1)

        # Calculate bar width and offsets for side-by-side grouped bars
        bar_width = (log_x.max() - log_x.min()) / (len(log_x) * 3.5)
        offset = bar_width / 2.0

        # Experimental Fit Residuals (Crimson)
        ax_inset.bar(
            log_x - offset,
            residuals_fit,
            width=bar_width,
            yerr=std_cp,
            color="crimson",
            alpha=0.7,
            edgecolor="darkred",
            ecolor="black",
            capsize=2,
            align="center",
            label="vs Fit",
            zorder=2,
        )

        # Ideal Nernst Residuals (Charcoal/Black)
        ax_inset.bar(
            log_x + offset,
            residuals_ideal,
            width=bar_width,
            yerr=std_cp,
            color="dimgray",
            alpha=0.85,
            edgecolor="black",
            ecolor="black",
            capsize=2,
            align="center",
            label="vs Ideal",
            zorder=2,
        )

        ax_inset.set_title("Residuals (±1 SD)", fontsize=8)
        ax_inset.set_xlabel(x_label_str, fontsize=7)
        ax_inset.set_ylabel(r"$\Delta y$ (mV)", fontsize=7)
        ax_inset.tick_params(labelsize=6)
        ax_inset.grid(True, linestyle=":", alpha=0.5)
        ax_inset.legend(loc="best", fontsize=6, framealpha=0.8)

# Set Subplot 2 custom axis limits if defined
if SUBPLOT2_X_LIMITS is not None:
    ax2.set_xlim(SUBPLOT2_X_LIMITS)
if SUBPLOT2_Y_LIMITS is not None:
    ax2.set_ylim(SUBPLOT2_Y_LIMITS)

ax2.set_title(
    f"L.J. CP Average vs. {x_label_str} ({title_metric_str})",
    fontsize=12,
    fontweight="bold",
)
ax2.set_xlabel(x_label_str, fontsize=11)
ax2.set_ylabel("Average L.J. CP (mV)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# ==========================================
# JSON UPDATE / EXPORT
# ==========================================
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
else:
    json_data = []

# Map list to dict by step number for easy updating
json_steps_map = {item.get("step"): item for item in json_data}

for step_id, mean_val in step_averages.items():
    std_val = step_std_devs.get(step_id, 0.0)

    if step_id not in json_steps_map:
        json_steps_map[step_id] = {"step": step_id}

    json_steps_map[step_id]["lj_cp_mean_mV"] = float(mean_val)
    json_steps_map[step_id]["lj_cp_std_mV"] = float(std_val)

# Convert back to a step-sorted list and write out
updated_json_data = [
    json_steps_map[k] for k in sorted(json_steps_map.keys())
]

with open(json_path, "w") as f:
    json.dump(updated_json_data, f, indent=4)

print(f"Successfully updated L.J. CP statistics in '{json_path}'.")