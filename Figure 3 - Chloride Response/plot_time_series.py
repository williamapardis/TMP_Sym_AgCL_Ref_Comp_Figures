import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
# Lookback window for step analysis: evaluated relative to step end time
# Set to an integer/float (e.g., 30.0 for the last 30 min), or None to use the full step
step_eval_window_min = 60.0

# Pulse Duration Filters (in minutes)
# Set thresholds to discard incomplete or stretched/abnormal pulses
min_pulse_duration_min = 3.5
max_pulse_duration_min = 5.0

# Figure 3 Analysis Windows (in minutes from pulse start):
window_1 = (1.7, 1.9)  # Baseline window
window_2 = (3.25, 3.75)  # Plateau window

# Toggle to enable/disable saving calculated statistics to titration_steps.json
save_stats_to_json = False

file_path = "cleaned.txt"
json_path = "titration_steps.json"

# Step selection
target_step = 6


# Temperature toggles for Figure 2 (Subplot 2):
plot_temp_setpoint = False
plot_avg_min_temp = True
plot_pid_setpoint = True
plot_avg_max_temp = True

# Manual Axis Limits Configuration (Set to None for automatic scaling):
f1_x_limits = None
f1_y_agcl_limits = None
f1_y_lj_limits = None
f1_y_vol_limits = None

f2_x_limits = None
f2_y_agcl_limits = None
f2_y_lj_limits = None
f2_y_temp_limits = None



# ==========================================
# LOAD JSON METADATA
# ==========================================
step_meta = {}
try:
    with open(json_path, "r") as f:
        meta_list = json.load(f)
        step_meta = {item["step"]: item for item in meta_list}
except FileNotFoundError:
    print(
        f"Warning: Metadata file '{json_path}' not found. Step titles will lack concentration data."
    )

# ==========================================
# DATA LOADING & PREPROCESSING
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

data_rows = []
with open(file_path, "r") as f:
    next(f)  # Skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1) if "\t" in line else line.split(",", 1)
        data_rows.append(
            [parts[0].strip()]
            + [float(x.strip()) for x in parts[1].split(",")]
        )

clean_df = pd.DataFrame(data_rows, columns=columns)
clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"])

# Filter data for target step
step_df = clean_df[clean_df["step"] == target_step].copy().reset_index(drop=True)

# Apply equilibration lookback window filter if defined
if step_eval_window_min is not None and not step_df.empty:
    step_end_time = step_df["timestamp"].max()
    window_start_time = step_end_time - pd.Timedelta(
        minutes=step_eval_window_min
    )
    step_df = (
        step_df[step_df["timestamp"] >= window_start_time]
        .copy()
        .reset_index(drop=True)
    )

if not step_df.empty:
    step_start_time = step_df["timestamp"].min()
    step_df["elapsed_min"] = (
        step_df["timestamp"] - step_start_time
    ).dt.total_seconds() / 60.0

meta_title_str = ""
if target_step in step_meta:
    conc = step_meta[target_step].get("concentration_M", "N/A")
    act = step_meta[target_step].get("activity", "N/A")
    meta_title_str = f" ([Cl⁻] = {conc} M, Activity = {act})"

color_agcl = "tab:blue"
color_lj = "tab:red"
color_vol = "tab:green"

# ==========================================
# FIGURE 1: Full Time Series (3 Y-Axes)
# ==========================================
fig1, ax1_agcl = plt.subplots(figsize=(13, 6))

ax1_agcl.set_xlabel("Timestamp")
ax1_agcl.set_ylabel("AgCl CP (mV)", color=color_agcl)
line1 = ax1_agcl.plot(
    clean_df["timestamp"],
    clean_df["AgCl CP"],
    color=color_agcl,
    label="AgCl CP",
)
ax1_agcl.tick_params(axis="y", labelcolor=color_agcl)
ax1_agcl.grid(True, linestyle="--", alpha=0.4)

ax1_lj = ax1_agcl.twinx()
ax1_lj.set_ylabel("L.J. CP (mV)", color=color_lj)
line2 = ax1_lj.plot(
    clean_df["timestamp"], clean_df["L.J. CP"], color=color_lj, label="L.J. CP"
)
ax1_lj.tick_params(axis="y", labelcolor=color_lj)

ax1_vol = ax1_agcl.twinx()
ax1_vol.spines["right"].set_position(("outward", 60))
ax1_vol.set_ylabel("Volume Dispensed (mL)", color=color_vol)
line3 = ax1_vol.plot(
    clean_df["timestamp"],
    clean_df["volume dispensed"],
    color=color_vol,
    linestyle="--",
    label="Volume Dispensed",
)
ax1_vol.tick_params(axis="y", labelcolor=color_vol)

if not step_df.empty:
    step_start = step_df["timestamp"].min()
    step_end = step_df["timestamp"].max()

    ax1_agcl.axvspan(
        step_start,
        step_end,
        color="gold",
        alpha=0.35,
        label=f"Step {target_step} Region",
    )
    ax1_agcl.text(
        step_start,
        ax1_agcl.get_ylim()[1],
        f" Step {target_step}",
        color="black",
        fontweight="bold",
        verticalalignment="bottom",
    )

handles = ax1_agcl.get_legend_handles_labels()[0]
handles2 = ax1_lj.get_legend_handles_labels()[0]
handles3 = ax1_vol.get_legend_handles_labels()[0]
labels = ax1_agcl.get_legend_handles_labels()[1]
labels2 = ax1_lj.get_legend_handles_labels()[1]
labels3 = ax1_vol.get_legend_handles_labels()[1]

ax1_agcl.legend(
    handles + handles2 + handles3,
    labels + labels2 + labels3,
    loc="upper left",
)
ax1_agcl.set_title(
    "Figure 1: Full Time Series - Cell Potential & Volume Dispensed",
    fontweight="bold",
)

if f1_x_limits:
    ax1_agcl.set_xlim(f1_x_limits)
if f1_y_agcl_limits:
    ax1_agcl.set_ylim(f1_y_agcl_limits)
if f1_y_lj_limits:
    ax1_lj.set_ylim(f1_y_lj_limits)
if f1_y_vol_limits:
    ax1_vol.set_ylim(f1_y_vol_limits)

plt.tight_layout()

# ==========================================
# FIGURE 2: Focused Continuous Step Analysis
# ==========================================
if not step_df.empty:
    fig2, (f2_ax1, f2_ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    f2_ax1.set_ylabel("AgCl CP (mV)", color=color_agcl)
    f2_line1 = f2_ax1.plot(
        step_df["elapsed_min"],
        step_df["AgCl CP"],
        color=color_agcl,
        label="AgCl CP",
    )
    f2_ax1.tick_params(axis="y", labelcolor=color_agcl)
    f2_ax1.grid(True, linestyle="--", alpha=0.5)

    f2_ax2 = f2_ax1.twinx()
    f2_ax2.set_ylabel("L.J. CP (mV)", color=color_lj)
    f2_line2 = f2_ax2.plot(
        step_df["elapsed_min"],
        step_df["L.J. CP"],
        color=color_lj,
        label="L.J. CP",
    )
    f2_ax2.tick_params(axis="y", labelcolor=color_lj)

    f2_lines = f2_line1 + f2_line2
    f2_labels = [l.get_label() for l in f2_lines]
    f2_ax1.legend(f2_lines, f2_labels, loc="upper left")
    f2_ax1.set_title(
        f"Figure 2: Step {target_step} Cell Potential Focus{meta_title_str}",
        fontweight="bold",
    )

    f2_ax3.plot(
        step_df["elapsed_min"],
        step_df["temperature"],
        label="Temperature",
        color="black",
        linewidth=1.5,
    )

    if plot_temp_setpoint:
        f2_ax3.plot(
            step_df["elapsed_min"],
            step_df["temperature setpoint"],
            label="Temp Setpoint",
            linestyle="--",
            color="blue",
        )
    if plot_avg_min_temp:
        f2_ax3.plot(
            step_df["elapsed_min"],
            step_df["avg min temp"],
            label="Avg Min Temp",
            linestyle=":",
            color="cyan",
        )
    if plot_pid_setpoint:
        f2_ax3.plot(
            step_df["elapsed_min"],
            step_df["PID setpoint"],
            label="PID Setpoint",
            linestyle="--",
            color="orange",
        )
    if plot_avg_max_temp:
        f2_ax3.plot(
            step_df["elapsed_min"],
            step_df["avg max temp"],
            label="Avg Max Temp",
            linestyle=":",
            color="red",
        )

    f2_ax3.set_title(
        f"Step {target_step}: Temperature Profile", fontweight="bold"
    )
    f2_ax3.set_xlabel("Elapsed Time (minutes)")
    f2_ax3.set_ylabel("Temperature (°C)")
    f2_ax3.legend(loc="upper left")
    f2_ax3.grid(True, linestyle="--", alpha=0.5)

    if f2_x_limits:
        f2_ax1.set_xlim(f2_x_limits)
    if f2_y_agcl_limits:
        f2_ax1.set_ylim(f2_y_agcl_limits)
    if f2_y_lj_limits:
        f2_ax2.set_ylim(f2_y_lj_limits)
    if f2_y_temp_limits:
        f2_ax3.set_ylim(f2_y_temp_limits)

    plt.tight_layout()

# ==========================================
# FIGURE 3: Categorized Overlay Pulse Analysis (Equilibrated Window Sliced)
# ==========================================
if not step_df.empty:
    # ----------------------------------------------------
    # SLICE DF TO EQUILIBRATED WINDOW ONLY
    # ----------------------------------------------------
    if step_eval_window_min is not None:
        step_end_time = step_df["timestamp"].max()
        window_start_time = step_end_time - pd.Timedelta(
            minutes=step_eval_window_min
        )
        step_df = (
            step_df[step_df["timestamp"] >= window_start_time]
            .copy()
            .reset_index(drop=True)
        )

    # Detect falling edges on the trimmed step segment
    step_df["pid_change"] = step_df["PID enable"].astype(int).diff()
    falling_edges = step_df.index[step_df["pid_change"] == -1].tolist()

    if 0 not in falling_edges:
        falling_edges.insert(0, 0)
    if len(step_df) - 1 not in falling_edges:
        falling_edges.append(len(step_df) - 1)

    pulses = {"Small (~1°C)": [], "Medium (~5°C)": [], "Large (~10°C)": []}
    discarded_count = 0

    for i in range(len(falling_edges) - 1):
        start_idx = falling_edges[i]
        end_idx = falling_edges[i + 1]

        pulse_data = step_df.iloc[start_idx:end_idx].copy()
        active_heating = pulse_data[pulse_data["PID enable"] == 1]

        if len(active_heating) < 3:
            continue

        t0 = pulse_data["timestamp"].min()
        pulse_data["pulse_time_min"] = (
            pulse_data["timestamp"] - t0
        ).dt.total_seconds() / 60.0
        total_duration = pulse_data["pulse_time_min"].max()

        if (
            total_duration < min_pulse_duration_min
            or total_duration > max_pulse_duration_min
        ):
            discarded_count += 1
            continue

        min_off_temp = pulse_data["temperature"].min()
        max_on_temp = active_heating["temperature"].max()
        delta_t = max_on_temp - min_off_temp

        if delta_t < 3.0:
            pulses["Small (~1°C)"].append(pulse_data)
        elif 3.0 <= delta_t < 8.0:
            pulses["Medium (~5°C)"].append(pulse_data)
        else:
            pulses["Large (~10°C)"].append(pulse_data)

    fig3, axes3 = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    categories = ["Small (~1°C)", "Medium (~5°C)", "Large (~10°C)"]
    cat_colors = ["tab:blue", "tab:orange", "tab:green"]

    exported_stats = {}

    for ax, cat_name, color in zip(axes3, categories, cat_colors):
        cat_pulses = pulses[cat_name]
        ax_temp = ax.twinx()

        for idx, pulse in enumerate(cat_pulses):
            ax.plot(
                pulse["pulse_time_min"],
                pulse["AgCl CP"],
                color=color,
                alpha=0.6,
                linewidth=1.2,
                label="AgCl CP" if idx == 0 else "",
            )
            ax_temp.plot(
                pulse["pulse_time_min"],
                pulse["temperature"],
                color="black",
                linestyle="--",
                alpha=0.3,
                label="Temperature (°C)" if idx == 0 else "",
            )

        # ----------------------------------------------------
        # STATS COMPUTATION
        # ----------------------------------------------------
        w1_samples, w2_samples = [], []
        delta_T_list = []
        delta_CP_list = []

        for pulse in cat_pulses:
            w1_mask = (pulse["pulse_time_min"] >= window_1[0]) & (
                pulse["pulse_time_min"] <= window_1[1]
            )
            w1_cp_list = pulse.loc[w1_mask, "AgCl CP"].tolist()
            w1_cp_mean = pulse.loc[w1_mask, "AgCl CP"].mean()
            w1_temp_mean = pulse.loc[w1_mask, "temperature"].mean()
            w1_samples.extend(w1_cp_list)

            w2_mask = (pulse["pulse_time_min"] >= window_2[0]) & (
                pulse["pulse_time_min"] <= window_2[1]
            )
            w2_cp_list = pulse.loc[w2_mask, "AgCl CP"].tolist()
            w2_cp_mean = pulse.loc[w2_mask, "AgCl CP"].mean()
            w2_temp_mean = pulse.loc[w2_mask, "temperature"].mean()
            w2_samples.extend(w2_cp_list)

            if not np.isnan(w1_temp_mean) and not np.isnan(w2_temp_mean):
                delta_T_list.append(w2_temp_mean - w1_temp_mean)
            if not np.isnan(w1_cp_mean) and not np.isnan(w2_cp_mean):
                delta_CP_list.append(w2_cp_mean - w1_cp_mean)

        w1_mean = (
            float(np.mean(w1_samples)) if len(w1_samples) > 0 else None
        )
        w1_std = float(np.std(w1_samples)) if len(w1_samples) > 0 else None

        w2_mean = (
            float(np.mean(w2_samples)) if len(w2_samples) > 0 else None
        )
        w2_std = float(np.std(w2_samples)) if len(w2_samples) > 0 else None

        delta_T_mean = (
            float(np.mean(delta_T_list)) if len(delta_T_list) > 0 else None
        )
        delta_T_std = (
            float(np.std(delta_T_list)) if len(delta_T_list) > 0 else None
        )

        delta_CP_mean = (
            float(np.mean(delta_CP_list)) if len(delta_CP_list) > 0 else None
        )
        delta_CP_std = (
            float(np.std(delta_CP_list)) if len(delta_CP_list) > 0 else None
        )

        exported_stats[cat_name] = {
            "count": len(cat_pulses),
            "w1_agcl_cp_mean_mV": w1_mean,
            "w1_agcl_cp_std_mV": w1_std,
            "w2_agcl_cp_mean_mV": w2_mean,
            "w2_agcl_cp_std_mV": w2_std,
            "delta_T_mean_C": delta_T_mean,
            "delta_T_std_C": delta_T_std,
            "delta_CP_mean_mV": delta_CP_mean,
            "delta_CP_std_mV": delta_CP_std,
        }

        # Visual Overlays
        ax.axvspan(
            window_1[0], window_1[1], color="purple", alpha=0.12, label="W1"
        )
        ax.axvspan(
            window_2[0], window_2[1], color="crimson", alpha=0.12, label="W2"
        )

        if w1_mean is not None:
            ax.hlines(
                w1_mean,
                window_1[0],
                window_1[1],
                colors="purple",
                linestyles="--",
                linewidth=2.0,
            )
            ax.fill_between(
                [window_1[0], window_1[1]],
                [w1_mean - w1_std, w1_mean - w1_std],
                [w1_mean + w1_std, w1_mean + w1_std],
                color="purple",
                alpha=0.2,
            )

        if w2_mean is not None:
            ax.hlines(
                w2_mean,
                window_2[0],
                window_2[1],
                colors="crimson",
                linestyles="--",
                linewidth=2.0,
            )
            ax.fill_between(
                [window_2[0], window_2[1]],
                [w2_mean - w2_std, w2_mean - w2_std],
                [w2_mean + w2_std, w2_mean + w2_std],
                color="crimson",
                alpha=0.2,
            )

        w1_str = (
            f"{w1_mean:.2f} ± {w1_std:.2f}" if w1_mean is not None else "N/A"
        )
        w2_str = (
            f"{w2_mean:.2f} ± {w2_std:.2f}" if w2_mean is not None else "N/A"
        )
        dt_str = (
            f"{delta_T_mean:.2f} ± {delta_T_std:.2f}"
            if delta_T_mean is not None
            else "N/A"
        )
        dcp_str = (
            f"{delta_CP_mean:.2f} ± {delta_CP_std:.2f}"
            if delta_CP_mean is not None
            else "N/A"
        )

        stats_text = (
            f"W1 ({window_1[0]}-{window_1[1]}m): {w1_str} mV\n"
            f"W2 ({window_2[0]}-{window_2[1]}m): {w2_str} mV\n"
            f"ΔT (W2 - W1): {dt_str} °C\n"
            f"ΔCP (W2 - W1): {dcp_str} mV"
        )
        ax.text(
            0.02,
            0.06,
            stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.85,
                edgecolor="gray",
            ),
        )

        ax.set_title(
            f"{cat_name} Pulses — Count: {len(cat_pulses)}", fontweight="bold"
        )
        ax.set_ylabel("AgCl CP (mV)", color=color)
        ax_temp.set_ylabel("Temp (°C)", color="black")
        ax.grid(True, linestyle="--", alpha=0.5)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_temp.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    axes3[-1].set_xlabel("Time from Pulse Start (minutes)")

    eval_str = (
        f" (Last {step_eval_window_min} min)"
        if step_eval_window_min is not None
        else ""
    )
    fig3.suptitle(
        f"Figure 3: Step {target_step} Categorized Pulse Analysis{eval_str}{meta_title_str}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    # ----------------------------------------------------
    # OPTIONAL JSON SAVE LOGIC
    # ----------------------------------------------------
    if save_stats_to_json:
        try:
            try:
                with open(json_path, "r") as f:
                    json_data = json.load(f)
            except FileNotFoundError:
                json_data = []

            step_entry = None
            for entry in json_data:
                if entry.get("step") == target_step:
                    step_entry = entry
                    break

            if step_entry is None:
                step_entry = {"step": target_step}
                json_data.append(step_entry)

            step_entry["pulse_analysis"] = {
                "eval_window_min": step_eval_window_min,
                "window_1_min": list(window_1),
                "window_2_min": list(window_2),
                "categories": exported_stats,
            }

            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=4)

            print(
                f"Successfully saved pulse stats for Step {target_step} to '{json_path}'"
            )

        except Exception as e:
            print(f"Error saving statistics to JSON: {e}")

plt.show()