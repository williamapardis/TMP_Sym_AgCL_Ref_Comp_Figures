import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

with open("all_window_pulse_metrics.json", "r", encoding="utf-8") as f:
  data = json.load(f)

targets = [1.0, 5.0, 10.0]
windows = ["1", "2", "3"]
colors = {"1": "#1f77b4", "2": "#ff7f0e", "3": "#2ca02c"}
markers = {"1": "o", "2": "s", "3": "^"}
offsets = {"1": -0.25, "2": 0.0, "3": 0.25}

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(
    1, 3, figsize=(13.5, 5.3), dpi=120, constrained_layout=True
)

for ax, target in zip(axes, targets):
  for w in windows:
    rows = [
        r
        for r in data[w]["condition_summaries"]
        if r["delta_temp_target_c"] == target
    ]
    rows.sort(key=lambda r: r["ambient_temp_setpoint_c"])

    x = np.array([r["ambient_temp_setpoint_c"] for r in rows])
    y = np.array([r["delta_e_mean_mv"] for r in rows])
    yerr = np.array([r["delta_e_std_mv"] for r in rows])

    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    r2 = r_value**2

    sign = "+" if intercept >= 0 else "-"
    eq_label = (
        f"Win {w}: y = {slope:.4f}x {sign} {abs(intercept):.3f} ($R^2$={r2:.3f})"
    )

    # Error bars at data points (±1 SD)
    ax.errorbar(
        x + offsets[w],
        y,
        yerr=yerr,
        fmt=markers[w],
        color=colors[w],
        markersize=6.5,
        capsize=4,
        capthick=1.2,
        elinewidth=1.2,
        alpha=0.95,
        zorder=4,
    )

    # Linear fit line + interpolated ±1 SD error band
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    sd_band = np.interp(x_line, x, yerr)

    ax.fill_between(
        x_line,
        y_line - sd_band,
        y_line + sd_band,
        color=colors[w],
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )

    ax.plot(
        x_line,
        y_line,
        linestyle="--",
        linewidth=1.6,
        color=colors[w],
        label=eq_label,
        zorder=3,
    )

  ax.set_title(
      f"$\\Delta T_{{target}} = {int(target)}\\ ^\\circ\\mathrm{{C}}$",
      fontsize=12,
      fontweight="bold",
      pad=8,
  )
  ax.set_xlabel(
      "Ambient Temp ($^\\circ\\mathrm{C}$)", fontsize=10, fontweight="bold"
  )
  ax.set_xticks([15, 25, 35])
  ax.set_xlim(12, 38)
  ax.tick_params(labelsize=9.5)
  ax.grid(True, linestyle=":", alpha=0.6)

  ax.legend(
      loc="upper center",
      bbox_to_anchor=(0.5, -0.16),
      ncol=1,
      fontsize=8.5,
      frameon=True,
      facecolor="white",
      edgecolor="lightgray",
  )

axes[0].set_ylabel(
    "Mean $\\Delta E$ (mV) $\\pm$ SD", fontsize=10.5, fontweight="bold"
)

fig.suptitle(
    "Mean $\\Delta E$ vs. Ambient Temperature Setpoint Across Windows (Shaded"
    " Band = $\\pm 1$ SD)",
    fontsize=13,
    fontweight="bold",
)

plt.savefig("delta_e_with_error_bands.png", dpi=300, bbox_inches="tight")
plt.show()