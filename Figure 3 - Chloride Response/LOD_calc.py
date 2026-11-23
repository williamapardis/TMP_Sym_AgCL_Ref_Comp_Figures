import math
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

# Constants
R = 8.314462618     # J/(mol·K)
F = 96485.33212     # C/mol
T_AMBIENT_C = 25.0  # Baseline ambient temperature (°C)
DAVIES_B = 0.30     # Empirical Davies b parameter (L/mol)

# Each row: (log10(activity), voltage SD in mV, measured ΔT in °C)
data = {
    "1°C": [
        (-2.0450, 0.0602, 1.11),
        (-1.8308, 0.0636, 1.10),
        (-1.5924, 0.0724, 1.09),
        (-1.3884, 0.0651, 1.08),
        (-1.1075, 0.0396, 1.08),
        (-0.8212, 0.0124, 1.09),
        (-0.6549, 0.0288, 1.06),
        (-0.5331, 0.0668, 1.07),
    ],
    "5°C": [
        (-2.0450, 0.1338, 5.06),
        (-1.8308, 0.0987, 5.05),
        (-1.5924, 0.0997, 5.06),
        (-1.3884, 0.0667, 5.06),
        (-1.1075, 0.0134, 5.06),
        (-0.8212, 0.0071, 5.06),
        (-0.6549, 0.0114, 5.06),
        (-0.5331, 0.0189, 5.07),
    ],
    "10°C": [
        (-2.0450, 0.0959, 10.03),
        (-1.8308, 0.1233, 10.04),
        (-1.5924, 0.0588, 10.04),
        (-1.3884, 0.0381, 10.04),
        (-1.1075, 0.0263, 10.05),
        (-0.8212, 0.0175, 10.05),
        (-0.6549, 0.0623, 10.05),
        (-0.5331, 0.0355, 10.04),
    ],
}

# Measured regression slopes in mV per log10(activity) unit.
calibration_slopes = {
    "1°C": -0.546,
    "5°C": -2.198,
    "10°C": -3.814,
}


def debye_huckel_A(t_c=T_AMBIENT_C):
    """Temperature-dependent Debye-Hückel A(T) via Malmberg-Maryott permittivity."""
    t_k = t_c + 273.15
    eps_r = (
        87.740
        - 0.40008 * t_c
        + 9.398e-4 * (t_c ** 2)
        - 1.410e-6 * (t_c ** 3)
    )
    return 1.824829e6 * ((eps_r * t_k) ** -1.5)


def davies_log_activity(conc_m, t_c=T_AMBIENT_C, b=DAVIES_B):
    """Forward Davies calculation: molar concentration (M) -> log10(activity)."""
    ionic_strength = conc_m  # 1:1 electrolyte: I = c_Cl-
    sqrt_i = math.sqrt(ionic_strength)
    A = debye_huckel_A(t_c)
    log_gamma = -A * (sqrt_i / (1.0 + sqrt_i) - b * ionic_strength)
    return log_gamma + math.log10(conc_m)


def conc_from_log_activity(target_log_a, t_c=T_AMBIENT_C, b=DAVIES_B):
    """Inverts the Davies equation via bisection to recover exact molarity (M)."""
    low, high = 1e-7, 1.5
    for _ in range(80):
        mid = 0.5 * (low + high)
        if davies_log_activity(mid, t_c, b) < target_log_a:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def theoretical_slope_mv_per_decade(delta_t_c):
    """Magnitude of (R ΔT / F) ln(10), in mV per decade."""
    return (R * abs(delta_t_c) / F) * math.log(10.0) * 1000.0


def add_horizontal_trace_toggle(fig, series_artists, rescale_callback, bar_width=0.44, bar_bottom=0.92, bar_height=0.045):
    """Adds a slim horizontal CheckButtons bar centered at the top of a figure."""
    series_keys = list(series_artists.keys())
    labels = [f"ΔT ≈ {s}" for s in series_keys]
    n_items = len(series_keys)

    bar_left = 0.50 - (bar_width / 2.0)
    ax_bar_bg = fig.add_axes([bar_left, bar_bottom, bar_width, bar_height])
    ax_bar_bg.set_xticks([])
    ax_bar_bg.set_yticks([])
    ax_bar_bg.patch.set_facecolor("#f7f7f7")
    for spine in ax_bar_bg.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)

    slot_width = bar_width / n_items
    widgets = []

    for idx, (s_key, lbl) in enumerate(zip(series_keys, labels)):
        ax_slot = fig.add_axes([
            bar_left + idx * slot_width + 0.01,
            bar_bottom + 0.004,
            slot_width - 0.02,
            bar_height - 0.008,
        ])
        ax_slot.set_frame_on(False)
        cb = CheckButtons(ax_slot, [lbl], [True])
        cb.labels[0].set_color(series_artists[s_key][0].get_color())
        cb.labels[0].set_fontsize(10)
        cb.labels[0].set_fontweight("bold")

        def make_callback(key):
            def _toggle(_label):
                for artist in series_artists[key]:
                    artist.set_visible(not artist.get_visible())
                rescale_callback()
                fig.canvas.draw_idle()
            return _toggle

        cb.on_clicked(make_callback(s_key))
        widgets.append(cb)

    return widgets


# ==============================================================================
# FIGURE A (Representation #2): Signal-to-Noise Ratio per Decade (|slope| / σ)
# ==============================================================================
fig_snr, axes_snr = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)
snr_artists = {series: [] for series in data.keys()}


def plot_snr_column(ax, method):
    for series, rows in data.items():
        x_conc_mmol = []
        y_snr = []

        for log_a, sd_mv, delta_t_c in rows:
            slope = (
                theoretical_slope_mv_per_decade(delta_t_c)
                if method == "theoretical"
                else abs(calibration_slopes[series])
            )
            c_mmol = conc_from_log_activity(log_a) * 1000.0
            snr_per_decade = slope / sd_mv  # Signal per decade (mV/dec) divided by noise (mV)

            x_conc_mmol.append(c_mmol)
            y_snr.append(snr_per_decade)

        (line,) = ax.plot(x_conc_mmol, y_snr, marker="o", label=f"ΔT ≈ {series}")
        snr_artists[series].append(line)

    ax.set_xlabel("Chloride Concentration $c_{\\mathrm{Cl}^-}$ (mmol/L)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


plot_snr_column(axes_snr[0], "theoretical")
plot_snr_column(axes_snr[1], "calibration")

axes_snr[0].set_title("Theoretical slope: (R ΔT / F) ln(10)")
axes_snr[1].set_title("Measured calibration slopes")

# Place shared y-axis ticks on inner spines and center the y-axis label at x = 0.50
axes_snr[0].yaxis.tick_right()
axes_snr[0].tick_params(axis="y", which="both", labelright=True, labelleft=False)
axes_snr[1].yaxis.tick_left()
axes_snr[1].tick_params(axis="y", which="both", labelleft=True, labelright=False)

fig_snr.subplots_adjust(left=0.05, right=0.95, top=0.82, bottom=0.13, wspace=0.28)
snr_center_y = 0.5 * (axes_snr[0].get_position().y0 + axes_snr[0].get_position().y1)
fig_snr.text(
    0.50,
    snr_center_y,
    "Signal-to-Noise Ratio per Decade  |slope| / σ",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=10,
)


def rescale_snr():
    visible_y = []
    for lines in snr_artists.values():
        if lines[0].get_visible():
            visible_y.extend(lines[0].get_ydata())
    if visible_y:
        axes_snr[0].set_ylim(min(visible_y) * 0.75, max(visible_y) * 1.35)


snr_widgets = add_horizontal_trace_toggle(
    fig_snr, snr_artists, rescale_snr, bar_width=0.44, bar_bottom=0.90, bar_height=0.06
)


# ==============================================================================
# FIGURE B (Representation #3): Calibration Response with ±3σ Resolution Envelopes
# ==============================================================================
fig_cal, axes_cal = plt.subplots(1, 2, figsize=(15, 6.0), sharex=True, sharey=True)
cal_artists = {series: [] for series in data.keys()}

# Reference anchor at lowest concentration (log10(a) = -2.0450 -> ΔE_ref = 0 mV)
LOG_A_REF = -2.0450


def plot_calibration_envelope(ax, method):
    for series, rows in data.items():
        x_conc_mmol = []
        y_de_mv = []
        y_upper_3sd = []
        y_lower_3sd = []

        for log_a, sd_mv, delta_t_c in rows:
            # Signed slope: negative for Cl- (higher [Cl-] -> more negative potential)
            slope_signed = (
                -theoretical_slope_mv_per_decade(delta_t_c)
                if method == "theoretical"
                else calibration_slopes[series]
            )
            c_mmol = conc_from_log_activity(log_a) * 1000.0
            delta_e = slope_signed * (log_a - LOG_A_REF)

            x_conc_mmol.append(c_mmol)
            y_de_mv.append(delta_e)
            y_upper_3sd.append(delta_e + 3.0 * sd_mv)
            y_lower_3sd.append(delta_e - 3.0 * sd_mv)

        (line,) = ax.plot(x_conc_mmol, y_de_mv, marker="o", linewidth=1.8, label=f"ΔT ≈ {series}")
        band = ax.fill_between(
            x_conc_mmol,
            y_lower_3sd,
            y_upper_3sd,
            color=line.get_color(),
            alpha=0.22,
        )
        # Also plot thin dashed ±3σ boundary lines so even tiny bands (e.g., 7 µV) remain visible
        (bound_hi,) = ax.plot(x_conc_mmol, y_upper_3sd, linestyle="--", linewidth=0.7, color=line.get_color(), alpha=0.6)
        (bound_lo,) = ax.plot(x_conc_mmol, y_lower_3sd, linestyle="--", linewidth=0.7, color=line.get_color(), alpha=0.6)

        cal_artists[series].extend([line, band, bound_hi, bound_lo])

    ax.set_xlabel("Chloride Concentration $c_{\\mathrm{Cl}^-}$ (mmol/L)")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)


plot_calibration_envelope(axes_cal[0], "theoretical")
plot_calibration_envelope(axes_cal[1], "calibration")

axes_cal[0].set_title("Theoretical Nernstian Response ± 3σ Band")
axes_cal[1].set_title("Measured Calibration Response ± 3σ Band")

axes_cal[0].yaxis.tick_right()
axes_cal[0].tick_params(axis="y", which="both", labelright=True, labelleft=False)
axes_cal[1].yaxis.tick_left()
axes_cal[1].tick_params(axis="y", which="both", labelleft=True, labelright=False)

fig_cal.subplots_adjust(left=0.05, right=0.95, top=0.82, bottom=0.13, wspace=0.28)
cal_center_y = 0.5 * (axes_cal[0].get_position().y0 + axes_cal[0].get_position().y1)
fig_cal.text(
    0.50,
    cal_center_y,
    "Relative Thermal Voltage $\\Delta E - \\Delta E_{\\mathrm{ref}}$ (mV)",
    ha="center",
    va="center",
    rotation="vertical",
    fontsize=10,
)


def rescale_cal():
    visible_y = []
    for artists in cal_artists.values():
        # Index 0 is main line; indices 2 & 3 are upper/lower ±3σ bounds
        if artists[0].get_visible():
            visible_y.extend(artists[2].get_ydata())
            visible_y.extend(artists[3].get_ydata())
    if visible_y:
        ymin, ymax = min(visible_y), max(visible_y)
        pad = (ymax - ymin) * 0.08 if ymax > ymin else 0.1
        axes_cal[0].set_ylim(ymin - pad, ymax + pad)


cal_widgets = add_horizontal_trace_toggle(
    fig_cal, cal_artists, rescale_cal, bar_width=0.44, bar_bottom=0.90, bar_height=0.06
)

plt.show()