# Figure 4. Thermal Response Stability

![Alt Text](delta_e_with_error_bands.png)


## Macro Selection and Parsing of Pulses (pulseParse.py)
Raw serial text data (closed_loop_titration.txt) is parsed into a pulse data structure and saved ina JSON file format. Macro-selection allows for some high level selection of the dataset.

### Time Series of Exp.
![Alt Text](parsed_pulse_timeseries.png)

### Parsed Pulses
<p float="left">
  <img src="raw_pulses.png" width="49%" />
  <img src="selected_pulses.png" width="49%" />
</p>

## Pulse Filtering and Endpoiont Calculation (pulseSelect.py)

![Alt Text](all_windows_timeseries_overlay.png)

<p float="left">
  <img src="window_1_filtered_pulses_overlay.png" width="32%" />
  <img src="window_2_filtered_pulses_overlay.png" width="32%" />
  <img src="window_3_filtered_pulses_overlay.png" width="32%" />
</p>