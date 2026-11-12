# Figure 3. Chloride Response

# Data Processing Steps

## Macro Selection and Parsing of Pulses (closedPulseParse.py)
Raw serial text data (closed_loop_titration.txt) is parsed into a pulse data structure and saved ina JSON file format. Macro-selection allows for some high level selection of the dataset.

### Time Series of Exp.
![Alt Text](parsed_pulse_timeseries.png)

### Parsed Pulses
<p float="left">
  <img src="raw_pulses.png" width="49%" />
  <img src="selected_pulses.png" width="49%" />
</p>

## Pulse Filtering (closedPulseSelect.py)

### Time Series of Exp.
![Alt Text](combined_filtered_pulses_timeseries.png)

### Filtered Pulses
![Alt Text](combined_filtered_pulses_overlay.png)

