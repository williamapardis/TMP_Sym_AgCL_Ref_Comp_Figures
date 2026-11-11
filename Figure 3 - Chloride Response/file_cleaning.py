from datetime import datetime, timedelta


def process_log(input_file, output_file, expected_columns=18):
    with open(input_file, "r") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if not raw_lines:
        return

    # Extract and retain header
    header = raw_lines[0]
    lines = raw_lines[1:]

    # Parse headers to locate CP and TRA column indices
    headers = [col.strip() for col in header.replace("\t", ",").split(",")]
    
    # The first column is the timestamp; the rest correspond to the comma-separated `data`
    data_headers = headers[1:] if len(headers) > 1 else headers
    
    # Identify 0-based indices within `data` that need division by 5
    scale_indices = {
        i for i, h in enumerate(data_headers)
        if "cp" in h.lower() or "tra" in h.lower()
    }

    blocks = []
    current_block = []

    for line in lines:
        parts = line.split("\t", 1)
        dt = None

        # Try parsing timestamp from the first column
        try:
            dt = datetime.strptime(parts[0].strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

        if dt is not None:
            if current_block:
                blocks.append(current_block)
            current_block = [(dt, parts[1] if len(parts) > 1 else parts[0])]
        else:
            if current_block:
                current_block.append((None, line))

    if current_block:
        blocks.append(current_block)

    cleaned_lines = []

    # Format header separator to match data (tabs to comma-space)
    if "\t" in header:
        cleaned_lines.append(header.replace("\t", ", ") + "\n")
    else:
        cleaned_lines.append(header + "\n")

    for i in range(len(blocks)):
        start_time = blocks[i][0][0]
        block_len = len(blocks[i])

        if i < len(blocks) - 1 and blocks[i + 1][0][0] is not None:
            end_time = blocks[i + 1][0][0]
            time_step = (end_time - start_time) / block_len
        else:
            time_step = timedelta(seconds=1) / block_len

        for index, (_, data) in enumerate(blocks[i]):
            cols = [c.strip() for c in data.split(",")]
            
            # Validate comma-separated column count
            if len(cols) != expected_columns:
                continue  # Skip lines that don't match the required length

            # Scale CP and TRA values by dividing by 5
            scaled_cols = []
            for col_idx, val in enumerate(cols):
                if col_idx in scale_indices:
                    try:
                        scaled_val = float(val) / 5.0
                        # Round/format neatly while removing trailing zero noise
                        scaled_cols.append(f"{scaled_val:.3f}".rstrip("0").rstrip("."))
                    except ValueError:
                        scaled_cols.append(val)
                else:
                    scaled_cols.append(val)

            row_time = start_time + (time_step * index)
            formatted_time = row_time.strftime("%Y-%m-%d %H:%M:%S")
            reconstructed_data = ",".join(scaled_cols)

            cleaned_lines.append(f"{formatted_time}, {reconstructed_data}\n")

    with open(output_file, "w") as f:
        f.writelines(cleaned_lines)


# Set expected_columns to match your data structure (18 for your sample)
process_log(
    "10p titration 2026-09-12 19-15_original.txt",
    "cleaned.txt",
    expected_columns=18,
)