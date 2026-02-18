import pandas as pd
from pathlib import Path


def get_label(label_csv_path, segment_name):
    """
    Given a CSV file with columns 'segment_name' and 'label',
    return the label for the given segment_name.
    Returns -1 if the segment_name is not found.
    """
    label_csv_path = Path(label_csv_path)
    if not label_csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {label_csv_path}")

    # Read only the two necessary columns
    df = pd.read_csv(label_csv_path, usecols=["segment_name", "label"])

    # Strip whitespace to avoid minor formatting issues
    df["segment_name"] = df["segment_name"].astype(str).str.strip()

    # Convert to dictionary for fast lookup
    label_dict = dict(zip(df["segment_name"], df["label"]))

    return label_dict.get(segment_name, -1)
