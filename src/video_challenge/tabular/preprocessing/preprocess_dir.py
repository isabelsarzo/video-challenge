from pathlib import Path

import pandas as pd

from src.video_challenge.tabular.preprocessing.compute_acc import process_single_file
from src.video_challenge.tabular.utils.extract_child_segment import parse_segment_name
from src.video_challenge.tabular.utils.check_file_label import get_label


def preprocess_directory_to_parquet(input_dir, output_dir, fps=30, label_csv_path=None):
    """
    Processes all .npy files, flattens them, and saves a single consolidated Parquet file.

    args:
        input_dir (str): Directory containing the .npy files.
        output_dir (str): Directory where the output Parquet file will be saved.
        fps (int): Frames per second of the original video, used to calculate time intervals.

    returns:
        None: Saves the processed data to a Parquet file.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    file_list = list(input_dir.glob("*.npy"))
    if not file_list:
        print(f"Warning: No .npy files found in {input_dir}.")
        return

    # Pre-generate column names: Acc_x_0, Acc_y_0, Acc_z_0 ... Acc_z_32
    columns = []
    for joint_idx in range(33):
        for coord in ["x", "y", "z"]:
            columns.append(f"Acc_{coord}_{joint_idx}")

    print(f"Processing {len(file_list)} files...")

    for file_path in file_list:
        # Get Acceleration (150, 33, 3)
        acc = process_single_file(file_path, fps=fps)

        # Reshape (150, 33, 3) -> (150, 99)
        flattened_acc = acc.reshape(acc.shape[0], -1)

        # Create DataFrame for this segment
        df = pd.DataFrame(flattened_acc, columns=columns)

        # Extract Metadata
        child_id, segment_idx = parse_segment_name(file_path.name)

        if label_csv_path:
            label = get_label(label_csv_path, f"{file_path.stem}.npy")

        # Add metadata columns (broadcasted to all 150 rows)
        df["child_id"] = child_id
        df["segment_id"] = segment_idx
        df["segment_name"] = file_path.stem

        if label_csv_path:
            df["label"] = label if label_csv_path else None

        # Concatenate everything and save
        output_file_path = output_dir / f"{file_path.stem}.parquet"
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file_path, engine="pyarrow", index=False)


#
# Execution Example
#
# preprocess_directory_to_parquet(
#     input_dir='./data/',
#     output_dir='./data_processed/'
# )
