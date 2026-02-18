import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from video_challenge.utils.metadata import list_parquet_files
from video_challenge.feature_extraction import features as ft

def extract_features(input_dir: Path, output_dir: Path) -> None:

    t1 = datetime.now()

    # Skip files that have already been analyzed
    records_input = list_parquet_files(input_dir)
    records_output = list_parquet_files(output_dir)

    n_records_input = len(records_input)

    if len(records_output) > 0:
        records_output = {Path(rs).stem for rs in records_output}
        records_input = [r for r in records_input if Path(r).stem not in records_output]
        n_records_input_clean = len(records_input)
        print(f"Skipping {n_records_input - n_records_input_clean} records that have already been analyzed in the past")

    for i, record in enumerate(records_input, start=1):
        print(f"Beginning analysis for record {i}/{len(records_input)}: {record}")
        data = pd.read_parquet(input_dir / record)

        # Keep track of record name, child, and segment
        if data["segment_name"].nunique() != 1:
            raise ValueError(f"Parquet {record} contains multiple segment names.")
        
        if data["child_id"].nunique() != 1:
            raise ValueError(f"Parquet {record} contains data from multiple children.")
        
        if data["segment_id"].nunique() != 1:
            raise ValueError(f"Parquet {record} contains data from multiple segments.")
        
        if data["label"].nunique() != 1:
            raise ValueError(f"Parquet {record} has multiple lables")
        
        if data["segment_name"].unique().tolist()[0] != Path(record).stem:
            raise ValueError(f"Segment name does not match file name {Path(record).stem}.")
        
        segment_name = data["segment_name"].unique().tolist()[0]
        child_id = data["child_id"].unique().tolist()[0]
        segment_id = data["segment_id"].unique().tolist()[0]
        label = data["label"].unique().tolist()[0]

        data = data.drop(columns=["segment_name", "child_id", "segment_id", "label"])
        data = data.to_numpy()

        # Extract features
        rms = ft.RMS(data)
        zcr = ft.ZCR(data, threshold=0)
        mf = ft.medFreq(data, fs=30)
        pkfreq = ft.peak_freq(data, fs=30)
        var = ft.variance(data)
        rp = ft.relative_power(data, fs=30, freqband=[2, 5])
        meanjk, maxjk, stdjk = ft.jerk(data, fs=30)
        iqr = ft.IQR(data)

        # Build list of feature names
        features = ["RMS", "ZCR", "MF", "PkF", "VAR", "RP", "MEANJK", "MAXJK", "STDJK", "IQR"]
        axes = ["X", "Y", "Z"]

        feature_names = []
        for f in features:
            for axis in axes:
                feature_names.extend([f"{f}_{axis}_{lmk}" for lmk in range(33)])

        # Build features dataframe
        feature_values_list = [rms, zcr, mf, pkfreq, var, rp, meanjk, maxjk, stdjk, iqr]
        features_stack = np.hstack([np.array(f).flatten() for f in feature_values_list]).reshape(1, -1)
        df = pd.DataFrame(features_stack, columns=feature_names)
        df.insert(0, "segment_name", segment_name)
        df.insert(1, "child_id", child_id)
        df.insert(2, "segment_id", segment_id)
        df.insert(3, "label", label)

        # Save features
        df.to_parquet(output_dir / f"{segment_name}.parquet", index=False)

        print("Features extracted successfully")

    t2 = datetime.now()
    print(f"Process was executed in {t2 - t1}")

    return None