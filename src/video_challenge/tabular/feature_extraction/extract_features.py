import pandas as pd
from pathlib import Path
from datetime import datetime
from video_challenge.tabular.utils.metadata import list_parquet_files
from video_challenge.tabular.feature_extraction import features as ft

def extract_features(input_dir: Path, output_dir: Path) -> None:
    """
    Extract temporal and spectral features from ACC data. 
    Features are extracted from the preprocessed landmarks (position data)
    obatined from videos previously processed with mediapipe.

    Args:
        input_dir (Path): 
            Directory containing the preprocessed data stored as parquet files.
        output_dir (Path): 
            Output directory where features will be saved.

    Raises:
        ValueError: 
            If multiple segment names (records) are found in a single parquet.
        ValueError: 
            If multiple children IDs are found in a single parquet.
        ValueError: 
            If multiple segment IDs are found in a single parquet.
        ValueError: 
            If segment name does not match file name.

    """

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
        data = pd.read_parquet(input_dir / record) # shape: (150, 103)

        # Keep track of record name, child, and segment
        if data["segment_name"].nunique() != 1:
            raise ValueError(f"Parquet {record} contains multiple segment names.")
        
        if data["child_id"].nunique() != 1:
            raise ValueError(f"Parquet {record} contains data from multiple children.")
        
        if data["segment_id"].nunique() != 1:
            raise ValueError(f"Parquet {record} contains data from multiple segments.")
        
        if data["segment_name"].unique().tolist()[0] != Path(record).stem:
            raise ValueError(f"Segment name does not match file name {Path(record).stem}.")
        
        segment_name = data["segment_name"].unique().tolist()[0]
        child_id = data["child_id"].unique().tolist()[0]
        segment_id = data["segment_id"].unique().tolist()[0]

        data = data.drop(
            columns=["segment_name", "child_id", "segment_id", "label"],
            errors="ignore"
        ) # (150, 103) --> (150, 99)

        data = data.to_numpy()

        # Extract features
        features = {}
        mag = ft.compute_magnitude(data) # ACC signal magnitude
        
        # --- temporal ---
        features["RMS"] = ft.RMS(mag)
        features["ZCR"] = ft.ZCR(mag, threshold=0)
        features["VAR"] = ft.variance(mag)
        features["IQR"] = ft.IQR(mag)
        features["SampEn"] = ft.sample_entropy(mag)
        features["Kurt"] = ft.kurtosis(mag)
        features["Skew"] = ft.skewness(mag)
        features["BurstDur"] = ft.burst_duration(mag)
        features["BurstAmp"] = ft.burst_amplitude(mag)
        features["RiseFall"] = ft.rise_fall_ratio(mag)
        
        # jerk
        meanjk, maxjk, stdjk = ft.jerk(mag, fs=30)
        features["MEANJK"] = meanjk
        features["MAXJK"] = maxjk
        features["STDJK"] = stdjk

        # axis relationships
        features["CORR"] = ft.axis_correlation(data)
        features["TAC"] = ft.tilt_angle_change(data)

        # --- spectral ---
        freqs, psd = ft.compute_psd(mag, fs=30)
        features["MF"] = ft.medfreq(freqs, psd)
        features["PkF"] = ft.peak_freq(freqs, psd)
        features["RP"] = ft.relative_power(freqs, psd, freqband=(2, 5))
        features["SpEntr"] = ft.spectral_entropy(psd)
        features["SpCen"] = ft.spectral_centroid(freqs, psd)
        features["RollOff"] = ft.spectral_rolloff(freqs, psd)
        features["BPR"] = ft.band_power_ratio(freqs, psd, band1=(0.5, 3), band2=(3, 10))

        # prepare dataframe
        df = ft.features_to_dataframe(features, n_channels=data.shape[1])
        df.insert(0, "segment_name", segment_name)
        df.insert(1, "child_id", child_id)
        df.insert(2, "segment_id", segment_id)

        print(f"Shape of features df: {df.shape}") # (1, n_features + 3) = (1, 795)

        # Save features
        df.to_parquet(output_dir / f"{segment_name}.parquet", index=False)

        print("Features extracted successfully")

    t2 = datetime.now()
    print(f"Process was executed in {t2 - t1}")

    return None