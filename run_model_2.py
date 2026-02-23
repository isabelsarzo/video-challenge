import pickle
import argparse
from pathlib import Path

import pandas as pd

from video_challenge.tabular.preprocessing.preprocess_dir import preprocess_directory_to_parquet
from video_challenge.tabular.feature_extraction.extract_features import extract_features
from video_challenge.tabular.feature_extraction.pull_features import pull_features

def run_model_2(input_dir: Path | str, output_csv: Path | str):
    print("RUNNING MODEL 2...")
    preprocessed_dir = Path("preprocessed")
    features_dir = Path("features")

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    # -------- 1. preprocess input data
    print("- Preprocessing data...")
    preprocess_directory_to_parquet(input_dir, preprocessed_dir)

    # -------- 2. extract features --------
    print("- Extracting features...")
    extract_features(preprocessed_dir, features_dir)

    # -------- 3. load features  --------
    features, _ = pull_features(features_dir, labels=None)

    # -------- 4. load pipeline  --------
    pipeline = pickle.load(open("models/tabnet.pkl"), 'rb')

    # -------- 5. make inferences  --------
    X_test = features.drop(columns=["segment_name", "child_id", "segment_id"])
    print("- Making infernce...")
    y_pred = pipeline.predict(X_test)

    # -------- 6. save predictions in output csv file  --------
    file_name = features["segment_name"].to_list()
    file_name_npy = [f"{f}.npy" for f in file_name]
    output = pd.DataFrame({
        "file_name": file_name_npy,
        "label": y_pred
    })
    output.to_csv(output_csv, index=False)
    print(f"SUCESS! Saved predictions to {output_csv}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", 
        required=True, 
        type=Path,
        help="Directory containing input npy files"
    )

    parser.add_argument(
        "--output", 
        required=True, 
        type=Path,
        help="Path to output csv file"
    )

    args = parser.parse_args()
    run_model_2(args.input, args.output)

if __name__ == "__main__":
    main()