import pickle
from pathlib import Path
import os

import pandas as pd

from video_challenge.preprocessing.preprocess_dir import preprocess_directory_to_parquet
from video_challenge.feature_extraction.extract_features import extract_features
from video_challenge.feature_extraction.pull_features import pull_features

def run_model_2():
    # read env variables
    input_rel_path = os.getenv("INPUT", "")
    output_rel_path = os.getenv("OUTPUT", "predictions.csv")

    input_dir = Path("/data") / input_rel_path
    output_csv = Path("/output") / output_rel_path

    # setup internal tmp dirs for preprocessed data and extracted features
    preprocessed_dir = Path("/tmp/preprocessed")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path("/tmp/features")
    features_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    #         Run Pipeline
    # --------------------------
    # -------- 1. preprocess input data
    preprocess_directory_to_parquet(input_dir, preprocessed_dir)

    # -------- 2. extract features --------
    extract_features(preprocessed_dir, features_dir)

    # -------- 3. load features  --------
    features, _ = pull_features(features_dir, labels=None)

    # -------- 4. load pipeline  --------
    pipeline = pickle.load(open("models/tabnet.pkl", 'rb'))

    # -------- 5. make inferences  --------
    X_test = features.drop(columns=["segment_name", "child_id", "segment_id"])
    y_pred = pipeline.predict(X_test)

    # -------- 6. save predictions in output csv file  --------
    file_name = features["segment_name"].to_list()
    file_name_npy = [f"{f}.npy" for f in file_name]
    output = pd.DataFrame({
        "file_name": file_name_npy,
        "label": y_pred
    })
    output.to_csv(output_csv, index=False)

    print("Pipeline complete. Submission file generated.")

def main():
    run_model_2()

if __name__ == "__main__":
    main()