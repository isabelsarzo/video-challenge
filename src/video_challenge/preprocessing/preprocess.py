import time

from src.video_challenge.preprocessing.preprocess_dir import (
    preprocess_directory_to_parquet,
)

time_start = time.time()
preprocess_directory_to_parquet(
    input_dir="./dataset/data/", output_dir="./dataset/data_preprocessed/"
)
time_end = time.time()

print(f"Total processing time: {time_end - time_start:.2f} seconds")
