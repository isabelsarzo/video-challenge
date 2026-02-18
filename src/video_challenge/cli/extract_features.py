import argparse
from pathlib import Path
from video_challenge.feature_extraction.extract_features import extract_features

def main():
    parser = argparse.ArgumentParser(
        description="Extract features from preprocessed (ACC) landmark data"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to preprocessed data directory"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output feature directory"
    )

    args = parser.parse_args()

    extract_features(
        input_dir=args.input,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()