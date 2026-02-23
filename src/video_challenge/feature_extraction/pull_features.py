import pandas as pd
import duckdb
from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class FeaturesDescription:
    """
    Describes a DataFrame of features loaded with the "pull_features" function.

    Attributes:
        n_rows (int): 
            Number of rows in the DataFrame.
        n_cols (int): 
            Number of columns in the DataFrame (feature columns + metadata columns).
        n_patients (int): 
            Number of unique patients.
        n_records (int): 
            Number of unique segment names.
        n_features (int): 
            Number of feature columns (n_cols - metadata columns).
        n_positive (int):
            Number of positive samples (label == 1).
        n_negative (int):
            Number of negative samples (label == 0).
        has_nans (bool):
            Whether the DataFrame contains any NaNs.
        n_rows_with_nan (int);
            Number of rows that contain at least one NaN.
        which_patients (list[str]):
            List of unique patients.
        which_records (list[str]):
            List of unique records.
        which_features (list[str]):
            List of feature column names.
    """

    n_rows: int
    n_cols: int
    n_patients: int
    n_records: int
    n_features: int

    n_positive: int | None
    n_negative: int | None

    has_nans: bool
    n_rows_with_nan: int

    which_patients: list[str]
    which_records: list[str]
    which_features: list[str]

    @property
    def positive_ratio(self) -> float | None:
        """Returns the fraction of positive samples (Label == 1)"""
        if self.n_positive is None or self.n_rows == 0:
            return None
        return self.n_positive / self.n_rows
    
    def __str__(self, show_lists: bool =  False) -> str:
        """
        Returns a readable summary of FeaturesDescription.

        Args:
            show_lists (bool, optional): 
                If True, also display the lists of unique patients, records, and feature columns 
                contained in the DataFrame. Defaults to False.

        Returns:
            str: Multi-line string summarizing the DataFrame.
        """

        s = (
            "FeaturesDescription\n"
            "-------------------\n"
            f"Samples            : {self.n_rows}\n"
            f"Columns            : {self.n_cols}\n"
            f"Patients           : {self.n_patients}\n"
            f"Records            : {self.n_records}\n"
            f"Features           : {self.n_features}\n"
        )

        if self.n_positive is not None:
            s += (
                f"Positive samples   : {self.n_positive}\n"
                f"Negative samples   : {self.n_negative}\n"
                f"Positive ratio     : {self.positive_ratio:.3e}\n"
            )
        else:
            s += "Labels             : None\n"

        s += (
            f"Has NaNs           : {self.has_nans}\n"
            f"Rows with NaNs     : {self.n_rows_with_nan}"
        )

        if show_lists:
            s += "\n\nUnique values:"
            s += f"\nPatients      : {self.which_patients}"
            s += f"\nRecords       : {self.which_records}"
            s += f"\nFeature cols  : {self.which_features}"

        return s
    
    def summary(self) -> str:
        """
        Returns a one-line summary of total, positive, and negative samples.
        If no labels available, returns only the number of samples.
        """
        if self.n_positive is None:
            return f"{self.n_rows} samples | no labels"
        return (
            f"{self.n_rows} samples | "
            f"{self.n_positive} positive | "
            f"{self.n_negative} negative"
        )
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "FeaturesDescription":
        """
        Constructs a FeaturesDescription from a pandas DataFrame.

        Args:
            df (pd.DataFrame): 
                DataFrame returned by pull_features containing feature columns and metadata columns (including 'label').

        Returns:
            FeaturesDescription: 
                Object that describes the DataFrame of loaded features.
        """

        meta_cols = {
            "child_id",
            "segment_name",
            "label",
            "segment_id"
        }

        feature_cols = [c for c in df.columns if c not in meta_cols]

        has_label = "label" in df.columns

        n_positive = (df["label"] == 1).sum() if has_label else None
        n_negative = (df["label"] == 0).sum() if has_label else None

        return cls(
            n_rows=len(df),
            n_cols=df.shape[1],
            n_patients=df["child_id"].nunique(),
            n_records=df["segment_name"].nunique(),
            n_features=len(feature_cols),
            n_positive=n_positive,
            n_negative=n_negative,
            has_nans=df.isna().any().any(),
            n_rows_with_nan=df.isna().any(axis=1).sum(),
            which_patients=df["child_id"].unique().tolist(),
            which_records=df["segment_name"].unique().tolist(),
            which_features=feature_cols
        )

def pull_features(dir: Path|str, labels: Path|str|None = None, patients: list[str] = "all") -> tuple[pd.DataFrame, FeaturesDescription]:
    """
    Loads extracted features and merges them with generated labels.

    This function:
        - Loads all parquet files in 'dir'.
        - Optionally filters for specific patients.
        - Optionally merges features with their corresponding labels (based on segment_name).
        - Returns a pandas DataFrame and a FeaturesDescription object.

    Args:
        dir (Path | str): 
            Directory containing feature parquet files.
        labels (Path | str, optional):
            Path to the CSV file with 'segment_name' and 'label'.
        patients (list[str] | 'all', optional):
            List of patient IDs to load features from.
            Defaults to 'all'.

    Returns:
        tuple[pd.DataFrame, FeaturesDescription]: 
            df: DataFrame containing features, metadata columns, and labels (optional).
            desc: FeaturesDescription object describing the DataFrame.

    Raises:
        FileNotFoundError: 
            If no parquet files are found in the given directory 'dir'.
    """
    dir = Path(dir)

    files = list(dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {dir}")
    
    files_str = ", ".join(f"'{str(f)}'" for f in files)
    patient_filter = ""
    if patients != "all":
        quoted = ", ".join(f"'{p}'" for p in patients)
        patient_filter = f"WHERE child_id IN ({quoted})"
    
    query = f"""
    SELECT *
    FROM read_parquet([{files_str}])
    {patient_filter}
    """
    
    with duckdb.connect() as conn:
        df_features = conn.execute(query).fetchdf()

    if labels is None:
        df = df_features # skip merge if no labels were passed
    else:
        labels = Path(labels)
        df_labels = pd.read_csv(labels)
        if "segment_name" in df_labels.columns: # remove .npy extension from segment names
            df_labels["segment_name"] = df_labels["segment_name"].str.replace(r"\.npy$", "", regex=True)

        # merge on segment_name
        df = df_features.merge(df_labels, on="segment_name", how="left")

    # Replace infinities with NaN
    df = df.replace([-np.inf, np.inf], np.nan)

    desc = FeaturesDescription.from_dataframe(df)

    return df, desc