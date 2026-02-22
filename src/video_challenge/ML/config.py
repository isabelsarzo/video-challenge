from pathlib import Path
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer

project_name = "video-challenge"

paths = {
    "results_root": Path("./results"),
    "features_dir": Path("./dataset/features_AXES-MAG"),
    "labels_file": Path("./dataset/data/train_data.csv"),
}

ncv = {"n_outer": 3, "n_inner": 4}

scoring = {
    "precision": make_scorer(precision_score, zero_division=np.nan),
    "recall": make_scorer(recall_score, zero_division=np.nan),
    "f1-score": make_scorer(f1_score, zero_division=0),
}
