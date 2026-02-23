from pathlib import Path
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer

MODEL_TYPE = "tabnet" # "xgboost" or "tabnet"

project_name = f"video-challenge_{MODEL_TYPE}_final"

n_trials = 200

cv_folds = 5

paths = {
    "results_root": Path("./results"),
    "features_dir": Path("./dataset/features_AXES-MAG"),
    "labels_file": Path("./dataset/data/train_data.csv"),
    "final_model_xgboost": Path("./models/xgboost.pkl"),
    "final_model_tabnet": Path("./models/tabnet.pkl"),
}

scoring = {
    "precision": make_scorer(precision_score, zero_division=np.nan),
    "recall": make_scorer(recall_score, zero_division=np.nan),
    "f1-score": make_scorer(f1_score, zero_division=0),
}