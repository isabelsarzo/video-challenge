from pathlib import Path
import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer

paths = {
    "results_root": Path("./results"),
    "features_dir": Path("./dataset/features"),
    "labels_file": Path("./dataset/data/train_data.csv"),
    "optuna_storage": "sqlite:////home/danielgalindo/projects/video-challenge/results/optuna/optuna.db",
}

ncv = {"n_outer": 5, "n_inner": 5}

clf = "XGBoost"

# XGBoost_hyperparam_grid = {
#     'feature_selection__k': [50, 100, 150, 200], # number of features to select
#     'model__classifier__max_depth': [6, 8, 10],
#     'model__classifier__min_child_weight': [3, 5, 7, 9],
#     'model__classifier__learning_rate': [0.1],  # 0.3 (default), 0.1
#     'model__classifier__max_delta_step': [1],  # default is 0 (balanced datasets), 1-10 helps with unbalanced datasets
#     'model__classifier__gamma': [1],  # 0 (default), 1 # 0.5, 1
#     'model__classifier__reg_lambda': [1], # L2 reg, default is 1
#     'model__classifier__reg_alpha': [0.5], # L1 reg, default is 0 # 0.1, 0.5, 1
#     'model__classifier__booster': ['gbtree'],  # default is gbtree
#     'model__classifier__n_jobs': [32], # same as challenge
#     'model__classifier__verbosity': [1],   # 0 for silent, 1 for warning (default), 2 for info, 3 for debug
#     'model__classifier__missing': [np.nan]
# }

XGBoost_hyperparam_grid = {
    "feature_selection__k": [50, 100, 150],  # number of features to select
    "model__max_depth": [6, 8, 10],
    "model__min_child_weight": [3, 5, 7, 9],
    "model__learning_rate": [0.1],  # 0.3 (default), 0.1
    "model__max_delta_step": [
        0,
        1,
    ],  # default is 0 (balanced datasets), 1-10 helps with unbalanced datasets
    "model__gamma": [1],  # 0 (default), 1 # 0.5, 1
    "model__reg_lambda": [1],  # L2 reg, default is 1
    "model__reg_alpha": [0, 0.1, 0.5],  # L1 reg, default is 0 # 0.1, 0.5, 1
    "model__booster": ["gbtree"],  # default is gbtree
    "model__n_jobs": [32],  # same as challenge
    "model__verbosity": [
        1
    ],  # 0 for silent, 1 for warning (default), 2 for info, 3 for debug
    "model__missing": [np.nan],
    "model__scale_pos_weight": [2.4],
}


XGBoost_hyperparam_space = {
    # ---------- feature selection ----------
    "feature_selection__k": optuna.distributions.IntDistribution(
        low=50, high=200, step=25
    ),
    # ---------- tree structure ----------
    "model__max_depth": optuna.distributions.IntDistribution(3, 10),
    "model__min_child_weight": optuna.distributions.FloatDistribution(
        1.0, 10.0, log=True
    ),
    # ---------- learning dynamics ----------
    "model__learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
    "model__max_delta_step": optuna.distributions.IntDistribution(0, 10),
    # ---------- regularization ----------
    "model__gamma": optuna.distributions.FloatDistribution(1e-8, 5.0, log=True),
    "model__reg_alpha": optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
    "model__reg_lambda": optuna.distributions.FloatDistribution(1e-3, 10.0, log=True),
    # ---------- class imbalance ----------
    "model__scale_pos_weight": optuna.distributions.FloatDistribution(
        1.0, 10.0, log=True
    ),
}

####################################### DO NOT EDIT BELOW ######################################

scoring = {
    "precision": make_scorer(precision_score, zero_division=np.nan),
    "recall": make_scorer(recall_score, zero_division=np.nan),
    "f1-score": make_scorer(f1_score, zero_division=0),
}

if clf == "XGBoost":
    clf_instance = xgb.XGBClassifier(
        n_jobs=32,
        booster="gbtree",
        verbosity=1,
        missing=np.nan,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=18,
        use_label_encoder=False,
    )
    hps_grid = XGBoost_hyperparam_grid
# NOTE: more classifiers and their hyperparams could be added to this condition

model = {
    "Classifier_name": clf,
    "Classifier_instance": clf_instance,
    "Hyperparameter_grid": hps_grid,
    "Hyperparameter_space": XGBoost_hyperparam_space,
}
