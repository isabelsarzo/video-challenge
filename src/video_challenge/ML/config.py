import xgboost as xgb
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, make_scorer

ncv = {
    "n_outer": 5,
    "n_inner": 5
}

clf = "XGBoost"

XGBoost_hyperparam_grid = {  
    'feature_selection__k': [50, 100, 150, 200], # number of features to select
    'model__classifier__max_depth': [6, 8, 10],
    'model__classifier__min_child_weight': [3, 5, 7, 9],
    'model__classifier__learning_rate': [0.1],  # 0.3 (default), 0.1
    'model__classifier__max_delta_step': [1],  # default is 0 (balanced datasets), 1-10 helps with unbalanced datasets
    'model__classifier__gamma': [1],  # 0 (default), 1 # 0.5, 1
    'model__classifier__reg_lambda': [1], # L2 reg, default is 1
    'model__classifier__reg_alpha': [0.5], # L1 reg, default is 0 # 0.1, 0.5, 1
    'model__classifier__booster': ['gbtree'],  # default is gbtree
    'model__classifier__n_jobs': [32], # same as challenge
    'model__classifier__verbosity': [1],   # 0 for silent, 1 for warning (default), 2 for info, 3 for debug
    'model__classifier__missing': [np.nan]
}

####################################### DO NOT EDIT BELOW ######################################

scoring = { 
    'precision': make_scorer(precision_score, zero_division=np.nan),
    'recall': make_scorer(recall_score, zero_division=np.nan),
    'f1-score': make_scorer(f1_score, zero_division=0),
}


if clf == "XGBoost":
    clf_instance = xgb.XGBClassifier()
    hps_grid = XGBoost_hyperparam_grid
# NOTE: more classifiers and their hyperparams could be added to this condition

model = {
    "Classifier_name": clf,
    "Classifier_instance": clf_instance,
    "Hyperparameter_grid": hps_grid,
}
