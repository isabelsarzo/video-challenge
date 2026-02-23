import pickle
from datetime import datetime

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from . import config as cfg
from ..feature_extraction.pull_features import pull_features
from .threshold_classifier import ThresholdedClassifier

t1 = datetime.now()

print("Loading features dataset...")
features, desc = pull_features(
    dir=cfg.paths["features_dir"],
    labels=cfg.paths["labels_file"]
)
print("----- Features loaded successfully!")

X = features.drop(columns=["segment_name", "segment_id", "child_id", "label"])
y = features["label"].to_numpy()

neg, pos = np.bincount(y)
spw = neg / pos

xgboost = xgb.XGBClassifier(
    max_depth= 3,
    min_child_weight= 2.3577683779128176,
    learning_rate= 0.041545340286190356,
    gamma= 1.917817734715575e-06,
    reg_alpha= 0.10068555337525686,
    reg_lambda= 3.576031708765849,
    max_delta_step= 0,
    scale_pos_weight= spw,
    n_estimators= 223, # average of cv folds
    random_state= 18,
    n_jobs= 32,
    tree_method="hist",
    objective="binary:logistic",
)

tabnet = TabNetClassifier(
    n_d= 8,
    n_a= 8,
    n_steps= 4,
    gamma= 1.569774781475759,
    n_independent= 1,
    n_shared= 1,
    lambda_sparse= 0.0003928179923800353,
    optimizer_params= {"lr": 0.004218844992715255},
    optimizer_fn= torch.optim.AdamW,
    mask_type= 'entmax',
    device_name='cuda' if torch.cuda.is_available() else 'cpu',
    seed=18,
)

pipeline_1 = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=150)),
        ("model", ThresholdedClassifier(xgboost, threshold=0.37724446452896593))
    ]
)

pipeline_2 = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=200)),
        ("model", ThresholdedClassifier(tabnet, threshold=0.3809105463249145))
    ]
)

print("Training xgboost...")
pipeline_1.fit(X, y)
print("---- Sucess!")

with open(cfg.paths["final_model_xgboost"], "wb") as f:
    pickle.dump(pipeline_1, f)
print("---- xgboost model saved")

print("Training tabnet...")
pipeline_2.fit(
    X, y,
    model__max_epochs=202,
    model__weights=1,
    model__batch_size=256,
    model__virtual_batch_size=64,
    model__drop_last=False,
)
print("---- Sucess!")

with open(cfg.paths["final_model_tabnet"], "wb") as f:
    pickle.dump(pipeline_2, f)
print("---- tabnet model saved")

t2 = datetime.now()

print("-------------------------------------")
print(f"Process executed in {t2 - t1}")