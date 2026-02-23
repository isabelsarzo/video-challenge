import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import wandb
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import xgboost as xgb
from .optuna_objective import objective, reset_wandb_env

from ..feature_extraction.pull_features import pull_features
from . import config as cfg
from .threshold_classifier import ThresholdedClassifier


t1 = datetime.now()
t1_timestamp = datetime.strftime(t1, "%Y%b%d-%H%M%S")

print(
    f"Initiated Cross-Validation script on {datetime.strftime(t1, '%d-%b-%Y %H:%M:%S')}"
)

wandb.login()

# results dir config
RESULTS = cfg.paths["results_root"] / f"results_run_{t1_timestamp}"
RESULTS.mkdir(parents=True, exist_ok=False)
SPLITS = RESULTS / "splits"
SPLITS.mkdir(parents=True, exist_ok=True)
OPTUNA_DIR = RESULTS / "optuna"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
optuna_db_file = OPTUNA_DIR / "optuna.db"
optuna_storage = f"sqlite:///{optuna_db_file}"
print(f"Results will be saved in: {RESULTS}")

# log config
log_dir = RESULTS / f"log_{t1_timestamp}.log"
logging.basicConfig(filename=log_dir, level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logger.info(f"CV started on: {datetime.strftime(t1, '%d-%b-%Y %H:%M:%S')}")

# load dataset
print("Loading dataset...")
features, desc = pull_features(
    dir=cfg.paths["features_dir"], labels=cfg.paths["labels_file"]
)

print(f"Dataset shape: {features.shape}")
print("Success!")
logger.info(f"Dataset: {desc}")

# cv configs
cv = StratifiedGroupKFold(n_splits=cfg.cv_folds)

# split data into training and testing (held-out test set)
split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=18) # just keep first fold
train_idx, test_idx = next(
    split.split(
        features, 
        features["label"], 
        groups=features["child_id"])
)
train_data = features.iloc[train_idx]
test_data = features.iloc[test_idx]

# -------- track splits --------
patients_in_training_set = train_data["child_id"].unique().tolist()
patients_in_testing_set = test_data["child_id"].unique().tolist()
logger.info(f"Patients in training set: {patients_in_training_set}")
logger.info(f"-------- n = {train_data["child_id"].nunique()}")
logger.info(f"Patients in testing set: {patients_in_testing_set}")
logger.info(f"-------- n = {test_data["child_id"].nunique()}")

pd.DataFrame(patients_in_training_set, columns=["child_id"]).to_csv(
    SPLITS / "patients_training.csv", index=False
)
pd.DataFrame(patients_in_testing_set, columns=["child_id"]).to_csv(
    SPLITS / "patients_testing.csv", index=False
)

# cv splits
for i, (subtrain, val) in enumerate(
    cv.split(
        X=train_data, y=train_data["label"], groups=train_data["child_id"]
    ),
    start=1,
):
    subtrain_patients = list(set(train_data["child_id"].values[subtrain]))
    val_patients = list(set(train_data["child_id"].values[val]))
    logger.info(f"----- CV FOLD {i}/{cv.get_n_splits()}:")
    logger.info(f"------ Patients in sub-training set: {subtrain_patients}")
    logger.info(f"--------- n = {len(subtrain_patients)}")
    logger.info(f"------ Patients in validation set: {val_patients}")
    logger.info(f"--------- n = {len(val_patients)}")

    pd.DataFrame(subtrain_patients, columns=["child_id"]).to_csv(
        SPLITS / f"cv_fold{i}_training.csv", index=False
    )
    pd.DataFrame(val_patients, columns=["child_id"]).to_csv(
        SPLITS / f"cv_fold{i}_validation.csv", index=False
    )

    # get subtrain labels
    y_subtrain = train_data["label"].iloc[subtrain]

    # count positives and negatives
    n_pos = (y_subtrain == 1).sum()
    n_neg = (y_subtrain == 0).sum()

    print(f"------ Positive samples in sub-training set: {n_pos}")
    print(f"------ Negative samples in sub-training set: {n_neg}")

    y_val = train_data["label"].iloc[val]
    n_pos_val = (y_val == 1).sum()
    n_neg_val = (y_val == 0).sum()

    print(f"------ Positive samples in validation set: {n_pos_val}")
    print(f"------ Negative samples in validation set: {n_neg_val}")

# ------ prepare data for pipeline ------
x_train = train_data.drop(columns=["segment_name", "segment_id", "child_id", "label"])
y_train = train_data["label"].to_numpy()

x_test = test_data.drop(columns=["segment_name", "segment_id", "child_id", "label"])
y_true = test_data["label"].to_numpy()

# execute optuna optimization for cv
logger.info("Executing Optuna optimization for CV...")
logger.info(f"Using model: {cfg.MODEL_TYPE}")

study = optuna.create_study(
    study_name="CV",
    storage=optuna_storage,
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=18),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=50,
    ),
)

study.optimize(
    lambda trial: objective(
        trial,
        x_train,
        y_train,
        train_data["child_id"].values,
        cv,
        wandb_dir=RESULTS,
        model_type=cfg.MODEL_TYPE
    ),
    n_trials=cfg.n_trials, # number of hp configs
)

# Get best hyperparms
best_params = study.best_params
best_k = best_params.pop("k")

reset_wandb_env()
run_name = "CV"
group_name_wandb = "cv_model"
wandb.init(
    project=cfg.project_name,
    name=run_name,
    config=best_params,
    reinit=True,
    group=group_name_wandb,
    dir=RESULTS,
)

# Get best model with best hyperparams
preprocessor = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler()),
        ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=best_k)),
    ]
)

X_train_transformed = preprocessor.fit_transform(x_train, y_train)
X_test_transformed = preprocessor.transform(x_test)

if cfg.MODEL_TYPE == "xgboost":

    neg, pos = np.bincount(y_train)
    spw = neg/pos
    
    best_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=1000,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=32,
        verbosity=1,
        random_state=18,
        scale_pos_weight=spw,
        early_stopping_rounds=50,
        callbacks=[
            wandb.xgboost.WandbCallback(),
        ],
    )

    best_model.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_test_transformed, y_true)],
    )

elif cfg.MODEL_TYPE == "tabnet":
    # Remove keys not accepted by TabNet
    tabnet_params = best_params.copy()
    for key in ["decision_threshold", "k"]:
        tabnet_params.pop(key, None)

    # Adjust TabNet-specific params
    if "lr" in tabnet_params:
        tabnet_params["optimizer_params"] = dict(lr=tabnet_params.pop("lr"))
    if "n_da" in tabnet_params:
        n_da = tabnet_params.pop("n_da")
        tabnet_params["n_d"] = n_da
        tabnet_params["n_a"] = n_da

    best_model = TabNetClassifier(
        **tabnet_params,
        seed=18,
        verbose=0,
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    )

    best_model.fit(
        X_train=X_train_transformed, 
        y_train=y_train,
        eval_set=[(X_test_transformed, y_true)],
        eval_name=['test'], 
        eval_metric=['logloss'],
        max_epochs=300, 
        patience=40,
        batch_size=256, 
        virtual_batch_size=64,
        weights=1,
        drop_last=False
    )

best_threshold = study.best_trial.params.get("decision_threshold", 0.5)
thresholded_model = ThresholdedClassifier(best_model, threshold=best_threshold)
pipeline = Pipeline(steps=preprocessor.steps + [("model", thresholded_model)])

# get best model and save it
model_path = RESULTS / f"CV_pipeline_{cfg.MODEL_TYPE}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

# get best model predictions on testing set
y_pred = thresholded_model.predict(X_test_transformed)

# -------- compute scores --------
recall = recall_score(y_true, y_pred, zero_division=np.nan)
precision = precision_score(y_true, y_pred, zero_division=np.nan)
f1score = f1_score(y_true, y_pred, zero_division=0)   

# save best hyperparams
best_params = study.best_params
best_params = pd.DataFrame([best_params])
best_params.to_csv(RESULTS / "best_params.csv", index=False)

scores = {
    "Recall": recall,
    "Precision": precision,
    "F1-score": f1score,
    "threshold": best_threshold
}

wandb.log(scores)
wandb.finish()

# save scores
scores = pd.DataFrame([scores])
logger.info(f"Scores on held-out test set: {scores}")
scores.to_csv(RESULTS / "scores.csv")

t2 = datetime.now()
t2_timestamp = datetime.strftime(t2, "%d-%b-%Y %H:%M:%S")
logger.info("----------------------------------------")
logger.info(f"Finished on: {t2_timestamp}")
logger.info(f"Training duration: {t2 - t1}")
logging.shutdown()
