import pandas as pd
import numpy as np
import pickle
import logging
from tqdm import tqdm
from datetime import datetime
import wandb
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from .optuna_objective import objective, reset_wandb_env

from ..feature_extraction.pull_features import pull_features
from . import config as cfg


t1 = datetime.now()
t1_timestamp = datetime.strftime(t1, "%Y%b%d-%H%M%S")

print(
    f"Initiated Nested Cross-Validation script on {datetime.strftime(t1, '%d-%b-%Y %H:%M:%S')}"
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
logger.info(f"NCV started on: {datetime.strftime(t1, '%d-%b-%Y %H:%M:%S')}")

# load dataset
print("Loading dataset...")
features, desc = pull_features(
    dir=cfg.paths["features_dir"], labels=cfg.paths["labels_file"]
)

print(f"Dataset shape: {features.shape}")
print("Success!")
logger.info(f"Dataset: {desc}")

# outer/inner loop configs
outercv = StratifiedGroupKFold(n_splits=cfg.ncv["n_outer"])
innercv = StratifiedGroupKFold(n_splits=cfg.ncv["n_inner"])

# initialize dict to store scores
outercv_scores = {
    "Fold": [],
    "Recall": [],
    "Precision": [],
    "F1-score": [],
}

# execute outer loop
for fold, (training, testing) in enumerate(
    tqdm(
        outercv.split(X=features, y=features["label"], groups=features["child_id"]),
        desc="Outer CV Progress",
        total=outercv.get_n_splits(),
    ),
    start=1,
):
    # split data
    train_data, test_data = features.iloc[training, :], features.iloc[testing, :]

    # track outer loop splits
    patients_in_training_set = train_data["child_id"].unique().tolist()
    patients_in_testing_set = test_data["child_id"].unique().tolist()
    logger.info(f"-- FOLD {fold}/{outercv.get_n_splits()} of outer loop:")
    logger.info(f"--- Patients in training set: {patients_in_training_set}")
    logger.info(f"--------- n = {len(patients_in_training_set)}")
    logger.info(f"--- Patients in testing set: {patients_in_testing_set}")
    logger.info(f"--------- n = {len(patients_in_testing_set)}")

    # Save outer fold patients
    pd.DataFrame(patients_in_training_set, columns=["child_id"]).to_csv(
        SPLITS / f"fold{fold}_outer_training.csv", index=False
    )
    pd.DataFrame(patients_in_testing_set, columns=["child_id"]).to_csv(
        SPLITS / f"fold{fold}_outer_testing.csv", index=False
    )

    # track inner loop splits
    for i, (subtrain, val) in enumerate(
        innercv.split(
            X=train_data, y=train_data["label"], groups=train_data["child_id"]
        ),
        start=1,
    ):
        subtrain_patients = list(set(train_data["child_id"].values[subtrain]))
        val_patients = list(set(train_data["child_id"].values[val]))
        logger.info(f"----- FOLD {i}/{innercv.get_n_splits()} of inner loop:")
        logger.info(f"------ Patients in sub-training set: {subtrain_patients}")
        logger.info(f"--------- n = {len(subtrain_patients)}")
        logger.info(f"------ Patients in validation set: {val_patients}")
        logger.info(f"--------- n = {len(val_patients)}")

        # Save inner fold patients
        pd.DataFrame(subtrain_patients, columns=["child_id"]).to_csv(
            SPLITS / f"fold{fold}_inner{i}_training.csv", index=False
        )
        pd.DataFrame(val_patients, columns=["child_id"]).to_csv(
            SPLITS / f"fold{fold}_inner{i}_eval.csv", index=False
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

    x_train = train_data.drop(
        columns=["segment_name", "segment_id", "child_id", "label"]
    )
    y_train = train_data["label"].to_numpy()

    x_test = test_data.drop(columns=["segment_name", "segment_id", "child_id", "label"])
    y_true = test_data["label"].to_numpy()

    # execute optuna optimization for inner loop
    logger.info("Executing Optuna optimization for inner loop...")

    # Training for inner loop
    study = optuna.create_study(
        study_name=f"outer_{fold}",
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
            innercv,
            fold,
            wandb_dir=RESULTS,
        ),
        n_trials=100, # number of hp configs
    )

    # Get best hyperparms
    best_params = study.best_params
    reset_wandb_env()
    run_name = f"outer_{fold}"
    group_name_wandb = "outer_loop_models"
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
            (
                "feature_selection",
                SelectKBest(
                    score_func=mutual_info_classif,
                    k=best_params.pop("k"),
                ),
            ),
        ]
    )

    X_train_transformed = preprocessor.fit_transform(x_train, y_train)
    X_test_transformed = preprocessor.transform(x_test)

    best_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=study.best_trial.user_attrs.get(
            "best_iteration", 300
        ),  # safe fallback
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=32,
        verbosity=1,
        random_state=18,
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

    pipeline = Pipeline(steps=preprocessor.steps + [("model", best_model)])

    # get best model and save it
    model_path = RESULTS / f"pipeline_fold{fold}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    # get best model predictions on testing set
    y_pred = best_model.predict(X_test_transformed)

    # compute and store fold performance
    outercv_scores["Fold"].append(fold)
    outercv_scores["Recall"].append(recall_score(y_true, y_pred, zero_division=np.nan))
    outercv_scores["Precision"].append(
        precision_score(y_true, y_pred, zero_division=np.nan)
    )
    outercv_scores["F1-score"].append(f1_score(y_true, y_pred, zero_division=0))

    # save best hyperparams
    best_params = study.best_params
    best_params = pd.DataFrame([best_params])
    best_params.to_csv(RESULTS / f"best_params_fold{fold}.csv", index=False)

    wandb.log(
        {
            "Recall": outercv_scores["Recall"][-1],
            "Precision": outercv_scores["Precision"][-1],
            "F1-score": outercv_scores["F1-score"][-1],
        }
    )
    wandb.finish()

# save outer scores
outercv_scores = pd.DataFrame(outercv_scores)
logger.info(f"Outer loop scores: {outercv_scores}")
outercv_scores.to_csv(RESULTS / "outercv_scores.csv")

# compute nested scores
logger.info("============ NESTED SCORES ===========")
logger.info(f"F1-score: {np.nanmean(outercv_scores['F1-score'].values)}")
logger.info(f"Recall: {np.nanmean(outercv_scores['Recall'].values)}")
logger.info(f"Precision: {np.nanmean(outercv_scores['Precision'].values)}")

t2 = datetime.now()
t2_timestamp = datetime.strftime(t2, "%d-%b-%Y %H:%M:%S")
logger.info("----------------------------------------")
logger.info(f"Finished on: {t2_timestamp}")
logger.info(f"Training duration: {t2 - t1}")
logging.shutdown()
