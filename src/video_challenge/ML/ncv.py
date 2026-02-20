import pandas as pd
import numpy as np
import pickle
import logging
import tqdm
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from video_challenge.feature_extraction.pull_features import pull_features
from video_challenge.ML.model import Model
from video_challenge.ML import config as cfg

t1 = datetime.now()
t1_timestamp = datetime.strftime(t1, "%Y%b%d-%H%M%S")

print(f"Initiated Nested Cross-Validation script on {datetime.strftime(t1, "%d-%b-%Y %H:%M:%S")}")

# results dir config
RESULTS_ROOT = Path("./results")
RESULTS = RESULTS_ROOT / f"results_run_{t1_timestamp}"
RESULTS.mkdir(parents=True, exist_ok=False)
print(f"Results will be saved in: {RESULTS}")

# log config
log_dir = RESULTS / f"log_{t1_timestamp}"
logging.basicConfig(
    filename=log_dir,
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"NCV started on: {datetime.strftime(t1, "%d-%b-%Y %H:%M:%S")}")

# load dataset
print("Loading dataset...")
features, desc = pull_features(
    dir=Path("./dataset/features"),
    labels=Path("./dataset/data/train_data.csv")
)

print(f"Dataset shape: {features.shape}")
print("Success!")
logger.info(f"Dataset: {desc}")

# outer/inner loop configs
outercv = StratifiedGroupKFold(n_splits=cfg.ncv["n_outer"])
innercv = StratifiedGroupKFold(n_splits=cfg.ncv["n_inner"])

# model config
model = cfg.model["Classifier_instance"]
clf_name = cfg.model["Classifier_name"]
hps_grid = cfg.model["Hyperparameter_grid"]

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
        outercv.split(
            X = features,
            y = features["label"],
            groups= features["child_id"]
        ),
        desc = "Outer CV Progress",
        total = outercv.get_n_splits()
    ),
    start=1
):
    
    # split data
    train_data, test_data = features.iloc[training, :], features.iloc[testing, :]

    # track outer loop splits
    logger.info(f"-- FOLD {fold}/{outercv.get_n_splits()} of outer loop:")
    logger.info(f"--- Patients in training set: {train_data['child_id'].unique().tolist()}")
    logger.info(f"--- Patients in testing set: {test_data['child_id'].unique().tolist()}")

    # track inner loop splits
    for i, (subtrain, val) in enumerate(
            innercv.split(
                X = train_data, 
                y = train_data['label'], 
                groups = train_data['child_id']
            ),
            start=1
        ):
        subtrain_patients = list(set(train_data['child_id'].values[subtrain]))
        val_patients = list(set(train_data['child_id'].values[val]))
        logger.info(f"----- FOLD {i}/{innercv.get_n_splits()} of inner loop:")
        logger.info(f"------ Patients in sub-training set: {subtrain_patients}")
        logger.info(f"------ Patients in validation set: {val_patients}")

    # build pipeline
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("feature_selection", SelectKBest(score_func=mutual_info_classif)),
        ('model', Model(
            classifier = model
        ))
    ])

    # safety check
    print(f"Safety check: {pipeline.get_params().keys()}")

    # execute grid search
    logger.info(f"Executing grid search for model: {clf_name}")
    grid_search = GridSearchCV(
        pipeline,
        hps_grid,
        scoring = cfg.scoring,
        cv = innercv,
        refit="f1-score",
        n_jobs=1,
        verbose=2
    )

    x_train = train_data.drop(colums=["segment_name", "segment_id", "child_id", "label"])
    y_train = train_data["label"].to_numpy()

    x_test = test_data.drop(columns=["segment_name", "segment_id", "child_id", "label"])
    y_true = test_data["label"].to_numpy()

    grid_search.fit(
        X = x_train,
        y = y_train,
        groups = train_data["child_id"].values
    )

    # get best model and save it
    best_model = grid_search.best_estimator_
    model_path = RESULTS / f"pipeline_fold{fold}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # get grid search cv results and save them
    gscv = pd.DataFrame(grid_search.cv_results_)
    gscv.to_csv(RESULTS / f'grid_search_cv_results_fold{fold}.csv')

    # get best model predictions on testing set
    y_pred = best_model.predict(x_test)

    # compute and store fold performance
    outercv_scores["Fold"].append(fold)
    outercv_scores["Recall"].append(recall_score(y_true, y_pred, zero_division=np.nan))
    outercv_scores["Precision"].append(precision_score(y_true, y_pred, zero_division=np.nan))
    outercv_scores["F1-score"].append(f1_score(y_true, y_pred, zero_division=0))

    # save best hyperparams
    best_params = grid_search.best_params_
    best_params = pd.DataFrame([best_params])
    best_params.to_csv(RESULTS / f"best_params_fold{fold}.csv", index=False)

# save outer scores
outercv_scores = pd.DataFrame(outercv_scores)
logger.info(f"Outer loop scores: {outercv_scores}")
outercv_scores.to_csv(RESULTS / "outercv_scores.csv")

# compute nested scores
logger.info("============ NESTED SCORES ===========")
logger.info(f"F1-score: {np.nanmean(outercv_scores["F1-score"].values)}")
logger.info(f"Recall: {np.nanmean(outercv_scores["Recall"].values)}")
logger.info(f"Precision: {np.nanmean(outercv_scores["Precision"].values)}")

t2 = datetime.now()
t2_timestamp = datetime.strftime(t2, "%d-%b-%Y %H:%M:%S")
logger.info("----------------------------------------")
logger.info(f"Finished on: {t2_timestamp}")
logger.info(f"Training duration: {t2 - t1}")
logging.shutdown()