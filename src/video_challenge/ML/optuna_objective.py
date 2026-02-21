import numpy as np
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import xgboost as xgb
from optuna.integration import XGBoostPruningCallback


def objective(trial, X, y, groups, cv):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 10.0, log=True
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 1.0, 10.0, log=True
        ),
    }

    f1_scores = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", MinMaxScaler()),
                (
                    "feature_selection",
                    SelectKBest(
                        score_func=mutual_info_classif,
                        k=trial.suggest_int("k", 50, 200, step=25),
                    ),
                ),
                (
                    "model",
                    xgb.XGBClassifier(
                        **params,
                        n_estimators=1000,  # IMPORTANT
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=32,
                        verbosity=0,
                        random_state=18,
                    ),
                ),
            ]
        )

        pipeline.fit(
            X_tr,
            y_tr,
            model__eval_set=[(X_val, y_val)],
            model__early_stopping_rounds=50,
            model__callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")],
        )

        preds = pipeline.predict(X_val)
        f1_scores.append(f1_score(y_val, preds, zero_division=0))

        trial.report(f1_score(y_val, preds, zero_division=0), len(f1_scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_scores)
