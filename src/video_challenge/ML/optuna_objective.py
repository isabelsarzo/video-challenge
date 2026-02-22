import os

import numpy as np
import wandb
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback

from src.video_challenge.ML import config as cfg


# Clear any existing W&B environment variables to avoid conflicts
def reset_wandb_env():
    exclude = {"WANDB_PROJECT", "WANDB_ENTITY", "WANDB_API_KEY"}
    for key in list(os.environ.keys()):
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]


def objective(trial, X, y, groups, cv, wandb_dir, model_type="xgboost"):

    # -------------------------
    # Model-specific parameters
    # -------------------------

    if model_type == "xgboost":
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 3.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
        }

    elif model_type == "tabnet":

        n_da = trial.suggest_categorical("n_da", [8, 16, 24])  # smaller upper bound for stability

        params = {
            "n_d": n_da,
            "n_a": n_da,
            "n_steps": trial.suggest_int("n_steps", 3, 6),
            "gamma": trial.suggest_float("gamma", 1.3, 1.7),
            "n_independent": trial.suggest_int("n_independent", 1, 3),
            "n_shared": trial.suggest_int("n_shared", 1, 3),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-4, 1e-3, log=True),
            "optimizer_params": dict(
                lr=trial.suggest_float("lr", 5e-4, 5e-3, log=True)  # safer LR upper bound
            ),
            "optimizer_fn": torch.optim.AdamW,
            "mask_type": trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
        }

    # -------------------------
    # Shared hyperparameters
    # -------------------------

    k_features = trial.suggest_int("k", 50, 200, step=25)

    # Global decision threshold tuned by Optuna
    decision_threshold = trial.suggest_float("decision_threshold", 0.3, 0.7)

    f1_scores = []
    recall_scores = []
    precision_scores = []

    # -------------------------
    # Cross-validation loop
    # -------------------------

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):

        run_name = f"cv_fold_{fold}_trial_{trial.number}"
        group_name_wandb = f"{fold}_trial_{trial.number}"

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Preprocessing pipeline
        preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
            ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=k_features)),
        ])

        X_tr_transformed = preprocessor.fit_transform(X_tr, y_tr)
        X_val_transformed = preprocessor.transform(X_val)

        reset_wandb_env()

        wandb.init(
            project=cfg.project_name,
            name=run_name,
            config={**params, "k": k_features, "decision_threshold": decision_threshold},
            reinit=True,
            group=group_name_wandb,
            dir=wandb_dir,
        )

        # -------------------------
        # Train model
        # -------------------------

        if model_type == "xgboost":

            neg, pos = np.bincount(y_tr)
            spw = neg/pos

            model = xgb.XGBClassifier(
                **params,
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
                    XGBoostPruningCallback(trial, "validation_0-logloss"),
                    wandb.xgboost.WandbCallback(),
                ],
            )

            model.fit(
                X_tr_transformed,
                y_tr,
                eval_set=[(X_val_transformed, y_val)],
            )

        elif model_type == "tabnet":

            model = TabNetClassifier(
                **params,
                seed=18,
                verbose=0,
                device_name='cuda' if torch.cuda.is_available() else 'cpu'
            )

            model.fit(
                X_train=X_tr_transformed,
                y_train=y_tr,
                eval_set=[(X_val_transformed, y_val)],
                eval_name=['val'],
                eval_metric=['logloss'],
                max_epochs=300,
                patience=40,
                batch_size=256,
                virtual_batch_size=64,
                weights=1,  # keeps minority class from collapsing
                drop_last=False
            )

        # -------------------------
        # Evaluation (F1 optimized)
        # -------------------------

        probs = model.predict_proba(X_val_transformed)[:, 1]
        preds = (probs > decision_threshold).astype(int)

        f1 = f1_score(y_val, preds, zero_division=0)
        f1_scores.append(f1)

        recall = recall_score(y_val, preds, zero_division=np.nan)
        recall_scores.append(recall)
        precision = precision_score(y_val, preds, zero_division=np.nan)
        precision_scores.append(precision)

        wandb.log({
            "f1_score": f1,
            "mean_f1_so_far": np.mean(f1_scores),
            "recall": recall,
            "mean_recall_so_far": np.nanmean(recall_scores),
            "precision": precision,
            "mean_precision_so_far": np.nanmean(precision_scores),
            "threshold": decision_threshold,
            "fold": fold
        })

        wandb.finish()

        # Report to Optuna (for pruning)
        trial.report(f1, fold)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_scores)
