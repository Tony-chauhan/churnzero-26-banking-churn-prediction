from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churnzero.data import coerce_binary_target, find_id_column, find_target_column, summarize_frame
from churnzero.features import build_preprocessor, choose_feature_columns, split_feature_types


def build_model_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    ensemble = VotingClassifier(
        estimators=[
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=260,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
            (
                "hist_gb",
                HistGradientBoostingClassifier(
                    learning_rate=0.045,
                    max_iter=190,
                    l2_regularization=0.04,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ],
        voting="soft",
        weights=[0.55, 0.45],
        n_jobs=1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", ensemble)])


def best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
    }


def train_churn_model(
    train_df: pd.DataFrame,
    target_column: str | None = None,
    id_column: str | None = None,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    target_column = find_target_column(train_df, target_column)
    id_column = find_id_column(train_df, id_column)

    y = coerce_binary_target(train_df[target_column])
    feature_columns, dropped_columns = choose_feature_columns(train_df, target_column, id_column)
    X = train_df[feature_columns].copy()
    numeric_features, categorical_features = split_feature_types(train_df, feature_columns)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=valid_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = build_model_pipeline(numeric_features, categorical_features, random_state)
    pipeline.fit(X_train, y_train)

    valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
    threshold = best_f1_threshold(y_valid, valid_probabilities)
    metrics = evaluate_predictions(y_valid, valid_probabilities, threshold)
    metrics["target_rate_train"] = float(y.mean())
    metrics["data_summary"] = summarize_frame(train_df)
    metrics["target_column"] = target_column
    metrics["id_column"] = id_column
    metrics["feature_count"] = len(feature_columns)
    metrics["numeric_feature_count"] = len(numeric_features)
    metrics["categorical_feature_count"] = len(categorical_features)
    metrics["dropped_columns"] = dropped_columns

    artifact = {
        "pipeline": pipeline,
        "target_column": target_column,
        "id_column": id_column,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": dropped_columns,
        "threshold": threshold,
        "metrics": metrics,
        "random_state": random_state,
    }
    return {
        "artifact": artifact,
        "metrics": metrics,
        "validation": {
            "X_valid": X_valid,
            "y_valid": y_valid,
            "probabilities": valid_probabilities,
            "predictions": (valid_probabilities >= threshold).astype(int),
        },
    }


def save_artifact(artifact: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path) -> dict[str, object]:
    return joblib.load(path)


def save_metrics(metrics: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def predict_with_artifact(artifact: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = artifact["feature_columns"]
    id_column = artifact.get("id_column")
    threshold = float(artifact.get("threshold", 0.5))

    X = df.copy()
    for column in feature_columns:
        if column not in X.columns:
            X[column] = np.nan
    X = X[feature_columns]

    probabilities = artifact["pipeline"].predict_proba(X)[:, 1]
    if id_column and id_column in df.columns:
        ids = df[id_column].reset_index(drop=True)
        id_name = id_column
    else:
        ids = pd.Series(np.arange(len(df)), name="row_id")
        id_name = "row_id"

    return pd.DataFrame(
        {
            id_name: ids,
            "churn_probability": np.round(probabilities, 6),
            "churn_prediction": (probabilities >= threshold).astype(int),
        }
    )


def compute_feature_importance(
    pipeline: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    random_state: int,
    max_rows: int = 2500,
) -> pd.DataFrame:
    if len(X_valid) > max_rows:
        X_valid = X_valid.sample(max_rows, random_state=random_state)
        y_valid = y_valid.loc[X_valid.index]

    result = permutation_importance(
        pipeline,
        X_valid,
        y_valid,
        scoring="average_precision",
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_valid.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
