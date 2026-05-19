from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def save_evaluation_figures(
    y_true,
    probabilities,
    predictions,
    feature_importance: pd.DataFrame,
    figures_dir: str | Path,
) -> dict[str, str]:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    sns.set_theme(style="whitegrid")

    cm = confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    paths["confusion_matrix"] = str(figures_dir / "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(paths["confusion_matrix"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("ROC Curve")
    paths["roc_curve"] = str(figures_dir / "roc_curve.png")
    fig.tight_layout()
    fig.savefig(paths["roc_curve"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("Precision-Recall Curve")
    paths["precision_recall_curve"] = str(figures_dir / "precision_recall_curve.png")
    fig.tight_layout()
    fig.savefig(paths["precision_recall_curve"], dpi=180)
    plt.close(fig)

    top_features = feature_importance.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=top_features, x="importance_mean", y="feature", color="#1769aa", ax=ax)
    ax.set_title("Top Churn Drivers")
    ax.set_xlabel("Permutation importance (average precision)")
    ax.set_ylabel("")
    paths["feature_importance"] = str(figures_dir / "feature_importance.png")
    fig.tight_layout()
    fig.savefig(paths["feature_importance"], dpi=180)
    plt.close(fig)

    return paths


def load_metrics(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
