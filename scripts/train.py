from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churnzero.data import read_csv
from churnzero.modeling import compute_feature_importance, predict_with_artifact, save_artifact, save_metrics, train_churn_model
from churnzero.reporting import save_evaluation_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Train banking customer churn model.")
    parser.add_argument("--train", default="data/raw/train.csv", help="Training CSV path.")
    parser.add_argument("--test", default=None, help="Optional test CSV path for prediction export.")
    parser.add_argument("--target", default=None, help="Target column name. Auto-detected if omitted.")
    parser.add_argument("--id-column", default=None, help="Customer ID column. Auto-detected if omitted.")
    parser.add_argument("--model-out", default="models/churn_model.joblib")
    parser.add_argument("--metrics-out", default="outputs/metrics.json")
    parser.add_argument("--predictions-out", default="outputs/predictions.csv")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_df = read_csv(args.train)
    result = train_churn_model(
        train_df,
        target_column=args.target,
        id_column=args.id_column,
        valid_size=args.valid_size,
        random_state=args.random_state,
    )
    artifact = result["artifact"]
    validation = result["validation"]

    feature_importance = compute_feature_importance(
        artifact["pipeline"],
        validation["X_valid"],
        validation["y_valid"],
        random_state=args.random_state,
    )
    figure_paths = save_evaluation_figures(
        validation["y_valid"],
        validation["probabilities"],
        validation["predictions"],
        feature_importance,
        args.figures_dir,
    )

    metrics = result["metrics"]
    metrics["figure_paths"] = figure_paths
    metrics["top_features"] = feature_importance.head(12).to_dict(orient="records")

    save_artifact(artifact, args.model_out)
    save_metrics(metrics, args.metrics_out)

    if args.test:
        test_df = read_csv(args.test)
        predictions = predict_with_artifact(artifact, test_df)
        predictions_path = Path(args.predictions_out)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(predictions_path, index=False)
        print(f"Wrote predictions: {predictions_path}")

    print(f"Model saved: {args.model_out}")
    print(f"Metrics saved: {args.metrics_out}")
    print(
        "Validation metrics: "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"AP={metrics['average_precision']:.4f}, "
        f"F1={metrics['f1']:.4f}, "
        f"Recall={metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
