from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churnzero.data import read_csv
from churnzero.modeling import load_artifact, predict_with_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate churn predictions from a trained model.")
    parser.add_argument("--model", default="models/churn_model.joblib")
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="outputs/predictions.csv")
    args = parser.parse_args()

    artifact = load_artifact(args.model)
    test_df = read_csv(args.test)
    predictions = predict_with_artifact(artifact, test_df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    print(f"Wrote predictions: {out_path}")


if __name__ == "__main__":
    main()
