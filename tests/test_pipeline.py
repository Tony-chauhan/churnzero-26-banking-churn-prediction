from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from churnzero.modeling import predict_with_artifact, train_churn_model
from make_sample_data import make_bank_churn_frame


class ChurnPipelineTest(unittest.TestCase):
    def test_train_and_predict_on_sample_schema(self) -> None:
        train_df = make_bank_churn_frame(700, 123, include_target=True)
        test_df = make_bank_churn_frame(50, 456, include_target=False)

        result = train_churn_model(train_df, target_column="Exited", valid_size=0.25, random_state=7)
        artifact = result["artifact"]
        metrics = result["metrics"]
        predictions = predict_with_artifact(artifact, test_df)

        self.assertGreater(metrics["roc_auc"], 0.6)
        self.assertEqual(len(predictions), len(test_df))
        self.assertIn("churn_probability", predictions.columns)
        self.assertTrue(predictions["churn_probability"].between(0, 1).all())

    def test_missing_test_column_is_handled(self) -> None:
        train_df = make_bank_churn_frame(500, 123, include_target=True)
        test_df = make_bank_churn_frame(20, 456, include_target=False).drop(columns=["Balance"])

        result = train_churn_model(train_df, target_column="Exited", valid_size=0.25, random_state=7)
        predictions = predict_with_artifact(result["artifact"], test_df)

        self.assertEqual(len(predictions), 20)


if __name__ == "__main__":
    unittest.main()
