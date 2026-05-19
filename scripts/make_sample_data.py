from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def make_bank_churn_frame(rows: int, seed: int, include_target: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    geography = rng.choice(["France", "Germany", "Spain"], size=rows, p=[0.5, 0.25, 0.25])
    gender = rng.choice(["Female", "Male"], size=rows, p=[0.46, 0.54])
    age = np.clip(rng.normal(39, 10.5, rows).round(), 18, 82).astype(int)
    tenure = rng.integers(0, 11, rows)
    credit_score = np.clip(rng.normal(650, 96, rows).round(), 350, 850).astype(int)
    balance = np.maximum(0, rng.normal(76000, 62000, rows))
    zero_balance = rng.random(rows) < 0.32
    balance[zero_balance] = 0
    num_products = rng.choice([1, 2, 3, 4], size=rows, p=[0.49, 0.44, 0.06, 0.01])
    has_card = rng.binomial(1, 0.7, rows)
    active = rng.binomial(1, 0.51, rows)
    salary = np.clip(rng.normal(100000, 57500, rows), 1000, 220000)

    logit = (
        -2.5
        + 0.065 * (age - 40)
        + 0.000006 * balance
        - 0.006 * (credit_score - 650)
        + 0.9 * (geography == "Germany")
        + 0.25 * (gender == "Female")
        + 1.25 * (num_products >= 3)
        - 1.15 * active
        - 0.045 * tenure
        + rng.normal(0, 0.25, rows)
    )
    churn_probability = sigmoid(logit)
    exited = rng.binomial(1, churn_probability)

    df = pd.DataFrame(
        {
            "RowNumber": np.arange(1, rows + 1),
            "CustomerId": 15_600_000 + np.arange(rows),
            "Surname": [f"Customer{i:05d}" for i in range(rows)],
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": np.round(balance, 2),
            "NumOfProducts": num_products,
            "HasCrCard": has_card,
            "IsActiveMember": active,
            "EstimatedSalary": np.round(salary, 2),
        }
    )
    if include_target:
        df["Exited"] = exited
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sample banking churn train/test CSV files.")
    parser.add_argument("--output-dir", default="data/sample")
    parser.add_argument("--train-rows", type=int, default=8000)
    parser.add_argument("--test-rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = make_bank_churn_frame(args.train_rows, args.seed, include_target=True)
    test_df = make_bank_churn_frame(args.test_rows, args.seed + 1, include_target=False)
    test_df["RowNumber"] = np.arange(args.train_rows + 1, args.train_rows + args.test_rows + 1)
    test_df["CustomerId"] = 15_600_000 + test_df["RowNumber"]

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    print(f"Wrote {output_dir / 'train.csv'} and {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
