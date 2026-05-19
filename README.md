# ChurnZero 26 - Banking Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-145DA0?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Round 2](https://img.shields.io/badge/ChurnZero%2026-Round%202-2E7D32?style=for-the-badge)

Round 2 submission for **ChurnZero 26 by IIT Kharagpur**. This repository builds a reproducible machine learning pipeline for predicting banking customer churn, exporting scored predictions, and generating a presentation-ready case-study deck.

## Team

| Role | Name |
| --- | --- |
| Team Leader | Dharmender Chauhan |
| Member | Rishita Sorout |

## What This Repo Delivers

- End-to-end churn prediction pipeline for banking customer data.
- Schema-tolerant preprocessing for numeric and categorical features.
- Soft-voting ensemble using Extra Trees and Histogram Gradient Boosting.
- Validation metrics, feature importance, and diagnostic plots.
- Scored prediction export in CSV format.
- Auto-generated Round 2 PPTX deck.

## Visual Report

| ROC Curve | Precision-Recall Curve |
| --- | --- |
| ![ROC Curve](reports/figures/roc_curve.png) | ![Precision Recall Curve](reports/figures/precision_recall_curve.png) |

| Churn Drivers | Confusion Matrix |
| --- | --- |
| ![Feature Importance](reports/figures/feature_importance.png) | ![Confusion Matrix](reports/figures/confusion_matrix.png) |

## Project Structure

```text
.
├── data/
│   ├── raw/                 # Place the official banking.csv file here
│   └── sample/              # Synthetic sample data for local verification
├── models/                  # Local trained model artifacts
├── outputs/                 # Metrics and prediction CSV outputs
├── reports/figures/         # Validation charts used in README and deck
├── scripts/
│   ├── make_sample_data.py
│   ├── train.py
│   ├── predict.py
│   └── make_deck.py
├── src/churnzero/           # Reusable data, feature, model, and report code
├── submission/              # Generated PPTX deck
└── tests/                   # Pipeline smoke tests
```

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the complete demo workflow:

```powershell
python scripts/make_sample_data.py
python scripts/train.py --train data/sample/train.csv --test data/sample/test.csv --target Exited
python scripts/make_deck.py
```

Run tests:

```powershell
python -m unittest discover -s tests
```

## Official Round 2 Run

Place the official dataset here:

```text
data/raw/banking.csv
```

Then run:

```powershell
python scripts/train.py --train data/raw/banking.csv --target y
python scripts/predict.py --model models/churn_model.joblib --test data/raw/banking.csv --out outputs/predictions.csv
python scripts/make_deck.py
```

The banking dataset uses `y` as the target column, and the training script auto-detects it when available.

## Generated Outputs

```text
models/churn_model.joblib
outputs/predictions.csv
outputs/metrics.json
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
reports/figures/precision_recall_curve.png
reports/figures/feature_importance.png
submission/ChurnZero_26_Round2_Presentation.pptx
```

## Modeling Summary

The model treats churn as a binary classification problem and optimizes both probability ranking and decision quality:

- median imputation and scaling for numeric fields
- most-frequent imputation and one-hot encoding for categorical fields
- automatic dropping of IDs, names, and high-cardinality text leakage risks
- class-weighted ensemble modeling for imbalance handling
- validation threshold selected by F1 score
- permutation importance for explainable churn drivers

## Model Performance & Evaluation

The final ensemble model was evaluated using a stratified 20% validation split from the official banking dataset. The metrics below demonstrate high predictive performance and balanced decision boundaries:

### Performance Metrics

| Metric | Value | Description |
| --- | --- | --- |
| **ROC-AUC** | `0.9455` | Excellent probability ranking and separation capability. |
| **Average Precision (PR-AUC)** | `0.6475` | Strong precision across recall thresholds on the imbalanced target. |
| **Accuracy** | `89.86%` | Overall classification correctness. |
| **Precision (Class 1)** | `53.27%` | Precision for predicting churners (minimizing false alarms). |
| **Recall (Class 1)** | `81.57%` | Sensitivity to capture actual churners (capturing 81.5% of true churners). |
| **F1-Score (Class 1)** | `0.6445` | Harmonized score balancing precision and recall. |
| **Log Loss** | `0.2689` | Well-calibrated probabilistic outputs. |
| **Decision Threshold** | `0.5740` | Optimally chosen threshold to maximize validation F1-score. |

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **0 (No Churn)** | 97.49% | 90.92% | 94.09% | 7,310 |
| **1 (Churn)** | 53.27% | 81.57% | 64.45% | 928 |
| **Macro Average** | 75.38% | 86.24% | 79.27% | 8,238 |
| **Weighted Average** | 92.51% | 89.86% | 90.75% | 8,238 |

### Key Churn Drivers (Permutation Importance)

The model features were analyzed using Permutation Feature Importance on the validation dataset to identify the true drivers of customer churn:

1. **`duration`** (Importance: `0.2779`): The length of the last customer contact in seconds. Longer call duration is the strongest predictor of customer engagement or resolution of issues.
2. **`emp_var_rate`** (Importance: `0.0286`): Employment variation rate (macroeconomic indicator). Indicates that broader economic stability affects customer churn.
3. **`euribor3m`** (Importance: `0.0223`): 3-month Euribor interest rate. Macroeconomic changes strongly influence financial decision-making and retention.
4. **`nr_employed`** (Importance: `0.0212`): Number of employees (macroeconomic indicator).
5. **`pdays`** (Importance: `0.0137`): Number of days since the customer was last contacted from a previous campaign.
6. **`day_of_week`** (Importance: `0.0133`): The day of the week of the contact.
7. **`poutcome`** (Importance: `0.0092`): Outcome of the previous marketing campaign.
8. **`contact`** (Importance: `0.0083`): Contact communication type (cellular vs. telephone).
9. **`default`** (Importance: `0.0081`): Whether the customer has credit in default.

## Submission Checklist

Round 2 expects:

- PPTX/PDF case-study presentation
- GitHub repository link with reproducible model code
- predictions CSV scored from the official banking dataset

Use [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) before uploading on Unstop.

