# ChurnZero 26 Round 2 Submission Checklist

Use this before uploading on Unstop.

- [ ] Put the official dataset in `data/raw/banking.csv`.
- [ ] Run `python scripts/train.py --train data/raw/banking.csv --target y`.
- [ ] Run `python scripts/predict.py --model models/churn_model.joblib --test data/raw/banking.csv --out outputs/predictions.csv`.
- [ ] Confirm `outputs/predictions.csv` has the required ID and prediction columns.
- [ ] Run `python scripts/make_deck.py`.
- [ ] Review `submission/ChurnZero_26_Round2_Presentation.pptx`.
- [ ] Export the PPTX to PDF if Unstop requires PDF.
- [ ] Push repository to GitHub.
- [ ] Submit PPTX/PDF, GitHub link, and `outputs/predictions.csv` on Unstop.

Round 2 window from screenshot: May 20, 2026 08:00 PM IST to May 24, 2026 11:59 PM IST.
