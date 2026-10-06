# Drug Review Dissatisfaction Classifier

INFO 4360 Course Project (Fall 2026): NLP for Real-World Problems

## Problem

A patient-experience team at an online pharmacy or health-information site receives thousands of drug reviews and cannot read them all. Dissatisfied patients (rating of 4 or lower out of 10) often describe side effects, lack of effectiveness, or reasons for stopping a medication. These reviews signal churn, safety concerns, and poor adherence.

**Stakeholder:** the patient-experience manager, who needs a ranked queue of likely-dissatisfied reviews for outreach, pharmacist follow-up, or safety escalation.

**Why it matters:** fewer missed safety signals, better patient retention, and staff time focused on the reviews that matter most.

## Approach (Path A: Classification)

- **Target:** `dissatisfied = 1` if `rating <= 4`, else `0` (created from the existing rating column).
- **Text field:** `review`
- **Control variables:** condition, drug name, review year, review length, exclamation count, caps ratio, log(usefulCount)
- **Models:**
  1. Logistic regression on TF-IDF text + controls (interpretable)
  2. Gradient boosting on SVD-reduced TF-IDF + controls (non-linear)
- **Ablation:** text only vs. controls only vs. text + controls, to show what the control variables add.
- **Leakage check:** `usefulCount` is only known after a review is posted and may act as a proxy for rating, so results are also run with `--drop-useful`.
- **Metrics:** precision, recall, F1, ROC-AUC, and PR-AUC on the held-out test file (dissatisfied reviews are the minority class).

## Data

UCI Machine Learning Repository, [Drug Reviews (Drugs.com)](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com): 215,063 reviews, split 75/25 into train and test files.

The full files are **not included** in this repo because the dataset is licensed for research use only with no redistribution. To reproduce:

1. Download the dataset from the UCI link above.
2. Put `drugsComTrain_raw.tsv` and `drugsComTest_raw.tsv` in the `data/` folder.

`data/sample_reviews.tsv` is a 500-row sample for a quick look at the format.

## How to Run

Requires Python 3 with `pandas`, `numpy`, and `scikit-learn` (1.2 or newer).

```bash
pip install pandas numpy scikit-learn

# Full run
python drug_review_models.py --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv

# Quick run on a subsample
python drug_review_models.py --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv --sample 30000

# Without usefulCount (leakage check)
python drug_review_models.py --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv --drop-useful
```

Outputs are saved to `results/`:
- `model_comparison.csv`: test-set metrics for all models
- `lr_coefficients.csv`: word and control-variable coefficients from Model 1

## Repository Structure

```
.
├── README.md
├── drug_review_models.py
├── data/
│   └── sample_reviews.tsv
└── results/
```

## Tools

Classical NLP (TF-IDF, logistic regression, gradient boosting) is the main approach. The Claude API may be used on a small sample of flagged reviews to categorize the reason for dissatisfaction as a qualitative check. The classifier, not Claude, makes the predictions.

## Status

Phase 1 (problem and data): complete. Results and interpretation will be added in later phases.

## Author

Liz
