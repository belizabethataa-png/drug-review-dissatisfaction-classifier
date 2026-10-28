# Drug Review Dissatisfaction: What Drives It, and How It Differs Across Drugs

INFO 4360 course project (Fall 2026). **Status: Phase 2 complete (data exploration, text mining basics, Phase 3 plan).**

## Problem
Patients who rate a drug 4 or lower (out of 10) are treated as dissatisfied. The rating already exists, so
the project asks a different question: which reasons (side effects, no effect, cost, stopping or switching)
drive dissatisfaction, and how do they differ across drugs treating the same condition? The intended
stakeholder is a drug-maker brand or safety (pharmacovigilance) team.

## Data
Drug Reviews (Drugs.com) from the UCI Machine Learning Repository, CC BY 4.0: 215,063 reviews
(161,297 train / 53,766 test). About 25% of reviews are dissatisfied. The data files are not included in this
repo; place `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` in `data/`.

## Phase 2 findings
- Ratings are skewed toward 10; dissatisfied is the minority class (about 25%).
- About 80% of rows share exact text with another row, and 32,130 distinct texts appear in both train and
  test. Phase 3 deduplicates before evaluating.
- Top TF-IDF terms differ strongly by condition, so comparisons are made within condition.
- VADER sentiment alone reaches ROC-AUC 0.73, so sentiment is only a moderate proxy for the rating.
- Rule-based sentence tagging cannot reliably separate condition from drug description; Claude extraction is planned for this.

## Repo contents
- `notebooks/phase2_exploration.ipynb`: exploration, text mining, Claude sample run, and Phase 3 plan
- `requirements.txt`: Python packages
- `data/`: instructions and a small sample (full files not redistributed)

## How to run
1. `pip install -r requirements.txt`
2. Put the two data files in `data/`.
3. Open `notebooks/phase2_exploration.ipynb` in Jupyter Lab and run all cells.
4. For the Claude sample, set `ANTHROPIC_API_KEY` when prompted. Never commit a key.

## Next (Phase 3)
Claude-extracted reasons, models comparing controls only vs. controls + text vs. controls + reasons,
evaluated after removing duplicate texts.
