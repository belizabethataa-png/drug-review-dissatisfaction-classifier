# Reading Between the Ratings: Predicting Dissatisfied Patient Drug Reviews

INFO 4360 NLP course project (Fall 2026). **Status: Phase 3 final deliverable.**
**Team:** Liz (solo project)

## Problem

A patient-experience manager at an online pharmacy or health-information site receives thousands of patient drug reviews and cannot read them all, yet reviews rated 4 out of 10 or lower signal churn, poor adherence and possible safety problems. This project builds a text classifier (Path A, prediction) that flags likely-dissatisfied reviews from their wording plus the drug and condition, producing a ranked queue for outreach, pharmacist follow-up or safety escalation. Claude labels the main reason behind a sample of dissatisfied reviews as context for the queue; those labels are hand-checked and never used as classifier training data.

## Data

Drug Reviews (Drugs.com), UCI Machine Learning Repository, CC BY 4.0: 215,063 reviews (161,297 train / 53,766 test) with drug, condition, review text, rating, date and usefulCount. Target: dissatisfied = rating of 4 or lower (about 25% of reviews). Duplicate texts are removed within each file, and test reviews whose text also appears in train are dropped. The raw files are not redistributed (see How to Run, step 4).

## Main results

<!-- RESULTS:START -->
Run `notebooks/final_report.ipynb` to generate this block (the last cell fills it in automatically).
<!-- RESULTS:END -->

## Repo contents

- `notebooks/final_report.ipynb` and `notebooks/final_report.html`: the final blog-style report (problem, Phase 2 recap, Claude integration, modeling, findings, business implications, ethics, AI usage log)
- `notebooks/phase2_exploration.ipynb`: Phase 2 exploration
- `drug_review_models.py`: reusable data loading, cleaning, features and model code (also runs as a script)
- `data/claude_labels.csv`, `data/claude_usage.csv`, `data/handcheck.csv`: saved Claude labels, token usage and the hand-check, so the report re-runs without API calls
- `results/`: tables written by the notebook
- `requirements.txt`, `.gitignore`, `LICENSE`

## How to Run

1. **Clone the repo and move into it**

   ```bash
   git clone https://github.com/belizabethataa-png/drug-review-dissatisfaction-classifier.git
   cd drug-review-dissatisfaction-classifier
   ```

2. **Create a virtual environment and install the packages**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Mac: source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Add your API key** (only needed to re-label with Claude)

   Create a file named `.env` in the project folder containing one line:

   ```
   ANTHROPIC_API_KEY=your-key-here
   ```

   `.env` is listed in `.gitignore`, so it is never committed.

4. **Get the data**

   Download the Drugs.com review files from the [UCI repository](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com) and save `drugsComTrain_raw.tsv` and `drugsComTest_raw.tsv` (a `.csv` copy also works) in the `data/` folder. They are git-ignored because of their size.

5. **Run the notebook**

   Open `notebooks/final_report.ipynb` and choose **Run All** (about 5 to 10 minutes on the full data; set `N_TRAIN = 30000` in the settings cell for a quick test). The Claude step reads the saved `data/claude_labels.csv`, so it makes no API calls unless you set `RELABEL = True`. Export the HTML with:

   ```bash
   jupyter nbconvert --to html notebooks/final_report.ipynb
   ```

Command-line alternative for the models only: `python drug_review_models.py --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv --drop-useful`

## API costs

Claude (`claude-haiku-5-5`, $0.10 per million input tokens and $0.50 per million output tokens for prompts up to 100K tokens) is used only to label the main reason for a sample of dissatisfied reviews. The exact tokens and cost of the run are reported in the notebook (section 3.5) and in the results block above. The classifier itself makes no API calls.

## Ethics and privacy

The dataset has no names or account IDs; the notebook scans review text for self-disclosed identifiers, sends only review text to the API, and checks model performance on rarely reviewed conditions. See section 7 of the report.

## License

MIT, see `LICENSE`.
