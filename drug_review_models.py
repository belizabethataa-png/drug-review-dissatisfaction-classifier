"""
INFO 4360 Project - Drug Review Dissatisfaction Classifier
Stakeholder: patient-experience / pharmacy-content team that wants to flag
dissatisfied patient reviews (rating <= 4) for follow-up.

Data: UCI "Drug Reviews (Drugs.com)" (drugsComTrain_raw.tsv, drugsComTest_raw.tsv)
      https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com

Models
  Model 1: Logistic Regression on TF-IDF text + control variables   (interpretable)
  Model 2: Gradient Boosting on SVD(TF-IDF) + control variables     (non-linear)
Ablation (shows what the control variables add):
  LR text only | LR controls only | LR text + controls

Usage
  python drug_review_models.py --train data/drugsComTrain_raw.tsv --test data/drugsComTest_raw.tsv
  python drug_review_models.py --train ... --test ... --sample 30000     # quick run
  python drug_review_models.py --train ... --test ... --drop-useful      # leakage check
  python drug_review_models.py --train ... --make-sample data/sample_reviews.tsv   # 500-row sample, then exit
Requires: pandas, numpy, scikit-learn>=1.2
"""
import argparse
import html
import os
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
NEG_CUTOFF = 4  # rating <= 4 -> dissatisfied (target = 1)


# --------------------------------------------------------------------------- #
# Data loading and feature engineering
# --------------------------------------------------------------------------- #
def load_raw(path):
    # Works for UCI .tsv (tab-separated) and Kaggle .csv (comma-separated) copies
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    first = df.columns[0]
    if first.startswith("Unnamed") or first == "":
        df = df.rename(columns={first: "uniqueID"})
    return df


def clean(df):
    df = df.copy()
    df = df.dropna(subset=["review", "condition", "rating", "date", "usefulCount"])
    # Some condition values are scraping artifacts like "3</span> users found this..."
    df = df[~df["condition"].astype(str).str.contains("</span>", regex=False)]
    df["review"] = (
        df["review"].astype(str).map(html.unescape).str.replace('"', "", regex=False).str.strip()
    )
    df = df[df["review"].str.len() > 0]
    df = df.drop_duplicates(subset=["review"])
    return df


def add_features(df, drop_useful=False):
    df = df.copy()
    df["target"] = (df["rating"] <= NEG_CUTOFF).astype(int)

    # Control variables (non-text)
    words = df["review"].str.split()
    df["n_words"] = words.str.len()
    df["exclaim_count"] = df["review"].str.count("!")
    df["caps_ratio"] = df["review"].map(
        lambda s: sum(c.isupper() for c in s) / max(len(s), 1)
    )
    dates = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = dates.dt.year
    df["log_useful"] = np.log1p(df["usefulCount"].astype(float))
    df["condition"] = df["condition"].astype(str).str.strip()
    df["drugName"] = df["drugName"].astype(str).str.strip()
    df = df.dropna(subset=["year"])
    return df


def feature_lists(drop_useful):
    numeric = ["n_words", "exclaim_count", "caps_ratio", "year", "log_useful"]
    if drop_useful:
        numeric.remove("log_useful")
    categorical = ["condition", "drugName"]
    return numeric, categorical


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #
def make_ohe(dense):
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=200, sparse_output=not dense
    )


def build_lr(numeric, categorical, use_text=True, use_controls=True, C=2.0):
    transformers = []
    if use_text:
        transformers.append(
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000,
                                     sublinear_tf=True, stop_words="english"), "review")
        )
    if use_controls:
        transformers.append(("num", StandardScaler(), numeric))
        transformers.append(("cat", make_ohe(dense=False), categorical))
    pre = ColumnTransformer(transformers)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                             solver="liblinear", random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_gbm(numeric, categorical):
    text_branch = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=20000,
                                  sublinear_tf=True, stop_words="english")),
        ("svd", TruncatedSVD(n_components=100, random_state=RANDOM_STATE)),
    ])
    pre = ColumnTransformer([
        ("text", text_branch, "review"),
        ("num", "passthrough", numeric),
        ("cat", make_ohe(dense=True), categorical),
    ])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=300, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# --------------------------------------------------------------------------- #
# Evaluation and interpretation
# --------------------------------------------------------------------------- #
def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    return row


def lr_coefficients(model):
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    names = pre.get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": clf.coef_[0]})
    coefs["abs_coef"] = coefs["coef"].abs()
    return coefs.sort_values("abs_coef", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--make-sample", default=None, help="write a 500-row sample of --train to this path and exit")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=None, help="rows to sample from train for a quick run")
    ap.add_argument("--drop-useful", action="store_true", help="drop usefulCount (possible leakage proxy)")
    args = ap.parse_args()

    if args.make_sample:
        folder = os.path.dirname(args.make_sample)
        if folder:
            os.makedirs(folder, exist_ok=True)
        load_raw(args.train).sample(500, random_state=RANDOM_STATE).to_csv(
            args.make_sample, sep="\t", index=False)
        print(f"Saved 500-row sample to {args.make_sample}")
        return
    if not args.test:
        ap.error("--test is required unless --make-sample is used")
    os.makedirs(args.out, exist_ok=True)

    train = add_features(clean(load_raw(args.train)), args.drop_useful)
    test = add_features(clean(load_raw(args.test)), args.drop_useful)
    if args.sample and args.sample < len(train):
        train = train.sample(args.sample, random_state=RANDOM_STATE)

    numeric, categorical = feature_lists(args.drop_useful)
    cols = ["review"] + numeric + categorical
    X_train, y_train = train[cols], train["target"]
    X_test, y_test = test[cols], test["target"]

    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
    print(f"Dissatisfied share  train: {y_train.mean():.3f}  test: {y_test.mean():.3f}")

    results = []

    # Ablation: what do the control variables add to text?
    ablations = {
        "LR text only": dict(use_text=True, use_controls=False),
        "LR controls only": dict(use_text=False, use_controls=True),
    }
    for name, kw in ablations.items():
        m = build_lr(numeric, categorical, **kw).fit(X_train, y_train)
        results.append(evaluate(name, m, X_test, y_test))
        print(f"done: {name}")

    # Model 1: Logistic Regression, text + controls
    lr = build_lr(numeric, categorical).fit(X_train, y_train)
    results.append(evaluate("Model 1: LR text + controls", lr, X_test, y_test))
    print("done: Model 1")

    # Model 2: Gradient boosting, SVD(text) + controls
    gbm = build_gbm(numeric, categorical).fit(X_train, y_train)
    results.append(evaluate("Model 2: GBM svd-text + controls", gbm, X_test, y_test))
    print("done: Model 2")

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n=== Test-set results ===")
    print(res.round(3).to_string(index=False))

    coefs = lr_coefficients(lr)
    coefs.to_csv(os.path.join(args.out, "lr_coefficients.csv"), index=False)
    text_coefs = coefs[coefs["feature"].str.startswith("text__")]
    ctrl_coefs = coefs[~coefs["feature"].str.startswith("text__")]
    print("\nTop words pushing toward DISSATISFIED:")
    print(text_coefs.sort_values("coef", ascending=False).head(15)[["feature", "coef"]].round(3).to_string(index=False))
    print("\nTop words pushing toward SATISFIED:")
    print(text_coefs.sort_values("coef").head(15)[["feature", "coef"]].round(3).to_string(index=False))
    print("\nLargest control-variable coefficients:")
    print(ctrl_coefs.head(15)[["feature", "coef"]].round(3).to_string(index=False))
    print(f"\nSaved outputs to ./{args.out}/")


if __name__ == "__main__":
    main()
