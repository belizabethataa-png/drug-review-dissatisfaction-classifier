"""Loading, cleaning, and basic feature construction for the UCI Drug Reviews data."""
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_FILE = "drugsComTrain_raw.tsv"
TEST_FILE = "drugsComTest_raw.tsv"

DISSATISFIED_CUTOFF = 4          # rating <= 4 -> dissatisfied
POST_HOC_COLS = ["usefulCount"]  # only known AFTER posting: never used as a control/feature


def load_reviews(data_dir=DATA_DIR, train_file=TRAIN_FILE, test_file=TEST_FILE):
    """Read the provided train/test TSVs and stack them with a `split` column."""
    frames = []
    for name, split in [(train_file, "train"), (test_file, "test")]:
        path = Path(data_dir) / name
        df = pd.read_csv(path, sep="\t")
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "uniqueID"})
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_review(text):
    """Unescape HTML entities (e.g. &#039;), drop wrapping quotes, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).strip().strip('"')
    return re.sub(r"\s+", " ", text)


def clean_condition(cond):
    """Some `condition` values are scraping artifacts (e.g. '3</span> users found this
    comment helpful.'). Set those to NaN instead of treating them as real conditions."""
    if not isinstance(cond, str) or "</span>" in cond or "users found" in cond:
        return np.nan
    return cond.strip()


def add_basic_features(df):
    """Add cleaned text, target, date parts, and simple style features.
    Style features use the ORIGINAL casing, so they are computed before lowercasing."""
    df = df.copy()
    df["review_clean"] = df["review"].map(clean_review)
    df["condition_clean"] = df["condition"].map(clean_condition)
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df["year"] = df["date"].dt.year
    df["dissatisfied"] = (df["rating"] <= DISSATISFIED_CUTOFF).astype(int)

    txt = df["review_clean"]
    df["n_chars"] = txt.str.len()
    df["n_words"] = txt.str.split().str.len()
    df["n_sentences"] = txt.map(lambda t: max(1, len(re.findall(r"[.!?]+", t))))
    df["exclam_count"] = txt.str.count("!")
    df["question_count"] = txt.str.count(r"\?")
    letters = txt.str.count(r"[A-Za-z]").replace(0, np.nan)
    df["caps_ratio"] = (txt.str.count(r"[A-Z]") / letters).fillna(0)
    return df


def _name_tokens(*names, min_len=4):
    toks = set()
    for n in names:
        if isinstance(n, str):
            toks |= {t.lower() for t in re.split(r"[^A-Za-z0-9]+", n) if len(t) >= min_len}
    return toks


def _sub_tokens(text, toks, placeholder):
    if not toks:
        return text
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(toks))) + r")\w*\b", re.I)
    return text.map(lambda t: pat.sub(placeholder, t))


def mask_drug_and_condition(df, text_col="review_clean", out_col="review_masked"):
    """Replace each review's OWN drug-name and condition words with placeholders so a
    text model cannot just learn 'which drug/condition is this'. Other drugs mentioned
    (e.g. 'switched from Zoloft') are kept on purpose: they carry switching information.
    Regexes are compiled once per (drugName, condition) pair, not once per row."""
    df = df.copy()
    df[out_col] = df[text_col]
    keys = df[["drugName", "condition_clean"]].fillna("")
    for _, idx in keys.groupby(["drugName", "condition_clean"]).groups.items():
        row = df.loc[idx[0]]
        text = df.loc[idx, text_col]
        text = _sub_tokens(text, _name_tokens(row["drugName"]), " drugx ")
        text = _sub_tokens(text, _name_tokens(row["condition_clean"]), " condx ")
        df.loc[idx, out_col] = text
    return df


def top_conditions(df, n=10, min_drugs=3):
    """Conditions with the most reviews that also have >= `min_drugs` distinct drugs,
    so that within-condition drug comparisons are possible."""
    g = df.groupby("condition_clean").agg(n_reviews=("rating", "size"), n_drugs=("drugName", "nunique"))
    return g[g.n_drugs >= min_drugs].sort_values("n_reviews", ascending=False).head(n)
