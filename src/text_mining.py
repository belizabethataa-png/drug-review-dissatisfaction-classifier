"""Keyword frequency, within-group distinctive terms, sentiment, and lexicon features."""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

DOMAIN_STOPWORDS = {
    "drug", "drugx", "condx", "medication", "medicine", "pill", "pills", "mg", "just", "like",
    "im", "ive", "ve", "don", "didn", "doesn", "wasn", "got", "get", "going", "took", "taking",
    "take", "taken", "day", "days", "ve", "ll", "really", "quot", "039",
}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS)


# ---------- frequency ----------
def top_terms(texts, n=20, ngram_range=(1, 1), min_df=5):
    """Most frequent terms (document frequency) after stop-word removal."""
    cv = CountVectorizer(stop_words=STOPWORDS, ngram_range=ngram_range, min_df=min_df)
    X = cv.fit_transform(texts)
    freq = np.asarray((X > 0).sum(axis=0)).ravel()
    s = pd.Series(freq, index=cv.get_feature_names_out()).sort_values(ascending=False)
    return (s / X.shape[0]).head(n)  # share of reviews containing the term


def tfidf_top_terms_by_group(df, group_col, groups, text_col="review_masked", n=15, min_df=5):
    """Top mean-TF-IDF terms fitted separately inside each group (e.g. each condition)."""
    out = {}
    for g in groups:
        texts = df.loc[df[group_col] == g, text_col]
        tv = TfidfVectorizer(stop_words=STOPWORDS, min_df=min(min_df, max(1, len(texts) // 20)))
        X = tv.fit_transform(texts)
        m = np.asarray(X.mean(axis=0)).ravel()
        terms = tv.get_feature_names_out()
        out[g] = list(terms[np.argsort(m)[::-1][:n]])
    return out


def jaccard_matrix(term_lists):
    """Pairwise Jaccard overlap between the top-term sets of each group."""
    keys = list(term_lists)
    M = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            A, B = set(term_lists[a]), set(term_lists[b])
            M.loc[a, b] = len(A & B) / max(1, len(A | B))
    return M


def weighted_log_odds(texts_a, texts_b, prior_texts, alpha0=500.0, min_count=10):
    """Monroe et al. (2008) log-odds ratio with an informative Dirichlet prior (z-scored).
    Positive z = term is distinctive of group A; negative = group B. Using the whole
    condition as the prior keeps rare words from dominating."""
    cv = CountVectorizer(stop_words=STOPWORDS, min_df=3)
    cv.fit(prior_texts)
    ya = np.asarray(cv.transform(texts_a).sum(axis=0)).ravel().astype(float)
    yb = np.asarray(cv.transform(texts_b).sum(axis=0)).ravel().astype(float)
    yp = np.asarray(cv.transform(prior_texts).sum(axis=0)).ravel().astype(float)
    alpha = alpha0 * yp / yp.sum()
    na, nb = ya.sum(), yb.sum()
    la = np.log((ya + alpha) / (na + alpha0 - ya - alpha))
    lb = np.log((yb + alpha) / (nb + alpha0 - yb - alpha))
    z = (la - lb) / np.sqrt(1.0 / (ya + alpha) + 1.0 / (yb + alpha))
    res = pd.DataFrame({"term": cv.get_feature_names_out(), "z": z, "count_a": ya, "count_b": yb})
    return res[(res.count_a + res.count_b) >= min_count].sort_values("z", ascending=False)


def within_condition_distinctive(df, conditions, text_col="review_masked", n=10):
    """For each condition: terms most tied to dissatisfied (z>0) vs satisfied (z<0) reviews."""
    rows = []
    for c in conditions:
        sub = df[df["condition_clean"] == c]
        a = sub.loc[sub.dissatisfied == 1, text_col]
        b = sub.loc[sub.dissatisfied == 0, text_col]
        if len(a) < 30 or len(b) < 30:
            continue
        res = weighted_log_odds(a, b, sub[text_col])
        rows.append({"condition": c,
                     "dissatisfied_terms": ", ".join(res.head(n).term),
                     "satisfied_terms": ", ".join(res.tail(n).term[::-1])})
    return pd.DataFrame(rows)


# ---------- sentiment ----------
def vader_compound(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return pd.Series([sia.polarity_scores(t)["compound"] for t in texts], index=getattr(texts, "index", None))


# ---------- lexicon features (cheap first pass at 'condition vs drug experience') ----------
CONDITION_CUES = r"\b(i have|i had|i was diagnosed|diagnosed|suffer\w*|my symptoms?|history of|for years|since i was|my doctor|my (anxiety|depression|pain|acne|periods?))\b"
DRUG_CUES = r"\b(side effects?|helped|worked|work(s|ing)?|started|stopped|switched|dose|dosage|weeks?|months?|nausea|headaches?|weight gain|gained|insomnia|dizz\w+|rash|refill|expensive|cost|insurance)\b"
DOSE_RE = r"\b\d+(\.\d+)?\s?(mg|mcg|ml|g)\b"
DURATION_RE = r"\b\d+\s?(day|days|week|weeks|month|months|year|years)\b"
SIDE_EFFECT_RE = r"\b(side effects?|nausea|headaches?|dizz\w+|weight gain|gained|insomnia|rash|vomit\w*|drows\w+|fatigue|tired|dry mouth|sweat\w*|anxious|mood swings?|bleeding|spotting|cramps?|libido|constipat\w+|diarrhea)\b"
INEFFECTIVE_RE = r"\b(did not work|didn'?t work|doesn'?t work|no effect|no difference|not working|stopped working|useless|waste)\b"
COST_RE = r"\b(expensive|cost|costs|insurance|copay|price|afford\w*)\b"
STOP_RE = r"\b(stopped|quit|discontinued|switched|came off|went off|taken off|got off)\b"


def _nc(pat):
    """Turn capturing groups into non-capturing ones (avoids pandas regex-group warnings)."""
    return re.sub(r"\((?!\?)", "(?:", pat)


def lexicon_features(texts):
    """Counts of cue words/phrases. Transparent baseline that Claude's extraction will be compared against."""
    t = texts.str.lower()
    return pd.DataFrame({
        "n_side_effect_terms": t.str.count(_nc(SIDE_EFFECT_RE)),
        "mentions_dose": t.str.contains(_nc(DOSE_RE)).astype(int),
        "mentions_duration": t.str.contains(_nc(DURATION_RE)).astype(int),
        "says_ineffective": t.str.contains(_nc(INEFFECTIVE_RE)).astype(int),
        "mentions_cost": t.str.contains(_nc(COST_RE)).astype(int),
        "stopped_or_switched": t.str.contains(_nc(STOP_RE)).astype(int),
    }, index=texts.index)


def tag_sentences(text):
    """Heuristic sentence tags: 'condition' (about the patient's own condition/history),
    'drug' (about the drug experience), 'mixed' (both kinds of cue), or 'other'."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    tags = []
    for s in sents:
        low = s.lower()
        c, d = bool(re.search(CONDITION_CUES, low)), bool(re.search(DRUG_CUES, low))
        tags.append("mixed" if (c and d) else "drug" if d else "condition" if c else "other")
    return tags


def sentence_tag_shares(texts):
    shares = []
    for t in texts:
        tags = tag_sentences(t)
        n = max(1, len(tags))
        shares.append({k: tags.count(k) / n for k in ["condition", "drug", "mixed", "other"]})
    return pd.DataFrame(shares, index=texts.index)
