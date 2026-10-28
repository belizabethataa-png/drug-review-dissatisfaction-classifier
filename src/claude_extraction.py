"""Claude-based extraction of WHY a patient is (dis)satisfied, kept separate from rating.

Design choices (see notebook Phase 3 plan):
 - The rating is NOT shown to Claude, so extracted reasons are independent of the label.
 - Output is strict JSON so it can become model features.
 - Only a small, sampled set of reviews is sent in Phase 2/3 (token cost + privacy).
Requires: `pip install anthropic` and ANTHROPIC_API_KEY in the environment (or a .env that is gitignored).
"""
import json
import os
from pathlib import Path

MODEL = "claude-haiku-5-5"   # small/cheap model is enough for extraction; swap if the class key limits models

SYSTEM_PROMPT = """You analyze patient drug reviews. Separate what the patient says about their OWN CONDITION
from what they say about the DRUG. Respond with ONLY a JSON object, no prose, in this schema:
{
  "condition_description": {"mentioned": true|false, "summary": "<=15 words or null"},
  "drug_experience": {
    "effectiveness": "worked" | "partial" | "did_not_work" | "not_stated",
    "side_effects": ["short phrase", ...],
    "dosage_or_duration": "short phrase or null",
    "cost_or_access_issue": true|false,
    "stopped_or_switched": true|false
  },
  "main_reason_for_dissatisfaction": "side_effects" | "ineffective" | "cost_access" | "condition_worsened" | "other" | "none_expressed"
}
Use only what the review states. Do not infer a rating."""


def build_user_message(review, drug, condition):
    return f"Drug: {drug}\nCondition: {condition}\nReview: {review}"


def extract_reasons(review, drug, condition, client, model=MODEL):
    """One review -> parsed dict (or {'error': ...} if the reply is not valid JSON)."""
    msg = client.messages.create(
        model=model, max_tokens=400, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(review, drug, condition)}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw": text[:300]}


def run_sample(df, n=20, out_path="results/claude_sample.jsonl", seed=42):
    """Stratified sample (half dissatisfied, half not) -> JSONL. Run by hand, review the output."""
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    half = n // 2
    import pandas as pd
    sample = pd.concat([df[df.dissatisfied == 1].sample(half, random_state=seed),
                        df[df.dissatisfied == 0].sample(half, random_state=seed)])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for _, r in sample.iterrows():
            res = extract_reasons(r.review_clean, r.drugName, r.condition_clean, client)
            f.write(json.dumps({"uniqueID": int(r.uniqueID), "rating": int(r.rating), "result": res}) + "\n")
    return out_path
