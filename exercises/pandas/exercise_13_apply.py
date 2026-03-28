# EXERCISE 13 — Pandas Apply, Map & Transform on NLP Model Outputs
#
# KEY CONCEPTS:
#   - df["col"].apply(func): apply func element-wise to a Series. Returns Series.
#   - df.apply(func, axis=1): apply func to each ROW. func receives a Series (the row).
#     Return a scalar → new column. Return a Series → expands to new columns.
#   - df["col"].map(dict_or_func): element-wise mapping. Faster than apply for simple lookups.
#   - df.apply(func, axis=1, result_type='expand'): when func returns a dict or Series,
#     automatically expands into multiple columns.
#   - Entropy: measure of prediction uncertainty. High entropy = model is unsure.
#     H = -sum(p * log(p)) for each class probability p.

import pandas as pd
import numpy as np
import json
import math
from typing import Dict

# ── Sample Dataset ─────────────────────────────────────────────────────────────
np.random.seed(1)
N = 40
LABELS = ["positive", "negative", "neutral"]

def _make_scores(true_label: str) -> str:
    """Generate realistic softmax scores as a JSON string."""
    scores = np.random.dirichlet(alpha=[3, 1, 1] if true_label == "positive"
                                 else [1, 3, 1] if true_label == "negative"
                                 else [1, 1, 3])
    return json.dumps({l: round(float(s), 4) for l, s in zip(LABELS, scores)})

true_labels = np.random.choice(LABELS, N)
df = pd.DataFrame({
    "text": [f"Sample text number {i} about {'good' if l=='positive' else 'bad' if l=='negative' else 'neutral'} things."
             for i, l in enumerate(true_labels)],
    "true_label": true_labels,
    "raw_scores": [_make_scores(l) for l in true_labels],
    "token_count": np.random.randint(5, 50, N),
})
# Predicted label = argmax of scores
df["predicted_label"] = df["raw_scores"].apply(
    lambda s: max(json.loads(s), key=json.loads(s).get)
)


# ── Functions ──────────────────────────────────────────────────────────────────

def parse_raw_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'raw_scores' JSON string column into separate score columns.
    Adds: score_positive, score_negative, score_neutral.
    """
    scores_df = df["raw_scores"].apply(json.loads).apply(pd.Series)
    scores_df.columns = [f"score_{c}" for c in scores_df.columns]
    return pd.concat([df, scores_df], axis=1)


def compute_entropy(df: pd.DataFrame) -> pd.Series:
    """
    Compute prediction entropy for each row: H = -sum(p * log2(p)).
    High entropy (close to log2(num_classes)) = uncertain prediction.
    Applied row-wise after parsing scores.
    """
    def row_entropy(scores_json: str) -> float:
        scores = json.loads(scores_json)
        return round(-sum(p * math.log2(p) for p in scores.values() if p > 0), 4)

    return df["raw_scores"].apply(row_entropy).rename("entropy")


def classify_confidence(df: pd.DataFrame) -> pd.Series:
    """
    Map max score to confidence category using .map() with a custom function.
    - 'high':   max score > 0.9
    - 'medium': max score 0.5–0.9
    - 'low':    max score < 0.5
    """
    def max_score(scores_json: str) -> float:
        return max(json.loads(scores_json).values())

    max_scores = df["raw_scores"].apply(max_score)
    return max_scores.map(
        lambda s: "high" if s > 0.9 else "medium" if s >= 0.5 else "low"
    ).rename("confidence_category")


def normalize_text(df: pd.DataFrame) -> pd.Series:
    """Normalize text: lowercase, strip whitespace, collapse multiple spaces."""
    import re
    return (
        df["text"]
        .str.lower()
        .str.strip()
        .apply(lambda t: re.sub(r"\s+", " ", t))
        .rename("normalized_text")
    )


def compute_per_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a function that returns multiple values per row.
    Adds columns: max_score, score_gap (top-2 difference), is_correct.
    """
    def row_metrics(row) -> Dict:
        scores = json.loads(row["raw_scores"])
        sorted_scores = sorted(scores.values(), reverse=True)
        return {
            "max_score": round(sorted_scores[0], 4),
            "score_gap": round(sorted_scores[0] - sorted_scores[1], 4),
            "is_correct": row["predicted_label"] == row["true_label"],
        }

    # result_type='expand' turns returned dict into separate columns
    new_cols = df.apply(row_metrics, axis=1, result_type="expand")
    return pd.concat([df, new_cols], axis=1)


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo():
    print("=== Parse Raw Scores ===")
    parsed = parse_raw_scores(df)
    score_cols = [c for c in parsed.columns if c.startswith("score_")]
    print(parsed[["predicted_label", "true_label"] + score_cols].head(4).to_string(index=False))

    print("\n=== Entropy ===")
    ent = compute_entropy(df)
    print(f"Mean entropy: {ent.mean():.4f}, Max: {ent.max():.4f}")
    print(ent.head(5).to_string())

    print("\n=== Confidence Categories ===")
    cats = classify_confidence(df)
    print(cats.value_counts().to_string())

    print("\n=== Normalized Text (first 3) ===")
    print(normalize_text(df).head(3).to_string())

    print("\n=== Per-Row Metrics ===")
    enriched = compute_per_row_metrics(df)
    print(enriched[["predicted_label", "true_label", "max_score", "score_gap", "is_correct"]].head(5).to_string(index=False))

    # ── Assertions ─────────────────────────────────────────────────────────────
    parsed = parse_raw_scores(df)
    assert "score_positive" in parsed.columns
    assert "score_negative" in parsed.columns
    score_sum = parsed[["score_positive", "score_negative", "score_neutral"]].sum(axis=1)
    assert (score_sum.round(3) == 1.0).all(), "Scores should sum to 1"
    print("\n✅ parse_raw_scores correct")

    ent = compute_entropy(df)
    max_entropy = math.log2(3)  # log2(num_classes)
    assert (ent >= 0).all() and (ent <= max_entropy + 0.01).all()
    print("✅ entropy in valid range [0, log2(3)]")

    cats = classify_confidence(df)
    assert set(cats.unique()).issubset({"high", "medium", "low"})
    print("✅ confidence categories valid")

    enriched = compute_per_row_metrics(df)
    assert "max_score" in enriched.columns
    assert "score_gap" in enriched.columns
    assert enriched["score_gap"].min() >= 0
    print("✅ per_row_metrics correct")

    print("\n🎉 Exercise 13 complete!")


run_demo()
