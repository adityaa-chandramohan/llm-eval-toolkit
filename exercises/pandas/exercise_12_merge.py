# EXERCISE 12 — Pandas Merge & Join on ML Datasets
#
# KEY CONCEPTS:
#   - pd.merge() default is inner join. Use how='left'/'right'/'outer' for other types.
#   - Anti-join (rows in A not in B): left merge + filter where right key is NaN.
#   - Suffixes: when merging on non-key columns with same names, use suffixes=('_a','_b').
#   - Joining on index: use left_index=True / right_index=True.
#   - merge vs join: df.join() is shorthand for merge on index.

import pandas as pd
import numpy as np

# ── Sample Datasets ────────────────────────────────────────────────────────────
np.random.seed(7)
N_SAMPLES = 60
N_MODELS = 2

LABELS = ["positive", "negative", "neutral"]
DIFFICULTIES = ["easy", "medium", "hard"]
SOURCES = ["twitter", "news", "reddit"]

sample_ids = [f"s{i:03d}" for i in range(N_SAMPLES)]

# Ground truth: every sample has a true label and metadata
ground_truth_df = pd.DataFrame({
    "sample_id": sample_ids,
    "true_label": np.random.choice(LABELS, N_SAMPLES),
    "difficulty": np.random.choice(DIFFICULTIES, N_SAMPLES),
    "source": np.random.choice(SOURCES, N_SAMPLES),
})

# Predictions: model A covers all samples, model B misses some
predictions_a = pd.DataFrame({
    "sample_id": sample_ids,
    "model": "model_a",
    "predicted_label": np.random.choice(LABELS, N_SAMPLES),
    "confidence": np.random.uniform(0.5, 1.0, N_SAMPLES).round(3),
    "timestamp": pd.date_range("2024-01-01", periods=N_SAMPLES, freq="1min"),
})

# Model B only covers 45 samples (misses 15)
b_ids = np.random.choice(sample_ids, size=45, replace=False)
predictions_b = pd.DataFrame({
    "sample_id": b_ids,
    "model": "model_b",
    "predicted_label": np.random.choice(LABELS, 45),
    "confidence": np.random.uniform(0.5, 1.0, 45).round(3),
    "timestamp": pd.date_range("2024-01-01", periods=45, freq="2min"),
})

model_metadata = pd.DataFrame({
    "model": ["model_a", "model_b", "model_c"],
    "architecture": ["BERT-base", "RoBERTa-large", "DistilBERT"],
    "parameters_M": [110, 355, 66],
    "training_dataset": ["mixed", "news-only", "mixed"],
})


# ── Functions ──────────────────────────────────────────────────────────────────

def join_predictions_with_truth(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Inner join predictions with ground truth on sample_id, add 'correct' column."""
    merged = pd.merge(preds, truth, on="sample_id", how="inner")
    merged["correct"] = merged["predicted_label"] == merged["true_label"]
    return merged


def find_missing_predictions(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Return ground truth rows for samples NOT covered by predictions (anti-join)."""
    merged = pd.merge(truth, preds[["sample_id"]], on="sample_id", how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    return missing.reset_index(drop=True)


def merge_model_metadata(preds: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with model metadata on 'model' column."""
    return pd.merge(preds, metadata, on="model", how="left")


def multi_model_comparison(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> dict:
    """
    Join two models' predictions side-by-side on sample_id.
    Returns merged DataFrame and agreement rate (fraction of samples where both agree).
    Only considers samples covered by BOTH models (inner join).
    """
    merged = pd.merge(
        preds_a[["sample_id", "predicted_label", "confidence"]],
        preds_b[["sample_id", "predicted_label", "confidence"]],
        on="sample_id",
        suffixes=("_a", "_b"),
    )
    merged["agree"] = merged["predicted_label_a"] == merged["predicted_label_b"]
    agreement_rate = merged["agree"].mean().round(4)
    return {"merged": merged, "agreement_rate": agreement_rate}


def enrich_with_difficulty(preds: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with ground truth, compute accuracy broken down by difficulty."""
    merged = join_predictions_with_truth(preds, truth)
    return (
        merged.groupby("difficulty")["correct"]
        .agg(accuracy="mean", count="size")
        .round({"accuracy": 4})
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo():
    print("=== Join Predictions with Ground Truth ===")
    joined = join_predictions_with_truth(predictions_a, ground_truth_df)
    print(f"Rows: {len(joined)}, Accuracy: {joined['correct'].mean():.4f}")
    print(joined[["sample_id", "predicted_label", "true_label", "correct", "difficulty"]].head(5).to_string(index=False))

    print("\n=== Missing Predictions (model_b) ===")
    missing = find_missing_predictions(predictions_b, ground_truth_df)
    print(f"Model B missing {len(missing)} samples:")
    print(missing[["sample_id", "true_label", "difficulty"]].head(5).to_string(index=False))

    print("\n=== Merge with Model Metadata ===")
    enriched = merge_model_metadata(predictions_a, model_metadata)
    print(enriched[["sample_id", "model", "architecture", "parameters_M"]].head(3).to_string(index=False))

    print("\n=== Model A vs B Comparison ===")
    result = multi_model_comparison(predictions_a, predictions_b)
    print(f"Agreement rate: {result['agreement_rate']}")
    print(f"Compared on {len(result['merged'])} shared samples")

    print("\n=== Accuracy by Difficulty ===")
    print(enrich_with_difficulty(predictions_a, ground_truth_df).to_string(index=False))

    # ── Assertions ─────────────────────────────────────────────────────────────
    assert len(joined) == N_SAMPLES
    assert "correct" in joined.columns
    print("\n✅ inner join correct size")

    assert len(missing) == N_SAMPLES - 45
    print("✅ anti-join finds correct missing count")

    enriched_meta = merge_model_metadata(predictions_a, model_metadata)
    assert "architecture" in enriched_meta.columns
    assert enriched_meta["architecture"].notna().all()
    print("✅ metadata join correct")

    comp = multi_model_comparison(predictions_a, predictions_b)
    assert 0 <= comp["agreement_rate"] <= 1
    print("✅ agreement rate in valid range")

    print("\n🎉 Exercise 12 complete!")


run_demo()
