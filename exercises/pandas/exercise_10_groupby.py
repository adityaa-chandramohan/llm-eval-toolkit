# EXERCISE 10 — Pandas GroupBy on ML Output Datasets
#
# KEY CONCEPTS:
#   - groupby().agg(): apply multiple aggregation functions at once.
#   - groupby().size() vs .count(): size includes NaN rows, count excludes them.
#   - Multi-level groupby: pass a list of column names.
#   - Named aggregation (pandas >= 0.25): agg(new_col=('src_col', 'func')).
#   - Resetting index after groupby: .reset_index() to get a flat DataFrame.

import pandas as pd
import numpy as np

# ── Sample Dataset ─────────────────────────────────────────────────────────────
np.random.seed(42)
N = 90

MODELS = ["bert-v1", "bert-v2", "gpt-small"]
CLASSES = ["positive", "negative", "neutral"]

df = pd.DataFrame({
    "model_name":      np.random.choice(MODELS, N),
    "class_label":     np.random.choice(CLASSES, N),
    "confidence":      np.random.uniform(0.5, 1.0, N).round(3),
})
# Simulate predicted label: mostly correct, sometimes wrong
df["predicted_label"] = df["class_label"].copy()
wrong_idx = np.random.choice(N, size=15, replace=False)
df.loc[wrong_idx, "predicted_label"] = np.random.choice(CLASSES, size=15)
df["correct"] = df["class_label"] == df["predicted_label"]


# ── Functions ──────────────────────────────────────────────────────────────────

def accuracy_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy = mean of correct column, grouped by model."""
    return (
        df.groupby("model_name")["correct"]
        .mean()
        .rename("accuracy")
        .round(4)
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )


def accuracy_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and sample count per true class label."""
    return (
        df.groupby("class_label")
        .agg(
            accuracy=("correct", "mean"),
            count=("correct", "size"),
        )
        .round({"accuracy": 4})
        .reset_index()
    )


def confidence_stats_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, std, min, max of confidence per model."""
    return (
        df.groupby("model_name")["confidence"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .reset_index()
        .rename(columns={"mean": "conf_mean", "std": "conf_std",
                         "min": "conf_min", "max": "conf_max"})
    )


def error_rate_by_model_and_class(df: pd.DataFrame) -> pd.DataFrame:
    """Error rate = 1 - accuracy, grouped by model and class."""
    return (
        df.groupby(["model_name", "class_label"])["correct"]
        .mean()
        .rsub(1)          # 1 - mean
        .rename("error_rate")
        .round(4)
        .reset_index()
        .sort_values("error_rate", ascending=False)
    )


def top_confused_pairs(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find N most common (true_label, predicted_label) pairs where prediction is wrong."""
    errors = df[df["class_label"] != df["predicted_label"]].copy()
    return (
        errors.groupby(["class_label", "predicted_label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .head(n)
    )


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo():
    print("=== Accuracy by Model ===")
    print(accuracy_by_model(df).to_string(index=False))

    print("\n=== Accuracy by Class ===")
    print(accuracy_by_class(df).to_string(index=False))

    print("\n=== Confidence Stats by Model ===")
    print(confidence_stats_by_model(df).to_string(index=False))

    print("\n=== Error Rate by Model × Class ===")
    print(error_rate_by_model_and_class(df).head(6).to_string(index=False))

    print("\n=== Top Confused Pairs ===")
    print(top_confused_pairs(df).to_string(index=False))

    # ── Assertions ─────────────────────────────────────────────────────────────
    acc = accuracy_by_model(df)
    assert set(acc["model_name"]) == set(MODELS), "Missing models"
    assert all(0 <= v <= 1 for v in acc["accuracy"]), "Accuracy out of range"
    print("\n✅ accuracy_by_model structure correct")

    cls_acc = accuracy_by_class(df)
    assert "count" in cls_acc.columns
    assert cls_acc["count"].sum() == N
    print("✅ accuracy_by_class counts sum to N")

    conf = confidence_stats_by_model(df)
    assert all(conf["conf_min"] >= 0.5)
    assert all(conf["conf_max"] <= 1.0)
    print("✅ confidence_stats bounds correct")

    pairs = top_confused_pairs(df)
    assert all(pairs["count"] > 0)
    print("✅ top_confused_pairs all have count > 0")

    print("\n🎉 Exercise 10 complete!")


run_demo()
