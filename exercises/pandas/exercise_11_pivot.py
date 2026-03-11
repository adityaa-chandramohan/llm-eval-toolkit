# EXERCISE 11 — Pandas Pivot Tables on ML Evaluation Results
#
# KEY CONCEPTS:
#   - pd.pivot_table(): more powerful than .pivot() — handles duplicates via aggfunc.
#   - groupby + unstack(): equivalent to pivot, but more explicit.
#     df.groupby([row, col])[val].mean().unstack(col)
#   - .rank(): rank values within a Series/DataFrame. ascending=False for "best first".
#   - fill_value in pivot_table: replace NaN with a default (e.g. 0).

import pandas as pd
import numpy as np

# ── Sample Dataset ─────────────────────────────────────────────────────────────
np.random.seed(0)
MODELS = ["bert", "gpt", "roberta", "distilbert"]
DATASETS = ["news", "reviews", "tweets"]
METRICS = ["precision", "recall", "f1"]
FOLDS = [1, 2, 3, 4, 5]

rows = []
for model in MODELS:
    for dataset in DATASETS:
        for fold in FOLDS:
            base = {"bert": 0.78, "gpt": 0.82, "roberta": 0.85, "distilbert": 0.75}[model]
            for metric in METRICS:
                # Slight variation per metric and fold
                val = round(base + np.random.uniform(-0.05, 0.05), 4)
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "metric": metric,
                    "value": val,
                    "fold": fold,
                })

df = pd.DataFrame(rows)


# ── Functions ──────────────────────────────────────────────────────────────────

def pivot_metrics_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rows = model, columns = metric, values = mean across all datasets and folds.
    """
    return pd.pivot_table(
        df,
        index="model",
        columns="metric",
        values="value",
        aggfunc="mean",
    ).round(4)


def pivot_fold_results(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """For one model, pivot: rows=fold, columns=metric, values=mean across datasets."""
    subset = df[df["model"] == model_name]
    return pd.pivot_table(
        subset,
        index="fold",
        columns="metric",
        values="value",
        aggfunc="mean",
    ).round(4)


def heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model × dataset pivot table of mean F1 scores.
    Suitable as input to seaborn.heatmap().
    """
    f1_df = df[df["metric"] == "f1"]
    return pd.pivot_table(
        f1_df,
        index="model",
        columns="dataset",
        values="value",
        aggfunc="mean",
    ).round(4)


def rank_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each metric, rank models from best (1) to worst.
    Returns DataFrame: rows=model, columns=metric, values=rank.
    """
    pivot = pivot_metrics_by_model(df)
    # For each metric column, rank descending (higher value = better = rank 1)
    return pivot.rank(ascending=False).astype(int)


def unstacking_demo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstrate groupby + unstack as equivalent to pivot_table.
    Returns same result as pivot_metrics_by_model.
    """
    return (
        df.groupby(["model", "metric"])["value"]
        .mean()
        .unstack("metric")
        .round(4)
    )


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo():
    print("=== Metrics by Model (pivot) ===")
    p1 = pivot_metrics_by_model(df)
    print(p1)

    print("\n=== Fold Results for 'bert' ===")
    print(pivot_fold_results(df, "bert"))

    print("\n=== F1 Heatmap Data ===")
    print(heatmap_data(df))

    print("\n=== Model Rankings per Metric ===")
    print(rank_models(df))

    print("\n=== Unstack Demo (should match pivot) ===")
    p2 = unstacking_demo(df)
    print(p2)

    # ── Assertions ─────────────────────────────────────────────────────────────
    assert list(p1.columns) == sorted(METRICS), "Columns should be sorted metrics"
    assert set(p1.index) == set(MODELS)
    print("\n✅ pivot shape correct")

    # pivot and unstack should give same result
    pd.testing.assert_frame_equal(p1, p2, check_like=True)
    print("✅ pivot == unstack result")

    heat = heatmap_data(df)
    assert heat.shape == (len(MODELS), len(DATASETS))
    print("✅ heatmap shape correct")

    ranks = rank_models(df)
    # roberta has highest base accuracy, so should rank 1 for all metrics
    assert ranks.loc["roberta"].max() <= 2, "roberta should be near top"
    print("✅ rank_models working")

    print("\n🎉 Exercise 11 complete!")


run_demo()
