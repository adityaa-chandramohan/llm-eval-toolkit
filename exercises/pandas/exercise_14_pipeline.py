# EXERCISE 14 — Full Pandas ML Results Analysis Pipeline
#
# This exercise chains groupby, merge, apply, and pivot into a complete
# end-to-end analysis of multi-model evaluation results.
#
# Pipeline:
#   generate_mock_results()
#   → join with ground truth
#   → compute_full_metrics()  (per-model, per-class precision/recall/F1)
#   → find_best_model()
#   → confidence_calibration_analysis()
#   → generate_report()

import pandas as pd
import numpy as np
import json
import math

# ── 1. Data Generation ─────────────────────────────────────────────────────────

def generate_mock_results(n_samples: int = 200, n_models: int = 3,
                          n_classes: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic mock dataset of model predictions.
    Returns a DataFrame with columns:
      sample_id, model, predicted_label, confidence, true_label, difficulty
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    classes = [f"class_{i}" for i in range(n_classes)]
    models = [f"model_{chr(65+i)}" for i in range(n_models)]  # model_A, model_B, ...
    difficulties = ["easy", "medium", "hard"]

    # Each model has a different base accuracy
    model_accuracy = {m: 0.7 + i * 0.05 for i, m in enumerate(models)}

    rows = []
    true_labels = rng.choice(classes, n_samples)
    sample_difficulties = rng.choice(difficulties, n_samples,
                                     p=[0.4, 0.4, 0.2])  # more easy/medium

    for model in models:
        acc = model_accuracy[model]
        for i in range(n_samples):
            true = true_labels[i]
            # Model is correct with probability = acc, wrong otherwise
            if rng.random() < acc:
                pred = true
            else:
                wrong_classes = [c for c in classes if c != true]
                pred = rng.choice(wrong_classes)
            # Confidence: higher when correct
            conf = round(float(rng.beta(8 if pred == true else 3, 2)), 4)
            rows.append({
                "sample_id": f"s{i:04d}",
                "model": model,
                "predicted_label": pred,
                "confidence": conf,
                "true_label": true,
                "difficulty": sample_difficulties[i],
            })

    return pd.DataFrame(rows)


# ── 2. Compute Full Metrics ────────────────────────────────────────────────────

def compute_full_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, class), compute precision, recall, F1 using pandas operations.
    Returns DataFrame with: model, class_label, precision, recall, f1, support.
    Also includes macro-averaged F1 per model.
    """
    classes = df["true_label"].unique()
    models = df["model"].unique()
    rows = []

    for model in models:
        model_df = df[df["model"] == model]
        for cls in classes:
            tp = ((model_df["predicted_label"] == cls) & (model_df["true_label"] == cls)).sum()
            fp = ((model_df["predicted_label"] == cls) & (model_df["true_label"] != cls)).sum()
            fn = ((model_df["predicted_label"] != cls) & (model_df["true_label"] == cls)).sum()
            support = (model_df["true_label"] == cls).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            rows.append({
                "model": model, "class_label": cls,
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "support":   int(support),
            })

    return pd.DataFrame(rows)


# ── 3. Find Best Model ─────────────────────────────────────────────────────────

def find_best_model(df: pd.DataFrame, metric: str = "f1") -> str:
    """
    Return model name with highest macro-averaged metric across all classes.
    df should be the output of compute_full_metrics().
    """
    macro = (
        df.groupby("model")[metric]
        .mean()
        .sort_values(ascending=False)
    )
    return macro.index[0]


# ── 4. Confidence Calibration Analysis ────────────────────────────────────────

def confidence_calibration_analysis(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Bin predictions by confidence decile, compute accuracy in each bin.
    A well-calibrated model has accuracy ≈ mean confidence in each bin.
    Returns DataFrame: bin_label, mean_confidence, accuracy, count, calibration_error.
    """
    df = df.copy()
    df["correct"] = df["predicted_label"] == df["true_label"]
    df["conf_bin"] = pd.cut(df["confidence"], bins=n_bins, labels=False)

    result = (
        df.groupby("conf_bin")
        .agg(
            mean_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            count=("correct", "size"),
        )
        .round(4)
        .reset_index()
    )
    result["calibration_error"] = (result["mean_confidence"] - result["accuracy"]).abs().round(4)
    return result


# ── 5. Generate Report ─────────────────────────────────────────────────────────

def generate_report(df: pd.DataFrame) -> None:
    """Chain all pipeline steps into a printed summary report."""
    print("=" * 60)
    print("  ML EVALUATION REPORT")
    print("=" * 60)

    # Per-model accuracy
    acc = df.groupby("model").apply(
        lambda g: (g["predicted_label"] == g["true_label"]).mean()
    ).rename("accuracy").round(4)
    print("\n[ Accuracy by Model ]")
    print(acc.to_string())

    # Full metrics
    metrics_df = compute_full_metrics(df)

    # Macro F1 per model
    macro_f1 = metrics_df.groupby("model")["f1"].mean().round(4)
    print("\n[ Macro F1 by Model ]")
    print(macro_f1.sort_values(ascending=False).to_string())

    # Best model
    best = find_best_model(metrics_df)
    print(f"\n[ Best Model ] → {best}")

    # Pivot: model × class F1
    f1_pivot = pd.pivot_table(
        metrics_df, index="model", columns="class_label", values="f1"
    ).round(4)
    print("\n[ F1 by Model × Class ]")
    print(f1_pivot.to_string())

    # Calibration for best model
    best_df = df[df["model"] == best]
    calib = confidence_calibration_analysis(best_df)
    print(f"\n[ Calibration Analysis for {best} ]")
    print(calib[["mean_confidence", "accuracy", "count", "calibration_error"]].to_string(index=False))

    # Accuracy by difficulty
    diff_acc = (
        df[df["model"] == best]
        .groupby("difficulty")
        .apply(lambda g: (g["predicted_label"] == g["true_label"]).mean())
        .rename("accuracy")
        .round(4)
    )
    print(f"\n[ {best} Accuracy by Difficulty ]")
    print(diff_acc.to_string())

    print("\n" + "=" * 60)


# ── Demo ───────────────────────────────────────────────────────────────────────

def run_demo():
    df = generate_mock_results()

    print(f"Generated {len(df)} rows × {df['model'].nunique()} models")
    print(df.head(4).to_string(index=False))

    metrics = compute_full_metrics(df)
    best = find_best_model(metrics)
    print(f"\nBest model by macro F1: {best}")

    # ── Assertions ─────────────────────────────────────────────────────────────
    assert len(df) == 200 * 3, "Expected 200 samples × 3 models"
    assert set(df.columns) >= {"sample_id", "model", "predicted_label", "confidence", "true_label"}
    print("✅ generate_mock_results shape correct")

    assert all(0 <= v <= 1 for v in metrics["f1"]), "F1 out of range"
    assert all(0 <= v <= 1 for v in metrics["precision"])
    assert all(0 <= v <= 1 for v in metrics["recall"])
    print("✅ compute_full_metrics values in range")

    assert best in df["model"].unique()
    print("✅ find_best_model returns valid model")

    calib = confidence_calibration_analysis(df[df["model"] == best])
    assert "calibration_error" in calib.columns
    assert (calib["accuracy"] >= 0).all()
    print("✅ confidence_calibration_analysis valid")

    print()
    generate_report(df)

    print("\n🎉 Exercise 14 complete!")


run_demo()
