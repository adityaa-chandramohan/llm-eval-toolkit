# Confusion Matrix basics:
#               Predicted Positive  Predicted Negative
# Actual Pos         TP                  FN
# Actual Neg         FP                  TN

# Precision = TP / (TP + FP) → "Of all I predicted positive, how many were actually positive?" (quality of positive predictions)
# Recall = TP / (TP + FN) → "Of all actual positives, how many did I catch?" (coverage)
# F1 = harmonic mean of both — punishes extremes (e.g., 100% precision + 0% recall = F1 of 0, not 50%)

# Why harmonic mean? Regular average of 1.0 and 0.0 = 0.5, which is misleadingly optimistic. Harmonic mean = 0.0, which reflects reality.



# You are given a list of model predictions and ground-truth labels for a binary classifier.
# Implement precision, recall, and F1 score WITHOUT using sklearn or any external library.

# Input:
#   y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
#   y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Expected output:
#   precision: 0.8, recall: 0.8, f1: 0.8

# Then extend your solution to handle multi-class classification using macro and weighted averaging.
# STARTER CODE

from typing import List, Dict

def compute_precision(y_true: List[int], y_pred: List[int], pos_label: int = 1) -> float:
    """Compute precision: TP / (TP + FP)"""
    if not y_true:
        raise ValueError("Input lists cannot be empty")
    
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def compute_recall(y_true: List[int], y_pred: List[int], pos_label: int = 1) -> float:
    """Compute recall: TP / (TP + FN)"""
    if not y_true:
        raise ValueError("Input lists cannot be empty")
    
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)
    
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def compute_f1(precision: float, recall: float) -> float:
    """Compute F1: harmonic mean of precision and recall"""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def classification_report(y_true: List[int], y_pred: List[int]) -> Dict:
    """Return full report for multi-class, macro + weighted avg"""
    if not y_true:
        raise ValueError("Input lists cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    classes = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    report = {}
    
    # Per-class metrics
    for cls in classes:
        p = compute_precision(y_true, y_pred, pos_label=cls)
        r = compute_recall(y_true, y_pred, pos_label=cls)
        f = compute_f1(p, r)
        support = sum(1 for t in y_true if t == cls)
        report[cls] = {"precision": p, "recall": r, "f1": f, "support": support}
    
    # Macro avg: simple average across classes (treats all classes equally)
    macro_p = sum(v["precision"] for v in report.values()) / len(classes)
    macro_r = sum(v["recall"] for v in report.values()) / len(classes)
    macro_f = sum(v["f1"] for v in report.values()) / len(classes)
    
    # Weighted avg: weight by class frequency (better for imbalanced datasets)
    weighted_p = sum(v["precision"] * v["support"] for v in report.values()) / n
    weighted_r = sum(v["recall"] * v["support"] for v in report.values()) / n
    weighted_f = sum(v["f1"] * v["support"] for v in report.values()) / n
    
    report["macro avg"] = {"precision": macro_p, "recall": macro_r, "f1": macro_f}
    report["weighted avg"] = {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f}
    
    return report


# --- TESTS ---
def run_tests():
    # Basic binary test
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    
    p = compute_precision(y_true, y_pred)
    r = compute_recall(y_true, y_pred)
    f = compute_f1(p, r)
    print(f"precision: {p}, recall: {r}, f1: {f}")  # Expected: 0.8, 0.8, 0.8
    assert round(p, 4) == 0.8
    assert round(r, 4) == 0.8
    assert round(f, 4) == 0.8
    print("✅ Basic binary test passed")

    # Perfect predictions
    y_perfect = [1, 0, 1, 0]
    p2 = compute_precision(y_perfect, y_perfect)
    r2 = compute_recall(y_perfect, y_perfect)
    f2 = compute_f1(p2, r2)
    assert p2 == r2 == f2 == 1.0
    print("✅ Perfect predictions test passed")

    # All wrong predictions
    y_true2 = [1, 1, 1, 1]
    y_pred2 = [0, 0, 0, 0]
    p3 = compute_precision(y_true2, y_pred2)
    r3 = compute_recall(y_true2, y_pred2)
    f3 = compute_f1(p3, r3)
    assert p3 == r3 == f3 == 0.0
    print("✅ All wrong predictions test passed")

    # Empty input raises ValueError
    try:
        compute_precision([], [])
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Empty input ValueError test passed")
    except ZeroDivisionError:
        assert False, "Should raise ValueError, not ZeroDivisionError"

    # Multi-class report
    y_true3 = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    y_pred3 = [0, 1, 2, 0, 2, 1, 0, 0, 2]
    rpt = classification_report(y_true3, y_pred3)
    print("\nMulti-class report:")
    for k, v in rpt.items():
        print(f"  {k}: {v}")

    # Imbalanced: macro != weighted
    y_true4 = [0]*90 + [1]*10  # Very imbalanced
    y_pred4 = [0]*90 + [0]*10  # Always predicts 0
    rpt4 = classification_report(y_true4, y_pred4)
    macro_f = rpt4["macro avg"]["f1"]
    weighted_f = rpt4["weighted avg"]["f1"]
    assert macro_f != weighted_f, "Macro and weighted should differ for imbalanced data"
    print(f"\n✅ Imbalanced test: macro_f1={macro_f:.4f}, weighted_f1={weighted_f:.4f} (correctly different)")

    print("\n🎉 All tests passed!")

run_tests()