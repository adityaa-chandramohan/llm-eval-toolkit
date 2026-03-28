# PROBLEM
# Build a confusion matrix from scratch for both binary and multi-class classification.
# Then write a function that prints a readable ASCII confusion matrix to stdout.

# For binary: return a 2x2 matrix with TP, FP, FN, TN clearly accessible.
# For multi-class: return an NxN matrix where entry [i][j] = number of samples
#   with true label i predicted as label j.

# Bonus: compute per-class accuracy (diagonal / row sum) from the matrix.
# STARTER CODE
# from typing import List, Dict
# import json

# def confusion_matrix(y_true: List, y_pred: List) -> List[List[int]]:
#     """
#     Build NxN confusion matrix.
#     Classes are inferred from sorted(set(y_true + y_pred)).
#     """
#     # Your code here
#     pass

# def print_confusion_matrix(matrix: List[List[int]], labels: List[str]) -> None:
#     """Pretty-print the matrix with label headers"""
#     # Your code here
#     pass

# def per_class_accuracy(matrix: List[List[int]]) -> Dict[int, float]:
#     """Return accuracy per class from confusion matrix diagonal"""
#     # Your code here
#     pass
# HINTS
# 	•	Use a dict to map class labels to indices: {label: idx for idx, label in enumerate(sorted_classes)}
# 	•	Initialize matrix as [[0]*n for _ in range(n)]
# 	•	For ASCII print: calculate max cell width first, then pad all values to that width
# 	•	Per-class accuracy: matrix[i][i] / sum(matrix[i]) for each row i


from typing import List, Dict

def confusion_matrix(y_true: List, y_pred: List) -> tuple[List[List[int]], List]:
    """
    Build NxN confusion matrix.
    Classes are inferred from sorted(set(y_true + y_pred)).
    Returns (matrix, labels) so callers always have the class order.
    """
    if not y_true or not y_pred:
        raise ValueError("Input lists cannot be empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    classes = sorted(set(y_true) | set(y_pred))
    idx = {label: i for i, label in enumerate(classes)}
    n = len(classes)

    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1   # row = true, col = predicted

    return matrix, classes


def print_confusion_matrix(matrix: List[List[int]], labels: List) -> None:
    """Pretty-print the matrix with label headers"""
    labels = [str(l) for l in labels]
    n = len(labels)

    # Width = widest of: all cell values, all label strings
    max_val_width = max(len(str(matrix[i][j])) for i in range(n) for j in range(n))
    max_lbl_width = max(len(l) for l in labels)
    col_w = max(max_val_width, max_lbl_width)

    # Row label column width
    row_lbl_w = max_lbl_width + 2

    # Header
    header_pad = " " * row_lbl_w
    header = header_pad + "  ".join(l.center(col_w) for l in labels)
    separator = "-" * len(header)

    print("\nPredicted →")
    print(header)
    print(separator)

    for i, row_label in enumerate(labels):
        prefix = f"{row_label:<{row_lbl_w}}"
        cells = "  ".join(str(matrix[i][j]).center(col_w) for j in range(n))
        print(f"{prefix}{cells}")

    print(separator)
    print("↑ Actual\n")


def per_class_accuracy(matrix: List[List[int]], labels: List = None) -> Dict:
    """Return accuracy per class: diagonal[i] / row_sum[i]"""
    n = len(matrix)
    result = {}
    for i in range(n):
        row_sum = sum(matrix[i])
        key = labels[i] if labels else i
        result[key] = round(matrix[i][i] / row_sum, 4) if row_sum > 0 else 0.0
    return result


# ── Binary helpers (TP/FP/FN/TN) ──────────────────────────────────────────────

def binary_components(matrix: List[List[int]]) -> Dict[str, int]:
    """Extract TP, FP, FN, TN from a 2x2 binary confusion matrix."""
    if len(matrix) != 2:
        raise ValueError("binary_components requires a 2x2 matrix")
    # Convention: label 0 = negative, label 1 = positive
    TN, FP = matrix[0][0], matrix[0][1]
    FN, TP = matrix[1][0], matrix[1][1]
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


# ── TESTS ──────────────────────────────────────────────────────────────────────

def run_tests():
    print("=" * 50)
    print("TEST 1: Binary classification")
    print("=" * 50)
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    mat, labels = confusion_matrix(y_true, y_pred)
    print_confusion_matrix(mat, labels)

    comps = binary_components(mat)
    print(f"TP={comps['TP']} FP={comps['FP']} FN={comps['FN']} TN={comps['TN']}")
    assert comps == {"TP": 4, "FP": 1, "FN": 1, "TN": 4}, f"Got {comps}"
    print("✅ Binary components correct\n")

    acc = per_class_accuracy(mat, labels)
    print(f"Per-class accuracy: {acc}")
    # Class 0: TN/(TN+FP) = 4/5 = 0.8
    # Class 1: TP/(TP+FN) = 4/5 = 0.8
    assert acc[0] == 0.8 and acc[1] == 0.8
    print("✅ Per-class accuracy correct\n")

    print("=" * 50)
    print("TEST 2: Multi-class (3 classes)")
    print("=" * 50)
    y_true2 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
    y_pred2 = [0, 1, 2, 0, 2, 1, 0, 0, 2, 1]
    mat2, labels2 = confusion_matrix(y_true2, y_pred2)
    print_confusion_matrix(mat2, labels2)

    acc2 = per_class_accuracy(mat2, labels2)
    print(f"Per-class accuracy: {acc2}")
    # Class 0: 3/3=1.0, Class 1: 2/4=0.5, Class 2: 2/3=0.667
    assert acc2[0] == 1.0
    assert acc2[1] == 0.5
    assert acc2[2] == round(2/3, 4)
    print("✅ Multi-class accuracy correct\n")

    print("=" * 50)
    print("TEST 3: String labels")
    print("=" * 50)
    y_true3 = ["cat", "dog", "cat", "bird", "dog"]
    y_pred3 = ["cat", "cat", "cat", "bird", "dog"]
    mat3, labels3 = confusion_matrix(y_true3, y_pred3)
    print_confusion_matrix(mat3, labels3)
    print("✅ String labels work\n")

    print("=" * 50)
    print("TEST 4: Perfect predictions")
    print("=" * 50)
    y_p = [0, 1, 2]
    mat4, labels4 = confusion_matrix(y_p, y_p)
    acc4 = per_class_accuracy(mat4, labels4)
    assert all(v == 1.0 for v in acc4.values())
    print("✅ Perfect predictions = 1.0 accuracy for all classes\n")

    print("=" * 50)
    print("TEST 5: Empty input raises ValueError")
    print("=" * 50)
    try:
        confusion_matrix([], [])
        assert False, "Should have raised"
    except ValueError:
        print("✅ Empty input raises ValueError\n")

    print("🎉 All tests passed!")


run_tests()


# **Output you'll see:**
# ```
# TEST 1: Binary classification

# Predicted →
#    0    1
# -----------
# 0  4    1
# 1  1    4
# -----------
# ↑ Actual

# TP=4 FP=1 FN=1 TN=4
# Per-class accuracy: {0: 0.8, 1: 0.8}
# ```

# ---

# ## 🧠 Key Concepts to Know Cold

# **Why row = actual, col = predicted?**
# Standard convention. Reading across a row tells you "for everything that *was* class X, where did predictions land?" A perfect model has all values on the diagonal.

# **Binary TP/FP/FN/TN mapping:**
# ```
#               Predicted 0   Predicted 1
# Actual 0   [    TN      ,      FP     ]
# Actual 1   [    FN      ,      TP     ]