# ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is used to evaluate
# summarization quality. Implement ROUGE-1, ROUGE-2, and ROUGE-L.

# ROUGE-N: n-gram recall, precision, and F1 between candidate and reference.
# ROUGE-L: based on Longest Common Subsequence (LCS) — captures sentence-level structure.

# Inputs:
#   candidate = 'the cat was found under the bed'
#   reference = 'the cat was under the bed'

# Return a dict: {'rouge-1': {'p':..,'r':..,'f':..}, 'rouge-2': {...}, 'rouge-l': {...}}

# Then build a rouge_summary() that aggregates scores across a corpus.
# STARTER CODE
# from typing import List, Dict
# from collections import Counter



from typing import List, Dict
from collections import Counter


def lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of Longest Common Subsequence using dynamic programming."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a token list."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate: str, reference: str, n: int) -> Dict[str, float]:
    """
    Compute ROUGE-N precision, recall, and F1.

    Precision = matched n-grams / candidate n-grams
    Recall    = matched n-grams / reference n-grams
    F1        = harmonic mean of P and R
    """
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()

    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)

    # Clipped overlap: count each n-gram match up to reference frequency
    overlap = sum((cand_ngrams & ref_ngrams).values())

    cand_count = sum(cand_ngrams.values())
    ref_count = sum(ref_ngrams.values())

    precision = overlap / cand_count if cand_count > 0 else 0.0
    recall    = overlap / ref_count  if ref_count  > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"p": round(precision, 4), "r": round(recall, 4), "f": round(f1, 4)}


def rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-L using Longest Common Subsequence.

    Precision = LCS / len(candidate)
    Recall    = LCS / len(reference)
    F1        = harmonic mean of P and R
    """
    cand_tokens = candidate.lower().split()
    ref_tokens  = reference.lower().split()

    lcs = lcs_length(cand_tokens, ref_tokens)

    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall    = lcs / len(ref_tokens)  if ref_tokens  else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"p": round(precision, 4), "r": round(recall, 4), "f": round(f1, 4)}


def rouge_score(candidate: str, reference: str) -> Dict[str, Dict[str, float]]:
    """Return ROUGE-1, ROUGE-2, and ROUGE-L scores for a single pair."""
    return {
        "rouge-1": rouge_n(candidate, reference, 1),
        "rouge-2": rouge_n(candidate, reference, 2),
        "rouge-l": rouge_l(candidate, reference),
    }


def rouge_summary(pairs: List[tuple]) -> Dict[str, Dict[str, float]]:
    """
    Average ROUGE scores across a corpus of (candidate, reference) pairs.

    Args:
        pairs: List of (candidate, reference) string tuples

    Returns:
        Macro-averaged ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    if not pairs:
        return {}

    totals = {
        "rouge-1": {"p": 0.0, "r": 0.0, "f": 0.0},
        "rouge-2": {"p": 0.0, "r": 0.0, "f": 0.0},
        "rouge-l": {"p": 0.0, "r": 0.0, "f": 0.0},
    }

    for candidate, reference in pairs:
        scores = rouge_score(candidate, reference)
        for metric, vals in scores.items():
            for k, v in vals.items():
                totals[metric][k] += v

    n = len(pairs)
    return {
        metric: {k: round(v / n, 4) for k, v in vals.items()}
        for metric, vals in totals.items()
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    candidate = "the cat was found under the bed"
    reference = "the cat was under the bed"

    print("=== Single Pair ===")
    scores = rouge_score(candidate, reference)
    for metric, vals in scores.items():
        print(f"{metric}: P={vals['p']}  R={vals['r']}  F1={vals['f']}")

    print("\n=== Corpus Summary ===")
    corpus = [
        ("the cat was found under the bed", "the cat was under the bed"),
        ("a dog ran fast in the park",       "the dog ran quickly through the park"),
        ("she sells seashells",              "she sells seashells by the seashore"),
    ]
    summary = rouge_summary(corpus)
    for metric, vals in summary.items():
        print(f"{metric}: P={vals['p']}  R={vals['r']}  F1={vals['f']}")