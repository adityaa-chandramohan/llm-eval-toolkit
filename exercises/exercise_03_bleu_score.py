# PROBLEM
# BLEU (Bilingual Evaluation Understudy) measures text generation quality by comparing
# n-gram overlap between a candidate and one or more reference texts.

# Implement BLEU-1 through BLEU-4 with brevity penalty from scratch.

# Inputs:
#   candidate = 'the cat sat on the mat'
#   references = ['the cat is on the mat', 'there is a cat on the mat']

# Steps:
#   1. Count n-gram matches (clipped by max count in any reference)
#   2. Compute modified n-gram precision for n=1,2,3,4
#   3. Apply brevity penalty: BP = 1 if len(cand) > len(ref), else exp(1 - r/c)
#   4. BLEU = BP * exp(sum of weighted log precisions)

# Then write a batch_bleu() that scores a list of (candidate, references) pairs.
# STARTER CODE
# from typing import List, Tuple
# from collections import Counter
# import math

# def get_ngrams(tokens: List[str], n: int) -> Counter:
#     """Extract n-grams from token list as a Counter"""
#     # Your code here
#     pass

# def clipped_precision(candidate: str, references: List[str], n: int) -> float:
#     """Compute modified n-gram precision with clipping"""
#     # Your code here
#     pass

# def brevity_penalty(candidate: str, references: List[str]) -> float:
#     """Compute BP using closest reference length"""
#     # Your code here
#     pass

# def bleu_score(candidate: str, references: List[str], max_n: int = 4) -> float:
#     """Compute BLEU-N score (default BLEU-4)"""
#     # Your code here
#     pass

# def batch_bleu(pairs: List[Tuple[str, List[str]]]) -> List[float]:
#     """Score a list of (candidate, references) pairs"""
#     # Your code here
#     pass


from typing import List, Tuple
from collections import Counter
import math


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from token list as a Counter"""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def clipped_precision(candidate: str, references: List[str], n: int) -> float:
    """
    Compute modified n-gram precision with clipping.
    Clipping: each n-gram count in candidate is capped by the MAX count
    of that n-gram across ANY reference (prevents gaming with repetition).
    """
    cand_tokens = candidate.split()
    cand_ngrams = get_ngrams(cand_tokens, n)

    if not cand_ngrams:
        return 0.0

    # For each n-gram in candidate, find its max count across all references
    clipped_count = 0
    for ngram, cand_count in cand_ngrams.items():
        max_ref_count = max(
            get_ngrams(ref.split(), n)[ngram]
            for ref in references
        )
        clipped_count += min(cand_count, max_ref_count)

    # Denominator = total n-grams in candidate
    total_cand = sum(cand_ngrams.values())
    return clipped_count / total_cand


def brevity_penalty(candidate: str, references: List[str]) -> float:
    """
    Compute BP using the closest reference length.
    If multiple refs are equidistant, prefer the shorter one.
    BP = 1              if c > r
       = exp(1 - r/c)   if c <= r
    """
    c = len(candidate.split())
    ref_lengths = [len(ref.split()) for ref in references]

    # Pick the reference length closest to candidate length
    r = min(ref_lengths, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c >= r:
        return 1.0
    return math.exp(1 - r / c)


def bleu_score(candidate: str, references: List[str], max_n: int = 4) -> float:
    """
    Compute BLEU-N score (default BLEU-4).
    BLEU = BP * exp( sum_n [ w_n * log(p_n) ] )
    Weights are uniform: w_n = 1/max_n for each n.
    """
    bp = brevity_penalty(candidate, references)

    weight = 1.0 / max_n
    log_avg = 0.0

    for n in range(1, max_n + 1):
        p_n = clipped_precision(candidate, references, n)
        if p_n == 0:
            # Any zero precision collapses the whole score to 0
            return 0.0
        log_avg += weight * math.log(p_n)

    return bp * math.exp(log_avg)


def batch_bleu(pairs: List[Tuple[str, List[str]]]) -> List[float]:
    """Score a list of (candidate, references) pairs"""
    return [bleu_score(candidate, references) for candidate, references in pairs]


# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    candidate   = "the cat sat on the mat"
    references  = ["the cat is on the mat", "there is a cat on the mat"]

    print("=" * 55)
    print(f"  Candidate : {candidate!r}")
    print(f"  Ref 1     : {references[0]!r}")
    print(f"  Ref 2     : {references[1]!r}")
    print("=" * 55)

    for n in range(1, 5):
        p = clipped_precision(candidate, references, n)
        print(f"  BLEU-{n} precision (p_{n})  : {p:.4f}")

    bp = brevity_penalty(candidate, references)
    print(f"\n  Brevity Penalty (BP)    : {bp:.4f}  (c={len(candidate.split())}, r=6)")

    for n in range(1, 5):
        score = bleu_score(candidate, references, max_n=n)
        print(f"  BLEU-{n} score           : {score:.4f}")

    print("\n  Batch BLEU demo:")
    pairs = [
        ("the cat sat on the mat",        ["the cat is on the mat", "there is a cat on the mat"]),
        ("a dog runs in the park",         ["a dog is running in the park"]),
        ("hello world",                    ["hello world"]),  # perfect match
        ("completely wrong sentence here", ["the cat is on the mat"]),
    ]
    scores = batch_bleu(pairs)
    for (cand, _), score in zip(pairs, scores):
        print(f"    {cand!r:<40} → {score:.4f}")
