# EXERCISE 6 — Dicts & Counters
#
# KEY CONCEPTS:
#   - Counter is a subclass of dict — use it for frequency counting in O(n).
#   - Grouping: use a dict where key = canonical form, value = list of members.
#     For anagrams: canonical = tuple(sorted(word)).
#   - Top-K: Counter.most_common(k) is O(n log k) — uses a heap internally.
#   - Frequency-based sort: sort by (-count, value) to get stable ordering.

from typing import List, Dict, Tuple
from collections import Counter
import heapq


# ── 1. Group Anagrams ──────────────────────────────────────────────────────────

def group_anagrams(words: List[str]) -> List[List[str]]:
    """
    Group words that are anagrams of each other.
    Key insight: two words are anagrams iff sorted(word) is identical.
    Using sorted tuple as dict key → O(n * k log k) where k = max word length.
    """
    groups: Dict[tuple, List[str]] = {}
    for word in words:
        key = tuple(sorted(word))
        groups.setdefault(key, []).append(word)
    return list(groups.values())


# ── 2. Top K Frequent Elements ─────────────────────────────────────────────────

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    Return k most frequent elements. Order within result does not matter.
    Approach: Counter + heapq.nlargest → O(n log k).
    """
    count = Counter(nums)
    # nlargest(k, iterable, key) picks k items with highest key values
    return heapq.nlargest(k, count, key=count.get)


def top_k_frequent_bucket(nums: List[int], k: int) -> List[int]:
    """
    Bucket sort approach → O(n).
    Bucket index = frequency. Max frequency = n (all same element).
    Iterate buckets from high to low, collect until we have k elements.
    """
    count = Counter(nums)
    # buckets[i] = list of elements with frequency i
    buckets: List[List[int]] = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)

    result = []
    for freq in range(len(buckets) - 1, 0, -1):
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result
    return result


# ── 3. Word Frequency ──────────────────────────────────────────────────────────

def word_frequency(text: str) -> List[Tuple[str, int]]:
    """
    Tokenize text and count word frequencies.
    Returns list of (word, count) sorted by count descending, then word ascending.
    """
    # Lowercase, split on whitespace, strip punctuation from edges
    import re
    words = re.findall(r"[a-z']+", text.lower())
    count = Counter(words)
    # Sort: primary = -count (descending), secondary = word (ascending)
    return sorted(count.items(), key=lambda x: (-x[1], x[0]))


# ── 4. Find All Duplicates ─────────────────────────────────────────────────────

def find_all_duplicates(nums: List[int]) -> List[int]:
    """
    Find all elements that appear exactly twice. Values in [1, n].
    O(n) time, O(1) extra space (modifies array in-place, then restores).
    Trick: use the sign of nums[abs(num)-1] as a visited marker.
    """
    result = []
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0:
            result.append(abs(num))  # seen twice
        else:
            nums[idx] = -nums[idx]  # mark as seen once

    # Restore array
    for i in range(len(nums)):
        nums[i] = abs(nums[i])

    return sorted(result)


def find_all_duplicates_counter(nums: List[int]) -> List[int]:
    """Simple O(n) space version using Counter."""
    return sorted(num for num, cnt in Counter(nums).items() if cnt == 2)


# ── TESTS ──────────────────────────────────────────────────────────────────────

def run_tests():
    # group_anagrams
    result = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    # Sort inner lists and outer list for comparison
    result_sorted = sorted([sorted(g) for g in result])
    assert result_sorted == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams([""]) == [[""]]
    assert group_anagrams(["a"]) == [["a"]]
    print("✅ group_anagrams passed")

    # top_k_frequent
    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    assert top_k_frequent([1], 1) == [1]
    # Bucket version
    assert sorted(top_k_frequent_bucket([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    assert top_k_frequent_bucket([1], 1) == [1]
    print("✅ top_k_frequent passed")

    # word_frequency
    freq = word_frequency("the cat sat on the mat the cat")
    assert freq[0] == ("the", 3)
    assert freq[1] == ("cat", 2)
    # "mat", "on", "sat" all appear once — sorted alphabetically
    once_words = [w for w, c in freq if c == 1]
    assert once_words == ["mat", "on", "sat"]
    print("✅ word_frequency passed")

    # find_all_duplicates
    assert find_all_duplicates([4, 3, 2, 7, 8, 2, 3, 1]) == [2, 3]
    assert find_all_duplicates([1, 1, 2]) == [1]
    assert find_all_duplicates([1]) == []
    # Counter version
    assert find_all_duplicates_counter([4, 3, 2, 7, 8, 2, 3, 1]) == [2, 3]
    print("✅ find_all_duplicates passed")

    print("\n🎉 All Exercise 6 tests passed!")


run_tests()
