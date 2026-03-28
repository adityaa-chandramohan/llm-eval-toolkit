# EXERCISE 7 — Sets
#
# KEY CONCEPTS:
#   - Set membership O(1) average vs list O(n). Convert list to set before lookups.
#   - Longest consecutive sequence: only start counting from sequence roots
#     (numbers where n-1 is NOT in the set). This ensures each number is
#     visited at most twice → O(n) total.
#   - Set intersection/union: Python set operators work in O(min(len(a), len(b))).
#   - Order-preserving deduplication: iterate the original list, add to seen set,
#     skip if already in seen. O(n) time and space.

from typing import List, Set, Any


# ── 1. Longest Consecutive Sequence ───────────────────────────────────────────

def longest_consecutive_sequence(nums: List[int]) -> int:
    """
    Find the length of the longest consecutive integer sequence.
    O(n) time: only begin counting at sequence starts (n-1 not in set).
    """
    num_set: Set[int] = set(nums)
    best = 0

    for n in num_set:
        if n - 1 not in num_set:  # n is a sequence start
            length = 1
            while n + length in num_set:
                length += 1
            best = max(best, length)

    return best


# ── 2. Find Missing Ranges ─────────────────────────────────────────────────────

def find_missing_ranges(nums: List[int], lo: int, hi: int) -> List[int]:
    """
    Return all integers in [lo, hi] that are NOT in nums.
    O(hi - lo) in worst case, but typically O(n + range_size).
    """
    present: Set[int] = set(nums)
    return [x for x in range(lo, hi + 1) if x not in present]


def find_missing_ranges_formatted(nums: List[int], lo: int, hi: int) -> List[str]:
    """
    Return missing ranges as strings: single missing = "n", range = "a->b".
    Useful for summarizing sparse missing data.
    """
    missing = find_missing_ranges(nums, lo, hi)
    if not missing:
        return []

    result = []
    start = missing[0]
    end = missing[0]

    for i in range(1, len(missing)):
        if missing[i] == end + 1:
            end = missing[i]
        else:
            result.append(str(start) if start == end else f"{start}->{end}")
            start = end = missing[i]
    result.append(str(start) if start == end else f"{start}->{end}")
    return result


# ── 3. Common Elements Across Multiple Lists ───────────────────────────────────

def common_elements(list_of_lists: List[List[Any]]) -> List[Any]:
    """
    Return elements present in ALL lists, preserving the order they appear
    in the first list.
    Converts all but the first list to sets for O(1) membership testing.
    """
    if not list_of_lists:
        return []

    first = list_of_lists[0]
    rest_sets = [set(lst) for lst in list_of_lists[1:]]

    seen = set()
    result = []
    for item in first:
        if item in seen:
            continue
        if all(item in s for s in rest_sets):
            result.append(item)
            seen.add(item)
    return result


# ── 4. Deduplicate Preserving Order ───────────────────────────────────────────

def deduplicate_preserving_order(items: List[Any]) -> List[Any]:
    """
    Remove duplicates but keep the FIRST occurrence of each item.
    O(n) time using a set to track what's been seen.
    Note: items must be hashable (ints, strings, tuples — not lists/dicts).
    """
    seen: Set[Any] = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


# ── TESTS ──────────────────────────────────────────────────────────────────────

def run_tests():
    # longest_consecutive_sequence
    assert longest_consecutive_sequence([100, 4, 200, 1, 3, 2]) == 4  # [1,2,3,4]
    assert longest_consecutive_sequence([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9  # [0..8]
    assert longest_consecutive_sequence([]) == 0
    assert longest_consecutive_sequence([5]) == 1
    assert longest_consecutive_sequence([1, 1, 1]) == 1
    print("✅ longest_consecutive_sequence passed")

    # find_missing_ranges
    assert find_missing_ranges([0, 1, 3, 50, 75], 0, 99) == (
        list(range(2, 3)) + list(range(4, 50)) + list(range(51, 75)) + list(range(76, 100))
    )
    assert find_missing_ranges([1, 2, 3], 1, 3) == []
    assert find_missing_ranges([], 1, 5) == [1, 2, 3, 4, 5]
    print("✅ find_missing_ranges passed")

    # find_missing_ranges_formatted
    assert find_missing_ranges_formatted([0, 1, 3, 50, 75], 0, 99) == ["2", "4->49", "51->74", "76->99"]
    assert find_missing_ranges_formatted([1, 2, 3], 1, 3) == []
    assert find_missing_ranges_formatted([1, 3], 1, 3) == ["2"]
    print("✅ find_missing_ranges_formatted passed")

    # common_elements
    assert common_elements([[1, 2, 3, 4], [2, 4, 6], [2, 4, 8]]) == [2, 4]
    assert common_elements([[1, 2, 3], [4, 5, 6]]) == []
    assert common_elements([[1, 2, 2, 3], [2, 3]]) == [2, 3]  # no duplicate 2 in output
    assert common_elements([]) == []
    assert common_elements([[1, 2, 3]]) == [1, 2, 3]
    print("✅ common_elements passed")

    # deduplicate_preserving_order
    assert deduplicate_preserving_order([1, 2, 3, 2, 1, 4]) == [1, 2, 3, 4]
    assert deduplicate_preserving_order([]) == []
    assert deduplicate_preserving_order([1]) == [1]
    assert deduplicate_preserving_order(["a", "b", "a", "c"]) == ["a", "b", "c"]
    print("✅ deduplicate_preserving_order passed")

    print("\n🎉 All Exercise 7 tests passed!")


run_tests()
