# EXERCISE 8 — Sorting
#
# KEY CONCEPTS:
#   - Custom sort key: Python's sort is stable and uses Timsort O(n log n).
#     Use key= with a tuple for multi-level sorting: key=lambda x: (-freq[x], x).
#   - Dutch National Flag (3-way partition): three pointers lo/mid/hi.
#     Invariant: [0..lo-1]=0, [lo..mid-1]=1, [mid..hi]=unsorted, [hi+1..n-1]=2.
#     Single pass O(n), O(1) space.
#   - Merge k sorted arrays: use a min-heap. Push (value, array_idx, element_idx).
#     Each pop gives the global minimum; push next element from same array. O(n log k).
#   - Wiggle sort: greedy single pass. Swap adjacent pairs that violate alternating property.

from typing import List
import heapq
from collections import Counter


# ── 1. Sort by Frequency ───────────────────────────────────────────────────────

def sort_by_frequency(nums: List[int]) -> List[int]:
    """
    Sort numbers by frequency descending. Break ties by value ascending.
    Returns a NEW list (does not modify input).
    """
    freq = Counter(nums)
    return sorted(nums, key=lambda x: (-freq[x], x))


# ── 2. Dutch National Flag ─────────────────────────────────────────────────────

def dutch_national_flag(nums: List[int]) -> None:
    """
    Sort array containing only 0s, 1s, 2s in-place in a single pass.
    Three-pointer approach:
      - lo: next position for 0
      - hi: next position for 2 (from right)
      - mid: current element under inspection
    When mid==0: swap with lo, advance both lo and mid.
    When mid==2: swap with hi, retreat hi only (mid stays to re-examine swapped val).
    When mid==1: just advance mid.
    """
    lo, mid, hi = 0, 0, len(nums) - 1

    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo += 1
            mid += 1
        elif nums[mid] == 2:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1
            # Don't advance mid — the swapped element needs inspection
        else:  # nums[mid] == 1
            mid += 1


# ── 3. Merge K Sorted Arrays ───────────────────────────────────────────────────

def merge_sorted_arrays(arrays: List[List[int]]) -> List[int]:
    """
    Merge k sorted arrays into one sorted list.
    Min-heap contains (value, array_index, element_index).
    Each extraction is O(log k), total n elements → O(n log k).
    """
    result = []
    heap = []

    # Initialize heap with first element of each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))

    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        # Push next element from the same array
        next_idx = elem_idx + 1
        if next_idx < len(arrays[arr_idx]):
            heapq.heappush(heap, (arrays[arr_idx][next_idx], arr_idx, next_idx))

    return result


# ── 4. Wiggle Sort ─────────────────────────────────────────────────────────────

def wiggle_sort(nums: List[int]) -> None:
    """
    Rearrange nums IN-PLACE so nums[0] < nums[1] > nums[2] < nums[3] ...
    Greedy single-pass: ensure each adjacent pair satisfies the pattern.
    At even index i: nums[i] should be < nums[i+1]. If not, swap.
    At odd index i:  nums[i] should be > nums[i+1]. If not, swap.
    Proof: each swap only fixes the local violation without breaking previous pairs
    because the value moved away is at most equal to its neighbor.
    O(n) time, O(1) space.
    """
    for i in range(len(nums) - 1):
        if (i % 2 == 0 and nums[i] > nums[i + 1]) or \
           (i % 2 == 1 and nums[i] < nums[i + 1]):
            nums[i], nums[i + 1] = nums[i + 1], nums[i]


def is_wiggle(nums: List[int]) -> bool:
    """Helper: verify array satisfies wiggle property."""
    for i in range(len(nums) - 1):
        if i % 2 == 0 and nums[i] >= nums[i + 1]:
            return False
        if i % 2 == 1 and nums[i] <= nums[i + 1]:
            return False
    return True


# ── TESTS ──────────────────────────────────────────────────────────────────────

def run_tests():
    # sort_by_frequency
    assert sort_by_frequency([1, 1, 2, 2, 2, 3]) == [2, 2, 2, 1, 1, 3]
    assert sort_by_frequency([2, 3, 1, 3, 2]) == [2, 2, 3, 3, 1]  # tie: 2<3 so 2 first
    assert sort_by_frequency([1]) == [1]
    print("✅ sort_by_frequency passed")

    # dutch_national_flag
    arr = [2, 0, 2, 1, 1, 0]
    dutch_national_flag(arr)
    assert arr == [0, 0, 1, 1, 2, 2]

    arr2 = [0]
    dutch_national_flag(arr2)
    assert arr2 == [0]

    arr3 = [2, 0, 1]
    dutch_national_flag(arr3)
    assert arr3 == [0, 1, 2]

    arr4 = [0, 0, 0]
    dutch_national_flag(arr4)
    assert arr4 == [0, 0, 0]
    print("✅ dutch_national_flag passed")

    # merge_sorted_arrays
    assert merge_sorted_arrays([[1, 4, 7], [2, 5, 8], [3, 6, 9]]) == list(range(1, 10))
    assert merge_sorted_arrays([[1], [], [2]]) == [1, 2]
    assert merge_sorted_arrays([]) == []
    assert merge_sorted_arrays([[1, 2, 3]]) == [1, 2, 3]
    print("✅ merge_sorted_arrays passed")

    # wiggle_sort
    arr5 = [3, 5, 2, 1, 6, 4]
    wiggle_sort(arr5)
    assert is_wiggle(arr5), f"Not wiggle: {arr5}"

    arr6 = [1, 2, 3, 4, 5]
    wiggle_sort(arr6)
    assert is_wiggle(arr6), f"Not wiggle: {arr6}"

    arr7 = [1]
    wiggle_sort(arr7)
    assert arr7 == [1]
    print("✅ wiggle_sort passed")

    print("\n🎉 All Exercise 8 tests passed!")


run_tests()
