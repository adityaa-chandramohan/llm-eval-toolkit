# EXERCISE 9 — Binary Search
#
# KEY CONCEPTS:
#   - Binary search requires a SORTED array. Each step halves the search space → O(log n).
#   - Template: lo=0, hi=len-1. Loop while lo<=hi. mid=(lo+hi)//2.
#     - Target found: return mid
#     - Target > mid: lo = mid + 1
#     - Target < mid: hi = mid - 1
#   - Leftmost/rightmost variants: don't return on match, keep narrowing.
#   - Rotated sorted array: one half is ALWAYS sorted. Determine which half,
#     then decide which side the target falls on.
#   - Find minimum in rotated array: minimum is where the "break" in sorted order is.
#     If mid > right, minimum is in right half; otherwise it's in left half (including mid).

from typing import List, Tuple


# ── 1. Standard Binary Search ──────────────────────────────────────────────────

def binary_search(arr: List[int], target: int) -> int:
    """Return index of target in sorted arr, or -1 if not found."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# ── 2. Search Insert Position ──────────────────────────────────────────────────

def search_insert_position(arr: List[int], target: int) -> int:
    """
    Return index where target is found, OR where it should be inserted
    to keep arr sorted (leftmost valid position).
    This is equivalent to: find first index i where arr[i] >= target.
    """
    lo, hi = 0, len(arr)  # hi = len (can insert at end)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid  # mid could be the answer; keep it in range
    return lo


# ── 3. Find First and Last Position ───────────────────────────────────────────

def find_first_last(arr: List[int], target: int) -> Tuple[int, int]:
    """
    Return (first_index, last_index) of target in sorted array.
    Returns (-1, -1) if target not found.
    Two binary searches: one to find leftmost, one for rightmost.
    """
    def find_first(arr, target):
        lo, hi = 0, len(arr) - 1
        result = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target:
                result = mid
                hi = mid - 1  # keep searching LEFT
            elif arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    def find_last(arr, target):
        lo, hi = 0, len(arr) - 1
        result = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr[mid] == target:
                result = mid
                lo = mid + 1  # keep searching RIGHT
            elif arr[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    return (find_first(arr, target), find_last(arr, target))


# ── 4. Search in Rotated Sorted Array ─────────────────────────────────────────

def search_rotated_array(arr: List[int], target: int) -> int:
    """
    Search for target in a rotated sorted array (no duplicates). Return index or -1.
    Key insight: one half of [lo..hi] is ALWAYS sorted.
    Determine which half is sorted, then check if target falls within it.
    O(log n).
    """
    lo, hi = 0, len(arr) - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid

        # Left half is sorted
        if arr[lo] <= arr[mid]:
            if arr[lo] <= target < arr[mid]:
                hi = mid - 1  # target is in left sorted half
            else:
                lo = mid + 1  # target must be in right half
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[hi]:
                lo = mid + 1  # target is in right sorted half
            else:
                hi = mid - 1  # target must be in left half

    return -1


# ── 5. Find Minimum in Rotated Sorted Array ────────────────────────────────────

def find_minimum_rotated(arr: List[int]) -> int:
    """
    Find minimum element in a rotated sorted array (no duplicates).
    Key insight: the minimum is where sorted order "breaks".
    If arr[mid] > arr[hi]: minimum is in RIGHT half (mid+1..hi).
    Else: minimum is in LEFT half including mid (lo..mid).
    O(log n).
    """
    lo, hi = 0, len(arr) - 1

    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] > arr[hi]:
            lo = mid + 1  # minimum is past mid
        else:
            hi = mid  # mid might be the minimum; include it

    return arr[lo]


# ── TESTS ──────────────────────────────────────────────────────────────────────

def run_tests():
    # binary_search
    assert binary_search([1, 3, 5, 7, 9], 5) == 2
    assert binary_search([1, 3, 5, 7, 9], 1) == 0
    assert binary_search([1, 3, 5, 7, 9], 9) == 4
    assert binary_search([1, 3, 5, 7, 9], 4) == -1
    assert binary_search([], 1) == -1
    assert binary_search([1], 1) == 0
    print("✅ binary_search passed")

    # search_insert_position
    assert search_insert_position([1, 3, 5, 6], 5) == 2
    assert search_insert_position([1, 3, 5, 6], 2) == 1
    assert search_insert_position([1, 3, 5, 6], 7) == 4
    assert search_insert_position([1, 3, 5, 6], 0) == 0
    assert search_insert_position([], 5) == 0
    print("✅ search_insert_position passed")

    # find_first_last
    assert find_first_last([5, 7, 7, 8, 8, 10], 8) == (3, 4)
    assert find_first_last([5, 7, 7, 8, 8, 10], 6) == (-1, -1)
    assert find_first_last([1, 1, 1, 1], 1) == (0, 3)
    assert find_first_last([1], 1) == (0, 0)
    assert find_first_last([], 0) == (-1, -1)
    print("✅ find_first_last passed")

    # search_rotated_array
    assert search_rotated_array([4, 5, 6, 7, 0, 1, 2], 0) == 4
    assert search_rotated_array([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert search_rotated_array([1], 0) == -1
    assert search_rotated_array([1], 1) == 0
    assert search_rotated_array([1, 2, 3, 4, 5], 3) == 2  # not rotated
    assert search_rotated_array([5, 1, 2, 3, 4], 5) == 0
    print("✅ search_rotated_array passed")

    # find_minimum_rotated
    assert find_minimum_rotated([3, 4, 5, 1, 2]) == 1
    assert find_minimum_rotated([4, 5, 6, 7, 0, 1, 2]) == 0
    assert find_minimum_rotated([11, 13, 15, 17]) == 11  # not rotated
    assert find_minimum_rotated([2, 1]) == 1
    assert find_minimum_rotated([1]) == 1
    print("✅ find_minimum_rotated passed")

    print("\n🎉 All Exercise 9 tests passed!")


run_tests()
