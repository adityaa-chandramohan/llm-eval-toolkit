# EXERCISE 5 — Lists, Two Pointers & Sliding Window
#
# KEY CONCEPTS:
#   - Fixed sliding window: move a window of size k across array,
#     update sum by adding new right element and dropping old left element.
#     Avoids recomputing sum from scratch each time → O(n) vs O(n*k).
#   - Variable sliding window: expand right until condition breaks,
#     then shrink left until condition is restored.
#   - Prefix sums: prefix[i] = sum(nums[0..i-1]). Subarray sum [i..j] = prefix[j+1] - prefix[i].
#     Store {prefix_sum: earliest_index} to find subarrays summing to target in O(n).
#   - Two pointer: sort first, then use left/right pointers converging inward.

from typing import List, Tuple


# ── 1. Fixed sliding window ────────────────────────────────────────────────────

def max_subarray_sum(nums: List[int], k: int) -> int:
    """Return the maximum sum of any contiguous subarray of length k."""
    if len(nums) < k:
        raise ValueError(f"Array length {len(nums)} is less than k={k}")

    # Build initial window
    window_sum = sum(nums[:k])
    best = window_sum

    # Slide: add right element, drop left element
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        best = max(best, window_sum)

    return best


# ── 2. Variable sliding window + prefix sum ────────────────────────────────────

def longest_subarray_with_sum(nums: List[int], target: int) -> int:
    """
    Return length of longest contiguous subarray summing to target.
    Works for non-negative integers only (variable window shrink works).
    For arrays with negatives, use the prefix-sum approach below.
    """
    left = 0
    current_sum = 0
    best = 0

    for right in range(len(nums)):
        current_sum += nums[right]
        # Shrink window from left while sum exceeds target
        while current_sum > target and left <= right:
            current_sum -= nums[left]
            left += 1
        if current_sum == target:
            best = max(best, right - left + 1)

    return best


def longest_subarray_with_sum_v2(nums: List[int], target: int) -> int:
    """
    Prefix-sum + dict approach — handles negative numbers too.
    Key insight: if prefix[j] - prefix[i] == target, subarray (i..j-1) sums to target.
    Store the FIRST occurrence of each prefix sum to maximize length.
    """
    prefix_sum = 0
    # Map prefix_sum → earliest index where it was seen
    first_seen = {0: -1}  # empty prefix has sum 0, "seen" at index -1
    best = 0

    for i, num in enumerate(nums):
        prefix_sum += num
        complement = prefix_sum - target
        if complement in first_seen:
            best = max(best, i - first_seen[complement])
        # Only store FIRST occurrence (to maximize subarray length)
        if prefix_sum not in first_seen:
            first_seen[prefix_sum] = i

    return best


# ── 3. Two Sum ─────────────────────────────────────────────────────────────────

def two_sum(nums: List[int], target: int) -> Tuple[int, int]:
    """
    Return indices (i, j) where nums[i] + nums[j] == target, i < j.
    O(n) time using a dict: for each number, check if complement was seen.
    """
    seen = {}  # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    raise ValueError("No two sum solution found")


# ── 4. Three Sum ───────────────────────────────────────────────────────────────

def three_sum(nums: List[int]) -> List[Tuple[int, int, int]]:
    """
    Return all unique triplets summing to 0.
    Approach: sort, fix one element, use two-pointer for the rest.
    Skip duplicates at each level to avoid duplicate triplets.
    O(n²) time.
    """
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicate values for the fixed element
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append((nums[i], nums[left], nums[right]))
                # Skip duplicates for left and right pointers
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1

    return result


# ── TESTS ──────────────────────────────────────────────────────────────────────

def run_tests():
    # max_subarray_sum
    assert max_subarray_sum([2, 1, 5, 1, 3, 2], 3) == 9   # [5,1,3]
    assert max_subarray_sum([1, 4, 2, 9, 7, 3, 4], 4) == 23  # [9,7,3,4]=23
    assert max_subarray_sum([5], 1) == 5
    print("✅ max_subarray_sum passed")

    # longest_subarray_with_sum (non-negative)
    assert longest_subarray_with_sum([1, 2, 3, 1, 1, 1], 6) == 4  # [3,1,1,1]=6, len 4
    assert longest_subarray_with_sum([1, 1, 1, 1, 1], 3) == 3
    assert longest_subarray_with_sum([5, 4, 3], 3) == 1
    print("✅ longest_subarray_with_sum passed")

    # longest_subarray_with_sum_v2 (handles negatives)
    assert longest_subarray_with_sum_v2([1, -1, 5, -2, 3], 3) == 4  # [1,-1,5,-2]=3
    assert longest_subarray_with_sum_v2([-2, -1, 2, 1], 1) == 2    # [2,1] or [-1,2]=1 len 2
    assert longest_subarray_with_sum_v2([1, 2, 3], 6) == 3
    print("✅ longest_subarray_with_sum_v2 passed")

    # two_sum
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert two_sum([3, 2, 4], 6) == (1, 2)
    assert two_sum([3, 3], 6) == (0, 1)
    try:
        two_sum([1, 2, 3], 100)
        assert False, "Should raise"
    except ValueError:
        pass
    print("✅ two_sum passed")

    # three_sum
    assert sorted(three_sum([-1, 0, 1, 2, -1, -4])) == sorted([(-1, -1, 2), (-1, 0, 1)])
    assert three_sum([0, 0, 0]) == [(0, 0, 0)]
    assert three_sum([1, 2, 3]) == []
    assert three_sum([]) == []
    print("✅ three_sum passed")

    print("\n🎉 All Exercise 5 tests passed!")


run_tests()
