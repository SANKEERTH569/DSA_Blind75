
# ESSENTIAL PYTHON CODE SNIPPETS FOR INTERVIEWS

#=========================================
# 1. ARRAY & STRING PATTERNS
#=========================================

# Two Sum - Hash Map Approach
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Valid Palindrome - Two Pointers
def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

# Maximum Subarray (Kadane's Algorithm)
def max_subarray(nums):
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# Sliding Window Maximum Sum
def max_sum_subarray_k(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i-k]
        max_sum = max(max_sum, window_sum)
    return max_sum

#=========================================
# 2. LINKED LIST PATTERNS
#=========================================

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Reverse Linked List
def reverse_list(head):
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev

# Detect Cycle (Floyd's Algorithm)
def has_cycle(head):
    if not head or not head.next:
        return False

    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Merge Two Sorted Lists
def merge_lists(l1, l2):
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    current.next = l1 or l2
    return dummy.next

#=========================================
# 3. TREE PATTERNS
#=========================================

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Tree Traversals
def inorder(root):
    result = []
    def helper(node):
        if node:
            helper(node.left)
            result.append(node.val)
            helper(node.right)
    helper(root)
    return result

def preorder(root):
    result = []
    def helper(node):
        if node:
            result.append(node.val)
            helper(node.left)
            helper(node.right)
    helper(root)
    return result

def postorder(root):
    result = []
    def helper(node):
        if node:
            helper(node.left)
            helper(node.right)
            result.append(node.val)
    helper(root)
    return result

# Level Order Traversal (BFS)
def level_order(root):
    if not root:
        return []

    result = []
    queue = [root]

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.pop(0)
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)
    return result

# Maximum Depth
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Valid BST
def is_valid_bst(root):
    def validate(node, min_val, max_val):
        if not node:
            return True

        if node.val <= min_val or node.val >= max_val:
            return False

        return (validate(node.left, min_val, node.val) and 
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))

#=========================================
# 4. GRAPH PATTERNS
#=========================================

# DFS Template
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    print(start, end=' ')

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# BFS Template
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=' ')

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# Number of Islands
def num_islands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0'):
            return

        grid[r][c] = '0'  # Mark as visited

        # Check all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)

    return islands

#=========================================
# 5. DYNAMIC PROGRAMMING PATTERNS
#=========================================

# Fibonacci - Bottom Up
def fibonacci(n):
    if n <= 1:
        return n

    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1

# Climbing Stairs
def climb_stairs(n):
    if n <= 2:
        return n

    first, second = 1, 2
    for i in range(3, n + 1):
        third = first + second
        first, second = second, third

    return second

# Coin Change
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Longest Increasing Subsequence
def lis_length(nums):
    if not nums:
        return 0

    dp = [1] * len(nums)

    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

#=========================================
# 6. SEARCHING & SORTING
#=========================================

# Binary Search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

# Merge Sort
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

#=========================================
# 7. STACK & QUEUE PATTERNS
#=========================================

# Valid Parentheses
def valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            stack.append(char)

    return not stack

# Min Stack Implementation
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x):
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        if self.stack:
            if self.stack[-1] == self.min_stack[-1]:
                self.min_stack.pop()
            return self.stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]

    def get_min(self):
        if self.min_stack:
            return self.min_stack[-1]

#=========================================
# 8. HEAP PATTERNS
#=========================================

import heapq

# Top K Frequent Elements
def top_k_frequent(nums, k):
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1

    heap = []
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)

    return [num for freq, num in heap]

# Kth Largest Element
def find_kth_largest(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)

    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)

    return heap[0]

#=========================================
# 9. USEFUL PYTHON TRICKS
#=========================================

# Counter for frequency counting
from collections import Counter, defaultdict

def group_anagrams(strs):
    anagram_map = defaultdict(list)

    for s in strs:
        # Sort the string to use as key
        key = ''.join(sorted(s))
        anagram_map[key].append(s)

    return list(anagram_map.values())

# Using enumerate for index tracking
def two_sum_enumerate(nums, target):
    for i, num1 in enumerate(nums):
        for j, num2 in enumerate(nums[i+1:], i+1):
            if num1 + num2 == target:
                return [i, j]
    return []

# List comprehensions
def squares_of_evens(nums):
    return [x*x for x in nums if x % 2 == 0]

# Dictionary comprehension
def char_count(s):
    return {char: s.count(char) for char in set(s)}

# Using zip for parallel iteration
def merge_alternating(list1, list2):
    return [item for pair in zip(list1, list2) for item in pair]

#=========================================
# 10. COMMON EDGE CASES TO REMEMBER
#=========================================

def handle_edge_cases_example(arr):
    # Check for None/empty input
    if not arr:
        return []

    # Check for single element
    if len(arr) == 1:
        return arr

    # Check for negative values
    if any(x < 0 for x in arr):
        # Handle negative values
        pass

    # Check for duplicates
    if len(arr) != len(set(arr)):
        # Handle duplicates
        pass

    return arr

# Test your functions
if __name__ == "__main__":
    # Test examples
    print("Two Sum:", two_sum([2, 7, 11, 15], 9))
    print("Max Subarray:", max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    print("Fibonacci(10):", fibonacci(10))
    print("Binary Search:", binary_search([1, 3, 5, 7, 9], 5))
