# Memoization

### Link : [NoteBook](https://github.com/KevalSolanki77/Deep-Learning-Journey/blob/main/Day-06-Memoization/memoization_demo.ipynb)  

## What It Is

Memoization stores a function's return value keyed by its arguments. A later call with the same arguments skips the computation and reads the stored value instead. It's a specific form of caching, and it's the mechanism behind the top-down style of dynamic programming: write the plain recursive solution first, add a cache, done.

## The Problem: Naive Recursive Fibonacci

```python
import time
def fib(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

start = time.time()
fib(40)
print(f"Time Taken : {time.time() - start}")
```

```text
Time Taken : 44.88774514198303
```

Each call to `fib(n)` spawns two more calls, so the recursion tree for `fib(n)` has close to `2^n` nodes. `fib(40)` runs over a billion calls, and the same subproblems recur constantly: `fib(38)` gets recomputed from scratch inside the branch for `fib(39)` and again inside the branch for `fib(40)`, and every value below it repeats the same way. Almost 45 seconds goes into recomputing answers the function already had.

## The Fix: Cache the Subproblems

```python
def fib(n, d):
    if n in d:
        return d[n]
    else:
        d[n] = fib(n-1, d) + fib(n-2, d)
        return d[n]

d = {0: 1, 1: 1}
start = time.time()
fib(40, d)
print(f"Time Taken : {time.time() - start}")
```

```text
Time Taken : 0.0
```

The dictionary `d` holds every `fib(n)` value the first time it's computed. Each recursive call checks `d` before doing any work: a hit returns the stored value in constant time, a miss computes it once and stores it before returning. `fib(40)` now touches exactly 41 unique subproblems (`fib(0)` through `fib(40)`), each computed once. Every one of the billion-plus redundant calls from the naive version turns into a dictionary lookup.

## Why It Cuts Computation

| | Naive recursion | Memoized |
|---|---|---|
| `fib(40)` time | 44.89s | under 1ms |
| Time complexity | O(2^n) | O(n) |
| Space complexity | O(n) call stack | O(n) call stack + O(n) cache |

The complexity drop from exponential to linear comes from one change: computing each unique value once instead of recomputing it on every branch that needs it. Trading a small amount of memory (the cache) for that reduction in repeated work is the entire idea.

## Where the Same Idea Shows Up

**Dynamic programming.** Memoization is the top-down half of DP. Tabulation, the bottom-up version, fills the same cache with a loop instead of on-demand recursive calls, but the goal matches: compute each subproblem once.

**Backpropagation.** Forward propagation computes each layer's activation and stores it (`A1` in the Day 4 backprop example). The backward pass reads that stored value directly instead of recomputing the forward pass from the input. Textbook memoization caches results across repeated calls carrying the same arguments; forward-pass caching stores each value once within a single training step, since a training step never recomputes `A1` for the same input twice in a normal run. Different mechanism, same target: skip a computation you already paid for.

### Link : [NoteBook](https://github.com/KevalSolanki77/Deep-Learning-Journey/blob/main/Day-06-Memoization/memoization_demo.ipynb)  