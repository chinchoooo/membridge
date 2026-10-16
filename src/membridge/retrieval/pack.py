"""Exact context packing: multiple-choice 0/1 knapsack by dynamic programming.

Each item offers one or more *options* (full evidence, shortened evidence); the DP picks at most one option per
item to maximise total value within the token budget. Exact for the sizes involved (n <= ~60 items, 1-token
resolution), so it is strictly at least as good as greedy first-fit, which it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Option:
    value: float
    cost: int  # tokens


def pack(groups: list[list[Option]], budget: int, *, granularity: int = 1, must: set[int] | None = None) -> list[int | None]:
    """Return, per group, the chosen option index or None. `must` groups get a dominating bonus (kept if at all feasible)."""
    must = must or set()
    cap = max(0, budget) // granularity
    n = len(groups)
    bonus = 1e6
    neg = float("-inf")
    dp = [0.0] * (cap + 1)
    choice: list[list[int]] = [[-1] * (cap + 1) for _ in range(n)]
    for i, opts in enumerate(groups):
        new = dp[:]  # option "skip"
        for b in range(cap + 1):
            for j, o in enumerate(opts):
                c = -(-o.cost // granularity)  # ceil
                if c <= b and dp[b - c] != neg:
                    v = dp[b - c] + o.value + (bonus if i in must else 0.0)
                    if v > new[b]:
                        new[b] = v
                        choice[i][b] = j
        dp = new
    # reconstruct from the best capacity
    b = max(range(cap + 1), key=lambda x: dp[x])
    picked: list[int | None] = [None] * n
    for i in range(n - 1, -1, -1):
        j = choice[i][b]
        if j >= 0:
            picked[i] = j
            b -= -(-groups[i][j].cost // granularity)
    return picked
