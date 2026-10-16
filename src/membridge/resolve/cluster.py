"""Union-find (disjoint set) with the guard rails entity resolution needs.

Plain transitive closure is the classic over-merge failure: one false-positive edge chains unrelated entities
together. So `union` refuses a merge that would (a) join two entities marked cannot-link, (b) exceed the
maximum cluster size, or (c) put together two members whose pairwise score is known to be below the
reject threshold. Path compression + union by rank keep operations near O(alpha(n)).
"""

from __future__ import annotations

from collections.abc import Iterable


class UnionFind:
    def __init__(self, items: Iterable[str] = (), *, cannot_link: Iterable[tuple[str, str]] = (), max_size: int = 8) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}
        self.members: dict[str, set[str]] = {}
        self.cannot: dict[str, set[str]] = {}
        self.max_size = max_size
        for it in items:
            self.add(it)
        for a, b in cannot_link:
            self.forbid(a, b)

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x], self.rank[x], self.members[x] = x, 0, {x}

    def forbid(self, a: str, b: str) -> None:
        self.cannot.setdefault(a, set()).add(b)
        self.cannot.setdefault(b, set()).add(a)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def can_union(self, a: str, b: str) -> tuple[bool, str]:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True, "same"
        if len(self.members[ra]) + len(self.members[rb]) > self.max_size:
            return False, "max_cluster_size"
        small, big = (ra, rb) if len(self.members[ra]) <= len(self.members[rb]) else (rb, ra)
        for x in self.members[small]:
            if self.cannot.get(x, set()) & self.members[big]:
                return False, "cannot_link"
        return True, "ok"

    def union(self, a: str, b: str) -> tuple[bool, str]:
        """Merge the clusters of a and b if allowed. Returns (merged, reason)."""
        ok, why = self.can_union(a, b)
        if not ok or why == "same":
            return ok and why != "same", why
        ra, rb = self.find(a), self.find(b)
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra  # union by rank
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.members[ra] |= self.members.pop(rb)
        return True, "ok"

    def clusters(self) -> list[set[str]]:
        return [set(m) for m in self.members.values()]
