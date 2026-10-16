"""Personalized PageRank by forward push (Andersen, Chung & Lang 2006) with hub damping and path attribution.

Why this and not power iteration: forward push is *local*. Its cost is O(1/(alpha*eps)) and independent of graph
size, which is what lets retrieval run on a bounded neighbourhood in milliseconds. Following HippoRAG, seeds are
weighted by node specificity and transitions are damped toward high-degree nodes so random walks do not drift
into hubs (the known weakness of fixed transition probabilities).

Invariants (tested): estimate p and residual r are non-negative and sum(p)+sum(r) == 1; p lower-bounds the exact
PPR vector and its L1 error is at most eps * volume; the result is deterministic.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .adjacency import Graph


@dataclass
class PPRResult:
    p: dict[str, float]
    r: dict[str, float]
    parent: dict[str, tuple[str, str]] = field(default_factory=dict)  # node -> (source node, edge type) of its biggest inflow
    pushes: int = 0
    source: str = "live"  # live forward-push | cache (summed precomputed vectors)

    def mass(self) -> float:
        return sum(self.p.values()) + sum(self.r.values())

    def path(self, node: str, max_len: int = 6) -> list[tuple[str, str]]:
        """[(from_node, edge_type)] chain from a seed to `node` (seed first)."""
        chain: list[tuple[str, str]] = []
        seen = {node}
        cur = node
        while cur in self.parent and len(chain) < max_len:
            src, et = self.parent[cur]
            if src in seen:
                break
            chain.append((src, et))
            seen.add(src)
            cur = src
        chain.reverse()
        return chain


def specificity(degree: int, beta: float = 1.0) -> float:
    """Inverse-degree damping in (0,1]: a hub with many neighbours is less specific evidence than a rare node."""
    return float(1.0 / (1.0 + math.log1p(degree)) ** beta)


def graph_context(g: Graph, hub_beta: float = 1.0) -> tuple[dict[str, int], dict[str, float]]:
    """Degree and specificity of every node. Compute once when running many queries on the same graph."""
    deg = {n: len(nb) for n, nb in g.adj.items()}
    return deg, {n: specificity(d, hub_beta) for n, d in deg.items()}


def personalized_pagerank(
    g: Graph,
    seeds: dict[str, float],
    *,
    alpha: float = 0.15,
    eps: float = 1e-4,
    hub_beta: float = 1.0,
    seed_specificity: bool = True,
    max_pushes: int = 200_000,
    context: tuple[dict[str, int], dict[str, float]] | None = None,
) -> PPRResult:
    seeds = {s: w for s, w in seeds.items() if w > 0 and s in g.adj}
    if not seeds:
        return PPRResult({}, {})
    deg, spec = context if context is not None else graph_context(g, hub_beta)

    start = {s: w * (spec[s] if seed_specificity else 1.0) for s, w in seeds.items()}
    total = sum(start.values())
    r = {s: w / total for s, w in start.items()}
    p: dict[str, float] = {}
    parent: dict[str, tuple[str, str]] = {}
    best_in: dict[str, float] = {}

    trans: dict[str, list[tuple[str, str, float]]] = {}

    def out(u: str) -> list[tuple[str, str, float]]:
        t = trans.get(u)
        if t is None:
            nbrs = g.neighbors(u)
            z = sum(w * spec[v] for v, _, w in nbrs)
            t = [(v, et, (w * spec[v]) / z) for v, et, w in nbrs] if z > 0 else []
            trans[u] = t
        return t

    heap: list[tuple[float, str]] = []

    def maybe_push(u: str) -> None:
        if r.get(u, 0.0) > eps * max(1, deg.get(u, 1)):
            heapq.heappush(heap, (-r[u] / max(1, deg.get(u, 1)), u))

    for s in r:
        maybe_push(s)
    pushes = 0
    while heap and pushes < max_pushes:
        _, u = heapq.heappop(heap)
        ru = r.get(u, 0.0)
        if ru <= eps * max(1, deg.get(u, 1)):
            continue  # stale heap entry
        pushes += 1
        edges = out(u)
        r[u] = 0.0
        if not edges:  # dangling: nothing to diffuse to, so the whole residual becomes estimate
            p[u] = p.get(u, 0.0) + ru
            continue
        p[u] = p.get(u, 0.0) + alpha * ru
        spread = (1.0 - alpha) * ru
        for v, et, prob in edges:
            c = spread * prob
            r[v] = r.get(v, 0.0) + c
            if c > best_in.get(v, 0.0):
                best_in[v], parent[v] = c, (u, et)
            maybe_push(v)
    return PPRResult(p, {k: v for k, v in r.items() if v > 0}, parent, pushes)


def exact_ppr(g: Graph, seeds: dict[str, float], *, alpha: float = 0.15, hub_beta: float = 1.0, seed_specificity: bool = True, iters: int = 400) -> dict[str, float]:
    """Reference power iteration used only by tests to measure forward-push error."""
    seeds = {s: w for s, w in seeds.items() if w > 0 and s in g.adj}
    deg = {n: len(nb) for n, nb in g.adj.items()}
    spec = {n: specificity(d, hub_beta) for n, d in deg.items()}
    start = {s: w * (spec[s] if seed_specificity else 1.0) for s, w in seeds.items()}
    z0 = sum(start.values())
    s0 = {k: v / z0 for k, v in start.items()}
    x = dict(s0)
    for _ in range(iters):
        nxt = {k: alpha * v for k, v in s0.items()}
        for u, xu in x.items():
            nbrs = g.neighbors(u)
            z = sum(w * spec[v] for v, _, w in nbrs)
            if z <= 0:
                nxt[u] = nxt.get(u, 0.0) + (1 - alpha) * xu  # dangling mass stays (matches absorbing push)
                continue
            for v, _, w in nbrs:
                nxt[v] = nxt.get(v, 0.0) + (1 - alpha) * xu * (w * spec[v]) / z
        x = nxt
    return x
