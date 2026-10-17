"""In-memory heterogeneous graph used by retrieval math (PPR etc.), loadable from SQLite *or* FalkorDB.

Both loaders produce the same structure from the same tables, so retrieval behaves identically with and
without Docker (degraded mode loses speed of extraction, not capability). Loading is bounded: BFS from the
seeds, `hops` deep, stopping at `cap` nodes, so cost never depends on total graph size.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..store.sqlite import Store

# Relative traversal weight per edge type. History edges are traversable but weaker than live structure.
TYPE_WEIGHT = {
    "ABOUT": 1.0, "REL": 0.9, "MENTIONED_IN": 0.5, "SUPERSEDES": 0.6, "CONTRADICTS": 0.6,
    "RELATED_TO": 0.3, "SAME_AS": 0.4, "DEPENDS_ON": 0.5, "LEARNED_FROM": 0.4, "DERIVED_FROM": 0.4,
}  # fmt: skip
ROLE_WEIGHT = {"subject": 1.0, "object": 1.0, "about": 0.7}


@dataclass
class Graph:
    """Undirected, typed, weighted multigraph collapsed to one edge per (pair, type) keeping the max weight."""

    adj: dict[str, dict[tuple[str, str], float]] = field(default_factory=lambda: defaultdict(dict))

    def add(self, a: str, b: str, etype: str, w: float) -> None:
        if a == b:
            return
        for x, y in ((a, b), (b, a)):
            cur = self.adj[x].get((y, etype))
            if cur is None or w > cur:
                self.adj[x][(y, etype)] = w

    def neighbors(self, n: str) -> list[tuple[str, str, float]]:
        return [(y, t, w) for (y, t), w in self.adj.get(n, {}).items()]

    def nodes(self) -> set[str]:
        return set(self.adj)

    def signature(self) -> set[tuple[str, str, str, float]]:
        """Order-independent edge set, used to prove the SQLite and FalkorDB loaders agree."""
        return {(a, b, t, round(w, 6)) for a, nb in self.adj.items() for (b, t), w in nb.items() if a < b}


def edge_weight(etype: str, *, role: str | None = None, count: int | None = None, weight: float | None = None) -> float:
    base = TYPE_WEIGHT.get(etype, 0.3)
    if etype == "ABOUT":
        return base * ROLE_WEIGHT.get(role or "about", 0.7)
    if etype == "MENTIONED_IN":
        return base * (0.5 + 0.5 * min(1.0, math.log1p(count or 1) / math.log(6)))
    if etype in ("REL", "RELATED_TO") and weight is not None:
        return base * min(1.0, 0.4 + 0.6 * math.log1p(weight) / math.log(6)) if etype == "RELATED_TO" else base * max(0.1, min(1.0, weight))
    return base


# ---- SQLite loader ------------------------------------------------------------------------------------
def _sql_neighbors(store: Store, ids: list[str], include_sensitive: bool = False) -> list[tuple[str, str, str, float]]:
    """All (a, b, etype, weight) edges touching `ids` (each pair appears from either side)."""
    if not ids:
        return []
    c = store.conn
    ents = "live_entities" if include_sensitive else "visible_entities"
    sm = "" if include_sensitive else " AND m.sensitivity='normal'"
    sa = "" if include_sensitive else " AND a.sensitivity='normal' AND b.sensitivity='normal'"
    q = ",".join("?" * len(ids))
    out: list[tuple[str, str, str, float]] = []
    for r in c.execute(
        f"SELECT me.memory_id AS m, me.entity_id AS e, me.role AS role FROM memory_entities me "
        f"JOIN memories m ON m.id=me.memory_id JOIN {ents} e ON e.id=me.entity_id "
        f"WHERE m.status!='deleted'{sm} AND (me.memory_id IN ({q}) OR me.entity_id IN ({q}))",
        ids + ids,
    ):
        out.append((r["m"], r["e"], "ABOUT", edge_weight("ABOUT", role=r["role"])))
    for r in c.execute(
        f"SELECT x.entity_id AS e, x.conversation_id AS v, x.count AS n FROM entity_mentions x JOIN {ents} e ON e.id=x.entity_id "
        f"WHERE x.entity_id IN ({q}) OR x.conversation_id IN ({q})",
        ids + ids,
    ):
        out.append((r["e"], r["v"], "MENTIONED_IN", edge_weight("MENTIONED_IN", count=r["n"])))
    for r in c.execute(
        f"SELECT l.src_id AS a, l.dst_id AS b, l.rel AS rel FROM memory_links l JOIN memories a ON a.id=l.src_id "
        f"JOIN memories b ON b.id=l.dst_id WHERE a.status!='deleted' AND b.status!='deleted'{sa} "
        f"AND (l.src_id IN ({q}) OR l.dst_id IN ({q}))",
        ids + ids,
    ):
        out.append((r["a"], r["b"], r["rel"], edge_weight(r["rel"])))
    for r in c.execute(
        f"SELECT src_entity AS a, dst_entity AS b, MAX(weight) AS w FROM entity_edges WHERE src_entity IN ({q}) OR dst_entity IN ({q}) "
        f"GROUP BY src_entity, dst_entity",
        ids + ids,
    ):
        out.append((r["a"], r["b"], "REL", edge_weight("REL", weight=r["w"])))
    for r in c.execute(
        f"SELECT a_entity AS a, b_entity AS b, weight AS w FROM entity_cooccurrence WHERE a_entity IN ({q}) OR b_entity IN ({q})",
        ids + ids,
    ):
        out.append((r["a"], r["b"], "RELATED_TO", edge_weight("RELATED_TO", weight=r["w"])))
    for r in c.execute(
        f"SELECT a_id, b_id FROM entity_link_candidates WHERE status='pending' AND (a_id IN ({q}) OR b_id IN ({q}))", ids + ids
    ):
        out.append((r["a_id"], r["b_id"], "SAME_AS", edge_weight("SAME_AS")))
    return out


def load_from_sqlite(store: Store, seeds: list[str], hops: int = 2, cap: int = 2000, include_sensitive: bool = False) -> Graph:
    g = Graph()
    frontier, visited = list(dict.fromkeys(seeds)), set(seeds)
    for _ in range(max(1, hops)):
        nxt: list[str] = []
        for a, b, t, w in _sql_neighbors(store, frontier, include_sensitive):
            g.add(a, b, t, w)
            for n in (a, b):
                if n not in visited and len(visited) < cap:
                    visited.add(n)
                    nxt.append(n)
        frontier = nxt
        if not frontier or len(visited) >= cap:
            break
    return g


# ---- FalkorDB loader ------------------------------------------------------------------------------------
_REL_TYPES = "ABOUT|REL|SUPERSEDES|CONTRADICTS|MENTIONED_IN|SAME_AS|RELATED_TO|DEPENDS_ON|LEARNED_FROM|DERIVED_FROM"


def load_from_graph(graph: Any, seeds: list[str], hops: int = 2, cap: int = 2000) -> Graph:
    g = Graph()
    frontier, visited = list(dict.fromkeys(seeds)), set(seeds)
    for _ in range(max(1, hops)):
        if not frontier:
            break
        rows = graph.query(
            f"MATCH (a)-[r:{_REL_TYPES}]-(b) WHERE a.id IN $ids AND coalesce(b.status,'') <> 'deleted' "
            "RETURN a.id, b.id, type(r), r.role, r.count, r.weight, r.status",
            {"ids": frontier},
        )
        nxt: list[str] = []
        for a, b, t, role, count, weight, status in rows:
            if t == "SAME_AS" and status not in (None, "pending"):
                continue
            g.add(a, b, t, edge_weight(t, role=role, count=count, weight=weight))
            for n in (a, b):
                if n not in visited and len(visited) < cap:
                    visited.add(n)
                    nxt.append(n)
        frontier = nxt
        if len(visited) >= cap:
            break
    return g
