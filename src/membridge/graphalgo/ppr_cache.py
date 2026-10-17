"""Precomputed Personalized PageRank vectors (the "linearity shortcut").

PPR is linear in its restart distribution:  PPR(sum_i w_i * seed_i) = sum_i w_i * PPR(seed_i).
So instead of loading a neighbourhood and pushing at query time (tens of ms), consolidation stores one sparse
vector per entity (top-K facts plus the ancestors needed to explain them) and a query just sums a few of them.
The cache is stamped with the content generation it was built from; any later write makes it stale and retrieval
falls back to the live forward-push (always correct, just slower) until a rebuild finishes.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from ..store.sqlite import Store
from .adjacency import Graph, edge_weight
from .ppr import PPRResult, graph_context, personalized_pagerank, specificity


def load_all(conn: sqlite3.Connection) -> Graph:
    """The whole heterogeneous graph from SQLite in six scans (same edge set and weights as the bounded loaders)."""
    g = Graph()
    for r in conn.execute(
        "SELECT me.memory_id AS m, me.entity_id AS e, me.role AS role FROM memory_entities me "
        "JOIN memories m ON m.id=me.memory_id JOIN live_entities e ON e.id=me.entity_id "
        "WHERE m.status!='deleted' AND m.sensitivity='normal'"  # the cache is the default (no-grant) view
    ):
        g.add(r["m"], r["e"], "ABOUT", edge_weight("ABOUT", role=r["role"]))
    for r in conn.execute("SELECT x.entity_id AS e, x.conversation_id AS v, x.count AS n FROM entity_mentions x JOIN live_entities e ON e.id=x.entity_id"):
        g.add(r["e"], r["v"], "MENTIONED_IN", edge_weight("MENTIONED_IN", count=r["n"]))
    for r in conn.execute(
        "SELECT l.src_id AS a, l.dst_id AS b, l.rel AS rel FROM memory_links l JOIN memories a ON a.id=l.src_id "
        "JOIN memories b ON b.id=l.dst_id WHERE a.status!='deleted' AND b.status!='deleted' "
        "AND a.sensitivity='normal' AND b.sensitivity='normal'"
    ):
        g.add(r["a"], r["b"], r["rel"], edge_weight(r["rel"]))
    for r in conn.execute("SELECT src_entity AS a, dst_entity AS b, MAX(weight) AS w FROM entity_edges GROUP BY src_entity, dst_entity"):
        g.add(r["a"], r["b"], "REL", edge_weight("REL", weight=r["w"]))
    for r in conn.execute("SELECT a_entity AS a, b_entity AS b, weight AS w FROM entity_cooccurrence"):
        g.add(r["a"], r["b"], "RELATED_TO", edge_weight("RELATED_TO", weight=r["w"]))
    for r in conn.execute("SELECT a_id, b_id FROM entity_link_candidates WHERE status='pending'"):
        g.add(r["a_id"], r["b_id"], "SAME_AS", edge_weight("SAME_AS"))
    return g


def build_ppr_cache(store: Store, *, k: int = 64, eps: float = 1e-3, alpha: float = 0.15, hub_beta: float = 1.0) -> dict[str, Any]:
    """(Re)build every entity's vector. Loads and computes on a private read-only connection (readers are not blocked)
    and only takes the store lock for the final bulk write."""
    t0 = time.perf_counter()
    ro = sqlite3.connect(f"file:{store.path}?mode=ro", uri=True, timeout=30)
    ro.row_factory = sqlite3.Row
    try:
        # One read transaction: the generation stamped on the cache is the generation of the data it was built from. Reading
        # it on the shared connection could include another thread's uncommitted bump and mark a stale cache as fresh.
        ro.execute("BEGIN")
        row = ro.execute("SELECT v FROM meta WHERE k='gen'").fetchone()
        gen = int(row[0]) if row else 0
        g = load_all(ro)
        ents = [r[0] for r in ro.execute("SELECT id FROM live_entities")]
    finally:
        ro.close()
    rows: list[tuple[str, str, float, str | None, str | None]] = []
    ctx = graph_context(g, hub_beta)  # once, not per entity
    for e in ents:
        if e not in g.adj:
            continue
        res = personalized_pagerank(g, {e: 1.0}, alpha=alpha, eps=eps, hub_beta=hub_beta, seed_specificity=False, context=ctx)
        top = sorted(((n, v) for n, v in res.p.items() if n.startswith("mem_")), key=lambda kv: (-kv[1], kv[0]))[:k]
        keep: dict[str, float] = {n: v for n, v in top}
        for n, _ in top:  # keep the ancestors so a path can be explained from cache alone
            for src, _et in res.path(n):
                if src != e and src not in keep:
                    keep[src] = res.p.get(src, 0.0)
        keep[e] = res.p.get(e, 0.0)
        for n, v in keep.items():
            par = res.parent.get(n)
            rows.append((e, n, v, par[0] if par else None, par[1] if par else None))
    with store.lock:
        store.conn.execute("DELETE FROM ppr_cache")
        store.conn.executemany("INSERT INTO ppr_cache(entity_id, node_id, p, parent_id, etype) VALUES (?,?,?,?,?)", rows)
        store.conn.execute("INSERT OR REPLACE INTO ppr_cache_state(id, gen, built_at, entities) VALUES (1,?,?,?)", (gen, time.time(), len(ents)))
        store.conn.commit()
    return {"entities": len(ents), "rows": len(rows), "seconds": round(time.perf_counter() - t0, 2), "gen": gen}


def cache_is_fresh(store: Store) -> bool:
    row = store.conn.execute("SELECT gen FROM ppr_cache_state WHERE id=1").fetchone()
    return row is not None and int(row[0]) == store.generation()


def cached_ppr(store: Store, seeds: dict[str, float], fact_entities: dict[str, list[str]], degree: dict[str, int], alpha: float = 0.15) -> PPRResult | None:
    """Answer a PPR query from stored vectors, or None if any seed has no vector / the cache is stale.

    Entity seeds are weighted by w * specificity (as the live algorithm does); a fact seed contributes its own restart mass
    plus, through its ABOUT entities, those entities' vectors."""
    if not cache_is_fresh(store):
        return None
    mix: dict[str, float] = {}  # entity -> weight
    direct: dict[str, float] = {}
    for node, w in seeds.items():
        if node.startswith("ent_"):
            mix[node] = mix.get(node, 0.0) + w * specificity(degree.get(node, 1))
        elif node.startswith("mem_"):
            es = fact_entities.get(node, [])
            direct[node] = direct.get(node, 0.0) + w
            for e in es:
                mix[e] = mix.get(e, 0.0) + 0.5 * w / len(es) * specificity(degree.get(e, 1))
    total = sum(mix.values()) + sum(direct.values())
    if total <= 0:
        return None
    p: dict[str, float] = {}
    parent: dict[str, tuple[str, str]] = {}
    best: dict[str, float] = {}
    for e, w in mix.items():
        rows = store.conn.execute("SELECT node_id, p, parent_id, etype FROM ppr_cache WHERE entity_id=?", (e,)).fetchall()
        if not rows:
            return None  # entity added after the cache was built: use the live algorithm
        for r in rows:
            c = w / total * r["p"]
            p[r["node_id"]] = p.get(r["node_id"], 0.0) + c
            if r["parent_id"] and c > best.get(r["node_id"], 0.0):
                best[r["node_id"]], parent[r["node_id"]] = c, (r["parent_id"], r["etype"])
    for f, w in direct.items():
        p[f] = p.get(f, 0.0) + alpha * w / total
    return PPRResult(p, {}, parent, 0, "cache")
