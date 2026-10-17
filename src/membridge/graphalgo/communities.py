"""Hierarchical community detection over the entity graph (GraphRAG-style), with stable ids for incremental refresh.

Level 1 = finest communities, level 2 = coarser groupings of them (Louvain's own nested passes). Leiden is used
when `igraph` + `leidenalg` are installed (extra `leiden`; it guarantees connected communities), otherwise
NetworkX Louvain. Detection is seeded, so the same graph always yields the same partition. A community's id is a hash
of its sorted members: unchanged membership keeps its id (and its summary); only changed communities are rebuilt.
ArchRAG's critique applies (topology-only communities can mix themes), so summaries stay linked to leaf facts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..memory.models import sha
from ..store.sqlite import Store


@dataclass
class CommunityReport:
    method: str = ""
    levels: dict[int, int] = field(default_factory=dict)  # level -> number of communities
    entities: int = 0
    new_ids: set[str] = field(default_factory=set)
    kept_ids: set[str] = field(default_factory=set)
    removed_ids: set[str] = field(default_factory=set)


def build_entity_graph(store: Store) -> nx.Graph[Any]:
    """Weighted undirected graph: co-occurrence in facts, typed relations, and (weakly) shared conversations."""
    from ..resolve.relations import rebuild_entity_edges

    rebuild_entity_edges(store)  # refresh typed relations + the co-occurrence table this graph is built from
    g: nx.Graph[Any] = nx.Graph()
    c = store.conn
    live = {r[0] for r in c.execute("SELECT DISTINCT me.entity_id FROM memory_entities me JOIN memories m ON m.id=me.memory_id "
                                    "JOIN visible_entities e ON e.id=me.entity_id WHERE m.status!='deleted' AND m.sensitivity='normal'")}  # fmt: skip
    for eid in live:
        g.add_node(eid)

    def bump(a: str, b: str, w: float) -> None:
        if a != b and a in live and b in live:
            g.add_edge(a, b, weight=g.get_edge_data(a, b, {}).get("weight", 0.0) + w)

    for r in c.execute("SELECT a_entity AS a, b_entity AS b, weight AS n FROM entity_cooccurrence"):
        bump(r["a"], r["b"], float(r["n"]))
    for r in c.execute("SELECT src_entity AS a, dst_entity AS b, MAX(weight) AS w FROM entity_edges GROUP BY src_entity, dst_entity"):
        bump(r["a"], r["b"], float(r["w"]))
    for r in c.execute(
        "SELECT x.entity_id AS a, y.entity_id AS b FROM entity_mentions x JOIN entity_mentions y "
        "ON x.conversation_id=y.conversation_id AND x.entity_id<y.entity_id"
    ):
        bump(r["a"], r["b"], 0.2)
    return g


def _leiden(g: nx.Graph[Any], seed: int) -> list[list[set[str]]] | None:
    try:
        import igraph as ig  # type: ignore[import-untyped, unused-ignore]
        import leidenalg  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        return None
    nodes = sorted(g.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in g.edges], directed=False)
    ig_g.es["weight"] = [g[a][b]["weight"] for a, b in g.edges]
    part = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition, weights="weight", seed=seed)
    fine = [{nodes[i] for i in cl} for cl in part]
    return [fine, _coarsen(g, fine, seed)]


def _coarsen(g: nx.Graph[Any], fine: list[set[str]], seed: int) -> list[set[str]]:
    """Second level: cluster the aggregate graph whose nodes are the fine communities."""
    owner = {e: i for i, cl in enumerate(fine) for e in cl}
    agg: nx.Graph[Any] = nx.Graph()
    agg.add_nodes_from(range(len(fine)))
    for a, b, d in g.edges(data=True):
        ia, ib = owner[a], owner[b]
        if ia != ib:
            agg.add_edge(ia, ib, weight=agg.get_edge_data(ia, ib, {}).get("weight", 0.0) + d["weight"])
    groups = nx.community.louvain_communities(agg, weight="weight", resolution=0.5, seed=seed)
    return [{e for i in grp for e in fine[i]} for grp in groups]


def detect(g: nx.Graph[Any], *, resolution: float = 1.0, seed: int = 42, prefer_leiden: bool = True) -> tuple[str, list[list[set[str]]]]:
    """Return (method, partitions from finest to coarsest). Isolated entities are left out (no structure to cluster)."""
    core = g.subgraph([n for n in g.nodes if g.degree(n) > 0]).copy()
    if core.number_of_nodes() == 0:
        return "none", []
    if prefer_leiden and (lv := _leiden(core, seed)) is not None:
        return "leiden", lv
    parts = [[set(cl) for cl in p] for p in nx.community.louvain_partitions(core, weight="weight", resolution=resolution, seed=seed)]
    return "louvain", parts or [[set(core.nodes)]]


def member_hash(members: set[str]) -> str:
    return sha(*sorted(members))[:16]


def build_communities(store: Store, *, resolution: float = 1.0, seed: int = 42, prefer_leiden: bool = True) -> CommunityReport:
    """(Re)detect communities and persist them. Ids are member-hashes, so unchanged communities keep identity."""
    rep = CommunityReport()
    g = build_entity_graph(store)
    rep.entities = g.number_of_nodes()
    method, parts = detect(g, resolution=resolution, seed=seed, prefer_leiden=prefer_leiden)
    rep.method = method
    c = store.conn
    old_ids = {r[0] for r in c.execute("SELECT id FROM communities")}
    c.execute("DELETE FROM community_members")
    c.execute("DELETE FROM communities")
    now = time.time()
    fine = parts[0] if parts else []
    coarse = parts[-1] if len(parts) > 1 else ([set().union(*fine)] if fine else [])
    level_sets = {1: fine, 2: coarse}
    ids: dict[tuple[int, str], str] = {}
    for level, groups in level_sets.items():
        for members in groups:
            if not members:
                continue
            cid = f"com{level}_{member_hash(members)}"
            ids[(level, member_hash(members))] = cid
    for level, groups in level_sets.items():
        for members in groups:
            if not members:
                continue
            cid = ids[(level, member_hash(members))]
            parent = None
            if level == 1:
                parent = next((ids[(2, member_hash(p))] for p in coarse if members <= p), None)
            c.execute(
                "INSERT OR REPLACE INTO communities(id, level, parent_id, method, resolution, member_hash, created_at) VALUES (?,?,?,?,?,?,?)",
                (cid, level, parent, method, resolution, member_hash(members), now),
            )
            c.executemany("INSERT OR IGNORE INTO community_members(community_id, entity_id) VALUES (?,?)", [(cid, e) for e in sorted(members)])
        rep.levels[level] = len([m for m in groups if m])
    new_ids = {r[0] for r in c.execute("SELECT id FROM communities")}
    rep.kept_ids, rep.new_ids, rep.removed_ids = old_ids & new_ids, new_ids - old_ids, old_ids - new_ids
    c.commit()
    return rep
