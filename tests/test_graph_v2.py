from __future__ import annotations

from pathlib import Path

import pytest

from membridge.graphalgo.adjacency import load_from_sqlite
from membridge.importers.service import run_import
from membridge.memory.consolidate import upsert
from membridge.memory.models import Candidate, EntityRef
from membridge.memory.pipeline import run_extraction
from membridge.memory.service import MemoryService
from membridge.resolve.entities import resolve_entities
from membridge.resolve.relations import rebuild_entity_edges
from membridge.store.graph import GraphStore
from membridge.store.sqlite import Store


def _uses(store: Store, text: str, project: str, tech: str, ts: float, aspect: str = "storage") -> str:
    return upsert(
        store,
        Candidate(statement=text, category="decision", subject=project.lower(), predicate=f"uses:{aspect}", object=tech,
                  entities=[EntityRef(project, "project"), EntityRef(tech)], ref=f"r-{ts}", ts=ts),
    ).memory_id  # fmt: skip


@pytest.fixture
def built(export_file: Path, store: Store) -> Store:
    run_import(store, export_file)
    run_extraction(store)
    return store


def test_typed_relation_edges_follow_the_fact_lifecycle(store: Store) -> None:
    a = _uses(store, "I decided to use PostgreSQL for Atlas storage.", "Atlas", "PostgreSQL", 1000.0)
    rebuild_entity_edges(store)
    e = store.conn.execute("SELECT * FROM entity_edges WHERE fact_id=?", (a,)).fetchone()
    assert e is not None and e["predicate"] == "USES" and e["invalid_at"] is None and e["weight"] > 0.5
    _uses(store, "I decided to use SQLite for Atlas storage.", "Atlas", "SQLite", 2000.0)
    rebuild_entity_edges(store)
    rows = {r["fact_id"]: r for r in store.conn.execute("SELECT * FROM entity_edges")}
    assert len(rows) == 2
    assert rows[a]["invalid_at"] == 2000.0 and rows[a]["weight"] < 0.5  # history is kept but weaker
    assert store.conn.execute("SELECT subject_entity_id IS NOT NULL AND object_entity_id IS NOT NULL FROM memories WHERE id=?", (a,)).fetchone()[0] == 1


def test_projection_matches_sqlite_and_is_idempotent(built: Store, graph: GraphStore) -> None:
    graph.rebuild(built)
    c1 = graph.counts()
    live_entities = built.conn.execute("SELECT COUNT(*) FROM live_entities").fetchone()[0]
    assert c1["nodes_entity"] == live_entities and c1["nodes_memory"] == built.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert graph.query("MATCH ()-[x:ABOUT]->() RETURN count(x)")[0][0] == built.conn.execute(
        "SELECT COUNT(*) FROM memory_entities me JOIN live_entities e ON e.id=me.entity_id").fetchone()[0]  # fmt: skip
    graph.sync(built)
    assert graph.counts() == c1


def test_incremental_sync_equals_full_rebuild(built: Store, graph: GraphStore) -> None:
    graph.rebuild(built)
    svc = MemoryService(built, graph)
    svc.record_decision("Atlas", "Use DuckDB for analytics", client="c1")  # incremental path
    svc.save_memory("The Atlas demo runs on port 8123", "fact", "user_stated", client="c1")
    incremental = graph.counts()
    other = GraphStore(name=graph.name + "_full")
    try:
        other.rebuild(built)
        assert other.counts() == incremental
    finally:
        other.drop()


def test_sqlite_and_falkordb_adjacency_are_identical(built: Store, graph: GraphStore) -> None:
    built.conn.execute("UPDATE entities SET type='project' WHERE canonical='atlas'")
    _uses(built, "I decided to use DuckDB for Atlas analytics.", "Atlas", "DuckDB", 5000.0, "analytics")
    rebuild_entity_edges(built)
    graph.rebuild(built)
    atlas = built.conn.execute("SELECT id FROM live_entities WHERE canonical='atlas'").fetchone()[0]
    for hops in (1, 2):
        sql_g = load_from_sqlite(built, [atlas], hops=hops, cap=500)
        fal_g = graph.adjacency([atlas], hops=hops, cap=500)
        assert sql_g.signature() == fal_g.signature(), f"loaders disagree at {hops} hops"
        assert len(sql_g.nodes()) > 5


def test_merge_removes_absorbed_node_and_split_restores_it(built: Store, graph: GraphStore) -> None:

    for i, name in enumerate(["Postgres", "PostgreSQL"]):
        upsert(built, Candidate(statement=f"I rely on {name} {i}.", category="fact", entities=[EntityRef(name)], ref=f"m{i}", ts=10.0 + i))
    graph.rebuild(built)
    before = graph.counts()["nodes_entity"]
    svc = MemoryService(built, graph, human=True)  # splitting a merge is the user's call
    assert svc.resolve_entities()["auto_merged"]
    assert graph.counts()["nodes_entity"] == before - 1
    merge_id = built.conn.execute("SELECT id FROM entity_merges WHERE reversed_at IS NULL").fetchone()[0]
    svc.split_entity_merge(merge_id)
    assert graph.counts()["nodes_entity"] == before


def test_entity_found_by_alias_through_fulltext(built: Store, graph: GraphStore) -> None:
    for i, name in enumerate(["Postgres", "PostgreSQL"]):
        upsert(built, Candidate(statement=f"I rely on {name} {i}.", category="fact", entities=[EntityRef(name)], ref=f"m{i}", ts=10.0 + i))
    resolve_entities(built)
    graph.rebuild(built)
    rows = graph.query("CALL db.idx.fulltext.queryNodes('Entity', 'postgres') YIELD node RETURN node.name, node.aliases")
    assert rows and "postgres" in rows[0][1] and "postgresql" in rows[0][1]


def test_pending_candidates_appear_as_same_as_edges(built: Store, graph: GraphStore) -> None:
    upsert(built, Candidate(statement="I use reactnative daily.", category="fact", entities=[EntityRef("ReactNative")], ref="a", ts=1.0))
    upsert(built, Candidate(statement="React Native builds are slow.", category="fact", entities=[EntityRef("React Nativ")], ref="b", ts=2.0))
    assert resolve_entities(built).pending == 1
    graph.rebuild(built)
    assert graph.query("MATCH (:Entity)-[x:SAME_AS]->(:Entity) RETURN count(x)")[0][0] == 1


def test_global_metrics_run_on_falkordb(built: Store, graph: GraphStore) -> None:
    graph.rebuild(built)
    out = graph.compute_global_metrics()
    assert out.get("pagerank", 0) > 0 and "pagerank_error" not in out and "wcc_error" not in out
    total = graph.query("MATCH (e:Entity) WHERE e.pr IS NOT NULL RETURN sum(e.pr)")[0][0]
    assert 0 < total <= 1.0 + 1e-6
