from __future__ import annotations

from pathlib import Path

from membridge.importers.service import run_import
from membridge.memory.pipeline import run_extraction
from membridge.store.graph import GraphStore, fulltext_query
from membridge.store.sqlite import Store


def _built(export_file: Path, store: Store, graph: GraphStore) -> dict[str, int]:
    run_import(store, export_file)
    run_extraction(store)
    return graph.rebuild(store)


def test_projection_counts_match_sqlite_and_is_idempotent(export_file: Path, store: Store, graph: GraphStore) -> None:
    n = _built(export_file, store, graph)
    c1 = graph.counts()
    assert c1["nodes_memory"] == store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == n["memories"]
    graph.sync(store)  # second sync must not duplicate anything
    assert graph.counts() == c1


def test_supersede_chain_and_provenance_paths(export_file: Path, store: Store, graph: GraphStore) -> None:
    _built(export_file, store, graph)
    rows = graph.query(
        "MATCH (n:Memory)-[r:SUPERSEDES]->(o:Memory)-[:SUPPORTED_BY]->(g:Message)-[:PART_OF]->(v:Conversation) "
        "RETURN n.statement, o.statement, r.reason, v.title"
    )
    assert len(rows) >= 1
    assert any("SQLite" in r[0] and "PostgreSQL" in r[1] and r[2] == "explicit_switch" for r in rows)


def test_entity_connects_memories_across_conversations(export_file: Path, store: Store, graph: GraphStore) -> None:
    _built(export_file, store, graph)
    rows = graph.query(
        "MATCH (e:Entity {canonical:'atlas'})<-[:ABOUT]-(m:Memory)-[:SUPPORTED_BY]->(:Message)-[:PART_OF]->(v:Conversation) "
        "RETURN count(DISTINCT m), count(DISTINCT v)"
    )
    assert rows[0][0] >= 4 and rows[0][1] >= 2  # fragments from several conversations meet at one node


def test_fulltext_seed_and_safe_query_builder(export_file: Path, store: Store, graph: GraphStore) -> None:
    _built(export_file, store, graph)
    q = fulltext_query('Atlas database"; MATCH (n) DETACH DELETE n //')
    assert q is not None and '"' not in q and ";" not in q
    rows = graph.query("CALL db.idx.fulltext.queryNodes('Memory', $q) YIELD node, score RETURN node.id, score", {"q": q})
    assert rows
    assert graph.counts()["nodes_memory"] > 0  # the injection attempt deleted nothing


def test_deleted_memory_leaves_graph_on_sync(export_file: Path, store: Store, graph: GraphStore) -> None:
    _built(export_file, store, graph)
    mid = store.conn.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.conn.execute("UPDATE memories SET status='deleted' WHERE id=?", (mid,))
    graph.sync(store)
    assert graph.query("MATCH (m:Memory {id:$id}) RETURN count(m)", {"id": mid})[0][0] == 0


def test_neighborhood_is_bounded_and_connects_memory_to_sources(export_file: Path, store: Store, graph: GraphStore) -> None:
    _built(export_file, store, graph)
    mid = store.conn.execute("SELECT id FROM memories WHERE statement LIKE '%switched Atlas%'").fetchone()[0]
    nb = graph.neighborhood(mid, 2, limit=40)
    labels = {n["label"] for n in nb["nodes"]}
    assert {"Memory", "Entity", "Message"} <= labels and any(e["type"] == "SUPERSEDES" for e in nb["edges"])
    assert len(nb["nodes"]) <= 80 and all(e["source"] in {n["id"] for n in nb["nodes"]} for e in nb["edges"])
