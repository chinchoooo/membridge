"""Every ```cypher block in docs/graph-schema.md must execute against a live, fully populated graph."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from membridge.importers.service import run_import
from membridge.memory.pipeline import run_extraction
from membridge.store.graph import GraphStore
from membridge.store.sqlite import Store

DOC = Path(__file__).resolve().parents[1] / "docs" / "graph-schema.md"


def _blocks() -> list[str]:
    return re.findall(r"```cypher\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)


def test_doc_has_the_expected_number_of_examples() -> None:
    assert len(_blocks()) >= 11


def test_every_documented_query_runs_and_core_ones_return_rows(export_file: Path, store: Store, graph: GraphStore) -> None:
    run_import(store, export_file)
    run_extraction(store)
    graph.rebuild(store)
    graph.compute_global_metrics()
    mem = store.conn.execute("SELECT id FROM memories WHERE statement LIKE '%switched Atlas%'").fetchone()[0]
    params = {"t": 1_700_000_000.0 + 15 * 86400, "id": mem}
    results = {}
    for q in _blocks():
        try:
            results[q.split("\n")[0][:60]] = graph.query(q, params if "$" in q else {})
        except Exception as e:  # pragma: no cover - failure message is the point
            pytest.fail(f"documented query failed: {q.splitlines()[0]}... -> {e}")
    nonempty = {k for k, v in results.items() if v}
    for must in ("MATCH (e:Entity {canonical:'atlas'})<-[:ABOUT]-(m:Memory)", "MATCH (newer:Memory)-[r:SUPERSEDES]->(older:Memory)",
                 "MATCH (m:Memory {id:$id})-[:SUPPORTED_BY]->(g:Message)-[:PART_OF]->(c:Conversation)"):  # fmt: skip
        assert must[:60] in nonempty, f"expected rows from: {must}"
