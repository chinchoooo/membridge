from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import networkx as nx
import pytest

from membridge.graphalgo.communities import build_communities, build_entity_graph, detect, member_hash
from membridge.importers.service import run_import
from membridge.memory.community_summaries import refresh_community_summaries, summarize_communities_llm
from membridge.memory.consolidate import upsert
from membridge.memory.models import Candidate, EntityRef
from membridge.memory.pipeline import run_extraction
from membridge.memory.service import MemoryService
from membridge.retrieval.engine import Retriever, global_intent
from membridge.store.graph import GraphStore
from membridge.store.sqlite import Store

ROOT = Path(__file__).resolve().parents[1]


def _clusters(store: Store, n_groups: int = 3, size: int = 4) -> None:
    """Dense groups of entities (every pair co-mentioned) joined by a single weak bridge."""
    ents = [[f"G{g}E{i}" for i in range(size)] for g in range(n_groups)]
    ts = 1000.0
    for g, names in enumerate(ents):
        for i in range(size):
            for j in range(i + 1, size):
                ts += 1
                upsert(store, Candidate(statement=f"{names[i]} works with {names[j]} in group {g}.", category="fact",
                                        entities=[EntityRef(names[i]), EntityRef(names[j])], ref=f"r{ts}", ts=ts))  # fmt: skip
    upsert(store, Candidate(statement="G0E0 once met G1E0.", category="fact", entities=[EntityRef("G0E0"), EntityRef("G1E0")], ref="bridge", ts=5000.0))


def test_detection_recovers_planted_groups_and_is_deterministic(store: Store) -> None:
    _clusters(store)
    g = build_entity_graph(store)
    method, parts = detect(g, prefer_leiden=False)
    assert method == "louvain"
    fine = [set(p) for p in parts[0]]
    names = {r["id"]: r["name"] for r in store.conn.execute("SELECT id, name FROM live_entities")}
    found = sorted(sorted(names[e][:2] for e in cl) for cl in fine)
    assert found == [["G0"] * 4, ["G1"] * 4, ["G2"] * 4], found
    assert detect(g, prefer_leiden=False)[1] == parts  # same graph, same partition
    assert detect(nx.Graph(), prefer_leiden=False) == ("none", [])


def test_hierarchy_is_nested_persisted_and_ids_are_member_hashes(store: Store) -> None:
    _clusters(store)
    rep = build_communities(store, prefer_leiden=False)
    assert rep.levels[1] == 3 and rep.levels[2] >= 1 and rep.entities == 12
    c = store.conn
    for com in c.execute("SELECT * FROM communities WHERE level=1"):
        members = {r[0] for r in c.execute("SELECT entity_id FROM community_members WHERE community_id=?", (com["id"],))}
        assert com["id"] == f"com1_{member_hash(members)}" and com["parent_id"]
        parent = {r[0] for r in c.execute("SELECT entity_id FROM community_members WHERE community_id=?", (com["parent_id"],))}
        assert members <= parent, "every level-1 community must nest inside its level-2 parent"


def test_summaries_are_bottom_up_and_incremental(store: Store) -> None:
    _clusters(store)
    build_communities(store, prefer_leiden=False)
    first = refresh_community_summaries(store)
    assert first["community-L1"] == 3 and first["community-L2"] >= 1
    assert refresh_community_summaries(store) == {"community-L1": 0, "community-L2": 0, "removed": 0}  # nothing changed -> nothing rewritten
    # a new fact inside group 2 changes that community only
    upsert(store, Candidate(statement="G2E0 also trusts G2E3 deeply.", category="fact", entities=[EntityRef("G2E0"), EntityRef("G2E3")], ref="new", ts=9000.0))
    rep = build_communities(store, prefer_leiden=False)
    assert rep.removed_ids == set() and len(rep.kept_ids) == len(rep.kept_ids | rep.new_ids)  # same membership -> same ids
    second = refresh_community_summaries(store)
    assert second["community-L1"] == 1, "exactly the touched community is rewritten"
    assert second["community-L2"] <= 1


def test_summary_text_keeps_links_to_leaf_facts(store: Store) -> None:
    _clusters(store)
    build_communities(store, prefer_leiden=False)
    refresh_community_summaries(store)
    texts = [r[0] for r in store.conn.execute("SELECT text FROM summaries WHERE level='community-L1'")]
    assert all(t.startswith("Theme:") and "works with" in t for t in texts)
    area = store.conn.execute("SELECT text FROM summaries WHERE level='community-L2'").fetchone()[0]
    assert area.startswith("Area:") and area.count("\n- ") >= 3


def test_vanished_communities_lose_their_summaries(store: Store) -> None:
    _clusters(store)
    build_communities(store, prefer_leiden=False)
    refresh_community_summaries(store)
    store.conn.execute("UPDATE memories SET status='deleted'")
    build_communities(store, prefer_leiden=False)
    assert refresh_community_summaries(store)["removed"] >= 4
    assert store.conn.execute("SELECT COUNT(*) FROM summaries WHERE level LIKE 'community-%'").fetchone()[0] == 0


def test_empty_and_tiny_graphs_do_not_crash(store: Store) -> None:
    assert build_communities(store, prefer_leiden=False).levels == {1: 0, 2: 0}
    upsert(store, Candidate(statement="Lonely fact about Zed.", category="fact", entities=[EntityRef("Zed")], ref="z", ts=1.0))
    assert refresh_community_summaries(store)["community-L1"] == 0


def test_fidelity_gate_for_llm_community_summaries(store: Store) -> None:
    _clusters(store)
    build_communities(store, prefer_leiden=False)
    refresh_community_summaries(store)

    class Fake:
        name, is_remote = "fake", False

        def __init__(self, text: str) -> None:
            self.text = text

        def complete_json(self, system: str, user: str, schema: dict, *, max_tokens: int = 1200) -> dict:  # type: ignore[type-arg]
            return {"summary": self.text}

    good = summarize_communities_llm(store, Fake("Theme about G0E0 working with G0E1 in group 0. Several current facts are listed."))
    assert good["written"] >= 1
    bad_store_count = store.conn.execute("SELECT COUNT(*) FROM summaries WHERE method='extractive' AND level='community-L1'").fetchone()[0]
    out = summarize_communities_llm(store, Fake("Quantum Zebra Holdings acquired Narnia Industries in 2031 for 900 million."))
    assert out["rejected_fidelity"] == bad_store_count and out["written"] == 0


# ---- global questions + integration ------------------------------------------------------------------------------
@pytest.fixture
def multihop_store(tmp_path: Path) -> Store:
    spec = importlib.util.spec_from_file_location("multihop", ROOT / "eval" / "multihop.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    convs, _ = mod.build(n_projects=6, n_distractors=12)
    f = tmp_path / "e" / "conversations.json"
    f.parent.mkdir()
    f.write_text(json.dumps(convs), encoding="utf-8")
    s = Store(tmp_path / "m.db")
    run_import(s, f)
    run_extraction(s)  # pipeline builds communities + summaries
    return s


def test_global_intent_detection() -> None:
    assert global_intent("What am I working on overall?") and global_intent("summarize all my projects")
    assert not global_intent("What database does Orion use?")


def test_pipeline_builds_communities_and_global_question_uses_them(multihop_store: Store) -> None:
    s = multihop_store
    assert s.conn.execute("SELECT COUNT(*) FROM communities WHERE level=1").fetchone()[0] >= 2
    pkg = Retriever(s).search("What am I working on overall across my projects?", token_budget=1500)
    assert pkg["sufficient_evidence"] and pkg["stats"]["global_intent"] is True
    assert pkg["community_summaries"] and pkg["community_summaries"][0]["summary"].startswith(("Area:", "Theme:"))
    assert Retriever(s).search("What does Orion use for storage now?")["community_summaries"] == []  # specific questions skip them


def test_overview_service_and_graph_projection(multihop_store: Store, graph: GraphStore) -> None:
    svc = MemoryService(multihop_store, graph)
    ov = svc.get_overview()
    assert ov["areas"] or ov["themes"]
    graph.rebuild(multihop_store)
    n_com = multihop_store.conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0]
    assert graph.query("MATCH (c:Community) RETURN count(c)")[0][0] == n_com
    assert graph.query("MATCH (:Entity)-[x:MEMBER_OF]->(:Community) RETURN count(x)")[0][0] > 0
    assert graph.query("MATCH (:Community)-[x:CHILD_OF]->(:Community) RETURN count(x)")[0][0] > 0


def test_leiden_path_recovers_groups_and_yields_connected_communities(store: Store) -> None:
    pytest.importorskip("leidenalg")
    _clusters(store)
    rep = build_communities(store, prefer_leiden=True)
    assert rep.method == "leiden" and rep.levels[1] == 3 and rep.levels[2] >= 1
    g = build_entity_graph(store)
    for com in store.conn.execute("SELECT id FROM communities WHERE level=1"):
        members = [r[0] for r in store.conn.execute("SELECT entity_id FROM community_members WHERE community_id=?", (com["id"],))]
        assert nx.is_connected(g.subgraph(members)), "Leiden guarantees connected communities"
    m1 = {r[0] for r in store.conn.execute("SELECT id FROM communities")}
    build_communities(store, prefer_leiden=True)
    assert {r[0] for r in store.conn.execute("SELECT id FROM communities")} == m1  # deterministic with the fixed seed
