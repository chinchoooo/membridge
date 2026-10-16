from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from membridge.importers.service import run_import
from membridge.memory.pipeline import run_extraction
from membridge.memory.service import MemoryService
from membridge.retrieval.engine import RetrievalConfig, Retriever, coverage, estimate_tokens, temporal_intent
from membridge.store.graph import GraphStore
from membridge.store.sqlite import Store

ROOT = Path(__file__).resolve().parents[1]


def _multihop():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("multihop", ROOT / "eval" / "multihop.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sample(export_file: Path, store: Store) -> Store:
    run_import(store, export_file)
    run_extraction(store)
    return store


@pytest.fixture
def mh(tmp_path: Path) -> tuple[Store, list[dict]]:  # type: ignore[type-arg]
    convs, qs = _multihop().build(n_projects=6, n_distractors=15)
    f = tmp_path / "mh" / "conversations.json"
    f.parent.mkdir()
    f.write_text(json.dumps(convs), encoding="utf-8")
    s = Store(tmp_path / "mh.db")
    run_import(s, f)
    run_extraction(s)
    return s, qs


@pytest.mark.parametrize(
    ("q", "intent"),
    [("What does Atlas use now?", "current"), ("Which database did Atlas use before SQLite?", "history"),
     ("What was the original plan?", "history"), ("Tell me about Atlas", "neutral"), ("what is the latest and the old plan", "neutral")],
)  # fmt: skip
def test_temporal_intent(q: str, intent: str) -> None:
    assert temporal_intent(q) == intent


def test_coverage_reranker_uses_light_stemming() -> None:
    assert coverage("I switched Atlas to SQLite", ["switching", "atlas"]) == 1.0
    assert coverage("unrelated sentence", ["atlas", "demo"]) == 0.0 and coverage("x", []) == 0.0


def test_history_question_surfaces_the_superseded_decision(sample: Store) -> None:
    pkg = Retriever(sample).search("Which database did Atlas use before SQLite?", limit=5)
    assert pkg["stats"]["temporal_intent"] == "history"
    assert any("decided to use PostgreSQL" in m["statement"] and m["status"] == "superseded" for m in pkg["memories"][:3])


def test_current_question_leads_with_the_superseder_via_lift(sample: Store) -> None:
    pkg = Retriever(sample).search("What storage does Atlas use now?", limit=5)
    assert pkg["stats"]["temporal_intent"] == "current"
    assert "SQLite" in pkg["memories"][0]["statement"] and pkg["memories"][0]["status"] == "current"
    assert any("supersession_lift" in m["channels"] for m in pkg["memories"])


def test_results_are_unique_sorted_and_explained(sample: Store) -> None:
    pkg = Retriever(sample).search("Atlas database storage deadline", limit=8)
    ids = [m["id"] for m in pkg["memories"]]
    assert len(ids) == len(set(ids)) and ids
    scores = [m["score"] for m in pkg["memories"]]
    assert scores == sorted(scores, reverse=True)
    assert all(m["why_selected"] and m["evidence"] for m in pkg["memories"])
    assert pkg["stats"]["channels"] and pkg["stats"]["graph_nodes_loaded"] > 0 and pkg["stats"]["ppr_pushes"] > 0


def test_token_budget_is_never_exceeded(sample: Store) -> None:
    r = Retriever(sample)
    for budget in (200, 350, 600, 1200, 4000):
        pkg = r.search("What did I decide about Atlas?", token_budget=budget)
        assert pkg["stats"]["tokens_estimate"] <= budget
        assert sum(estimate_tokens(str(m)) for m in pkg["memories"]) <= budget


def test_insufficient_evidence_stays_honest(sample: Store) -> None:
    pkg = Retriever(sample).search("What is my favourite colour and my shoe size?")
    assert pkg["sufficient_evidence"] is False and pkg["memories"] == []


@pytest.mark.parametrize("flag", ["use_ppr", "use_entities", "use_mmr", "use_knapsack"])
def test_every_ablation_switch_still_returns_valid_packages(sample: Store, flag: str) -> None:
    pkg = Retriever(sample, None, replace(RetrievalConfig(), **{flag: False})).search("What storage does Atlas use?", limit=5)
    assert pkg["stats"]["tokens_estimate"] <= 1500 and all(m["evidence"] for m in pkg["memories"])


def test_lexical_gap_questions_need_the_graph(mh: tuple[Store, list[dict]]) -> None:  # type: ignore[type-arg]
    store, qs = mh
    gap = [q for q in qs if q["kind"] == "lexical-gap-multihop"]
    full = Retriever(store)
    text_only = Retriever(store, None, replace(RetrievalConfig(), use_entities=False, use_ppr=False))

    def hits(r: Retriever) -> int:
        n = 0
        for q in gap:
            top = r.search(q["q"], limit=3)["memories"]
            n += any(q["expected"][0] in m["statement"] for m in top)
        return n

    assert hits(text_only) == 0 and hits(full) >= len(gap) - 1  # the graph is the only thing that answers these


def test_hub_distractors_do_not_swamp_the_answer(mh: tuple[Store, list[dict]]) -> None:  # type: ignore[type-arg]
    store, qs = mh
    direct = [q for q in qs if q["kind"] == "direct"]
    ok = sum(any(q["expected"][0] in m["statement"] for m in Retriever(store).search(q["q"], limit=3)["memories"]) for q in direct)
    assert ok == len(direct)  # 15 projects all using the "Python" hub must not drown the real answers


def test_graph_and_sqlite_modes_return_identical_rankings(sample: Store, graph: GraphStore) -> None:
    graph.rebuild(sample)
    for q in ("What storage does Atlas use now?", "Which database did Atlas use before SQLite?", "Atlas prototype deadline", "pgbouncer connection slots"):
        g = Retriever(sample, graph).search(q, limit=6)
        d = Retriever(sample, None).search(q, limit=6)
        assert g["stats"]["mode"] == "graph" and d["stats"]["mode"] == "sqlite-degraded"
        assert [m["id"] for m in g["memories"]] == [m["id"] for m in d["memories"]], q


def test_as_of_does_not_apply_the_history_penalty(sample: Store) -> None:
    import datetime as dt

    ts = dt.datetime(2023, 11, 20, tzinfo=dt.UTC).timestamp()
    pkg = MemoryService(sample).search_memory("Atlas storage database", filters={"as_of": ts})
    assert any("PostgreSQL" in m["statement"] for m in pkg["memories"][:2])
