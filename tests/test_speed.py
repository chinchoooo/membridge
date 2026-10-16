from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

from membridge.graphalgo.ppr_cache import build_ppr_cache, cache_is_fresh
from membridge.importers.service import run_import
from membridge.memory.pipeline import run_extraction
from membridge.memory.service import MemoryService
from membridge.retrieval.engine import RetrievalConfig, Retriever
from membridge.store.sqlite import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample(export_file: Path, store: Store) -> Store:
    run_import(store, export_file)
    run_extraction(store)
    return store


@pytest.fixture
def mh(tmp_path: Path) -> tuple[Store, list[dict]]:  # type: ignore[type-arg]
    spec = importlib.util.spec_from_file_location("multihop", ROOT / "eval" / "multihop.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    convs, qs = mod.build(n_projects=8, n_distractors=20)
    f = tmp_path / "mh" / "conversations.json"
    f.parent.mkdir()
    f.write_text(json.dumps(convs), encoding="utf-8")
    s = Store(tmp_path / "mh.db")
    run_import(s, f)
    run_extraction(s)
    return s, qs


# ---- generation counter ---------------------------------------------------------------------------------
def test_generation_bumps_on_content_writes_but_not_on_read_audit(sample: Store) -> None:
    g0 = sample.generation()
    sample.audit_read("t", "search", None, q=1)
    sample.flush_audit()
    assert sample.generation() == g0, "audit traffic must not invalidate caches"
    MemoryService(sample).save_memory("The demo port is 8123", "fact", "user_stated")
    assert sample.generation() > g0


def test_generation_is_visible_across_connections(sample: Store) -> None:
    other = Store(sample.path)
    g = other.generation()
    MemoryService(sample).save_memory("Another fact about Atlas deployment", "fact", "user_stated")
    assert other.generation() > g  # a write by one process invalidates another process's cache
    other.close()


# ---- result cache -------------------------------------------------------------------------------------------
def test_repeat_queries_hit_the_cache_and_are_far_faster(sample: Store) -> None:
    r = Retriever(sample)
    first = r.search("What storage does Atlas use now?", limit=5)
    assert first["stats"]["cache"] == "miss"
    times = []
    for _ in range(20):
        t = time.perf_counter()
        again = r.search("what storage does   ATLAS use now?", limit=5)  # case/whitespace-insensitive key
        times.append((time.perf_counter() - t) * 1000)
    assert again["stats"]["cache"] == "hit" and [m["id"] for m in again["memories"]] == [m["id"] for m in first["memories"]]
    assert sorted(times)[len(times) // 2] < 1.0, "a cached memory call must cost well under a millisecond"


def test_cache_is_invalidated_by_any_write_and_respects_parameters(sample: Store) -> None:
    svc = MemoryService(sample)
    a = svc.search_memory("Atlas demo port")
    assert svc.search_memory("Atlas demo port")["stats"]["cache"] == "hit"
    svc.save_memory("The Atlas demo runs on port 8123", "fact", "user_stated")
    b = svc.search_memory("Atlas demo port")
    assert b["stats"]["cache"] == "miss" and any("8123" in m["statement"] for m in b["memories"]) and not any("8123" in m["statement"] for m in a["memories"])
    assert svc.search_memory("Atlas demo port", limit=3)["stats"]["cache"] == "miss"  # different parameters, different entry
    other = MemoryService(Store(sample.path))  # a separate process/connection writes
    other.save_memory("Atlas demo also exposes metrics", "fact", "user_stated")
    assert svc.search_memory("Atlas demo port")["stats"]["cache"] == "miss"


# ---- warm index ---------------------------------------------------------------------------------------------------
def test_index_links_entities_by_name_and_alias_in_linear_time(sample: Store) -> None:
    from membridge.memory.consolidate import upsert
    from membridge.memory.models import Candidate, EntityRef
    from membridge.resolve.entities import resolve_entities

    for i, n in enumerate(["Postgres", "PostgreSQL"]):
        upsert(sample, Candidate(statement=f"I rely on {n} {i}.", category="fact", entities=[EntityRef(n)], ref=f"x{i}", ts=9.0 + i))
    resolve_entities(sample)
    ix = Retriever(sample)._ix()
    assert {n for n, _ in ix.link("tell me about atlas and pg please").values()} >= {"Atlas", "PostgreSQL"}
    assert ix.link("nothing relevant here") == {}
    t = time.perf_counter()
    for _ in range(200):
        ix.link("what is the status of the atlas project with postgres and sqlite")
    assert (time.perf_counter() - t) / 200 < 0.002  # well under 2 ms per link even in Python


def test_index_rebuilds_only_after_a_write(sample: Store) -> None:
    r = Retriever(sample)
    ix1 = r._ix()
    assert r._ix() is ix1
    MemoryService(sample).save_memory("Index rebuild check about Atlas", "fact", "user_stated")
    assert r._ix() is not ix1


# ---- PPR cache --------------------------------------------------------------------------------------------------------
def test_ppr_cache_is_fresh_after_consolidation_and_stale_after_a_write(sample: Store) -> None:
    assert cache_is_fresh(sample)
    r = Retriever(sample)
    assert r.search("What storage does Atlas use?")["stats"]["ppr_source"] == "cache"
    sample.conn.execute("UPDATE memories SET statement=statement || ' ' WHERE id=(SELECT id FROM memories LIMIT 1)")
    sample.conn.commit()
    assert not cache_is_fresh(sample)
    assert r.search("What storage does Atlas use??")["stats"]["ppr_source"] == "live"  # stale cache -> correct live fallback
    build_ppr_cache(sample)
    assert cache_is_fresh(sample)


def test_cached_ppr_agrees_with_live_ppr(mh: tuple[Store, list[dict]]) -> None:  # type: ignore[type-arg]
    store, qs = mh
    cached = Retriever(store, None, RetrievalConfig(use_ppr_cache=True))
    live = Retriever(store, None, RetrievalConfig(use_ppr_cache=False))
    top1 = overlap = n = 0
    for q in qs[:32]:
        a = cached.search(q["q"], limit=5, filters=q.get("filters"))
        b = live.search(q["q"], limit=5, filters=q.get("filters"))
        assert a["stats"]["ppr_source"] == "cache" and b["stats"]["ppr_source"] == "live"
        ia, ib = [m["id"] for m in a["memories"]], [m["id"] for m in b["memories"]]
        n += 1
        top1 += bool(ia and ib and ia[0] == ib[0])
        overlap += len(set(ia[:5]) & set(ib[:5])) / max(1, len(set(ib[:5])))
    assert top1 / n >= 0.95 and overlap / n >= 0.9, (top1 / n, overlap / n)


def test_cached_answers_stay_correct_on_the_benchmark(mh: tuple[Store, list[dict]]) -> None:  # type: ignore[type-arg]
    store, qs = mh
    r = Retriever(store)
    ok = 0
    for q in qs:
        mems = r.search(q["q"], limit=5, filters=q.get("filters"))["memories"]
        hit = all(any(e in m["statement"] for m in mems) for e in q["expected"])
        cur = all(any(c in m["statement"] and m["status"] == "current" for m in mems) for c in q.get("current", []))
        bad = any(f in m["statement"] for m in mems for f in q.get("forbid", []))
        ok += hit and cur and not bad
    assert ok == len(qs)


def test_explanations_survive_the_cache(sample: Store) -> None:
    pkg = Retriever(sample).search("What is the Atlas storage decision?", limit=5)
    assert pkg["stats"]["ppr_source"] == "cache" and pkg["graph_paths"], "graph paths must be explainable from cached parents"


def test_idf_pruning_keeps_the_rare_term_and_leaves_small_corpora_alone(sample: Store, tmp_path: Path) -> None:
    r = Retriever(sample)
    assert r._fts_terms("messages_fts", ["what", "atlas", "now"]) == ["what", "atlas", "now"]  # 16 messages: untouched
    spec = importlib.util.spec_from_file_location("scale", ROOT / "eval" / "scale.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = tmp_path / "big" / "conversations.json"
    f.parent.mkdir()
    f.write_text(json.dumps(mod.build(400)), encoding="utf-8")
    big = Store(tmp_path / "big.db")
    run_import(big, f)
    run_extraction(big)
    rb = Retriever(big)
    kept = rb._fts_terms("messages_fts", ["kamilu0", "storage", "service", "project"])
    assert "kamilu0" in kept or len(kept) == 1
    assert len(kept) < 4, "ubiquitous terms must be dropped"
    assert rb._fts_terms("messages_fts", ["zzzznotaword", "qqqqnope"]) == []
    pkg = rb.search(f"What does {mod.name(5)} use for storage now?", limit=5)
    assert mod.name(5) in pkg["memories"][0]["statement"]
