from __future__ import annotations

import math

import pytest

from membridge.embed.embedder import HashingEmbedder, build_embedder
from membridge.embed.index import (
    index_embeddings,
    pack_vec,
    rebuild_embeddings,
    similar_entities,
    unpack_vec,
    vector_search,
)
from membridge.memory.consolidate import upsert
from membridge.memory.models import Candidate, EntityRef
from membridge.resolve.entities import resolve_entities
from membridge.retrieval.engine import RetrievalConfig, Retriever
from membridge.store.graph import GraphStore
from membridge.store.sqlite import Store

FACTS = ["I prefer dark mode in every editor.", "My manager is Priya Raman.", "The staging database runs on PostgreSQL 15.", "I live with two cats named Pixel and Byte."]


def _fill(store: Store) -> list[str]:
    return [upsert(store, Candidate(statement=f, category="fact", ref=f"r{i}", ts=1000.0 + i, verification="user_stated")).memory_id for i, f in enumerate(FACTS)]


def test_hashing_embedder_is_deterministic_unit_length_and_lexically_sensible() -> None:
    e = HashingEmbedder()
    a, b = e.embed(["the staging database runs on PostgreSQL", "PostgreSQL database for staging"])
    c = e.embed(["my manager is Priya"])[0]
    assert e.embed(["x y z"]) == e.embed(["x y z"]) and math.isclose(sum(x * x for x in a), 1.0, abs_tol=1e-9)
    assert sum(x * y for x, y in zip(a, b, strict=True)) > sum(x * y for x, y in zip(a, c, strict=True))
    assert build_embedder("none") is None and build_embedder("hashing").dim == 256  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        build_embedder("bogus")


def test_vector_pack_round_trip() -> None:
    v = [0.25, -1.5, 3.0, 0.0]
    assert unpack_vec(pack_vec(v)) == v


def test_indexing_is_incremental_and_cleans_up(store: Store) -> None:
    emb = HashingEmbedder()
    ids = _fill(store)
    assert index_embeddings(store, emb, ("fact",)) == {"embedded": 4, "unchanged": 0, "removed": 0}
    assert index_embeddings(store, emb, ("fact",)) == {"embedded": 0, "unchanged": 4, "removed": 0}
    store.conn.execute("UPDATE memories SET statement='I prefer light mode.' WHERE id=?", (ids[0],))
    assert index_embeddings(store, emb, ("fact",))["embedded"] == 1  # only the changed text is re-embedded
    store.conn.execute("UPDATE memories SET status='deleted' WHERE id=?", (ids[1],))
    assert index_embeddings(store, emb, ("fact",))["removed"] == 1
    assert store.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 3


def test_rebuild_reproduces_identical_vectors_and_models_never_mix(store: Store) -> None:
    emb = HashingEmbedder()
    _fill(store)
    index_embeddings(store, emb, ("fact",))
    before = sorted(tuple(r) for r in store.conn.execute("SELECT record_id, vector FROM embeddings"))
    rebuild_embeddings(store, emb, ("fact",))
    assert sorted(tuple(r) for r in store.conn.execute("SELECT record_id, vector FROM embeddings")) == before
    other = HashingEmbedder(128)
    index_embeddings(store, other, ("fact",))
    assert {r[0] for r in store.conn.execute("SELECT DISTINCT model FROM embeddings")} == {emb.name, other.name}
    assert all(len(v) == 128 for _, v in vector_search_dim(store, other))


def vector_search_dim(store: Store, emb: HashingEmbedder) -> list[tuple[str, list[float]]]:
    return [(r["record_id"], unpack_vec(r["vector"])) for r in store.conn.execute("SELECT record_id, vector FROM embeddings WHERE model=?", (emb.name,))]


def test_vector_search_orders_by_cosine_and_honours_floor(store: Store) -> None:
    emb = HashingEmbedder()
    ids = _fill(store)
    index_embeddings(store, emb, ("fact",))
    hits = vector_search(store, emb, "which database does staging run on PostgreSQL", k=4)
    assert hits[0][0] == ids[2] and [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
    assert vector_search(store, emb, "zzz qqq", k=4, min_sim=0.9) == []


def test_retriever_vector_channel_is_opt_in(store: Store) -> None:
    emb = HashingEmbedder()
    ids = _fill(store)
    index_embeddings(store, emb, ("fact",))
    off = Retriever(store, None, RetrievalConfig(use_vector=False), emb).search("staging database PostgreSQL")
    on = Retriever(store, None, RetrievalConfig(use_vector=True, vector_min_sim=0.2), emb).search("staging database PostgreSQL")
    assert "vector" not in off["stats"]["channels"] and "vector" in on["stats"]["channels"]
    assert on["memories"][0]["id"] == ids[2]


def test_graph_vector_index_matches_brute_force(store: Store, graph: GraphStore) -> None:
    emb = HashingEmbedder()
    _fill(store)
    index_embeddings(store, emb, ("fact",))
    graph.rebuild(store)
    assert graph.sync_embeddings(store, emb.name, emb.dim) == 4
    q = "which database does staging run on PostgreSQL"
    via_graph = graph.vector_query(emb.embed_query(q), 4)
    brute = vector_search(store, emb, q, k=4)
    assert via_graph[0][0] == brute[0][0]  # same nearest neighbour as exact search
    assert via_graph[0][1] == pytest.approx(brute[0][1], abs=1e-3)  # distance -> cosine conversion is right
    on = Retriever(store, graph, RetrievalConfig(use_vector=True, vector_min_sim=0.2, vector_backend="graph"), emb).search(q)
    assert "vector" in on["stats"]["channels"]


class _FixedEmbedder:
    """Controlled vectors: 'alpha systems' and 'beta platform' are near-identical, 'gamma' is orthogonal."""

    name, dim = "fixed-3", 3
    _V = {"alpha systems": [1.0, 0.0, 0.0], "beta platform": [0.97, 0.243, 0.0], "gamma": [0.0, 0.0, 1.0]}

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            key = next(k for k in self._V if t.lower().startswith(k))
            v = self._V[key]
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def test_semantic_blocking_proposes_candidates_but_never_auto_merges(store: Store) -> None:
    emb = _FixedEmbedder()
    for i, name in enumerate(["Alpha Systems", "Beta Platform", "Gamma"]):
        upsert(store, Candidate(statement=f"We rely on {name} {i}.", category="fact", entities=[EntityRef(name)], ref=f"e{i}", ts=10.0 + i))
    index_embeddings(store, emb, ("entity",))  # type: ignore[arg-type]
    close = similar_entities(store, emb, min_sim=0.9)  # type: ignore[arg-type]
    assert len(close) == 1  # only the alpha/beta pair; gamma is orthogonal
    rep = resolve_entities(store, embedder=emb)
    assert rep.candidate_pairs == 1 and rep.auto_merged == []  # embedding evidence alone never merges
    assert store.conn.execute("SELECT COUNT(*) FROM live_entities").fetchone()[0] == 3


def _real() -> object:
    pytest.importorskip("fastembed")
    try:
        return build_embedder("fastembed")
    except Exception as e:  # model files unavailable offline
        pytest.skip(f"fastembed model unavailable: {type(e).__name__}")


def test_real_model_captures_paraphrase_that_lexical_search_misses(store: Store) -> None:
    emb = _real()
    ids = _fill(store)
    for i in range(20):
        upsert(store, Candidate(statement=f"Note number {i} about the quarterly planning cycle.", category="fact", ref=f"n{i}", ts=2000.0 + i))
    index_embeddings(store, emb, ("fact",))  # type: ignore[arg-type]
    q = "What engine backs the pre-production datastore?"  # no content word shared with the fact
    lexical = Retriever(store).search(q, limit=3)["memories"]
    semantic = Retriever(store, None, RetrievalConfig(use_vector=True), emb).search(q, limit=3)["memories"]  # type: ignore[arg-type]
    assert ids[2] not in [m["id"] for m in lexical]
    assert semantic[0]["id"] == ids[2]
