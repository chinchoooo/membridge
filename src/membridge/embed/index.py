"""Embedding index: stored in SQLite (canonical, rebuildable), searched by cosine, optionally mirrored to FalkorDB.

Every vector is keyed (record_id, kind, model) with a hash of the exact text that was embedded, so re-indexing
only touches records whose text changed, and a model switch never mixes vector spaces.
"""

from __future__ import annotations

import array
from typing import Any

from ..memory.models import sha
from ..store.sqlite import Store
from .embedder import Embedder


def pack_vec(v: list[float]) -> bytes:
    return array.array("f", v).tobytes()


def unpack_vec(b: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(b)
    return list(a)


def _texts(store: Store, kind: str) -> dict[str, str]:
    c = store.conn
    if kind == "fact":
        return {r[0]: r[1] for r in c.execute("SELECT id, statement FROM memories WHERE status!='deleted' AND sensitivity='normal'")}
    if kind == "entity":
        out = {}
        for e in c.execute("SELECT id, name FROM live_entities"):
            aliases = [r[0] for r in c.execute("SELECT alias_norm FROM entity_aliases WHERE entity_id=?", (e["id"],))]
            out[e["id"]] = e["name"] + (" (" + ", ".join(a for a in aliases if a != e["name"].lower())[:120] + ")" if len(aliases) > 1 else "")
        return out
    raise ValueError(kind)


def index_embeddings(store: Store, emb: Embedder, kinds: tuple[str, ...] = ("fact", "entity"), batch: int = 64) -> dict[str, int]:
    """Embed new/changed records, drop vectors of records that no longer exist. Idempotent."""
    c = store.conn
    stats = {"embedded": 0, "unchanged": 0, "removed": 0}
    for kind in kinds:
        texts = _texts(store, kind)
        have = {r[0]: r[1] for r in c.execute("SELECT record_id, text_hash FROM embeddings WHERE kind=? AND model=?", (kind, emb.name))}
        todo = [(rid, t) for rid, t in texts.items() if have.get(rid) != sha(emb.name, t)]
        stats["unchanged"] += len(texts) - len(todo)
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            for (rid, t), vec in zip(chunk, emb.embed([t for _, t in chunk]), strict=True):
                c.execute(
                    "INSERT OR REPLACE INTO embeddings(record_id, kind, model, dim, vector, text_hash) VALUES (?,?,?,?,?,?)",
                    (rid, kind, emb.name, emb.dim, pack_vec(vec), sha(emb.name, t)),
                )
            stats["embedded"] += len(chunk)
        gone = [rid for rid in have if rid not in texts]
        for rid in gone:
            c.execute("DELETE FROM embeddings WHERE record_id=? AND kind=? AND model=?", (rid, kind, emb.name))
        stats["removed"] += len(gone)
    c.commit()
    return stats


def rebuild_embeddings(store: Store, emb: Embedder, kinds: tuple[str, ...] = ("fact", "entity")) -> dict[str, int]:
    """Drop this model's vectors and recompute them from canonical text (the vectors are derived data)."""
    store.conn.execute("DELETE FROM embeddings WHERE model=?", (emb.name,))
    store.conn.commit()
    return index_embeddings(store, emb, kinds)


def vector_search(store: Store, emb: Embedder, query: str, *, k: int = 20, kind: str = "fact", min_sim: float = 0.0) -> list[tuple[str, float]]:
    """Cosine top-k over stored vectors (vectors are unit length, so cosine == dot product)."""
    rows = store.conn.execute("SELECT record_id, vector FROM embeddings WHERE kind=? AND model=?", (kind, emb.name)).fetchall()
    if not rows:
        return []
    q = emb.embed_query(query)
    try:
        import numpy as np

        m = np.array([unpack_vec(r["vector"]) for r in rows], dtype="float32")
        sims = (m @ np.array(q, dtype="float32")).tolist()
    except ImportError:  # no numpy: pure Python, fine for personal-scale corpora
        sims = [sum(a * b for a, b in zip(unpack_vec(r["vector"]), q, strict=True)) for r in rows]
    ranked = sorted(((rows[i]["record_id"], float(s)) for i, s in enumerate(sims) if s >= min_sim), key=lambda x: (-x[1], x[0]))
    return ranked[:k]


def similar_entities(store: Store, emb: Embedder, k: int = 3, min_sim: float = 0.82) -> dict[tuple[str, str], float]:
    """Entity pairs whose name embeddings are close: *candidates* for resolution (semantic blocking), never merges."""
    rows = store.conn.execute("SELECT record_id, vector FROM embeddings WHERE kind='entity' AND model=?", (emb.name,)).fetchall()
    if len(rows) < 2:
        return {}
    ids = [r["record_id"] for r in rows]
    try:
        import numpy as np

        m = np.array([unpack_vec(r["vector"]) for r in rows], dtype="float32")
        sim = m @ m.T
    except ImportError:
        vs = [unpack_vec(r["vector"]) for r in rows]
        sim = [[sum(a * b for a, b in zip(x, y, strict=True)) for y in vs] for x in vs]
    out: dict[tuple[str, str], float] = {}
    for i in range(len(ids)):
        row: Any = sim[i]
        order = sorted((j for j in range(len(ids)) if j != i), key=lambda j: -float(row[j]))[:k]
        for j in order:
            s = float(row[j])
            if s >= min_sim:
                a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
                out[(a, b)] = max(out.get((a, b), 0.0), s)
    return out
