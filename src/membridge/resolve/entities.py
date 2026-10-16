"""Entity resolution orchestration: block -> score -> tier -> cluster -> merge (reversibly).

Three tiers (see ADR): score >= tau_hi merges automatically; tau_lo <= score < tau_hi becomes a pending
`entity_link_candidates` row for a human or an LLM adjudicator; below tau_lo is treated as distinct (and acts
as a soft cannot-link inside clusters). Every merge is logged with a snapshot so `split_merge` can undo it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..memory.models import canonical
from ..store.sqlite import Store
from .blocking import candidate_pairs, compact, reduction_ratio
from .cluster import UnionFind
from .score import ALIAS_DICT, cannot_link, features, score

TAU_HI = 0.88
TAU_LO = 0.65


@dataclass
class ResolutionReport:
    entities: int = 0
    candidate_pairs: int = 0
    reduction_ratio: float = 0.0
    auto_merged: list[dict[str, Any]] = field(default_factory=list)
    pending: int = 0
    blocked_cannot_link: int = 0
    refused_by_guard: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False


def seed_aliases(store: Store) -> int:
    """Every live entity is an alias of itself; makes alias lookup the single entry point."""
    n = 0
    now = time.time()
    for e in store.conn.execute("SELECT id, canonical, name FROM live_entities").fetchall():
        cur = store.conn.execute(
            "INSERT OR IGNORE INTO entity_aliases(entity_id, alias_norm, alias, source, first_seen) VALUES (?,?,?,?,?)",
            (e["id"], e["canonical"], e["name"], "extracted", now),
        )
        n += cur.rowcount
    return n


def lookup_alias(store: Store, text: str) -> str | None:
    """Entity id (a live survivor) that `text` refers to, via aliases and merge pointers."""
    norm = canonical(text)
    row = store.conn.execute(
        "SELECT a.entity_id FROM entity_aliases a JOIN live_entities e ON e.id=a.entity_id WHERE a.alias_norm=? LIMIT 1", (norm,)
    ).fetchone()
    return None if row is None else str(row[0])


def root_entity(store: Store, eid: str) -> str:
    seen: set[str] = set()
    while eid not in seen:
        seen.add(eid)
        row = store.conn.execute("SELECT merged_into FROM entities WHERE id=?", (eid,)).fetchone()
        if row is None or row[0] is None:
            return eid
        eid = str(row[0])
    return eid


def _context_sets(store: Store) -> dict[str, set[str]]:
    ctx: dict[str, set[str]] = {}
    for r in store.conn.execute("SELECT entity_id, conversation_id FROM entity_mentions"):
        ctx.setdefault(r[0], set()).add("c:" + r[1])
    for r in store.conn.execute(
        "SELECT a.entity_id AS e, b.entity_id AS o FROM memory_entities a JOIN memory_entities b "
        "ON a.memory_id=b.memory_id AND a.entity_id!=b.entity_id"
    ):
        ctx.setdefault(r["e"], set()).add("e:" + r["o"])
    return ctx


def _alias_targets() -> dict[str, str]:
    return {compact(k): compact(v) for k, v in ALIAS_DICT.items()}


def resolve_entities(
    store: Store, *, dry_run: bool = False, tau_hi: float = TAU_HI, tau_lo: float = TAU_LO, max_cluster: int = 8,
    decided_by: str = "auto", embedder: Any = None,
) -> ResolutionReport:  # fmt: skip
    rep = ResolutionReport(dry_run=dry_run)
    c = store.conn
    if not dry_run:
        seed_aliases(store)
    ents = {r["id"]: r for r in c.execute("SELECT id, canonical, name, type FROM live_entities")}
    mem_counts = {r[0]: r[1] for r in c.execute("SELECT entity_id, COUNT(*) FROM memory_entities GROUP BY entity_id")}
    rep.entities = len(ents)
    ctx = _context_sets(store)
    decided = {(r["a_id"], r["b_id"]): r for r in c.execute("SELECT * FROM entity_link_candidates WHERE status IN ('accepted','rejected')")}

    pairs = candidate_pairs({i: e["canonical"] for i, e in ents.items()}, _alias_targets())
    emb_sim: dict[tuple[str, str], float] = {}
    if embedder is not None:  # semantic blocking: close name embeddings become *candidates*, never automatic merges
        from ..embed.index import index_embeddings, similar_entities

        index_embeddings(store, embedder, ("entity",))
        emb_sim = similar_entities(store, embedder)
        pairs |= {p for p in emb_sim if p[0] in ents and p[1] in ents}
    rep.candidate_pairs = len(pairs)
    rep.reduction_ratio = round(reduction_ratio(len(ents), len(pairs)), 4)

    uf = UnionFind(ents, max_size=max_cluster)
    scored: dict[tuple[str, str], tuple[float, str, Any]] = {}
    for a, b in sorted(pairs):
        ea, eb = ents[a], ents[b]
        prior = decided.get((a, b))
        if prior is not None and prior["status"] == "rejected":
            uf.forbid(a, b)
            continue
        if (why := cannot_link(ea["canonical"], eb["canonical"], ea["type"], eb["type"])) and not (prior and prior["status"] == "accepted"):
            uf.forbid(a, b)
            rep.blocked_cannot_link += 1
            continue
        f = features(ea["canonical"], eb["canonical"], ctx.get(a, set()), ctx.get(b, set()), emb_sim.get((a, b)))
        sc, method = score(f)
        scored[(a, b)] = (sc, method, f)
        if sc < tau_lo:
            uf.forbid(a, b)  # known-different: soft cannot-link inside any cluster
    accepted: list[tuple[str, str, float, str, Any]] = []
    for (a, b), (sc, method, f) in sorted(scored.items(), key=lambda kv: -kv[1][0]):
        prior = decided.get((a, b))
        if prior is not None and prior["status"] == "accepted":
            accepted.append((a, b, 1.0, "user", f))
        elif sc >= tau_hi:
            accepted.append((a, b, sc, method, f))
        elif sc >= tau_lo:
            rep.pending += 1
            if not dry_run:
                c.execute(
                    "INSERT OR IGNORE INTO entity_link_candidates(a_id, b_id, score, features_json, status) VALUES (?,?,?,?,'pending')",
                    (a, b, sc, json.dumps(f.as_dict())),
                )
    for a, b, _sc, _method, _f in accepted:  # strongest first, so weak edges cannot chain clusters
        ok, why = uf.union(a, b)
        if not ok and why != "same":
            rep.refused_by_guard.append({"a": ents[a]["name"], "b": ents[b]["name"], "score": sc, "reason": why})
    for members in uf.clusters():
        if len(members) < 2:
            continue
        keep = max(members, key=lambda i: (mem_counts.get(i, 0), -len(ents[i]["canonical"]), i))
        for absorbed in sorted(members - {keep}):
            pair = (absorbed, keep) if absorbed < keep else (keep, absorbed)
            sc, method, f = scored.get(pair, (1.0, "user", None))
            rep.auto_merged.append({"absorbed": ents[absorbed]["name"], "into": ents[keep]["name"], "score": sc, "method": method})
            if not dry_run:
                merge_entities(store, absorbed, keep, score=sc, method=method if method in ("exact", "alias_dict", "lsh_fuzzy", "embedding", "user") else "lsh_fuzzy",
                               features=f.as_dict() if f else {}, decided_by=decided_by)  # fmt: skip
    if not dry_run:
        c.commit()
    return rep


def merge_entities(store: Store, absorbed: str, into: str, *, score: float, method: str, features: dict[str, Any], decided_by: str) -> int:
    """Re-point everything from `absorbed` to `into`, logging a snapshot so the merge can be undone."""
    c = store.conn
    if absorbed == into:
        raise ValueError("cannot merge an entity into itself")
    snap: dict[str, Any] = {"memory_entities": [], "mentions": [], "aliases": [], "edges": [], "subject_memories": []}
    keep_me = {(r[0], r[1]) for r in c.execute("SELECT memory_id, role FROM memory_entities WHERE entity_id=?", (into,))}
    for r in c.execute("SELECT memory_id, role FROM memory_entities WHERE entity_id=?", (absorbed,)).fetchall():
        snap["memory_entities"].append({"memory_id": r[0], "role": r[1], "dup": (r[0], r[1]) in keep_me})
        c.execute("INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, role) VALUES (?,?,?)", (r[0], into, r[1]))
    c.execute("DELETE FROM memory_entities WHERE entity_id=?", (absorbed,))
    keep_mn = {r[0]: r[1] for r in c.execute("SELECT conversation_id, count FROM entity_mentions WHERE entity_id=?", (into,))}
    for r in c.execute("SELECT conversation_id, count FROM entity_mentions WHERE entity_id=?", (absorbed,)).fetchall():
        snap["mentions"].append({"conversation_id": r[0], "count": r[1], "had": r[0] in keep_mn})
        c.execute("INSERT OR REPLACE INTO entity_mentions(entity_id, conversation_id, count) VALUES (?,?,?)", (into, r[0], keep_mn.get(r[0], 0) + r[1]))
    c.execute("DELETE FROM entity_mentions WHERE entity_id=?", (absorbed,))
    for r in c.execute("SELECT alias_norm, alias, source FROM entity_aliases WHERE entity_id=?", (absorbed,)).fetchall():
        snap["aliases"].append(dict(r))
        c.execute("INSERT OR IGNORE INTO entity_aliases(entity_id, alias_norm, alias, source, first_seen) VALUES (?,?,?,'merge',?)", (into, r["alias_norm"], r["alias"], time.time()))
    c.execute("DELETE FROM entity_aliases WHERE entity_id=?", (absorbed,))
    for r in c.execute("SELECT * FROM entity_edges WHERE src_entity=? OR dst_entity=?", (absorbed, absorbed)).fetchall():
        snap["edges"].append(dict(r))
    c.execute("UPDATE OR IGNORE entity_edges SET src_entity=? WHERE src_entity=?", (into, absorbed))
    c.execute("UPDATE OR IGNORE entity_edges SET dst_entity=? WHERE dst_entity=?", (into, absorbed))
    c.execute("DELETE FROM entity_edges WHERE src_entity=? OR dst_entity=?", (absorbed, absorbed))
    a_row = c.execute("SELECT canonical, type FROM entities WHERE id=?", (absorbed,)).fetchone()
    i_row = c.execute("SELECT canonical, type FROM entities WHERE id=?", (into,)).fetchone()
    for r in c.execute("SELECT id FROM memories WHERE subject=? OR subject_entity_id=?", (a_row["canonical"], absorbed)).fetchall():
        snap["subject_memories"].append({"id": r[0], "old_subject": a_row["canonical"]})
    c.execute("UPDATE memories SET subject=?, subject_entity_id=? WHERE subject=? OR subject_entity_id=?", (i_row["canonical"], into, a_row["canonical"], absorbed))
    snap["absorbed_type"], snap["into_type_before"] = a_row["type"], i_row["type"]
    if i_row["type"] == "concept" and a_row["type"] != "concept":
        c.execute("UPDATE entities SET type=? WHERE id=?", (a_row["type"], into))
    c.execute("INSERT OR IGNORE INTO entity_aliases(entity_id, alias_norm, alias, source, first_seen) VALUES (?,?,?,'merge',?)",
              (into, a_row["canonical"], a_row["canonical"], time.time()))  # fmt: skip
    c.execute("UPDATE entities SET merged_into=? WHERE id=?", (into, absorbed))
    cur = c.execute(
        "INSERT INTO entity_merges(absorbed_id, into_id, score, method, features_json, decided_by, decided_at, snapshot_json) VALUES (?,?,?,?,?,?,?,?)",
        (absorbed, into, score, method, json.dumps(features), decided_by, time.time(), json.dumps(snap)),
    )
    a, b = (absorbed, into) if absorbed < into else (into, absorbed)
    c.execute("INSERT OR REPLACE INTO entity_link_candidates(a_id, b_id, score, features_json, status, decided_by, decided_at) VALUES (?,?,?,?, 'accepted', ?, ?)",
              (a, b, score, json.dumps(features), decided_by, time.time()))  # fmt: skip
    store.audit(decided_by, "merge_entities", absorbed, into=into, score=score, method=method)
    return int(cur.lastrowid or 0)


def split_merge(store: Store, merge_id: int, decided_by: str = "user") -> dict[str, Any]:
    """Undo a merge exactly, and record the pair as cannot-link so it is not merged again."""
    c = store.conn
    m = c.execute("SELECT * FROM entity_merges WHERE id=? AND reversed_at IS NULL", (merge_id,)).fetchone()
    if m is None:
        raise ValueError(f"no active merge with id {merge_id}")
    snap = json.loads(m["snapshot_json"])
    ab, into = m["absorbed_id"], m["into_id"]
    for r in snap["memory_entities"]:
        c.execute("INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, role) VALUES (?,?,?)", (r["memory_id"], ab, r["role"]))
        if not r["dup"]:
            c.execute("DELETE FROM memory_entities WHERE memory_id=? AND entity_id=? AND role=?", (r["memory_id"], into, r["role"]))
    for r in snap["mentions"]:
        cur_n = c.execute("SELECT count FROM entity_mentions WHERE entity_id=? AND conversation_id=?", (into, r["conversation_id"])).fetchone()
        c.execute("INSERT OR REPLACE INTO entity_mentions(entity_id, conversation_id, count) VALUES (?,?,?)", (ab, r["conversation_id"], r["count"]))
        if cur_n is not None:
            left = cur_n[0] - r["count"]
            if left > 0 and r["had"]:
                c.execute("UPDATE entity_mentions SET count=? WHERE entity_id=? AND conversation_id=?", (left, into, r["conversation_id"]))
            else:
                c.execute("DELETE FROM entity_mentions WHERE entity_id=? AND conversation_id=?", (into, r["conversation_id"]))
    for r in snap["aliases"]:
        c.execute("DELETE FROM entity_aliases WHERE entity_id=? AND alias_norm=?", (into, r["alias_norm"]))
        c.execute("INSERT OR IGNORE INTO entity_aliases(entity_id, alias_norm, alias, source, first_seen) VALUES (?,?,?,?,?)", (ab, r["alias_norm"], r["alias"], r["source"], time.time()))
    absorbed_canon = c.execute("SELECT canonical FROM entities WHERE id=?", (ab,)).fetchone()[0]
    c.execute("DELETE FROM entity_aliases WHERE entity_id=? AND alias_norm=? AND source='merge'", (into, absorbed_canon))
    for r in snap["edges"]:
        c.execute("DELETE FROM entity_edges WHERE fact_id=? AND predicate=?", (r["fact_id"], r["predicate"]))
        c.execute("INSERT OR IGNORE INTO entity_edges(src_entity, predicate, dst_entity, fact_id, weight, valid_at, invalid_at) VALUES (?,?,?,?,?,?,?)",
                  (r["src_entity"], r["predicate"], r["dst_entity"], r["fact_id"], r["weight"], r["valid_at"], r["invalid_at"]))  # fmt: skip
    for r in snap["subject_memories"]:
        c.execute("UPDATE memories SET subject=?, subject_entity_id=? WHERE id=?", (r["old_subject"], ab, r["id"]))
    c.execute("UPDATE entities SET type=? WHERE id=?", (snap["into_type_before"], into))
    c.execute("UPDATE entities SET merged_into=NULL WHERE id=?", (ab,))
    c.execute("UPDATE entity_merges SET reversed_at=? WHERE id=?", (time.time(), merge_id))
    siblings = [r[0] for r in c.execute("SELECT id FROM entities WHERE merged_into=? AND id!=?", (into, ab))]
    for other in siblings:
        x, y = (ab, other) if ab < other else (other, ab)
        c.execute(
            "INSERT OR REPLACE INTO entity_link_candidates(a_id, b_id, score, features_json, status, decided_by, decided_at) "
            "VALUES (?,?,?,?, 'rejected', ?, ?)",
            (x, y, m["score"], "{}", decided_by, time.time()),
        )
    a, b = (ab, into) if ab < into else (into, ab)
    c.execute("INSERT OR REPLACE INTO entity_link_candidates(a_id, b_id, score, features_json, status, decided_by, decided_at) VALUES (?,?,?,?, 'rejected', ?, ?)",
              (a, b, m["score"], m["features_json"], decided_by, time.time()))  # fmt: skip
    store.audit(decided_by, "split_merge", ab, into=into, merge_id=merge_id)
    c.commit()
    return {"restored": ab, "from": into, "now_cannot_link": True, "also_rejected_against": siblings}


def decide_pair(store: Store, a: str, b: str, accept: bool, decided_by: str = "user") -> dict[str, Any]:
    """Human (or LLM) verdict on a pending candidate. Accept merges now; reject records a cannot-link."""
    x, y = (a, b) if a < b else (b, a)
    c = store.conn
    row = c.execute("SELECT * FROM entity_link_candidates WHERE a_id=? AND b_id=?", (x, y)).fetchone()
    if not accept:
        c.execute("INSERT OR REPLACE INTO entity_link_candidates(a_id, b_id, score, features_json, status, decided_by, decided_at) VALUES (?,?,?,?, 'rejected', ?, ?)",
                  (x, y, row["score"] if row else 0.0, row["features_json"] if row else "{}", decided_by, time.time()))  # fmt: skip
        c.commit()
        return {"status": "rejected"}
    counts = {r[0]: r[1] for r in c.execute("SELECT entity_id, COUNT(*) FROM memory_entities WHERE entity_id IN (?,?) GROUP BY entity_id", (x, y))}
    keep, absorbed = (x, y) if counts.get(x, 0) >= counts.get(y, 0) else (y, x)
    mid = merge_entities(store, absorbed, keep, score=row["score"] if row else 1.0, method="user", features={}, decided_by=decided_by)
    c.commit()
    return {"status": "merged", "merge_id": mid, "into": keep}
