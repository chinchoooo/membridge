"""Consolidation: dedup, confirmation, explicit supersede/contradiction handling, scoring.

Nothing is ever overwritten or silently deleted. A newer statement either confirms an existing
memory (same meaning), supersedes it (explicit link + the old one stays retrievable as history),
or is linked as an unresolved contradiction. Rules are deterministic; ambiguous cases are left
for the optional LLM classifier (see llm/classify.py).
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass

from ..resolve.time import parse_date
from ..store.sqlite import Store
from .models import (
    CATEGORIES,
    Candidate,
    EntityRef,
    canonical,
    entity_id,
    memory_id,
    normalize_statement,
    sha,
)

_SUPERSEDABLE = {
    "uses",
    "deadline",
    "status",
}  # one current value per (subject, predicate)
_OPPOSITE = {"prefers": "avoids", "avoids": "prefers"}
_NEAR_DUP = 0.85
_SIMULTANEOUS = (
    3600.0  # seconds: opposing statements this close together are a real contradiction
)


@dataclass
class Outcome:
    memory_id: str
    action: str  # created | confirmed | skipped_deleted


def _tokens(s: str) -> set[str]:
    return set(normalize_statement(s).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


_IMPORTANCE = {
    "decision": 0.9, "constraint": 0.85, "profile": 0.8, "project": 0.75, "preference": 0.7, "skill": 0.6,
    "education": 0.6, "research": 0.6, "lesson": 0.6, "health": 0.5, "fact": 0.5, "interest": 0.5, "session": 0.3,
}  # fmt: skip
_ACTR_DECAY = 0.5  # ACT-R base-level learning decay d


def importance_of(category: str, verification: str) -> float:
    """Deterministic prior in (0,1): what the user said directly about durable things matters more.

    Bonus is applied to the *remaining headroom* (base + (1-base)*b), so the ordering
    inferred < stated < verified holds for every category and never saturates.
    """
    base = _IMPORTANCE.get(category, 0.5)
    bonus = {"user_stated": 0.4, "user_verified": 0.6}.get(verification, 0.0)
    return base + (1.0 - base) * bonus


def activation_of(access_times: list[float], now: float) -> float:
    """ACT-R base-level activation B = ln(sum_j (dt_j+1)^-d) over accesses (days), squashed to (0,1).

    Every confirmation of a fact is an access, so repetition raises activation and age lowers it, without
    letting frequency alone dominate (importance and confidence are separate terms).
    """
    if not access_times:
        return 0.0
    total = sum((max(0.0, now - t) / 86400.0 + 1.0) ** -_ACTR_DECAY for t in access_times)
    return 1.0 / (1.0 + math.exp(-math.log(total)))


def score_memory(
    row: sqlite3.Row | dict[str, object], now: float | None = None, access_times: list[float] | None = None
) -> float:
    now = now or time.time()
    explicit = {"user_stated": 1.0, "user_verified": 1.0, "provider_stated": 0.7, "model_inferred": 0.4}.get(
        str(row["verification"]), 0.4
    )
    last = float(row["last_seen"] or row["created_at"] or now)  # type: ignore[arg-type]
    act = activation_of(access_times if access_times else [last], now)
    imp = importance_of(str(row["category"]), str(row["verification"]))
    confirm = min(1.0, math.log1p(int(str(row["mention_count"])) - 1) / math.log(6))
    durable = str(row["category"]) in {"decision", "constraint", "profile", "preference", "project"}
    score = 0.28 * explicit + 0.20 * act + 0.17 * imp + 0.10 * confirm + 0.25 * float(row["confidence"])  # type: ignore[arg-type]
    if durable and row["status"] == "current":
        score = max(score, 0.5)  # a one-off old decision must stay retrievable
    if row["status"] in ("superseded", "unresolved"):
        score *= 0.5
    return round(score, 4)


def rescore(store: Store, mem_id: str, now: float | None = None) -> None:
    c = store.conn
    row = c.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
    now = now or time.time()
    times = [
        float(r[0])
        for r in c.execute(
            "SELECT g.create_time FROM memory_evidence e JOIN messages g ON g.id=e.ref "
            "WHERE e.memory_id=? AND g.create_time IS NOT NULL",
            (mem_id,),
        )
    ]
    c.execute(
        "UPDATE memories SET score=?, importance=?, activation=? WHERE id=?",
        (
            score_memory(row, now, times),
            importance_of(row["category"], row["verification"]),
            activation_of(times or [float(row["last_seen"] or row["created_at"] or now)], now),
            mem_id,
        ),
    )


def rescore_all(store: Store, now: float | None = None) -> int:
    """Recompute score/importance/activation for every live memory. Idempotent; fills rows that predate the
    bi-temporal migration and keeps recency honest as time passes."""
    ids = [r[0] for r in store.conn.execute("SELECT id FROM memories WHERE status!='deleted'")]
    for mid in ids:
        rescore(store, mid, now)
    store.conn.commit()
    return len(ids)


def revision(store: Store, mem_id: str, actor: str, reason: str) -> None:
    row = store.conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
    if row is not None:
        store.conn.execute(
            "INSERT INTO memory_revisions(memory_id, ts, actor, reason, old_json) VALUES (?,?,?,?,?)",
            (mem_id, time.time(), actor, reason, json.dumps(dict(row))),
        )


def upsert_entity(store: Store, ref: EntityRef) -> str | None:
    """Return the live entity id for `ref`, creating it if new. Aliases and merge pointers are honoured, so
    'Postgres' resolves to the merged 'PostgreSQL' entity instead of resurrecting a duplicate."""
    canon = canonical(ref.name)
    if len(canon) < 2:
        return None
    c = store.conn
    hit = c.execute(
        "SELECT a.entity_id, e.type FROM entity_aliases a JOIN live_entities e ON e.id=a.entity_id WHERE a.alias_norm=? LIMIT 1", (canon,)
    ).fetchone()
    if hit is not None:
        if hit["type"] == "concept" and ref.type != "concept":
            c.execute("UPDATE entities SET type=? WHERE id=?", (ref.type, hit["entity_id"]))
        return str(hit["entity_id"])
    eid = entity_id(canon)
    row = c.execute("SELECT type, merged_into FROM entities WHERE id=?", (eid,)).fetchone()
    if row is None:
        c.execute(
            "INSERT INTO entities(id, canonical, name, type, created_at) VALUES (?,?,?,?,?)", (eid, canon, ref.name, ref.type, time.time())
        )
        c.execute(
            "INSERT OR IGNORE INTO entity_aliases(entity_id, alias_norm, alias, source, first_seen) VALUES (?,?,?,'extracted',?)",
            (eid, canon, ref.name, time.time()),
        )
        return eid
    if row["merged_into"] is not None:
        from ..resolve.entities import root_entity

        return root_entity(store, eid)
    if row["type"] == "concept" and ref.type != "concept":
        c.execute("UPDATE entities SET type=? WHERE id=?", (ref.type, eid))
    return eid


def add_evidence(store: Store, mem_id: str, c: Candidate) -> None:
    store.conn.execute(
        "INSERT OR IGNORE INTO memory_evidence(memory_id, ref, ref_kind, quote) VALUES (?,?,?,?)",
        (mem_id, c.ref, c.ref_kind, c.quote[:400]),
    )


def _find_existing(store: Store, c: Candidate, dk: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = store.conn.execute(
        "SELECT * FROM memories WHERE dedup_key=?", (dk,)
    ).fetchone()
    if row is not None:
        return row
    # Identical wording filed under another category (e.g. rule says "constraint", model says "decision")
    # is still the same memory.
    norm = normalize_statement(c.statement)
    other_keys = [sha(cat, norm) for cat in sorted(CATEGORIES) if cat != c.category]
    marks = ",".join("?" * len(other_keys))
    row = store.conn.execute(f"SELECT * FROM memories WHERE dedup_key IN ({marks}) LIMIT 1", other_keys).fetchone()
    if row is not None:
        return row
    # Near-duplicate: same category/predicate/object-identity and very similar wording.
    oc = canonical(c.object) if c.object else None
    mine = _tokens(c.statement)
    for r in store.conn.execute(
        "SELECT * FROM memories WHERE subject IS ? AND predicate IS ? AND category=?",  # null-safe AND index-friendly
        (c.subject, c.predicate, c.category),
    ):
        same_obj = (
            oc is not None and r["object"] is not None and canonical(r["object"]) == oc
        )
        if same_obj and _jaccard(mine, _tokens(r["statement"])) >= _NEAR_DUP:
            return r  # type: ignore[no-any-return]
    return None


def upsert(store: Store, c: Candidate) -> Outcome:
    now = time.time()
    if c.subject and c.subject != "user":
        hit = store.conn.execute(
            "SELECT e.id, e.canonical FROM entity_aliases a JOIN live_entities e ON e.id=a.entity_id WHERE a.alias_norm=? LIMIT 1",
            (c.subject,),
        ).fetchone()
        if hit is not None:
            c.subject = hit["canonical"]
    dk = sha(c.category, normalize_statement(c.statement))
    existing = _find_existing(store, c, dk)
    if existing is not None:
        mid = existing["id"]
        if existing["status"] == "deleted":
            return Outcome(
                mid, "skipped_deleted"
            )  # the user forgot this; never resurrect it
        add_evidence(store, mid, c)
        n = store.conn.execute(
            "SELECT COUNT(*) FROM memory_evidence WHERE memory_id=?", (mid,)
        ).fetchone()[0]
        store.conn.execute(
            "UPDATE memories SET mention_count=?, first_seen=MIN(IFNULL(first_seen,?),?), "
            "last_seen=MAX(IFNULL(last_seen,?),?), updated_at=? WHERE id=?",
            (
                max(n, existing["mention_count"]),
                c.ts or now,
                c.ts or now,
                c.ts or 0,
                c.ts or 0,
                now,
                mid,
            ),
        )
        rescore(store, mid)
        return Outcome(mid, "confirmed")

    mid = memory_id(dk)
    ts = c.ts or now
    base_pred = (c.predicate or "").split(":")[0]
    obj_value = c.object_value or (parse_date(c.object, c.ts) if base_pred == "deadline" and c.object else None)
    c.object_value = obj_value
    store.conn.execute(
        "INSERT INTO memories(id, statement, kind, category, subject, predicate, object, status, verification, "
        "confidence, extraction_method, sensitivity, valid_from, first_seen, last_seen, dedup_key, origin, "
        "created_at, updated_at, valid_at, recorded_at, object_value) "
        "VALUES (?,?,?,?,?,?,?, 'current', ?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, c.statement, c.kind, c.category, c.subject, c.predicate, c.object, c.verification, c.confidence,
         c.method, c.sensitivity, ts, ts, ts, dk, c.origin, now, now, ts, now, obj_value),
    )  # fmt: skip
    store.conn.execute(
        "INSERT INTO memories_fts(memory_id, statement) VALUES (?,?)",
        (mid, c.statement),
    )
    add_evidence(store, mid, c)
    for ref in c.entities:
        eid = upsert_entity(store, ref)
        if eid:
            role = (
                "object"
                if c.object and canonical(ref.name) == canonical(c.object)
                else "about"
            )
            store.conn.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, role) VALUES (?,?,?)",
                (mid, eid, role),
            )
    if c.subject and c.subject != "user":
        eid = upsert_entity(store, EntityRef(c.subject))
        if eid:
            store.conn.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, role) VALUES (?,?,'subject')",
                (mid, eid),
            )
    _resolve_conflicts(store, mid, c)
    rescore(store, mid)
    return Outcome(mid, "created")


def _link(store: Store, src: str, rel: str, dst: str, reason: str) -> None:
    store.conn.execute(
        "INSERT OR IGNORE INTO memory_links(src_id, rel, dst_id, reason, created_at) VALUES (?,?,?,?,?)",
        (src, rel, dst, reason, time.time()),
    )


def _supersede(
    store: Store, newer: str, older: str, reason: str, at: float | None, retract: bool = False
) -> None:
    """Invalidate `older` (world time) without deleting it.

    retract=True is a *correction*: the old statement was never true, so its validity collapses to zero length
    and `expired_at` records when we learned that (knowledge time). Otherwise it simply stopped being true at `at`.
    """
    revision(store, older, "consolidator", f"superseded by {newer}: {reason}")
    now = time.time()
    if retract:
        store.conn.execute(
            "UPDATE memories SET status='superseded', valid_to=valid_from, invalid_at=valid_at, expired_at=?, updated_at=? WHERE id=?",
            (now, now, older),
        )
    else:
        store.conn.execute(
            "UPDATE memories SET status='superseded', valid_to=?, invalid_at=?, updated_at=? WHERE id=?",
            (at, at, now, older),
        )
    _link(store, newer, "SUPERSEDES", older, reason)
    rescore(store, older)


def _resolve_conflicts(store: Store, new_id: str, c: Candidate) -> None:
    if not c.subject or not c.predicate or not c.object:
        return
    new_ts = c.ts or time.time()
    new_obj = c.object_value or canonical(c.object)
    base = c.predicate.split(":")[0]
    rows = store.conn.execute(
        "SELECT * FROM memories WHERE id!=? AND status IN ('current','unresolved') AND subject=? "
        "AND (predicate=? OR predicate=? OR predicate LIKE ?)",
        (new_id, c.subject, c.predicate, _OPPOSITE.get(c.predicate, c.predicate), base + ":%"),
    ).fetchall()
    for old in rows:
        old_obj = old["object_value"] or canonical(old["object"] or "")
        old_ts = old["valid_from"] or 0.0
        # A project uses many technologies: only the same *aspect* (predicate "uses:storage") is replaced,
        # or anything the user explicitly switched away from ("switched X from A to B").
        explicit = c.old_object is not None and canonical(c.old_object) == old_obj and old["predicate"].split(":")[0] == base
        if base in _SUPERSEDABLE and old_obj != new_obj and (old["predicate"] == c.predicate or explicit):
            reason = "correction" if c.correction else "explicit_switch" if explicit else {
                "uses": "changed_decision", "deadline": "constraint_updated"}.get(base, "updated")  # fmt: skip
            if new_ts >= old_ts:
                _supersede(store, new_id, old["id"], reason, new_ts, retract=c.correction)
            else:  # an older statement arrived late: it is history, not current
                _supersede(store, old["id"], new_id, reason, old_ts)
                store.conn.execute(
                    "UPDATE memories SET valid_to=? WHERE id=?", (old_ts, new_id)
                )
        elif (
            c.predicate in _OPPOSITE
            and old["predicate"] == _OPPOSITE[c.predicate]
            and old_obj == new_obj
        ):
            if abs(new_ts - old_ts) < _SIMULTANEOUS:
                _link(
                    store,
                    new_id,
                    "CONTRADICTS",
                    old["id"],
                    "opposing_statements_unresolved",
                )
                for m in (new_id, old["id"]):
                    store.conn.execute(
                        "UPDATE memories SET status='unresolved' WHERE id=?", (m,)
                    )
                    rescore(store, m)
            elif new_ts > old_ts:
                _supersede(store, new_id, old["id"], "changed_preference", new_ts)
            else:
                _supersede(store, old["id"], new_id, "changed_preference", old_ts)


def refresh_entity_links(store: Store) -> int:
    """Link entities to every conversation whose messages mention them or any alias of them."""
    n = 0
    store.conn.execute("DELETE FROM entity_mentions")
    for e in store.conn.execute("SELECT id, canonical FROM live_entities").fetchall():
        names = {e["canonical"]} | {r[0] for r in store.conn.execute("SELECT alias_norm FROM entity_aliases WHERE entity_id=?", (e["id"],))}
        phrases = ['"' + " ".join(re.sub(r"[^a-z0-9 ]", " ", nm).split()) + '"' for nm in sorted(names)]
        phrases = [p for p in phrases if p != '""']
        if not phrases:
            continue
        try:
            rows = store.conn.execute(
                "SELECT m.conversation_id AS cid, COUNT(*) AS n FROM messages_fts f "
                "JOIN messages m ON m.id=f.message_id WHERE messages_fts MATCH ? GROUP BY m.conversation_id "
                "ORDER BY n DESC LIMIT 25",
                (" OR ".join(phrases),),
            ).fetchall()
        except sqlite3.OperationalError:
            continue
        for r in rows:
            store.conn.execute(
                "INSERT OR REPLACE INTO entity_mentions(entity_id, conversation_id, count) VALUES (?,?,?)",
                (e["id"], r["cid"], r["n"]),
            )
            n += 1
    return n
