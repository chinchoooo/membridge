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


def score_memory(
    row: sqlite3.Row | dict[str, object], now: float | None = None
) -> float:
    now = now or time.time()
    explicit = {
        "user_stated": 1.0,
        "user_verified": 1.0,
        "provider_stated": 0.7,
        "model_inferred": 0.4,
    }.get(str(row["verification"]), 0.4)
    last = float(row["last_seen"] or row["created_at"] or now)  # type: ignore[arg-type]
    recency = math.exp(-max(0.0, now - last) / 86400.0 / 365.0)
    confirm = min(1.0, math.log1p(int(str(row["mention_count"])) - 1) / math.log(6))
    durable = str(row["category"]) in {
        "decision",
        "constraint",
        "profile",
        "preference",
        "project",
    }
    score = (
        0.30 * explicit
        + 0.15 * recency
        + 0.15 * confirm
        + 0.15 * (1.0 if durable else 0.6)
    )
    score += 0.25 * float(row["confidence"])  # type: ignore[arg-type]
    if durable and row["status"] == "current":
        score = max(score, 0.5)  # a one-off old decision must stay retrievable
    if row["status"] in ("superseded", "unresolved"):
        score *= 0.5
    return round(score, 4)


def rescore(store: Store, mem_id: str) -> None:
    row = store.conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
    store.conn.execute(
        "UPDATE memories SET score=? WHERE id=?", (score_memory(row), mem_id)
    )


def revision(store: Store, mem_id: str, actor: str, reason: str) -> None:
    row = store.conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
    if row is not None:
        store.conn.execute(
            "INSERT INTO memory_revisions(memory_id, ts, actor, reason, old_json) VALUES (?,?,?,?,?)",
            (mem_id, time.time(), actor, reason, json.dumps(dict(row))),
        )


def upsert_entity(store: Store, ref: EntityRef) -> str | None:
    canon = canonical(ref.name)
    if len(canon) < 2:
        return None
    eid = entity_id(canon)
    row = store.conn.execute("SELECT type FROM entities WHERE id=?", (eid,)).fetchone()
    if row is None:
        store.conn.execute(
            "INSERT INTO entities(id, canonical, name, type, created_at) VALUES (?,?,?,?,?)",
            (eid, canon, ref.name, ref.type, time.time()),
        )
    elif row["type"] == "concept" and ref.type != "concept":
        store.conn.execute(
            "UPDATE entities SET type=? WHERE id=?", (ref.type, eid)
        )  # upgrade
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
        "SELECT * FROM memories WHERE category=? AND IFNULL(predicate,'')=IFNULL(?,'') "
        "AND IFNULL(subject,'')=IFNULL(?,'')",
        (c.category, c.predicate, c.subject),
    ):
        same_obj = (
            oc is not None and r["object"] is not None and canonical(r["object"]) == oc
        )
        if same_obj and _jaccard(mine, _tokens(r["statement"])) >= _NEAR_DUP:
            return r  # type: ignore[no-any-return]
    return None


def upsert(store: Store, c: Candidate) -> Outcome:
    now = time.time()
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
    store.conn.execute(
        "INSERT INTO memories(id, statement, kind, category, subject, predicate, object, status, verification, "
        "confidence, extraction_method, sensitivity, valid_from, first_seen, last_seen, dedup_key, origin, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?, 'current', ?,?,?,?,?,?,?,?,?,?,?)",
        (mid, c.statement, c.kind, c.category, c.subject, c.predicate, c.object, c.verification, c.confidence,
         c.method, c.sensitivity, ts, ts, ts, dk, c.origin, now, now),
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
    store: Store, newer: str, older: str, reason: str, at: float | None
) -> None:
    revision(store, older, "consolidator", f"superseded by {newer}: {reason}")
    store.conn.execute(
        "UPDATE memories SET status='superseded', valid_to=?, updated_at=? WHERE id=?",
        (at, time.time(), older),
    )
    _link(store, newer, "SUPERSEDES", older, reason)
    rescore(store, older)


def _resolve_conflicts(store: Store, new_id: str, c: Candidate) -> None:
    if not c.subject or not c.predicate or not c.object:
        return
    new_ts = c.ts or time.time()
    new_obj = canonical(c.object)
    base = c.predicate.split(":")[0]
    rows = store.conn.execute(
        "SELECT * FROM memories WHERE id!=? AND status IN ('current','unresolved') AND IFNULL(subject,'')=? "
        "AND (predicate=? OR predicate=? OR predicate LIKE ?)",
        (new_id, c.subject, c.predicate, _OPPOSITE.get(c.predicate, c.predicate), base + ":%"),
    ).fetchall()
    for old in rows:
        old_obj = canonical(old["object"] or "")
        old_ts = old["valid_from"] or 0.0
        # A project uses many technologies: only the same *aspect* (predicate "uses:storage") is replaced,
        # or anything the user explicitly switched away from ("switched X from A to B").
        explicit = c.old_object is not None and canonical(c.old_object) == old_obj and old["predicate"].split(":")[0] == base
        if base in _SUPERSEDABLE and old_obj != new_obj and (old["predicate"] == c.predicate or explicit):
            reason = "explicit_switch" if explicit else {
                "uses": "changed_decision", "deadline": "constraint_updated"}.get(base, "updated")  # fmt: skip
            if new_ts >= old_ts:
                _supersede(store, new_id, old["id"], reason, new_ts)
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
    """Link entities to every conversation whose messages mention them (cross-conversation glue)."""
    n = 0
    store.conn.execute("DELETE FROM entity_mentions")
    for e in store.conn.execute("SELECT id, canonical FROM entities").fetchall():
        words = re.sub(r"[^a-z0-9 ]", " ", e["canonical"]).split()
        if not words:
            continue
        query = '"' + " ".join(words) + '"'
        try:
            rows = store.conn.execute(
                "SELECT m.conversation_id AS cid, COUNT(*) AS n FROM messages_fts f "
                "JOIN messages m ON m.id=f.message_id WHERE messages_fts MATCH ? GROUP BY m.conversation_id "
                "ORDER BY n DESC LIMIT 25",
                (query,),
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
