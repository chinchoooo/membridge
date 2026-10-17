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
    trust_rank,
)

_SUPERSEDABLE = {
    "uses",
    "deadline",
    "status",
}  # one current value per (subject, predicate)
# One current value per (subject, predicate) about the user: an employer, a school, a city, a name, a favourite X.
# Skills, interests and tools are many-valued and never conflict with each other.
_SINGLE_VALUED = {
    "work_at", "study_at", "live_at", "am_based_at", "name_is", "editor_is", "ide_is", "os_is", "degree_is", "major_is",
    "university_is", "college_is", "company_is", "employer_is", "team_is", "thesis_is",
}  # fmt: skip
_OPPOSITE = {"prefers": "avoids", "avoids": "prefers"}


def single_valued(predicate: str | None) -> bool:
    p = (predicate or "").split(":")[0]
    return p in _SINGLE_VALUED or (p.startswith("favo") and p.endswith("_is"))

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
    bonus = {"user_stated": 0.4, "user_verified": 0.6, "client_asserted": 0.2}.get(verification, 0.0)
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
    explicit = {"user_stated": 1.0, "user_verified": 1.0, "client_asserted": 0.8, "provider_stated": 0.7, "model_inferred": 0.4}.get(
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


def dedup_key_of(c: Candidate) -> str:
    """category + wording, plus the predicate when there is one: a single sentence can carry two different facts
    ("my favourite editor is Vim and I work at Acme") and they must not collapse into one memory."""
    norm = normalize_statement(c.statement)
    return sha(c.category, norm, c.predicate) if c.predicate else sha(c.category, norm)


def _find_existing(store: Store, c: Candidate, dk: str) -> sqlite3.Row | None:
    conn = store.conn
    row: sqlite3.Row | None = conn.execute("SELECT * FROM memories WHERE dedup_key=?", (dk,)).fetchone()
    if row is not None:
        return row
    norm = normalize_statement(c.statement)
    # Rows stored before predicates joined the key use the plain (category, wording) key.
    legacy = conn.execute("SELECT * FROM memories WHERE dedup_key=?", (sha(c.category, norm),)).fetchone()
    if legacy is not None and (legacy["predicate"] or None) == (c.predicate or None):
        return legacy  # type: ignore[no-any-return]
    # Identical wording filed under another category (rule says "constraint", model says "decision") is the same
    # memory, but only when it asserts the same thing (same predicate): another triple is another fact.
    keys = [sha(cat, norm, c.predicate) for cat in sorted(CATEGORIES) if cat != c.category] if c.predicate else []
    keys += [sha(cat, norm) for cat in sorted(CATEGORIES) if cat != c.category]
    marks = ",".join("?" * len(keys))
    for r in conn.execute(f"SELECT * FROM memories WHERE dedup_key IN ({marks})", keys):
        if (r["predicate"] or None) == (c.predicate or None):
            return r  # type: ignore[no-any-return]
    # Near-duplicate: same category/predicate/object-identity and very similar wording.
    oc = canonical(c.object) if c.object else None
    if oc is None:
        return None  # nothing to compare an object on: scanning every subject-less row is O(n^2) for bulk imports
    mine = _tokens(c.statement)
    for r in conn.execute(
        "SELECT * FROM memories WHERE subject IS ? AND predicate IS ? AND category=?",  # null-safe AND index-friendly
        (c.subject, c.predicate, c.category),
    ):
        same_obj = r["object"] is not None and canonical(r["object"]) == oc
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
    dk = dedup_key_of(c)
    existing = _find_existing(store, c, dk)
    if existing is not None:
        return _confirm(store, existing, c, now)
    return _create(store, c, dk, now)


def _link_entities(store: Store, mid: str, c: Candidate) -> None:
    for ref in c.entities:
        eid = upsert_entity(store, ref)
        if eid:
            role = "object" if c.object and canonical(ref.name) == canonical(c.object) else "about"
            store.conn.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, role) VALUES (?,?,?)", (mid, eid, role)
            )
    if c.subject and c.subject != "user":
        eid = upsert_entity(store, EntityRef(c.subject))
        if eid:
            store.conn.execute(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, role) VALUES (?,?,'subject')", (mid, eid)
            )


def _confirm(store: Store, existing: sqlite3.Row, c: Candidate, now: float) -> Outcome:
    """The same thing was said again. It adds evidence; it also lets the memory *inherit* the strongest trust, any
    sensitivity, a known date and the entities of every statement of it (a fact the user later states themselves
    must not stay 'provider_stated')."""
    mid = existing["id"]
    if existing["status"] == "deleted":
        return Outcome(mid, "skipped_deleted")  # the user forgot this; never resurrect it
    conn = store.conn
    add_evidence(store, mid, c)
    n = conn.execute("SELECT COUNT(*) FROM memory_evidence WHERE memory_id=?", (mid,)).fetchone()[0]
    conn.execute(
        "UPDATE memories SET mention_count=?, first_seen=MIN(IFNULL(first_seen,?),?), "
        "last_seen=MAX(IFNULL(last_seen,?),?), updated_at=? WHERE id=?",
        (max(n, existing["mention_count"]), c.ts or now, c.ts or now, c.ts or 0, c.ts or 0, now, mid),
    )
    changes: dict[str, object] = {}
    if trust_rank(c.verification) > trust_rank(existing["verification"]):
        changes["verification"] = c.verification
    if c.sensitivity == "sensitive" and existing["sensitivity"] != "sensitive":
        changes["sensitivity"] = "sensitive"
    if c.confidence > existing["confidence"] and trust_rank(c.verification) >= trust_rank(existing["verification"]):
        changes["confidence"] = c.confidence
    if c.ts is not None and not existing["valid_time_known"]:
        changes.update(valid_at=c.ts, valid_from=c.ts, valid_time_known=1)
    if changes:
        revision(store, mid, "consolidator", "re-stated: merged " + ",".join(sorted(changes)))
        sets = ", ".join(f"{k}=?" for k in changes)
        conn.execute(f"UPDATE memories SET {sets} WHERE id=?", [*changes.values(), mid])
    _link_entities(store, mid, c)
    rescore(store, mid)
    if existing["status"] == "superseded" and _should_reinstate(store, existing, c):
        _reinstate(store, mid, c)
    return Outcome(mid, "confirmed")


def _should_reinstate(store: Store, existing: sqlite3.Row, c: Candidate) -> bool:
    """A superseded value that is stated again, later and with at least the trust of whatever replaced it, is the
    current value again (A -> B -> A). Re-importing the same old source must not do this: it needs a later date."""
    base = (c.predicate or "").split(":")[0]
    if not c.subject or not c.predicate or not c.object or not (base in _SUPERSEDABLE or single_valued(c.predicate)):
        return False
    if c.ts is None or (existing["invalid_at"] is not None and c.ts <= existing["invalid_at"]):
        return False
    for r in store.conn.execute(
        "SELECT m.verification FROM memory_links l JOIN memories m ON m.id=l.src_id "
        "WHERE l.dst_id=? AND l.rel='SUPERSEDES' AND m.status='current'",
        (existing["id"],),
    ):
        if trust_rank(r["verification"]) > trust_rank(c.verification):
            return False  # a more trusted statement replaced it; a weaker re-statement does not undo that
    return True


def _reinstate(store: Store, mid: str, c: Candidate) -> None:
    revision(store, mid, "consolidator", "restated later: current again")
    now = time.time()
    store.conn.execute(
        "UPDATE memories SET status='current', valid_to=NULL, invalid_at=NULL, expired_at=NULL, valid_at=?, valid_from=?, "
        "valid_time_known=1, updated_at=? WHERE id=?",
        (c.ts, c.ts, now, mid),
    )
    _resolve_conflicts(store, mid, c)  # whatever is current in between is now history
    rescore(store, mid)


def _create(store: Store, c: Candidate, dk: str, now: float) -> Outcome:
    mid = memory_id(dk)
    ts = c.ts if c.ts is not None else now
    base_pred = (c.predicate or "").split(":")[0]
    obj_value = c.object_value or (parse_date(c.object, c.ts) if base_pred == "deadline" and c.object else None)
    c.object_value = obj_value
    store.conn.execute(
        "INSERT INTO memories(id, statement, kind, category, subject, predicate, object, status, verification, "
        "confidence, extraction_method, sensitivity, valid_from, first_seen, last_seen, dedup_key, origin, "
        "created_at, updated_at, valid_at, recorded_at, object_value, valid_time_known) "
        "VALUES (?,?,?,?,?,?,?, 'current', ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, c.statement, c.kind, c.category, c.subject, c.predicate, c.object, c.verification, c.confidence,
         c.method, c.sensitivity, ts, ts, ts, dk, c.origin, now, now, ts, now, obj_value, 0 if c.ts is None else 1),
    )  # fmt: skip
    store.conn.execute("INSERT INTO memories_fts(memory_id, statement) VALUES (?,?)", (mid, c.statement))
    add_evidence(store, mid, c)
    _link_entities(store, mid, c)
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
    and `expired_at` records when we learned that (knowledge time). Otherwise it simply stopped being true at `at`
    (when `at` is unknown it falls back to when the newer statement began, then to when we recorded it: never NULL,
    which `as_of` would read as 'valid forever').
    """
    if at is None:
        row = store.conn.execute("SELECT valid_at, recorded_at FROM memories WHERE id=?", (newer,)).fetchone()
        at = (row["valid_at"] or row["recorded_at"]) if row is not None else None
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
            (at or now, at or now, now, older),
        )
    _link(store, newer, "SUPERSEDES", older, reason)
    rescore(store, older)


def _outrank(store: Store, winner: str, loser: str, reason: str) -> None:
    """The loser was *not believed*, which is different from having stopped being true: zero-length validity, no
    retraction timestamp, and the winner is linked as its superseder so the history stays explainable."""
    revision(store, loser, "consolidator", f"outranked by {winner}: {reason}")
    now = time.time()
    store.conn.execute(
        "UPDATE memories SET status='superseded', valid_to=valid_from, invalid_at=IFNULL(valid_at, valid_from), updated_at=? WHERE id=?",
        (now, loser),
    )
    _link(store, winner, "SUPERSEDES", loser, reason)
    rescore(store, loser)


def _pending(store: Store, claim: str, standing: str, reason: str) -> None:
    """A less trusted but NEWER claim contradicts something the user stands behind. The user's statement stays current;
    only the claim waits for the person (so a connected assistant cannot take your own statements out of circulation)."""
    _link(store, claim, "CONTRADICTS", standing, reason)
    revision(store, claim, "consolidator", f"pending: {reason}")
    store.conn.execute("UPDATE memories SET status='unresolved', updated_at=? WHERE id=?", (time.time(), claim))
    rescore(store, claim)


def _unresolved(store: Store, a: str, b: str, reason: str) -> None:
    _link(store, a, "CONTRADICTS", b, reason)
    for m in (a, b):
        revision(store, m, "consolidator", f"unresolved: {reason}")
        store.conn.execute("UPDATE memories SET status='unresolved', updated_at=? WHERE id=?", (time.time(), m))
        rescore(store, m)


_ORG_SUFFIXES = {
    "inc", "corp", "corporation", "ltd", "limited", "llc", "co", "company", "gmbh", "labs", "lab", "technologies", "systems",
    "university", "institute", "group", "robotics",
}  # fmt: skip


def _alias_key(store: Store | None, value: str) -> str:
    """'Postgres' and 'PostgreSQL' (and 'pg') are one value: the curated alias table and any entity merge the user made."""
    from ..resolve.blocking import compact
    from ..resolve.score import ALIAS_DICT

    k = compact(value)
    if store is not None:
        row = store.conn.execute(
            "SELECT e.canonical FROM entity_aliases a JOIN live_entities e ON e.id=a.entity_id WHERE a.alias_norm=? LIMIT 1", (value,)
        ).fetchone()
        if row is not None:
            return compact(str(row[0]))
    return ALIAS_DICT.get(k, k)


def _same_value(a: str, b: str, store: Store | None = None) -> bool:
    """'Acme' and 'Acme Corp' are one employer, 'Postgres' and 'PostgreSQL' one database; 'Visual Studio' and 'Visual Studio Code'
    are two editors, and 'Postgres' and 'MySQL' two databases."""
    if a == b or _alias_key(store, a) == _alias_key(store, b):
        return True
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb or not (ta <= tb or tb <= ta):
        return False
    extra = (tb - ta) if ta <= tb else (ta - tb)
    return extra <= _ORG_SUFFIXES  # only a corporate suffix is allowed to be the difference


def remembered_resolution(store: Store, subject: str, predicate: str, a_obj: str, b_obj: str) -> str | None:
    """If the user already settled this exact disagreement, return the value they kept ('a' or 'b')."""
    row = store.conn.execute(
        "SELECT kept_obj, dropped_obj FROM conflict_resolutions WHERE subject=? AND predicate=? "
        "AND ((kept_obj=? AND dropped_obj=?) OR (kept_obj=? AND dropped_obj=?)) ORDER BY resolved_at DESC LIMIT 1",
        (subject, predicate, a_obj, b_obj, b_obj, a_obj),
    ).fetchone()
    if row is None:
        return None
    return "a" if row["kept_obj"] == a_obj else "b"


def _arbitrate(store: Store, new_id: str, old: sqlite3.Row, c: Candidate, kind: str, reason: str, explicit: bool) -> None:
    """Decide between two statements about the same thing.

    1. The user already settled this exact disagreement -> same answer.
    2. Different trust: the more trusted one wins whatever the dates, and the other stays as history. The one
       exception: a connected agent's *newer* claim against a more trusted older one is a genuine change the user
       must settle, not something to overrule silently.
    3. Equal trust: the newer dated statement wins; an explicit correction/switch wins; anything else (an undated
       or simultaneous clash) is left unresolved for the user. Nothing is guessed from "now".
    """
    old_id = old["id"]
    new_obj = c.object_value or canonical(c.object or "")
    old_obj = old["object_value"] or canonical(old["object"] or "")
    if c.subject and c.predicate:
        kept = remembered_resolution(store, c.subject, c.predicate, new_obj, old_obj)
        if kept is not None:
            winner, loser = (new_id, old_id) if kept == "a" else (old_id, new_id)
            _outrank(store, winner, loser, "user_resolution_remembered")
            return
    r_new, r_old = trust_rank(c.verification), trust_rank(old["verification"])
    new_ts = c.ts
    old_ts = old["valid_at"] if old["valid_time_known"] else None
    both = new_ts is not None and old_ts is not None
    if r_new != r_old:
        new_wins = r_new > r_old
        low_verif = old["verification"] if new_wins else c.verification
        low_ts, high_ts = (old_ts, new_ts) if new_wins else (new_ts, old_ts)
        if low_verif == "client_asserted" and both and low_ts is not None and high_ts is not None and low_ts > high_ts:
            _pending(store, old_id if new_wins else new_id, new_id if new_wins else old_id, "newer_agent_claim_vs_more_trusted")
        elif new_wins:
            _outrank(store, new_id, old_id, "outranked_by_trust")
        else:
            _outrank(store, old_id, new_id, "outranked_by_trust")
        return
    if explicit or c.correction:
        _supersede(store, new_id, old_id, reason, new_ts if new_ts is not None else old_ts, retract=c.correction)
        return
    if new_ts is None or old_ts is None:
        _unresolved(store, new_id, old_id, "undated_conflict_equal_trust")
        return
    if kind == "opposite":
        if abs(new_ts - old_ts) < _SIMULTANEOUS:
            _unresolved(store, new_id, old_id, "opposing_statements_unresolved")
        elif new_ts > old_ts:
            _supersede(store, new_id, old_id, "changed_preference", new_ts)
        else:
            _supersede(store, old_id, new_id, "changed_preference", old_ts)
        return
    if new_ts > old_ts:
        _supersede(store, new_id, old_id, reason, new_ts)
    elif new_ts < old_ts:  # an older statement arrived late: it is history, not current
        _supersede(store, old_id, new_id, reason, old_ts)
        store.conn.execute("UPDATE memories SET valid_to=? WHERE id=?", (old_ts, new_id))
    else:
        _unresolved(store, new_id, old_id, "simultaneous_conflict_equal_trust")


def _resolve_conflicts(store: Store, new_id: str, c: Candidate) -> None:
    if not c.subject or not c.predicate or not c.object:
        return
    new_obj = c.object_value or canonical(c.object)
    base = c.predicate.split(":")[0]
    rows = store.conn.execute(
        "SELECT * FROM memories WHERE id!=? AND status IN ('current','unresolved') AND subject=? "
        "AND (predicate=? OR predicate=? OR predicate LIKE ?)",
        (new_id, c.subject, c.predicate, _OPPOSITE.get(c.predicate, c.predicate), base + ":%"),
    ).fetchall()
    for old in rows:
        old_obj = old["object_value"] or canonical(old["object"] or "")
        # A project uses many technologies: only the same *aspect* (predicate "uses:storage") is replaced,
        # or anything the user explicitly switched away from ("switched X from A to B").
        explicit = c.old_object is not None and canonical(c.old_object) == old_obj and old["predicate"].split(":")[0] == base
        if (base in _SUPERSEDABLE or single_valued(c.predicate)) and not _same_value(old_obj, new_obj, store) and (
            old["predicate"] == c.predicate or explicit
        ):
            reason = "correction" if c.correction else "explicit_switch" if explicit else {
                "uses": "changed_decision", "deadline": "constraint_updated"}.get(base, "updated")  # fmt: skip
            _arbitrate(store, new_id, old, c, "replace", reason, explicit)
        elif c.predicate in _OPPOSITE and old["predicate"] == _OPPOSITE[c.predicate] and old_obj == new_obj:
            _arbitrate(store, new_id, old, c, "opposite", "changed_preference", False)


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
