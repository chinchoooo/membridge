"""Import a ChatGPT 'data-portability dump' as provider-stated memories.

Two inputs are understood:
  * the strict line format asked for by docs/CHATGPT_DUMP_PROMPT.md (FACT / CHANGE / ENTITY / EPISODE | ...);
  * the looser numbered-prose dump ("C12. text [source: memory] [confidence: high]" under ### headings).

Everything lands as provider_stated (or model_inferred for ChatGPT's own inferences), never user_stated: ChatGPT is
reporting what it believes, not quoting the user. A dump has no dates unless it says so, so undated facts are stored
with `valid_time_known=0` and never count as *newer* than something you dated yourself. Sensitive facts are flagged so
default retrieval never returns them.
"""

from __future__ import annotations

import calendar
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..store.sqlite import Store
from .consolidate import (
    _link,
    _resolve_conflicts,
    _supersede,
    dedup_key_of,
    refresh_entity_links,
    rescore,
    upsert,
    upsert_entity,
)
from .models import Candidate, EntityRef, canonical, contains_secret, sensitivity_of, sha
from .pipeline import _doc
from .rules import extract_from_message
from .summaries import refresh_summaries

_CATEGORY = {
    "identity": "profile", "location": "profile", "education": "education", "work": "project", "project": "project",
    "skill": "skill", "tool": "skill", "goal": "interest", "decision": "decision", "preference": "preference",
    "dislike": "preference", "habit": "constraint", "health": "health", "finance": "constraint",
    "relationship": "fact", "other": "fact",
}  # fmt: skip
_HEADING_CATEGORY = {
    "IDENTITY": "identity", "WORK": "work", "SKILLS": "skill", "GOALS": "goal", "DECISIONS": "decision",
    "LIKES": "preference", "HABITS": "habit", "PEOPLE": "entity", "FACTS THAT CHANGED": "change", "THINGS I ASKED": "other",
}  # fmt: skip
_CONF = {"high": 0.8, "medium": 0.6, "low": 0.4}
_META = re.compile(
    r"data[- ]portability|memory dump|comprehensive review of information|account data export|"
    r"this export is compiled|not a verbatim archive",
    re.IGNORECASE,
)
_TAGS = re.compile(r"\s*\[source:\s*(?P<src>[\w-]+)\]\s*(?:\[confidence:\s*(?P<conf>\w+)\])?\s*$", re.IGNORECASE)
_NUMBERED = re.compile(r"^(?P<k>[ABCD])(?P<n>\d+)\.\s+(?P<body>.+)$")
_EPISODE = re.compile(r"^(?P<title>.+?)\s+\|\s+(?P<date>(?:approx\.\s*)?[\w -]+?)\.\s+(?P<desc>.+)$")
_NA = re.compile(r"not available to me|^(?:exact )?(?:change )?(?:date|year|dates)[^.]*:", re.IGNORECASE)
_ENTITY_TYPES = {"person", "organization", "place", "project", "tool", "course", "event", "concept"}


@dataclass
class Item:
    kind: str  # fact | change | entity | episode
    text: str
    category: str = "other"
    source: str = "memory"
    confidence: str = "medium"
    when: str | None = None
    sensitive: bool = False
    chain: list[str] = field(default_factory=list)  # change: [old, ..., new]
    subject: str = ""
    etype: str = "concept"
    quote: str = ""


def parse_when(s: str | None) -> float | None:
    """'2026-10-17' / '2026-10' / '2026' -> UTC timestamp; anything vaguer ('approx. 2026-10', '?') -> a month or None."""
    if not s:
        return None
    m = re.search(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", s)
    if not m:
        return None
    y, mo, d = int(m[1]), int(m[2] or 1), int(m[3] or 1)
    if not (1970 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return float(calendar.timegm((y, mo, d, 0, 0, 0)))


def _yes(flag: str) -> bool:
    """'Y', 'yes', and the label-included 'sensitive: Y' that ChatGPT often writes."""
    return bool(re.match(r"(?:sensitive\s*:?\s*)?y(?:es)?\b", flag.strip(), re.IGNORECASE))


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _drop_unavailable(s: str) -> str:
    """Remove 'Exact date: not available to me.' style sentences; what is left (possibly nothing) is the fact."""
    kept = [x for x in re.split(r"(?<=[.!?])\s+", s) if x and not _NA.search(x)]
    return " ".join(kept)


def _split_tags(body: str) -> tuple[str, str, str]:
    m = _TAGS.search(body)
    if not m:
        return body, "memory", "medium"
    return body[: m.start()], m["src"].lower(), (m["conf"] or "medium").lower()


def parse_dump(text: str) -> list[Item]:
    items: list[Item] = []
    heading = "other"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("END ") or line == "CONTINUE":
            continue
        if line.startswith("#"):
            head = line.lstrip("#").strip().upper()
            heading = next((v for k, v in _HEADING_CATEGORY.items() if head.startswith(k)), "other")
            continue
        parts = [p.strip() for p in line.split("|")]
        tag = parts[0].upper()
        if tag == "FACT" and len(parts) >= 4:
            if _META.search(parts[2]):
                continue
            cat = parts[1].lower()
            sens = len(parts) > 6 and _yes(parts[6])
            items.append(Item("fact", _clean(parts[2]), cat, (parts[3] or "memory").lower(),
                              (parts[4] if len(parts) > 4 else "medium").lower(), parts[5] if len(parts) > 5 else None,
                              sens or cat in {"health", "finance"}, quote=line))  # fmt: skip
        elif tag == "CHANGE" and len(parts) >= 3 and "->" in parts[2]:
            chain = [_clean(x) for x in parts[2].split("->")]
            items.append(Item("change", parts[2], "other", (parts[4] if len(parts) > 4 else "memory").lower(),
                              (parts[5] if len(parts) > 5 else "medium").lower(), parts[3] if len(parts) > 3 else None,
                              chain=chain, subject=_clean(parts[1]), quote=line))  # fmt: skip
        elif tag == "ENTITY" and len(parts) >= 3:
            items.append(Item("entity", _clean(parts[3]) if len(parts) > 3 else "", "other", etype=parts[2].lower(),
                              subject=_clean(parts[1]), quote=line))  # fmt: skip
        elif tag == "EPISODE" and len(parts) >= 4:
            items.append(Item("episode", f"{_clean(parts[2])}: {_clean(parts[3])}", "session", when=parts[1], quote=line))
        elif m := _NUMBERED.match(line):
            _legacy_line(items, m["k"], m["body"], heading, line)
    return items


def _legacy_line(items: list[Item], kind: str, body: str, heading: str, line: str) -> None:
    if kind == "A" or _META.search(body):
        return
    if kind == "D":
        e = _EPISODE.match(body)
        if e:
            items.append(Item("episode", f"{_clean(e['title'])}: {_clean(e['desc'])}", "session", when=e["date"], quote=line))
        return
    body, src, conf = _split_tags(body)
    body = _clean(_drop_unavailable(_clean(body)))
    if not body or body.lower() == "none":
        return
    if kind == "B":  # custom instructions: keep only the verbatim quote; B1 repeats B2-B4 and B5 is a limitation
        q = re.search(r'Verbatim:\s*"(.+)"\s*$', body)
        if q and not q[1].startswith("Other Information"):
            items.append(Item("fact", f"Custom instruction: {_clean(q[1])}", "preference", "custom-instructions", "high", quote=line))
        return
    if heading == "entity":
        name = re.split(r"[,.]| in the context| referenced", body, maxsplit=1)[0].strip()
        items.append(Item("entity", body, "other", subject=name, quote=line))
        return
    if heading == "change" and "->" in body:
        subj, _, rest = body.partition(":")
        chain = [_clean(re.sub(r"\s*\(.*$|\s*\bwith\b.*$|,? because .*$", "", x)).rstrip(".") for x in rest.split("->")]
        items.append(Item("change", body, "other", src, conf, subject=_clean(subj), chain=chain, quote=line))
        return
    cat = heading if heading in _CATEGORY else "other"
    items.append(Item("fact", body, cat, src, conf, sensitive=cat == "health", quote=line))


# ---- structure: make plain sentences able to conflict --------------------------------------------------------------
_VERBS = {
    "prefers": "prefer", "likes": "like", "loves": "love", "enjoys": "enjoy", "hates": "hate", "dislikes": "dislike",
    "avoids": "avoid", "uses": "use", "works": "work", "studies": "study", "lives": "live", "is": "am",
    "has": "have", "wants": "want", "knows": "know", "codes": "code", "writes": "write", "builds": "build",
}  # fmt: skip
_VERB_ALT = "|".join(sorted(_VERBS, key=len, reverse=True))


def infer_user_subject(items: list[Item]) -> str | None:
    """A dump is about one person: the capitalised name that most often starts its sentences is that person."""
    c: Counter[str] = Counter()
    for it in items:
        if it.kind != "fact":
            continue
        m = re.match(rf"^((?:[A-Z][\w.]*)(?:\s[A-Z][\w.]*){{0,2}})\s+(?:{_VERB_ALT})\b", it.text)
        if m and m[1].lower() not in {"the", "this", "that"}:
            c[m[1]] += 1
    top = c.most_common(1)
    return top[0][0] if top and top[0][1] >= 3 else None


def first_person(statement: str, user: str | None) -> str | None:
    """'The user prefers C++ over Rust.' / 'Alex studies at Northfield University.' -> 'I prefer C++ over Rust.'"""
    names = ["the user", "user", "you"] + ([re.escape(user)] if user else [])
    m = re.match(rf"^(?:{'|'.join(names)})(?P<poss>'s)?\s+", statement, re.IGNORECASE)
    if not m:
        return None
    rest = statement[m.end():]
    if m["poss"]:
        return "My " + rest
    v = re.match(rf"^({_VERB_ALT})\b", rest, re.IGNORECASE)
    if v:
        rest = _VERBS[v[1].lower()] + rest[v.end():]
    rest = re.sub(r"^are\b", "am", rest, flags=re.IGNORECASE)
    return "I " + rest


def derive_triple(statement: str, user: str | None) -> Candidate | None:
    """Reuse the first-person rules on the statement; accept only an unambiguous single structured reading."""
    fp = first_person(statement, user)
    if not fp or len(fp) < 12:
        return None
    cands, _ = extract_from_message(fp, message_id="dump", ts=None, context_entity=None, known=set())
    structured = [c for c in cands if c.subject and c.predicate and c.object]
    return structured[0] if len(structured) == 1 else None


def _candidate(it: Item, statement: str, did: str, ts: float | None, user: str | None, *, category: str | None = None,
               derive: bool = True) -> Candidate:  # fmt: skip
    cat = _CATEGORY.get(category or it.category, category or it.category)
    conf = _CONF.get(it.confidence, 0.6)
    if it.source == "inferred":
        conf = min(conf, 0.5)
    sens = it.sensitive or sensitivity_of(statement) == "sensitive"
    c = Candidate(
        statement=statement[:400], category=cat if cat in _CATEGORY.values() or cat == "session" else "fact",
        kind="episodic" if it.kind == "episode" else "semantic", ref=did, ref_kind="document", quote=it.quote[:400],
        ts=ts, confidence=conf, verification="model_inferred" if it.source == "inferred" else "provider_stated",
        method="chatgpt_dump_import", sensitivity="sensitive" if sens else "normal", origin="chatgpt_memory",
    )  # fmt: skip
    if derive and it.kind == "fact":
        t = derive_triple(statement, user)
        if t is not None:
            c.subject, c.predicate, c.object, c.old_object, c.entities = t.subject, t.predicate, t.object, t.old_object, t.entities
            if it.category == "other":
                c.category = t.category
    return c


def seed_from_text(store: Store, path: Path, *, dry_run: bool = False) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    items = parse_dump(text)
    counts = {"facts": 0, "changes": 0, "entities": 0, "episodes": 0, "sensitive": 0, "structured": 0, "secrets_dropped": 0,
              "total_parsed": len(items)}  # fmt: skip
    user = infer_user_subject(items)
    if dry_run:
        for it in items:
            counts[{"fact": "facts", "change": "changes", "entity": "entities", "episode": "episodes"}[it.kind]] += 1
            if it.kind == "fact":
                c = _candidate(it, it.text, "dry", parse_when(it.when), user)
                counts["sensitive"] += int(c.sensitivity == "sensitive")
                counts["structured"] += int(bool(c.predicate))
        return counts
    job = "seedtext_" + sha(path.name, sha(text))[:10]
    did = _doc(store, "chatgpt_dump", path.name, text, None, job)
    for it in items:
        if contains_secret(it.text) or contains_secret(" ".join(it.chain)):
            counts["secrets_dropped"] += 1
            continue
        ts = parse_when(it.when)
        if it.kind == "entity":
            upsert_entity(store, EntityRef(it.subject, it.etype if it.etype in _ENTITY_TYPES else "concept"))
            counts["entities"] += 1
        elif it.kind == "change" and len(it.chain) >= 2:
            _seed_change(store, it, did, ts, user)
            counts["changes"] += 1
        elif it.kind in {"fact", "episode"}:
            c = _candidate(it, it.text, did, ts, user)
            upsert(store, c)
            counts["episodes" if it.kind == "episode" else "facts"] += 1
            counts["sensitive"] += int(c.sensitivity == "sensitive")
            counts["structured"] += int(bool(c.predicate))
    store.conn.commit()
    refresh_entity_links(store)
    refresh_summaries(store)
    store.conn.commit()
    return counts


def _set_current(store: Store, mid: str, ts: float | None) -> None:
    """Make a previously superseded value current again (A -> B -> A)."""
    at = ts if ts is not None else time.time()
    store.conn.execute(
        "UPDATE memories SET status='current', valid_to=NULL, invalid_at=NULL, expired_at=NULL, valid_at=?, valid_from=?, "
        "updated_at=? WHERE id=?",
        (at, at, time.time(), mid),
    )


def _status(store: Store, mid: str) -> str | None:
    row = store.conn.execute("SELECT status FROM memories WHERE id=?", (mid,)).fetchone()
    return None if row is None else str(row["status"])


def _seed_change(store: Store, it: Item, did: str, ts: float | None, user: str | None) -> None:
    """chain [a, b, c] -> one memory per value; every value but the last becomes history, linked to what replaced it.
    A value that comes back later in the chain (a, b, a) is current again and the one in between becomes history.

    Written as 'compute the end state, change only what differs' so that re-importing the same file is a no-op."""
    n = len(it.chain)
    reason = f"chatgpt_change: {canonical(it.subject)}"
    ids: list[str] = []
    whens: list[float | None] = []
    for i, val in enumerate(it.chain):
        when = ts - 86400.0 * (n - 1 - i) if ts is not None else None
        c = _candidate(it, f"{it.subject}: {val}", did, when, user, category="other", derive=False)
        ids.append(upsert(store, c).memory_id)
        whens.append(when)
    for i in range(1, n):
        if ids[i] != ids[i - 1]:
            _link(store, ids[i], "SUPERSEDES", ids[i - 1], reason)
    last = ids[-1]
    if _status(store, last) == "superseded":
        _set_current(store, last, whens[-1])
    for i, mid in enumerate(ids[:-1]):
        if mid != last and _status(store, mid) == "current":
            _supersede(store, ids[i + 1], mid, reason, whens[i + 1])


def restructure_existing(store: Store, *, dry_run: bool = False) -> dict[str, int]:
    """Give plain facts imported earlier (before structured triples existed) their triple, in place, and then let the
    normal conflict rules see them. Idempotent: only rows without a predicate are touched. Nothing is deleted or merged
    with anything else; a conflict found now is recorded the way it would have been at import time."""
    rows = store.conn.execute(
        "SELECT * FROM memories WHERE origin='chatgpt_memory' AND predicate IS NULL AND status!='deleted' AND kind='semantic'"
    ).fetchall()
    user = infer_user_subject([Item("fact", r["statement"]) for r in rows])
    out = {"candidates": len(rows), "structured": 0, "conflicts_found": 0, "skipped_key_clash": 0}
    for r in rows:
        t = derive_triple(r["statement"], user)
        if t is None:
            continue
        out["structured"] += 1
        c = Candidate(
            statement=r["statement"], category=r["category"], subject=t.subject, predicate=t.predicate, object=t.object,
            old_object=t.old_object, entities=t.entities, ref="", confidence=r["confidence"], verification=r["verification"],
            ts=r["valid_at"] if r["valid_time_known"] else None, origin=r["origin"],
        )  # fmt: skip
        dk = dedup_key_of(c)
        if store.conn.execute("SELECT 1 FROM memories WHERE dedup_key=? AND id!=?", (dk, r["id"])).fetchone():
            out["skipped_key_clash"] += 1
            continue
        if dry_run:
            continue
        before = store.conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]
        store.conn.execute("UPDATE memories SET subject=?, predicate=?, object=?, dedup_key=?, updated_at=? WHERE id=?",
                           (c.subject, c.predicate, c.object, dk, time.time(), r["id"]))  # fmt: skip
        for ref in t.entities:
            eid = upsert_entity(store, ref)
            if eid:
                store.conn.execute("INSERT OR IGNORE INTO memory_entities(memory_id, entity_id, role) VALUES (?,?,'about')", (r["id"], eid))
        _resolve_conflicts(store, r["id"], c)
        rescore(store, r["id"])
        out["conflicts_found"] += store.conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0] - before
    if not dry_run:
        store.conn.commit()
    return out


__all__ = ["parse_dump", "parse_when", "seed_from_text", "derive_triple", "infer_user_subject", "first_person", "restructure_existing"]
