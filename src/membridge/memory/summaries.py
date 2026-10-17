"""Hierarchical summaries (extractive, deterministic). LLM summaries may replace them (method='llm').

Summaries are derived data: each stores a hash of its inputs so it is rebuilt only when the
underlying memories change, and none of them replaces the raw archive.
"""

from __future__ import annotations

import time

from ..store.sqlite import Store
from .models import sha


def _put(store: Store, level: str, ref: str, text: str, source_hash: str) -> bool:
    sid = f"sum_{level}_{ref}"
    row = store.conn.execute(
        "SELECT source_hash, method FROM summaries WHERE id=?", (sid,)
    ).fetchone()
    if row is not None and (
        row["source_hash"] == source_hash or row["method"] == "llm"
    ):
        return False
    store.conn.execute(
        "INSERT OR REPLACE INTO summaries(id, level, ref, text, source_hash, method, updated_at) VALUES (?,?,?,?,?,'extractive',?)",
        (sid, level, ref, text, source_hash, time.time()),
    )
    return True


def refresh_summaries(store: Store, *, only_projects: set[str] | None = None) -> dict[str, int]:
    """Rebuild stale summaries. `only_projects` (entity ids) limits the work to those projects plus the personal summary:
    a single new decision must not re-read every conversation."""
    changed = {"conversation": 0, "project": 0, "personal": 0}
    c = store.conn
    # conversation level: title, opening user message, memories supported by it
    for conv in [] if only_projects is not None else c.execute(
        "SELECT id, title, message_count FROM conversations"
    ).fetchall():
        first = c.execute(
            "SELECT text FROM messages WHERE conversation_id=? AND role='user' AND text!='' AND sensitivity='normal' ORDER BY seq LIMIT 1",
            (conv["id"],),
        ).fetchone()
        mems = c.execute(
            "SELECT DISTINCT m.id, m.statement, m.updated_at FROM memories m JOIN memory_evidence e ON e.memory_id=m.id "
            "JOIN messages g ON g.id=e.ref WHERE g.conversation_id=? AND m.status!='deleted' AND m.sensitivity='normal' "
            "ORDER BY m.score DESC LIMIT 5",
            (conv["id"],),
        ).fetchall()
        sh = sha(
            conv["id"],
            first["text"][:200] if first else "",
            *[f"{m['id']}{m['updated_at']}" for m in mems],
        )
        lines = [f"{conv['title'] or 'Untitled'} ({conv['message_count']} messages)"]
        if first:
            lines.append("Opens with: " + first["text"][:200].replace("\n", " "))
        lines += [f"- {m['statement'][:160]}" for m in mems]
        changed["conversation"] += _put(
            store, "conversation", conv["id"], "\n".join(lines), sh
        )
    # project level: current memories about the project entity, plus history count
    for e in c.execute("SELECT id, name FROM live_entities WHERE type='project'").fetchall():
        if only_projects is not None and e["id"] not in only_projects:
            continue
        mems = c.execute(
            "SELECT DISTINCT m.id, m.statement, m.status, m.category, m.updated_at FROM memories m "
            "JOIN memory_entities me ON me.memory_id=m.id WHERE me.entity_id=? AND m.status!='deleted' AND m.sensitivity='normal' "
            "ORDER BY m.status='current' DESC, m.score DESC",
            (e["id"],),
        ).fetchall()
        cur = [m for m in mems if m["status"] == "current"]
        hist = [m for m in mems if m["status"] != "current"]
        sh = sha(e["id"], *[f"{m['id']}{m['status']}{m['updated_at']}" for m in mems])
        lines = [
            f"Project {e['name']}: {len(cur)} current, {len(hist)} superseded/unresolved"
        ]
        lines += [f"- [{m['category']}] {m['statement'][:200]}" for m in cur[:12]]
        lines += [
            f"- (history, {m['status']}) {m['statement'][:160]}" for m in hist[:6]
        ]
        changed["project"] += _put(store, "project", e["id"], "\n".join(lines), sh)
    # personal level: highest-scoring current memories by category
    top = c.execute(
        "SELECT id, category, statement, updated_at FROM memories WHERE status='current' AND sensitivity='normal' "
        "ORDER BY score DESC LIMIT 40"
    ).fetchall()
    by: dict[str, list[str]] = {}
    for m in top:
        by.setdefault(m["category"], []).append(m["statement"][:180])
    text = "\n".join(
        f"{cat}:\n" + "\n".join(f"- {s}" for s in items[:6])
        for cat, items in by.items()
    )
    changed["personal"] += _put(
        store,
        "personal",
        "me",
        text or "No memories yet.",
        sha(*[f"{m['id']}{m['updated_at']}" for m in top]),
    )
    c.commit()
    return changed
