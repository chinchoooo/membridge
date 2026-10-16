"""Bottom-up community summaries (GraphRAG/RAPTOR style): level-1 from facts, level-2 from level-1 summaries.

Extractive by default (deterministic, always faithful to the leaves). Every summary stores a hash of its inputs;
refresh rewrites only summaries whose inputs changed, and summaries of vanished communities are removed. An LLM
version can replace an extractive one only if it passes the fidelity check (see llm_tasks.summary_fidelity).
"""

from __future__ import annotations

import time
from typing import Any

from ..store.sqlite import Store
from .llm_tasks import summary_fidelity
from .models import sha


def _title(store: Store, members: list[str]) -> str:
    rows = store.conn.execute(
        f"SELECT e.name, (SELECT COUNT(*) FROM memory_entities me WHERE me.entity_id=e.id) AS n FROM live_entities e "
        f"WHERE e.id IN ({','.join('?' * len(members))}) ORDER BY n DESC, e.name LIMIT 3",
        members,
    ).fetchall()
    return " / ".join(r["name"] for r in rows) or "(unnamed)"


def _put(store: Store, level: str, ref: str, text: str, child_hash: str) -> bool:
    sid = f"sum_{level}_{ref}"
    row = store.conn.execute("SELECT child_hash, method FROM summaries WHERE id=?", (sid,)).fetchone()
    if row is not None and row["child_hash"] == child_hash:
        return False
    if row is not None and row["method"] == "llm" and row["child_hash"] == child_hash:
        return False
    store.conn.execute(
        "INSERT OR REPLACE INTO summaries(id, level, ref, text, source_hash, method, updated_at, child_hash) VALUES (?,?,?,?,?,'extractive',?,?)",
        (sid, level, ref, text, child_hash, time.time(), child_hash),
    )
    return True


def refresh_community_summaries(store: Store) -> dict[str, int]:
    c = store.conn
    changed = {"community-L1": 0, "community-L2": 0, "removed": 0}
    live_ids = {r[0] for r in c.execute("SELECT id FROM communities")}
    for r in c.execute("SELECT id, level, ref FROM summaries WHERE level IN ('community-L1','community-L2')").fetchall():
        if r["ref"] not in live_ids:
            c.execute("DELETE FROM summaries WHERE id=?", (r["id"],))
            changed["removed"] += 1
    l1 = c.execute("SELECT id, parent_id FROM communities WHERE level=1 ORDER BY id").fetchall()
    heads: dict[str, list[str]] = {}
    hashes: dict[str, list[str]] = {}
    for com in l1:
        members = [r[0] for r in c.execute("SELECT entity_id FROM community_members WHERE community_id=?", (com["id"],))]
        marks = ",".join("?" * len(members))
        facts = c.execute(
            f"SELECT DISTINCT m.id, m.statement, m.status, m.category, m.updated_at, m.score FROM memories m JOIN memory_entities me ON me.memory_id=m.id "
            f"WHERE me.entity_id IN ({marks}) AND m.status!='deleted' AND m.sensitivity='normal' ORDER BY (m.status='current') DESC, m.score DESC",
            members,
        ).fetchall()
        cur = [f for f in facts if f["status"] == "current"]
        hist = [f for f in facts if f["status"] != "current"]
        title = _title(store, members)
        h = sha(com["id"], *[f"{f['id']}{f['status']}{f['updated_at']}" for f in facts])
        text = "\n".join(
            [f"Theme: {title} ({len(members)} entities, {len(cur)} current facts, {len(hist)} superseded/unresolved)"]
            + [f"- [{f['category']}] {f['statement'][:180]}" for f in cur[:8]]
            + [f"- (history) {f['statement'][:140]}" for f in hist[:3]]
        )
        changed["community-L1"] += _put(store, "community-L1", com["id"], text, h)
        if com["parent_id"]:
            heads.setdefault(com["parent_id"], []).append(f"{title}: {cur[0]['statement'][:110] if cur else 'no current facts'}")
            hashes.setdefault(com["parent_id"], []).append(h)
    for com in c.execute("SELECT id FROM communities WHERE level=2").fetchall():
        kids = heads.get(com["id"], [])
        h = sha(com["id"], *sorted(hashes.get(com["id"], [])))
        members = [r[0] for r in c.execute("SELECT entity_id FROM community_members WHERE community_id=?", (com["id"],))]
        text = "\n".join([f"Area: {_title(store, members)} ({len(kids)} themes)"] + [f"- {k}" for k in sorted(kids)[:12]])
        changed["community-L2"] += _put(store, "community-L2", com["id"], text, h)
    c.commit()
    return changed


def summarize_communities_llm(store: Store, llm: Any, min_fidelity: float = 0.8) -> dict[str, int]:
    """Optional: replace level-1 extractive summaries with model prose iff it stays faithful to the leaf facts."""
    from ..llm.base import LLMError
    from .llm_tasks import SYSTEM_SUMMARY, _Sum

    out = {"written": 0, "rejected_fidelity": 0, "errors": 0, "skipped": 0}
    for s in store.conn.execute("SELECT id, ref, text, child_hash, method FROM summaries WHERE level='community-L1'").fetchall():
        if s["method"] == "llm":
            out["skipped"] += 1
            continue
        try:
            res = _Sum.model_validate(llm.complete_json(SYSTEM_SUMMARY, f"<untrusted_statements>\n{s['text']}\n</untrusted_statements>", _Sum.model_json_schema(), max_tokens=300))
        except (LLMError, ValueError):
            out["errors"] += 1
            continue
        if summary_fidelity(res.summary, [s["text"]]) < min_fidelity:
            out["rejected_fidelity"] += 1
            continue
        store.conn.execute("UPDATE summaries SET text=?, method='llm', updated_at=? WHERE id=?", (res.summary, time.time(), s["id"]))
        out["written"] += 1
    store.conn.commit()
    return out
