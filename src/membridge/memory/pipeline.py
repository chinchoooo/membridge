"""Incremental extraction pipeline: messages -> memories/entities -> links -> summaries."""

from __future__ import annotations

import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..store.sqlite import Store
from .consolidate import refresh_entity_links, upsert, upsert_entity
from .models import (
    Candidate,
    EntityRef,
    canonical,
    contains_secret,
    sensitivity_of,
    sha,
)
from .rules import EXTRACTOR_VERSION, extract_from_message, find_entities
from .summaries import refresh_summaries


@dataclass
class ExtractionStats:
    messages_scanned: int = 0
    messages_skipped_done: int = 0
    candidates: int = 0
    created: int = 0
    confirmed: int = 0
    skipped_deleted: int = 0
    secrets_dropped: int = 0
    entities_merged: int = 0
    by_category: dict[str, int] = field(default_factory=dict)


def _known_entities(store: Store) -> set[str]:
    return {
        r[0]
        for r in store.conn.execute(
            "SELECT canonical FROM live_entities WHERE type='project'"
        )
    }


def run_extraction(
    store: Store, *, limit_conversations: int | None = None, embedder: Any = None
) -> ExtractionStats:
    """Process user messages not yet seen by this extractor version (safe to re-run)."""
    st = ExtractionStats()
    known = _known_entities(store)
    convs = store.conn.execute(
        "SELECT id FROM conversations ORDER BY IFNULL(create_time,0), id"
        + (f" LIMIT {int(limit_conversations)}" if limit_conversations else "")
    ).fetchall()
    for conv in convs:
        context_entity: str | None = None
        rows = store.conn.execute(
            "SELECT m.id, m.text, m.create_time FROM messages m WHERE m.conversation_id=? AND m.role='user' "
            "AND m.on_current_branch=1 AND m.text!='' ORDER BY m.seq",
            (conv["id"],),
        ).fetchall()
        for r in rows:
            if store.conn.execute(
                "SELECT 1 FROM extraction_state WHERE message_id=? AND extractor=?",
                (r["id"], EXTRACTOR_VERSION),
            ).fetchone():
                st.messages_skipped_done += 1
                continue
            st.messages_scanned += 1
            cands, dropped = extract_from_message(
                r["text"],
                message_id=r["id"],
                ts=r["create_time"],
                context_entity=context_entity,
                known=known,
            )
            st.secrets_dropped += dropped
            for c in cands:
                st.candidates += 1
                out = upsert(store, c)
                if out.action == "created":
                    st.created += 1
                    st.by_category[c.category] = st.by_category.get(c.category, 0) + 1
                elif out.action == "confirmed":
                    st.confirmed += 1
                else:
                    st.skipped_deleted += 1
                if c.category == "project" and c.object:
                    known.add(canonical(c.object))
                    context_entity = canonical(c.object)
            if context_entity is None:
                ents = [
                    canonical(e.name)
                    for e in find_entities(r["text"][:400], known)
                    if canonical(e.name) in known
                ]
                context_entity = ents[0] if ents else None
            store.conn.execute(
                "INSERT OR IGNORE INTO extraction_state(message_id, extractor, processed_at) VALUES (?,?,?)",
                (r["id"], EXTRACTOR_VERSION, time.time()),
            )
        store.conn.commit()
    refresh_entity_links(store)
    from ..resolve.entities import resolve_entities

    st.entities_merged = len(resolve_entities(store, embedder=embedder).auto_merged)
    from ..resolve.relations import rebuild_entity_edges

    rebuild_entity_edges(store)
    from ..graphalgo.communities import build_communities
    from .community_summaries import refresh_community_summaries

    build_communities(store)
    refresh_summaries(store)
    refresh_community_summaries(store)
    if embedder is not None:
        from ..embed.index import index_embeddings

        index_embeddings(store, embedder)
    store.audit(
        "extractor",
        "extract",
        None,
        **{k: v for k, v in st.__dict__.items() if k != "by_category"},
    )
    store.conn.commit()
    return st


# ---- seeding from Claude's own memory files and projects -----------------------------------
_PATH_CATEGORY = {
    "profile": "profile", "projects": "project", "tech-stack": "skill", "education": "education",
    "diet-and-health": "health", "research-and-internships": "research", "artist-extension": "interest",
}  # fmt: skip
_BULLET = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")


def _doc(
    store: Store, kind: str, name: str, text: str, updated: str | None, job: str
) -> str:
    did = "doc_" + sha(kind, name)[:24]
    store.conn.execute(
        "INSERT OR REPLACE INTO documents(id, kind, name, text, content_hash, updated_at, import_job) VALUES (?,?,?,?,?,?,?)",
        (did, kind, name, text, sha(text), updated, job),
    )
    return did


_PREFIX_TAG = re.compile(r"^\[(?P<tag>[A-Za-z]{3,12})\]\s*")
_SOURCE_TAG = re.compile(r"\s*\[s:(?P<src>[0-9a-f]{6,})\]\s*$")


def clean_claude_bullet(text: str) -> tuple[str, str | None, str | None]:
    """'[stated] Prefers C++ [s:d015e9]' -> ('Prefers C++', 'stated', 'd015e9'). The markup is Claude's own bookkeeping:
    the tag says how Claude came to know it, the id points at the conversation it came from."""
    tag = src = None
    if m := _PREFIX_TAG.match(text):
        tag, text = m["tag"].lower(), text[m.end():]
    if m := _SOURCE_TAG.search(text):
        src, text = m["src"], text[: m.start()]
    return text.strip(), tag, src


def purge_claude_seeds(store: Store) -> int:
    """Hard-delete memories derived purely from Claude's memory/project files so they can be re-derived cleanly."""
    c = store.conn
    ids = [r[0] for r in c.execute("SELECT id FROM memories WHERE origin IN ('claude_memory','claude_project')")]
    for mid in ids:
        for sql in (
            "DELETE FROM memory_evidence WHERE memory_id=?", "DELETE FROM memories_fts WHERE memory_id=?",
            "DELETE FROM memory_entities WHERE memory_id=?", "DELETE FROM memory_links WHERE src_id=? OR dst_id=?",
            "DELETE FROM entity_edges WHERE fact_id=?", "DELETE FROM embeddings WHERE record_id=?",
            "DELETE FROM memory_revisions WHERE memory_id=?", "DELETE FROM memories WHERE id=?",
        ):  # fmt: skip
            c.execute(sql, (mid, mid) if sql.count("?") == 2 else (mid,))
    c.commit()
    return len(ids)


def seed_from_claude_export(store: Store, zips: list[Path], *, replace: bool = False) -> dict[str, int]:
    """memories-*.zip bullets -> provider_stated memories; projects-*.zip -> project entities."""
    job = "seed_" + sha(*[str(z) for z in zips])[:10]
    counts = {"memory_files": 0, "memories": 0, "projects": 0}
    if replace:
        counts["purged"] = purge_claude_seeds(store)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                data = json.loads(zf.read(name))
                for mf in (
                    data.get("memory_files", []) if isinstance(data, dict) else []
                ):
                    path = str(mf.get("path", ""))
                    text = str(mf.get("content", ""))
                    did = _doc(
                        store, "claude_memory", path, text, mf.get("updated_at"), job
                    )
                    counts["memory_files"] += 1
                    stem = path.rsplit("/", 1)[-1].removesuffix(".md")
                    category = _PATH_CATEGORY.get(stem, "fact")
                    for line in text.splitlines():
                        m = _BULLET.match(line)
                        if not m or contains_secret(m.group(1)):
                            continue
                        original = m.group(1)
                        stmt, tag, _src = clean_claude_bullet(original)
                        if not stmt:
                            continue
                        c = Candidate(
                            statement=stmt, category="health" if sensitivity_of(stmt) == "sensitive" and category == "fact" else category,
                            kind="semantic", ref=did, ref_kind="document", quote=original,
                            entities=find_entities(stmt), confidence=0.8, verification="model_inferred" if tag == "inferred" else "provider_stated",
                            method="claude_memory_import", sensitivity=sensitivity_of(stmt), origin="claude_memory",
                        )  # fmt: skip
                        upsert(store, c)
                        counts["memories"] += 1
                if (
                    isinstance(data, dict)
                    and "uuid" in data
                    and "name" in data
                    and "docs" in data
                ):
                    desc = str(data.get("description") or "")
                    did = _doc(
                        store,
                        "claude_project",
                        str(data["name"]),
                        json.dumps({k: data.get(k) for k in ("name", "description")}),
                        data.get("updated_at"),
                        job,
                    )
                    upsert_entity(store, EntityRef(str(data["name"]), "project"))
                    if desc:
                        upsert(store, Candidate(
                            statement=f"Project {data['name']}: {desc}"[:400], category="project", subject=canonical(str(data["name"])),
                            predicate="is_project", object=str(data["name"]), ref=did, ref_kind="document", quote=desc[:400],
                            entities=[EntityRef(str(data["name"]), "project")], confidence=0.8,
                            verification="provider_stated", method="claude_project_import", origin="claude_project",
                        ))  # fmt: skip
                    counts["projects"] += 1
    store.conn.commit()
    refresh_entity_links(store)
    refresh_summaries(store)
    store.conn.commit()
    return counts
