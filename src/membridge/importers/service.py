"""Import orchestration: dry-run, batching, checkpoints/resume, idempotent re-import."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..store.sqlite import Store
from .base import ImportAdapter, ParseFailure, RawConversation
from .chatgpt import ChatGPTAdapter
from .claude import ClaudeAdapter

ADAPTERS: dict[str, ImportAdapter] = {
    "chatgpt": ChatGPTAdapter(),
    "claude": ClaudeAdapter(),
}


def _h(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def conversation_id(source: str, external_id: str) -> str:
    return "conv_" + _h(source, external_id)[:32]


def message_id(source: str, conv_external_id: str, msg_external_id: str) -> str:
    return "msg_" + _h(source, conv_external_id, msg_external_id)[:32]


def _conv_hash(c: RawConversation) -> str:
    payload = json.dumps(
        [c.title, [(m.external_id, m.role, m.text, m.create_time) for m in c.messages]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_files(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        with f.open("rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


@dataclass
class Manifest:
    job_id: str
    adapter: str
    source_path: str
    source_sha256: str
    dry_run: bool
    status: str = "completed"  # completed | interrupted | already_imported | failed
    resumed_from_index: int | None = None
    conversations_seen: int = 0
    conversations_new: int = 0
    conversations_updated: int = 0
    conversations_unchanged: int = 0
    messages_new: int = 0
    messages_existing: int = 0
    message_revisions_new: int = 0
    messages_with_limitations: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class ImportError_(Exception):
    pass


def detect_adapter(path: Path, name: str | None = None) -> ImportAdapter:
    if name:
        if name not in ADAPTERS:
            raise ImportError_(
                f"unknown adapter '{name}'. available: {', '.join(ADAPTERS)}"
            )
        adapter = ADAPTERS[name]
        if not adapter.detect(path):
            raise ImportError_(f"'{path}' does not look like a {name} export")
        return adapter
    for adapter in ADAPTERS.values():
        if adapter.detect(path):
            return adapter
    raise ImportError_(
        f"no adapter recognises '{path}'. Expected conversations.json, a folder or a .zip"
    )


def run_import(
    store: Store | None,
    path: Path,
    *,
    adapter_name: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    batch_size: int = 50,
    report_dir: Path | None = None,
    stop_after: int | None = None,  # test hook: simulate an interruption after N items
) -> Manifest:
    path = path.expanduser().resolve()
    if not path.exists():
        raise ImportError_(f"path not found: {path}")
    if not dry_run and store is None:
        raise ImportError_("a store is required unless --dry-run")
    adapter = detect_adapter(path, adapter_name)
    sha = sha256_files(adapter.source_files(path))
    job_id = "job_" + uuid.uuid4().hex[:12]
    m = Manifest(job_id, adapter.name, str(path), sha, dry_run)

    start_index = 0
    if store is not None and not dry_run:
        done = store.conn.execute(
            "SELECT id FROM import_jobs WHERE source_sha256=? AND status='completed'",
            (sha,),
        ).fetchone()
        if done and not force:
            m.status = "already_imported"
            m.job_id = done[0]
            m.notes.append(
                "This exact archive was already imported; nothing to do (use --force to re-scan)."
            )
            return m
        prev = store.conn.execute(
            "SELECT id, last_index FROM import_jobs WHERE source_sha256=? "
            "AND status IN ('running','interrupted') ORDER BY started_at DESC LIMIT 1",
            (sha,),
        ).fetchone()
        if prev:
            m.job_id, start_index = prev[0], int(prev[1]) + 1
            m.resumed_from_index = start_index
            store.conn.execute(
                "UPDATE import_jobs SET status='running' WHERE id=?", (m.job_id,)
            )
        else:
            store.conn.execute(
                "INSERT INTO import_jobs(id, adapter, source_path, source_sha256, status, started_at) "
                "VALUES (?,?,?,?,'running',?)",
                (m.job_id, adapter.name, str(path), sha, time.time()),
            )
        store.conn.commit()

    index = -1
    try:
        for index, item in enumerate(adapter.iter_items(path)):
            if index < start_index:
                continue
            if stop_after is not None and index - start_index >= stop_after:
                m.status = "interrupted"
                break
            m.conversations_seen += 1
            if isinstance(item, ParseFailure):
                m.failures.append(
                    {"index": index, "ref": item.ref, "error": item.error}
                )
                if store is not None and not dry_run:
                    store.conn.execute(
                        "INSERT INTO import_errors(job_id, item_index, ref, error) VALUES (?,?,?,?)",
                        (m.job_id, index, item.ref, item.error),
                    )
            else:
                _ingest(store, adapter.name, item, m, m.job_id, dry_run)
            if (
                store is not None
                and not dry_run
                and (m.conversations_seen % batch_size == 0)
            ):
                _checkpoint(store, m.job_id, index)
        else:
            m.status = "completed"
    except BaseException:
        if store is not None and not dry_run:
            store.conn.rollback()
            store.conn.execute(
                "UPDATE import_jobs SET status='interrupted' WHERE id=?", (m.job_id,)
            )
            store.conn.commit()
        raise

    if store is not None and not dry_run:
        last = index if m.status == "completed" else index - 1
        store.conn.execute(
            "UPDATE import_jobs SET status=?, last_index=?, finished_at=?, stats_json=? WHERE id=?",
            (
                m.status,
                last,
                time.time() if m.status == "completed" else None,
                m.to_json(),
                m.job_id,
            ),
        )
        store.audit("importer", "import", m.job_id, status=m.status, source_sha256=sha)
        store.conn.commit()
    if report_dir is not None and not dry_run:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{m.job_id}.manifest.json").write_text(
            m.to_json(), encoding="utf-8"
        )
    return m


def _checkpoint(store: Store, job_id: str, index: int) -> None:
    store.conn.execute(
        "UPDATE import_jobs SET last_index=? WHERE id=?", (index, job_id)
    )
    store.conn.commit()


def _ingest(
    store: Store | None,
    source: str,
    c: RawConversation,
    m: Manifest,
    job_id: str,
    dry_run: bool,
) -> None:
    cid = conversation_id(source, c.external_id)
    chash = _conv_hash(c)
    existing = store.conversation_hash(cid) if store is not None else None
    if existing == chash:
        m.conversations_unchanged += 1
        m.messages_existing += len(c.messages)
        return
    m.messages_with_limitations += sum(1 for x in c.messages if x.limitations)
    if existing is None:
        m.conversations_new += 1
    else:
        m.conversations_updated += 1
        m.notes.append(
            f"conversation {c.external_id} changed since last import; originals preserved"
        )
    if store is None or dry_run:
        m.messages_new += len(c.messages)
        return

    conn = store.conn
    now = time.time()
    if existing is None:
        conn.execute(
            "INSERT INTO conversations(id, source, external_id, title, create_time, update_time, "
            "current_node, content_hash, message_count, limitations_json, metadata_json, first_job, last_job) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                source,
                c.external_id,
                c.title,
                c.create_time,
                c.update_time,
                c.current_node,
                chash,
                len(c.messages),
                json.dumps(c.limitations),
                json.dumps(c.metadata),
                job_id,
                job_id,
            ),
        )
    else:
        conn.execute(
            "UPDATE conversations SET title=?, update_time=?, current_node=?, content_hash=?, "
            "message_count=?, limitations_json=?, last_job=? WHERE id=?",
            (
                c.title,
                c.update_time,
                c.current_node,
                chash,
                len(c.messages),
                json.dumps(c.limitations),
                job_id,
                cid,
            ),
        )
    for msg in c.messages:
        mid = message_id(source, c.external_id, msg.external_id)
        mhash = _h(msg.role, msg.text)
        row = conn.execute(
            "SELECT content_hash FROM messages WHERE id=?", (mid,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO messages(id, conversation_id, external_id, parent_external_id, seq, role, "
                "author_name, create_time, content_type, text, content_hash, on_current_branch, "
                "limitations_json, metadata_json, import_job) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mid,
                    cid,
                    msg.external_id,
                    msg.parent_external_id,
                    msg.seq,
                    msg.role,
                    msg.author_name,
                    msg.create_time,
                    msg.content_type,
                    msg.text,
                    mhash,
                    int(msg.on_current_branch),
                    json.dumps(msg.limitations),
                    json.dumps(msg.metadata),
                    job_id,
                ),
            )
            conn.execute(
                "INSERT INTO messages_fts(message_id, text) VALUES (?,?)",
                (mid, msg.text),
            )
            m.messages_new += 1
        elif row[0] == mhash:
            m.messages_existing += 1
        else:
            cur = conn.execute(
                "INSERT OR IGNORE INTO message_revisions(message_id, content_hash, text, import_job, seen_at) "
                "VALUES (?,?,?,?,?)",
                (mid, mhash, msg.text, job_id, now),
            )
            m.message_revisions_new += cur.rowcount
