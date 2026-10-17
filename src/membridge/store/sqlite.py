"""SQLite system of record: raw archive, import jobs, FTS, audit log."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .memory_schema import MEMORY_SCHEMA
from .migrations import migrate

SCHEMA = """
CREATE TABLE IF NOT EXISTS import_jobs (
  id TEXT PRIMARY KEY,
  adapter TEXT NOT NULL,
  source_path TEXT NOT NULL,
  source_sha256 TEXT NOT NULL,
  status TEXT NOT NULL,              -- running | interrupted | completed | failed
  started_at REAL NOT NULL,
  finished_at REAL,
  last_index INTEGER NOT NULL DEFAULT -1,
  stats_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_jobs_sha ON import_jobs(source_sha256);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,               -- conv_<hash(source, external_id)>
  source TEXT NOT NULL,
  external_id TEXT NOT NULL,
  title TEXT,
  create_time REAL,
  update_time REAL,
  current_node TEXT,
  content_hash TEXT NOT NULL,
  message_count INTEGER NOT NULL,
  limitations_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  first_job TEXT NOT NULL,
  last_job TEXT NOT NULL,
  UNIQUE (source, external_id)
);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,               -- msg_<hash(source, conv external id, node id)>
  conversation_id TEXT NOT NULL REFERENCES conversations(id),
  external_id TEXT NOT NULL,
  parent_external_id TEXT,
  seq INTEGER NOT NULL,
  role TEXT NOT NULL,
  author_name TEXT,
  create_time REAL,
  content_type TEXT NOT NULL,
  text TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  on_current_branch INTEGER NOT NULL,
  limitations_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  import_job TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, seq);

-- A later import carried different content for an already-stored message.
-- The original row is never overwritten; the variant is kept here.
CREATE TABLE IF NOT EXISTS message_revisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id TEXT NOT NULL REFERENCES messages(id),
  content_hash TEXT NOT NULL,
  text TEXT NOT NULL,
  import_job TEXT NOT NULL,
  seen_at REAL NOT NULL,
  UNIQUE (message_id, content_hash)
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  message_id UNINDEXED, text, tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS import_errors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  item_index INTEGER NOT NULL,
  ref TEXT,
  error TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  target TEXT,
  detail_json TEXT NOT NULL DEFAULT '{}'
);
"""


class VaultLockedError(RuntimeError):
    """The database is encrypted at rest (`membridge lock`). Opening it would silently create an empty one beside it."""


class Store:
    """Thin wrapper over a SQLite connection (WAL, foreign keys on)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() and path.with_suffix(path.suffix + ".enc").exists():
            raise VaultLockedError(f"{path.name} is locked (encrypted at rest). Run `membridge unlock` first.")
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.execute("PRAGMA busy_timeout=30000")  # a CLI import and a running server wait for each other, not fail
        self._audit_buf: list[tuple[float, str, str, str | None, str]] = []
        self._audit_thread: threading.Thread | None = None
        self._closing = threading.Event()
        self.lock = threading.RLock()  # callers (MemoryService) serialise access to the shared connection
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # safe in WAL mode; avoids an fsync per commit
        self.conn.execute("PRAGMA cache_size=-65536")  # 64 MB page cache
        self.conn.execute("PRAGMA mmap_size=268435456")  # memory-map up to 256 MB: reads skip a copy
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA + MEMORY_SCHEMA)
        migrate(self.conn)
        self.conn.commit()

    def close(self) -> None:
        self._closing.set()
        try:
            self.flush_audit()
        finally:
            self.conn.close()

    # ---- read-path audit: buffered, so a search never commits ---------------------------------------------
    def audit_read(self, actor: str, action: str, target: str | None = None, **detail: Any) -> None:
        """Audit a READ without touching the database: append to memory; a background thread flushes in batches."""
        self._audit_buf.append((time.time(), actor, action, target, json.dumps(detail, sort_keys=True)))
        if self._audit_thread is None:
            self._audit_thread = threading.Thread(target=self._audit_loop, daemon=True, name="membridge-audit")
            self._audit_thread.start()
        if len(self._audit_buf) >= 500:
            self.flush_audit()

    def _audit_loop(self) -> None:
        while not self._closing.wait(2.0):
            try:
                self.flush_audit()
            except Exception:  # connection closed or db busy: the next tick retries
                pass

    def flush_audit(self) -> int:
        with self.lock:
            buf, self._audit_buf = self._audit_buf, []
            if not buf:
                return 0
            prev = self._last_hash()
            rows = []
            for ts, actor, action, target, detail in buf:
                h = _row_hash(prev, ts, actor, action, target, detail)
                rows.append((ts, actor, action, target, detail, prev, h))
                prev = h
            self.conn.executemany(
                "INSERT INTO audit_log(ts, actor, action, target, detail_json, prev_hash, hash) VALUES (?,?,?,?,?,?,?)", rows
            )
            self.conn.commit()
            return len(buf)

    def generation(self) -> int:
        """Content generation: bumped by triggers on any write to retrieval-relevant tables, by any process."""
        row = self.conn.execute("SELECT v FROM meta WHERE k='gen'").fetchone()
        return int(row[0]) if row else 0

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    # ---- audit -------------------------------------------------------
    def _last_hash(self) -> str:
        row = self.conn.execute("SELECT hash FROM audit_log WHERE hash IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
        return "" if row is None else str(row[0])

    def audit(self, actor: str, action: str, target: str | None = None, **detail: Any) -> None:
        """Append to the audit log. Each row commits to the one before it, so deleting or editing history is detectable."""
        ts, d = time.time(), json.dumps(detail, sort_keys=True)
        prev = self._last_hash()
        self.conn.execute(
            "INSERT INTO audit_log(ts, actor, action, target, detail_json, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
            (ts, actor, action, target, d, prev, _row_hash(prev, ts, actor, action, target, d)),
        )

    def audit_sync(self, actor: str, action: str, target: str | None = None, **detail: Any) -> None:
        """Audit and commit now. Used for access to private data and credentials: these must survive a killed server."""
        self.audit(actor, action, target, **detail)
        self.conn.commit()

    # ---- read helpers ------------------------------------------------
    def get_message(self, message_id: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM messages WHERE id=?", (message_id,)
        ).fetchone()
        return row

    def conversation_hash(self, conversation_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM conversations WHERE id=?", (conversation_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def search_messages(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        # Quote each term so user text can never be parsed as FTS5 syntax.
        terms = ['"' + t.replace('"', '""') + '"' for t in query.split() if t]
        if not terms:
            return []
        return self.conn.execute(
            "SELECT m.* FROM messages_fts f JOIN messages m ON m.id=f.message_id "
            "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
            (" ".join(terms), limit),
        ).fetchall()

    def stats(self) -> dict[str, int]:
        def one(q: str) -> int:
            return int(self.conn.execute(q).fetchone()[0])

        return {
            "conversations": one("SELECT COUNT(*) FROM conversations"),
            "messages": one("SELECT COUNT(*) FROM messages"),
            "message_revisions": one("SELECT COUNT(*) FROM message_revisions"),
            "import_jobs": one("SELECT COUNT(*) FROM import_jobs"),
            "import_errors": one("SELECT COUNT(*) FROM import_errors"),
        }


def _row_hash(prev: str, ts: float, actor: str, action: str, target: str | None, detail: str) -> str:
    return hashlib.sha256(f"{prev}|{ts!r}|{actor}|{action}|{target}|{detail}".encode()).hexdigest()


def verify_audit_chain(store: Store) -> dict[str, Any]:
    """Re-derive every row hash and check each row points at the previous one. Rows written before the chain existed
    have no hash and are skipped. Two processes appending at the same instant can also leave a benign fork; a break
    that is not adjacent to such a pair means a row was edited or removed."""
    rows = store.conn.execute(
        "SELECT id, ts, actor, action, target, detail_json, prev_hash, hash FROM audit_log WHERE hash IS NOT NULL ORDER BY id"
    ).fetchall()
    breaks: list[int] = []
    prev = ""
    for r in rows:
        if r["hash"] != _row_hash(r["prev_hash"] or "", r["ts"], r["actor"], r["action"], r["target"], r["detail_json"]):
            breaks.append(r["id"])  # the row itself was altered
        elif (r["prev_hash"] or "") != prev:
            breaks.append(r["id"])  # something between this row and the one before it is gone (or a concurrent fork)
        prev = r["hash"]
    return {"hashed_rows": len(rows), "breaks": breaks, "ok": not breaks}
