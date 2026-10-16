"""SQLite system of record: raw archive, import jobs, FTS, audit log."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .memory_schema import MEMORY_SCHEMA

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


class Store:
    """Thin wrapper over a SQLite connection (WAL, foreign keys on)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()  # callers (MemoryService) serialise access to the shared connection
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA + MEMORY_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    # ---- audit -------------------------------------------------------
    def audit(
        self, actor: str, action: str, target: str | None = None, **detail: Any
    ) -> None:
        self.conn.execute(
            "INSERT INTO audit_log(ts, actor, action, target, detail_json) VALUES (?,?,?,?,?)",
            (time.time(), actor, action, target, json.dumps(detail)),
        )

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
