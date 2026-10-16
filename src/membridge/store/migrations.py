"""Ordered, idempotent schema migrations (additive only: nothing is ever dropped).

The base schemas in sqlite.py / memory_schema.py describe version 1. Every later change lives here, so a
fresh database and an upgraded old database end up identical.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable

Migration = Callable[[sqlite3.Connection], None]


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _add_columns(conn: sqlite3.Connection, table: str, cols: dict[str, str]) -> None:
    have = _columns(conn, table)
    for name, decl in cols.items():
        if name not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _m2_bitemporal_and_graph_tables(conn: sqlite3.Connection) -> None:
    """Bi-temporal facts (Zep/Graphiti style), structured triples, entity resolution, relations, communities."""
    _add_columns(conn, "memories", {
        "valid_at": "REAL",          # world time: when the fact became true
        "invalid_at": "REAL",        # world time: when it stopped being true
        "recorded_at": "REAL",       # knowledge time: when we learned it
        "expired_at": "REAL",        # knowledge time: when we learned it was never true (retraction)
        "importance": "REAL NOT NULL DEFAULT 0",
        "activation": "REAL NOT NULL DEFAULT 0",
        "subject_entity_id": "TEXT",
        "object_entity_id": "TEXT",
        "object_value": "TEXT",      # normalised literal, e.g. ISO date
    })  # fmt: skip
    # Backfill from the v1 columns (valid_from/valid_to/created_at). Safe to re-run: only fills NULLs.
    conn.execute("UPDATE memories SET valid_at=valid_from WHERE valid_at IS NULL")
    conn.execute("UPDATE memories SET invalid_at=valid_to WHERE invalid_at IS NULL AND valid_to IS NOT NULL")
    conn.execute("UPDATE memories SET recorded_at=created_at WHERE recorded_at IS NULL")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_triple ON memories(subject_entity_id, predicate, valid_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_asof ON memories(valid_at, invalid_at)")
    _add_columns(conn, "summaries", {"child_hash": "TEXT"})
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entity_aliases (
          entity_id TEXT NOT NULL REFERENCES entities(id),
          alias_norm TEXT NOT NULL,
          alias TEXT NOT NULL,
          source TEXT NOT NULL,              -- extracted | merge | user
          first_seen REAL NOT NULL,
          PRIMARY KEY (entity_id, alias_norm)
        );
        CREATE INDEX IF NOT EXISTS idx_alias_norm ON entity_aliases(alias_norm);

        CREATE TABLE IF NOT EXISTS entity_merges (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          absorbed_id TEXT NOT NULL,
          into_id TEXT NOT NULL,
          score REAL NOT NULL,
          method TEXT NOT NULL,              -- exact | lsh_fuzzy | embedding | user | llm
          features_json TEXT NOT NULL DEFAULT '{}',
          decided_by TEXT NOT NULL,
          decided_at REAL NOT NULL,
          reversed_at REAL,
          snapshot_json TEXT NOT NULL DEFAULT '{}'   -- what was re-pointed, so a split can replay it
        );

        CREATE TABLE IF NOT EXISTS entity_link_candidates (
          a_id TEXT NOT NULL,
          b_id TEXT NOT NULL,
          score REAL NOT NULL,
          features_json TEXT NOT NULL DEFAULT '{}',
          status TEXT NOT NULL DEFAULT 'pending',    -- pending | accepted | rejected (cannot-link)
          decided_by TEXT,
          decided_at REAL,
          PRIMARY KEY (a_id, b_id)
        );

        CREATE TABLE IF NOT EXISTS entity_edges (
          src_entity TEXT NOT NULL REFERENCES entities(id),
          predicate TEXT NOT NULL,
          dst_entity TEXT NOT NULL REFERENCES entities(id),
          fact_id TEXT NOT NULL REFERENCES memories(id),
          weight REAL NOT NULL DEFAULT 1.0,
          valid_at REAL,
          invalid_at REAL,
          PRIMARY KEY (src_entity, predicate, dst_entity, fact_id)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_dst ON entity_edges(dst_entity);

        CREATE TABLE IF NOT EXISTS communities (
          id TEXT PRIMARY KEY,
          level INTEGER NOT NULL,
          parent_id TEXT,
          method TEXT NOT NULL,
          resolution REAL,
          member_hash TEXT NOT NULL,
          created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS community_members (
          community_id TEXT NOT NULL REFERENCES communities(id),
          entity_id TEXT NOT NULL REFERENCES entities(id),
          PRIMARY KEY (community_id, entity_id)
        );

        CREATE TABLE IF NOT EXISTS embeddings (
          record_id TEXT NOT NULL,
          kind TEXT NOT NULL,                -- fact | entity
          model TEXT NOT NULL,
          dim INTEGER NOT NULL,
          vector BLOB NOT NULL,
          text_hash TEXT NOT NULL,
          PRIMARY KEY (record_id, kind, model)
        );
        """
    )


def _m3_entity_merge_pointer(conn: sqlite3.Connection) -> None:
    """Merged-away entities keep their row (so merges are reversible) but leave every listing."""
    _add_columns(conn, "entities", {"merged_into": "TEXT"})
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_merged ON entities(merged_into)")
    conn.execute("DROP VIEW IF EXISTS live_entities")
    conn.execute("CREATE VIEW live_entities AS SELECT * FROM entities WHERE merged_into IS NULL")


MIGRATIONS: list[tuple[int, str, Migration]] = [
    (1, "baseline (v1 schema from sqlite.py/memory_schema.py)", lambda c: None),
    (2, "bitemporal facts, triples, aliases, merges, relations, communities, embeddings", _m2_bitemporal_and_graph_tables),
    (3, "entity merge pointer + live_entities view", _m3_entity_merge_pointer),
]
LATEST = MIGRATIONS[-1][0]


def current_version(conn: sqlite3.Connection) -> int:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL)"
    )
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0] or 0)


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations in order, each in its own transaction. Returns the versions applied."""
    done = current_version(conn)
    applied: list[int] = []
    for version, name, fn in MIGRATIONS:
        if version <= done:
            continue
        try:
            fn(conn)
            conn.execute("INSERT INTO schema_version(version, name, applied_at) VALUES (?,?,?)", (version, name, time.time()))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        applied.append(version)
    return applied
