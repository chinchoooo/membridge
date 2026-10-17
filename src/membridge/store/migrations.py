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


def _m4_scale_indexes(conn: sqlite3.Connection) -> None:
    """Found by the scale test: summary/provenance queries scanned memory_evidence once per conversation."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_ref ON memory_evidence(ref)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_links_dst ON memory_links(dst_id, rel)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mentions_conv ON entity_mentions(conversation_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_extract_msg ON extraction_state(message_id)")
    conn.execute("ANALYZE")


def _m5_cooccurrence(conn: sqlite3.Connection) -> None:
    """Materialised entity co-occurrence (weak RELATED_TO signal). Recomputing the self-join per query dominated latency."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS entity_cooccurrence (a_entity TEXT NOT NULL, b_entity TEXT NOT NULL, weight REAL NOT NULL, "
        "PRIMARY KEY (a_entity, b_entity))"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cooc_b ON entity_cooccurrence(b_entity)")
    conn.execute("DELETE FROM entity_cooccurrence")
    conn.execute("INSERT INTO entity_cooccurrence(a_entity, b_entity, weight) SELECT a.entity_id, b.entity_id, COUNT(*) FROM memory_entities a JOIN memory_entities b ON a.memory_id=b.memory_id AND a.entity_id<b.entity_id JOIN memories m ON m.id=a.memory_id JOIN live_entities ea ON ea.id=a.entity_id JOIN live_entities eb ON eb.id=b.entity_id WHERE m.status!='deleted' GROUP BY a.entity_id, b.entity_id")


# Tables whose content can change a retrieval answer. Any write by ANY process bumps meta.gen, which is what the
# result cache / warm index / PPR cache key on (PRAGMA data_version is unusable: our own audit flushes would bump it).
GEN_TABLES = (
    "memories", "memory_links", "memory_entities", "memory_evidence", "entities", "entity_aliases", "entity_edges",
    "entity_mentions", "entity_cooccurrence", "messages", "conversations", "summaries", "communities", "community_members", "embeddings",
)


def _m6_generation(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v INTEGER NOT NULL)")
    conn.execute("INSERT OR IGNORE INTO meta(k, v) VALUES ('gen', 1)")
    for t in GEN_TABLES:
        for ev in ("INSERT", "UPDATE", "DELETE"):
            conn.execute(f"DROP TRIGGER IF EXISTS gen_{t}_{ev.lower()}")
            conn.execute(f"CREATE TRIGGER gen_{t}_{ev.lower()} AFTER {ev} ON {t} BEGIN UPDATE meta SET v = v + 1 WHERE k='gen'; END")
    conn.execute("UPDATE meta SET v = v + 1 WHERE k='gen'")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS ppr_cache (
          entity_id TEXT NOT NULL, node_id TEXT NOT NULL, p REAL NOT NULL, parent_id TEXT, etype TEXT,
          PRIMARY KEY (entity_id, node_id)
        );
        CREATE TABLE IF NOT EXISTS ppr_cache_state (id INTEGER PRIMARY KEY CHECK (id = 1), gen INTEGER NOT NULL, built_at REAL NOT NULL, entities INTEGER NOT NULL);
        """
    )


def _m7_trust_and_visibility(conn: sqlite3.Connection) -> None:
    """Trust-aware conflicts, visibility of raw text, sensitive-access grants, tamper-evident audit.

    * memories.valid_time_known: 0 when the source gave no date (a ChatGPT/Claude summary); such a statement is never
      treated as *newer* than a dated one.
    * messages/documents.sensitivity: raw text is classified too, so it cannot leak around the memory-level filter.
    * sensitive_grants / settings_kv: time-limited, client-bound consent to see sensitive memories; PIN hash.
    * audit_log.prev_hash/hash: each row commits to the one before it.
    Existing rows are re-classified with the current lexicon; they are only ever *upgraded* to sensitive.
    """
    from ..memory.models import sensitivity_of

    _add_columns(conn, "memories", {"valid_time_known": "INTEGER NOT NULL DEFAULT 1"})
    _add_columns(conn, "messages", {"sensitivity": "TEXT NOT NULL DEFAULT 'normal'"})
    _add_columns(conn, "documents", {"sensitivity": "TEXT NOT NULL DEFAULT 'normal'"})
    _add_columns(conn, "audit_log", {"prev_hash": "TEXT", "hash": "TEXT"})
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sensitive_grants (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          client TEXT NOT NULL,
          created_at REAL NOT NULL,
          expires_at REAL NOT NULL,
          uses_left INTEGER NOT NULL,
          revoked INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_grants_client ON sensitive_grants(client, expires_at);
        CREATE TABLE IF NOT EXISTS settings_kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS conflict_resolutions (
          subject TEXT NOT NULL, predicate TEXT NOT NULL, kept_obj TEXT NOT NULL, dropped_obj TEXT NOT NULL,
          resolved_at REAL NOT NULL, PRIMARY KEY (subject, predicate, kept_obj, dropped_obj)
        );
        CREATE INDEX IF NOT EXISTS idx_mem_sens ON memories(sensitivity, status);
        """
    )
    conn.execute(
        "UPDATE memories SET valid_time_known=0 WHERE origin IN ('claude_memory','claude_project','chatgpt_memory') "
        "AND NOT EXISTS (SELECT 1 FROM memory_evidence e WHERE e.memory_id=memories.id AND e.ref_kind='message')"
    )
    stale = [(r[0],) for r in conn.execute("SELECT id, text FROM summaries").fetchall() if sensitivity_of(str(r[1])) == "sensitive"]
    conn.executemany("DELETE FROM summaries WHERE id=?", stale)  # regenerated by refresh_summaries without private text
    for table, col in (("memories", "statement"), ("messages", "text"), ("documents", "text")):
        pk = "id"
        rows = conn.execute(f"SELECT {pk}, {col} FROM {table} WHERE sensitivity!='sensitive'").fetchall()
        flag = [(r[0],) for r in rows if r[1] and sensitivity_of(str(r[1])) == "sensitive"]
        for i in range(0, len(flag), 500):
            conn.executemany(f"UPDATE {table} SET sensitivity='sensitive' WHERE {pk}=?", flag[i : i + 500])


def _m8_visible_entities(conn: sqlite3.Connection) -> None:
    """An entity that exists only through private memories is itself private (its name says what the memory says).

    * visible_entities: live entities with at least one normal, live memory, or with no live memory at all.
    * Rows an assistant saved as `user_stated` under v0.1 (it could) become `client_asserted`; the dashboard's own are kept.
    * Community summaries are rebuilt on the next consolidation (they were built with the old entity set).
    """
    conn.execute("DROP VIEW IF EXISTS visible_entities")
    conn.execute(
        "CREATE VIEW visible_entities AS SELECT e.* FROM live_entities e WHERE "
        "EXISTS (SELECT 1 FROM memory_entities me JOIN memories m ON m.id=me.memory_id "
        "        WHERE me.entity_id=e.id AND m.status!='deleted' AND m.sensitivity='normal') "
        "OR NOT EXISTS (SELECT 1 FROM memory_entities me JOIN memories m ON m.id=me.memory_id "
        "               WHERE me.entity_id=e.id AND m.status!='deleted')"
    )
    conn.execute(
        "UPDATE memories SET verification='client_asserted' WHERE origin='client' AND verification='user_stated' "
        "AND extraction_method NOT IN ('client:dashboard','client:cli')"
    )
    conn.execute("DELETE FROM summaries WHERE level IN ('community-L1','community-L2')")


MIGRATIONS: list[tuple[int, str, Migration]] = [
    (1, "baseline (v1 schema from sqlite.py/memory_schema.py)", lambda c: None),
    (2, "bitemporal facts, triples, aliases, merges, relations, communities, embeddings", _m2_bitemporal_and_graph_tables),
    (3, "entity merge pointer + live_entities view", _m3_entity_merge_pointer),
    (4, "scale indexes (evidence.ref, links.dst, mentions.conv)", _m4_scale_indexes),
    (5, "materialised entity co-occurrence", _m5_cooccurrence),
    (6, "generation counter triggers + PPR cache tables", _m6_generation),
    (7, "trust tiers, raw-text sensitivity, sensitive grants, audit hash chain", _m7_trust_and_visibility),
    (8, "visible_entities view, v0.1 agent-forged trust, community summaries rebuilt", _m8_visible_entities),
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
