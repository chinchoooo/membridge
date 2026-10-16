from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from membridge.importers.service import run_import
from membridge.memory.consolidate import activation_of, importance_of, upsert
from membridge.memory.models import Candidate, EntityRef
from membridge.memory.pipeline import run_extraction
from membridge.memory.service import MemoryService
from membridge.resolve.time import parse_date
from membridge.store.memory_schema import MEMORY_SCHEMA
from membridge.store.migrations import LATEST, current_version
from membridge.store.sqlite import SCHEMA, Store


def _ts(y: int, m: int, d: int) -> float:
    return datetime(y, m, d, 12, tzinfo=UTC).timestamp()


# ---- migrations ----------------------------------------------------------------------------------
def test_fresh_database_is_at_latest_version_and_idempotent(tmp_path: Path) -> None:
    s = Store(tmp_path / "a.db")
    assert current_version(s.conn) == LATEST
    cols = {r[1] for r in s.conn.execute("PRAGMA table_info(memories)")}
    assert {"valid_at", "invalid_at", "recorded_at", "expired_at", "importance", "activation", "object_value"} <= cols
    s.close()
    s2 = Store(tmp_path / "a.db")  # reopening applies nothing twice
    assert s2.conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == LATEST
    s2.close()


def test_v1_database_is_upgraded_and_backfilled_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.executescript(SCHEMA + MEMORY_SCHEMA)  # exactly what a pre-migration install had
    raw.execute(
        "INSERT INTO memories(id, statement, kind, category, status, verification, confidence, extraction_method, "
        "valid_from, valid_to, dedup_key, created_at, updated_at) VALUES ('mem_old','old fact','semantic','fact','superseded',"
        "'user_stated',0.9,'rule',100.0,200.0,'dk',50.0,50.0)"
    )
    raw.commit()
    raw.close()
    s = Store(path)
    row = s.conn.execute("SELECT valid_at, invalid_at, recorded_at, statement FROM memories WHERE id='mem_old'").fetchone()
    assert (row["valid_at"], row["invalid_at"], row["recorded_at"], row["statement"]) == (100.0, 200.0, 50.0, "old fact")
    assert current_version(s.conn) == LATEST
    s.close()


# ---- as-of queries ---------------------------------------------------------------------------------
@pytest.fixture
def svc(export_file: Path, store: Store) -> MemoryService:
    run_import(store, export_file)
    run_extraction(store)
    return MemoryService(store)


def test_as_of_returns_the_world_as_it_was(svc: MemoryService) -> None:
    # sample timeline: PostgreSQL decided ~2023-11-14, switched to SQLite ~2023-12-14
    before = svc.search_memory("Atlas storage database", filters={"as_of": _ts(2023, 11, 20)})
    after = svc.search_memory("Atlas storage database", filters={"as_of": _ts(2024, 1, 20)})
    b = [m["statement"] for m in before["memories"]]
    a = [m["statement"] for m in after["memories"]]
    assert any("PostgreSQL for Atlas storage" in s for s in b) and not any("switched Atlas" in s for s in b)
    assert any("switched Atlas" in s for s in a) and not any("decided to use PostgreSQL" in s for s in a)


def test_known_at_ignores_facts_we_had_not_learned_yet(svc: MemoryService, store: Store) -> None:
    long_ago = 1_000.0
    assert svc.search_memory("Atlas", filters={"known_at": long_ago})["memories"] == []
    assert svc.search_memory("Atlas", filters={"known_at": 4_000_000_000.0})["memories"]


def test_correction_retracts_instead_of_merely_ending_validity(store: Store) -> None:
    a = upsert(store, Candidate(statement="The Atlas deadline is 1 March.", category="constraint", subject="atlas",
                                predicate="deadline", object="1 March", entities=[EntityRef("Atlas")], ref="m1", ts=_ts(2026, 1, 1))).memory_id  # fmt: skip
    upsert(store, Candidate(statement="Actually, the Atlas deadline is 8 March.", category="constraint", subject="atlas",
                            predicate="deadline", object="8 March", entities=[EntityRef("Atlas")], ref="m2", ts=_ts(2026, 1, 5), correction=True))  # fmt: skip
    old = store.conn.execute("SELECT status, valid_at, invalid_at, expired_at FROM memories WHERE id=?", (a,)).fetchone()
    assert old["status"] == "superseded" and old["invalid_at"] == old["valid_at"] and old["expired_at"] is not None
    link = store.conn.execute("SELECT reason FROM memory_links WHERE dst_id=?", (a,)).fetchone()
    assert link["reason"] == "correction"


def test_changed_decision_ends_validity_but_is_not_retracted(store: Store) -> None:
    a = upsert(store, Candidate(statement="I decided to use Redis for Atlas cache.", category="decision", subject="atlas",
                                predicate="uses:cache", object="Redis", ref="m1", ts=_ts(2026, 1, 1))).memory_id  # fmt: skip
    upsert(store, Candidate(statement="I decided to use Memcached for Atlas cache.", category="decision", subject="atlas",
                            predicate="uses:cache", object="Memcached", ref="m2", ts=_ts(2026, 2, 1)))  # fmt: skip
    old = store.conn.execute("SELECT invalid_at, expired_at FROM memories WHERE id=?", (a,)).fetchone()
    assert old["invalid_at"] == _ts(2026, 2, 1) and old["expired_at"] is None


def test_equal_dates_in_different_words_are_not_a_contradiction(store: Store) -> None:
    a = upsert(store, Candidate(statement="The Atlas deadline is Friday.", category="constraint", subject="atlas", predicate="deadline",
                                object="Friday", ref="m1", ts=_ts(2026, 10, 12))).memory_id  # Monday 12 Oct 2026 -> Fri 16 Oct
    upsert(store, Candidate(statement="The Atlas deadline is 16 October.", category="constraint", subject="atlas", predicate="deadline",
                            object="16 October", ref="m2", ts=_ts(2026, 10, 13)))  # fmt: skip
    assert store.conn.execute("SELECT status FROM memories WHERE id=?", (a,)).fetchone()[0] == "current"
    assert store.conn.execute("SELECT object_value FROM memories WHERE id=?", (a,)).fetchone()[0] == "2026-10-16"


# ---- date parser -------------------------------------------------------------------------------------
REF = _ts(2026, 10, 14)  # a Wednesday


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tomorrow", "2026-10-15"), ("today", "2026-10-14"), ("yesterday", "2026-10-13"),
        ("friday", "2026-10-16"), ("next friday", "2026-10-23"), ("next wednesday", "2026-10-21"),
        ("in two weeks", "2026-10-28"), ("in 3 days", "2026-10-17"), ("2 weeks ago", "2026-09-30"),
        ("5 January", "2027-01-05"), ("January 5th", "2027-01-05"), ("15 December 2026", "2026-12-15"),
        ("2026-11-02", "2026-11-02"), ("end of october", "2026-10-31"), ("31 February", None), ("whenever", None),
    ],
)  # fmt: skip
def test_relative_date_parser(text: str, expected: str | None) -> None:
    assert parse_date(text, REF) == expected


def test_parser_never_guesses_without_a_reference_for_relative_words() -> None:
    assert parse_date("tomorrow", None) is None and parse_date("2026-01-02", None) == "2026-01-02"


# ---- activation / importance ---------------------------------------------------------------------------
def test_actr_activation_rises_with_repetition_and_falls_with_age() -> None:
    now = _ts(2026, 10, 14)
    one_old = activation_of([_ts(2025, 10, 14)], now)
    many_old = activation_of([_ts(2025, 10, 14), _ts(2025, 11, 1), _ts(2026, 1, 1), _ts(2026, 3, 1)], now)
    one_new = activation_of([_ts(2026, 10, 13)], now)
    assert one_old < many_old and one_old < one_new and 0 < one_old < 1 and activation_of([], now) == 0.0
    assert activation_of([now], now) == pytest.approx(0.5)


def test_importance_prior_prefers_user_stated_durable_facts() -> None:
    assert importance_of("decision", "user_stated") > importance_of("fact", "user_stated") > importance_of("fact", "model_inferred")
    assert importance_of("decision", "user_verified") > importance_of("decision", "user_stated")
    assert importance_of("session", "model_inferred") < 0.5


def test_old_one_off_decision_stays_retrievable(store: Store) -> None:
    mid = upsert(store, Candidate(statement="I chose Rust for the firmware.", category="decision", subject="fw", predicate="decided",
                                  object="Rust", ref="m1", ts=_ts(2015, 1, 1))).memory_id  # fmt: skip
    assert store.conn.execute("SELECT score FROM memories WHERE id=?", (mid,)).fetchone()[0] >= 0.5


def test_rescore_all_backfills_rows_that_predate_the_migration(store: Store) -> None:
    from membridge.memory.consolidate import rescore_all

    mid = upsert(store, Candidate(statement="I decided to use Rust for firmware.", category="decision", ref="m", ts=_ts(2026, 1, 1), verification="user_stated")).memory_id
    store.conn.execute("UPDATE memories SET importance=0, activation=0, score=0 WHERE id=?", (mid,))  # as a migrated legacy row
    assert rescore_all(store, now=_ts(2026, 1, 2)) == 1
    row = store.conn.execute("SELECT importance, activation, score FROM memories WHERE id=?", (mid,)).fetchone()
    assert row["importance"] > 0.9 and 0.3 < row["activation"] < 1.0 and row["score"] >= 0.5
