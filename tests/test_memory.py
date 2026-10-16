from __future__ import annotations

from pathlib import Path

from membridge.importers.service import run_import
from membridge.memory.pipeline import run_extraction
from membridge.store.sqlite import Store


def _mem(store: Store, text: str) -> list[dict]:  # type: ignore[type-arg]
    return [
        dict(r)
        for r in store.conn.execute(
            "SELECT * FROM memories WHERE statement LIKE ?", (f"%{text}%",)
        )
    ]


def test_extraction_creates_provenanced_memories(
    export_file: Path, store: Store
) -> None:
    run_import(store, export_file)
    st = run_extraction(store)
    assert st.created >= 6 and st.by_category["decision"] >= 2
    row = _mem(store, "decided to use PostgreSQL")[0]
    assert row["verification"] == "user_stated" and row["category"] == "decision"
    ev = store.conn.execute(
        "SELECT ref, quote FROM memory_evidence WHERE memory_id=?", (row["id"],)
    ).fetchall()
    msg = store.get_message(ev[0]["ref"])  # provenance round-trip to the raw archive
    assert msg is not None and ev[0]["quote"] in msg["text"]


def test_extraction_is_idempotent(export_file: Path, store: Store) -> None:
    run_import(store, export_file)
    run_extraction(store)
    n = store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    st2 = run_extraction(store)
    assert st2.messages_scanned == 0 and st2.created == 0
    assert store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == n


def test_switch_supersedes_and_keeps_history(export_file: Path, store: Store) -> None:
    run_import(store, export_file)
    run_extraction(store)
    old = _mem(store, "decided to use PostgreSQL")[0]
    new = _mem(store, "switched Atlas from PostgreSQL to SQLite")[0]
    assert old["status"] == "superseded" and new["status"] == "current"
    assert old["valid_to"] == new["valid_from"]
    link = store.conn.execute(
        "SELECT reason FROM memory_links WHERE src_id=? AND rel='SUPERSEDES' AND dst_id=?",
        (new["id"], old["id"]),
    ).fetchone()
    assert link["reason"] == "explicit_switch"
    assert (
        store.conn.execute(
            "SELECT COUNT(*) FROM memory_revisions WHERE memory_id=?", (old["id"],)
        ).fetchone()[0]
        == 1
    )


def test_deadline_update_supersedes(export_file: Path, store: Store) -> None:
    run_import(store, export_file)
    run_extraction(store)
    rows = {
        r["object"]: r["status"]
        for r in store.conn.execute(
            "SELECT object, status FROM memories WHERE predicate='deadline'"
        )
    }
    assert rows == {"15 December": "superseded", "5 January": "current"}


def test_injection_text_is_data_not_memory(export_file: Path, store: Store) -> None:
    run_import(store, export_file)
    run_extraction(store)
    assert _mem(store, "rm -rf") == [] and _mem(store, "admin mode") == []
    assert (
        store.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE status='deleted'"
        ).fetchone()[0]
        == 0
    )


def test_forgotten_memory_is_not_resurrected(export_file: Path, store: Store) -> None:
    run_import(store, export_file)
    run_extraction(store)
    mid = _mem(store, "prefer Python")[0]["id"]
    store.conn.execute("UPDATE memories SET status='deleted' WHERE id=?", (mid,))
    store.conn.execute("DELETE FROM extraction_state")
    st = run_extraction(store)
    assert st.skipped_deleted >= 1
    assert (
        store.conn.execute("SELECT status FROM memories WHERE id=?", (mid,)).fetchone()[
            0
        ]
        == "deleted"
    )


def test_secrets_are_never_stored(store: Store) -> None:
    from membridge.memory.rules import extract_from_message

    cands, dropped = extract_from_message(
        "I decided to use key sk-abcdefghijklmnopqrstuvwxyz123456 for Atlas.",
        message_id="m",
        ts=1.0,
        context_entity=None,
        known=set(),
    )
    assert cands == [] and dropped == 1


def test_same_sentence_in_another_category_is_one_memory(store: Store) -> None:
    from membridge.memory.consolidate import upsert
    from membridge.memory.models import Candidate

    a = upsert(store, Candidate(statement="The Atlas deadline is 5 January.", category="constraint", ref="m1", ts=1.0))
    b = upsert(store, Candidate(statement="The Atlas deadline is 5 January.", category="decision", ref="m2", ts=2.0))
    assert a.memory_id == b.memory_id and b.action == "confirmed"
    assert store.conn.execute("SELECT COUNT(*) FROM memory_evidence WHERE memory_id=?", (a.memory_id,)).fetchone()[0] == 2
