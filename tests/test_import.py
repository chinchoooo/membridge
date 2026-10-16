from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from membridge.importers.base import ParseFailure, RawConversation
from membridge.importers.chatgpt import ChatGPTAdapter
from membridge.importers.service import ImportError_, run_import
from membridge.store.sqlite import Store


def test_adapter_parses_branches_images_and_failures(export_file: Path) -> None:
    items = list(ChatGPTAdapter().iter_items(export_file))
    convs = [i for i in items if isinstance(i, RawConversation)]
    fails = [i for i in items if isinstance(i, ParseFailure)]
    assert len(convs) == 6 and len(fails) == 1
    kick = next(c for c in convs if c.external_id == "c-atlas-kickoff")
    assert len(kick.messages) == 4  # 3 turns + regenerated alternative kept
    alt = next(m for m in kick.messages if m.external_id == "c-atlas-kickoff-alt")
    assert alt.on_current_branch is False
    assert all(
        m.on_current_branch for m in kick.messages if m.external_id != alt.external_id
    )
    diagram = next(c for c in convs if c.external_id == "c-diagram")
    assert any("image not included" in lim for lim in diagram.messages[0].limitations)
    assert "Here is my Atlas diagram" in diagram.messages[0].text


def test_dry_run_writes_nothing(export_file: Path, store: Store) -> None:
    m = run_import(None, export_file, dry_run=True)
    assert m.conversations_new == 6 and m.messages_new == 16 and len(m.failures) == 1
    assert store.stats()["conversations"] == 0


def test_import_is_idempotent_and_provenance_round_trips(
    export_file: Path, store: Store
) -> None:
    m1 = run_import(store, export_file)
    assert m1.status == "completed" and m1.conversations_new == 6
    s1 = store.stats()
    assert (
        s1["conversations"] == 6 and s1["messages"] == 16 and s1["import_errors"] == 1
    )
    # same archive again -> short-circuit
    assert run_import(store, export_file).status == "already_imported"
    # forced re-scan -> everything unchanged, no new rows
    m3 = run_import(store, export_file, force=True)
    assert m3.conversations_unchanged == 6 and m3.messages_new == 0
    assert store.stats()["messages"] == s1["messages"]
    # original message recoverable exactly
    hit = store.search_messages("PostgreSQL Atlas storage")[0]
    row = store.get_message(hit["id"])
    assert row is not None and "I decided to use PostgreSQL" in row["text"]
    assert row["role"] == "user" and row["conversation_id"].startswith("conv_")


def test_changed_archive_adds_new_data_and_preserves_originals(
    export_file: Path, store: Store, sample_module
) -> None:  # type: ignore[no-untyped-def]
    run_import(store, export_file)
    data = sample_module.build_export()
    data[0]["mapping"]["c-atlas-kickoff-n2"]["message"]["content"]["parts"] = [
        "EDITED text"
    ]
    data.insert(
        0,
        sample_module.conv(
            "c-new", "New chat", 1.8e9, [("user", "a brand new conversation")]
        ),
    )
    p2 = export_file.parent.parent / "export_v2" / "conversations.json"
    p2.parent.mkdir()
    p2.write_text(json.dumps(data), encoding="utf-8")
    m = run_import(store, p2)
    assert (
        m.conversations_new == 1
        and m.conversations_updated == 1
        and m.message_revisions_new == 1
    )
    orig = store.search_messages("decided PostgreSQL")[0]
    assert "I decided to use PostgreSQL" in orig["text"]  # original not overwritten
    assert store.stats()["message_revisions"] == 1


def test_interrupted_import_resumes_without_duplicates(
    export_file: Path, store: Store
) -> None:
    m1 = run_import(store, export_file, stop_after=3)
    assert m1.status == "interrupted" and m1.conversations_seen == 3
    m2 = run_import(store, export_file)
    assert m2.status == "completed" and m2.resumed_from_index == 3
    st = store.stats()
    assert (
        st["conversations"] == 6 and st["messages"] == 16 and st["import_errors"] == 1
    )


def test_zip_and_directory_inputs(
    export_file: Path, tmp_path: Path, store: Store
) -> None:
    z = tmp_path / "export.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(export_file, "conversations.json")
        zf.writestr("chat.html", "<html></html>")
    assert run_import(None, z, dry_run=True).conversations_new == 6
    assert run_import(None, export_file.parent, dry_run=True).conversations_new == 6


def test_errors_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(ImportError_, match="path not found"):
        run_import(None, tmp_path / "nope.json", dry_run=True)
    bad = tmp_path / "other.txt"
    bad.write_text("x")
    with pytest.raises(ImportError_, match="no adapter recognises"):
        run_import(None, bad, dry_run=True)


def test_truncated_json_is_reported_not_fatal(tmp_path: Path, sample_module) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "conversations.json"
    text = json.dumps(sample_module.build_export())
    p.write_text(text[: len(text) // 2], encoding="utf-8")
    m = run_import(None, p, dry_run=True)
    assert m.failures and any("invalid JSON" in f["error"] for f in m.failures)


def test_fts_query_cannot_inject_syntax(export_file: Path, store: Store) -> None:
    run_import(store, export_file)
    store.search_messages('" OR text:* NEAR(')  # must not raise an FTS5 syntax error
    assert store.search_messages("") == []


def test_chatgpt_float_timestamps_are_preserved(export_file: Path, store: Store) -> None:
    run_import(store, export_file)
    assert store.conn.execute("SELECT COUNT(*) FROM conversations WHERE create_time IS NULL").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM messages WHERE create_time IS NULL").fetchone()[0] == 0
