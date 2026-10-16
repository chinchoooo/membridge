from __future__ import annotations

import json
import zipfile
from pathlib import Path

from membridge.importers.base import RawConversation
from membridge.importers.chatgpt import ChatGPTAdapter
from membridge.importers.claude import ClaudeAdapter
from membridge.importers.service import detect_adapter, run_import
from membridge.store.sqlite import Store

ROOT = "00000000-0000-4000-8000-000000000000"


def _block(t: str, **kw: object) -> dict[str, object]:
    return {"type": t, "start_timestamp": "2026-01-01T00:00:00Z", **kw}


def _export() -> list[object]:
    msgs = [
        {
            "uuid": "m1",
            "sender": "human",
            "text": "I prefer Rust for CLI tools",
            "parent_message_uuid": ROOT,
            "created_at": "2026-01-01T10:00:00Z",
            "attachments": [{"file_name": "notes.pdf"}],
            "files": [],
            "content": [
                _block("text", text="I prefer Rust for CLI tools"),
                _block("image", file_uuid="f1"),
            ],
        },
        {
            "uuid": "m2",
            "sender": "assistant",
            "text": "Noted.",
            "parent_message_uuid": "m1",
            "created_at": "2026-01-01T10:00:05Z",
            "attachments": [],
            "files": [],
            "content": [
                _block("thinking", thinking="secret reasoning"),
                _block("tool_use", name="web_search", input={"q": "x"}),
                _block("tool_result", content="HUGE RESULT"),
                _block("text", text="Noted."),
            ],
        },
        {
            "uuid": "m2b",
            "sender": "assistant",
            "text": "Alt.",
            "parent_message_uuid": "m1",
            "created_at": "2026-01-01T10:00:04Z",
            "attachments": [],
            "files": [],
            "content": [],
        },
    ]
    return [
        {
            "uuid": "c1",
            "name": "Tools",
            "summary": "",
            "created_at": "2026-01-01T10:00:00Z",
            "updated_at": "2026-01-01T10:01:00Z",
            "account": {},
            "chat_messages": msgs,
        },
        42,
    ]


def _zip(tmp: Path) -> Path:
    z = tmp / "conversations-000.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("conversations.json", json.dumps(_export()))
    return z


def test_claude_parsing(tmp_path: Path) -> None:
    items = list(ClaudeAdapter().iter_items(_zip(tmp_path)))
    conv = items[0]
    assert (
        isinstance(conv, RawConversation) and len(items) == 2
    )  # second item is a parse failure
    by = {m.external_id: m for m in conv.messages}
    assert by["m1"].role == "user" and by["m1"].parent_external_id is None
    assert any("image not included" in lim for lim in by["m1"].limitations)
    assert any("attachment" in lim for lim in by["m1"].limitations)
    assert by["m2"].text == "Noted."  # reasoning and tool output are not stored as text
    assert (
        "secret reasoning" not in by["m2"].text and "HUGE RESULT" not in by["m2"].text
    )
    assert by["m2"].metadata == {"tools_used": {"web_search": 1}}
    assert by["m2b"].on_current_branch is False and by["m2"].on_current_branch is True


def test_adapters_are_told_apart_by_content(tmp_path: Path, export_file: Path) -> None:
    z = _zip(tmp_path)
    assert isinstance(detect_adapter(z), ClaudeAdapter)
    assert isinstance(detect_adapter(export_file), ChatGPTAdapter)


def test_claude_import_idempotent(tmp_path: Path, store: Store) -> None:
    z = _zip(tmp_path)
    m = run_import(store, z)
    assert m.conversations_new == 1 and len(m.failures) == 1
    assert run_import(store, z).status == "already_imported"
    assert store.stats()["messages"] == 3
    assert store.search_messages("Rust CLI")[0]["role"] == "user"
