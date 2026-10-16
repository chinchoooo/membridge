from __future__ import annotations

import json
from pathlib import Path

from membridge.importers.base import ParseFailure, RawConversation
from membridge.importers.claude_code import ClaudeCodeAdapter
from membridge.importers.notes import NotesAdapter
from membridge.importers.service import detect_adapter, run_import
from membridge.memory.pipeline import run_extraction
from membridge.store.sqlite import Store


def _session(tmp: Path) -> Path:
    d = tmp / "proj"
    d.mkdir()
    lines = [
        {"type": "mode", "sessionId": "s1", "mode": "x"},
        {"type": "ai-title", "sessionId": "s1", "aiTitle": "Atlas storage"},
        {"type": "user", "sessionId": "s1", "uuid": "u1", "parentUuid": None, "timestamp": "2026-01-01T10:00:00Z", "cwd": "C:/work/atlas",
         "message": {"role": "user", "content": "I decided to use SQLite for Atlas storage."}},
        {"type": "assistant", "sessionId": "s1", "uuid": "a1", "parentUuid": "u1", "timestamp": "2026-01-01T10:00:05Z",
         "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "tool_use", "name": "Bash", "input": {}}, {"type": "text", "text": "Noted."}]}},
        {"type": "user", "sessionId": "s1", "uuid": "u2", "parentUuid": "a1", "timestamp": "2026-01-01T10:00:09Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "content": "HUGE OUTPUT"}]}},
        {"type": "user", "sessionId": "s1", "uuid": "u3", "parentUuid": "a1", "isMeta": True, "message": {"role": "user", "content": "meta"}},
        {"type": "assistant", "sessionId": "s1", "uuid": "side", "parentUuid": "a1", "isSidechain": True, "timestamp": "2026-01-01T10:00:20Z",
         "message": {"role": "assistant", "content": "A sub-agent said something."}},
        {"type": "attachment", "sessionId": "s1", "uuid": "x1"},
    ]
    (d / "s1.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\nnot json at all\n", encoding="utf-8")
    return d


def test_claude_code_sessions_keep_words_not_machine_output(tmp_path: Path) -> None:
    d = _session(tmp_path)
    assert ClaudeCodeAdapter().detect(d)
    items = list(ClaudeCodeAdapter().iter_items(d))
    c = items[0]
    assert isinstance(c, RawConversation) and c.external_id == "s1" and c.title == "Atlas storage" and c.metadata["cwd"] == "C:/work/atlas"
    texts = {m.external_id: m for m in c.messages}
    assert set(texts) == {"u1", "a1", "side"}  # tool_result-only turn and isMeta are dropped
    assert texts["a1"].text == "Noted." and "HUGE OUTPUT" not in " ".join(m.text for m in c.messages)
    assert texts["side"].on_current_branch is False and texts["u1"].create_time is not None
    assert any("tool call" in lim for lim in c.limitations) and any("unparseable" in lim for lim in c.limitations)


def test_claude_code_import_is_idempotent_and_feeds_extraction(tmp_path: Path, store: Store) -> None:
    d = _session(tmp_path)
    assert type(detect_adapter(d)).__name__ == "ClaudeCodeAdapter"
    m = run_import(store, d)
    assert m.status == "completed" and m.conversations_new == 1 and m.messages_new == 3
    assert run_import(store, d).status == "already_imported"
    run_extraction(store)
    assert store.conn.execute("SELECT COUNT(*) FROM memories WHERE statement LIKE '%SQLite for Atlas%'").fetchone()[0] == 1


def test_notes_folder_sections_become_messages(tmp_path: Path, store: Store) -> None:
    n = tmp_path / "notes"
    (n / "sub").mkdir(parents=True)
    (n / "plan.md").write_text("intro line here\n# Goals\nI prefer Python for tooling.\n## Risks\nThe deadline moved to 5 January.\n", encoding="utf-8")
    (n / "sub" / "todo.txt").write_text("buy a keyboard and a monitor arm", encoding="utf-8")
    (n / "image.png").write_bytes(b"\x89PNG")
    ad = NotesAdapter()
    assert ad.detect(n) and len(ad.source_files(n)) == 2
    items = [i for i in ad.iter_items(n) if isinstance(i, RawConversation)]
    plan = next(i for i in items if i.external_id == "plan.md")
    assert [m.metadata.get("heading") for m in plan.messages] == [None, "Goals", "Risks"] and plan.title == "Goals"
    m = run_import(store, n, adapter_name="notes")
    assert m.conversations_new == 2
    run_extraction(store)
    assert store.conn.execute("SELECT COUNT(*) FROM memories WHERE statement LIKE '%prefer Python%'").fetchone()[0] == 1


def test_detection_is_exclusive(export_file: Path, tmp_path: Path) -> None:
    assert not NotesAdapter().detect(export_file.parent)  # a conversation export folder is never "notes"
    assert not ClaudeCodeAdapter().detect(export_file.parent)
    big = tmp_path / "big"
    big.mkdir()
    (big / "huge.md").write_bytes(b"x" * 5_100_000)
    out = list(NotesAdapter().iter_items(big))
    assert isinstance(out[0], ParseFailure) and "larger" in out[0].error
