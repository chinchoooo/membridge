"""First-run experience for a normal user (all synthetic)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from membridge import quickstart as qs
from membridge.cli import app
from membridge.demo_data import build_export
from membridge.store.sqlite import Store

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setenv("MEMBRIDGE_DATA_DIR", str(d))
    monkeypatch.delenv("MEMBRIDGE_DASHBOARD_TOKEN", raising=False)
    return d


def test_demo_walkthrough_works_end_to_end_without_questions(home: Path) -> None:
    r = runner.invoke(app, ["quickstart", "--demo", "--yes", "--no-serve"])
    assert r.exit_code == 0, r.output
    for step in ("[1/6]", "[2/6]", "[3/6]", "[4/6]", "[5/6]", "[6/6]"):
        assert step in r.output
    assert "SQLite" in r.output  # the Atlas answer follows the change
    assert "nothing stored about that, and it does not guess" in r.output  # abstention is shown, not just claimed
    assert "claude mcp add membridge" in r.output and "membridge.mcp_server" in r.output
    s = Store(home / "membridge.db")
    assert s.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] >= 15
    assert s.conn.execute("SELECT COUNT(*) FROM memories WHERE sensitivity!='normal'").fetchone()[0] >= 3  # private facts really are private
    s.close()


def test_running_the_demo_twice_does_not_duplicate_anything(home: Path) -> None:
    runner.invoke(app, ["quickstart", "--demo", "--yes", "--no-serve"])
    n = Store(home / "membridge.db").conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    again = runner.invoke(app, ["quickstart", "--demo", "--yes", "--no-serve"])
    assert "leaving it as it is" in again.output
    assert Store(home / "membridge.db").conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == n


def test_importing_your_own_export_through_the_wizard(home: Path, tmp_path: Path) -> None:
    exp = tmp_path / "export" / "conversations.json"
    exp.parent.mkdir()
    exp.write_text(json.dumps(build_export()), encoding="utf-8")
    r = runner.invoke(app, ["quickstart", "--yes", "--no-serve", "--export", str(exp)])
    assert r.exit_code == 0, r.output
    assert "Preview:" in r.output and "Imported." in r.output
    s = Store(home / "membridge.db")
    assert s.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 6
    assert s.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] > 0
    s.close()
    assert "PIN" in r.output and "membridge set-pin" in r.output  # told how to protect private facts; --yes never invents a PIN


def test_a_wrong_path_is_explained_not_a_crash(home: Path, tmp_path: Path) -> None:
    r = runner.invoke(app, ["quickstart", "--yes", "--no-serve", "--export", str(tmp_path / "nope.zip")])
    assert r.exit_code == 0 and "Could not import that" in r.output


def test_without_an_export_it_tells_you_what_to_do_next(home: Path) -> None:
    r = runner.invoke(app, ["quickstart", "--yes", "--no-serve"])
    assert r.exit_code == 0 and "Skipped." in r.output and "membridge import" in r.output and "Nothing stored yet" in r.output


def test_demo_never_touches_the_real_default_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MEMBRIDGE_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # a pretend home directory
    r = runner.invoke(app, ["quickstart", "--demo", "--yes", "--no-serve"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / ".membridge-demo" / "membridge.db").exists() and not (tmp_path / ".membridge").exists()


def test_the_packaged_chatgpt_prompt_is_the_documented_one() -> None:
    doc = (ROOT / "docs" / "CHATGPT_DUMP_PROMPT.md").read_text(encoding="utf-8")
    a = doc.index("```\n", doc.index("## Prompt")) + 4
    assert qs.prompt_text().strip() == doc[a : doc.index("\n```", a)].strip()
    out = runner.invoke(app, ["dump-prompt"]).output
    assert "data-portability request" in out and "END |" in out


def test_the_assistant_connection_command_points_at_this_python() -> None:
    import sys

    assert sys.executable in qs.mcp_command() and "MEMBRIDGE_CLIENT_NAME=claude-code" in qs.mcp_command()
