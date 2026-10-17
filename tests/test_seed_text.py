"""ChatGPT dump importer. All fixtures are synthetic: no real person's data belongs in this repository."""

from __future__ import annotations

from pathlib import Path

from membridge.memory.seed_text import (
    derive_triple,
    first_person,
    infer_user_subject,
    parse_dump,
    parse_when,
    seed_from_text,
)
from membridge.store.sqlite import Store

STRICT = """\
FACT | education | Alex Rivera studies Computer Science at Northfield University. | memory | high | 2026-05 | N
FACT | health | Alex Rivera reports taking a daily antidepressant for anxiety. | memory | high | ? | Y
FACT | preference | Alex Rivera prefers C++ over Rust because Alex does not know Rust. | inferred | medium | ? | N
FACT | other | Alex Rivera key is sk-abcdefghijklmnopqrstuvwx | memory | high | ? | N
CHANGE | GitHub username | alex-old -> alex-new | 2026-08 | memory | high
CHANGE | Weight | 72 kg -> 73 kg -> 87 kg | ? | memory | high
CHANGE | Editor | VS Code -> Neovim -> VS Code | ? | memory | high
ENTITY | Northfield University | organization | Where Alex studies
EPISODE | 2026-10-16 | Hackathon planning | Asked for a portable memory engine hosted on the laptop.
END | facts=4 | changes=3 | entities=1 | episodes=1
"""

LEGACY = """\
## B. CUSTOM INSTRUCTIONS
B1. Other Information. Verbatim: "Other Information: I am a student" [source: custom-instructions] [confidence: high]
B2. User profile: occupation. Verbatim: "I am a student" [source: custom-instructions] [confidence: high]
B5. Other fields: not available to me. [source: custom-instructions] [confidence: high]
### GOALS & PLANS
C171. You want research experience. [source: memory] [confidence: high]
C344. The exact identity of your parent is not available to me. [source: memory] [confidence: high]
C130. You have asked for a comprehensive review of information held about you. [source: recalled-chat] [confidence: high]
### PEOPLE, ORGANIZATIONS, PLACES
C301. Northfield University. [source: memory] [confidence: high]
### FACTS THAT CHANGED OVER TIME (OLD -> NEW)
C352. GitHub username: alex-old -> alex-new. Exact change date: not available to me. [source: memory] [confidence: high]
## D. CHAT HISTORY RECALL
D44. Agricultural Modeling | approx. 2026. You compared crop models with and without rainfall.
D1. Data portability and complete memory dump | 2026-10-17. You asked for a dump.
"""


def _rows(store: Store, sql: str, *a: object) -> list[dict[str, object]]:
    return [dict(r) for r in store.conn.execute(sql, a).fetchall()]


def _status(store: Store, statement: str) -> str:
    return str(_rows(store, "SELECT status FROM memories WHERE statement=?", statement)[0]["status"])


def test_parse_when() -> None:
    assert parse_when("2026-10-17") and parse_when("approx. 2026-10") and parse_when("2026")
    assert parse_when("?") is None and parse_when(None) is None and parse_when("recently") is None


def test_parse_strict_and_legacy() -> None:
    s = parse_dump(STRICT)
    assert [i.kind for i in s].count("fact") == 4 and sum(i.kind == "change" for i in s) == 3
    assert next(i for i in s if i.chain == ["72 kg", "73 kg", "87 kg"])
    legacy = parse_dump(LEGACY)
    kinds = [(i.kind, i.text) for i in legacy]
    assert ("fact", "Custom instruction: I am a student") in kinds  # B1 duplicate and B5 limitation are dropped
    assert not any("Other Information" in t for _, t in kinds)
    assert not any("parent" in t or "comprehensive review" in t for _, t in kinds)  # unavailable + meta lines skipped
    assert next(i for i in legacy if i.kind == "change").chain == ["alex-old", "alex-new"]  # date note stripped
    assert sum(i.kind == "episode" for i in legacy) == 1  # D1 is about the dump itself


def test_sensitive_flag_accepts_label_form() -> None:
    line = (
        "FACT | identity | The user is Alex. | memory | high | ? | sensitive: Y\n"
        "FACT | identity | The user studies. | memory | high | ? | sensitive: N\n"
    )
    a, b = parse_dump(line)
    assert a.sensitive and not b.sensitive
    assert not parse_dump("FACT | other | This export is compiled from the memory summary. | memory | high | ? | N")


def test_first_person_and_triples() -> None:
    assert first_person("The user prefers Vim over Emacs.", None) == "I prefer Vim over Emacs."
    assert first_person("Alex works at Acme Corp.", "Alex") == "I work at Acme Corp."
    assert first_person("The user's editor is Vim.", None) == "My editor is Vim."
    assert first_person("Northfield offers courses.", "Alex") is None
    t = derive_triple("Alex works at Acme Corp.", "Alex")
    assert t is not None and t.predicate == "work_at" and t.object == "Acme Corp"
    items = parse_dump("\n".join(f"FACT | identity | Alex does thing {i}. | memory | high | ? | N" for i in range(3))
                       .replace("does", "uses"))  # fmt: skip
    assert infer_user_subject(items) == "Alex"


def test_import_flags_sensitive_and_never_user_stated(tmp_path: Path, store: Store) -> None:
    f = tmp_path / "d.txt"
    f.write_text(STRICT, encoding="utf-8")
    counts = seed_from_text(store, f)
    assert counts["secrets_dropped"] == 1 and counts["entities"] == 1 and counts["episodes"] == 1
    mem = _rows(store, "SELECT statement, verification, sensitivity, origin, confidence, valid_time_known FROM memories")
    assert all(m["verification"] in {"provider_stated", "model_inferred"} for m in mem)
    assert all(m["origin"] == "chatgpt_memory" for m in mem)
    assert not any("sk-abcdef" in str(m["statement"]) for m in mem)
    assert next(m for m in mem if "antidepressant" in str(m["statement"]))["sensitivity"] == "sensitive"
    inferred = next(m for m in mem if "C++" in str(m["statement"]))
    assert inferred["verification"] == "model_inferred" and float(inferred["confidence"]) <= 0.5  # type: ignore[arg-type]
    dated = next(m for m in mem if "Computer Science" in str(m["statement"]))
    assert dated["valid_time_known"] == 1 and inferred["valid_time_known"] == 0  # only a dated claim has a time


def test_change_chain_supersedes_and_keeps_history(tmp_path: Path, store: Store) -> None:
    f = tmp_path / "d.txt"
    f.write_text(STRICT, encoding="utf-8")
    seed_from_text(store, f)
    assert [_status(store, f"Weight: {v}") for v in ("72 kg", "73 kg", "87 kg")] == ["superseded", "superseded", "current"]
    assert _status(store, "GitHub username: alex-old") == "superseded" and _status(store, "GitHub username: alex-new") == "current"
    assert _rows(store, "SELECT sensitivity FROM memories WHERE statement='Weight: 87 kg'")[0]["sensitivity"] == "sensitive"


def test_returning_value_is_current_again(tmp_path: Path, store: Store) -> None:
    """VS Code -> Neovim -> VS Code must leave exactly one current value (the old code left none)."""
    f = tmp_path / "d.txt"
    f.write_text(STRICT, encoding="utf-8")
    seed_from_text(store, f)
    assert _status(store, "Editor: VS Code") == "current" and _status(store, "Editor: Neovim") == "superseded"
    assert _rows(store, "SELECT COUNT(*) AS n FROM memories WHERE statement LIKE 'Editor:%' AND status='current'")[0]["n"] == 1


def test_import_idempotent(tmp_path: Path, store: Store) -> None:
    f = tmp_path / "d.txt"
    f.write_text(STRICT + LEGACY, encoding="utf-8")
    seed_from_text(store, f)
    sql = "SELECT (SELECT COUNT(*) FROM memories) AS m, (SELECT COUNT(*) FROM memory_links) AS l, (SELECT COUNT(*) FROM memory_revisions) AS r"
    before = _rows(store, sql)[0]
    seed_from_text(store, f)
    assert _rows(store, sql)[0] == before
    assert _status(store, "Weight: 87 kg") == "current" and _status(store, "Editor: VS Code") == "current"


def test_second_dump_with_same_name_keeps_first_documents_evidence(tmp_path: Path, store: Store) -> None:
    a, b = tmp_path / "a" / "chatgpt-dump.txt", tmp_path / "b" / "chatgpt-dump.txt"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text("FACT | skill | Alex uses Python for data work. | memory | high | ? | N\n", encoding="utf-8")
    b.write_text("FACT | skill | Alex uses Go for services. | memory | high | ? | N\n", encoding="utf-8")
    seed_from_text(store, a)
    seed_from_text(store, b)
    doc_ids = {r["ref"] for r in _rows(store, "SELECT ref FROM memory_evidence")}
    assert len(doc_ids) == 2  # the first dump's text was not overwritten by the second
    for d in doc_ids:
        assert _rows(store, "SELECT text FROM documents WHERE id=?", d)[0]["text"]


def test_dry_run_writes_nothing(tmp_path: Path, store: Store) -> None:
    f = tmp_path / "d.txt"
    f.write_text(STRICT, encoding="utf-8")
    c = seed_from_text(store, f, dry_run=True)
    assert c["facts"] == 4 and c["changes"] == 3
    assert _rows(store, "SELECT COUNT(*) AS n FROM memories")[0]["n"] == 0


def test_restructure_upgrades_older_plain_facts_and_finds_the_conflict(tmp_path: Path, store: Store) -> None:
    from membridge.memory.consolidate import upsert as raw_upsert
    from membridge.memory.models import Candidate
    from membridge.memory.seed_text import restructure_existing

    def plain(text: str, ts: float | None = None) -> None:  # how earlier imports stored facts: no subject/predicate/object
        raw_upsert(store, Candidate(statement=text, category="profile", ref="d", ref_kind="document", origin="chatgpt_memory",
                                    verification="provider_stated", ts=ts))  # fmt: skip

    for t in ("Alex works at Acme Robotics.", "Alex works at Globex Labs.", "Alex studies at Northfield University.",
              "Alex uses Python for data work.", "Alex uses Go for services."):  # fmt: skip
        plain(t)
    dry = restructure_existing(store, dry_run=True)
    assert dry["structured"] >= 3 and store.conn.execute("SELECT COUNT(*) FROM memories WHERE predicate IS NOT NULL").fetchone()[0] == 0
    out = restructure_existing(store)
    assert out["structured"] >= 3 and out["conflicts_found"] >= 1
    states = {r["statement"]: r["status"] for r in _rows(store, "SELECT statement, status FROM memories")}
    assert states["Alex works at Acme Robotics."] == "unresolved" and states["Alex works at Globex Labs."] == "unresolved"  # undated + equal trust
    assert states["Alex uses Python for data work."] == "current" and states["Alex uses Go for services."] == "current"  # many-valued
    again = restructure_existing(store)
    assert again["structured"] == 0 and again["conflicts_found"] == 0  # idempotent: nothing left to upgrade
