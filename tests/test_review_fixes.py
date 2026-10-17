"""Regressions for the second (independent) review. All fixtures are synthetic."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from membridge.llm.base import LLM
from membridge.memory.consolidate import _same_value, upsert
from membridge.memory.llm_tasks import extract_with_llm
from membridge.memory.models import Candidate, EntityRef, contains_secret, safe_title
from membridge.memory.pipeline import purge_claude_seeds
from membridge.memory.rules import extract_from_message
from membridge.memory.seed_text import seed_from_text
from membridge.memory.service import MemoryService, ServiceError
from membridge.store.graph import GraphStore
from membridge.store.migrations import MIGRATIONS
from membridge.store.sqlite import Store

T0 = 1_700_000_000.0
DAY = 86400.0
PRIVATE = "I take sertraline 50 mg every morning for my anxiety"


def _msg(store: Store, conv: str, n: int, text: str, sens: str = "normal") -> str:
    mid = f"m_{conv}_{n}"
    store.conn.execute(
        "INSERT INTO messages(id, conversation_id, external_id, seq, role, create_time, content_type, text, content_hash, on_current_branch, "
        "import_job, sensitivity) VALUES (?,?,?,?,?,?,?,?,?,1,'j',?)", (mid, conv, mid, n, "user", T0 + n, "text", text, mid, sens))  # fmt: skip
    store.conn.execute("INSERT INTO messages_fts(message_id, text) VALUES (?,?)", (mid, text))
    return mid


def _conv(store: Store, cid: str, title: str) -> None:
    store.conn.execute(
        "INSERT INTO conversations(id, source, external_id, title, create_time, content_hash, message_count, first_job, last_job) "
        "VALUES (?,'t',?,?,?, 'h', 2, 'j','j')", (cid, cid, title, T0))  # fmt: skip


@pytest.fixture
def world(store: Store) -> dict[str, str]:
    _conv(store, "c1", "Atlas planning")
    _conv(store, "c2", "Sertraline side effects and my therapist")
    m1 = _msg(store, "c1", 1, "I decided to use PostgreSQL for Atlas storage.")
    m2 = _msg(store, "c2", 1, PRIVATE + ". Also the Atlas demo is on Friday. token ghp_abcdefghijklmnopqrstuvwxyz0123", "sensitive")
    ids = {"m1": m1, "m2": m2}
    ids["normal"] = upsert(store, Candidate(statement="I decided to use PostgreSQL for Atlas storage.", category="decision", kind="episodic", subject="atlas",
                                            predicate="uses:storage", object="PostgreSQL", entities=[EntityRef("Atlas", "project")], ref=m1, ts=T0)).memory_id  # fmt: skip
    ids["private"] = upsert(store, Candidate(statement=PRIVATE, category="health", entities=[EntityRef("Sertraline"), EntityRef("Atlas", "project")],
                                             ref=m2, ts=T0 + 5, sensitivity="sensitive")).memory_id  # fmt: skip
    # a NORMAL memory that happens to be supported by the private message (the leak path for graph snippets / titles)
    ids["demo"] = upsert(store, Candidate(statement="The Atlas demo is on Friday.", category="constraint", entities=[EntityRef("Atlas", "project")],
                                          ref=m2, quote="the Atlas demo is on Friday", ts=T0 + 6)).memory_id  # fmt: skip
    store.conn.commit()
    return ids


@pytest.fixture
def agent(store: Store, world: dict[str, str]) -> MemoryService:
    return MemoryService(store, actor="claude-code")


@pytest.fixture
def me(store: Store, world: dict[str, str]) -> MemoryService:
    return MemoryService(store, actor="me", human=True)


def blob(obj: object) -> str:
    import json

    return json.dumps(obj, default=str).lower()


# ---- private-only entities and conversation titles --------------------------------------------------------------------------
def test_an_entity_that_exists_only_through_private_memories_is_invisible(agent: MemoryService, store: Store, world: dict[str, str]) -> None:
    visible = {r[0] for r in store.conn.execute("SELECT name FROM visible_entities")}
    assert "Sertraline" not in visible and "Atlas" in visible
    for res in (agent.get_overview("everything"), agent.get_project_context("Atlas"), agent.list_projects(), agent.get_memory(world["normal"]),
                agent.search_memory("Atlas storage", limit=20)):  # fmt: skip
        assert "sertraline" not in blob(res)
    assert agent.get_project_context("Sertraline")["found"] is False == agent.get_project_context("Prozac")["found"]  # same answer as unknown
    assert agent.get_memory_statistics()["entities"] == store.conn.execute("SELECT COUNT(*) FROM visible_entities").fetchone()[0]


def test_a_private_conversation_title_is_never_shown(agent: MemoryService, store: Store, world: dict[str, str]) -> None:
    assert safe_title("Sertraline side effects and my therapist") == "[private conversation]"
    res = [agent.get_conversation("c2"), agent.search_conversations("Atlas demo Friday"), agent.trace_memory_provenance(world["demo"]),
           agent.search_memory("Atlas demo Friday")]  # fmt: skip
    assert "sertraline" not in blob(res) and "therapist" not in blob(res)
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123" not in blob(res)


def test_graph_never_gets_private_snippets_titles_or_entities(store: Store, world: dict[str, str], graph: GraphStore) -> None:
    graph.rebuild(store)
    assert not any("sertraline" in str(r).lower() or "therapist" in str(r).lower() for r in graph.query("MATCH (n) RETURN properties(n)"))
    assert not any("ghp_" in str(r) for r in graph.query("MATCH (g:Message) RETURN g.snippet"))
    assert graph.query("MATCH (e:Entity {name:'Sertraline'}) RETURN count(e)")[0][0] == 0
    assert graph.query("MATCH (e:Entity {name:'Atlas'}) RETURN count(e)")[0][0] == 1


# ---- erase / make private is complete ---------------------------------------------------------------------------------------
def test_forget_and_set_private_leave_no_trace_in_summaries_entities_index_or_file(me: MemoryService, agent: MemoryService, store: Store) -> None:
    secret = me.save_memory("The vault codeword is zephyrquartz", "fact", "user_stated")["memory_id"]
    me.save_memory("Quartzmaster is a person I work with on Atlas", "fact", "user_stated")
    store.conn.execute("INSERT OR REPLACE INTO summaries(id, level, ref, text, source_hash, method, updated_at, child_hash) VALUES "
                       "('sum_community-L1_x','community-L1','x','Theme: zephyrquartz codeword','h','llm',0,'h')")  # a polished summary
    store.conn.commit()
    me.set_private(secret)
    assert "zephyrquartz" not in " ".join(r[0] for r in store.conn.execute("SELECT text FROM summaries")).lower()
    assert "zephyrquartz" not in blob(agent.get_overview("codeword"))
    me.set_private(secret, False)
    me.forget_memory(secret, me.forget_memory(secret)["confirm_token"])
    store.conn.commit()
    store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = Path(store.path).read_bytes().lower()
    assert b"zephyrquartz" not in raw, "the forgotten text is still in the database file (free pages / FTS segments / WAL)"


def test_forgetting_removes_the_entity_it_alone_created(me: MemoryService, store: Store) -> None:
    mid = me.save_memory("My private codename is Zorblax", "fact", "user_stated")["memory_id"]
    assert store.conn.execute("SELECT COUNT(*) FROM entities WHERE canonical='zorblax'").fetchone()[0] == 1
    me.forget_memory(mid, me.forget_memory(mid)["confirm_token"])
    assert store.conn.execute("SELECT COUNT(*) FROM entities WHERE canonical='zorblax'").fetchone()[0] == 0


# ---- trust: edits, dumps, oracles, poisoning -----------------------------------------------------------------------------------
def test_an_agent_that_rewrites_a_memory_does_not_inherit_the_users_trust(agent: MemoryService, me: MemoryService, store: Store) -> None:
    mid = me.save_memory("The Atlas database is PostgreSQL", "fact", "user_stated")["memory_id"]
    assert store.conn.execute("SELECT verification FROM memories WHERE id=?", (mid,)).fetchone()[0] == "user_stated"
    agent.update_memory(mid, {"statement": "The Atlas database is MongoDB and never PostgreSQL"}, "fix")
    assert store.conn.execute("SELECT verification FROM memories WHERE id=?", (mid,)).fetchone()[0] == "client_asserted"


def test_a_dump_change_line_cannot_overrule_what_the_user_stated(me: MemoryService, store: Store, tmp_path: Path) -> None:
    user = me.save_memory("Atlas uses Postgres", "fact", "user_stated")["memory_id"]
    dump = tmp_path / "d.txt"
    dump.write_text("CHANGE | Atlas | uses Postgres -> uses MongoDB | ? | memory | high\n", encoding="utf-8")
    seed_from_text(store, dump)
    st = {r["statement"]: r["status"] for r in store.conn.execute("SELECT statement, status FROM memories")}
    assert store.conn.execute("SELECT status FROM memories WHERE id=?", (user,)).fetchone()[0] == "current"
    assert st["Atlas: uses MongoDB"] == "superseded"  # the provider's newer claim is outranked, not the user's statement


def test_saving_text_that_matches_a_private_memory_reveals_nothing(agent: MemoryService, me: MemoryService, world: dict[str, str]) -> None:
    res = agent.save_memory(PRIVATE, "health", "model_inferred")
    assert "memory_id" not in res and res["result"] == "saved"
    fresh = agent.save_memory("My salary is paid on the first of the month", "fact", "model_inferred")
    assert fresh == res | {} or ("memory_id" not in fresh and fresh["result"] == "saved")  # same shape for new and existing private text
    a = agent.save_memory("I take a daily walk for fitness", "health", "model_inferred")["memory_id"]
    with pytest.raises(ServiceError) as e:
        agent.update_memory(a, {"statement": PRIVATE}, "dup")
    assert world["private"] not in str(e.value)


def test_an_agent_cannot_take_the_users_statement_out_of_circulation(agent: MemoryService, store: Store) -> None:
    user = upsert(store, Candidate(statement="Atlas caching uses Redis", category="decision", subject="atlas", predicate="uses:cache", object="Redis",
                                   ref="m", ts=T0, verification="user_stated")).memory_id  # fmt: skip
    for i in range(6):
        agent.record_decision("Atlas", f"use Backend{i} for cache")
    assert store.conn.execute("SELECT status FROM memories WHERE id=?", (user,)).fetchone()[0] == "current"
    top = agent.search_memory("Atlas cache")["memories"][0]
    assert top["id"] == user


# ---- extraction false positives -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "Write a cover letter that says I work at Google.",
    "Translate to French: I live in Berlin",
    "My friend said: I work at Initech",
    'He told me "I use Rust for everything"',
    "Draft an email saying I am building Atlas",
])  # fmt: skip
def test_instructions_and_reported_speech_are_not_the_users_own_statements(text: str) -> None:
    assert extract_from_message(text, message_id="m", ts=1.0, context_entity=None, known=set())[0] == []


def test_real_first_person_statements_still_pass() -> None:
    for t in ("I work at Acme Corp.", "We are building Atlas.", "I use Python for data analysis."):
        assert extract_from_message(t, message_id="m", ts=1.0, context_entity=None, known=set())[0], t


# ---- values and aliases ---------------------------------------------------------------------------------------------------------
def test_aliases_are_one_value_but_neighbouring_names_are_not(store: Store) -> None:
    assert _same_value("postgres", "postgresql", store) and _same_value("vscode", "visual studio code", store)
    assert _same_value("acme", "acme corp", store) and _same_value("northfield", "northfield university", store)
    assert not _same_value("visual studio", "visual studio code", store) and not _same_value("postgres", "mysql", store)


def test_record_decision_with_an_alias_does_not_supersede_the_same_choice(agent: MemoryService, store: Store) -> None:
    first = upsert(store, Candidate(statement="I decided to use PostgreSQL for Atlas storage.", category="decision", subject="atlas", predicate="uses:storage",
                                    object="PostgreSQL", ref="m", ts=T0, verification="user_stated")).memory_id  # fmt: skip
    agent.record_decision("Atlas", "use Postgres for storage")
    assert store.conn.execute("SELECT status FROM memories WHERE id=?", (first,)).fetchone()[0] == "current"


def test_a_contradiction_can_only_be_settled_between_memories_in_conflict(me: MemoryService, store: Store) -> None:
    a = me.save_memory("Alpha fact one", "fact", "user_stated")["memory_id"]
    b = me.save_memory("Beta fact two", "fact", "user_stated")["memory_id"]
    store.conn.execute("UPDATE memories SET status='unresolved' WHERE id IN (?,?)", (a, b))
    store.conn.commit()
    with pytest.raises(ServiceError, match="not recorded as being in conflict"):
        me.resolve_contradiction(a, b)


# ---- credentials ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "my passcode is 482913xx", "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "my 2FA backup code is 84729-11823",
    "client_secret: abcd1234efgh5678", "recovery code is ABCD-EFGH-1234",
])  # fmt: skip
def test_more_credential_shapes_are_recognised(text: str) -> None:
    assert contains_secret(text), text


# ---- LLM extraction ------------------------------------------------------------------------------------------------------------------
class _Remote:
    is_remote = True
    name = "fake-remote:1"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def complete_json(self, system: str, user: str, schema: dict, *, max_tokens: int = 1200) -> dict:  # type: ignore[type-arg]
        self.seen.append(user)
        return {"memories": []}


def test_private_raw_messages_never_go_to_a_remote_model(store: Store, world: dict[str, str]) -> None:
    llm: LLM = _Remote()  # type: ignore[assignment]
    extract_with_llm(store, llm)
    assert llm.seen and not any("sertraline" in u.lower() for u in llm.seen)  # type: ignore[attr-defined]
    assert any("PostgreSQL" in u for u in llm.seen)  # type: ignore[attr-defined]


# ---- migration, purge ---------------------------------------------------------------------------------------------------------------
def test_migration_8_downgrades_agent_forged_v01_rows_but_keeps_the_dashboards_own(tmp_path: Path) -> None:
    p = tmp_path / "old.db"
    s = Store(p)
    for i, (method, ver) in enumerate([("client:claude-code", "user_stated"), ("client:dashboard", "user_stated"), ("rule", "user_stated")]):
        upsert(s, Candidate(statement=f"Row number {i} says something", category="fact", ref="r", verification=ver, origin="client" if method.startswith("client") else "import",
                            method=method))  # fmt: skip
    s.conn.execute("DELETE FROM schema_version WHERE version=8")
    s.conn.execute("UPDATE memories SET verification='user_stated'")
    s.conn.commit()
    MIGRATIONS[-1][2](s.conn)
    got = {r["extraction_method"]: r["verification"] for r in s.conn.execute("SELECT extraction_method, verification FROM memories")}
    assert got == {"client:claude-code": "client_asserted", "client:dashboard": "user_stated", "rule": "user_stated"}
    s.close()
    sqlite3.connect(p).close()


def test_purging_seeds_does_not_leave_the_conflict_partner_unresolved_for_ever(store: Store) -> None:
    base = dict(category="fact", subject="user", predicate="work_at", ref="d", ref_kind="document", verification="provider_stated")
    a = upsert(store, Candidate(statement="Alex works at Acme", object="Acme", origin="claude_memory", **base)).memory_id  # type: ignore[arg-type]
    b = upsert(store, Candidate(statement="Alex works at Globex", object="Globex", origin="import", **base)).memory_id  # type: ignore[arg-type]
    assert store.conn.execute("SELECT status FROM memories WHERE id=?", (b,)).fetchone()[0] == "unresolved"
    store.conn.execute("INSERT INTO documents(id, kind, name, text, content_hash) VALUES ('d','claude_memory','n','t','h')")
    store.conn.commit()
    purge_claude_seeds(store)
    assert store.conn.execute("SELECT 1 FROM memories WHERE id=?", (a,)).fetchone() is None
    assert store.conn.execute("SELECT status FROM memories WHERE id=?", (b,)).fetchone()[0] == "current"
