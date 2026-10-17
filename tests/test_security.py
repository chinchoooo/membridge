"""Security regressions from the architecture review. Every fixture is synthetic."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from membridge.crypto import lock_database
from membridge.memory import access
from membridge.memory.consolidate import upsert
from membridge.memory.models import Candidate, EntityRef
from membridge.memory.seed_text import seed_from_text
from membridge.memory.service import SENSITIVE_NOTICE, MemoryService, ServiceError
from membridge.memory.summaries import refresh_summaries
from membridge.store.graph import GraphStore
from membridge.store.sqlite import Store, VaultLockedError, verify_audit_chain

PRIVATE_TEXT = "I was diagnosed with an anxiety disorder and my therapist adjusts the dosage"
PIN = "246810"
T0 = 1_700_000_000.0


def _message(store: Store, conv: str, n: int, role: str, text: str) -> str:
    from membridge.memory.models import sensitivity_of

    mid = f"msg_{conv}_{n}"
    store.conn.execute(
        "INSERT INTO messages(id, conversation_id, external_id, seq, role, create_time, content_type, text, content_hash, "
        "on_current_branch, import_job, sensitivity) VALUES (?,?,?,?,?,?,?,?,?,1,'j',?)",
        (mid, conv, mid, n, role, T0 + n, "text", text, mid, sensitivity_of(text)),
    )  # fmt: skip
    store.conn.execute("INSERT INTO messages_fts(message_id, text) VALUES (?,?)", (mid, text))
    return mid


@pytest.fixture
def world(store: Store) -> dict[str, str]:
    """A project (Atlas), a normal decision, a private memory attached to the same project and conversation."""
    store.conn.execute(
        "INSERT INTO conversations(id, source, external_id, title, create_time, content_hash, message_count, first_job, last_job) "
        "VALUES ('conv_a','t','a','Atlas planning',?, 'h', 3, 'j','j')", (T0,)
    )  # fmt: skip
    m1 = _message(store, "conv_a", 1, "user", "I decided to use PostgreSQL for Atlas storage.")
    m2 = _message(store, "conv_a", 2, "user", PRIVATE_TEXT + ". My api key is sk-abcdefghijklmnopqrstuvwx for Atlas.")
    m3 = _message(store, "conv_a", 3, "user", "Remind me about the Atlas demo on Friday.")
    ids: dict[str, str] = {"m_private": m2, "m_normal": m1}
    ids["normal"] = upsert(store, Candidate(
        statement="I decided to use PostgreSQL for Atlas storage.", category="decision", kind="episodic", subject="atlas",
        predicate="uses:storage", object="PostgreSQL", entities=[EntityRef("Atlas", "project"), EntityRef("PostgreSQL")], ref=m1,
        quote="I decided to use PostgreSQL for Atlas storage.", ts=T0 + 1, verification="user_stated")).memory_id  # fmt: skip
    ids["private"] = upsert(store, Candidate(
        statement=PRIVATE_TEXT, category="health", entities=[EntityRef("Atlas", "project"), EntityRef("Zoloft")], ref=m2, quote=PRIVATE_TEXT,
        ts=T0 + 2, verification="user_stated", sensitivity="sensitive")).memory_id  # fmt: skip
    ids["other"] = upsert(store, Candidate(
        statement="The Atlas demo is on Friday.", category="constraint", entities=[EntityRef("Atlas", "project")], ref=m3,
        quote="demo on Friday", ts=T0 + 3, verification="user_stated")).memory_id  # fmt: skip
    store.conn.execute("INSERT OR IGNORE INTO memory_links(src_id, rel, dst_id, reason, created_at) VALUES (?,?,?,?,0)",
                       (ids["private"], "RELATED_TO", ids["normal"], "t"))  # fmt: skip
    store.conn.commit()
    refresh_summaries(store)
    return ids


@pytest.fixture
def agent(store: Store, world: dict[str, str]) -> MemoryService:
    return MemoryService(store, actor="claude-code")


@pytest.fixture
def me(store: Store, world: dict[str, str]) -> MemoryService:
    return MemoryService(store, actor="me", human=True)


def _everything(obj: object) -> str:
    """Everything a client would see, except the words it typed itself (the query is echoed back)."""

    def strip(o: object) -> object:
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("query", "will_forget")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o

    return json.dumps(strip(obj), default=str).lower()


# ---- visibility -----------------------------------------------------------------------------------------------------
def test_private_memory_is_indistinguishable_from_a_missing_one(agent: MemoryService, world: dict[str, str]) -> None:
    with pytest.raises(ServiceError) as hidden:
        agent.get_memory(world["private"])
    with pytest.raises(ServiceError) as missing:
        agent.get_memory("mem_" + "0" * 24)
    assert str(hidden.value).replace(world["private"], "X") == str(missing.value).replace("mem_" + "0" * 24, "X")
    for fn in (agent.trace_memory_provenance, agent.get_related_memories):
        with pytest.raises(ServiceError):
            fn(world["private"])


def test_no_read_path_returns_private_text(agent: MemoryService, store: Store, world: dict[str, str]) -> None:
    results = {
        "search_private": agent.search_memory("anxiety disorder therapist dosage"),
        "search_project": agent.search_memory("Atlas storage database", limit=20),
        "context": agent.get_relevant_context("Atlas"),
        "overview": agent.get_overview("everything about Atlas"),
        "related": agent.get_related_memories(world["normal"], 3),
        "get": agent.get_memory(world["normal"]),
        "project": agent.get_project_context("Atlas"),
        "unresolved": agent.list_unresolved(),
        "projects": agent.list_projects(),
        "messages": agent.search_conversations("therapist anxiety dosage"),
        "conversation": agent.get_conversation("conv_a"),
        "provenance": agent.trace_memory_provenance(world["normal"]),
    }
    for name, res in results.items():
        blob = _everything(res)
        for needle in ("anxiety", "therapist", "dosage", "zoloft", "diagnosed"):
            assert needle not in blob, f"{name} leaked {needle!r}"
    assert results["search_private"]["sufficient_evidence"] is False
    assert any(m["text"] == "[withheld: private]" for m in results["conversation"]["messages"])
    assert world["private"] not in _everything(results)
    summaries = " ".join(r[0] for r in store.conn.execute("SELECT text FROM summaries"))
    assert "anxiety" not in summaries.lower() and "therapist" not in summaries.lower()


def test_private_memory_does_not_influence_ranking_or_stats(agent: MemoryService, me: MemoryService, world: dict[str, str]) -> None:
    assert agent.get_memory_statistics()["memories"] == 2 and "memories_private" not in agent.get_memory_statistics()
    assert me.get_memory_statistics()["memories_private"] == 1
    a = agent.search_memory("Atlas")
    assert world["private"] not in [m["id"] for m in a["memories"]]
    assert "private" not in _everything(a["stats"]).replace("private memories", "")


def test_every_result_carries_the_same_constant_notice(agent: MemoryService) -> None:
    a = agent.search_memory("anxiety therapist")
    b = agent.search_memory("pizza topping")
    c = agent.search_memory("Atlas storage")
    assert a["sensitive_notice"] == b["sensitive_notice"] == c["sensitive_notice"] == SENSITIVE_NOTICE
    assert not (a["sensitive_included"] or b["sensitive_included"] or c["sensitive_included"])  # the note cannot answer "do I have X?"


def test_secrets_in_the_raw_archive_are_redacted_on_the_way_out(agent: MemoryService, me: MemoryService, store: Store) -> None:
    _message(store, "conv_a", 4, "user", "The Atlas CI deploys with the key sk-abcdefghijklmnopqrstuvwx and a postgres://root:hunter2hunter2@db/app url")
    store.conn.commit()
    got = [agent.search_conversations("CI deploys key"), me.get_conversation("conv_a"), me.search_conversations("CI deploys key"),
           agent.search_memory("Atlas CI deploys key", limit=20)]  # fmt: skip
    blob = _everything(got)
    assert "sk-abcdefghijklmnopqrstuvwx" not in blob and "hunter2hunter2" not in blob
    assert blob.count("[redacted]") >= 2  # the message was returned, with the credentials blanked out


# ---- consent: PIN + grants -------------------------------------------------------------------------------------------
def test_asking_for_private_memories_without_a_grant_is_refused_and_audited(agent: MemoryService, store: Store) -> None:
    pkg = agent.search_memory("anxiety disorder therapist", include_sensitive=True)
    assert pkg["sufficient_evidence"] is False and pkg["sensitive_included"] is False
    row = store.conn.execute("SELECT action, actor FROM audit_log WHERE action LIKE 'sensitive_%' ORDER BY id DESC LIMIT 1").fetchone()
    assert (row["action"], row["actor"]) == ("sensitive_denied", "claude-code")


def test_agent_cannot_create_its_own_grant_or_pin(agent: MemoryService) -> None:
    for call in (lambda: agent.set_pin("999999"), lambda: agent.grant_sensitive("999999", "claude-code"), agent.list_grants,
                 lambda: agent.revoke_sensitive("claude-code")):  # fmt: skip
        with pytest.raises(ServiceError, match="needs the user"):
            call()


def test_grant_lifecycle_pin_expiry_uses_client_binding_revocation(agent: MemoryService, me: MemoryService, store: Store, world: dict[str, str]) -> None:
    with pytest.raises(ServiceError, match="set-pin"):
        me.grant_sensitive(PIN, "claude-code")
    me.set_pin(PIN)
    with pytest.raises(ServiceError, match="wrong PIN"):
        me.grant_sensitive("not-the-pin", "claude-code")
    assert me.grant_sensitive(PIN, "claude-code", minutes=5, uses=2)["uses"] == 2
    other = MemoryService(store, actor="some-other-client")
    assert other.search_memory("anxiety disorder therapist", include_sensitive=True)["sensitive_included"] is False  # bound to one client
    got = agent.search_memory("anxiety disorder therapist dosage", include_sensitive=True)
    assert got["sensitive_included"] is True and any(world["private"] == m["id"] for m in got["memories"])
    assert agent.get_memory(world["private"], include_sensitive=True)["statement"] == PRIVATE_TEXT  # second use
    with pytest.raises(ServiceError):  # a third read: the grant had two uses
        agent.get_memory(world["private"], include_sensitive=True)
    me.grant_sensitive(PIN, "claude-code", minutes=5, uses=5)
    store.conn.execute("UPDATE sensitive_grants SET expires_at=? WHERE revoked=0 AND uses_left>0", (time.time() - 1,))
    store.conn.commit()
    assert agent.search_memory("anxiety disorder therapist", include_sensitive=True)["sensitive_included"] is False  # expired
    me.grant_sensitive(PIN, "claude-code", minutes=5, uses=5)
    assert me.revoke_sensitive("claude-code")["revoked"] >= 1
    assert agent.search_memory("anxiety disorder therapist", include_sensitive=True)["sensitive_included"] is False
    acts = [r[0] for r in store.conn.execute("SELECT action FROM audit_log WHERE action LIKE 'sensitive_%' OR action='pin_set'")]
    assert "sensitive_grant" in acts and "sensitive_revoke" in acts and "sensitive_access" in acts and "pin_set" in acts


def test_wrong_pins_are_rate_limited(me: MemoryService) -> None:
    me.set_pin(PIN)
    for _ in range(access.LOCKOUT_AFTER):
        with pytest.raises(ServiceError, match="wrong PIN"):
            me.grant_sensitive("nope-nope", "c")
    with pytest.raises(ServiceError, match="too many wrong PINs"):
        me.grant_sensitive(PIN, "c")  # even the right PIN waits


def test_the_pin_is_stored_hashed(me: MemoryService, store: Store) -> None:
    me.set_pin(PIN)
    stored = store.conn.execute("SELECT v FROM settings_kv WHERE k='pin_hash'").fetchone()[0]
    assert PIN not in stored and "$" in stored and len(stored) > 80


# ---- trust cannot be forged by an agent ------------------------------------------------------------------------------
def test_agent_writes_are_recorded_as_client_asserted_not_user_stated(agent: MemoryService, me: MemoryService, store: Store) -> None:
    a = agent.save_memory("The deploy target is Fly.io", "fact", "user_stated")["memory_id"]
    d = agent.record_decision("Atlas", "use SQLite for storage")["memory_id"]
    h = me.save_memory("I prefer tabs", "preference", "user_stated")["memory_id"]
    ver = {r["id"]: r["verification"] for r in store.conn.execute("SELECT id, verification FROM memories WHERE id IN (?,?,?)", (a, d, h))}
    assert ver == {a: "client_asserted", d: "client_asserted", h: "user_stated"}
    with pytest.raises(ServiceError):
        agent.save_memory("I am the user", "fact", "user_verified")


def test_agents_cannot_change_meaning_only_the_user_can(agent: MemoryService, me: MemoryService, world: dict[str, str]) -> None:
    with pytest.raises(ServiceError, match="user's call"):
        agent.link_memories(world["normal"], "SUPERSEDES", world["other"])
    agent.link_memories(world["normal"], "RELATED_TO", world["other"])
    with pytest.raises(ServiceError, match="needs the user"):
        agent.resolve_contradiction(world["normal"], world["other"])
    with pytest.raises(ServiceError, match="needs the user"):
        agent.decide_entity_pair("a", "b", True)
    with pytest.raises(ServiceError, match="user's decisions"):
        agent.update_memory(world["normal"], {"status": "superseded"}, "x")
    me.update_memory(world["normal"], {"confidence": 0.5}, "tune")


def test_an_edit_can_make_a_memory_private_but_never_public_and_cannot_collide(agent: MemoryService, store: Store, world: dict[str, str]) -> None:
    agent.update_memory(world["other"], {"statement": "The Atlas client's salary is paid on Friday"}, "reword")
    assert store.conn.execute("SELECT sensitivity FROM memories WHERE id=?", (world["other"],)).fetchone()[0] == "sensitive"
    assert world["other"] not in [m["id"] for m in agent.search_memory("Atlas salary Friday")["memories"]]
    with pytest.raises(ServiceError):  # now private, so an agent cannot even address it
        agent.update_memory(world["other"], {"statement": "x y z"}, "again")
    agent.save_memory("Atlas uses Redis for caching", "fact", "model_inferred")
    other = agent.save_memory("Atlas uses Valkey for caching", "fact", "model_inferred")["memory_id"]
    with pytest.raises(ServiceError, match="already says this"):  # editing one memory into another would merge them silently
        agent.update_memory(other, {"statement": "Atlas uses Redis for caching"}, "make it the same")
    assert store.conn.execute("SELECT statement FROM memories WHERE id=?", (other,)).fetchone()[0] == "Atlas uses Valkey for caching"


# ---- forgetting is real ------------------------------------------------------------------------------------------------
def test_forget_scrubs_every_copy_of_the_text(me: MemoryService, store: Store, tmp_path: Path) -> None:
    dump = tmp_path / "chatgpt-dump.txt"
    dump.write_text("FACT | skill | Alex uses Haskell for compilers. | memory | high | ? | N\nFACT | skill | Alex uses Go for services. | memory | high | ? | N\n", encoding="utf-8")
    seed_from_text(store, dump)
    note = me.save_memory("Remember the zebra codeword for the demo", "fact", "user_stated")["memory_id"]
    haskell = store.conn.execute("SELECT id FROM memories WHERE statement LIKE '%Haskell%'").fetchone()[0]
    for mid in (note, haskell):
        me.forget_memory(mid, me.forget_memory(mid)["confirm_token"])
    blob = _everything([dict(r) for r in store.conn.execute("SELECT * FROM memories")] +
                       [dict(r) for r in store.conn.execute("SELECT * FROM documents")] +
                       [dict(r) for r in store.conn.execute("SELECT * FROM memory_evidence")] +
                       [dict(r) for r in store.conn.execute("SELECT * FROM memory_revisions")] +
                       [dict(r) for r in store.conn.execute("SELECT * FROM summaries")] +
                       [dict(r) for r in store.conn.execute("SELECT * FROM memories_fts")])  # fmt: skip
    assert "haskell" not in blob and "zebra" not in blob
    assert "go for services" in blob  # unrelated lines of the same dump are untouched
    again = me.save_memory("Remember the zebra codeword for the demo", "fact", "user_stated")
    assert again["result"] == "skipped_deleted"


# ---- contradictions the user settled stay settled ------------------------------------------------------------------------
def test_a_settled_contradiction_does_not_come_back(me: MemoryService, store: Store) -> None:
    base = dict(category="decision", kind="episodic", subject="atlas", predicate="uses:cache", ref="d", ref_kind="document", verification="provider_stated")
    a = upsert(store, Candidate(statement="Atlas caching uses Redis", object="Redis", **base)).memory_id  # type: ignore[arg-type]
    b = upsert(store, Candidate(statement="Atlas caching uses Memcached", object="Memcached", **base)).memory_id  # type: ignore[arg-type]
    assert {r["status"] for r in store.conn.execute("SELECT status FROM memories WHERE id IN (?,?)", (a, b))} == {"unresolved"}
    me.resolve_contradiction(a, b, "redis it is")
    st = {r["id"]: r["status"] for r in store.conn.execute("SELECT id, status FROM memories WHERE id IN (?,?)", (a, b))}
    assert st == {a: "current", b: "superseded"}
    c = upsert(store, Candidate(statement="Atlas caching uses Memcached again", object="Memcached", **base)).memory_id  # type: ignore[arg-type]
    assert store.conn.execute("SELECT status FROM memories WHERE id=?", (c,)).fetchone()[0] == "superseded"  # not unresolved again
    assert store.conn.execute("SELECT status FROM memories WHERE id=?", (a,)).fetchone()[0] == "current"


# ---- the vault and the audit trail --------------------------------------------------------------------------------------
def test_a_locked_vault_is_not_replaced_by_an_empty_database(tmp_path: Path) -> None:
    db = tmp_path / "v" / "membridge.db"
    Store(db).close()
    lock_database(db, "correct horse battery staple")
    assert not db.exists()
    with pytest.raises(VaultLockedError, match="membridge unlock"):
        Store(db)
    assert not db.exists()  # and nothing was created beside the encrypted copy


def test_audit_log_is_a_hash_chain_that_notices_edits_and_removals(store: Store) -> None:
    for i in range(6):
        store.audit_sync("a", "act", f"t{i}", n=i)
    assert verify_audit_chain(store) == {"hashed_rows": 6, "breaks": [], "ok": True}
    store.conn.execute("UPDATE audit_log SET actor='mallory' WHERE target='t2'")
    store.conn.commit()
    assert verify_audit_chain(store)["ok"] is False
    store.conn.execute("UPDATE audit_log SET actor='a' WHERE target='t2'")
    store.conn.execute("DELETE FROM audit_log WHERE target='t4'")
    store.conn.commit()
    out = verify_audit_chain(store)
    assert out["ok"] is False and out["breaks"]


# ---- derived structures --------------------------------------------------------------------------------------------------
def test_derived_edges_and_cooccurrence_ignore_private_memories(store: Store, world: dict[str, str]) -> None:
    from membridge.resolve.relations import rebuild_entity_edges

    rebuild_entity_edges(store)
    names = {r[0] for r in store.conn.execute(
        "SELECT ea.name FROM entity_cooccurrence c JOIN entities ea ON ea.id=c.a_entity UNION "
        "SELECT eb.name FROM entity_cooccurrence c JOIN entities eb ON eb.id=c.b_entity")}  # fmt: skip
    assert "Zoloft" not in names  # the only memory linking Zoloft to Atlas is private


def test_graph_projection_never_contains_private_memories(store: Store, world: dict[str, str], graph: GraphStore) -> None:
    graph.rebuild(store)
    rows = graph.query("MATCH (m:Memory) RETURN m.id, m.statement")
    assert rows and world["private"] not in [r[0] for r in rows] and not any("anxiety" in str(r[1]).lower() for r in rows)
    assert graph.query("MATCH (m:Memory {id:$i}) RETURN count(m)", {"i": world["private"]})[0][0] == 0
    edges = graph.query("MATCH (a)-[r]->(b) RETURN a.id, b.id")
    assert not any(world["private"] in (e[0], e[1]) for e in edges)
    assert not any("anxiety" in str(r).lower() for r in graph.query("MATCH (g:Message) RETURN g.snippet"))


def test_the_user_can_mark_a_memory_private_when_the_detector_missed_it(agent: MemoryService, me: MemoryService, store: Store, world: dict[str, str]) -> None:
    mid = me.save_memory("My grandmother's maiden name is Okonkwo", "fact", "user_stated")["memory_id"]
    assert agent.get_memory(mid)["statement"].startswith("My grandmother")  # the lexicon did not catch it
    with pytest.raises(ServiceError, match="needs the user"):
        agent.set_private(mid)
    me.set_private(mid)
    with pytest.raises(ServiceError, match="no memory with id"):
        agent.get_memory(mid)
    assert mid not in [m["id"] for m in agent.search_memory("grandmother maiden name Okonkwo")["memories"]]
    assert "okonkwo" not in " ".join(r[0] for r in store.conn.execute("SELECT text FROM summaries")).lower()
    me.set_private(mid, False)
    assert agent.get_memory(mid)["statement"].startswith("My grandmother")
    assert store.conn.execute("SELECT COUNT(*) FROM audit_log WHERE action IN ('set_private','set_public')").fetchone()[0] == 2
