"""Trust-aware conflict handling, found by the architecture review (all fixtures synthetic)."""

from __future__ import annotations

from membridge.memory.consolidate import single_valued, upsert
from membridge.memory.models import Candidate, redact_secrets, sensitivity_of
from membridge.memory.models import contains_secret as secret
from membridge.memory.pipeline import purge_claude_seeds
from membridge.store.sqlite import Store

DAY = 86400.0
T0 = 1_700_000_000.0


def cand(statement: str, obj: str, *, ver: str = "user_stated", ts: float | None = T0, pred: str = "uses:cache", subj: str = "atlas",
         cat: str = "decision", origin: str = "import", **kw: object) -> Candidate:  # fmt: skip
    return Candidate(statement=statement, category=cat, kind="episodic", subject=subj, predicate=pred, object=obj, ref=f"r-{statement}",
                     ref_kind="document", quote=statement, ts=ts, verification=ver, origin=origin, **kw)  # type: ignore[arg-type]


def status(store: Store, mid: str) -> str:
    return str(store.conn.execute("SELECT status FROM memories WHERE id=?", (mid,)).fetchone()["status"])


def row(store: Store, mid: str) -> dict[str, object]:
    return dict(store.conn.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone())


def test_undated_provider_fact_never_overrules_dated_user_statement(store: Store) -> None:
    redis = upsert(store, cand("Atlas caching uses Redis", "Redis", ts=T0)).memory_id
    memc = upsert(store, cand("Atlas caching uses Memcached", "Memcached", ver="provider_stated", ts=None)).memory_id
    assert status(store, redis) == "current" and status(store, memc) == "superseded"
    # ...and a later dated statement from the user still moves things forward, with the provider fact staying history
    valkey = upsert(store, cand("Atlas caching uses Valkey", "Valkey", ts=T0 + 30 * DAY)).memory_id
    assert status(store, valkey) == "current" and status(store, redis) == "superseded" and status(store, memc) == "superseded"
    assert row(store, memc)["invalid_at"] is not None  # history has an end: as_of never reads it as 'valid forever'


def test_higher_trust_wins_whatever_the_date(store: Store) -> None:
    user = upsert(store, cand("Atlas caching uses Redis", "Redis", ts=T0)).memory_id
    prov = upsert(store, cand("Atlas caching uses Memcached", "Memcached", ver="provider_stated", ts=T0 + 100 * DAY)).memory_id
    assert status(store, user) == "current" and status(store, prov) == "superseded"


def test_newer_agent_claim_against_user_statement_is_left_for_the_user(store: Store) -> None:
    user = upsert(store, cand("Atlas caching uses Redis", "Redis", ts=T0)).memory_id
    agent = upsert(store, cand("Atlas caching uses Valkey", "Valkey", ver="client_asserted", ts=T0 + DAY)).memory_id
    assert status(store, user) == "unresolved" and status(store, agent) == "unresolved"
    assert store.conn.execute("SELECT COUNT(*) FROM memory_links WHERE rel='CONTRADICTS'").fetchone()[0] == 1


def test_equal_trust_undated_conflict_is_unresolved_not_guessed(store: Store) -> None:
    a = upsert(store, cand("Atlas caching uses Redis", "Redis", ver="provider_stated", ts=None)).memory_id
    b = upsert(store, cand("Atlas caching uses Memcached", "Memcached", ver="provider_stated", ts=None)).memory_id
    assert status(store, a) == "unresolved" and status(store, b) == "unresolved"


def test_a_b_a_leaves_exactly_one_current_value(store: Store) -> None:
    pg1 = upsert(store, cand("Atlas storage uses Postgres", "Postgres", pred="uses:storage", ts=T0)).memory_id
    my = upsert(store, cand("Atlas storage uses MySQL", "MySQL", pred="uses:storage", ts=T0 + DAY)).memory_id
    assert status(store, pg1) == "superseded" and status(store, my) == "current"
    pg2 = upsert(store, cand("Atlas storage uses Postgres", "Postgres", pred="uses:storage", ts=T0 + 2 * DAY)).memory_id
    assert pg2 == pg1
    assert status(store, pg1) == "current" and status(store, my) == "superseded"
    cur = store.conn.execute("SELECT COUNT(*) FROM memories WHERE predicate='uses:storage' AND status='current'").fetchone()[0]
    assert cur == 1


def test_restating_an_old_value_from_the_same_old_source_does_not_reinstate_it(store: Store) -> None:
    pg = upsert(store, cand("Atlas storage uses Postgres", "Postgres", pred="uses:storage", ts=T0)).memory_id
    upsert(store, cand("Atlas storage uses MySQL", "MySQL", pred="uses:storage", ts=T0 + DAY))
    upsert(store, cand("Atlas storage uses Postgres", "Postgres", pred="uses:storage", ts=T0))  # same old message again
    assert status(store, pg) == "superseded"


def test_restating_never_reinstates_against_a_more_trusted_value(store: Store) -> None:
    pg = upsert(store, cand("Atlas storage uses Postgres", "Postgres", pred="uses:storage", ver="provider_stated", ts=T0)).memory_id
    my = upsert(store, cand("Atlas storage uses MySQL", "MySQL", pred="uses:storage", ts=T0 + DAY)).memory_id
    upsert(store, cand("Atlas storage uses Postgres", "Postgres", pred="uses:storage", ver="provider_stated", ts=T0 + 9 * DAY))
    assert status(store, my) == "current" and status(store, pg) == "superseded"


def test_confirmation_merges_trust_sensitivity_and_time(store: Store) -> None:
    first = upsert(store, Candidate(statement="Alex uses an insulin pump", category="health", ref="d1", ref_kind="document",
                                    verification="provider_stated", ts=None, confidence=0.5)).memory_id  # fmt: skip
    again = upsert(store, Candidate(statement="Alex uses an insulin pump", category="health", ref="m1", ref_kind="message",
                                    verification="user_stated", sensitivity="sensitive", ts=T0, confidence=0.9)).memory_id  # fmt: skip
    assert first == again
    r = row(store, first)
    assert r["verification"] == "user_stated" and r["sensitivity"] == "sensitive" and r["valid_time_known"] == 1
    assert r["confidence"] == 0.9 and r["valid_at"] == T0


def test_two_facts_in_one_sentence_stay_two_memories(store: Store) -> None:
    s = "My favourite editor is Vim and I work at Acme Corp"
    a = upsert(store, cand(s, "Vim", pred="editor_is", subj="user", cat="preference")).memory_id
    b = upsert(store, cand(s, "Acme Corp", pred="work_at", subj="user", cat="profile")).memory_id
    assert a != b
    assert store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2


def test_single_valued_user_attribute_supersedes_but_alias_does_not_conflict(store: Store) -> None:
    assert single_valued("work_at") and single_valued("favourite_editor_is") and not single_valued("uses_tool")
    a = upsert(store, cand("I work at Acme", "Acme", pred="work_at", subj="user", cat="profile", ts=T0)).memory_id
    same = upsert(store, cand("I work at Acme Corp", "Acme Corp", pred="work_at", subj="user", cat="profile", ts=T0 + DAY)).memory_id
    assert status(store, a) == "current" and status(store, same) == "current"  # one employer, two spellings
    b = upsert(store, cand("I work at Globex", "Globex", pred="work_at", subj="user", cat="profile", ts=T0 + 2 * DAY)).memory_id
    assert status(store, b) == "current" and status(store, a) == "superseded" and status(store, same) == "superseded"


def test_many_valued_skills_never_conflict(store: Store) -> None:
    a = upsert(store, cand("I use Python for data", "Python", pred="uses_tool", subj="user", cat="skill")).memory_id
    b = upsert(store, cand("I use Go for services", "Go", pred="uses_tool", subj="user", cat="skill", ts=T0 + DAY)).memory_id
    assert status(store, a) == "current" and status(store, b) == "current"


def test_purge_keeps_tombstones_and_user_confirmed_rows(store: Store) -> None:
    seed = upsert(store, Candidate(statement="Seeded fact one", category="fact", ref="d", ref_kind="document", origin="claude_memory",
                                   verification="provider_stated")).memory_id  # fmt: skip
    gone = upsert(store, Candidate(statement="Forgotten fact two", category="fact", ref="d", ref_kind="document", origin="claude_memory",
                                   verification="provider_stated")).memory_id  # fmt: skip
    confirmed = upsert(store, Candidate(statement="Confirmed fact three", category="fact", ref="m", ref_kind="message", origin="claude_memory",
                                        verification="user_stated")).memory_id  # fmt: skip
    store.conn.execute("UPDATE memories SET status='deleted', statement='[forgotten]' WHERE id=?", (gone,))
    store.conn.execute("INSERT INTO documents(id, kind, name, text, content_hash) VALUES ('d','claude_memory','n','t','h')")
    store.conn.commit()
    assert purge_claude_seeds(store) == 1
    left = {r["id"] for r in store.conn.execute("SELECT id FROM memories")}
    assert seed not in left and gone in left and confirmed in left


def test_sensitivity_lexicon_is_general_and_precise() -> None:
    for text in ("I take 20 mg of sertraline", "My salary is high", "I earn ₹50,000 a month", "My mom lives in Pune", "I weigh 80 kg",
                 "I was diagnosed with diabetes", "He has insomnia"):  # fmt: skip
        assert sensitivity_of(text) == "sensitive", text
    for text in ("The loan prediction model scored 0.9", "Open the parent directory", "I use Python for data analysis"):
        assert sensitivity_of(text) == "normal", text


def test_secret_detection_and_redaction() -> None:
    for text in ("my password is hunter2hunter2", "postgres://admin:s3cr3t@db.local/app", "key AIzaSyA-1234567890abcdefghijklmnopqrstuv",
                 "token github_pat_11ABCDEFG0123456789abcdef", "card 4111 1111 1111 1111"):  # fmt: skip
        assert secret(text), text
        assert "[REDACTED]" in redact_secrets(text)
    assert not secret("created at 1700000000000 ms")  # a 13-digit timestamp is not a card number
    assert not secret("spin: values are fine")
