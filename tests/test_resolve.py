from __future__ import annotations

import random
import string
from itertools import combinations
from pathlib import Path

import pytest

from membridge.importers.service import run_import
from membridge.memory.consolidate import refresh_entity_links, upsert, upsert_entity
from membridge.memory.models import Candidate, EntityRef, canonical
from membridge.memory.pipeline import run_extraction
from membridge.resolve.blocking import candidate_pairs, estimate_jaccard, minhash, reduction_ratio, shingles
from membridge.resolve.cluster import UnionFind
from membridge.resolve.entities import _alias_targets, decide_pair, lookup_alias, resolve_entities, split_merge
from membridge.resolve.score import cannot_link, features, score
from membridge.store.sqlite import Store

TAU_HI, TAU_LO = 0.88, 0.65

# ---- labelled evaluation set --------------------------------------------------------------------
GOLD_CLUSTERS = [
    ["postgresql", "postgres", "pg"], ["vs code", "vscode", "visual studio code"], ["javascript", "js"],
    ["kubernetes", "k8s"], ["react", "reactjs", "react js"], ["node", "nodejs", "node js"], ["tensorflow", "tf"],
    ["mongodb", "mongo"], ["python", "py"], ["typescript", "ts"],
]  # fmt: skip
SINGLETONS = ["pandas", "numpy", "docker", "redis", "sqlite", "mysql", "sql", "rust", "go", "c", "c++", "java", "react native",
              "reactor", "atlas", "atlas shrugged", "nodemon", "python 2", "python 3", "gpt 3", "gpt 4", "kafka", "graphql"]  # fmt: skip
ALL = [n for c in GOLD_CLUSTERS for n in c] + SINGLETONS
CLUSTER_OF = {n: i for i, c in enumerate(GOLD_CLUSTERS) for n in c}


def _gold_same(a: str, b: str) -> bool:
    return a in CLUSTER_OF and b in CLUSTER_OF and CLUSTER_OF[a] == CLUSTER_OF[b]


def _decide(a: str, b: str, ctx: tuple[set[str], set[str]] = (set(), set())) -> tuple[bool, float]:
    if cannot_link(canonical(a), canonical(b), "concept", "concept"):
        return False, 0.0
    sc, _ = score(features(canonical(a), canonical(b), *ctx))
    return sc >= TAU_HI, sc


def test_pairwise_resolution_quality_on_labelled_set() -> None:
    ids = {n: n for n in ALL}
    cands = candidate_pairs({n: canonical(n) for n in ALL}, _alias_targets())
    tp = fp = fn = 0
    missed_blocking = []
    for a, b in combinations(ALL, 2):
        pred, _ = _decide(a, b)
        gold = _gold_same(a, b)
        key = (a, b) if a < b else (b, a)
        if gold and key not in {(x, y) if x < y else (y, x) for x, y in cands}:
            missed_blocking.append(key)
        tp += pred and gold
        fp += pred and not gold
        fn += (not pred) and gold
    assert ids  # keep the mapping referenced
    precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
    print(f"pairwise precision={precision:.3f} recall={recall:.3f} tp={tp} fp={fp} fn={fn}")
    assert fp == 0, "hard negatives (Python 2/3, Java/JavaScript, Atlas/Atlas Shrugged...) must never auto-merge"
    assert recall >= 0.9
    assert not missed_blocking, f"blocking dropped true pairs: {missed_blocking}"  # blocking recall must be 1.0 here


def test_cluster_purity_after_union_find() -> None:
    uf = UnionFind(ALL, max_size=8)
    for a, b in combinations(ALL, 2):
        pred, _ = _decide(a, b)
        if pred:
            uf.union(a, b)
    for cl in uf.clusters():
        golds = {CLUSTER_OF.get(n, f"s:{n}") for n in cl}
        assert len(golds) == 1, f"impure cluster {cl}"
    assert max(len(c) for c in uf.clusters()) <= 3


def test_blocking_prunes_almost_all_pairs_at_scale() -> None:
    rnd = random.Random(7)
    words = {f"w{i}": "".join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(5, 12))) for i in range(600)}
    words["a1"], words["a2"] = "postgresql", "postgres"
    pairs = candidate_pairs(words)
    assert reduction_ratio(len(words), len(pairs)) > 0.97
    assert ("a1", "a2") in pairs or ("a2", "a1") in pairs


def test_minhash_is_deterministic_and_estimates_jaccard() -> None:
    a, b, c = shingles("postgresql"), shingles("postgres"), shingles("kubernetes")
    assert minhash(a) == minhash(a)
    true = len(a & b) / len(a | b)
    assert abs(estimate_jaccard(minhash(a), minhash(b)) - true) < 0.2
    assert estimate_jaccard(minhash(a), minhash(c)) < 0.2


def test_union_find_guards() -> None:
    uf = UnionFind(["a", "b", "c", "d"], cannot_link=[("a", "d")], max_size=3)
    assert uf.union("a", "b") == (True, "ok") and uf.union("b", "c") == (True, "ok")
    assert uf.union("a", "c") == (False, "same")  # already together: not a new merge
    assert uf.union("c", "d") == (False, "max_cluster_size") or uf.union("c", "d")[1] in ("cannot_link", "max_cluster_size")
    uf2 = UnionFind(["a", "b", "c"], cannot_link=[("a", "c")], max_size=8)
    uf2.union("a", "b")
    assert uf2.union("b", "c") == (False, "cannot_link")  # transitive chain a-b-c is blocked
    assert uf2.find("a") == uf2.find("b") != uf2.find("c")


def test_three_tiers_typo_with_and_without_context() -> None:
    no_ctx = score(features("postgres", "postgress", set(), set()))[0]
    with_ctx = score(features("postgres", "postgress", {"c:1", "c:2"}, {"c:1", "c:2"}))[0]
    assert TAU_LO <= no_ctx < TAU_HI, "a bare typo is only a candidate for review"
    assert with_ctx >= TAU_HI, "the same typo sharing conversations is strong evidence"


# ---- end to end on the database ------------------------------------------------------------------
def _mem(store: Store, text: str, ents: list[str], ref: str, ts: float = 1.0, category: str = "fact") -> str:
    return upsert(store, Candidate(statement=text, category=category, entities=[EntityRef(e) for e in ents], ref=ref, ts=ts)).memory_id


def _dump(store: Store) -> dict[str, list[tuple[object, ...]]]:
    c = store.conn
    return {
        "memory_entities": sorted(tuple(r) for r in c.execute("SELECT memory_id, entity_id, role FROM memory_entities")),
        "mentions": sorted(tuple(r) for r in c.execute("SELECT entity_id, conversation_id, count FROM entity_mentions")),
        "aliases": sorted(tuple(r) for r in c.execute("SELECT entity_id, alias_norm FROM entity_aliases")),
        "live": sorted(tuple(r) for r in c.execute("SELECT id, type FROM live_entities")),
        "subjects": sorted(tuple(r) for r in c.execute("SELECT id, subject, subject_entity_id FROM memories")),
    }


@pytest.fixture
def populated(export_file: Path, store: Store) -> Store:
    run_import(store, export_file)
    _mem(store, "I decided to use PostgreSQL for Atlas storage.", ["PostgreSQL", "Atlas"], "x1", 1.0, "decision")
    _mem(store, "The pg connection pool kept failing under load.", ["pg", "Atlas"], "x2", 2.0)
    _mem(store, "I like Postgres extensions a lot.", ["Postgres"], "x3", 3.0)
    _mem(store, "Atlas Shrugged is on my reading list.", ["Atlas Shrugged"], "x4", 4.0)
    _mem(store, "I am learning Python 3 typing.", ["Python 3"], "x5", 5.0)
    _mem(store, "Legacy scripts still run on Python 2.", ["Python 2"], "x6", 6.0)
    refresh_entity_links(store)
    store.conn.commit()
    return store


def test_end_to_end_merge_repoints_and_blocks_hard_negatives(populated: Store) -> None:
    s = populated
    before = s.conn.execute("SELECT COUNT(*) FROM live_entities").fetchone()[0]
    rep = resolve_entities(s)
    live = {r["canonical"] for r in s.conn.execute("SELECT canonical FROM live_entities")}
    assert s.conn.execute("SELECT COUNT(*) FROM live_entities").fetchone()[0] == before - 2  # postgres + pg absorbed
    assert {"python 2", "python 3", "atlas", "atlas shrugged"} <= live
    assert len(rep.auto_merged) == 2 and rep.blocked_cannot_link >= 1
    survivor = lookup_alias(s, "Postgres")
    assert survivor == lookup_alias(s, "pg") == lookup_alias(s, "PostgreSQL")
    n = s.conn.execute("SELECT COUNT(DISTINCT memory_id) FROM memory_entities WHERE entity_id=?", (survivor,)).fetchone()[0]
    assert n == 3  # all three memories now hang off one node
    assert upsert_entity(s, EntityRef("Postgres")) == survivor  # a new mention does not resurrect a duplicate
    assert s.conn.execute("SELECT COUNT(*) FROM entity_merges WHERE reversed_at IS NULL").fetchone()[0] == 2


def test_resolution_is_idempotent_and_dry_run_changes_nothing(populated: Store) -> None:
    dry = resolve_entities(populated, dry_run=True)
    assert dry.dry_run and len(dry.auto_merged) == 2 and populated.conn.execute("SELECT COUNT(*) FROM entity_merges").fetchone()[0] == 0
    resolve_entities(populated)
    snap = _dump(populated)
    again = resolve_entities(populated)
    assert again.auto_merged == [] and _dump(populated) == snap


def test_split_restores_state_exactly_and_prevents_remerge(populated: Store) -> None:
    s = populated
    seed = resolve_entities(s, dry_run=True)  # noqa: F841
    from membridge.resolve.entities import seed_aliases

    seed_aliases(s)
    s.conn.commit()
    before = _dump(s)
    resolve_entities(s)
    assert _dump(s) != before
    for m in s.conn.execute("SELECT id FROM entity_merges WHERE reversed_at IS NULL ORDER BY id DESC").fetchall():
        split_merge(s, m["id"])
    assert _dump(s) == before  # every table is back to exactly what it was
    rep = resolve_entities(s)
    assert rep.auto_merged == [], "a split pair is cannot-link: it must not be merged again"


def test_pending_candidates_can_be_decided_by_a_human(store: Store) -> None:
    _mem(store, "I use reactnative daily.", ["ReactNative"], "m1", 1.0)
    _mem(store, "React Native builds are slow.", ["React Nativ"], "m2", 2.0)  # typo variant, no shared context
    rep = resolve_entities(store)
    assert rep.pending == 1 and rep.auto_merged == []
    cand = store.conn.execute("SELECT a_id, b_id FROM entity_link_candidates WHERE status='pending'").fetchone()
    out = decide_pair(store, cand["a_id"], cand["b_id"], accept=True)
    assert out["status"] == "merged" and store.conn.execute("SELECT COUNT(*) FROM live_entities").fetchone()[0] == 1


def test_rejected_pair_is_never_merged(store: Store) -> None:
    _mem(store, "I use postgres daily.", ["Postgres"], "m1", 1.0)
    _mem(store, "I run PostgreSQL in prod.", ["PostgreSQL"], "m2", 2.0)
    ids = [r["id"] for r in store.conn.execute("SELECT id FROM live_entities")]
    decide_pair(store, ids[0], ids[1], accept=False)
    assert resolve_entities(store).auto_merged == []


def test_pipeline_runs_resolution_and_real_sample_has_no_false_merges(export_file: Path, store: Store) -> None:
    run_import(store, export_file)
    st = run_extraction(store)
    assert st.entities_merged == 0  # nothing in the clean sample is an alias of anything else
    assert store.conn.execute("SELECT COUNT(*) FROM entity_merges").fetchone()[0] == 0


def test_numeronyms_are_not_versions_but_real_versions_still_block() -> None:
    assert cannot_link("k8s", "kubernetes", "concept", "concept") is None
    assert cannot_link("i18n", "internationalization", "concept", "concept") is None
    assert cannot_link("python 2", "python 3", "concept", "concept") == "version_mismatch"
    assert cannot_link("gpt4", "gpt 3", "concept", "concept") == "version_mismatch"
    assert cannot_link("python", "python 3", "concept", "concept") == "version_mismatch"
    assert cannot_link("alice", "acme", "person", "organisation") == "type_conflict"


def test_labelled_set_reaches_full_recall_after_the_numeronym_fix() -> None:
    fn = [(a, b) for a, b in combinations(ALL, 2) if _gold_same(a, b) and not _decide(a, b)[0]]
    assert fn == []
