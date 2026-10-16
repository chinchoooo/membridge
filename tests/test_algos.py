from __future__ import annotations

import random
from itertools import product

import pytest

from membridge.graphalgo.adjacency import Graph
from membridge.graphalgo.ppr import exact_ppr, personalized_pagerank, specificity
from membridge.retrieval.fusion import mmr, rrf, token_jaccard, weighted_sum
from membridge.retrieval.pack import Option, pack


def _graph() -> Graph:
    """12-ring + two chords + a hub (h) with 20 leaves attached to ring node n0."""
    g = Graph()
    for i in range(12):
        g.add(f"n{i}", f"n{(i + 1) % 12}", "REL", 0.9)
    g.add("n0", "n6", "REL", 0.5)
    g.add("n3", "n9", "ABOUT", 1.0)
    g.add("n0", "h", "ABOUT", 1.0)
    for j in range(20):
        g.add("h", f"leaf{j}", "ABOUT", 1.0)
    return g


def _volume(g: Graph) -> int:
    return sum(len(g.neighbors(n)) for n in g.nodes())


# ---- personalized PageRank -------------------------------------------------------------------------
def test_mass_is_conserved_and_nonnegative() -> None:
    g = _graph()
    res = personalized_pagerank(g, {"n0": 1.0, "n5": 0.5}, eps=1e-5)
    assert res.mass() == pytest.approx(1.0, abs=1e-9)
    assert all(v >= 0 for v in res.p.values()) and all(v >= 0 for v in res.r.values())


def test_forward_push_error_is_bounded_and_shrinks_with_eps() -> None:
    g = _graph()
    seeds = {"n0": 1.0}
    truth = exact_ppr(g, seeds)
    errs = {}
    for eps in (1e-2, 1e-3, 1e-5):
        est = personalized_pagerank(g, seeds, eps=eps).p
        errs[eps] = sum(abs(truth.get(n, 0.0) - est.get(n, 0.0)) for n in set(truth) | set(est))
        assert all(est.get(n, 0.0) <= truth.get(n, 0.0) + 1e-9 for n in est), "push estimate must lower-bound the exact PPR"
        assert errs[eps] <= eps * _volume(g) + 1e-9, f"L1 error {errs[eps]} exceeds eps*vol"
    assert errs[1e-5] < errs[1e-3] < errs[1e-2]


def test_deterministic() -> None:
    g = _graph()
    a = personalized_pagerank(g, {"n0": 1.0, "n7": 0.3})
    b = personalized_pagerank(g, {"n7": 0.3, "n0": 1.0})
    assert a.p == b.p and a.pushes == b.pushes


def test_multi_hop_reach_and_path_attribution() -> None:
    g = Graph()
    g.add("seed", "a", "ABOUT", 1.0)
    g.add("a", "b", "REL", 0.9)
    g.add("b", "target", "ABOUT", 1.0)
    res = personalized_pagerank(g, {"seed": 1.0}, eps=1e-6)
    assert res.p.get("target", 0.0) > 0, "mass must reach a node three hops away"
    path = res.path("target")
    assert [n for n, _ in path] == ["seed", "a", "b"] and [t for _, t in path] == ["ABOUT", "REL", "ABOUT"]


def test_hub_damping_reduces_hub_drift() -> None:
    g = _graph()
    plain = personalized_pagerank(g, {"n2": 1.0}, hub_beta=0.0, eps=1e-6).p.get("h", 0.0)
    damped = personalized_pagerank(g, {"n2": 1.0}, hub_beta=2.0, eps=1e-6).p.get("h", 0.0)
    assert damped < plain
    assert specificity(1) > specificity(100) > 0


def test_edge_weights_steer_the_walk() -> None:
    g = Graph()
    g.add("s", "strong", "ABOUT", 1.0)
    g.add("s", "weak", "RELATED_TO", 0.1)
    res = personalized_pagerank(g, {"s": 1.0}, hub_beta=0.0, eps=1e-7)
    assert res.p["strong"] > res.p["weak"]


def test_unknown_or_empty_seeds_return_empty() -> None:
    assert personalized_pagerank(_graph(), {"nope": 1.0}).p == {}
    assert personalized_pagerank(Graph(), {}).p == {}


def test_graph_collapses_parallel_edges_to_max_weight() -> None:
    g = Graph()
    g.add("a", "b", "REL", 0.2)
    g.add("b", "a", "REL", 0.7)
    g.add("a", "b", "ABOUT", 0.4)
    assert dict(((n, t), w) for n, t, w in g.neighbors("a")) == {("b", "REL"): 0.7, ("b", "ABOUT"): 0.4}
    assert g.signature() == {("a", "b", "REL", 0.7), ("a", "b", "ABOUT", 0.4)}


# ---- fusion ---------------------------------------------------------------------------------------------
def test_rrf_rewards_agreement_and_respects_weights() -> None:
    scores, by = rrf({"a": ["x", "y", "z"], "b": ["y", "x"], "c": ["y"]})
    assert max(scores, key=scores.__getitem__) == "y" and by["y"] == ["a", "b", "c"] and "z" in scores
    heavy, _ = rrf({"a": ["x"], "b": ["y"]}, {"a": 1.0, "b": 3.0})
    assert heavy["y"] > heavy["x"]
    assert rrf({}) == ({}, {})


def test_weighted_sum_uses_score_magnitude() -> None:
    s, _ = weighted_sum({"a": {"x": 10.0, "y": 1.0}, "b": {"y": 5.0}})
    assert s["x"] == pytest.approx(1.0) and s["y"] == pytest.approx(0.1 + 1.0)


def test_mmr_trades_relevance_for_diversity() -> None:
    toks = {"a": {"red", "apple", "fruit"}, "a2": {"red", "apple", "fruit", "sweet"}, "b": {"blue", "car"}}
    rel = {"a": 1.0, "a2": 0.95, "b": 0.8}
    sim = lambda x, y: token_jaccard(toks[x], toks[y])  # noqa: E731
    assert mmr(["a", "a2", "b"], rel, sim, lam=0.7, k=2) == ["a", "b"]
    assert mmr(["a", "a2", "b"], rel, sim, lam=1.0, k=2) == ["a", "a2"]  # pure relevance keeps the near-duplicate
    assert mmr(["a", "a2", "b"], rel, sim, lam=0.7, k=2, must=["b", "b"])[0] == "b"  # must first, de-duplicated


# ---- knapsack -----------------------------------------------------------------------------------------------
def _brute(groups: list[list[Option]], budget: int) -> float:
    best = 0.0
    for combo in product(*[[None, *range(len(g))] for g in groups]):
        cost = sum(groups[i][j].cost for i, j in enumerate(combo) if j is not None)
        if cost <= budget:
            best = max(best, sum(groups[i][j].value for i, j in enumerate(combo) if j is not None))
    return best


def test_pack_is_optimal_against_brute_force() -> None:
    rnd = random.Random(3)
    for _ in range(60):
        n = rnd.randint(1, 6)
        groups = []
        for _ in range(n):
            full = rnd.randint(20, 120)
            v = rnd.random()
            groups.append([Option(v, full), Option(v * 0.9, max(5, full // 2))])
        budget = rnd.randint(30, 300)
        picks = pack(groups, budget)
        cost = sum(groups[i][j].cost for i, j in enumerate(picks) if j is not None)
        val = sum(groups[i][j].value for i, j in enumerate(picks) if j is not None)
        assert cost <= budget
        assert val == pytest.approx(_brute(groups, budget), abs=1e-9)


def test_pack_prefers_shortening_over_dropping_and_handles_edges() -> None:
    groups = [[Option(1.0, 100), Option(0.9, 40)], [Option(0.8, 100), Option(0.72, 40)]]
    assert pack(groups, 140) == [0, 1]  # full+short (1.72) beats short+short (1.62)
    assert pack(groups, 80) == [1, 1]  # shortening both beats dropping one (0.9 + 0.72 vs 0.9)
    assert pack(groups, 200) == [0, 0]
    assert pack(groups, 0) == [None, None] and pack([], 100) == []
    assert pack([[Option(1.0, 500)]], 100) == [None]
