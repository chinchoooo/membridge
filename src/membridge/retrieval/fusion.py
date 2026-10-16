"""Rank fusion (RRF) and diversity (MMR). Both are rank/score-scale agnostic, so BM25, graph diffusion and
vector similarity can be combined without calibrating their raw scores."""

from __future__ import annotations

from collections.abc import Callable


def rrf(
    rankings: dict[str, list[str]], weights: dict[str, float] | None = None, k: int = 60
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Reciprocal Rank Fusion: score(d) = sum_channels w_c / (k + rank_c(d)). Returns (scores, channels per doc)."""
    weights = weights or {}
    scores: dict[str, float] = {}
    hit_by: dict[str, list[str]] = {}
    for ch, ranked in rankings.items():
        w = weights.get(ch, 1.0)
        for i, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + i)
            hit_by.setdefault(doc, []).append(ch)
    return scores, hit_by


def weighted_sum(channel_scores: dict[str, dict[str, float]], weights: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Baseline fusion for ablations: sum of per-channel scores normalised to [0,1] (needs comparable scales)."""
    weights = weights or {}
    scores: dict[str, float] = {}
    hit_by: dict[str, list[str]] = {}
    for ch, sc in channel_scores.items():
        top = max(sc.values(), default=0.0) or 1.0
        for doc, v in sc.items():
            scores[doc] = scores.get(doc, 0.0) + weights.get(ch, 1.0) * v / top
            hit_by.setdefault(doc, []).append(ch)
    return scores, hit_by


def token_jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def mmr(
    candidates: list[str],
    relevance: dict[str, float],
    similarity: Callable[[str, str], float],
    *,
    lam: float = 0.7,
    k: int = 10,
    must: list[str] | None = None,
) -> list[str]:
    """Maximal Marginal Relevance: greedily pick argmax lam*rel(d) - (1-lam)*max_sim(d, selected).

    `must` documents are placed first (they still count as selected for the diversity penalty).
    """
    chosen: list[str] = list(dict.fromkeys(m for m in (must or []) if m in relevance))[:k]  # de-duplicated
    pool = [c for c in candidates if c not in chosen]
    top = max(relevance.values(), default=0.0) or 1.0
    while pool and len(chosen) < k:
        def value(d: str) -> float:
            div = max((similarity(d, s) for s in chosen), default=0.0)
            return lam * relevance[d] / top - (1.0 - lam) * div

        best = max(pool, key=lambda d: (value(d), relevance[d], d))
        chosen.append(best)
        pool.remove(best)
    return chosen
