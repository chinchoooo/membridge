"""Pairwise entity scoring: deterministic rules first, then a weighted blend of similarity features.

Hard constraints (cannot-link) are checked before any score is trusted: version numbers must agree
("Python 2" != "Python 3"), two different non-generic types never merge, and a user's earlier "these are
different" decision is final.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import compact

# Curated, high-precision aliases (canonical lower-case, no punctuation). Users can add more via entity_aliases.
ALIAS_DICT: dict[str, str] = {
    "postgres": "postgresql", "pg": "postgresql", "psql": "postgresql", "js": "javascript", "ts": "typescript",
    "k8s": "kubernetes", "py": "python", "gh": "github", "vscode": "visualstudiocode", "vsc": "visualstudiocode",
    "tf": "tensorflow", "sklearn": "scikitlearn", "np": "numpy", "gpt": "gpt", "reactjs": "react", "nodejs": "node",
    "node js": "node", "golang": "go", "cpp": "c++", "mongo": "mongodb", "gcp": "googlecloud", "aws": "amazonwebservices",
}  # fmt: skip
GENERIC_TYPES = {"concept"}


@dataclass
class Features:
    name: float
    jaro_winkler: float
    containment: float
    context: float
    rule: str | None
    embedding: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _canon_alias(c: str) -> str:
    k = compact(c)
    return ALIAS_DICT.get(k, ALIAS_DICT.get(c, k))


def _digits(c: str) -> tuple[str, ...]:
    """Version-like digit runs. A run with a letter on both sides is a numeronym (k8s, i18n), not a version."""
    out = []
    for m in re.finditer(r"\d+", c):
        before = c[m.start() - 1] if m.start() > 0 else ""
        after = c[m.end()] if m.end() < len(c) else ""
        if not (before.isalpha() and after.isalpha()):
            out.append(m.group(0))
    return tuple(out)


def cannot_link(a_canon: str, b_canon: str, a_type: str, b_type: str) -> str | None:
    """Reason these two can never be the same entity, or None."""
    curated = _canon_alias(a_canon) == _canon_alias(b_canon)  # curated alias knowledge beats the version heuristic
    if not curated and _digits(a_canon) != _digits(b_canon):
        return "version_mismatch"
    if a_type not in GENERIC_TYPES and b_type not in GENERIC_TYPES and a_type != b_type:
        return "type_conflict"
    return None


def _containment(a: str, b: str) -> float:
    """Prefix abbreviation signal ('postgres' in 'postgresql'): only for length>=4 and a close length ratio."""
    ca, cb = compact(a), compact(b)
    short, long_ = sorted((ca, cb), key=len)
    if len(short) >= 4 and long_.startswith(short) and len(short) / len(long_) >= 0.7:
        return 1.0
    return 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def features(a_canon: str, b_canon: str, a_ctx: set[str], b_ctx: set[str], embedding: float | None = None) -> Features:
    rule = None
    if compact(a_canon) == compact(b_canon):
        rule = "compact"
    elif _canon_alias(a_canon) == _canon_alias(b_canon):
        rule = "alias_dict"
    return Features(
        name=fuzz.token_sort_ratio(a_canon, b_canon) / 100.0,
        jaro_winkler=JaroWinkler.normalized_similarity(compact(a_canon), compact(b_canon)),
        containment=_containment(a_canon, b_canon),
        context=jaccard(a_ctx, b_ctx),
        rule=rule,
        embedding=embedding,
    )


def score(f: Features) -> tuple[float, str]:
    """Return (score in [0,1], method). Rules override the blend; the blend stays conservative."""
    if f.rule == "compact":
        return 1.0, "exact"
    if f.rule == "alias_dict":
        return 0.97, "alias_dict"
    name = max(f.name, f.jaro_winkler)
    blend = 0.55 * name + 0.20 * f.containment + 0.25 * f.context
    method = "lsh_fuzzy"
    if f.embedding is not None:
        blend = 0.75 * blend + 0.25 * f.embedding
        method = "embedding"
    # an abbreviation-style match that also shares real context is strong evidence
    if f.containment and f.context >= 0.2 and name >= 0.9:
        blend = max(blend, 0.9)
    return round(min(blend, 0.99), 4), method
