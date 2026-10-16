"""Blocking: cheap candidate-pair generation so resolution never compares all O(n^2) pairs.

Four independent blockers are unioned (recall comes from the union, precision from the later scorer):
exact compact key, curated alias dictionary, MinHash-LSH over character 3-gram shingles (typos,
abbreviations), and shared rare tokens. Deterministic: fixed seeds, no randomness at run time.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from itertools import combinations

_PRIME = (1 << 61) - 1
_NUM_PERM = 64
_BANDS = 16  # 16 bands x 4 rows: collides pairs with Jaccard >~ 0.5 with high probability
_ROWS = _NUM_PERM // _BANDS
_MAX_BLOCK = 30  # a block bigger than this is a stop-word-like token, not evidence


def compact(canon: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", canon)


def shingles(canon: str, n: int = 3) -> set[str]:
    s = f"^{compact(canon)}$"
    return {s[i : i + n] for i in range(max(1, len(s) - n + 1))}


def _h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


_rng = random.Random(0xB10C)
_PERMS = [(_rng.randrange(1, _PRIME), _rng.randrange(0, _PRIME)) for _ in range(_NUM_PERM)]


def minhash(shingle_set: set[str]) -> list[int]:
    hashes = [_h64(s) for s in shingle_set] or [0]
    return [min((a * h + b) % _PRIME for h in hashes) for a, b in _PERMS]


def estimate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    return sum(x == y for x, y in zip(sig_a, sig_b, strict=True)) / len(sig_a)


def lsh_pairs(items: dict[str, str]) -> set[tuple[str, str]]:
    """items: id -> canonical name. Returns candidate id pairs sharing at least one LSH band bucket."""
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = defaultdict(list)
    for eid, canon in items.items():
        sig = minhash(shingles(canon))
        for band in range(_BANDS):
            buckets[(band, tuple(sig[band * _ROWS : (band + 1) * _ROWS]))].append(eid)
    return _pairs_from_buckets(buckets.values())


def _pairs_from_buckets(groups: object) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for g in groups:  # type: ignore[attr-defined]
        if 1 < len(g) <= _MAX_BLOCK:
            out.update((a, b) if a < b else (b, a) for a, b in combinations(sorted(set(g)), 2))
    return out


def candidate_pairs(items: dict[str, str], alias_targets: dict[str, str] | None = None) -> set[tuple[str, str]]:
    """Union of all blockers. `alias_targets` maps a compact alias key to a canonical compact key."""
    alias_targets = alias_targets or {}
    by_key: dict[str, list[str]] = defaultdict(list)
    by_token: dict[str, list[str]] = defaultdict(list)
    for eid, canon in items.items():
        key = compact(canon)
        by_key[alias_targets.get(key, key)].append(eid)
        if key in alias_targets:
            by_key[key].append(eid)
        for tok in canon.split():
            if len(tok) >= 4:
                by_token[tok].append(eid)
    pairs = _pairs_from_buckets(by_key.values()) | _pairs_from_buckets(by_token.values()) | lsh_pairs(items)
    return pairs


def reduction_ratio(n_entities: int, n_pairs: int) -> float:
    total = n_entities * (n_entities - 1) // 2
    return 1.0 - (n_pairs / total) if total else 1.0
