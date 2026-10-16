"""Embedders. Optional everywhere: nothing in the core pipeline requires one.

FastEmbedder  real semantic model (BAAI/bge-small-en-v1.5, ONNX, CPU, ~130 MB, extra `embeddings`).
HashingEmbedder  dependency-free feature hashing of word uni/bigrams: deterministic and fast, captures *lexical*
                 overlap only. Used in tests and as a fallback; it is NOT semantic and is labelled as such.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim, self.name = dim, f"hashing-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            toks = re.findall(r"[a-z0-9+#]{2,}", t.lower())
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            v = [0.0] * self.dim
            for g in grams:
                h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collision bias
            out.append(_norm(v))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FastEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # lazy: only when this embedder is actually requested

        self._m: Any = TextEmbedding(model)
        self.name, self.dim = f"fastembed:{model}", 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_norm([float(x) for x in v]) for v in self._m.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [_norm([float(x) for x in v]) for v in self._m.query_embed([text])][0]


def build_embedder(kind: str) -> Embedder | None:
    if kind == "none":
        return None
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedder()
    raise ValueError(f"unknown embedder '{kind}' (none | fastembed | hashing)")
