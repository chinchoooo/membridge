"""Warm in-memory index: everything a search needs that is not full-text, held in plain dicts.

Built in one pass from SQLite and rebuilt lazily when `Store.generation()` changes (trigger-maintained, so a write by any
process invalidates it). A search then needs no SQL for entity linking, fact loading, per-entity fact lists, supersede
chains or evidence resolution: only the two FTS5 lookups remain. Memory cost is a few MB per 10k facts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..store.sqlite import Store


def canon(text: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip()


@dataclass
class WarmIndex:
    gen: int
    full: bool = False  # True only for the grant-authorised view that includes sensitive memories
    built_ms: float = 0.0
    facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    phrases: dict[str, list[tuple[str, str, float]]] = field(default_factory=dict)  # phrase -> [(entity_id, name, strength)]
    max_words: int = 1
    ent_name: dict[str, str] = field(default_factory=dict)
    fact_entities: dict[str, list[str]] = field(default_factory=dict)
    degree: dict[str, int] = field(default_factory=dict)
    entity_facts: dict[str, list[str]] = field(default_factory=dict)  # sorted: current first, score desc
    conv_projects: dict[str, list[str]] = field(default_factory=dict)
    conv_title: dict[str, str] = field(default_factory=dict)
    ref_facts: dict[str, list[str]] = field(default_factory=dict)  # message/document id -> fact ids it supports
    superseders: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # older -> [(newer id, newer status)] newest first
    evidence: dict[tuple[str, int], list[dict[str, Any]]] = field(default_factory=dict)  # lazy, per generation

    def link(self, query: str) -> dict[str, tuple[str, float]]:
        """Entities named in `query` (canonical names and aliases) by word n-gram lookup: O(query length)."""
        words = canon(query).split()
        hits: dict[str, tuple[str, float]] = {}
        for n in range(1, min(self.max_words, len(words)) + 1):
            for i in range(len(words) - n + 1):
                for eid, name, strength in self.phrases.get(" ".join(words[i : i + n]), ()):
                    if eid not in hits or strength > hits[eid][1]:
                        hits[eid] = (name, strength)
        return hits


def build_index(store: Store, include_sensitive: bool = False) -> WarmIndex:
    """The default view contains no sensitive memory at all, so nothing downstream (entity lists, PPR seeds, supersede
    chains, evidence maps) can rank by, count, or even label one."""
    t0 = time.perf_counter()
    c = store.conn
    ix = WarmIndex(gen=store.generation(), full=include_sensitive)
    sens = "" if include_sensitive else " AND sensitivity='normal'"
    ix.facts = {r["id"]: dict(r) for r in c.execute(f"SELECT * FROM memories WHERE status!='deleted'{sens}")}
    names: dict[str, str] = {}
    for r in c.execute("SELECT id, name, canonical FROM live_entities"):
        names[r["id"]] = r["name"]
        ix.phrases.setdefault(r["canonical"], []).append((r["id"], r["name"], 1.0))
    for r in c.execute("SELECT a.entity_id AS id, a.alias_norm AS alias FROM entity_aliases a JOIN live_entities e ON e.id=a.entity_id"):
        if r["alias"] and all(x[0] != r["id"] for x in ix.phrases.get(r["alias"], ())):
            ix.phrases.setdefault(r["alias"], []).append((r["id"], names[r["id"]], 0.95))
    ix.ent_name = names
    # curated aliases ("pg", "k8s", "vscode") also work as *query* words for whichever entity they stand for
    from ..resolve.blocking import compact
    from ..resolve.score import ALIAS_DICT

    by_compact: dict[str, tuple[str, str]] = {}
    for phrase, entries in list(ix.phrases.items()):
        for eid, name, _ in entries:
            by_compact.setdefault(compact(phrase), (eid, name))
    for alias, target in ALIAS_DICT.items():
        hit = by_compact.get(compact(target))
        a = canon(alias)
        if hit and a and all(x[0] != hit[0] for x in ix.phrases.get(a, ())):
            ix.phrases.setdefault(a, []).append((hit[0], hit[1], 0.9))
    ix.max_words = min(6, max((len(p.split()) for p in ix.phrases), default=1))
    per: dict[str, set[str]] = {}
    for r in c.execute("SELECT entity_id, memory_id FROM memory_entities"):
        if r["memory_id"] in ix.facts and r["entity_id"] in names:
            per.setdefault(r["entity_id"], set()).add(r["memory_id"])
    for eid, ids in per.items():
        ix.degree[eid] = len(ids)
        for m in ids:
            ix.fact_entities.setdefault(m, []).append(eid)
        ix.entity_facts[eid] = sorted(ids, key=lambda m: (ix.facts[m]["status"] != "current", -float(ix.facts[m]["score"]), m))
    for r in c.execute("SELECT x.conversation_id AS cid, x.entity_id AS eid FROM entity_mentions x JOIN live_entities e ON e.id=x.entity_id WHERE e.type='project'"):
        ix.conv_projects.setdefault(r["cid"], []).append(r["eid"])
    ix.conv_title = {r["id"]: r["title"] or "" for r in c.execute("SELECT id, title FROM conversations")}
    for r in c.execute("SELECT ref, memory_id FROM memory_evidence"):
        if r["memory_id"] in ix.facts:
            ix.ref_facts.setdefault(r["ref"], []).append(r["memory_id"])
    for r in c.execute(
        f"SELECT l.dst_id AS dst, l.src_id AS src, m.status AS st FROM memory_links l JOIN memories m ON m.id=l.src_id "
        f"WHERE l.rel='SUPERSEDES' AND m.status!='deleted'{sens.replace('sensitivity', 'm.sensitivity')} ORDER BY l.created_at DESC"
    ):
        ix.superseders.setdefault(r["dst"], []).append((r["src"], r["st"]))
    ix.built_ms = round((time.perf_counter() - t0) * 1000, 2)
    return ix
