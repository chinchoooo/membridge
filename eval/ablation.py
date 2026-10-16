"""Retrieval ablation on the synthetic multi-hop benchmark (measured, never invented).

Run: uv run python eval/ablation.py        -> table on stdout + eval/results/ablation-<ts>.json
Variants add one mechanism at a time so each row's delta is attributable to that mechanism.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import multihop  # noqa: E402

from membridge.importers.service import run_import  # noqa: E402
from membridge.memory.pipeline import run_extraction  # noqa: E402
from membridge.retrieval.engine import RetrievalConfig, Retriever, estimate_tokens  # noqa: E402
from membridge.store.graph import GraphStore  # noqa: E402
from membridge.store.sqlite import Store  # noqa: E402

VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("A  bm25 only (facts + messages)", dict(use_entities=False, use_ppr=False, fusion="rrf", use_mmr=False, use_knapsack=False)),
    ("B  + entity/hop channels, sum fusion", dict(use_entities=True, use_ppr=False, fusion="sum", use_mmr=False, use_knapsack=False)),
    ("C  + personalized PageRank (sum)", dict(use_entities=True, use_ppr=True, fusion="sum", use_mmr=False, use_knapsack=False)),
    ("D  RRF fusion instead of sum", dict(use_entities=True, use_ppr=True, fusion="rrf", use_mmr=False, use_knapsack=False)),
    ("E  + MMR diversity", dict(use_entities=True, use_ppr=True, fusion="rrf", use_mmr=True, use_knapsack=False)),
    ("F  + knapsack packing (full)", dict(use_entities=True, use_ppr=True, fusion="rrf", use_mmr=True, use_knapsack=True)),
]
K = 5


def build_store(workdir: Path) -> Store:
    convs, qs = multihop.build()
    export = workdir / "export" / "conversations.json"
    export.parent.mkdir(parents=True)
    export.write_text(json.dumps(convs), encoding="utf-8")
    store = Store(workdir / "ablation.db")
    run_import(store, export)
    run_extraction(store)
    store._questions = qs  # type: ignore[attr-defined]
    return store


def _ndcg(rels: list[int], n_expected: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:K]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(K, n_expected)))
    return dcg / ideal if ideal else 0.0


def evaluate(store: Store, graph: GraphStore | None, cfg: RetrievalConfig, label: str, budget: int = 1200) -> dict[str, Any]:
    r = Retriever(store, graph, cfg)
    per: list[dict[str, Any]] = []
    raw_cache: dict[str, int] = {}
    for q in store._questions:  # type: ignore[attr-defined]
        pkg = r.search(q["q"], limit=K, token_budget=budget, filters=q.get("filters"))
        stmts = pkg["memories"]
        rels = [1 if any(e in m["statement"] for e in q["expected"]) else 0 for m in stmts]
        found = [e for e in q["expected"] if any(e in m["statement"] for m in stmts)]
        recall = len(found) / len(q["expected"])
        mrr = next((1 / (i + 1) for i, x in enumerate(rels) if x), 0.0)
        ok_current = all(any(c in m["statement"] and m["status"] == "current" for m in stmts) for c in q.get("current", []))
        ok_forbid = not any(f in m["statement"] for m in stmts for f in q.get("forbid", []))
        convs = {e["conversation_id"] for m in stmts for e in m["evidence"] if e.get("conversation_id")}
        raw = 0
        for cid in convs:
            if cid not in raw_cache:
                raw_cache[cid] = sum(estimate_tokens(t[0]) for t in store.conn.execute("SELECT text FROM messages WHERE conversation_id=?", (cid,)))
            raw += raw_cache[cid]
        per.append({
            "id": q["id"], "kind": q["kind"], "recall": recall, "mrr": mrr, "ndcg": _ndcg(rels, len(q["expected"])),
            "pass": recall == 1.0 and ok_current and ok_forbid, "prov": all(m["evidence"] for m in stmts),
            "tokens": pkg["stats"]["tokens_estimate"], "compression": (pkg["stats"]["tokens_estimate"] / raw) if raw else None,
            "ms": pkg["stats"]["latency_ms"],
        })  # fmt: skip
    ms = sorted(p["ms"] for p in per)
    comp = [p["compression"] for p in per if p["compression"] is not None]
    kinds: dict[str, float] = {}
    for kind in sorted({p["kind"] for p in per}):
        rows = [p for p in per if p["kind"] == kind]
        kinds[kind] = round(sum(p["pass"] for p in rows) / len(rows), 3)
    return {
        "variant": label, "n": len(per), "pass_rate": round(sum(p["pass"] for p in per) / len(per), 3),
        "recall@5": round(statistics.mean(p["recall"] for p in per), 3), "mrr": round(statistics.mean(p["mrr"] for p in per), 3),
        "ndcg@5": round(statistics.mean(p["ndcg"] for p in per), 3), "provenance_ok": all(p["prov"] for p in per),
        "tokens_mean": round(statistics.mean(p["tokens"] for p in per), 1), "context_vs_raw": round(statistics.mean(comp), 3) if comp else None,
        "failed": [p["id"] for p in per if not p["pass"]], "ms_p50": round(statistics.median(ms), 1), "ms_p95": round(ms[int(0.95 * (len(ms) - 1))], 1), "pass_by_kind": kinds,
    }  # fmt: skip


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        store = build_store(Path(td))
        stats = store.stats()
        rows = [evaluate(store, None, replace(RetrievalConfig(), **kw), name) for name, kw in VARIANTS]
        full = RetrievalConfig()
        rows.append(evaluate(store, None, replace(full, use_knapsack=False), "E2 tight budget (350 tok), greedy packing", 350))
        rows.append(evaluate(store, None, full, "F2 tight budget (350 tok), knapsack", 350))
        g = GraphStore(name="membridge_ablation")
        if g.available():
            g.rebuild(store)
            rows.append(evaluate(store, g, RetrievalConfig(), "F' full system, FalkorDB graph mode"))
            g.drop()
        else:
            print("NOTE: FalkorDB unreachable: graph-mode row skipped")
        store.close()
    print(f"archive: {stats['conversations']} conversations, {stats['messages']} messages; {len(store._questions) if hasattr(store, '_questions') else ''}")  # noqa: SLF001
    hdr = f"{'variant':42} {'pass':>5} {'R@5':>6} {'MRR':>6} {'nDCG':>6} {'tok':>6} {'ctx/raw':>8} {'p50ms':>7} {'p95ms':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['variant']:42} {r['pass_rate']:>5} {r['recall@5']:>6} {r['mrr']:>6} {r['ndcg@5']:>6} {r['tokens_mean']:>6} "
              f"{str(r['context_vs_raw']):>8} {r['ms_p50']:>7} {r['ms_p95']:>7}")  # fmt: skip
    for r in rows:
        if r["failed"]:
            print(f"  failed [{r['variant'][:30]}]: {r['failed'][:8]}{' ...' if len(r['failed']) > 8 else ''}")
    print("\npass rate by question kind:")
    kinds = sorted(rows[0]["pass_by_kind"])
    print(f"{'variant':42} " + " ".join(f"{k[:14]:>14}" for k in kinds))
    for r in rows:
        print(f"{r['variant']:42} " + " ".join(f"{r['pass_by_kind'][k]:>14}" for k in kinds))
    out = ROOT / "eval" / "results" / f"ablation-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"archive": stats, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
