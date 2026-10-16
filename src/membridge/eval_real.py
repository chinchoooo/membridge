"""Real-data evaluation without invented labels.

`gold_init` drafts questions from your own memories and, for each, a *pool* of candidate answers (the system's top results
united with plain keyword matches, TREC-style, so recall is not flattered by judging only what the system already returned).
You flip `relevant` to true for the right ones. `eval_real` then reports precision/recall/MRR/nDCG, and refuses to report
anything until enough questions have been judged.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .retrieval.engine import Retriever
from .store.sqlite import Store

MIN_JUDGED = 10
K = 5
_TEMPLATES = {
    "decision": "What did I decide about {x}?", "preference": "What are my preferences about {x}?", "project": "What do I know about {x}?",
    "constraint": "What constraints apply to {x}?", "skill": "What do I use {x} for?", "education": "Tell me about my education ({x}).",
    "research": "What are my research interests ({x})?", "profile": "Tell me about myself ({x}).", "interest": "What interests me about {x}?",
}  # fmt: skip


def _focus(store: Store, mem: Any) -> str:
    r = store.conn.execute(
        "SELECT e.name FROM memory_entities me JOIN live_entities e ON e.id=me.entity_id WHERE me.memory_id=? ORDER BY "
        "(me.role='subject') DESC, (me.role='object') DESC LIMIT 1",
        (mem["id"],),
    ).fetchone()
    if r:
        return str(r[0])
    words = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{3,}", mem["statement"])
    return words[0] if words else "this"


def gold_init(store: Store, retriever: Retriever, out: Path, n: int = 25) -> dict[str, Any]:
    """Draft up to n questions (diverse categories, current facts first) with candidate pools. Existing judgments are kept."""
    prior: dict[str, dict[str, bool]] = {}
    if out.exists():
        for q in json.loads(out.read_text(encoding="utf-8")).get("questions", []):
            prior[q["q"]] = {c["id"]: c["relevant"] for c in q["pool"]}
    rows = store.conn.execute(
        "SELECT * FROM memories WHERE status='current' AND sensitivity='normal' AND category!='session' ORDER BY score DESC"
    ).fetchall()
    seen_focus: set[str] = set()
    seen_cat: dict[str, int] = {}
    questions: list[dict[str, Any]] = []
    cap = max(3, n // 5)

    def add(m: Any, focus: str) -> None:
        seen_focus.add(focus.lower())
        seen_cat[m["category"]] = seen_cat.get(m["category"], 0) + 1
        q = _TEMPLATES.get(m["category"], "What do I know about {x}?").format(x=focus)
        pkg = retriever.search(q, limit=8, token_budget=4000)
        pool: dict[str, str] = {x["id"]: x["statement"] for x in pkg["memories"]}
        for r in store.conn.execute("SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ? LIMIT 10", ('"' + focus.replace('"', "") + '"',)):
            row = store.conn.execute("SELECT statement FROM memories WHERE id=? AND status!='deleted' AND sensitivity='normal'", (r[0],)).fetchone()
            if row:
                pool.setdefault(r[0], row[0])
        questions.append({"q": q, "pool": [{"id": i, "statement": st, "relevant": prior.get(q, {}).get(i, False)} for i, st in pool.items()]})

    for diverse_pass in (True, False):  # first prefer variety across categories, then fill any remaining slots
        for m in rows:
            if len(questions) >= n:
                break
            focus = _focus(store, m)
            if focus.lower() in seen_focus or (diverse_pass and seen_cat.get(m["category"], 0) >= cap):
                continue
            add(m, focus)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"instructions": "Set relevant=true on every pool item that correctly answers or supports the question.", "questions": questions}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"questions": len(questions), "path": str(out)}


def _ndcg(rels: list[int], total: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:K]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(K, total)))
    return dcg / ideal if ideal else 0.0


def eval_real(retriever: Retriever, gold: Path) -> dict[str, Any]:
    data = json.loads(gold.read_text(encoding="utf-8"))["questions"]
    judged = [q for q in data if any(c["relevant"] for c in q["pool"])]
    if len(judged) < MIN_JUDGED:
        return {"status": "not_enough_judgments", "judged_questions": len(judged), "needed": MIN_JUDGED,
                "message": f"Mark at least {MIN_JUDGED} questions (set relevant=true on correct pool items in {gold}); no numbers are reported without them."}  # fmt: skip
    rec, prec, mrr, ndcg, fails = [], [], [], [], []
    for q in judged:
        relevant = {c["id"] for c in q["pool"] if c["relevant"]}
        got = [m["id"] for m in retriever.search(q["q"], limit=K, token_budget=4000)["memories"]]
        rels = [1 if i in relevant else 0 for i in got]
        hit = sum(rels)
        rec.append(hit / len(relevant))
        prec.append(hit / len(got) if got else 0.0)
        mrr.append(next((1 / (i + 1) for i, r in enumerate(rels) if r), 0.0))
        ndcg.append(_ndcg(rels, len(relevant)))
        if hit == 0:
            fails.append(q["q"])
    n = len(judged)
    return {"status": "ok", "judged_questions": n, f"recall@{K}": round(sum(rec) / n, 3), f"precision@{K}": round(sum(prec) / n, 3),
            "mrr": round(sum(mrr) / n, 3), f"ndcg@{K}": round(sum(ndcg) / n, 3), "questions_with_no_relevant_result": fails,
            "caveat": "judgments are pooled from system results + keyword matches; relevant facts found by neither are unseen."}  # fmt: skip
