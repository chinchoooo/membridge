from __future__ import annotations

import json
from pathlib import Path

from membridge.eval_real import eval_real, gold_init
from membridge.importers.service import run_import
from membridge.memory.consolidate import upsert
from membridge.memory.models import Candidate, EntityRef
from membridge.memory.pipeline import run_extraction
from membridge.retrieval.engine import Retriever
from membridge.store.sqlite import Store


def _store_with_facts(store: Store, n: int = 14) -> None:
    for i in range(n):
        upsert(store, Candidate(statement=f"I decided to use Tool{i} for Project{i} storage.", category="decision", subject=f"project{i}", predicate=f"uses:storage{i}",
                                object=f"Tool{i}", entities=[EntityRef(f"Project{i}", "project"), EntityRef(f"Tool{i}")], ref=f"r{i}", ts=1000.0 + i, verification="user_stated"))  # fmt: skip
    run_extraction(store)


def test_gold_init_builds_pools_and_refuses_to_score_without_judgments(store: Store, tmp_path: Path) -> None:
    _store_with_facts(store)
    r = Retriever(store)
    out = tmp_path / "gold.json"
    info = gold_init(store, r, out, n=12)
    data = json.loads(out.read_text())
    assert info["questions"] == 12 and all(q["pool"] and not any(c["relevant"] for c in q["pool"]) for q in data["questions"])
    res = eval_real(r, out)
    assert res["status"] == "not_enough_judgments" and "judged_questions" in res and "recall@5" not in res


def test_scores_are_computed_from_your_judgments_and_preserved_on_regeneration(store: Store, tmp_path: Path) -> None:
    _store_with_facts(store)
    r = Retriever(store)
    out = tmp_path / "gold.json"
    gold_init(store, r, out, n=12)
    data = json.loads(out.read_text())
    for q in data["questions"]:  # judge: the pool item whose statement mentions the question's focus project is the right one
        focus = q["q"].split("Project")[1].split()[0].rstrip("?)").strip("(")
        for c in q["pool"]:
            c["relevant"] = f"Project{focus} " in c["statement"] or f"Project{focus}." in c["statement"]
    out.write_text(json.dumps(data))
    res = eval_real(r, out)
    assert res["status"] == "ok" and res["judged_questions"] >= 10
    assert res["recall@5"] == 1.0 and res["mrr"] > 0.9 and 0 < res["precision@5"] <= 1.0
    gold_init(store, r, out, n=12)  # regenerate: judgments survive
    assert sum(c["relevant"] for q in json.loads(out.read_text())["questions"] for c in q["pool"]) >= 10


def test_a_system_that_misses_answers_scores_low(store: Store, tmp_path: Path) -> None:
    _store_with_facts(store)
    r = Retriever(store)
    out = tmp_path / "gold.json"
    gold_init(store, r, out, n=12)
    data = json.loads(out.read_text())
    for q in data["questions"]:
        q["pool"].append({"id": "mem_never_returned", "statement": "unreachable", "relevant": True})  # relevant but never retrievable
    out.write_text(json.dumps(data))
    res = eval_real(r, out)
    assert res["recall@5"] == 0.0 and len(res["questions_with_no_relevant_result"]) == res["judged_questions"]


def test_eval_real_on_the_sample_archive(export_file: Path, store: Store, tmp_path: Path) -> None:
    run_import(store, export_file)
    run_extraction(store)
    out = tmp_path / "g.json"
    assert gold_init(store, Retriever(store), out, n=25)["questions"] >= 2
