"""Memory-call latency: cold (never-seen query), repeat (same query again) and end-to-end through MemoryService.

Run: uv run python eval/latency.py real          # a copy of your real store (never modifies it)
     uv run python eval/latency.py synth 1500    # synthetic archive of N conversations
All numbers are wall-clock milliseconds measured in-process (the MCP/HTTP transport adds its own fixed overhead).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from membridge.config import Settings  # noqa: E402
from membridge.memory.service import MemoryService  # noqa: E402
from membridge.store.sqlite import Store  # noqa: E402

REAL_QUERIES = [
    "What tech stack do I use?", "What projects am I working on?", "What is my educational background?", "Which languages do I prefer?",
    "What did I build with React?", "Tell me about my research interests", "What internships have I done?", "What do I eat?",
    "Which database do I use for projects?", "What tools do I use for deployment?", "Do I prefer C++ or Rust?", "What is my name and role?",
    "What hackathons or competitions did I join?", "What am I learning right now?", "What is my deadline manager project?",
    "How do I authenticate users in my apps?", "Which cloud services do I use?", "What are my hobbies?", "What do I study at university?",
    "What machine learning topics interest me?",
]  # fmt: skip


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))], 3)


def summarize(xs: list[float]) -> dict[str, float]:
    return {"p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "p99": pct(xs, 0.99), "mean": round(statistics.mean(xs), 3), "n": len(xs)}


def measure(svc: MemoryService, queries: list[str]) -> dict[str, Any]:
    cold: list[float] = []
    again: list[float] = []
    for q in queries:  # first sighting of each query
        t = time.perf_counter()
        svc.search_memory(q, limit=5)
        cold.append((time.perf_counter() - t) * 1000)
    for _ in range(5):  # the same queries again (cache hits where a cache exists)
        for q in queries:
            t = time.perf_counter()
            svc.search_memory(q, limit=5)
            again.append((time.perf_counter() - t) * 1000)
    return {"cold_ms": summarize(cold), "repeat_ms": summarize(again)}


def main() -> None:
    mode = sys.argv[1]
    out: dict[str, Any] = {"mode": mode}
    with tempfile.TemporaryDirectory() as td:
        if mode == "real":
            src = Settings().db_path
            dst = Path(td) / "real.db"
            a = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            b = sqlite3.connect(dst)
            a.backup(b)
            a.close()
            b.close()
            queries = REAL_QUERIES
        else:
            import scale

            n = int(sys.argv[2])
            f = Path(td) / "e" / "conversations.json"
            f.parent.mkdir()
            f.write_text(json.dumps(scale.build(n)), encoding="utf-8")
            from membridge.importers.service import run_import
            from membridge.memory.pipeline import run_extraction

            dst = Path(td) / "synth.db"
            st = Store(dst)
            run_import(st, f)
            run_extraction(st)
            st.close()
            queries = [f"What does {scale.name(i * 7)} use for storage now?" for i in range(30)]
            out["conversations"] = n
        store = Store(dst)
        out["memories"] = store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        svc = MemoryService(store, None, actor="latency")
        svc.search_memory(queries[0], limit=5)  # open connections, import modules
        out.update(measure(svc, queries))
        store.close()
    print(json.dumps(out, indent=2))
    res = ROOT / "eval" / "results" / f"latency-{mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _ = shutil


if __name__ == "__main__":
    main()
