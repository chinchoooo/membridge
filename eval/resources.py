"""Resource and scale measurements on a real export (in a throwaway database; nothing is written to your data dir).

Run: uv run python eval/resources.py <path to conversations zip/folder/json>
Reports wall time, peak RSS, storage growth, per-stage timings, FalkorDB memory and query latency. Measured, not estimated.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from membridge.importers.service import run_import  # noqa: E402
from membridge.memory.pipeline import run_extraction  # noqa: E402
from membridge.retrieval.engine import Retriever  # noqa: E402
from membridge.store.graph import GraphStore  # noqa: E402
from membridge.store.sqlite import Store  # noqa: E402

QUERIES = [
    "What projects am I working on?", "What did I decide about the database?", "Which tools do I prefer?", "What is my background?",
    "What went wrong with the deployment?", "Which framework did I choose?", "What are my current constraints?", "Tell me about my side project",
    "What programming languages do I use?", "What did I learn recently?",
]  # fmt: skip


class PeakRSS:
    def __init__(self) -> None:
        self.p, self.peak, self._stop = psutil.Process(), 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.p.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self) -> PeakRSS:
        self._t.start()
        return self

    def __exit__(self, *a: object) -> None:
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.p.memory_info().rss)


def timed(fn: Any) -> tuple[Any, float, float]:
    with PeakRSS() as pk:
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
    return out, dt, pk.peak / 2**20


def docker_mem(name: str = "membridge-falkordb") -> str | None:
    try:
        r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}} / cpu {{.CPUPerc}}", name], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or None
    except Exception:
        return None


def main(src: str) -> None:
    src_path = Path(src)
    src_bytes = sum(f.stat().st_size for f in (src_path.rglob("*") if src_path.is_dir() else [src_path]) if f.is_file())
    out: dict[str, Any] = {"source_mb": round(src_bytes / 2**20, 1), "cpu_threads": psutil.cpu_count(), "ram_total_mb": round(psutil.virtual_memory().total / 2**20)}
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "res.db")
        m, dt, rss = timed(lambda: run_import(store, src_path))
        out["import"] = {"seconds": round(dt, 2), "peak_rss_mb": round(rss), "conversations": m.conversations_new, "messages": m.messages_new, "failures": len(m.failures)}
        db1 = store.path.stat().st_size
        st, dt, rss = timed(lambda: run_extraction(store))
        db2 = store.path.stat().st_size
        out["extraction_and_consolidation"] = {"seconds": round(dt, 2), "peak_rss_mb": round(rss), "memories_created": st.created, "entities_merged": st.entities_merged}
        out["storage_mb"] = {"after_import": round(db1 / 2**20, 1), "after_consolidation": round(db2 / 2**20, 1), "ratio_db_to_source": round(db2 / src_bytes, 2)}
        counts = {t: store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("memories", "entities", "communities", "summaries")}
        out["counts"] = {**store.stats(), **counts}
        g = GraphStore(name="membridge_resources")
        if g.available():
            _, dt, rss = timed(lambda: g.rebuild(store))
            out["graph_rebuild"] = {"seconds": round(dt, 2), "peak_rss_mb": round(rss), **g.counts(), "falkordb_container": docker_mem()}
        for label, graph in (("graph_mode", g if g.available() else None), ("sqlite_degraded", None)):
            r = Retriever(store, graph)
            lat = []
            for q in QUERIES * 3:
                lat.append(r.search(q, limit=5)["stats"]["latency_ms"])
            lat.sort()
            out[f"query_latency_ms_{label}"] = {"n": len(lat), "p50": round(statistics.median(lat), 1), "p95": round(lat[int(0.95 * (len(lat) - 1))], 1), "max": round(lat[-1], 1)}
        g.drop()
        store.close()
    out["process_idle_rss_mb"] = round(psutil.Process().memory_info().rss / 2**20)
    print(json.dumps(out, indent=2))
    res = ROOT / "eval" / "results" / f"resources-{time.strftime('%Y%m%d-%H%M%S')}.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
