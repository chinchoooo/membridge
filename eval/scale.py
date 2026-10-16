"""Scale test: a synthetic archive thousands of conversations large (deterministic), to check the algorithms'
claimed scaling (LSH blocking is sub-quadratic, PPR cost is independent of total graph size).

Run: uv run python eval/scale.py [n_conversations]   (default 3000)
"""

from __future__ import annotations

import importlib.util
import json
import random
import statistics
import sys
import tempfile
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

_spec = importlib.util.spec_from_file_location("make_sample", ROOT / "samples" / "make_sample.py")
assert _spec and _spec.loader
_ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ms)

TECH = ["MySQL", "Redis", "MongoDB", "Cassandra", "DynamoDB", "Neo4j", "ClickHouse", "CockroachDB", "Firebird", "InfluxDB", "RocksDB", "Timescale",
        "Kafka", "RabbitMQ", "Elasticsearch", "Postgres", "SQLite", "DuckDB", "Pulsar", "NATS", "Memcached", "Etcd", "Consul", "Vault", "Nomad"]  # fmt: skip
SYL = ["ka", "to", "mi", "re", "lu", "va", "no", "se", "di", "pa", "xo", "ru", "te", "gi", "ha", "bo"]


def name(i: int) -> str:
    return "".join(SYL[(i // 16**k) % 16] for k in range(3)).capitalize() + str(i % 7)


def build(n: int, seed: int = 5) -> list[Any]:
    rnd = random.Random(seed)
    convs = []
    t0 = 1_600_000_000.0
    for i in range(n):
        p = name(i)
        a, b = rnd.sample(TECH, 2)
        turns: list[tuple[str, str]] = [("user", f"I'm starting a project called {p}: a service number {i}. I decided to use {a} for {p} storage.")]
        if rnd.random() < 0.6:
            turns.append(("user", f"I switched {p} from {a} to {b} because the load grew."))
        if rnd.random() < 0.3:
            turns.append(("user", f"I prefer {rnd.choice(TECH)} for caching."))
        convs.append(_ms.conv(f"s-{i}", f"{p} notes", t0 + i * 3600, turns))
    return convs


def main(n: int) -> None:
    out: dict[str, Any] = {"conversations": n}
    proc = psutil.Process()
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "e" / "conversations.json"
        f.parent.mkdir()
        t = time.perf_counter()
        f.write_text(json.dumps(build(n)), encoding="utf-8")
        out["generated_mb"] = round(f.stat().st_size / 2**20, 1)
        store = Store(Path(td) / "scale.db")
        t = time.perf_counter()
        m = run_import(store, f)
        out["import_s"] = round(time.perf_counter() - t, 2)
        out["messages"] = m.messages_new
        stages: dict[str, float] = {}
        t = time.perf_counter()
        st = run_extraction(store)
        stages["extract_resolve_relations_communities_s"] = round(time.perf_counter() - t, 2)
        out["stages"] = stages
        out["memories"] = store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        out["entities"] = store.conn.execute("SELECT COUNT(*) FROM live_entities").fetchone()[0]
        out["entities_merged"] = st.entities_merged
        ne = store.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        pairs_cap = ne * (ne - 1) // 2
        out["all_pairs_if_naive"] = pairs_cap
        out["communities"] = store.conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0]
        out["db_mb"] = round(store.path.stat().st_size / 2**20, 1)
        out["peak_rss_mb"] = round(proc.memory_info().rss / 2**20)
        g = GraphStore(name="membridge_scale")
        if g.available():
            t = time.perf_counter()
            g.rebuild(store)
            out["graph_rebuild_s"] = round(time.perf_counter() - t, 2)
            out["graph"] = g.counts()
        rnd = random.Random(9)
        qs = [f"What does {name(rnd.randrange(n))} use for storage now?" for _ in range(60)]
        for label, graph in (("sqlite", None), ("graph", g if g.available() else None)):
            if label == "graph" and graph is None:
                continue
            r = Retriever(store, graph)
            lat, ok = [], 0
            for q in qs:
                pkg = r.search(q, limit=5)
                lat.append(pkg["stats"]["latency_ms"])
                ok += bool(pkg["memories"]) and name_in(q) in pkg["memories"][0]["statement"]
            lat.sort()
            out[f"query_{label}"] = {"p50_ms": round(statistics.median(lat), 1), "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1), "top1_mentions_project": f"{ok}/{len(qs)}"}
        g.drop()
        store.close()
    print(json.dumps(out, indent=2))
    res = ROOT / "eval" / "results" / f"scale-{n}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(out, indent=2), encoding="utf-8")


def name_in(q: str) -> str:
    return q.split("does ")[1].split(" use")[0]


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
