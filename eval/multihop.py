"""Synthetic multi-hop benchmark generator (deterministic).

Each project has a fact chain spread over THREE conversations:
  conv1  "I'm starting a project called P. I decided to use T1 for P storage."
  conv2  a message containing a unique token (ZXn) that creates no fact, an assistant reply,
         then "I switched P from T1 to T2 because ..."   (the answer to the lexical-gap question)
  conv3  "The P prototype deadline moved to <date>."
Hub distractors: dozens of other projects all deciding to use "Python", so Python is a high-degree hub that a
naive graph walk drifts into. Questions have gold answers known by construction (nothing is invented).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_sample", ROOT / "samples" / "make_sample.py")
assert _spec and _spec.loader
_ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ms)

DAY = 86_400.0
NAMES = ["Orion", "Vega", "Lyra", "Draco", "Cygnus", "Aquila", "Hydra", "Pavo", "Lupus", "Carina", "Fornax", "Vela", "Mensa", "Norma"]
TECH = ["MySQL", "Redis", "MongoDB", "Cassandra", "DynamoDB", "Neo4j", "ClickHouse", "CockroachDB", "Firebird", "InfluxDB", "RocksDB", "Timescale"]


def build(n_projects: int = 12, n_distractors: int = 36) -> tuple[list[Any], list[dict[str, Any]]]:
    t0 = 1_700_000_000.0
    convs: list[Any] = []
    qs: list[dict[str, Any]] = []
    for i in range(n_projects):
        p, t1, t2 = NAMES[i], TECH[i], TECH[(i + 5) % len(TECH)]
        base = t0 + i * 3 * DAY
        convs.append(_ms.conv(f"mh-{i}-a", f"{p} kickoff", base, [
            ("user", f"I'm starting a project called {p}: an analytics service. I decided to use {t1} for {p} storage."),
            ("assistant", "Understood."),
        ]))  # fmt: skip
        convs.append(_ms.conv(f"mh-{i}-b", f"{p} trouble", base + 30 * DAY, [
            ("user", f"The nightly ZX{i} timeout keeps killing the {p} job and nobody knows why."),
            ("assistant", "That usually means the datastore cannot keep up; consider changing engines."),
            ("user", f"I switched {p} from {t1} to {t2} because the engine could not keep up."),
        ]))  # fmt: skip
        convs.append(_ms.conv(f"mh-{i}-c", f"{p} schedule", base + 60 * DAY, [
            ("user", f"The {p} prototype deadline moved to {5 + i} February."),
        ]))  # fmt: skip
        gold_new = f"switched {p} from {t1} to {t2}"
        gold_old = f"decided to use {t1} for {p} storage"
        qs += [
            {"id": f"mh{i}-direct", "kind": "direct", "q": f"What does {p} use for storage now?", "expected": [gold_new], "current": [gold_new]},
            {"id": f"mh{i}-gap", "kind": "lexical-gap-multihop", "q": f"What did I change after the ZX{i} timeout?", "expected": [gold_new], "current": [gold_new]},
            {"id": f"mh{i}-history", "kind": "history", "q": f"Which database did {p} use before {t2}?", "expected": [gold_old]},
            {"id": f"mh{i}-asof", "kind": "as-of", "q": f"{p} storage database", "expected": [gold_old], "forbid": [gold_new],
             "filters": {"as_of": base + 15 * DAY}},
        ]  # fmt: skip
    for j in range(n_distractors):  # hubs: many unrelated projects all using Python
        q = f"Proj{j}x"
        convs.append(_ms.conv(f"hub-{j}", f"{q} scripts", t0 + (j + 3) * DAY, [
            ("user", f"I'm starting a project called {q}: some tooling. I decided to use Python for {q} scripting."),
        ]))  # fmt: skip
    convs.append(_ms.conv("hub-pref", "Languages", t0 + 40 * DAY, [("user", "I prefer Python for everything I build.")]))
    return convs, qs
