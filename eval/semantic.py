"""Paraphrase benchmark: queries that share (almost) no content words with the fact that answers them.

Measures what the optional embedding channel adds on top of full-text + graph retrieval. Gold is by construction.
Run: uv run --extra embeddings python eval/semantic.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from membridge.embed.embedder import build_embedder  # noqa: E402
from membridge.embed.index import index_embeddings  # noqa: E402
from membridge.memory.consolidate import upsert  # noqa: E402
from membridge.memory.models import Candidate  # noqa: E402
from membridge.retrieval.engine import RetrievalConfig, Retriever  # noqa: E402
from membridge.store.sqlite import Store  # noqa: E402

PAIRS = [
    ("I prefer dark mode in every editor.", "Which colour theme do I like for my IDE?"),
    ("My flight to Berlin leaves at 7am on Friday.", "When do I depart for Germany?"),
    ("I work remotely from Lisbon on weekdays.", "Where is my usual place of employment?"),
    ("I use Neovim as my primary text editor.", "What do I write code with?"),
    ("I switched our CI from Jenkins to GitHub Actions.", "How are my builds automated now?"),
    ("The budget for the Q3 campaign is 40k euros.", "How much money is allocated for the third-quarter marketing push?"),
    ("I train for a marathon every Sunday morning.", "What athletic activity do I do on weekends?"),
    ("I live with two cats named Pixel and Byte.", "Which pets do I have?"),
    ("My manager is Priya Raman.", "Who do I report to?"),
    ("I speak Spanish and Portuguese fluently.", "Which languages am I good at?"),
    ("The staging database runs on PostgreSQL 15.", "What engine backs the pre-production datastore?"),
    ("I drink oat milk flat whites.", "What kind of coffee order do I have?"),
]  # fmt: skip
DISTRACTORS = [
    "The quarterly report is due next month.", "I bought a new keyboard last week.", "The garden needs watering twice a week.",
    "My laptop battery lasts about six hours.", "The meeting room on the third floor is booked.", "I read a book about Roman history.",
    "The invoice was paid on the fifteenth.", "Our team uses a shared calendar for planning.", "The train to the city takes forty minutes.",
    "I prefer window seats on long journeys.", "The backup runs every night at two.", "I keep my passwords in a manager.",
    "The office printer is out of toner.", "I usually walk to the market on Saturdays.", "The conference talk was about compilers.",
    "My phone plan renews in March.", "The API rate limit is one hundred per minute.", "I plan to repaint the kitchen soon.",
    "The review process takes two rounds.", "I take the stairs instead of the lift.", "Our release cadence is every second Tuesday.",
    "The cafeteria closes at three.", "I subscribe to a weekly newsletter on design.", "The old server was retired last year.",
    "I carry a notebook in my bag.", "The deadline for the grant is in April.", "My monitor is mounted on an arm.",
    "The standup lasts fifteen minutes.", "I stream music while I work.", "The warehouse ships orders before noon.",
    "I joined a book club in the spring.", "The firewall blocks outbound mail.", "Our logo uses two shades of green.",
    "I keep a spare charger at the office.", "The lease on the apartment ends in June.", "The dashboard refreshes every minute.",
    "I enjoy hiking in the autumn.", "The vendor sent a revised quote.", "My desk faces the east window.", "The audit found no issues.",
]  # fmt: skip


def main() -> None:
    emb = build_embedder("fastembed")
    assert emb is not None
    with tempfile.TemporaryDirectory() as td:
        store = Store(Path(td) / "sem.db")
        ts = 1_000_000.0
        for i, (fact, _) in enumerate(PAIRS):
            upsert(store, Candidate(statement=fact, category="fact", ref=f"g{i}", ts=ts + i, verification="user_stated"))
        for j, d in enumerate(DISTRACTORS):
            upsert(store, Candidate(statement=d, category="fact", ref=f"d{j}", ts=ts + 100 + j, verification="user_stated"))
        t0 = time.perf_counter()
        stats = index_embeddings(store, emb)
        index_s = time.perf_counter() - t0
        rows = {}
        for label, cfg in (("lexical + graph (no vectors)", RetrievalConfig()), ("+ vector channel (bge-small)", replace(RetrievalConfig(), use_vector=True))):
            r = Retriever(store, None, cfg, emb if cfg.use_vector else None)
            hit1 = hit3 = 0
            rr = 0.0
            for fact, q in PAIRS:
                mems = [m["statement"] for m in r.search(q, limit=3)["memories"]]
                hit1 += bool(mems and mems[0] == fact)
                hit3 += fact in mems
                rr += next((1 / (i + 1) for i, m in enumerate(mems) if m == fact), 0.0)
            n = len(PAIRS)
            rows[label] = {"recall@1": round(hit1 / n, 3), "recall@3": round(hit3 / n, 3), "mrr": round(rr / n, 3)}
        print(f"corpus: {len(PAIRS)} gold facts + {len(DISTRACTORS)} distractors; embedded {stats['embedded']} records in {index_s:.2f}s with {emb.name}")
        print(f"{'variant':36} {'R@1':>6} {'R@3':>6} {'MRR':>6}")
        for k, v in rows.items():
            print(f"{k:36} {v['recall@1']:>6} {v['recall@3']:>6} {v['mrr']:>6}")
        store.close()
        out = ROOT / "eval" / "results" / f"semantic-{time.strftime('%Y%m%d-%H%M%S')}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"model": emb.name, "index_seconds": round(index_s, 2), "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
