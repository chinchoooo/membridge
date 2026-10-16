# Evaluation (all numbers measured on this machine; none estimated)

Hardware: i9-13900H (20 threads), 16 GB RAM, no GPU, Windows 11, FalkorDB 4.22 in Docker. Reproduce with the scripts named
in each section; raw JSON lands in `eval/results/`. **Caveat:** the benchmarks are synthetic with gold answers by
construction (the real archive has no ground-truth labels), so they show *which mechanisms matter*, not production accuracy.

## 1. Retrieval ablation — `uv run python eval/ablation.py`
48 questions over 73 conversations (12 projects × 4 question kinds, plus 37 hub-distractor conversations where one tool,
"Python", links dozens of projects). Pass = all gold facts retrieved (and current where required).

| Variant | Pass | Recall@5 | MRR | nDCG@5 | p50 ms |
|---|---|---|---|---|---|
| A text only (BM25 over facts + messages) | 0.750 | 0.750 | 0.550 | 0.597 | 1.2 |
| B + entity/hop channels (sum fusion) | 0.917 | 0.917 | 0.667 | 0.730 | 1.8 |
| C + Personalized PageRank (sum fusion) | 1.000 | 1.000 | 0.826 | 0.869 | 5.8 |
| D RRF fusion instead of sum | 1.000 | 1.000 | 0.812 | 0.858 | 5.8 |
| E + MMR | 1.000 | 1.000 | 0.812 | 0.858 | 5.8 |
| F + exact knapsack packing (**full system**) | 1.000 | 1.000 | 0.812 | 0.858 | 8.3 |
| F′ full system via FalkorDB | 1.000 | 1.000 | 0.812 | 0.858 | 15.6 |

By question kind (pass rate) — text-only A: as-of 1.0, direct 1.0, history 1.0, **lexical-gap multi-hop 0.0**; full system: all 1.0.

What this shows, including the unflattering parts:
- The graph is the **only** thing that answers the 12 "lexical-gap" questions (the answer shares no words with the question; it is reached
  message → conversation → project → facts). Text-only gets 0/12; the full system 12/12.
- Entity channels alone (B) *hurt* history questions (0.667) until temporal intent and PPR were added: every mechanism was needed.
- **RRF was not better than score-sum fusion here** (MRR 0.812 vs 0.826; not significant at n=48). RRF is kept because it needs no score
  calibration when the optional vector channel is added.
- **MMR and knapsack show no measurable effect** at normal budgets on this data (no redundancy to remove; everything fits). At a
  350-token budget greedy and knapsack tie (both 0.75): only one memory fits, so the *history* kind fails (0.0) by arithmetic, not by ranking.
- Graph mode is slower than SQLite (15.6 vs 8.3 ms); results are identical. FalkorDB's value is capability (Cypher exploration, algorithms,
  indexes), not latency, at personal scale.
- "ctx/raw" ratios (≈10×) are meaningless on 1–3-sentence synthetic chats; context compression must be judged on real long conversations.

Bugs this benchmark found and that are fixed and covered by tests: duplicate must-include items, knapsack rounding, missing temporal
intent, hop channel ordered by fact score instead of conversation rank, entity extractor dropping "The Atlas", ABOUT edges collapsing roles.

## 2. Behaviour set — `uv run python eval/run.py` (9 questions on the demo archive)
Hybrid 9/9, SQLite-degraded 9/9, text-only baseline 8/9 (fails the cross-conversation hop). Provenance resolves for every returned memory.

## 3. Entity resolution — `tests/test_resolve.py` (labelled set: 47 names, 10 alias clusters + 23 singletons including hard negatives)
Pairwise **precision 1.000, recall 1.000** (18 true pairs, 0 false merges; hard negatives include Python 2/3, Java/JavaScript,
Atlas/Atlas Shrugged, React/React Native, GPT-3/GPT-4, k8s numeronym). Blocking recall 1.0 *when the alias is in the curated dictionary*;
abbreviations absent from it (no shared character shingles) depend on the optional embedding blocker or on user-added aliases.
Over 600 random names + 1 alias pair, blocking removes >97% of pair comparisons. Merges are reversible (state restored exactly; tested).

## 4. Semantic channel — `uv run --extra embeddings python eval/semantic.py`
12 paraphrase queries with no shared content words + 40 distractors. Text+graph: R@1 0.0, R@3 0.0. **With bge-small vectors:
R@1 0.583, R@3 0.667, MRR 0.611.** Embedding 52 records takes 0.10 s; the model uses ~210 MB RSS and loads in ~1 s. Embeddings are optional,
never auto-merge entities, and are rebuildable from text (`membridge embed --rebuild`).

## 5. Real archive (Claude export, 80 conversations / 2,062 messages) — `uv run python eval/resources.py <zip>`
Import 0.74 s at 59 MB peak RSS (streams the 47 MB JSON from an 8.9 MB zip); database 9.0 MB (1.01× the zip); consolidation 0.42 s;
FalkorDB container 97 MiB; query latency p50 4.8 ms graph / 2.0 ms SQLite. Real retrieval (with the 43 seeded provider memories + vectors):
~50 ms, ~800 tokens. Rule extraction found only 2 new memories in 945 user messages (real chat is not phrased "I decided to…").
Local `phi3` (3.8B, CPU) takes ~59 s per window; 9 of 12 proposed memories were rejected for non-verbatim quotes; survivors are labelled
`model_inferred` at 0.6 confidence and are plausible but sometimes over-generalise (a one-off task constraint recorded as permanent).
A full pass would take ~4 h; it is opt-in and resumable (`membridge extract-llm`). No precision/recall on real data is reported: it needs
labels only you can provide.

## 6. Speed and scale — `eval/latency.py`, `eval/scale.py`
Memory-call latency, in-process, wall-clock (the MCP/HTTP transport adds its own fixed cost). *Cold* = a query never seen before;
*repeat* = the same query again (result cache). The machine is shared with a browser, so single runs vary ±50%; ranges are honest.

| Store | Memories | Cold p50 (before → after) | Cold p95 | Repeat p50 (before → after) |
|---|---|---|---|---|
| **Your real archive** | 57 | 20.4 ms → **2.5 ms** | 3.1 ms | 17.8 ms → **0.021 ms** |
| 1,500 conversations | 3,928 | 137 ms → **5.6 ms** (8.7–22 ms in noisy runs) | 6.6 ms | 138 ms → **0.084 ms** |
| 5,000 conversations | 13,064 | 152 ms → **2.7 ms** | 6.3 ms | – → **0.069 ms** |

Where the time went and what removed it (each found by profiling, not guessing):
1. **Writes on the read path**: every search committed an audit row. Now buffered in memory, flushed in batches by a background thread.
2. **Result cache** keyed on a trigger-maintained *content generation*, so any write by any process invalidates it (repeat calls are ~20–80 µs).
3. **PPR linearity shortcut**: PPR(Σwᵢ·seedᵢ) = Σwᵢ·PPR(seedᵢ), so consolidation precomputes a sparse vector per entity and a query sums a few
   stored vectors (no graph load, no pushes). Cached and live PPR agree on the top answer ≥95% and on ≥90% of the top 5 (`tests/test_speed.py`);
   a stale cache transparently falls back to the live algorithm.
4. **Warm in-memory index** (entity/alias n-gram table, fact rows, per-entity fact lists, supersede chains) replaces ~40 SQL statements per query.
5. **IDF-pruned full-text** lookups: OR-queries over ubiquitous words cost 3–13 ms; pruning terms in >50 docs (never the rarest) cuts them ~25×.
6. **numpy knapsack** (exact, tested equal to the pure-Python DP) and one generation read per search instead of ~10.

The ablation (§1) is unchanged by all of this: the full system still passes 48/48 with MRR 0.812; its median latency fell from 8.3 to 1.9 ms.

Consolidation (import → rules → resolution → relations → communities → summaries → PPR cache):

| Conversations | Memories | Entities | Before | After | Peak RSS |
|---|---|---|---|---|---|
| 1,500 | 3,928 | 1,550 | 101.7 s | **8.5 s** | 81 MB |
| 3,000 | – | 3,050 | – | 23.5 s | – |
| 5,000 | 13,064 | 5,050 | 110.7 s | **44.3 s** | 160 MB |

Fixes: quadratic known-entity scan; table scans from `IFNULL(...)` in two upsert lookups (now null-safe `IS`, index-friendly); missing `memory_evidence(ref)` index
(15 s → 0.1 s); co-occurrence self-join materialised; PPR graph context computed once (cache build 8.8 s → 2.0 s). Growth is now ≈ n^1.3.
**Target missed:** I aimed for <20 s at 5,000 conversations and got 44 s (5,000 conversations is ~60× the real archive, which consolidates in under a second).

## 7. Tests
`uv run pytest` → 231 tests (unit, algorithm properties incl. PPR error bound vs exact power iteration and brute-force knapsack optimality,
real MCP over stdio in separate processes, live FalkorDB parity, API security guards, documented-Cypher execution). mypy `--strict` and ruff clean.

## 8. Known weaknesses
Rule extraction has low recall on casual chat (broadened rules still find 2 candidates in your 945 real messages: your messages are task prompts, so only the LLM tier can help); local-LLM extraction is slow and noisy;
blocking recall depends on aliases; tight token budgets can fit only one memory; consolidation missed its 20 s target at 5,000 conversations;
the live database is encrypted only while locked; Cursor/VS Code/opencode/ChatGPT connectors are unverified.

## 9. Extraction, import, encryption, packaging (new in v2.1)
- **Deterministic rules** now also catch "I'm using X", "I'm learning X", "my stack/editor/degree/… is X", "I work/study at X", "my name is X".
  On a labelled corpus: 10/10 statements caught and 0/10 chatter or questions falsely extracted (`tests/test_rules_corpus.py`).
  On your real messages the yield did not change (2 candidates): the archive simply contains few self-descriptions.
- **Hosted extraction** (`MEMBRIDGE_LLM_PROVIDER=claude_cli`, opt-in): Haiku via your logged-in Claude Code, all tools disabled, run in an empty temp directory,
  redacted first, JSON-schema output, verbatim-quote guard, and a guard rejecting any number the user never wrote. `extract-llm --dry-run` writes the exact payloads and
  sends nothing. Measured dry run on your archive: 89 calls / 866 messages / ~412k characters / estimated ≈ $4.18; the redactor matched nothing (no emails, keys or cards in the payload).
- **New importers**: Claude Code session transcripts and Markdown/text note folders (synthetic fixtures; idempotent; tool output is never stored as text).
- **Encryption**: `backup|export --encrypt` (scrypt + streaming AES-256-GCM; tampering, truncation and chunk reordering are rejected) and `lock`/`unlock`
  to keep the database encrypted whenever it is not in use. **SQLCipher was tried and is unavailable on Windows** (no wheel), so the live database is plaintext while unlocked.
- **Packaging**: MIT license; `Dockerfile` + `docker compose --profile app up` (built and run: healthy, non-root user, published on 127.0.0.1 only, bad `Host` rejected).
- **Independent client**: the official MCP Inspector CLI (TypeScript SDK) listed all 20 tools, wrote a decision in one process, read it in another, and had an unsafe call refused.
- **Real-data evaluation tooling** (`gold-init`, `eval-real`): pooled candidate answers for you to judge; refuses to report numbers until ≥10 questions are judged.
