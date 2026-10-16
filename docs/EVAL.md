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

## 6. Scale — `uv run python eval/scale.py N`
| Conversations | Memories | Entities | Consolidation | Query p50 (SQLite) | Top-1 correct | Peak RSS |
|---|---|---|---|---|---|---|
| 1,500 (before fixes) | 3,928 | 1,550 | 101.7 s | 123 ms | 60/60 | 80 MB |
| 1,500 (after) | 3,928 | 1,550 | **8.5 s** | **50 ms** | 60/60 | 81 MB |
| 5,000 (after) | 13,064 | 5,050 | 110.7 s | 152 ms | 60/60 | 89 MB |

Fixes found by profiling: a quadratic known-entity scan (4 of 7.7 s), a missing `memory_evidence(ref)` index (15 s → 0.1 s for summaries),
and the co-occurrence self-join recomputed per query (65 of 98 ms; now materialised). Growth is still super-linear (≈n^1.7); 5,000
conversations is ~60× the real archive, so we stopped there. The dashboard's initial bundle is 170 kB (graph library lazy-loaded).

## 7. Tests
`uv run pytest` → 174 tests (unit, algorithm properties incl. PPR error bound vs exact power iteration and brute-force knapsack optimality,
real MCP over stdio in separate processes, live FalkorDB parity, API security guards, documented-Cypher execution). mypy `--strict` and ruff clean.

## 8. Known weaknesses
Rule extraction has low recall on conversational text; LLM extraction is slow on CPU and noisy; blocking recall depends on aliases;
tight token budgets can fit only one memory; not encrypted at rest; Cursor/VS Code/opencode/ChatGPT connectors unverified.
