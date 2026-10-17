# Architecture decision record (final)

Status: implemented and measured. Numbers refer to `docs/EVAL.md`. Spec: `MEMBRIDGE_SPEC.md`.

## Decisions

| Concern | Decision | Alternatives rejected, and why |
|---|---|---|
| System of record | **SQLite (WAL)**: raw archive (content-hashed ids), facts with full revision history, aliases/merge log, jobs, audit. FTS5 for messages and facts. | Putting the archive in the graph DB: poor bulk text/backup/transactions. A second database per capability: unjustified for a single-user store. |
| Graph | **FalkorDB 4.22** (Docker, `127.0.0.1` only) as a **rebuildable projection**: bounded neighbourhoods, Cypher path explanation, full-text + vector indexes, global algorithms (`pageRank`, `WCC`). | Neo4j (heavier; not the target). Making the graph the source of truth (a rebuild would lose data). |
| Retrieval math | **In Python on an in-memory graph** loaded from FalkorDB *or* SQLite: forward-push Personalized PageRank (HippoRAG-style, hub-damped), RRF, MMR, exact knapsack. FalkorDB has no personalized PageRank, and putting the math in Python makes graph mode and degraded mode return identical results (tested). | Delegating to `algo.pageRank` (global, not personalised); a vector DB (would hide structure; FalkorDB's own HNSW is mirrored optionally). |
| Temporal model | **Bi-temporal facts** (valid vs recorded time), invalidate-never-delete, corrections retract (Zep/Graphiti). | Overwriting or deleting old facts (spec forbids; destroys history). |
| Entity resolution | Block (exact/alias dictionary/**MinHash-LSH**/rare tokens/optional embeddings) → score (fuzzy + containment + context) → three tiers → **guarded union-find** (cannot-link, size cap) → **reversible** merges + human review queue. | Name-string identity (fragments the graph); aggressive transitive closure (over-merge). |
| Extraction | Deterministic rules first (cheap, exact, provenance-perfect); optional LLM with schema-constrained output, **verbatim-quote guard**, fenced untrusted input, no tools. | LLM-only (cost, privacy, injection surface); regex-only (measured recall: 2 memories from 945 real messages). |
| LLM | Provider interface: `none` (default) → local **Ollama** → opt-in hosted OpenAI-compatible with redaction. Local default `phi3` was measured, not assumed: usable but slow and noisy on CPU (see EVAL §5). A bake-off against other local models was **not** run (RAM/time); only `phi3` was measured on real data. | Hosted-by-default (violates privacy requirement); large local models (too slow on CPU). |
| Embeddings | **Optional** (`fastembed` bge-small, ONNX, CPU): +0.58 R@1 on paraphrase queries where text+graph scores 0. Never auto-merges entities; rebuildable. | Embeddings as the core (cannot give history, provenance or multi-hop structure; measured). |
| Communities | **Leiden** (igraph/leidenalg) when installed, else NetworkX Louvain; stable hash ids so only changed communities are re-summarised; summaries keep links to leaf facts. | One global summary (no drill-down); LLM-only summaries (fidelity gate required). |
| MCP | Official `mcp` SDK 2.x (`MCPServer`), stdio primary, loopback HTTP + bearer token optional. Two-step destructive ops, tool annotations, audit identity. | FastMCP standalone (extra dependency, no need). |
| UI | FastAPI + React/TypeScript/Vite; Cytoscape lazy-loaded; localhost-hardened. | A hosted dashboard (violates local-first). |
| Speed | Content-generation counter maintained by SQL triggers (any process invalidates) → result cache; warm in-memory index; **precomputed per-entity PPR vectors summed at query time (PPR is linear in its seeds)** with live fallback; IDF-pruned FTS; numpy knapsack; no writes on the read path. Measured: 2.5 ms cold / 21 µs repeat on the real archive. | A separate cache server or vector DB (a process boundary costs more than the work); precomputing per-query results (unbounded key space). |
| Packaging | Modular monolith (`uv` Python package); `docker compose` runs FalkorDB, optionally the app. | Microservices/broker/k8s (no demonstrated need). |
| Who vouches | **Trust order** `user_verified > user_stated > client_asserted > provider_stated > model_inferred` decides conflicts before time does; undated claims are never "newer"; equal-trust undated clashes become *unresolved* and the person's decision is remembered. Agents cannot mint the top two tiers. | "Newest wins" (a summary with no date overruled things the user dated); first-writer-wins verification (a fact the user later states stayed `provider_stated`). |
| Private data | One rule, applied by **construction**: the default search index, PageRank graph/cache, derived edges, summaries, graph projection, embeddings and exports simply contain no private memory; a separate uncached view exists only under a grant. Raw messages are classified too. | A filter at the end of each query path (the review found eight ways around it). |
| Consent | PIN (scrypt, rate-limited) typed at a hidden prompt/dashboard creates a **short, client-bound, use-counted grant**; every private access is audited synchronously; audit rows are hash-chained. | Letting an agent pass a flag or a token typed in chat (it would pass through the model and could be replayed); a bearer secret the agent can read from its own environment. |
| Answer or abstain | A memory is returned only with **per-item support for this query**: lexical (IDF-weighted, must contain the most informative term), entity-anchored, message-anchored (a near-unique token or two rare words), or vector. Structured memories carry the words their predicate answers (company/city/tools). | A score threshold on a self-normalised score (it could never fail: asking for a pizza topping returned "favourite editor"). |

## Data flow
`export zip` → streaming adapter (ChatGPT/Claude) → raw archive (SQLite) → rules (+ optional LLM) → consolidation (dedup, supersede/correct,
ACT-R activation + importance scoring, entity resolution, typed relations, communities, summaries) → graph projection (+ optional embeddings)
→ retrieval (channels → PPR → fusion → MMR → packing) → MCP/API/UI. Clients write back via `record_decision` / `save_memory` / `commit_session_summary`.

## Resources (measured)
Import 0.74 s / 59 MB RSS for a 47 MB JSON export; DB ≈ source size; FalkorDB ≈ 100 MB; optional embedding model ≈ 210 MB; CPU only.
Local LLM extraction ≈ 59 s per window on `phi3`. Running cost: $0 (hosted inference only if you opt in).

## Failure modes
FalkorDB down → identical answers via SQLite loader (tested), slower graph features. LLM down → rules only. Interrupted import/extraction → resume.
Bad records → reported, never fatal. Wrong merge → undo (`entity-review`). Wrong fact → edit/forget; history preserved.

## Migration and deployment
Schema versions 1–7 are additive and idempotent (`store/migrations.py`; migration 7 adds trust/time flags, raw-text sensitivity, grants, the PIN
store and the audit hash chain, and re-classifies existing data, only ever upgrading to private). Backup/restore and export need no AI provider.
`docker compose up -d` then `uv run membridge …`; `docker compose --profile app up -d` runs the dashboard (token required).
