# MemBridge — Architecture Decision Record + Milestone Plan

## Context
User wants MemBridge (spec: `C:\Users\aravi\mem\MEMBRIDGE_SPEC.md`): a local-first personal AI memory system for the Graph Hacks hackathon — imports ChatGPT exports, builds provenance-linked memories + a FalkorDB graph, exposes them via MCP to multiple AI clients, with a dashboard. Repo `C:\Users\aravi\mem` is empty except the spec (Phase 0 done).

**Environment (Phase 0):** Win 11, i9-13900H 14C/20T, 15.7 GB RAM, 178 GB free, no GPU needed. Python 3.13.1 + uv 0.7.4, Node 22.14, Docker 29.3 (daemon NOT running), Ollama 0.9.0 (old; has llama3, phi3, llava), MCP-capable clients installed: Claude Code, Cursor, VS Code, opencode. No ChatGPT export found on disk.

## ADR summary (recommended choices)

| Concern | Decision | Rejected / why |
|---|---|---|
| System of record | **SQLite** (WAL): raw conversations/messages (content-hashed IDs), memories + full revision history, import jobs/checkpoints, audit log. FTS5 over messages. | Raw archive in FalkorDB: worse for bulk text, backup, transactions. |
| Graph + memory search | **FalkorDB** (Docker `falkordb/falkordb`, port bound to 127.0.0.1, named volume). Holds Memory, Entity (Project/Person/Org/Concept/Tool), Decision, Conversation, cited Message nodes + typed edges. Uses its built-in **full-text index** and **vector index** — no extra DBs. | Neo4j (heavier, not hackathon target); Chroma/Qdrant (unnecessary — FalkorDB has HNSW vector index). |
| Graph role | Projection of SQLite canonical records, **rebuildable** (`membridge graph rebuild`). Retrieval = FalkorDB full-text seed → Cypher 1–2 hop expansion (entity → related memories/decisions, SUPERSEDES chains, SUPPORTED_BY → messages). Graph is in the critical path, not decoration. Degraded mode: SQLite FTS only if FalkorDB is down. | |
| LLM | Provider interface: `none` (deterministic only) / `ollama` (default local) / `openai-compatible` + `anthropic` hosted (opt-in, redaction pass first). Default local model: **qwen3.5:4b** (fallback gemma4:e4b) via Ollama with JSON-schema-constrained output; thinking off. Pick final default by a bake-off on sample data (M5), not by claims. | Large local models (24B+) too slow on CPU/16 GB; hosted-by-default violates privacy requirement. |
| Deterministic vs LLM | Deterministic: parsing, hashing, segmentation, explicit-statement patterns ("I prefer/decided/switched to/use X"), alias normalisation, exact/near-dup (normalised text + trigram), supersede rules on same (subject, predicate) key, scoring. LLM: open-ended fact/entity/relation extraction, conversation & project summaries, contradiction classification of candidate pairs (changed preference / correction / temporary / unresolved). | |
| Embeddings | Optional, off for MVP slice. When on: **fastembed bge-small-en-v1.5** (384-d ONNX, CPU, ~130 MB), stored in FalkorDB vector index with model-version tag; rebuildable. | sentence-transformers (pulls torch, heavy). |
| MCP | Official **`mcp` Python SDK 2.x** (`MCPServer`; verify API at install). Transports: stdio (Claude Code, Cursor, VS Code, opencode) + optional Streamable HTTP on 127.0.0.1 with bearer token; remote via user's tunnel (Tailscale/Cloudflare) documented, not default. | FastMCP 4 standalone: extra dependency for no MVP need. |
| API/UI | FastAPI backend (same Python package) + **React/TS (Vite)** dashboard, **Cytoscape.js** for neighbourhood graph views (never whole graph). | |
| Deploy | Modular monolith: `membridge` Python package (uv), CLI (`typer`), FalkorDB via `docker compose`. No broker/k8s. | |

**Resources (est.):** FalkorDB ~100–300 MB RAM for a personal archive; app ~200 MB; Ollama 4B Q4 ~3–4 GB RAM, ~10–20 tok/s CPU → full LLM extraction of a large archive (1k+ convs) takes hours, so it runs as a resumable background job, prioritising user-authored messages; deterministic pass completes in minutes. Running cost: $0 local; hosted mode only if opted in.

**Security / threat model:** imported text is untrusted data — never interpolated into Cypher (parameters only), extraction prompts fence content and validate output against Pydantic schemas, extracted text cannot trigger tools. MCP tools are fixed operations with input limits; no raw Cypher/SQL/shell/filesystem. `forget_memory` and bulk ops require a two-step confirm token; deletes are soft with tombstone + audit. Audit log for reads/writes. Sensitivity tag (normal/sensitive/secret); sensitive excluded from hosted inference and from MCP responses unless allowed by config. Secrets via env/`.env` (gitignored). DB ports bound to localhost only.

**Failure modes:** FalkorDB down → degraded SQLite retrieval + `graph rebuild` on recovery; LLM unavailable → deterministic extraction only, job resumes later; interrupted import → checkpoint resume; malformed records → error report, import continues.

## Repo layout
```
mem/
  pyproject.toml, docker-compose.yml, .env.example, README.md
  docs/ADR.md, docs/threat-model.md, docs/graph-schema.md
  src/membridge/
    config.py          # pydantic-settings
    store/sqlite.py    # schema, migrations, canonical CRUD, FTS5, audit
    store/graph.py     # FalkorDB projection, indexes, Cypher queries
    importers/base.py, importers/chatgpt.py   # adapter interface + ChatGPT (ijson streaming)
    extract/rules.py, extract/llm.py, extract/pipeline.py
    consolidate/dedup.py, consolidate/contradiction.py, consolidate/summarise.py
    llm/base.py, llm/ollama.py, llm/openai_compat.py, llm/redact.py
    retrieval/engine.py, retrieval/budget.py
    mcp_server.py      # MCP tools
    api/app.py         # FastAPI
    cli.py
  web/                 # Vite React TS dashboard
  tests/  eval/  samples/ (synthetic, no private data)
```

## Milestones (each ends with tests run + report)
1. **M1 Core + importer:** package skeleton, config, SQLite schema, ChatGPT adapter (mapping tree → current branch + branch metadata, non-text parts recorded not invented), content-hash IDs, dry-run, manifest/error report, checkpoint resume, idempotent re-import. Synthetic sample export. Tests.
2. **M2 Memory + graph:** start FalkorDB via compose; deterministic extraction (rules), memory model (category, provenance, confidence, method, validity, sensitivity, verification, user-stated vs inferred), revisions + SUPERSEDES/CONTRADICTS, dedup, graph projection + indexes + rebuild. Tests incl. duplicate import → no duplicate memories.
3. **M3 Retrieval:** FTS seed + graph expansion + ranking (explicitness, recency, confirmation, source reliability, confidence) + token budget + explanation (paths) + "insufficient evidence". Eval set with ground-truth Q&A on sample data; precision/recall + latency script.
4. **M4 MCP server:** spec tools (retrieval/persistence/maintenance), confirm-token for destructive ops, idempotent writes. Verify for real: Claude Code (client 1) saves a decision; **Cursor or opencode** (client 2) retrieves it. Config examples for others.
5. **M5 LLM layer:** provider interface, Ollama (upgrade Ollama, pull qwen3.5:4b + gemma4:e4b), extraction/summaries/contradiction classification, redaction for hosted; bake-off on sample set → pick default. Graceful no-LLM mode.
6. **M6 Dashboard:** overview, memory explorer (edit/delete/verify), graph explorer, maintenance (import, consolidation, contradictions, export/backup), retrieval tester (paths, evidence, latency, tokens).
7. **M7 Ops + eval:** backup/restore (SQLite file + FalkorDB dump), export (JSON/Markdown), optional embeddings, restart-persistence test, prompt-injection tests, resource measurement, eval report.
8. **M8 Submission:** README, ADR, schema + example Cypher, demo script, video plan, limitations.

## User prerequisites (needed before M2 / M5)
- Start Docker Desktop (needed from M2).
- Update Ollama (0.9.0 too old for current models) — M5.
- Real ChatGPT export ZIP path — for the real-data demo; development uses synthetic samples until then.

## Verification
- `uv run pytest` per milestone (importer idempotency, provenance round-trip, supersede logic, retrieval eval, MCP tool contract via in-process MCP client, injection fixtures).
- `uv run membridge import samples/chatgpt_export.zip --dry-run` then real import twice → identical counts.
- `docker compose restart` → data persists; `membridge stats` unchanged.
- Cross-client: decision saved from Claude Code, retrieved from second client against the same DB (verified, not simulated).
- `eval/run.py` prints measured precision/recall/latency — no invented numbers.
