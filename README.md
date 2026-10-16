# MemBridge

A **local-first personal AI memory**: import your ChatGPT / Claude history once, keep a provenance-linked, bi-temporal knowledge graph on
your own machine, and let any MCP-capable assistant read and update it. No GPU, no mandatory cloud, no dependence on any provider's memory feature.

```
export zip ─▶ raw archive (SQLite) ─▶ memories + entities (rules, optional local LLM)
                  │                         │  dedup · supersede/correct · entity resolution · communities
                  ▼                         ▼
             exact messages          FalkorDB graph (rebuildable) ─▶ retrieval (BM25 + graph + PPR + RRF + MMR + packing)
                                                                          ├─ MCP server (stdio / loopback HTTP)
                                                                          └─ dashboard + HTTP API (localhost)
```

What makes the graph matter: on a 48-question multi-hop benchmark, text-only retrieval fails every question whose answer shares no words
with the query (0/12); the graph-based system answers all of them. See [`docs/EVAL.md`](docs/EVAL.md) for every measured number and the
weaknesses we found.

## Quick start (Windows/macOS/Linux)

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker, Node 20+ (dashboard only).

```bash
uv sync                                   # Python deps
docker compose up -d                      # FalkorDB on 127.0.0.1:6390
uv run membridge import <export.zip|folder|conversations.json> --dry-run
uv run membridge import <export>          # ChatGPT or Claude export, auto-detected; resumable, idempotent
uv run membridge seed memories-000.zip projects-000.zip   # optional: Claude's own memory/project files
uv run membridge extract                  # rules + entity resolution + relations + communities + summaries
uv run membridge graph-rebuild
uv run membridge ask "What did I decide about the database?"
cd web && npm install && npm run build && cd ..
uv run membridge serve                    # dashboard at http://127.0.0.1:8787
uv run membridge mcp                      # MCP server over stdio (see docs/CLIENTS.md)
```

Optional extras: `uv sync --extra embeddings` (semantic channel: `membridge embed`), `--extra leiden` (Leiden communities).
Optional local LLM extraction: install Ollama, `MEMBRIDGE_LLM_PROVIDER=ollama MEMBRIDGE_LLM_MODEL=phi3 uv run membridge extract-llm`
(slow on CPU, resumable, labelled `model_inferred`). Hosted inference requires `MEMBRIDGE_ALLOW_HOSTED=true` and redacts first.

Run the whole demo: `uv run python scripts/demo.py` ([`docs/DEMO.md`](docs/DEMO.md)). Tests: `uv run pytest` (tests that need FalkorDB skip automatically when it is not running).

## Commands
`import · seed · extract · extract-llm · classify · summarize-llm · resolve · entity-review · communities · embed · graph-rebuild · ask · search · stats ·
backup · restore · export · health · serve · mcp`

## Configuration (env vars, prefix `MEMBRIDGE_`, or `.env`)
`DATA_DIR` (default `~/.membridge`), `FALKOR_HOST/PORT/GRAPH` (127.0.0.1 / 6390 / membridge), `LLM_PROVIDER` (none|ollama|openai_compat),
`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY_ENV` (name of the variable holding a key), `ALLOW_HOSTED`, `REDACT_HOSTED`, `EMBEDDER` (none|fastembed|hashing),
`MCP_TOKEN` (HTTP transport), `CLIENT_NAME` (audit identity). Keys are never stored in files.

## Documentation
[Architecture decisions](docs/ADR.md) · [Graph schema + Cypher](docs/graph-schema.md) · [Threat model](docs/threat-model.md) ·
[Connecting clients](docs/CLIENTS.md) · [Evaluation](docs/EVAL.md) · [Demo plan](docs/DEMO.md) · [Original spec](MEMBRIDGE_SPEC.md)

## Troubleshooting
- *FalkorDB not reachable*: `docker compose up -d`; everything still works in degraded SQLite mode (same answers, no graph UI).
- *Port 6390 busy*: set `MEMBRIDGE_FALKOR_PORT`. *No dashboard*: build it (`cd web && npm run build`).
- *Import says already_imported*: the exact archive was imported; use `--force` to re-scan.
- *Ollama slow/timeouts*: lower `--window-chars`; it resumes where it stopped.

## Limitations (read these)
- Rule extraction is conservative; real conversational data needs the optional LLM tier, which is slow and noisy on CPU.
- Benchmarks are synthetic; no precision/recall is claimed on real data (that needs labels only the owner can supply).
- Only Claude Code and a generic MCP SDK client are verified; Cursor/VS Code/opencode configs are unverified; hosted/ChatGPT connectors are untested.
- The data is not encrypted at rest. The app itself has no Dockerfile (compose runs FalkorDB only).
- Consolidation is ~n^1.7: 5,000 conversations take ~2 minutes (the real archive: <1 s).
- Only the ChatGPT and Claude export formats have adapters.

## License
Add a license of your choice before publishing.
