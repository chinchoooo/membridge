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
uv run membridge serve                    # prints a link with this launch's access token: open that link (dashboard on 127.0.0.1:8787)
uv run membridge mcp                      # MCP server over stdio (see docs/CLIENTS.md)
```

Optional extras: `uv sync --extra embeddings` (semantic channel: `membridge embed`), `--extra leiden` (Leiden communities).
Optional local LLM extraction: install Ollama, `MEMBRIDGE_LLM_PROVIDER=ollama MEMBRIDGE_LLM_MODEL=phi3 uv run membridge extract-llm`
(slow on CPU, resumable, labelled `model_inferred`). Hosted inference requires `MEMBRIDGE_ALLOW_HOSTED=true` and redacts first.

Run the whole demo: `uv run python scripts/demo.py` ([`docs/DEMO.md`](docs/DEMO.md)). Tests: `uv run pytest` (tests that need FalkorDB skip automatically when it is not running).

## Commands
`import · seed · seed-text · restructure · extract · extract-llm · classify · summarize-llm · resolve · entity-review · communities · embed · ppr-cache · graph-rebuild · ask · search · stats ·
backup · restore · export · lock · unlock · gold-init · eval-real · health · serve · mcp`
Privacy and consent: `set-pin · grant-sensitive · revoke-sensitive · grants · mark-private · forget · forget-requests · audit-verify`

## Private memories and who can see them
Health, money, identity-document and family facts are classified as **private** (also in raw messages) and are invisible to every assistant by default: absent from
search, summaries, the graph, exports, even from rankings. Mark anything the detector missed with `membridge mark-private <id>`.
To let one assistant read them for a short time: `membridge set-pin` once, then `membridge grant-sensitive <client> --minutes 10 --uses 25` (hidden PIN prompt; a program
without a terminal cannot answer it). The assistant then passes `include_sensitive=true`. Assistants cannot create grants, cannot mark a memory `user_verified`, cannot settle
contradictions, and can only *request* a `forget`; you confirm it (`membridge forget <id>` or the dashboard). See [threat model](docs/threat-model.md) for exactly what this does and does not protect.

## Importing what ChatGPT already knows about you
Paste the prompt in [`docs/CHATGPT_DUMP_PROMPT.md`](docs/CHATGPT_DUMP_PROMPT.md) into ChatGPT, save the reply as `chatgpt-dump.txt`, then `membridge seed-text chatgpt-dump.txt --dry-run` and without `--dry-run`.
Everything lands as `provider_stated` (ChatGPT's belief, never "you said it"), undated facts never outrank something you dated, and conflicts are left for you.
Your official export (`membridge import`) later outranks the dump. Facts imported before structured triples existed can be upgraded in place with `membridge restructure`.

## Configuration (env vars, prefix `MEMBRIDGE_`, or `.env`)
`DATA_DIR` (default `~/.membridge`), `FALKOR_HOST/PORT/GRAPH` (127.0.0.1 / 6390 / membridge), `LLM_PROVIDER` (none|ollama|openai_compat),
`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY_ENV` (name of the variable holding a key), `ALLOW_HOSTED`, `REDACT_HOSTED`, `EMBEDDER` (none|fastembed|hashing),
`MCP_TOKEN` (HTTP transport), `CLIENT_NAME` (audit identity and the name a grant is bound to), `DASHBOARD_TOKEN` (fixed dashboard token; otherwise one is generated per launch).
`ALLOW_REMOTE_GRAPH=1` must be set in the process environment to use a non-loopback FalkorDB host. An optional `.env` is read from `~/.membridge/.env` only (never from the current directory). Keys are never stored in files.

## Documentation
[Architecture decisions](docs/ADR.md) · [Graph schema + Cypher](docs/graph-schema.md) · [Threat model](docs/threat-model.md) · [Importing ChatGPT memory](docs/CHATGPT_DUMP_PROMPT.md) ·
[Connecting clients](docs/CLIENTS.md) · [Evaluation](docs/EVAL.md) · [Demo plan](docs/DEMO.md) · [Original spec](MEMBRIDGE_SPEC.md)

## Troubleshooting
- *FalkorDB not reachable*: `docker compose up -d`; everything still works in degraded SQLite mode (same answers, no graph UI).
- *Port 6390 busy*: set `MEMBRIDGE_FALKOR_PORT`. *No dashboard*: build it (`cd web && npm run build`).
- *Import says already_imported*: the exact archive was imported; use `--force` to re-scan.
- *Ollama slow/timeouts*: lower `--window-chars`; it resumes where it stopped.

## Speed
Memory calls are cheap: on a real 2,000-message archive a new question takes **~14 ms** (p50; p95 21 ms) and a repeated one **~30 µs**
(measured after the v0.2 evidence gate, which costs ~11 ms per new question: it counts how rare each query word is; before it, a cold call was 2.5 ms).
The 5,000-conversation / 13k-memory figures in [`docs/EVAL.md`](docs/EVAL.md) §6 (3-6 ms cold) predate the gate and have not been re-measured.
Mechanisms: content-generation cache, warm in-memory index, precomputed PageRank vectors, IDF-pruned full-text, vectorised packing.
Run `uv run python eval/latency.py real` to measure your own store.

## Encryption and backups
`membridge backup --encrypt`, `membridge export --encrypt` (passphrase from a prompt or an env var *name*), `membridge restore <file.enc>`, and
`membridge lock` / `unlock` to keep the database encrypted whenever it isn't in use. See [`docs/threat-model.md`](docs/threat-model.md).

## Docker
`docker compose up -d` runs FalkorDB; `MEMBRIDGE_DASHBOARD_TOKEN=<long random string> docker compose --profile app up -d` also runs the dashboard/API (published on `127.0.0.1:8787` only, non-root; open `http://127.0.0.1:8787/?token=<that string>`).

## Limitations (read these)
- Rule extraction is conservative and finds only 2 candidates in the author's 945 real messages (they are task prompts). Real recall needs the LLM tier:
  local `phi3` is slow (~59 s/window), and hosted extraction (`claude_cli`, opt-in, redacted, `--dry-run` first) costs money (~$4 for that archive).
- Benchmarks are synthetic; no precision/recall is claimed on real data until you judge answers with `membridge gold-init` / `eval-real`.
- Verified clients: Claude Code, the Python MCP SDK client, and the official MCP Inspector CLI. Cursor/VS Code/opencode configs are provided but unverified; hosted/ChatGPT connectors are untested.
- The live database is plaintext while unlocked (no SQLCipher build for Windows); encrypted backups/exports and `lock`/`unlock` are provided. An assistant that can run shell commands as you can read the file directly: the PIN protects the MCP/dashboard path, `lock` protects the file.
- The answer-or-abstain gate is measured on a synthetic persona (33 answerable / 34 unanswerable questions: 34/34 unanswerable declined, 32/33 answerable answered; the miss needs world knowledge that Rust is a programming language, i.e. the optional embedder). It has not been scored on your real data until you judge answers with `gold-init` / `eval-real`.
- The privacy detector is a pattern lexicon: it can miss unusual phrasing (use `mark-private`) and it over-flags on purpose.
- Consolidation is ≈ n^1.3: 5,000 conversations take ~44 s (missed my 20 s target); your real archive takes under a second.
- Import formats: ChatGPT, Claude, Claude Code sessions, Markdown/text notes.

## License
MIT (see `LICENSE`).
