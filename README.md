# MemBridge

Local-first, cross-assistant personal AI memory (FalkorDB graph + SQLite + MCP). Full spec: `MEMBRIDGE_SPEC.md`; architecture decisions and milestones: `docs/ADR.md`.

## Status
- **M1 done:** ChatGPT importer (dry-run, resumable, idempotent), SQLite archive with FTS5.
- Next: M2 memory extraction + FalkorDB graph.

## Quick start
```bash
uv sync
uv run python samples/make_sample.py                 # synthetic export, no private data
uv run membridge import samples/chatgpt_export --dry-run
uv run membridge import <your export .zip | folder | conversations.json>
uv run membridge search "deadline"
uv run membridge stats
uv run pytest
```
Data lives in `~/.membridge` (override with `MEMBRIDGE_DATA_DIR`). Never commit it.
