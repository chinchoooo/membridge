# Connecting AI clients (MCP)

MemBridge exposes the same store to any MCP client. **MCP cannot capture a client's conversations by itself**: memories
are saved only when the client calls `record_decision`, `save_memory` or `commit_session_summary` (or you import an export).

Set `MEMBRIDGE_CLIENT_NAME` so the audit log shows who did what. Generate ready-to-paste configs with the dashboard
(Maintenance → Connect AI clients) or `GET /api/clients`.

## Verification status (honest)

| Client | Status | How it was verified |
|---|---|---|
| **Claude Code** | ✅ verified end to end | Headless `claude -p --mcp-config … --allowedTools mcp__membridge__record_decision` wrote a decision (see `scripts/demo.py --claude`, step 4); a separate process then read it from the same database. |
| **Generic MCP SDK client (stdio)** | ✅ verified | `tests/test_mcp.py` spawns the server as a separate process per call: write, read, restart-persistence, forget, errors, audit identity. `scripts/second_client.py` is the reusable second client. |
| **Official MCP Inspector CLI** (TypeScript SDK, an independent implementation) | ✅ verified | `bash scripts/inspector_check.sh`: lists all 20 tools, writes a decision in one process, reads it in another, and an unsafe `save_memory` is refused (`isError: true`). |
| Cursor, VS Code (Copilot agent mode), opencode | ⚠️ configuration provided, **not verified** here | Installed on the dev machine but they cannot be driven headlessly; the JSON below follows each client's documented format. |
| ChatGPT / hosted connectors | ⚠️ **not verified**, may be unavailable | Custom MCP connectors depend on plan and usually need a public HTTPS URL. Use `membridge mcp --transport http` behind an authenticated TLS tunnel you control. |

## Configurations

Claude Code (`claude mcp add` or `.mcp.json`):
```json
{"mcpServers": {"membridge": {"command": "<path>/.venv/Scripts/python.exe", "args": ["-m", "membridge.mcp_server"],
  "env": {"MEMBRIDGE_DATA_DIR": "<your data dir>", "MEMBRIDGE_CLIENT_NAME": "claude-code"}}}}
```
Cursor (`~/.cursor/mcp.json`): same `mcpServers` block, `MEMBRIDGE_CLIENT_NAME=cursor`.
VS Code (`.vscode/mcp.json`): `{"servers": {"membridge": {"type": "stdio", "command": …, "args": […], "env": {…}}}}`.
opencode (`opencode.json`): `{"mcp": {"membridge": {"type": "local", "command": ["<python>", "-m", "membridge.mcp_server"], "environment": {…}}}}`.

Remote (optional): `MEMBRIDGE_MCP_TOKEN=<random 24+ chars> membridge mcp --transport http` listens on `127.0.0.1:8765` and
requires `Authorization: Bearer <token>`. It refuses non-loopback binds by design.

## Tools

Retrieval: `search_memory`, `get_relevant_context`, `get_memory`, `get_related_memories`, `trace_memory_provenance`,
`get_conversation`, `search_conversations`, `get_project_context`, `get_overview`, `list_projects`, `get_memory_statistics`,
`list_unresolved_contradictions`. Filters: `category, status, project, since, until, as_of, known_at, verification`.
Persistence: `save_memory`, `record_decision`, `commit_session_summary`, `update_memory`, `link_memories`.
Maintenance: `run_consolidation`, `export_memory`, `forget_memory` (two-step).
