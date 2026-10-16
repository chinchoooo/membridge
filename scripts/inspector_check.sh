#!/usr/bin/env bash
# Independent MCP client check: drives the stdio server with the official MCP Inspector CLI (TypeScript SDK, a different
# implementation from this project's Python SDK). Requires Node/npx and network access for `npx`.
# Usage: bash scripts/inspector_check.sh
set -euo pipefail
DATA="$(mktemp -d)"
export MEMBRIDGE_DATA_DIR="$DATA" MEMBRIDGE_CLIENT_NAME=inspector-cli
PY="${PYTHON:-$(pwd)/.venv/Scripts/python.exe}"
[ -f "$PY" ] || PY="$(pwd)/.venv/bin/python"
run() { npx -y @modelcontextprotocol/inspector --cli "$PY" -m membridge.mcp_server "$@" 2>&1; }

echo "== tools/list =="; run --method tools/list | grep -c '"name"' | xargs echo "tools:"
echo "== write (process 1) =="; run --method tools/call --tool-name record_decision --tool-arg project=Atlas --tool-arg "decision=Use DuckDB for analytics" | grep -E '"status"|memory_id' | head -2
echo "== read (process 2) =="; run --method tools/call --tool-name get_project_context --tool-arg project=Atlas | grep -E 'DuckDB' | head -1
echo "== unsafe input is refused =="; run --method tools/call --tool-name save_memory --tool-arg "content=key sk-abcdefghijklmnopqrstuvwxyz123456" | grep -E '"isError"'
rm -rf "$DATA"
