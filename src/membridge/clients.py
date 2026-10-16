"""MCP client connection instructions. `verified` only means we actually exercised that client in tests/demo."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

from .config import Settings

# Honest status: what has really been run against this server, and how.
VERIFIED = {
    "claude-code": "Verified: headless `claude -p` called record_decision over stdio (see docs/CLIENTS.md).",
    "mcp-sdk-client": "Verified: independent MCP SDK stdio client (scripts/second_client.py and tests/test_mcp.py).",
}


def _server_entry(s: Settings, client: str) -> dict[str, Any]:
    return {
        "command": sys.executable,
        "args": ["-m", "membridge.mcp_server"],
        "env": {"MEMBRIDGE_DATA_DIR": str(s.data_dir), "MEMBRIDGE_CLIENT_NAME": client},
    }


def client_configs(s: Settings) -> dict[str, Any]:
    root = Path(sys.executable).parents[2] if "Scripts" in sys.executable or "bin" in sys.executable else Path.cwd()
    return {
        "note": "MCP cannot capture a client's conversations by itself: memories are saved only when the client calls the "
                "persistence tools (record_decision, save_memory, commit_session_summary).",  # fmt: skip
        "project_root": str(root),
        "clients": [
            {"id": "claude-code", "name": "Claude Code", "installed": shutil.which("claude") is not None,
             "status": VERIFIED["claude-code"], "how": "claude mcp add membridge -- <command> <args>  (or a .mcp.json file)",
             "config": {"mcpServers": {"membridge": _server_entry(s, "claude-code")}}},  # fmt: skip
            {"id": "cursor", "name": "Cursor", "installed": shutil.which("cursor") is not None,
             "status": "Configuration provided, NOT verified end-to-end in this project.", "how": "Add to ~/.cursor/mcp.json",
             "config": {"mcpServers": {"membridge": _server_entry(s, "cursor")}}},  # fmt: skip
            {"id": "vscode", "name": "VS Code (Copilot agent mode)", "installed": shutil.which("code") is not None,
             "status": "Configuration provided, NOT verified end-to-end in this project.", "how": "Add to .vscode/mcp.json",
             "config": {"servers": {"membridge": {"type": "stdio", **_server_entry(s, "vscode")}}}},  # fmt: skip
            {"id": "opencode", "name": "opencode", "installed": shutil.which("opencode") is not None,
             "status": "Configuration provided, NOT verified end-to-end in this project.", "how": "Add to opencode.json under `mcp`",
             "config": {"mcp": {"membridge": {"type": "local", "command": [sys.executable, "-m", "membridge.mcp_server"],
                                              "environment": {"MEMBRIDGE_DATA_DIR": str(s.data_dir), "MEMBRIDGE_CLIENT_NAME": "opencode"}}}}},  # fmt: skip
            {"id": "remote-http", "name": "Remote / ChatGPT-style connectors", "installed": None,
             "status": "Not verified. Custom MCP connectors depend on the client and plan; many require a public HTTPS URL.",
             "how": "Run `membridge mcp --transport http` with MEMBRIDGE_MCP_TOKEN set, then publish 127.0.0.1:8765 through an "
                    "authenticated TLS tunnel you control. The server refuses to start without a token or on a public bind.",  # fmt: skip
             "config": {"url": "https://<your-tunnel-host>/mcp", "headers": {"Authorization": "Bearer <MEMBRIDGE_MCP_TOKEN>"}}},
        ],
    }
