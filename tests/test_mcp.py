"""MCP tests over real stdio: every call spawns the server as a separate process (a fresh 'client session')."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from membridge.importers.service import run_import
from membridge.memory.pipeline import run_extraction
from membridge.store.sqlite import Store

EXPECTED_TOOLS = {
    "search_memory", "get_memory", "get_related_memories", "get_conversation", "search_conversations",
    "get_project_context", "get_relevant_context", "trace_memory_provenance", "save_memory", "update_memory",
    "record_decision", "commit_session_summary", "link_memories", "list_projects", "get_memory_statistics",
    "run_consolidation", "export_memory", "forget_memory",
}  # fmt: skip


@pytest.fixture
def data_dir(tmp_path: Path, export_file: Path) -> Path:
    d = tmp_path / "data"
    s = Store(d / "membridge.db")
    run_import(s, export_file)
    run_extraction(s)
    s.close()
    return d


def _env(data_dir: Path, client: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(MEMBRIDGE_DATA_DIR=str(data_dir), MEMBRIDGE_FALKOR_PORT="1", MEMBRIDGE_CLIENT_NAME=client)
    return env


async def _session(data_dir: Path, client: str, fn: Any) -> Any:
    params = StdioServerParameters(command=sys.executable, args=["-m", "membridge.mcp_server"], env=_env(data_dir, client))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        return await fn(session)


def _payload(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content
    if getattr(result, "structured_content", None):
        return dict(result.structured_content)
    return dict(json.loads(result.content[0].text))


def call(data_dir: Path, client: str, tool: str, **args: Any) -> dict[str, Any]:
    async def go(session: ClientSession) -> dict[str, Any]:
        return _payload(await session.call_tool(tool, args))

    return dict(asyncio.run(_session(data_dir, client, go)))


def test_server_lists_all_spec_tools_with_safety_annotations(data_dir: Path) -> None:
    async def go(session: ClientSession) -> Any:
        return (await session.list_tools()).tools

    tools = asyncio.run(_session(data_dir, "t", go))
    names = {t.name for t in tools}
    assert EXPECTED_TOOLS <= names
    forget = next(t for t in tools if t.name == "forget_memory")
    assert forget.annotations is not None and forget.annotations.destructive_hint is True
    assert next(t for t in tools if t.name == "search_memory").annotations.read_only_hint is True  # type: ignore[union-attr]
    assert not {"execute_cypher", "run_sql", "shell"} & names  # no raw database/shell access


def test_cross_client_write_then_read_against_same_store(data_dir: Path) -> None:
    saved = call(data_dir, "client-a", "record_decision", project="Atlas", decision="Adopt DuckDB for analytics", rationale="fast scans")
    assert saved["status"] == "current"
    ctx = call(data_dir, "client-b", "get_project_context", project="Atlas")  # a different process, a different client name
    assert any("DuckDB" in m["statement"] for m in ctx["current"])
    assert any("SQLite" in m["statement"] for m in ctx["history"] + ctx["current"])
    # persistence across 'restart': a brand-new server process sees the earlier write
    found = call(data_dir, "client-c", "search_memory", query="DuckDB analytics")
    assert found["memories"] and "DuckDB" in found["memories"][0]["statement"]


def test_contradiction_preserves_history_over_mcp(data_dir: Path) -> None:
    call(data_dir, "a", "record_decision", project="Atlas", decision="Use DuckDB for analytics")
    r = call(data_dir, "b", "record_decision", project="Atlas", decision="Use Polars for analytics")
    assert r["supersedes"]
    ctx = call(data_dir, "c", "get_project_context", project="Atlas")
    assert any("Polars" in m["statement"] for m in ctx["current"]) and any("DuckDB" in m["statement"] for m in ctx["history"])


def test_forget_is_two_step_over_mcp(data_dir: Path) -> None:
    mid = call(data_dir, "a", "save_memory", content="Remember the Atlas demo is on Friday", category="fact", source="user_stated")["memory_id"]
    first = call(data_dir, "a", "forget_memory", memory_id=mid)
    assert first["confirmation_required"] is True
    assert call(data_dir, "a", "get_memory", memory_id=mid)["status"] == "current"  # nothing deleted yet
    assert call(data_dir, "a", "forget_memory", memory_id=mid, confirm_token=first["confirm_token"]) == {"forgotten": mid}


def test_invalid_input_returns_tool_error_not_crash(data_dir: Path) -> None:
    async def go(session: ClientSession) -> Any:
        a = await session.call_tool("save_memory", {"content": "key sk-abcdefghijklmnopqrstuvwxyz123456"})
        b = await session.call_tool("search_memory", {"query": "x", "filters": {"drop_table": 1}})
        c = await session.call_tool("get_memory", {"memory_id": "mem_nope"})
        return a, b, c

    a, b, c = asyncio.run(_session(data_dir, "t", go))
    assert a.is_error and b.is_error and c.is_error


def test_audit_log_records_client_identity(data_dir: Path) -> None:
    call(data_dir, "claude-code", "save_memory", content="The audit trail note about Atlas", source="user_stated")
    s = Store(data_dir / "membridge.db")
    row = s.conn.execute("SELECT actor FROM audit_log WHERE action='save_memory'").fetchone()
    s.close()
    assert row is not None and row[0] == "claude-code"


def test_http_transport_refuses_without_token_or_public_bind(data_dir: Path) -> None:
    env = _env(data_dir, "t")
    env.pop("MEMBRIDGE_MCP_TOKEN", None)
    p = subprocess.run([sys.executable, "-m", "membridge.mcp_server", "--transport", "http"], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and "MEMBRIDGE_MCP_TOKEN" in (p.stderr + p.stdout)
    env["MEMBRIDGE_MCP_TOKEN"] = "x" * 24
    p = subprocess.run([sys.executable, "-m", "membridge.mcp_server", "--transport", "http", "--host", "0.0.0.0"], env=env, capture_output=True, text=True, timeout=60)  # noqa: S104
    assert p.returncode != 0 and "loopback" in (p.stderr + p.stdout)
