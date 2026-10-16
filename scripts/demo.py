"""Reproducible end-to-end demo of the six hackathon steps on a fresh, synthetic archive (no private data).

Run:  docker compose up -d && uv run python scripts/demo.py [--claude]
--claude additionally drives the real Claude Code CLI as the first MCP client (costs a few cents of API usage).
Everything is written to .membridge-demo/ (git-ignored) and uses graph name 'membridge_demo'.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEMO = ROOT / ".membridge-demo"
os.environ.update(MEMBRIDGE_DATA_DIR=str(DEMO), MEMBRIDGE_FALKOR_GRAPH="membridge_demo", MEMBRIDGE_FALKOR_PORT=os.environ.get("MEMBRIDGE_FALKOR_PORT", "6390"))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from membridge.config import Settings  # noqa: E402
from membridge.importers.service import run_import  # noqa: E402
from membridge.memory.pipeline import run_extraction  # noqa: E402
from membridge.memory.service import MemoryService  # noqa: E402
from membridge.ops import backup, export_memory, restore  # noqa: E402
from membridge.store.graph import GraphStore  # noqa: E402
from membridge.store.sqlite import Store  # noqa: E402


def h(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def show(pkg: dict[str, Any], n: int = 4) -> None:
    s = pkg["stats"]
    print(f"  [{s['mode']}] {s['latency_ms']} ms, {s['tokens_estimate']} tokens, channels={list(s['channels'])}, intent={s['temporal_intent']}")
    for m in pkg["memories"][:n]:
        ev = m["evidence"][0] if m["evidence"] else {}
        print(f"  - ({m['status']}/{m['verification']}) {m['statement'][:92]}")
        print(f"      from: {ev.get('conversation', ev.get('source', '?'))}")
    for p in pkg["graph_paths"][:2]:
        print(f"  path: {p[:160]}")


async def mcp_call(client: str, tool: str, **args: Any) -> dict[str, Any]:
    cfg = Settings()
    env = {**os.environ, "MEMBRIDGE_CLIENT_NAME": client}
    params = StdioServerParameters(command=sys.executable, args=["-m", "membridge.mcp_server"], env=env)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        res = await s.call_tool(tool, args)
        assert not res.is_error, res.content
        _ = cfg
        return dict(res.structured_content or json.loads(res.content[0].text))


def main() -> None:
    use_claude = "--claude" in sys.argv
    if DEMO.exists():
        shutil.rmtree(DEMO)
    DEMO.mkdir()
    sys.path.insert(0, str(ROOT / "samples"))
    import make_sample  # type: ignore[import-not-found]

    export = DEMO / "export" / "conversations.json"
    export.parent.mkdir()
    export.write_text(json.dumps(make_sample.build_export()), encoding="utf-8")
    settings = Settings()
    graph = GraphStore(settings.falkor_host, settings.falkor_port, settings.falkor_graph)
    if not graph.available():
        print("FalkorDB is not running: start it with `docker compose up -d` (the demo needs the graph).")
        raise SystemExit(2)

    h("STEP 1  Historical migration: import -> memories -> graph")
    store = Store(settings.db_path)
    m = run_import(store, export)
    print(f"  imported {m.conversations_new} conversations / {m.messages_new} messages, {len(m.failures)} malformed record reported, status={m.status}")
    again = run_import(store, export)
    print(f"  re-import of the same archive: {again.status} (no duplicates)")
    run_extraction(store)
    graph.rebuild(store)
    svc = MemoryService(store, graph, actor="demo")
    stats = svc.get_memory_statistics()
    print(f"  memories={stats['memories']} entities={stats['entities']} graph={stats['graph']}")

    h("STEP 2  Cross-conversation retrieval, traced to sources")
    show(svc.search_memory("Why did I stop using Postgres for Atlas?"))
    tr = svc.trace_memory_provenance(svc.search_memory("switched Atlas from PostgreSQL to SQLite")["memories"][0]["id"])
    print(f"  provenance: {tr['sources'][0]['conversation']!r}, quote verified in original: {tr['sources'][0]['quote_found_in_original']}")

    h("STEP 3  Graph reasoning: a question that shares no words with the answer")
    print('  Q: "fatal remaining connection slots reserved"  (only the error message says this; the decision is in another conversation)')
    show(svc.get_relevant_context("fatal remaining connection slots reserved"), 3)

    h("STEP 4  Cross-client portability (two separate MCP client processes, one store)")
    if use_claude:
        cfg = DEMO / "mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"membridge": {"command": sys.executable, "args": ["-m", "membridge.mcp_server"],
                        "env": {"MEMBRIDGE_DATA_DIR": str(DEMO), "MEMBRIDGE_FALKOR_GRAPH": "membridge_demo", "MEMBRIDGE_CLIENT_NAME": "claude-code"}}}}), encoding="utf-8")
        r = subprocess.run(["claude", "-p", "Call the membridge MCP tool record_decision with project='Atlas' and decision='Archive raw sensor data as Parquet files for cold storage'. Reply with only the memory_id.",
                            "--mcp-config", str(cfg), "--strict-mcp-config", "--allowedTools", "mcp__membridge__record_decision", "--model", "haiku"],
                           capture_output=True, text=True, timeout=240)
        print("  client A = Claude Code ->", r.stdout.strip()[:80] or r.stderr[:120])
    else:
        out = asyncio.run(mcp_call("client-A", "record_decision", project="Atlas", decision="Archive raw sensor data as Parquet files for cold storage", rationale="cheap columnar archive"))
        print(f"  client A (MCP SDK) recorded decision {out['memory_id']} status={out['status']}")
    ctx = asyncio.run(mcp_call("client-B", "get_project_context", project="Atlas"))
    print(f"  client B (separate process) sees: {[x['statement'][:60] for x in ctx['current'] if 'Parquet' in x['statement']]}")
    audit = Store(settings.db_path).conn.execute("SELECT DISTINCT actor FROM audit_log WHERE action IN ('record_decision','get_project_context')").fetchall()
    print(f"  audit log actors: {sorted(r[0] for r in audit)}")

    h("STEP 5  Update and contradiction: history preserved, current state updated")
    svc2 = MemoryService(Store(settings.db_path), graph, actor="demo")
    svc2.record_decision("Atlas", "Use DuckDB for analytics")
    r2 = svc2.record_decision("Atlas", "Use Polars for analytics")
    print(f"  second decision on the same aspect supersedes the first: {bool(r2['supersedes'])}")
    c = svc2.get_project_context("Atlas")
    print("  CURRENT :", [x["statement"][:50] for x in c["current"] if "analytics" in x["statement"]])
    print("  HISTORY :", [(x["statement"][:44], x["status"]) for x in c["history"]][:4])
    show(svc2.search_memory("Which database did Atlas use before SQLite?"), 2)
    show(svc2.search_memory("Atlas storage database", filters={"as_of": 1_700_000_000 + 15 * 86400}), 2)
    print("  (above: history question, then as-of 'two weeks after the first conversation')")

    h("STEP 6  Local ownership: restart, export, back up, restore")
    before = svc2.get_memory_statistics()["memories"]
    if shutil.which("docker"):
        subprocess.run(["docker", "compose", "restart", "falkordb"], cwd=ROOT, capture_output=True, timeout=120)
        for _ in range(30):
            if graph.available():
                break
            import time as _t

            _t.sleep(1)
    fresh_store = Store(settings.db_path)
    fresh = MemoryService(fresh_store, GraphStore(settings.falkor_host, settings.falkor_port, settings.falkor_graph), actor="demo")
    after = fresh.get_memory_statistics()
    print(f"  after restarting FalkorDB and reopening everything: memories {before} -> {after['memories']}, graph={after['graph']}")
    print("  answer after restart:", fresh.search_memory("What does Atlas use for analytics?")["memories"][0]["statement"][:70])
    exp = export_memory(fresh_store, DEMO / "exports", "json", include_raw=True)
    bak = backup(fresh_store, DEMO / "backups")
    print(f"  export: {exp['memories']} memories -> {Path(exp['path']).name}; backup: {Path(bak['path']).name} ({bak['bytes'] // 1024} KB, integrity {bak['integrity']})")
    tmp = Path(tempfile.mkdtemp()) / "restored.db"
    res = restore(tmp, Path(bak["path"]))
    print(f"  restore into a new location: {res['stats']['conversations']} conversations, {res['stats']['messages']} messages recovered with no AI provider involved")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
