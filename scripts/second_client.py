"""Second, independent MCP client: reads what another client wrote, over real stdio.

Usage: uv run python scripts/second_client.py <mcp.json> <project>
Used in the demo and docs/CLIENTS.md to prove two clients share one persistent store.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(cfg_path: str, project: str) -> None:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))["mcpServers"]["membridge"]
    env = {**cfg.get("env", {}), "MEMBRIDGE_CLIENT_NAME": "second-client"}
    params = StdioServerParameters(command=cfg["command"], args=cfg["args"], env=env)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        res = await s.call_tool("get_project_context", {"project": project})
        data = res.structured_content or json.loads(res.content[0].text)
        print(f"[second-client] project={data['project']} current={len(data['current'])} history={len(data['history'])}")
        for m in data["current"]:
            print("  CURRENT :", m["statement"][:110])
        for m in data["history"]:
            print(f"  HISTORY ({m['status']}):", m["statement"][:110])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
