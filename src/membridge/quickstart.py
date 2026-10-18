"""First-run wizard: from nothing to a working, private memory in a few minutes.

`membridge quickstart`          guide a real user through importing their own history
`membridge quickstart --demo`   the same walkthrough on synthetic data (a separate ~/.membridge-demo store): nothing personal needed

Every step explains what it does and what it did not touch, and works without Docker (the graph view is an optional extra).
"""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Settings
from .store.graph import GraphStore
from .store.sqlite import Store
from .term import hidden, interactive

DEMO_QUESTIONS = [
    ("What database does Atlas use now?", "it follows the change you made: PostgreSQL, then SQLite. The old value is kept as history"),
    ("Where do I work?", "the employer you told it, not the different one in the imported ChatGPT summary"),
    ("What is my favourite pizza topping?", "it was never stored, so it says so instead of guessing"),
]
SUGGESTIONS = ["What am I working on overall?", "What did I decide about <a project you remember>?", "Where do I study / work?"]

Emit = Callable[[str], None]


def _step(emit: Emit, n: int, total: int, title: str) -> None:
    emit(f"\n[{n}/{total}] {title}\n" + "-" * (len(title) + 8))


def mcp_command(client: str = "claude-code") -> str:
    exe = Path(sys.executable)
    return f'claude mcp add membridge --env MEMBRIDGE_CLIENT_NAME={client} -- "{exe}" -m membridge.mcp_server'


def _answer(emit: Emit, store: Store, q: str) -> dict[str, Any]:
    from .memory.service import MemoryService

    r = MemoryService(store, actor="quickstart", human=True).search_memory(q, limit=3)
    emit(f'  Q: {q}')
    if r["sufficient_evidence"]:
        for m in r["memories"][:2]:
            emit(f'     -> [{m["status"]}, {m["verification"]}] {m["statement"][:110]}')
    else:
        emit("     -> (nothing stored about that, and it does not guess)")
    return r


def run(settings: Settings, *, demo: bool, export: Path | None, yes: bool, serve: Callable[[], None] | None, emit: Emit = print) -> dict[str, Any]:
    """Run the wizard. Returns a summary (used by tests). `serve` starts the dashboard when given."""
    total = 6
    ask_user = interactive() and not yes
    result: dict[str, Any] = {"demo": demo}
    emit("MemBridge: your AI memory, on your computer.")
    emit("It imports what your AI assistants know about you, keeps it in one place you own, and lets assistants read it under your rules.")
    emit(f"Data folder: {settings.data_dir}" + ("   (a separate demo store: your real one, if any, is untouched)" if demo else ""))

    _step(emit, 1, total, "Checking your setup")
    graph = GraphStore(settings.falkor_host, settings.falkor_port, settings.falkor_graph)
    result["graph"] = graph.available()
    emit("  Python and the local database: ok")
    emit("  Graph view: " + ("FalkorDB is running" if result["graph"] else "not running. That is fine: everything works without it. For the graph tab: docker compose up -d"))

    store = Store(settings.db_path)
    try:
        _step(emit, 2, total, "Bringing in your history" if not demo else "Loading a synthetic example")
        have = store.conn.execute("SELECT COUNT(*) FROM memories WHERE status!='deleted'").fetchone()[0]
        if demo:
            if have:
                emit(f"  The demo store already holds {have} memories; leaving it as it is.")
            else:
                from .demo_data import seed_demo_store

                emit("  Importing a made-up person: a project (Atlas) whose database changed, an employer, and a few private facts...")
                result["seed"] = seed_demo_store(store)
                emit(f"  Done: {result['seed']['memories']} memories ({result['seed']['private']} flagged private).")
        else:
            path = export
            if path is None and ask_user:
                emit("  You can import a ChatGPT or Claude export (.zip, a folder, or conversations.json).")
                emit("  Don't have one yet? ChatGPT: Settings > Data controls > Export data (it can take days).")
                emit("  Meanwhile run `membridge dump-prompt`, paste it into ChatGPT, and import its reply with `membridge seed-text`.")
                raw = input("  Path to your export (Enter to skip): ").strip().strip('"')
                path = Path(raw).expanduser() if raw else None
            if path is None:
                emit("  Skipped. Import any time with: membridge import <export>   (add --dry-run to preview)")
            else:
                from .importers.service import ImportError_, run_import
                from .memory.pipeline import run_extraction

                try:
                    pre = run_import(store, path, dry_run=True)
                    emit(f"  Preview: {pre.conversations_new} conversations, {pre.messages_new} messages ({pre.adapter}); nothing written yet.")
                    if ask_user and input("  Import it? [Y/n] ").strip().lower() in ("n", "no"):
                        emit("  Okay, nothing imported.")
                    else:
                        run_import(store, path)
                        emit(f"  Imported. Finding facts worth remembering (this reads only your own messages)... {run_extraction(store).created} found.")
                        emit("  Few facts is normal for task-style chats: it keeps only things you plainly said about yourself or your work.")
                except ImportError_ as e:
                    emit(f"  Could not import that: {e}")
                    result["import_error"] = str(e)

        _step(emit, 3, total, "What it knows")
        stats = {
            "memories": store.conn.execute("SELECT COUNT(*) FROM memories WHERE status!='deleted'").fetchone()[0],
            "private": store.conn.execute("SELECT COUNT(*) FROM memories WHERE status!='deleted' AND sensitivity!='normal'").fetchone()[0],
            "superseded": store.conn.execute("SELECT COUNT(*) FROM memories WHERE status='superseded'").fetchone()[0],
            "conversations": store.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
        }
        result.update(stats)
        emit(f"  {stats['conversations']} conversations, {stats['memories']} memories ({stats['superseded']} old values kept as history, {stats['private']} private).")
        if stats["memories"]:
            if demo:
                for q, why in DEMO_QUESTIONS:
                    _answer(emit, store, q)
                    emit(f"     ({why})")
            else:
                emit("  Try asking it things:")
                for s in SUGGESTIONS:
                    emit(f'    membridge ask "{s}"')
        else:
            emit("  Nothing stored yet. Import an export, or try `membridge quickstart --demo`.")

        _step(emit, 4, total, "Privacy: private memories")
        emit("  Health, money, identity-document and family facts are hidden from every assistant by default.")
        emit("  Only you can open them for one assistant, for a few minutes, with a PIN you choose.")
        from .memory import access

        if access.has_pin(store):
            emit("  A PIN is already set.")
        elif demo:
            emit("  (Demo: not setting a PIN. For your own store run `membridge set-pin`.)")
        elif ask_user:
            if input("  Set a PIN now? [Y/n] ").strip().lower() not in ("n", "no"):
                try:
                    pin = hidden("  New PIN (6+ characters): ")
                    if hidden("  Repeat PIN: ") != pin:
                        emit("  The PINs did not match; skipped. Run `membridge set-pin` later.")
                    else:
                        access.set_pin(store, pin)
                        emit("  PIN set. It is stored hashed and never shown to an assistant.")
                        result["pin_set"] = True
                except (access.AccessError, EOFError, OSError) as e:
                    emit(f"  Skipped ({e}). Run `membridge set-pin` later in a normal terminal.")
        else:
            emit("  Set one later with: membridge set-pin")

        _step(emit, 5, total, "Connecting an assistant (optional)")
        emit("  Claude Code:")
        emit(f"    {mcp_command()}")
        emit("  Other MCP clients (Cursor, VS Code, ...): see docs/CLIENTS.md. An assistant can read your memory and add to it;")
        emit("  it cannot mark things as verified, settle disagreements, or forget anything without you.")
    finally:
        store.close()

    _step(emit, 6, total, "The dashboard")
    if serve is None:
        emit("  Open it any time with: membridge serve   (it prints a private link)")
    else:
        emit("  Starting it now. Close this window (Ctrl+C) to stop it.")
        serve()
    return result


def open_dashboard(host: str = "127.0.0.1", port: int = 8787, *, browser: bool = True) -> None:
    """Start the dashboard, print the access link, and open it."""
    import secrets

    import uvicorn

    token = os.environ.get("MEMBRIDGE_DASHBOARD_TOKEN") or secrets.token_urlsafe(24)
    os.environ["MEMBRIDGE_DASHBOARD_TOKEN"] = token
    url = f"http://{host}:{port}/?token={token}"
    print(f"\n  Dashboard (this link carries this launch's private access token):\n    {url}\n")
    if browser:
        try:
            webbrowser.open(url)
        except Exception:  # no browser available: the printed link is enough
            pass
    uvicorn.run("membridge.api.app:create_app", factory=True, host=host, port=port, log_level="warning", access_log=False)


def prompt_text() -> str:
    return (Path(__file__).parent / "_data" / "chatgpt_dump_prompt.txt").read_text(encoding="utf-8")


def summary_json(r: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in r.items() if k != "seed"}, default=str)
