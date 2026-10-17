"""MemBridge command line interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .config import Settings
from .importers.service import ImportError_, run_import
from .store.sqlite import Store

app = typer.Typer(
    help="MemBridge: local-first personal AI memory.", no_args_is_help=True
)


@app.command("import")
def import_cmd(
    path: Annotated[
        Path, typer.Argument(help="conversations.json, export folder, or .zip")
    ],
    dry_run: Annotated[
        bool, typer.Option(help="Validate and report without writing")
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Re-scan an archive that was already imported")
    ] = False,
    adapter: Annotated[
        str | None, typer.Option(help="Force an adapter (default: auto-detect)")
    ] = None,
) -> None:
    """Import a conversation export into the local archive (idempotent, resumable)."""
    settings = Settings()
    store = None if dry_run else Store(settings.db_path)
    try:
        m = run_import(
            store,
            path,
            adapter_name=adapter,
            dry_run=dry_run,
            force=force,
            batch_size=settings.import_batch_size,
            report_dir=settings.import_dir,
        )
    except ImportError_ as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    finally:
        if store:
            store.close()
    typer.echo(m.to_json())
    raise typer.Exit(0 if m.status in ("completed", "already_imported") else 1)


@app.command()
def stats() -> None:
    """Show archive counts."""
    store = Store(Settings().db_path)
    for k, v in store.stats().items():
        typer.echo(f"{k}: {v}")
    store.close()


@app.command()
def search(query: str, limit: int = 5) -> None:
    """Full-text search over raw messages (exact-evidence lookup)."""
    store = Store(Settings().db_path)
    for r in store.search_messages(query, limit):
        typer.echo(f"{r['id']} [{r['role']}] {r['text'][:120]!r}")
    store.close()


def _graph(settings: Settings):  # type: ignore[no-untyped-def]
    from .store.graph import GraphStore

    return GraphStore(settings.falkor_host, settings.falkor_port, settings.falkor_graph)


@app.command()
def seed(
    zips: Annotated[list[Path], typer.Argument(help="Claude export memories-*.zip / projects-*.zip")],
    replace: Annotated[bool, typer.Option(help="Re-derive: delete earlier Claude-seeded memories first")] = False,
) -> None:
    """Seed memories and project entities from a Claude export's memory/project files."""
    from .memory.pipeline import seed_from_claude_export

    store = Store(Settings().db_path)
    typer.echo(json.dumps(seed_from_claude_export(store, zips, replace=replace)))
    store.close()


@app.command("seed-text")
def seed_text(
    dump: Annotated[Path, typer.Argument(help="ChatGPT data-portability dump (see docs/CHATGPT_DUMP_PROMPT.md)")],
    dry_run: Annotated[bool, typer.Option(help="Parse and count only; write nothing")] = False,
) -> None:
    """Import a ChatGPT memory dump as provider-stated memories (sensitive ones stay out of default retrieval)."""
    from .memory.seed_text import seed_from_text

    store = Store(Settings().db_path)
    typer.echo(json.dumps(seed_from_text(store, dump, dry_run=dry_run)))
    store.close()


@app.command()
def extract() -> None:
    """Extract memories from imported messages (incremental, idempotent)."""
    from .memory.pipeline import run_extraction

    store = Store(Settings().db_path)
    typer.echo(json.dumps(run_extraction(store).__dict__, indent=2))
    store.close()


@app.command("graph-rebuild")
def graph_rebuild() -> None:
    """Rebuild the FalkorDB graph from the SQLite records."""
    settings = Settings()
    g = _graph(settings)
    if not g.available():
        typer.secho("FalkorDB is not reachable. Run: docker compose up -d", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    store = Store(settings.db_path)
    typer.echo(json.dumps({"synced": g.rebuild(store), "graph": g.counts()}, indent=2))
    store.close()


def _llm_or_exit(settings: Settings):  # type: ignore[no-untyped-def]
    from .llm.base import LLMError
    from .llm.factory import build_llm

    try:
        llm = build_llm(settings)
    except LLMError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    if llm is None:
        typer.secho("No LLM configured (MEMBRIDGE_LLM_PROVIDER=ollama|openai_compat). Rule-based extraction needs none.",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(2)
    return llm


@app.command("extract-llm")
def extract_llm(
    max_windows: Annotated[int | None, typer.Option(help="Stop after N prompt windows (resumable)")] = None,
    window_chars: Annotated[int, typer.Option(help="Characters of conversation per LLM call")] = 1600,
    msg_chars: Annotated[int, typer.Option(help="Max characters taken from one message")] = 900,
    workers: Annotated[int, typer.Option(help="Concurrent LLM calls (hosted providers)")] = 1,
    dry_run: Annotated[bool, typer.Option(help="Write the exact payloads to a file and send NOTHING")] = False,
) -> None:
    """LLM extraction of memories (local by default; hosted needs MEMBRIDGE_ALLOW_HOSTED=true)."""
    from .memory.llm_tasks import extract_with_llm

    settings = Settings()
    llm = _llm_or_exit(settings)
    store = Store(settings.db_path)
    dry = (settings.data_dir / "llm-dryrun" / f"payloads-{int(__import__('time').time())}.jsonl") if dry_run else None
    st = extract_with_llm(store, llm, max_windows=max_windows, window_chars=window_chars, msg_chars=msg_chars, workers=workers, dry_run_path=dry,
                          progress=lambda s: typer.echo(f"windows={s.windows} created={s.created} errors={s.errors}", err=True))  # fmt: skip
    typer.echo(json.dumps({**st.__dict__, "provider": llm.name, "remote": llm.is_remote,
                           "chars_sent_remote": getattr(llm, "sent_chars", 0), "redactions": getattr(llm, "redactions", {})}, indent=2))  # fmt: skip
    store.close()


@app.command("classify")
def classify(max_pairs: int = 40) -> None:
    """Use the LLM to classify memory pairs the rules could not settle."""
    from .memory.llm_tasks import classify_conflicts

    settings = Settings()
    llm = _llm_or_exit(settings)
    store = Store(settings.db_path)
    typer.echo(json.dumps(classify_conflicts(store, llm, max_pairs=max_pairs).__dict__, indent=2))
    store.close()


@app.command("summarize-llm")
def summarize_llm() -> None:
    """LLM project summaries (rejected unless faithful to the stored memories)."""
    from .memory.llm_tasks import summarize_projects_llm

    settings = Settings()
    llm = _llm_or_exit(settings)
    store = Store(settings.db_path)
    typer.echo(json.dumps(summarize_projects_llm(store, llm)))
    store.close()


@app.command("ask")
def ask(
    query: str, budget: int = 1500, graph: bool = True,
    private: Annotated[bool, typer.Option("--private", help="Include private memories (asks for your PIN at a hidden prompt)")] = False,
) -> None:  # fmt: skip
    """Retrieve context for a question (what an AI client would get)."""
    from .retrieval.engine import Retriever

    settings = Settings()
    store = Store(settings.db_path)
    if private:
        _require_pin(store)
        store.audit_sync("cli", "sensitive_access", "ask")
    g = _graph(settings) if graph else None
    typer.echo(json.dumps(Retriever(store, g).search(query, token_budget=budget, full=private), indent=2, default=str))
    store.close()


def _require_pin(store: Store) -> None:
    """Hidden prompt for the PIN. A program running this command without a terminal cannot answer it."""
    import getpass

    from .memory import access

    if not access.has_pin(store):
        typer.secho("No PIN is set yet: run `membridge set-pin` first.", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    try:
        ok = access.verify_pin(store, getpass.getpass("PIN: "))
    except (access.AccessError, EOFError, OSError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    if not ok:
        typer.secho("Wrong PIN.", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _human_service(settings: Settings):  # type: ignore[no-untyped-def]
    from .memory.service import MemoryService

    return MemoryService(Store(settings.db_path), _graph(settings), actor="cli", human=True)


@app.command("set-pin")
def set_pin() -> None:
    """Set or change the PIN that guards private memories (health, money, family...). Typed at a hidden prompt."""
    import getpass

    from .memory import access

    store = Store(Settings().db_path)
    try:
        old = getpass.getpass("Current PIN: ") if access.has_pin(store) else None
        new = getpass.getpass("New PIN (6+ characters): ")
        if getpass.getpass("Repeat PIN: ") != new:
            raise access.AccessError("the PINs do not match")
        access.set_pin(store, new, old)
    except (access.AccessError, EOFError, OSError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    typer.echo("PIN set. Nothing is shared with any assistant until you run `membridge grant-sensitive <client>`.")


@app.command("grant-sensitive")
def grant_sensitive(
    client: Annotated[str, typer.Argument(help="The client name the assistant connects as (MEMBRIDGE_CLIENT_NAME)")],
    minutes: Annotated[int, typer.Option(help="How long the grant lasts")] = 10,
    uses: Annotated[int, typer.Option(help="How many private reads it allows")] = 25,
) -> None:
    """Let ONE assistant read private memories for a short time. Needs your PIN (hidden prompt)."""
    import getpass

    from .memory import access

    store = Store(Settings().db_path)
    try:
        out = access.create_grant(store, getpass.getpass("PIN: "), client, minutes, uses)
    except (access.AccessError, EOFError, OSError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    typer.echo(json.dumps(out))


@app.command("revoke-sensitive")
def revoke_sensitive(client: Annotated[str | None, typer.Argument(help="Omit to revoke every grant")] = None) -> None:
    """End access to private memories now."""
    from .memory import access

    typer.echo(json.dumps({"revoked": access.revoke(Store(Settings().db_path), client)}))


@app.command("grants")
def grants() -> None:
    """Who may currently read private memories."""
    from .memory import access

    store = Store(Settings().db_path)
    typer.echo(json.dumps({"pin_set": access.has_pin(store), "grants": access.list_grants(store)}, indent=2))


@app.command("forget")
def forget(memory_id: str, yes: Annotated[bool, typer.Option("--yes", help="Do not ask")] = False) -> None:
    """Permanently forget a memory (confirm here; an assistant can only request it)."""
    svc = _human_service(Settings())
    first = svc.forget_memory(memory_id, actor="cli")
    typer.echo(f"Will forget: {first['will_forget']}")
    if not yes and not typer.confirm("Forget this permanently?"):
        raise typer.Exit(1)
    typer.echo(json.dumps(svc.forget_memory(memory_id, first["confirm_token"], actor="cli")))


@app.command("forget-requests")
def forget_requests() -> None:
    """Memories an assistant has asked to forget and is waiting on you for."""
    typer.echo(json.dumps(_human_service(Settings()).list_forget_requests(), indent=2))


@app.command("audit-verify")
def audit_verify() -> None:
    """Check the audit log's hash chain: any edited or removed entry is reported."""
    from .store.sqlite import verify_audit_chain

    store = Store(Settings().db_path)
    store.flush_audit()
    out = verify_audit_chain(store)
    typer.echo(json.dumps(out))
    raise typer.Exit(0 if out["ok"] else 1)


def _secure(path: str, encrypt: bool, passphrase_env: str | None) -> str:
    """Encrypt `path` in place (plaintext removed after the encrypted copy is written) when requested."""
    from .crypto import CryptoError, encrypt_file, get_passphrase

    if not encrypt:
        return path
    try:
        enc = encrypt_file(Path(path), Path(path + ".enc"), get_passphrase(passphrase_env, confirm=passphrase_env is None))
    except CryptoError as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    Path(path).unlink()
    return str(enc)


@app.command()
def backup(
    dest: Annotated[Path | None, typer.Option(help="Backup folder")] = None,
    encrypt: Annotated[bool, typer.Option(help="AES-256-GCM encrypt with a passphrase")] = False,
    passphrase_env: Annotated[str | None, typer.Option(help="NAME of an env var holding the passphrase (default: prompt)")] = None,
) -> None:
    """Consistent copy of the database (the graph is rebuildable)."""
    from .ops import backup as do_backup

    settings = Settings()
    store = Store(settings.db_path)
    out = do_backup(store, dest or settings.data_dir / "backups")
    out["path"] = _secure(out["path"], encrypt, passphrase_env)
    out["encrypted"] = encrypt
    typer.echo(json.dumps(out))
    store.close()


@app.command()
def restore(
    backup_file: Path,
    passphrase_env: Annotated[str | None, typer.Option(help="NAME of an env var holding the passphrase (default: prompt)")] = None,
) -> None:
    """Restore a backup (.db or encrypted .enc). The current database is moved aside, never overwritten."""
    import tempfile

    from .crypto import CryptoError, decrypt_file, get_passphrase
    from .ops import restore as do_restore

    settings = Settings()
    if backup_file.suffix == ".enc":
        with tempfile.TemporaryDirectory() as td:
            try:
                plain = decrypt_file(backup_file, Path(td) / "restored.db", get_passphrase(passphrase_env))
            except CryptoError as e:
                typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
                raise typer.Exit(2) from e
            typer.echo(json.dumps(do_restore(settings.db_path, plain, _graph(settings)), indent=2))
        return
    typer.echo(json.dumps(do_restore(settings.db_path, backup_file, _graph(settings)), indent=2))


@app.command()
def export(
    fmt: Annotated[str, typer.Option("--format")] = "json",
    raw: bool = False,
    sensitive: bool = False,
    encrypt: Annotated[bool, typer.Option(help="AES-256-GCM encrypt with a passphrase")] = False,
    passphrase_env: Annotated[str | None, typer.Option(help="NAME of an env var holding the passphrase (default: prompt)")] = None,
) -> None:
    """Export memories (and optionally the raw archive) to the exports folder."""
    from .ops import export_memory

    settings = Settings()
    store = Store(settings.db_path)
    out = export_memory(store, settings.data_dir / "exports", fmt, include_raw=raw, include_sensitive=sensitive)
    out["path"] = _secure(out["path"], encrypt, passphrase_env)
    out["encrypted"] = encrypt
    typer.echo(json.dumps(out))
    store.close()


@app.command()
def lock(passphrase_env: Annotated[str | None, typer.Option(help="NAME of an env var holding the passphrase")] = None) -> None:
    """Encrypt the database at rest (membridge.db -> membridge.db.enc) and remove the plaintext. Unlock before use."""
    from .crypto import CryptoError, get_passphrase, lock_database

    s = Settings()
    try:
        enc = lock_database(s.db_path, get_passphrase(passphrase_env, confirm=passphrase_env is None))
    except (CryptoError, FileNotFoundError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    typer.echo(f"locked: {enc}  (stop the dashboard/MCP server first; run `membridge unlock` before using MemBridge again)")


@app.command()
def unlock(passphrase_env: Annotated[str | None, typer.Option(help="NAME of an env var holding the passphrase")] = None) -> None:
    """Decrypt the vault back to membridge.db."""
    from .crypto import CryptoError, get_passphrase, unlock_database

    try:
        unlock_database(Settings().db_path, get_passphrase(passphrase_env))
    except (CryptoError, FileNotFoundError) as e:
        typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from e
    typer.echo("unlocked")


@app.command()
def health() -> None:
    """Check SQLite, FalkorDB, memory and disk."""
    from .ops import health as do_health

    typer.echo(json.dumps(do_health(Settings()), indent=2))


@app.command()
def mcp(transport: str = "stdio", port: int = 8765) -> None:
    """Run the MCP server (stdio for local clients; http binds 127.0.0.1 and needs MEMBRIDGE_MCP_TOKEN)."""
    from .mcp_server import main as mcp_main

    mcp_main(["--transport", transport, "--port", str(port)])


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    """Run the dashboard/API (bound to localhost)."""
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1") and os.environ.get("MEMBRIDGE_IN_CONTAINER") != "1":
        typer.secho("Refusing to bind a non-loopback address (set only inside the official container).", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    import secrets

    token = os.environ.get("MEMBRIDGE_DASHBOARD_TOKEN") or secrets.token_urlsafe(24)
    os.environ["MEMBRIDGE_DASHBOARD_TOKEN"] = token  # the app factory below reads it
    typer.echo(f"\nOpen the dashboard (this link carries this launch's access token):\n  http://{host}:{port}/?token={token}\n")
    uvicorn.run("membridge.api.app:create_app", factory=True, host=host, port=port, log_level="info")


@app.command()
def resolve(dry_run: bool = False) -> None:
    """Merge alias entities (Postgres/PostgreSQL/pg). Reversible; uncertain pairs go to the review queue."""
    from .resolve.entities import resolve_entities

    store = Store(Settings().db_path)
    rep = resolve_entities(store, dry_run=dry_run)
    typer.echo(json.dumps(rep.__dict__, indent=2, default=str))
    store.close()


@app.command("entity-review")
def entity_review() -> None:
    """List pending entity pairs (uncertain matches) and recent merges."""
    store = Store(Settings().db_path)
    rows = store.conn.execute(
        "SELECT a.name AS a, b.name AS b, c.score, c.a_id, c.b_id FROM entity_link_candidates c "
        "JOIN entities a ON a.id=c.a_id JOIN entities b ON b.id=c.b_id WHERE c.status='pending' ORDER BY c.score DESC"
    ).fetchall()
    typer.echo(json.dumps({"pending": [dict(r) for r in rows],
                           "merges": [dict(r) for r in store.conn.execute(
                               "SELECT id, absorbed_id, into_id, score, method, decided_by FROM entity_merges WHERE reversed_at IS NULL ORDER BY id DESC LIMIT 20")]}, indent=2))  # fmt: skip
    store.close()


@app.command()
def communities() -> None:
    """Detect entity communities and (re)build the hierarchical summaries."""
    from .graphalgo.communities import build_communities
    from .memory.community_summaries import refresh_community_summaries

    store = Store(Settings().db_path)
    rep = build_communities(store)
    typer.echo(json.dumps({"method": rep.method, "levels": rep.levels, "new": len(rep.new_ids), "kept": len(rep.kept_ids),
                           "removed": len(rep.removed_ids), "summaries": refresh_community_summaries(store)}, indent=2))  # fmt: skip
    store.close()


@app.command("embed")
def embed_cmd(rebuild: bool = False, to_graph: bool = False) -> None:
    """Build the optional semantic index (MEMBRIDGE_EMBEDDER=fastembed). --rebuild recomputes from canonical text."""
    from .embed.embedder import build_embedder
    from .embed.index import index_embeddings, rebuild_embeddings

    settings = Settings()
    emb = build_embedder(settings.embedder if settings.embedder != "none" else "fastembed")
    assert emb is not None
    store = Store(settings.db_path)
    out = rebuild_embeddings(store, emb) if rebuild else index_embeddings(store, emb)
    if to_graph and (g := _graph(settings)).available():
        out["mirrored_to_graph"] = g.sync_embeddings(store, emb.name, emb.dim)
    typer.echo(json.dumps({"model": emb.name, "dim": emb.dim, **out}))
    store.close()


@app.command("ppr-cache")
def ppr_cache_cmd() -> None:
    """(Re)build the precomputed PageRank vectors that make searches ~free (done automatically by `extract`)."""
    from .graphalgo.ppr_cache import build_ppr_cache

    store = Store(Settings().db_path)
    typer.echo(json.dumps(build_ppr_cache(store)))
    store.close()


@app.command("gold-init")
def gold_init_cmd(n: int = 25) -> None:
    """Draft ~N questions with candidate answers into eval/gold_real.json for YOU to judge (never committed)."""
    from .eval_real import gold_init
    from .retrieval.engine import Retriever

    settings = Settings()
    store = Store(settings.db_path)
    out = Path("eval/gold_real.json")
    typer.echo(json.dumps(gold_init(store, Retriever(store), out, n)))
    typer.echo(f"Open {out}, set relevant=true on the correct pool items, then run: membridge eval-real")
    store.close()


@app.command("eval-real")
def eval_real_cmd(gold: Path = Path("eval/gold_real.json")) -> None:
    """Score retrieval against YOUR judgments (refuses to report until enough questions are judged)."""
    from .eval_real import eval_real
    from .retrieval.engine import Retriever

    store = Store(Settings().db_path)
    typer.echo(json.dumps(eval_real(Retriever(store), gold), indent=2))
    store.close()
