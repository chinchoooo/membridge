"""MemBridge command line interface."""

from __future__ import annotations

import json
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
def ask(query: str, budget: int = 1500, graph: bool = True) -> None:
    """Retrieve context for a question (what an AI client would get)."""
    from .retrieval.engine import Retriever

    settings = Settings()
    store = Store(settings.db_path)
    g = _graph(settings) if graph else None
    typer.echo(json.dumps(Retriever(store, g).search(query, token_budget=budget), indent=2, default=str))
    store.close()


@app.command()
def backup(dest: Annotated[Path | None, typer.Option(help="Backup folder")] = None) -> None:
    """Consistent copy of the database (the graph is rebuildable)."""
    from .ops import backup as do_backup

    settings = Settings()
    store = Store(settings.db_path)
    typer.echo(json.dumps(do_backup(store, dest or settings.data_dir / "backups")))
    store.close()


@app.command()
def restore(backup_file: Path) -> None:
    """Restore a backup. The current database is moved aside, never overwritten."""
    from .ops import restore as do_restore

    settings = Settings()
    typer.echo(json.dumps(do_restore(settings.db_path, backup_file, _graph(settings)), indent=2))


@app.command()
def export(fmt: Annotated[str, typer.Option("--format")] = "json", raw: bool = False,
           sensitive: bool = False) -> None:  # fmt: skip
    """Export memories (and optionally the raw archive) to the exports folder."""
    from .ops import export_memory

    settings = Settings()
    store = Store(settings.db_path)
    typer.echo(json.dumps(export_memory(store, settings.data_dir / "exports", fmt, include_raw=raw, include_sensitive=sensitive)))
    store.close()


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

    if host not in ("127.0.0.1", "localhost", "::1"):
        typer.secho("Refusing to bind a non-loopback address.", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
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
