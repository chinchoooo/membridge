"""Operations: export, backup/restore, health. All local, none requires an AI provider."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

import psutil

from .config import Settings
from .store.graph import GraphStore
from .store.sqlite import Store


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def export_memory(
    store: Store, dest_dir: Path, fmt: str = "json", *, include_raw: bool = False, include_sensitive: bool = False
) -> dict[str, Any]:
    """Write a portable export to dest_dir and return its path and counts (never the contents)."""
    if fmt not in {"json", "markdown"}:
        raise ValueError("format must be 'json' or 'markdown'")
    c = store.conn
    sens = "" if include_sensitive else " AND sensitivity='normal'"
    mems = [dict(r) for r in c.execute(f"SELECT * FROM memories WHERE status!='deleted'{sens} ORDER BY IFNULL(valid_from,0)")]
    ids = {m["id"] for m in mems}
    data: dict[str, Any] = {
        "format": "membridge-export-1",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "memories": mems,
        "evidence": [dict(r) for r in c.execute("SELECT * FROM memory_evidence") if r["memory_id"] in ids],
        "links": [dict(r) for r in c.execute("SELECT * FROM memory_links") if r["src_id"] in ids and r["dst_id"] in ids],
        "entities": [dict(r) for r in c.execute("SELECT * FROM entities")],
        "memory_entities": [dict(r) for r in c.execute("SELECT * FROM memory_entities") if r["memory_id"] in ids],
        "summaries": [dict(r) for r in c.execute("SELECT * FROM summaries")],
    }
    if include_raw:
        data["conversations"] = [dict(r) for r in c.execute("SELECT * FROM conversations")]
        data["messages"] = [dict(r) for r in c.execute("SELECT * FROM messages")]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"membridge-export-{_stamp()}.{'json' if fmt == 'json' else 'md'}"
    if fmt == "json":
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        lines = ["# MemBridge export", ""]
        by: dict[str, list[dict[str, Any]]] = {}
        for m in mems:
            by.setdefault(m["category"], []).append(m)
        for cat, items in sorted(by.items()):
            lines += [f"## {cat}", ""]
            for m in items:
                lines.append(f"- [{m['status']}, {m['verification']}] {m['statement']}")
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
    store.audit("export", "export_memory", str(out), fmt=fmt, raw=include_raw)
    store.conn.commit()
    return {"path": str(out), "format": fmt, "memories": len(mems), "includes_raw_archive": include_raw}


def backup(store: Store, dest_dir: Path) -> dict[str, Any]:
    """Consistent online copy of the SQLite database (the system of record). The graph is rebuildable."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"membridge-{_stamp()}.db"
    dst = sqlite3.connect(out)
    try:
        store.conn.backup(dst)
        ok = dst.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        dst.close()
    if ok != "ok":
        out.unlink(missing_ok=True)
        raise RuntimeError(f"backup failed integrity check: {ok}")
    store.audit("backup", "backup", str(out))
    store.conn.commit()
    return {"path": str(out), "bytes": out.stat().st_size, "integrity": ok}


def restore(db_path: Path, backup_file: Path, graph: GraphStore | None = None) -> dict[str, Any]:
    """Replace the database with a backup. The current database is moved aside, never overwritten."""
    if not backup_file.is_file():
        raise FileNotFoundError(f"backup not found: {backup_file}")
    probe = sqlite3.connect(f"file:{backup_file}?mode=ro", uri=True)
    try:
        ok = probe.execute("PRAGMA integrity_check").fetchone()[0]
        probe.execute("SELECT COUNT(*) FROM memories")
    finally:
        probe.close()
    if ok != "ok":
        raise RuntimeError(f"backup is corrupt: {ok}")
    aside = None
    if db_path.exists():
        aside = db_path.with_suffix(f".pre-restore-{_stamp()}.db")
        shutil.move(str(db_path), aside)
        for suffix in ("-wal", "-shm"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
    shutil.copy2(backup_file, db_path)
    store = Store(db_path)
    rebuilt = None
    if graph is not None and graph.available():
        rebuilt = graph.rebuild(store)
    stats = store.stats()
    store.close()
    return {"restored_from": str(backup_file), "previous_database": str(aside) if aside else None, "stats": stats, "graph_rebuilt": rebuilt}


def health(settings: Settings) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True}
    try:
        store = Store(settings.db_path)
        out["sqlite"] = {"ok": True, "path": str(settings.db_path), "bytes": settings.db_path.stat().st_size, **store.stats()}
        store.close()
    except Exception as e:
        out["sqlite"] = {"ok": False, "error": str(e)}
        out["ok"] = False
    g = GraphStore(settings.falkor_host, settings.falkor_port, settings.falkor_graph)
    out["graph"] = {"ok": g.available(), "host": settings.falkor_host, "port": settings.falkor_port}
    p = psutil.Process()
    out["process"] = {"rss_mb": round(p.memory_info().rss / 2**20, 1), "cpu_percent": p.cpu_percent(interval=0.1)}
    vm = psutil.virtual_memory()
    out["system"] = {"ram_free_mb": round(vm.available / 2**20), "ram_total_mb": round(vm.total / 2**20),
                     "disk_free_gb": round(shutil.disk_usage(settings.data_dir.anchor or ".").free / 2**30, 1)}  # fmt: skip
    return out
