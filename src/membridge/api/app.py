"""Dashboard/API backend (FastAPI). Local only.

Hardening: loopback bind (enforced by the CLI), a per-launch access token (an HttpOnly SameSite=Strict cookie for the
UI, a bearer header for scripts: another program on this machine, or a web page, cannot just call the API), a Host-header
allow-list (blocks DNS rebinding), an Origin check, no CORS, and every mutating request must carry a custom header.
Destructive actions use a two-step confirmation that only this (human) interface can complete.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Settings
from ..importers.service import ImportError_, run_import
from ..memory.service import MemoryService, ServiceError
from ..ops import backup as do_backup
from ..ops import export_memory, health
from ..store.graph import GraphStore
from ..store.sqlite import Store

MUTATING_HEADER = "x-membridge"
COOKIE = "mb_token"
LOOPBACK = frozenset({"127.0.0.1", "localhost", "[::1]"})


class RetrieveIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    token_budget: int = Field(1500, ge=200, le=8000)
    limit: int = Field(10, ge=1, le=50)
    use_graph: bool = True
    use_vector: bool = False
    as_of: str | None = None
    known_at: str | None = None
    category: str | None = None
    project: str | None = None


class PairDecision(BaseModel):
    a_id: str
    b_id: str
    accept: bool


class SplitIn(BaseModel):
    merge_id: int


class ResolveIn(BaseModel):
    dry_run: bool = True


class ContradictionIn(BaseModel):
    keep_id: str
    drop_id: str
    reason: str = Field("resolved in dashboard", max_length=300)


class MemoryIn(BaseModel):
    content: str = Field(min_length=3, max_length=1000)
    category: str = "fact"
    source: str = "user_stated"
    confidence: float = Field(0.9, ge=0, le=1)
    project: str | None = None


class MemoryPatch(BaseModel):
    changes: dict[str, Any]
    reason: str = Field("edited in dashboard", max_length=300)


class ImportIn(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    dry_run: bool = True


class ConsolidateIn(BaseModel):
    scope: str = "incremental"


class PinIn(BaseModel):
    new_pin: str = Field(min_length=6, max_length=200)
    old_pin: str | None = Field(None, max_length=200)


class GrantIn(BaseModel):
    pin: str = Field(min_length=1, max_length=200)
    client: str = Field(min_length=1, max_length=80)
    minutes: int = Field(10, ge=1, le=120)
    uses: int = Field(25, ge=1, le=200)


class RevokeIn(BaseModel):
    client: str | None = Field(None, max_length=80)


class ExportIn(BaseModel):
    format: str = "json"
    include_raw: bool = False
    include_sensitive: bool = False


def create_app(settings: Settings | None = None, *, token: str | None = None, extra_hosts: frozenset[str] = frozenset()) -> FastAPI:
    s = settings or Settings()
    store = Store(s.db_path)
    graph = GraphStore(s.falkor_host, s.falkor_port, s.falkor_graph)
    svc = MemoryService(store, graph, actor="dashboard", human=True)
    jobs: dict[str, dict[str, Any]] = {}
    app = FastAPI(title="MemBridge", docs_url=None, redoc_url=None, openapi_url=None)
    access_token = token or os.environ.get("MEMBRIDGE_DASHBOARD_TOKEN") or secrets.token_urlsafe(24)
    if len(access_token) < 16:
        raise ValueError("the dashboard token must be at least 16 characters")
    app.state.token = access_token
    app.state.store, app.state.svc = store, svc
    hosts = LOOPBACK | extra_hosts

    def authorised(request: Request) -> bool:
        bearer = request.headers.get("authorization", "")
        presented = bearer[7:] if bearer.lower().startswith("bearer ") else request.cookies.get(COOKIE, "")
        return bool(presented) and hmac.compare_digest(presented, access_token)

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower() if not (request.headers.get("host") or "").startswith("[") else (
            (request.headers.get("host") or "").split("]")[0] + "]")  # fmt: skip
        if host not in hosts:
            return JSONResponse({"error": "host not allowed"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != "null" and (origin.split("://", 1)[-1].rsplit(":", 1)[0].lower() not in hosts):
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
        if request.url.path == "/" and request.query_params.get("token"):  # the link `membridge serve` prints
            if hmac.compare_digest(request.query_params["token"], access_token):
                r = RedirectResponse("/", status_code=303)
                r.set_cookie(COOKIE, access_token, httponly=True, samesite="strict", path="/")
                return r
            return JSONResponse({"error": "wrong token"}, status_code=401)
        if request.url.path.startswith("/api/") and not authorised(request):
            return JSONResponse({"error": "unauthorised: open the link printed by `membridge serve`"}, status_code=401)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and MUTATING_HEADER not in request.headers:
            return JSONResponse({"error": f"missing {MUTATING_HEADER} header"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ServiceError)
    async def _svc_err(_: Request, e: ServiceError) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=400)

    # ---- overview / health -------------------------------------------------------
    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        return health(s)

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        stats = svc.get_memory_statistics()
        c = store.conn
        with store.lock:
            recent = [dict(r) for r in c.execute(
                "SELECT id, adapter, status, started_at, finished_at, last_index FROM import_jobs ORDER BY started_at DESC LIMIT 8")]  # fmt: skip
            errors = [dict(r) for r in c.execute("SELECT job_id, item_index, ref, error FROM import_errors ORDER BY id DESC LIMIT 20")]
            lim = c.execute("SELECT COUNT(*) FROM messages WHERE limitations_json != '[]'").fetchone()[0]
            ext = c.execute("SELECT COUNT(DISTINCT message_id) FROM extraction_state").fetchone()[0]
        return {"stats": stats, "imports": recent, "import_errors": errors, "messages_with_limitations": lim,
                "messages_extracted": ext, "live_jobs": list(jobs.values())[-5:]}  # fmt: skip

    # ---- memories ------------------------------------------------------------------
    @app.get("/api/memories")
    def memories(q: str = "", category: str = "", status: str = "", verification: str = "", limit: int = 50, offset: int = 0,
                 show_private: bool = False) -> dict[str, Any]:  # fmt: skip
        limit = max(1, min(limit, 200))
        with store.lock:
            if q.strip():
                f = {k: v for k, v in (("category", category), ("status", status), ("verification", verification)) if v}
                pkg = svc.search_memory(q, limit, f, include_sensitive=show_private)
                return {"items": pkg["memories"], "total": len(pkg["memories"]), "search": pkg}
            sql, args = "SELECT id, statement, category, status, verification, confidence, score, mention_count, valid_from, sensitivity FROM memories WHERE status!='deleted'", []
            if not show_private:
                sql += " AND sensitivity='normal'"
            for col, val in (("category", category), ("status", status), ("verification", verification)):
                if val:
                    sql += f" AND {col}=?"
                    args.append(val)
            total = store.conn.execute(f"SELECT COUNT(*) FROM ({sql})", args).fetchone()[0]
            rows = store.conn.execute(sql + " ORDER BY score DESC LIMIT ? OFFSET ?", [*args, limit, max(0, offset)]).fetchall()
        return {"items": [dict(r) for r in rows], "total": total}

    @app.get("/api/memories/{memory_id}")
    def memory_detail(memory_id: str) -> dict[str, Any]:
        return {"memory": svc.get_memory(memory_id, True), "provenance": svc.trace_memory_provenance(memory_id, True),
                "related": svc.get_related_memories(memory_id, 1, True)}

    @app.post("/api/memories")
    def memory_create(body: MemoryIn) -> dict[str, Any]:
        return svc.save_memory(body.content, body.category, body.source, body.confidence, "dashboard", body.project)

    @app.patch("/api/memories/{memory_id}")
    def memory_patch(memory_id: str, body: MemoryPatch) -> dict[str, Any]:
        return svc.update_memory(memory_id, body.changes, body.reason, actor="dashboard")

    @app.delete("/api/memories/{memory_id}")
    def memory_forget(memory_id: str, confirm_token: str | None = None) -> dict[str, Any]:
        return svc.forget_memory(memory_id, confirm_token, actor="dashboard")

    # ---- graph ----------------------------------------------------------------------
    @app.get("/api/graph/neighborhood")
    def neighborhood(center: str, depth: int = 1) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", center):
            raise HTTPException(400, "invalid center id")
        if not graph.available():
            raise HTTPException(503, "FalkorDB is not reachable. Start it with: docker compose up -d")
        return graph.neighborhood(center, max(1, min(depth, 2)))

    @app.get("/api/entities")
    def entities(q: str = "", limit: int = 30) -> dict[str, Any]:
        with store.lock:
            rows = store.conn.execute(
                "SELECT e.id, e.name, e.type, (SELECT COUNT(*) FROM memory_entities me WHERE me.entity_id=e.id) AS memories "
                "FROM live_entities e WHERE e.name LIKE ? ORDER BY memories DESC LIMIT ?", (f"%{q}%", max(1, min(limit, 100)))).fetchall()  # fmt: skip
        return {"items": [dict(r) for r in rows]}

    @app.get("/api/projects")
    def projects() -> dict[str, Any]:
        return svc.list_projects()

    @app.get("/api/projects/{name}")
    def project(name: str) -> dict[str, Any]:
        return svc.get_project_context(name)

    @app.get("/api/conversations/{cid}")
    def conversation(cid: str, show_private: bool = False) -> dict[str, Any]:
        return svc.get_conversation(cid, 120, show_private)

    @app.get("/api/contradictions")
    def contradictions() -> dict[str, Any]:
        return svc.list_unresolved()

    # ---- retrieval tester ----------------------------------------------------------------
    @app.post("/api/retrieve")
    def retrieve(body: RetrieveIn) -> dict[str, Any]:
        from ..embed.embedder import build_embedder
        from ..memory.filters import build_filters
        from ..retrieval.engine import RetrievalConfig, Retriever

        filters = build_filters({k: v for k, v in (("as_of", body.as_of), ("known_at", body.known_at), ("category", body.category), ("project", body.project)) if v})
        with store.lock:
            emb = None
            if body.use_vector:
                try:
                    emb = build_embedder(s.embedder if s.embedder != "none" else "fastembed")
                except Exception as e:  # optional extra not installed / model unavailable
                    raise HTTPException(400, f"semantic search unavailable: {type(e).__name__}") from e
            r = Retriever(store, graph if body.use_graph else None, RetrievalConfig(use_vector=emb is not None), emb)
            r.actor = "dashboard"
            return r.search(body.query, limit=body.limit, token_budget=body.token_budget, filters=filters)

    @app.get("/api/overview/global")
    def overview_global(q: str = "") -> dict[str, Any]:
        return svc.get_overview(q or None)

    @app.get("/api/entity-review")
    def entity_review() -> dict[str, Any]:
        return svc.entity_review()

    @app.post("/api/entity-review/decide")
    def entity_decide(body: PairDecision) -> dict[str, Any]:
        return svc.decide_entity_pair(body.a_id, body.b_id, body.accept)

    @app.post("/api/entity-review/split")
    def entity_split(body: SplitIn) -> dict[str, Any]:
        return svc.split_entity_merge(body.merge_id)

    @app.post("/api/resolve")
    def resolve(body: ResolveIn) -> dict[str, Any]:
        return svc.resolve_entities(body.dry_run)

    @app.post("/api/contradictions/resolve")
    def contradiction_resolve(body: ContradictionIn) -> dict[str, Any]:
        return svc.resolve_contradiction(body.keep_id, body.drop_id, body.reason, actor="dashboard")

    # ---- consent for private memories ---------------------------------------------------------
    @app.get("/api/private/status")
    def private_status() -> dict[str, Any]:
        return svc.list_grants()

    @app.post("/api/private/pin")
    def private_pin(body: PinIn) -> dict[str, Any]:
        return svc.set_pin(body.new_pin, body.old_pin)

    @app.post("/api/private/grant")
    def private_grant(body: GrantIn) -> dict[str, Any]:
        return svc.grant_sensitive(body.pin, body.client, body.minutes, body.uses)

    @app.post("/api/private/revoke")
    def private_revoke(body: RevokeIn) -> dict[str, Any]:
        return svc.revoke_sensitive(body.client)

    @app.get("/api/forget-requests")
    def forget_requests() -> dict[str, Any]:
        return svc.list_forget_requests()

    @app.get("/api/audit/verify")
    def audit_verify() -> dict[str, Any]:
        from ..store.sqlite import verify_audit_chain

        with store.lock:
            store.flush_audit()
            return verify_audit_chain(store)

    # ---- maintenance -----------------------------------------------------------------------
    @app.post("/api/imports")
    def start_import(body: ImportIn) -> dict[str, Any]:
        path = Path(body.path).expanduser()
        if not path.exists():
            raise HTTPException(400, f"path not found: {path}")
        job_id = "ui_" + uuid.uuid4().hex[:8]
        jobs[job_id] = {"id": job_id, "path": str(path), "dry_run": body.dry_run, "status": "running", "started": time.time()}

        def work() -> None:
            st = None if body.dry_run else Store(s.db_path)  # own connection: imports run in a thread
            try:
                m = run_import(st, path, dry_run=body.dry_run, batch_size=s.import_batch_size, report_dir=s.import_dir)
                jobs[job_id].update(status=m.status, manifest=json.loads(m.to_json()))
            except ImportError_ as e:
                jobs[job_id].update(status="failed", error=str(e))
            except Exception as e:  # keep the server alive; the job reports the failure
                jobs[job_id].update(status="failed", error=f"{type(e).__name__}: {e}")
            finally:
                if st:
                    st.close()

        threading.Thread(target=work, daemon=True).start()
        return jobs[job_id]

    @app.get("/api/imports/{job_id}")
    def import_status(job_id: str) -> dict[str, Any]:
        if job_id not in jobs:
            raise HTTPException(404, "unknown job")
        return jobs[job_id]

    @app.post("/api/consolidate")
    def consolidate(body: ConsolidateIn) -> dict[str, Any]:
        return svc.run_consolidation(body.scope)

    @app.post("/api/backup")
    def backup() -> dict[str, Any]:
        with store.lock:
            return do_backup(store, s.data_dir / "backups")

    @app.post("/api/export")
    def export(body: ExportIn) -> dict[str, Any]:
        with store.lock:
            return export_memory(store, s.data_dir / "exports", body.format, include_raw=body.include_raw, include_sensitive=body.include_sensitive)

    @app.get("/api/exports/{name}")
    def download(name: str) -> FileResponse:
        if not re.fullmatch(r"membridge-export-[0-9-]+\.(json|md)", name):
            raise HTTPException(400, "invalid file name")
        f = s.data_dir / "exports" / name
        if not f.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(f, filename=name)

    @app.get("/api/clients")
    def clients() -> dict[str, Any]:
        from ..clients import client_configs

        return client_configs(s)

    web = Path(__file__).resolve().parents[3] / "web" / "dist"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
