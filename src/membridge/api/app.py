"""Dashboard/API backend (FastAPI). Local only.

Hardening: loopback bind (enforced by the CLI), a per-launch access token (an HttpOnly SameSite=Strict cookie for the
UI, a bearer header for scripts: another program on this machine, or a web page, cannot just call the API), a Host-header
allow-list (blocks DNS rebinding), an Origin check, no CORS, and every mutating request must carry a custom header.
Destructive actions use a two-step confirmation that only this (human) interface can complete.
"""

from __future__ import annotations

import collections
import hmac
import json
import os
import re
import secrets
import tempfile
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
from ..memory import access
from ..memory.service import MemoryService, ServiceError
from ..ops import backup as do_backup
from ..ops import export_memory, health
from ..store.graph import GraphStore
from ..store.sqlite import Store

MUTATING_HEADER = "x-membridge"
COOKIE = "mb_token"
PRIV_COOKIE = "mb_priv"  # proof that the PIN was entered recently
PRIV_TTL = 600
# Hosted sandbox (synthetic data only): everything is read-only except these, and these leak nothing about the server.
SANDBOX_POST = {"/api/retrieve", "/api/private/unlock", "/api/private/lock"}
SANDBOX_HIDDEN = {"/api/health", "/api/clients", "/api/audit/verify", "/api/forget-requests", "/api/private/status"}
SANDBOX_RATE = (90, 60.0)  # requests per window (seconds) per client address
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                               "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
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


class PrivateIn(BaseModel):
    private: bool = True


class UnlockIn(BaseModel):
    pin: str = Field(min_length=1, max_length=200)


class RevokeIn(BaseModel):
    client: str | None = Field(None, max_length=80)


class ExportIn(BaseModel):
    format: str = "json"
    include_raw: bool = False
    include_sensitive: bool = False


def create_app(settings: Settings | None = None, *, token: str | None = None, extra_hosts: frozenset[str] = frozenset(),
               sandbox: bool | None = None) -> FastAPI:  # fmt: skip
    """`sandbox=True` (or MEMBRIDGE_SANDBOX=1) is the public demo: a throw-away store of SYNTHETIC data, read-only for visitors, no token,
    rate-limited. It never opens the user's own data directory."""
    sandbox = sandbox if sandbox is not None else os.environ.get("MEMBRIDGE_SANDBOX") == "1"
    s = Settings(data_dir=Path(tempfile.mkdtemp(prefix="membridge-sandbox-"))) if sandbox else (settings or Settings())
    store = Store(s.db_path)
    if sandbox:
        from ..demo_data import SANDBOX_PIN, seed_demo_store

        seed_demo_store(store)
        access.set_pin(store, SANDBOX_PIN)
    graph = GraphStore(s.falkor_host, s.falkor_port, s.falkor_graph)
    svc = MemoryService(store, graph, actor="dashboard", human=True)
    jobs: dict[str, dict[str, Any]] = {}
    app = FastAPI(title="MemBridge", docs_url=None, redoc_url=None, openapi_url=None)
    access_token = token or os.environ.get("MEMBRIDGE_DASHBOARD_TOKEN") or secrets.token_urlsafe(24)
    if len(access_token) < 16:
        raise ValueError("the dashboard token must be at least 16 characters")
    app.state.token = access_token
    app.state.store, app.state.svc = store, svc
    # the public hostname: named explicitly, or announced by the platform (Render sets RENDER_EXTERNAL_HOSTNAME, Fly sets FLY_APP_NAME)
    announced = [os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")] + ([f"{os.environ['FLY_APP_NAME']}.fly.dev"] if os.environ.get("FLY_APP_NAME") else [])
    public = frozenset(h.strip().lower() for h in [*os.environ.get("MEMBRIDGE_PUBLIC_HOST", "").split(","), *announced] if h.strip())
    hosts = LOOPBACK | extra_hosts | public
    trust_proxy = os.environ.get("MEMBRIDGE_TRUST_PROXY") == "1"  # behind a platform proxy the real client is the first X-Forwarded-For
    hits: dict[str, collections.deque[float]] = {}

    def rate_limited(request: Request) -> bool:
        fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip() if trust_proxy else ""
        ip = fwd or (request.client.host if request.client else "?")
        now, q = time.monotonic(), hits.setdefault(ip, collections.deque())
        while q and now - q[0] > SANDBOX_RATE[1]:
            q.popleft()
        if len(q) >= SANDBOX_RATE[0]:
            return True
        q.append(now)
        if len(hits) > 5000:  # bound the table
            for k in [k for k, v in hits.items() if not v or now - v[-1] > SANDBOX_RATE[1]][:2500]:
                hits.pop(k, None)
        return False

    def priv_cookie() -> str:
        exp = int(time.time()) + PRIV_TTL
        return f"{exp}.{hmac.new(access_token.encode(), f'priv:{exp}'.encode(), 'sha256').hexdigest()}"

    def require_private_session(request: Request) -> None:
        """Seeing private memories in the dashboard needs the PIN, typed within the last ten minutes. The dashboard token
        alone (which a log or a neighbouring process could expose) is not enough."""
        v = request.cookies.get(PRIV_COOKIE, "")
        try:
            exp_s, sig = v.split(".", 1)
            ok = int(exp_s) > time.time() and hmac.compare_digest(
                sig, hmac.new(access_token.encode(), f"priv:{int(exp_s)}".encode(), "sha256").hexdigest())
        except ValueError:
            ok = False
        if not ok:
            raise HTTPException(403, "Private memories need your PIN: unlock them in Maintenance -> Private memories.")

    def authorised(request: Request) -> bool:
        if sandbox:
            return True  # public by design; protected by being read-only, synthetic and rate-limited
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
        path = request.url.path
        if sandbox and path.startswith("/api/"):
            if path in SANDBOX_HIDDEN or path.startswith(("/api/exports", "/api/imports")):
                return JSONResponse({"error": "not available in the sandbox"}, status_code=404)
            if request.method not in ("GET", "HEAD") and path not in SANDBOX_POST:
                return JSONResponse({"error": "This is a read-only sandbox on synthetic data. Install MemBridge locally to change things: see the README."}, status_code=403)
            if rate_limited(request):
                return JSONResponse({"error": "slow down a little"}, status_code=429, headers={"Retry-After": "30"})
        if request.url.path == "/" and request.query_params.get("token") and not sandbox:  # the link `membridge serve` prints
            if hmac.compare_digest(request.query_params["token"], access_token):
                r = RedirectResponse("/", status_code=303)
                r.set_cookie(COOKIE, access_token, httponly=True, samesite="strict", path="/")
                return r
            return JSONResponse({"error": "wrong token"}, status_code=401)
        if request.url.path.startswith("/api/") and request.url.path != "/api/ping" and not authorised(request):
            return JSONResponse({"error": "unauthorised: open the link printed by `membridge serve`"}, status_code=401)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and MUTATING_HEADER not in request.headers:
            return JSONResponse({"error": f"missing {MUTATING_HEADER} header"}, status_code=403)
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    @app.exception_handler(ServiceError)
    async def _svc_err(_: Request, e: ServiceError) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=400)

    # ---- overview / health -------------------------------------------------------
    @app.get("/api/sandbox")
    def api_sandbox() -> dict[str, Any]:
        if not sandbox:
            return {"sandbox": False}
        from ..demo_data import SANDBOX_PIN

        return {"sandbox": True, "pin": SANDBOX_PIN, "note": "Synthetic data, read-only. The PIN is public here; on your own machine it is yours alone."}

    @app.get("/api/ping")
    def api_ping() -> dict[str, bool]:  # liveness only (container healthcheck): reveals nothing about the data
        return {"ok": True}

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
    def memories(request: Request, q: str = "", category: str = "", status: str = "", verification: str = "", limit: int = 50, offset: int = 0,
                 show_private: bool = False) -> dict[str, Any]:  # fmt: skip
        if show_private:
            require_private_session(request)
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
    def memory_detail(memory_id: str, request: Request) -> dict[str, Any]:
        row = store.conn.execute("SELECT sensitivity FROM memories WHERE id=? AND status!='deleted'", (memory_id,)).fetchone()
        if row is not None and row["sensitivity"] != "normal":
            require_private_session(request)
        return {"memory": svc.get_memory(memory_id, True), "provenance": svc.trace_memory_provenance(memory_id, True),
                "related": svc.get_related_memories(memory_id, 1, True)}

    @app.post("/api/memories")
    def memory_create(body: MemoryIn) -> dict[str, Any]:
        return svc.save_memory(body.content, body.category, body.source, body.confidence, "dashboard", body.project)

    @app.patch("/api/memories/{memory_id}")
    def memory_patch(memory_id: str, body: MemoryPatch) -> dict[str, Any]:
        return svc.update_memory(memory_id, body.changes, body.reason, actor="dashboard")

    @app.post("/api/memories/{memory_id}/private")
    def memory_private(memory_id: str, body: PrivateIn, request: Request) -> dict[str, Any]:
        if not body.private:
            require_private_session(request)  # making something visible to assistants again is the sensitive direction
        return svc.set_private(memory_id, body.private)

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
    def conversation(cid: str, request: Request, show_private: bool = False) -> dict[str, Any]:
        if show_private:
            require_private_session(request)
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

        if sandbox and body.use_vector:
            raise HTTPException(400, "semantic search is off in the sandbox (it needs a model download)")
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

    @app.post("/api/private/unlock")
    def private_unlock(body: UnlockIn) -> JSONResponse:
        if not access.has_pin(store):
            raise HTTPException(400, "set a PIN first")
        try:
            ok = access.verify_pin(store, body.pin)
        except access.AccessError as e:
            raise HTTPException(400, str(e)) from e
        if not ok:
            raise HTTPException(400, "wrong PIN")
        r = JSONResponse({"unlocked_for_seconds": PRIV_TTL})
        r.set_cookie(PRIV_COOKIE, priv_cookie(), httponly=True, samesite="strict", path="/", max_age=PRIV_TTL)
        return r

    @app.post("/api/private/lock")
    def private_lock() -> JSONResponse:
        r = JSONResponse({"locked": True})
        r.delete_cookie(PRIV_COOKIE, path="/")
        return r

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
    def export(body: ExportIn, request: Request) -> dict[str, Any]:
        if body.include_sensitive:
            require_private_session(request)
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

    # the bundle shipped inside the package (works after `pip install` / `uvx`), else a fresh `npm run build` in a source checkout
    web = Path(__file__).resolve().parents[1] / "_web"
    if not web.is_dir():
        web = Path(__file__).resolve().parents[3] / "web" / "dist"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app
