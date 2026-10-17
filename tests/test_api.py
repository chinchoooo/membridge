from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from membridge.api.app import create_app
from membridge.config import Settings
from membridge.importers.service import run_import
from membridge.memory.pipeline import run_extraction
from membridge.store.sqlite import Store

H = {"x-membridge": "1"}
TOKEN = "test-token-0123456789abcdef"


@pytest.fixture
def client(tmp_path: Path, export_file: Path) -> TestClient:
    s = Store(tmp_path / "d" / "membridge.db")
    run_import(s, export_file)
    run_extraction(s)
    s.close()
    settings = Settings(data_dir=tmp_path / "d", falkor_port=1)  # graph unreachable -> degraded
    c = TestClient(create_app(settings, token=TOKEN), base_url="http://127.0.0.1")
    c.headers["authorization"] = f"Bearer {TOKEN}"
    return c


def test_overview_counts(client: TestClient) -> None:
    d = client.get("/api/overview").json()
    assert d["stats"]["conversations"] == 6 and d["stats"]["memories"] >= 6 and d["stats"]["graph"] == "unavailable"
    assert d["stats"]["import_errors"] == 1 and d["messages_with_limitations"] >= 1


def test_host_header_and_csrf_guards(client: TestClient) -> None:
    assert client.get("/api/overview", headers={"host": "evil.example.com"}).status_code == 403  # DNS rebinding
    assert client.post("/api/consolidate", json={"scope": "summaries"}).status_code == 403  # no custom header
    assert client.post("/api/consolidate", json={"scope": "summaries"}, headers=H).status_code == 200


def test_memory_list_filter_detail_edit_and_two_step_delete(client: TestClient) -> None:
    items = client.get("/api/memories", params={"category": "decision"}).json()["items"]
    assert items and all(i["category"] == "decision" for i in items)
    mid = items[0]["id"]
    d = client.get(f"/api/memories/{mid}").json()
    assert d["provenance"]["sources"] and d["memory"]["id"] == mid
    r = client.patch(f"/api/memories/{mid}", json={"changes": {"verification": "user_verified"}, "reason": "confirmed"}, headers=H)
    assert r.status_code == 200
    bad = client.patch(f"/api/memories/{mid}", json={"changes": {"statement": "x" * 5000}}, headers=H)
    assert bad.status_code == 400 and "too long" in bad.json()["error"]
    first = client.delete(f"/api/memories/{mid}", headers=H).json()
    assert first["confirmation_required"] is True
    assert client.get(f"/api/memories/{mid}").status_code == 200  # still there
    done = client.delete(f"/api/memories/{mid}", params={"confirm_token": first["confirm_token"]}, headers=H).json()
    assert done == {"forgotten": mid}
    assert client.get(f"/api/memories/{mid}").status_code == 400


def test_retrieval_tester_returns_latency_tokens_and_evidence(client: TestClient) -> None:
    r = client.post("/api/retrieve", json={"query": "What database did I decide to use for Atlas?", "token_budget": 800}, headers=H).json()
    assert r["sufficient_evidence"] and r["stats"]["latency_ms"] >= 0 and r["stats"]["tokens_estimate"] <= 800
    assert r["memories"][0]["evidence"]


def test_graph_endpoint_reports_unavailable_cleanly(client: TestClient) -> None:
    r = client.get("/api/graph/neighborhood", params={"center": "user"})
    assert r.status_code == 503 and "docker compose" in r.json()["detail"]
    assert client.get("/api/graph/neighborhood", params={"center": "x;DROP"}).status_code == 400


def test_import_job_dry_run_via_api(client: TestClient, export_file: Path) -> None:
    import time

    j = client.post("/api/imports", json={"path": str(export_file), "dry_run": True}, headers=H).json()
    for _ in range(50):
        st = client.get(f"/api/imports/{j['id']}").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "completed" and st["manifest"]["conversations_new"] == 6
    assert client.post("/api/imports", json={"path": "C:/definitely/missing"}, headers=H).status_code == 400


def test_export_backup_and_client_instructions(client: TestClient) -> None:
    ex = client.post("/api/export", json={"format": "json"}, headers=H).json()
    assert ex["memories"] >= 6 and Path(ex["path"]).is_file()
    name = Path(ex["path"]).name
    assert client.get(f"/api/exports/{name}").status_code == 200
    assert client.get("/api/exports/..%2F..%2Fsecret").status_code in (400, 404)
    assert client.post("/api/backup", headers=H).json()["integrity"] == "ok"
    cl = client.get("/api/clients").json()
    ids = {c["id"]: c for c in cl["clients"]}
    assert "Verified" in ids["claude-code"]["status"] and "NOT verified" in ids["cursor"]["status"]
    assert "cannot capture" in cl["note"]


def test_as_of_retrieval_via_api(client: TestClient) -> None:
    now = client.post("/api/retrieve", json={"query": "What storage does Atlas use?", "limit": 5}, headers=H).json()
    old = client.post("/api/retrieve", json={"query": "Atlas storage database", "as_of": "2023-11-20"}, headers=H).json()
    assert "SQLite" in now["memories"][0]["statement"]
    assert any("PostgreSQL" in m["statement"] for m in old["memories"][:2]) and not any("switched" in m["statement"] for m in old["memories"])
    bad = client.post("/api/retrieve", json={"query": "x", "as_of": "not-a-date"}, headers=H)
    assert bad.status_code == 400


def test_global_overview_endpoint(client: TestClient) -> None:
    d = client.get("/api/overview/global").json()
    assert "areas" in d and "themes" in d and d["content_trust"]


def test_entity_review_loop_via_api(client: TestClient) -> None:
    r = client.post("/api/resolve", json={"dry_run": True}, headers=H).json()
    assert r["dry_run"] is True and "auto_merged" in r
    assert set(client.get("/api/entity-review").json()) == {"pending", "merges"}
    assert client.post("/api/entity-review/split", json={"merge_id": 999}, headers=H).status_code == 400


def test_contradiction_resolution_keeps_history_and_is_remembered(client: TestClient) -> None:
    items = client.get("/api/memories", params={"category": "decision"}).json()["items"]
    keep, drop = items[0]["id"], items[1]["id"]
    # two memories that were never in conflict cannot be "resolved"
    assert client.post("/api/contradictions/resolve", json={"keep_id": keep, "drop_id": drop}, headers=H).status_code == 400
    st = client.app.state.store  # type: ignore[attr-defined]
    st.conn.execute("UPDATE memories SET status='unresolved' WHERE id IN (?,?)", (keep, drop))
    st.conn.execute("INSERT INTO memory_links(src_id, rel, dst_id, reason, created_at) VALUES (?,?,?,?,0)", (keep, "CONTRADICTS", drop, "test"))
    st.conn.commit()
    out = client.post("/api/contradictions/resolve", json={"keep_id": keep, "drop_id": drop}, headers=H)
    assert out.status_code == 200 and out.json() == {"kept": keep, "superseded": drop}
    d = client.get(f"/api/memories/{drop}").json()["memory"]
    assert d["status"] == "superseded" and any(link["rel"] == "SUPERSEDES" for link in d["links"])
    assert client.post("/api/contradictions/resolve", json={"keep_id": keep, "drop_id": keep}, headers=H).status_code == 400


def test_api_needs_the_launch_token(client: TestClient) -> None:
    anon = TestClient(client.app, base_url="http://127.0.0.1")
    assert anon.get("/api/overview").status_code == 401  # another local program cannot just call the API
    assert anon.get("/api/overview", headers={"authorization": "Bearer wrong-token-0123456789"}).status_code == 401
    assert anon.get("/api/memories").status_code == 401
    assert anon.post("/api/backup", headers=H).status_code == 401
    # the link `membridge serve` prints sets an HttpOnly, SameSite=Strict cookie that then authorises the UI
    r = anon.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303 and "httponly" in r.headers["set-cookie"].lower() and "samesite=strict" in r.headers["set-cookie"].lower()
    assert anon.get("/api/overview").status_code == 200
    assert TestClient(client.app, base_url="http://127.0.0.1").get("/?token=nope-nope-nope-nope-nope").status_code == 401


def test_cross_origin_requests_are_refused(client: TestClient) -> None:
    assert client.post("/api/consolidate", json={"scope": "summaries"}, headers={**H, "origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/consolidate", json={"scope": "summaries"}, headers={**H, "origin": "http://127.0.0.1:8787"}).status_code == 200
    assert client.get("/api/overview", headers={"host": "testserver"}).status_code == 403  # not on the allow-list unless a test adds it


def test_private_memories_are_hidden_from_listings_until_asked(client: TestClient) -> None:
    c = client.post("/api/memories", json={"content": "I was diagnosed with an anxiety disorder", "category": "health", "source": "user_stated"}, headers=H)
    assert c.status_code == 200
    mid = c.json()["memory_id"]
    assert all(i["id"] != mid for i in client.get("/api/memories", params={"limit": 200}).json()["items"])
    assert any(i["id"] == mid for i in client.get("/api/memories", params={"limit": 200, "show_private": True}).json()["items"])
    assert client.get(f"/api/memories/{mid}").json()["memory"]["id"] == mid  # opening one deliberately is allowed for the user
    assert client.get("/api/audit/verify").json()["ok"] is True


def test_pin_grant_and_revoke_flow(client: TestClient) -> None:
    assert client.post("/api/private/grant", json={"pin": "123456", "client": "claude-code"}, headers=H).status_code == 400  # no PIN yet
    assert client.post("/api/private/pin", json={"new_pin": "12"}, headers=H).status_code == 422
    assert client.post("/api/private/pin", json={"new_pin": "246810"}, headers=H).status_code == 200
    assert client.post("/api/private/grant", json={"pin": "wrong-pin", "client": "claude-code"}, headers=H).status_code == 400
    ok = client.post("/api/private/grant", json={"pin": "246810", "client": "claude-code", "minutes": 5, "uses": 3}, headers=H)
    assert ok.status_code == 200 and ok.json()["uses"] == 3
    status = client.get("/api/private/status").json()
    assert status["pin_set"] and status["grants"][0]["active"]
    assert client.post("/api/private/revoke", json={"client": "claude-code"}, headers=H).json() == {"revoked": 1}
    assert not client.get("/api/private/status").json()["grants"][0]["active"]
