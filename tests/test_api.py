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


@pytest.fixture
def client(tmp_path: Path, export_file: Path) -> TestClient:
    s = Store(tmp_path / "d" / "membridge.db")
    run_import(s, export_file)
    run_extraction(s)
    s.close()
    settings = Settings(data_dir=tmp_path / "d", falkor_port=1)  # graph unreachable -> degraded
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


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
