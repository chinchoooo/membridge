"""The public demo: synthetic data only, read-only for strangers, rate-limited, and unable to reveal anything about the server."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from membridge.api.app import SANDBOX_POST, SANDBOX_RATE, create_app
from membridge.demo_data import SANDBOX_PIN

H = {"x-membridge": "1"}


@pytest.fixture(scope="module")
def sb() -> TestClient:
    return TestClient(create_app(sandbox=True), base_url="http://127.0.0.1")


def test_it_runs_on_synthetic_data_in_a_throwaway_folder(sb: TestClient) -> None:
    store = sb.app.state.store  # type: ignore[attr-defined]
    assert "membridge-sandbox-" in str(store.path) and str(Path.home() / ".membridge") not in str(store.path)
    assert sb.get("/api/sandbox").json()["sandbox"] is True and sb.get("/api/sandbox").json()["pin"] == SANDBOX_PIN
    names = {r[0] for r in store.conn.execute("SELECT title FROM conversations")}
    assert "Atlas project kickoff" in names  # the made-up story, nothing else


def test_anyone_can_read_without_a_token(sb: TestClient) -> None:
    anon = TestClient(sb.app, base_url="http://127.0.0.1")
    assert anon.get("/api/overview").status_code == 200
    items = anon.get("/api/memories", params={"limit": 100}).json()["items"]
    assert items and all(i["sensitivity"] == "normal" for i in items)  # private ones stay hidden by default


def test_nothing_can_be_changed_or_exfiltrated_by_a_visitor(sb: TestClient) -> None:
    for method, path, body in [
        ("post", "/api/memories", {"content": "defaced", "category": "fact", "source": "user_stated"}),
        ("post", "/api/imports", {"path": "C:/Windows", "dry_run": True}),
        ("post", "/api/backup", None), ("post", "/api/export", {"format": "json"}), ("post", "/api/consolidate", {"scope": "summaries"}),
        ("post", "/api/private/pin", {"new_pin": "999999"}), ("post", "/api/private/grant", {"pin": SANDBOX_PIN, "client": "x"}),
        ("post", "/api/contradictions/resolve", {"keep_id": "a", "drop_id": "b"}), ("post", "/api/resolve", {"dry_run": False}),
        ("patch", "/api/memories/x", {"changes": {"statement": "x y z"}, "reason": "r"}), ("delete", "/api/memories/x", None),
    ]:
        r = getattr(sb, method)(path, headers=H, **({"json": body} if body is not None else {}))
        assert r.status_code in (403, 404), (method, path, r.status_code)
    for path in ("/api/health", "/api/clients", "/api/exports/membridge-export-1.json", "/api/imports/x", "/api/private/status", "/api/audit/verify"):
        assert sb.get(path).status_code == 404, path  # nothing about the server: paths, memory use, jobs, grants
    assert set(SANDBOX_POST) == {"/api/retrieve", "/api/private/unlock", "/api/private/lock"}


def test_the_retrieval_tester_works_and_declines_what_it_does_not_know(sb: TestClient) -> None:
    ok = sb.post("/api/retrieve", json={"query": "What database does Atlas use now?"}, headers=H).json()
    assert ok["sufficient_evidence"] and "SQLite" in ok["memories"][0]["statement"]
    no = sb.post("/api/retrieve", json={"query": "What is my favourite pizza topping?"}, headers=H).json()
    assert no["sufficient_evidence"] is False and no["memories"] == []
    assert sb.post("/api/retrieve", json={"query": "x y", "use_vector": True}, headers=H).status_code == 400  # no model download on a public box


def test_the_private_gate_can_be_tried_with_the_published_pin(sb: TestClient) -> None:
    c = TestClient(sb.app, base_url="http://127.0.0.1")
    assert c.get("/api/memories", params={"show_private": True}).status_code == 403
    assert c.post("/api/private/unlock", json={"pin": "000000"}, headers=H).status_code == 400
    assert c.post("/api/private/unlock", json={"pin": SANDBOX_PIN}, headers=H).status_code == 200
    priv = [i for i in c.get("/api/memories", params={"show_private": True, "limit": 100}).json()["items"] if i["sensitivity"] != "normal"]
    assert len(priv) >= 3 and any("sertraline" in i["statement"] or "antidepressant" in i["statement"] for i in priv)


def test_it_is_rate_limited_per_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import membridge.api.app as appmod

    monkeypatch.setattr(appmod, "SANDBOX_RATE", (5, 3600.0))  # independent of how fast the machine is
    c = TestClient(create_app(sandbox=True), base_url="http://127.0.0.1")
    codes = [c.get("/api/overview").status_code for _ in range(8)]
    assert codes[:5] == [200] * 5 and codes[5:] == [429] * 3
    assert c.get("/api/overview").status_code == 429  # and it stays limited until the window passes
    assert SANDBOX_RATE[0] >= 30  # the production limit is generous enough for a person browsing


def test_security_headers_and_host_checks(sb: TestClient) -> None:
    r = sb.get("/api/sandbox")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"] and r.headers["x-content-type-options"] == "nosniff"
    assert sb.get("/api/sandbox", headers={"host": "evil.example"}).status_code == 403


def test_a_public_host_must_be_named_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMBRIDGE_PUBLIC_HOST", "demo.example.org")
    c = TestClient(create_app(sandbox=True), base_url="http://demo.example.org")
    assert c.get("/api/sandbox").status_code == 200
    assert TestClient(c.app, base_url="http://other.example.org").get("/api/sandbox").status_code == 403
