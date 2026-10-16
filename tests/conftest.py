from __future__ import annotations

import importlib.util
import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from membridge.store.graph import GraphStore
from membridge.store.sqlite import Store

ROOT = Path(__file__).resolve().parents[1]


def _load_sample_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "make_sample", ROOT / "samples" / "make_sample.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sample_module():  # type: ignore[no-untyped-def]
    return _load_sample_module()


@pytest.fixture
def export_file(tmp_path: Path, sample_module) -> Path:  # type: ignore[no-untyped-def]
    p = tmp_path / "export" / "conversations.json"
    p.parent.mkdir()
    p.write_text(json.dumps(sample_module.build_export()), encoding="utf-8")
    return p


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store(tmp_path / "db" / "test.db")
    yield s
    s.close()


@pytest.fixture
def graph() -> Iterator["GraphStore"]:  # noqa: UP037
    from membridge.store.graph import GraphStore

    g = GraphStore(name=f"membridge_test_{uuid.uuid4().hex[:8]}")
    if not g.available():
        pytest.skip("FalkorDB is not running (docker compose up -d)")
    yield g
    g.drop()
