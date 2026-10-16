"""Shared file handling for JSON conversation exports (file, folder, or .zip)."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import ijson  # type: ignore[import-untyped]


def is_conv_file(name: str) -> bool:
    base = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base.startswith("conversations") and base.endswith(".json")


def _is_zip(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def zip_members(path: Path) -> list[str]:
    if not _is_zip(path):
        return []
    with zipfile.ZipFile(path) as zf:
        return sorted(n for n in zf.namelist() if is_conv_file(n))


def source_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and is_conv_file(p.name))
    return [path]


def has_conversation_files(path: Path) -> bool:
    if path.is_dir():
        return bool(source_files(path))
    if _is_zip(path):
        return bool(zip_members(path))
    return path.is_file() and is_conv_file(path.name)


@contextmanager
def open_streams(path: Path) -> Iterator[list[IO[bytes]]]:
    if _is_zip(path):
        with zipfile.ZipFile(path) as zf:
            opened: list[IO[bytes]] = [zf.open(n) for n in zip_members(path)]
            try:
                yield opened
            finally:
                for f in opened:
                    f.close()
    else:
        handles: list[IO[bytes]] = [p.open("rb") for p in source_files(path)]
        try:
            yield handles
        finally:
            for f in handles:
                f.close()


def iter_json_items(path: Path) -> Iterator[Any]:
    """Yield each element of the top-level arrays (streamed). Raises ijson.JSONError on bad JSON."""
    with open_streams(path) as streams:
        for stream in streams:
            yield from ijson.items(stream, "item", use_float=True)


def sniff_keys(path: Path, limit: int = 5) -> set[str]:
    """Union of keys of the first few dict items; empty if unreadable. Tells exports apart."""
    keys: set[str] = set()
    try:
        for i, item in enumerate(iter_json_items(path)):
            if isinstance(item, dict):
                keys.update(item.keys())
            if i + 1 >= limit:
                break
    except (ijson.JSONError, OSError, zipfile.BadZipFile):
        pass
    return keys
