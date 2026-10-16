"""Adapter interface and provider-neutral import records."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RawMessage:
    external_id: str
    parent_external_id: str | None
    seq: int
    role: str
    author_name: str | None
    create_time: float | None
    content_type: str
    text: str
    on_current_branch: bool
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawConversation:
    external_id: str
    title: str | None
    create_time: float | None
    update_time: float | None
    current_node: str | None
    messages: list[RawMessage]
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseFailure:
    ref: str | None
    error: str


class ImportAdapter(Protocol):
    """One adapter per provider export format. Adapters never touch the database."""

    name: str

    def detect(self, path: Path) -> bool: ...

    def source_files(self, path: Path) -> list[Path]:
        """Files whose bytes identify this source (hashed for dedup of whole imports)."""
        ...

    def iter_items(self, path: Path) -> Iterator[RawConversation | ParseFailure]: ...
