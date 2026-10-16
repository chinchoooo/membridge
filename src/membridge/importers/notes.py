"""A folder of Markdown / plain-text notes. Each file is one conversation; each top-level section is one message from the owner.

Use explicitly (`membridge import ./notes --adapter notes`). It is also tried last during auto-detection, but only for folders that
contain no conversation exports.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from .base import ParseFailure, RawConversation, RawMessage
from .files import is_conv_file

_EXT = {".md", ".markdown", ".txt"}
_HEAD = re.compile(r"^(#{1,2})\s+(.*\S)\s*$")
_MAX_BYTES = 5_000_000


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in _EXT else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _EXT and ".git" not in p.parts)


class NotesAdapter:
    name = "notes"

    def source_files(self, path: Path) -> list[Path]:
        return _files(path)

    def detect(self, path: Path) -> bool:
        if path.is_file() and path.suffix.lower() not in {".md", ".markdown"}:
            return False  # a lone .txt could be anything; folders and Markdown files are claimed
        if path.is_dir() and any(is_conv_file(p.name) for p in path.iterdir() if p.is_file()):
            return False  # a conversation export folder, not notes
        return bool(_files(path))

    def iter_items(self, path: Path) -> Iterator[RawConversation | ParseFailure]:
        root = path if path.is_dir() else path.parent
        for f in _files(path):
            rel = f.relative_to(root).as_posix()
            try:
                if f.stat().st_size > _MAX_BYTES:
                    yield ParseFailure(rel, f"file larger than {_MAX_BYTES} bytes; skipped")
                    continue
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                yield ParseFailure(rel, f"unreadable: {e}")
                continue
            sections: list[tuple[str | None, list[str]]] = [(None, [])]
            for line in text.splitlines():
                if (m := _HEAD.match(line)) and not line.startswith("###"):
                    sections.append((m.group(2), []))
                else:
                    sections[-1][1].append(line)
            mtime = f.stat().st_mtime
            msgs: list[RawMessage] = []
            for head, body in sections:
                content = "\n".join(body).strip()
                full = f"{head}\n{content}".strip() if head else content
                if len(full) < 3:
                    continue
                msgs.append(
                    RawMessage(
                        external_id=f"{rel}#{len(msgs)}", parent_external_id=msgs[-1].external_id if msgs else None, seq=len(msgs),
                        role="user", author_name=None, create_time=mtime, content_type="text", text=full, on_current_branch=True,
                        metadata={"heading": head} if head else {},
                    )
                )  # fmt: skip
            first_head = next((h for h, _ in sections if h), None)
            yield RawConversation(
                external_id=rel, title=first_head or f.stem, create_time=mtime, update_time=mtime,
                current_node=msgs[-1].external_id if msgs else None, messages=msgs,
                limitations=["notes are the owner's own writing; per-message timestamps are the file's modified time"],
            )  # fmt: skip
