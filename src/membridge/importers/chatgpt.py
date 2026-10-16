"""ChatGPT data-export adapter (conversations.json, sharded conversations-NNN.json, or .zip).

OpenAI publishes no official schema; this follows the community-documented layout:
each conversation has a `mapping` tree of nodes (`message`, `parent`, `children`) and a
`current_node` pointing at the leaf of the branch the user last saw. Everything is parsed
defensively: a bad record becomes a ParseFailure, never a crash.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import-untyped]

from . import files
from .base import ParseFailure, RawConversation, RawMessage


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def extract_content(content: Any) -> tuple[str, str, list[str]]:
    """Return (content_type, text, limitations). Never invents content for non-text parts."""
    if not isinstance(content, dict):
        return "unknown", "", ["message has no content object"]
    ctype = str(content.get("content_type") or "unknown")
    limits: list[str] = []
    texts: list[str] = []
    parts = content.get("parts")
    if isinstance(parts, list):
        for p in parts:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, dict):
                ptype = p.get("content_type")
                if ptype == "image_asset_pointer":
                    limits.append(
                        f"image not included in export: {p.get('asset_pointer')}"
                    )
                elif isinstance(p.get("text"), str):
                    texts.append(p["text"])
                else:
                    limits.append(f"non-text part omitted: {ptype or 'unknown'}")
            elif p is not None:
                limits.append(f"unrecognised part of type {type(p).__name__} omitted")
    elif isinstance(content.get("text"), str):
        texts.append(content["text"])
    elif ctype in ("thoughts", "reasoning_recap"):
        limits.append(f"{ctype} content not imported")
    else:
        limits.append(f"no extractable text for content_type={ctype}")
    return ctype, "\n".join(texts), limits


def _parse_conversation(raw: Any, index: int) -> RawConversation | ParseFailure:
    if not isinstance(raw, dict):
        return ParseFailure(
            f"item[{index}]", f"expected object, got {type(raw).__name__}"
        )
    ext = raw.get("conversation_id") or raw.get("id")
    mapping = raw.get("mapping")
    if not isinstance(ext, str) or not ext:
        return ParseFailure(f"item[{index}]", "missing conversation id")
    if not isinstance(mapping, dict):
        return ParseFailure(ext, "missing or invalid 'mapping'")

    current = (
        raw.get("current_node") if isinstance(raw.get("current_node"), str) else None
    )
    # Nodes on the visible branch: walk parents from current_node (cycle-safe).
    on_branch: set[str] = set()
    cur = current
    while isinstance(cur, str) and cur in mapping and cur not in on_branch:
        on_branch.add(cur)
        node = mapping[cur]
        cur = node.get("parent") if isinstance(node, dict) else None

    roots = [
        k
        for k, n in mapping.items()
        if isinstance(n, dict)
        and (n.get("parent") is None or n.get("parent") not in mapping)
    ]
    messages: list[RawMessage] = []
    limitations: list[str] = []
    visited: set[str] = set()
    # Iterative DFS, children in listed order; the visible branch is flagged, not exclusive.
    stack = list(reversed(roots))
    while stack:
        nid = stack.pop()
        if nid in visited or nid not in mapping:
            continue
        visited.add(nid)
        node = mapping[nid]
        if not isinstance(node, dict):
            limitations.append(f"node {nid} is not an object; skipped")
            continue
        children = node.get("children")
        if isinstance(children, list):
            stack.extend(reversed([c for c in children if isinstance(c, str)]))
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue  # structural root nodes carry no message
        author = _dict(msg.get("author"))
        ctype, text, limits = extract_content(msg.get("content"))
        meta = _dict(msg.get("metadata"))
        keep: dict[str, Any] = {
            k: meta[k]
            for k in ("model_slug", "is_visually_hidden_from_conversation")
            if k in meta
        }
        messages.append(
            RawMessage(
                external_id=str(msg.get("id") or nid),
                parent_external_id=node.get("parent")
                if isinstance(node.get("parent"), str)
                else None,
                seq=len(messages),
                role=str(author.get("role") or "unknown"),
                author_name=author.get("name")
                if isinstance(author.get("name"), str)
                else None,
                create_time=_num(msg.get("create_time")),
                content_type=ctype,
                text=text,
                on_current_branch=nid in on_branch,
                limitations=limits,
                metadata=keep,
            )
        )
    unreachable = len(mapping) - len(visited)
    if unreachable:
        limitations.append(
            f"{unreachable} mapping node(s) unreachable from any root (cycle or orphan)"
        )
    title = raw.get("title")
    return RawConversation(
        external_id=ext,
        title=title if isinstance(title, str) else None,
        create_time=_num(raw.get("create_time")),
        update_time=_num(raw.get("update_time")),
        current_node=current,
        messages=messages,
        limitations=limitations,
    )


class ChatGPTAdapter:
    name = "chatgpt"

    def source_files(self, path: Path) -> list[Path]:
        return files.source_files(path)

    def detect(self, path: Path) -> bool:
        return files.has_conversation_files(path) and "mapping" in files.sniff_keys(
            path
        )

    def iter_items(self, path: Path) -> Iterator[RawConversation | ParseFailure]:
        index = 0
        try:
            for raw in files.iter_json_items(path):
                yield _parse_conversation(raw, index)
                index += 1
        except ijson.JSONError as e:  # truncated / invalid file: report and stop
            yield ParseFailure(f"item[{index}]", f"invalid JSON stream: {e}")
