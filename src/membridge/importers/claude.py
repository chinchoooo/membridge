"""Claude.ai data-export adapter (conversations.json inside conversations-000.zip).

Layout (observed on a real export): a list of conversations, each with `uuid`, `name`,
`summary`, `created_at`, `updated_at` and `chat_messages`. Each message has `sender`
(human|assistant), `text`, typed `content` blocks (text, thinking, tool_use, tool_result,
image, document, ...), `attachments`, `files` and `parent_message_uuid`.

Only user-visible text blocks become message text. Reasoning, tool calls/results and
injected system prompts are counted, not stored as conversation text, and the omission is
recorded as a limitation. Images and documents are not part of the export.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import ijson  # type: ignore[import-untyped]

from . import files
from .base import ParseFailure, RawConversation, RawMessage

_ROOT_PARENT = "00000000-0000-4000-8000-000000000000"
_SKIPPED = {
    "thinking": "reasoning",
    "tool_use": "tool call",
    "tool_result": "tool result",
    "injected_prompt_block": "injected system prompt",
    "token_budget": "token budget marker",
}


def _ts(v: Any) -> float | None:
    if not isinstance(v, str) or not v:
        return None
    try:
        return datetime.fromisoformat(v).timestamp()
    except ValueError:
        return None


def _message_text(m: dict[str, Any]) -> tuple[str, str, list[str], dict[str, Any]]:
    """Return (content_type, text, limitations, metadata) for one message."""
    limits: list[str] = []
    meta: dict[str, Any] = {}
    texts: list[str] = []
    skipped: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    blocks = m.get("content")
    if isinstance(blocks, list) and blocks:
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and isinstance(b.get("text"), str):
                texts.append(b["text"])
            elif t == "image":
                limits.append(
                    f"image not included in export (file {b.get('file_uuid')})"
                )
            elif t == "document":
                limits.append(
                    f"document not included in export ({b.get('title') or b.get('file_uuid')})"
                )
            elif t in _SKIPPED:
                skipped[_SKIPPED[t]] += 1
                if t == "tool_use" and isinstance(b.get("name"), str):
                    tools[b["name"]] += 1
            else:
                limits.append(f"unrecognised content block omitted: {t}")
    elif isinstance(m.get("text"), str):
        texts.append(m["text"])  # older exports: flat text only
    for what, n in skipped.items():
        if what != "token budget marker":
            limits.append(f"{n} {what} block(s) not stored as text")
    if tools:
        meta["tools_used"] = dict(tools)
    atts = m.get("attachments")
    if isinstance(atts, list) and atts:
        names = [a.get("file_name") for a in atts if isinstance(a, dict)]
        limits.append(
            f"{len(atts)} attachment(s) not stored: {', '.join(str(n) for n in names if n)}"
        )
    return ("text" if texts else "non_text"), "\n".join(texts), limits, meta


def _parse_conversation(raw: Any, index: int) -> RawConversation | ParseFailure:
    if not isinstance(raw, dict):
        return ParseFailure(
            f"item[{index}]", f"expected object, got {type(raw).__name__}"
        )
    ext = raw.get("uuid")
    msgs = raw.get("chat_messages")
    if not isinstance(ext, str) or not ext:
        return ParseFailure(f"item[{index}]", "missing conversation uuid")
    if not isinstance(msgs, list):
        return ParseFailure(ext, "missing or invalid 'chat_messages'")

    by_id: dict[str, dict[str, Any]] = {
        m["uuid"]: m
        for m in msgs
        if isinstance(m, dict) and isinstance(m.get("uuid"), str)
    }
    on_branch: set[str] = set()
    # Branch tip = the newest message by timestamp (list order breaks ties).
    ordered = [m for m in msgs if isinstance(m, dict) and m.get("uuid") in by_id]
    tip = max(
        enumerate(ordered),
        key=lambda im: (_ts(im[1].get("created_at")) or 0.0, im[0]),
        default=None,
    )
    cur = tip[1]["uuid"] if tip else None
    while isinstance(cur, str) and cur in by_id and cur not in on_branch:
        on_branch.add(cur)
        cur = by_id[cur].get("parent_message_uuid")

    out: list[RawMessage] = []
    limitations: list[str] = []
    for m in msgs:
        if not isinstance(m, dict) or not isinstance(m.get("uuid"), str):
            limitations.append("a message without uuid was skipped")
            continue
        ctype, text, limits, meta = _message_text(m)
        parent = m.get("parent_message_uuid")
        sender = m.get("sender")
        role = (
            "unknown"
            if not isinstance(sender, str)
            else {"human": "user"}.get(sender, sender)
        )
        out.append(
            RawMessage(
                external_id=m["uuid"],
                parent_external_id=parent
                if isinstance(parent, str) and parent != _ROOT_PARENT
                else None,
                seq=len(out),
                role=role,
                author_name=None,
                create_time=_ts(m.get("created_at")),
                content_type=ctype,
                text=text,
                on_current_branch=m["uuid"] in on_branch,
                limitations=limits,
                metadata=meta,
            )
        )
    name = raw.get("name")
    summary = raw.get("summary")
    return RawConversation(
        external_id=ext,
        title=name if isinstance(name, str) and name else None,
        create_time=_ts(raw.get("created_at")),
        update_time=_ts(raw.get("updated_at")),
        current_node=next(
            (x.external_id for x in reversed(out) if x.on_current_branch), None
        ),
        messages=out,
        limitations=limitations,
        metadata={"provider_summary": summary}
        if isinstance(summary, str) and summary
        else {},
    )


class ClaudeAdapter:
    name = "claude"

    def source_files(self, path: Path) -> list[Path]:
        return files.source_files(path)

    def detect(self, path: Path) -> bool:
        return files.has_conversation_files(
            path
        ) and "chat_messages" in files.sniff_keys(path)

    def iter_items(self, path: Path) -> Iterator[RawConversation | ParseFailure]:
        index = 0
        try:
            for raw in files.iter_json_items(path):
                yield _parse_conversation(raw, index)
                index += 1
        except ijson.JSONError as e:
            yield ParseFailure(f"item[{index}]", f"invalid JSON stream: {e}")
