"""Claude Code session transcripts (`~/.claude/projects/<project>/<session>.jsonl`).

One JSONL file = one conversation. Only `user` / `assistant` events with real text become messages: tool calls, tool results and
reasoning are counted as limitations (they are machine output, not the user's words), bookkeeping events (mode, attachments,
snapshots, queue operations...) are ignored, and sidechain (sub-agent) turns are kept but flagged off the main branch.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import ParseFailure, RawConversation, RawMessage

_SKIP_TOOLS = {"tool_use": "tool call", "tool_result": "tool result", "thinking": "reasoning"}


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".jsonl" else []
    return sorted(p for p in path.rglob("*.jsonl") if p.is_file())


def _ts(v: Any) -> float | None:
    if not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v).timestamp()
    except ValueError:
        return None


def _text(content: Any) -> tuple[str, Counter[str]]:
    skipped: Counter[str] = Counter()
    if isinstance(content, str):
        return content, skipped
    parts: list[str] = []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif t in _SKIP_TOOLS:
                skipped[_SKIP_TOOLS[t]] += 1
            elif t == "image":
                skipped["image"] += 1
    return "\n".join(parts), skipped


def _is_session_line(d: Any) -> bool:
    return isinstance(d, dict) and "sessionId" in d and d.get("type") in {"user", "assistant", "system", "mode", "attachment", "ai-title"}


class ClaudeCodeAdapter:
    name = "claude_code"

    def source_files(self, path: Path) -> list[Path]:
        return _files(path)

    def detect(self, path: Path) -> bool:
        for f in _files(path)[:3]:
            try:
                with f.open(encoding="utf-8") as fh:
                    for _ in range(25):
                        line = fh.readline()
                        if not line:
                            break
                        try:
                            if _is_session_line(json.loads(line)):
                                return True
                        except json.JSONDecodeError:
                            return False
            except OSError:
                return False
        return False

    def iter_items(self, path: Path) -> Iterator[RawConversation | ParseFailure]:
        for f in _files(path):
            try:
                yield self._parse_file(f)
            except (OSError, ValueError) as e:
                yield ParseFailure(f.name, f"unreadable session file: {e}")

    def _parse_file(self, f: Path) -> RawConversation:
        title: str | None = None
        session = f.stem
        cwd: str | None = None
        msgs: list[RawMessage] = []
        omitted: Counter[str] = Counter()
        bad_lines = 0
        first_ts = last_ts = None
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    bad_lines += 1
                    continue
                if not isinstance(d, dict):
                    continue
                session = d.get("sessionId") or session
                cwd = d.get("cwd") or cwd
                t = d.get("type")
                if t == "ai-title":
                    title = d.get("aiTitle") or d.get("title") or title
                    continue
                m = d.get("message")
                if t not in ("user", "assistant") or not isinstance(m, dict) or d.get("isMeta"):
                    continue
                text, skipped = _text(m.get("content"))
                omitted.update(skipped)
                text = text.strip()
                if not text or not isinstance(d.get("uuid"), str):
                    continue
                ts = _ts(d.get("timestamp"))
                first_ts = first_ts if first_ts is not None else ts
                last_ts = ts or last_ts
                parent = d.get("parentUuid")
                msgs.append(
                    RawMessage(
                        external_id=d["uuid"], parent_external_id=parent if isinstance(parent, str) else None, seq=len(msgs),
                        role=str(m.get("role") or t), author_name=None, create_time=ts, content_type="text", text=text,
                        on_current_branch=not d.get("isSidechain"), metadata={"sidechain": True} if d.get("isSidechain") else {},
                    )
                )  # fmt: skip
        if title is None and msgs:
            first_user = next((x.text for x in msgs if x.role == "user"), msgs[0].text)
            title = " ".join(first_user.split())[:70]
        lims = [f"{n} {what} block(s) not stored as text" for what, n in omitted.items()]
        if bad_lines:
            lims.append(f"{bad_lines} unparseable line(s) skipped")
        return RawConversation(
            external_id=str(session), title=title, create_time=first_ts, update_time=last_ts,
            current_node=msgs[-1].external_id if msgs else None, messages=msgs, limitations=lims,
            metadata={"cwd": cwd} if cwd else {},
        )  # fmt: skip
