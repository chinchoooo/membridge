"""Deterministic relative-date normalisation against a reference timestamp (the message's own time).

Graphiti resolves "two weeks ago" against the episode timestamp so facts get absolute validity; this is a
small rule-based equivalent. It only returns a date when it is confident; otherwise None (never guesses).
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}  # fmt: skip
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())} | {"sept": 9})
_DAYS = {d: i for i, d in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}
_DAYS.update({k[:3]: v for k, v in list(_DAYS.items())})
_UNITS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _n(tok: str) -> int | None:
    return int(tok) if tok.isdigit() else _NUM.get(tok)


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(text: str, ref_ts: float | None) -> str | None:
    """Return an ISO date (YYYY-MM-DD) for `text` relative to `ref_ts`, or None if not confidently parseable."""
    t = text.strip().lower().rstrip(".,;")
    ref = datetime.fromtimestamp(ref_ts, UTC).date() if ref_ts else None
    if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t):
        d = _safe(int(m[1]), int(m[2]), int(m[3]))
        return d.isoformat() if d else None
    # "5 january [2027]" / "january 5[th] [, 2027]" — a missing year is taken as the next occurrence on/after ref
    pats = [(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})(?:,?\s+(\d{4}))?", 1, 2), (r"([a-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?", 2, 1)]
    for pat, di, mi in pats:
        if (m := re.fullmatch(pat, t)) and m[mi] in _MONTHS:
            month, day = _MONTHS[m[mi]], int(m[di])
            if m[3]:
                d = _safe(int(m[3]), month, day)
            elif ref:
                d = _safe(ref.year, month, day)
                if d and d < ref:
                    d = _safe(ref.year + 1, month, day)
            else:
                return None
            return d.isoformat() if d else None
    if ref is None:
        return None
    if t in ("today", "tonight"):
        return ref.isoformat()
    if t == "tomorrow":
        return (ref + timedelta(days=1)).isoformat()
    if t == "yesterday":
        return (ref - timedelta(days=1)).isoformat()
    if m := re.fullmatch(r"(?:in\s+)?(\w+)\s+(days?|weeks?|months?)(?:\s+from now)?", t):
        n = _n(m[1])
        if n is not None and (t.startswith("in ") or t.endswith("from now")):
            return (ref + timedelta(days=n * _UNITS[m[2]])).isoformat()
    if m := re.fullmatch(r"(\w+)\s+(days?|weeks?|months?)\s+ago", t):
        n = _n(m[1])
        if n is not None:
            return (ref - timedelta(days=n * _UNITS[m[2]])).isoformat()
    if m := re.fullmatch(r"(next|this|last)?\s*([a-z]{3,9})", t):
        wd = _DAYS.get(m[2])
        if wd is not None:
            delta = (wd - ref.weekday()) % 7
            if m[1] == "next":
                delta = delta + 7 if delta else 7
            elif m[1] == "last":
                delta = (wd - ref.weekday()) % 7 - 7 if (wd - ref.weekday()) % 7 else -7
            return (ref + timedelta(days=delta)).isoformat()
    if m := re.fullmatch(r"end of ([a-z]{3,9})", t):
        if m[1] in _MONTHS:
            month = _MONTHS[m[1]]
            y = ref.year + (1 if _safe(ref.year, month, 1) and _safe(ref.year, month, 1) < ref.replace(day=1) else 0)  # type: ignore[operator]
            nxt = date(y + (month == 12), month % 12 + 1, 1)
            return (nxt - timedelta(days=1)).isoformat()
    return None
