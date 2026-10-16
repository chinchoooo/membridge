"""Shared retrieval-filter parsing for the MCP server and the HTTP API (one validated place)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .service import ServiceError

ALLOWED = {"category", "status", "project", "origin", "verification", "since", "until", "as_of", "known_at", "include_sensitive", "communities"}
_DATE_KEYS = ("since", "until", "as_of", "known_at")


def epoch(v: Any) -> float | None:
    """ISO date/datetime or epoch seconds -> epoch seconds (naive values are taken as UTC)."""
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        d = datetime.fromisoformat(str(v))
    except ValueError as e:
        raise ServiceError(f"invalid date '{v}': use ISO format like 2026-01-31") from e
    return (d if d.tzinfo else d.replace(tzinfo=UTC)).timestamp()


def build_filters(filters: dict[str, Any] | None, date_range: dict[str, Any] | None = None) -> dict[str, Any]:
    f = dict(filters or {})
    bad = set(f) - ALLOWED
    if bad:
        raise ServiceError(f"unknown filter(s) {sorted(bad)}; allowed: {sorted(ALLOWED - {'include_sensitive', 'communities'})}")
    for key, val in (date_range or {}).items():
        if key in ("since", "until"):
            f[key] = val
    for key in _DATE_KEYS:
        if key in f:
            v = epoch(f[key])
            if v is None:
                del f[key]
            else:
                f[key] = v
    f.pop("include_sensitive", None)  # an agent/UI request cannot opt itself into sensitive memories
    f.pop("communities", None)
    return f
