"""Redaction applied to any text before it leaves the machine (hosted inference only)."""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("API_KEY", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[bap]-[A-Za-z0-9-]{10,})\b")),
    ("SECRET", re.compile(r"(?i)\b(?:password|passwd|secret|token|api[_ -]?key)\s*[:=]\s*\S+")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\(\d{2,4}\)|\d{2,4})[ -]?\d{3,4}[ -]?\d{3,4}(?!\w)")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("URL_TOKEN", re.compile(r"https?://\S*[?&](?:token|key|secret|sig|signature|auth)=\S+")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace sensitive spans with typed placeholders. Returns (clean_text, counts_by_type)."""
    counts: dict[str, int] = {}
    for label, pat in _PATTERNS:

        def sub(_m: re.Match[str], label: str = label) -> str:
            counts[label] = counts.get(label, 0) + 1
            return f"[{label}]"

        text = pat.sub(sub, text)
    return text, counts
