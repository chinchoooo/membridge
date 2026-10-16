"""Memory candidates and shared helpers (normalisation, ids, sensitivity, secrets)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

CATEGORIES = {
    "preference", "decision", "project", "constraint", "profile", "skill", "fact",
    "lesson", "interest", "health", "education", "research", "session",
}  # fmt: skip
VERIFICATIONS = {"user_stated", "provider_stated", "model_inferred", "user_verified"}
LINK_RELS = {
    "SUPERSEDES",
    "CONTRADICTS",
    "RELATED_TO",
    "DEPENDS_ON",
    "LEARNED_FROM",
    "DERIVED_FROM",
}


def sha(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def canonical(text: str) -> str:
    """Lower-case alphanumerics only, single-spaced: the identity of an entity/object."""
    return re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip()


def normalize_statement(text: str) -> str:
    return canonical(text)


def memory_id(dedup_key: str) -> str:
    return "mem_" + sha(dedup_key)[:24]


def entity_id(canon: str) -> str:
    return "ent_" + sha(canon)[:20]


_SENSITIVE = re.compile(
    r"\b(diet|allerg\w*|medical|medication|diagnos\w+|therap\w+|depress\w+|anxiety|salary|income|"
    r"passport|ssn|social security|home address|phone number|religio\w+|health)\b",
    re.IGNORECASE,
)
# Anything that looks like a credential is never stored.
_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[bap]-[A-Za-z0-9-]{10,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:password|passwd|api[_ -]?key|secret|token)\s*[:=]\s*\S{6,}|"
    r"\b\d{13,19}\b)",
    re.IGNORECASE,
)


def sensitivity_of(text: str) -> str:
    return "sensitive" if _SENSITIVE.search(text) else "normal"


def contains_secret(text: str) -> bool:
    return bool(_SECRET.search(text))


@dataclass
class EntityRef:
    name: str
    type: str = "concept"


@dataclass
class Candidate:
    statement: str
    category: str
    kind: str = "semantic"  # episodic | semantic
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    old_object: str | None = None  # explicit "switched from X" evidence
    entities: list[EntityRef] = field(default_factory=list)
    ref: str = ""  # evidence message or document id
    ref_kind: str = "message"
    quote: str = ""
    ts: float | None = None
    confidence: float = 0.7
    verification: str = "user_stated"
    method: str = "rule"
    sensitivity: str = "normal"
    origin: str = "import"
