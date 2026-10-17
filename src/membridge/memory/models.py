"""Memory candidates and shared helpers (normalisation, ids, trust, sensitivity, secrets)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

CATEGORIES = {
    "preference", "decision", "project", "constraint", "profile", "skill", "fact",
    "lesson", "interest", "health", "education", "research", "session",
}  # fmt: skip
VERIFICATIONS = {"user_stated", "provider_stated", "model_inferred", "user_verified", "client_asserted"}
# Who vouches for a statement, strongest first. `client_asserted` is what a connected agent writes on the user's
# behalf: better than a provider's summary of you, worse than something you said or confirmed yourself.
TRUST = {"user_verified": 4, "user_stated": 3, "client_asserted": 2, "provider_stated": 1, "model_inferred": 0}
LINK_RELS = {
    "SUPERSEDES",
    "CONTRADICTS",
    "RELATED_TO",
    "DEPENDS_ON",
    "LEARNED_FROM",
    "DERIVED_FROM",
}


def trust_rank(verification: str | None) -> int:
    return TRUST.get(str(verification), 0)


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


# ---- sensitivity ------------------------------------------------------------------------------------------------
# Anything health, money, identity-document, body or family related. Wrongly-private is the safe failure: a private
# memory is merely hidden from default retrieval, a leaked one cannot be un-leaked.
_FAMILY = r"(?:(?:my|his|her|their|your|the user'?s)\s+(?:parents?|mother|father|mom|mum|dad|sibling|siblings|brother|sister|wife|husband|partner|spouse|son|daughter)|only child|one parent|parents? (?:is|are|live|lives|in)\b)"
_SENSITIVE = re.compile(
    r"\b(?:diet|allerg\w*|medical|medic(?:ation|ine)s?|diagnos\w+|therap\w+|depress\w+|anxiety|insomnia|disorder|"
    r"prescri\w+|symptoms?|surgery|dosage|pills?|tablets?|psychiatr\w+|counsell?\w+|mental health|health|"
    r"salary|income|earn(?:s|ed|ing)?|stipend|net worth|credit card|debt|(?:my|his|her|their) loan|"
    r"passport|ssn|social security|aadhaar|home address|phone number|religio\w+|"
    r"weigh(?:s|ed|t|ing)?|body weight|bmi|\d+(?:\.\d+)?\s?(?:kg|kgs|lbs?|cm)|\d+\s?mg|gym|sleep\w*|awake)\b"
    rf"|{_FAMILY}|[₹$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?(?:lpa|rupees|inr|usd|eur)\b",
    re.IGNORECASE,
)
# ---- secrets ----------------------------------------------------------------------------------------------------
# Anything that looks like a credential is never stored and never returned.
_SECRET_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{16,}",
    r"\bAKIA[0-9A-Z]{16}\b|\bASIA[0-9A-Z]{16}\b",
    r"\bghp_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|\bgho_[A-Za-z0-9]{20,}",
    r"\bxox[bapors]-[A-Za-z0-9-]{10,}",
    r"\bAIza[0-9A-Za-z_-]{30,}",
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@",
    r"\b(?:password|passwd|pwd|passphrase|passcode|pin|api[_ -]?key|(?:aws[_ ])?secret(?:[_ ]access[_ ]key)?|client[_ ]secret|"
    r"access[_ ]key|auth[_ ]token|token|(?:2fa|otp|mfa|backup|recovery)(?:[_ -](?:backup|recovery))?[_ -]?codes?)\s*(?:is|are|=|:)\s*\S{4,}",
)
_SECRET = re.compile("|".join(f"(?:{p})" for p in _SECRET_PATTERNS), re.IGNORECASE)
_CARD = re.compile(r"(?<![\w.])(?:\d[ -]?){13,19}(?![\w])")


def _luhn(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if flip:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        flip = not flip
    return total % 10 == 0


def _card_spans(text: str) -> list[tuple[int, int]]:
    out = []
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn(digits):  # a 13-digit millisecond timestamp is not a card number
            out.append(m.span())
    return out


def sensitivity_of(text: str) -> str:
    return "sensitive" if _SENSITIVE.search(text) else "normal"


def safe_title(title: str | None) -> str:
    """A conversation title as an assistant may see it: private titles are replaced, credentials blanked."""
    t = title or ""
    return "[private conversation]" if sensitivity_of(t) == "sensitive" else redact_secrets(t)


def contains_secret(text: str) -> bool:
    return bool(_SECRET.search(text)) or bool(_card_spans(text))


def redact_secrets(text: str) -> str:
    """The text with every credential-looking span replaced by [REDACTED] (used on everything returned to a client)."""
    for a, b in reversed(_card_spans(text)):
        text = text[:a] + "[REDACTED]" + text[b:]
    return _SECRET.sub("[REDACTED]", text)


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
    ts: float | None = None  # None = the source gave no date; such a statement is never "newer" than a dated one
    confidence: float = 0.7
    verification: str = "user_stated"
    method: str = "rule"
    sensitivity: str = "normal"
    origin: str = "import"
    correction: bool = False  # user explicitly corrected an earlier statement (retracts it)
    object_value: str | None = None  # normalised literal (ISO date) for comparison
