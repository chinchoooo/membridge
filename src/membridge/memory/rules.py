"""Deterministic extraction of explicit first-person statements.

Only things the user literally said are extracted (verification=user_stated); the statement
is the user's own sentence, never a paraphrase. Open-ended inference belongs to the optional
LLM extractor. Content is treated as untrusted data: text is only ever matched, never executed
or interpolated into queries.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .models import Candidate, EntityRef, canonical, contains_secret, sensitivity_of

EXTRACTOR_VERSION = "rules-1"

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_FIRST_PERSON = re.compile(
    r"\b(i|i'm|i've|i'll|i'd|my|we|we're|we've|we'll|our)\b", re.IGNORECASE
)

_STOP_CAPS = {
    "i", "i'm", "i've", "i'll", "i'd", "the", "a", "an", "my", "we", "it", "this", "that", "these", "those",
    "and", "but", "or", "so", "if", "when", "also", "then", "yes", "no", "ok", "okay", "please", "thanks",
    "what", "why", "how", "who", "where", "which", "can", "could", "should", "would", "do", "does", "did",
    "is", "are", "was", "were", "be", "have", "has", "had", "deadline", "here", "there", "now", "just",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "ignore", "system", "note", "great", "good", "sure", "hello", "hi",
}  # fmt: skip
_CAP_RUN = re.compile(r"[A-Z][A-Za-z0-9+#.]*(?:\s+[A-Z][A-Za-z0-9+#.]*){0,3}")
_ARTICLE = re.compile(r"^(?:a|an|the|my|our|some)\s+", re.IGNORECASE)

_P = r"(?:i|we)(?:'ve| have)?"
_DECIDE = re.compile(
    rf"\b{_P}\s+(?:decided|chose|picked|settled)\s+(?:to\s+)?(?:use|go with|choose|adopt|pick|try)?\s*"
    r"(?P<obj>[^,.;:]+?)(?:\s+(?:for|in|on|as)\s+(?P<subj>[^,.;:]+?))?\s*(?:[,.;:]|\bbecause\b|\bsince\b|$)",
    re.IGNORECASE,
)
_INTENT = re.compile(
    r"\b(?:i|we)(?:'ll| will|'m going to| am going to|'re going to)\s+(?:use|go with)\s+"
    r"(?P<obj>[^,.;:]+?)(?:\s+(?:for|in|on|as)\s+(?P<subj>[^,.;:]+?))?\s*(?:[,.;:]|\bbecause\b|$)",
    re.IGNORECASE,
)
_SWITCH = re.compile(
    rf"\b{_P}\s+(?P<verb>switched|moved|migrated|changed)\s+(?:(?P<subj>[^,.;:]+?)\s+)?"
    r"(?:from\s+(?P<old>[^,.;:]+?)\s+)?to\s+(?P<obj>[^,.;:]+?)\s*(?:[,.;:]|\bbecause\b|\bsince\b|\bas\b|\bso\b|$)",
    re.IGNORECASE,
)
_PREFER = re.compile(
    r"\bi\s+(?:really\s+|strongly\s+|generally\s+|always\s+)?"
    r"(?P<verb>prefer|like|love|enjoy|hate|dislike|avoid|can't stand)\s+(?P<obj>[^,.;:!?]+)",
    re.IGNORECASE,
)
_PROJECT_NAMED = re.compile(
    r"\b(?:project|app|tool|system|pipeline|platform|library|startup|product)\s+(?:called|named)\s+"
    r"(?P<obj>[A-Z][\w.+#-]*(?:\s[A-Z][\w.+#-]*){0,3})"
)
_PROJECT_WORKING = re.compile(
    rf"\b{_P}\s*(?:am\s+|are\s+)?(?:'m\s+|'re\s+)?"
    r"(?:starting|building|working on|developing|creating|writing)\s+(?:a |an |the |my |our )?"
    r"(?P<obj>[A-Z][\w.+#-]*(?:\s[A-Z][\w.+#-]*){0,3})"
)
_DEADLINE = re.compile(
    r"\bdeadline\b[^.!?\n]*?\b(?:is|moved to|changed to|now|extended to|pushed to|set to)\s+(?P<obj>[^,.;:!?]+)",
    re.IGNORECASE,
)
_PROFILE = re.compile(
    r"\bi(?:'m| am)\s+(?:a |an )?(?P<obj>(?:student|engineer|developer|researcher|designer|artist|"
    r"data scientist|professor|founder|intern|teacher|musician|writer)[^,.;:!?]{0,50})",
    re.IGNORECASE,
)
_SKILL = re.compile(
    r"\bi\s+(?:use|work with|know|code in|program in|write)\s+(?P<obj>[A-Z][\w.+#-]*(?:\s[A-Z][\w.+#-]*){0,2})"
    r"\s+(?:for|to|at)\b",
)  # fmt: skip
_POSITIVE = {"prefer", "like", "love", "enjoy"}


def aspect_of(phrase: str | None, subject: str | None) -> str:
    """The facet a decision is about ('Atlas storage' -> 'storage'): only the same facet is superseded."""
    if not phrase:
        return ""
    own = set((subject or "").split())
    words = [w for w in canonical(phrase).split() if w not in _STOP_CAPS and w not in own]
    return " ".join(words[:3])


def clean_object(text: str, max_words: int = 6) -> str | None:
    t = _ARTICLE.sub("", text.strip().strip("\"'`()[]"))
    t = re.sub(
        r"\s+(?:because|since|but|and then|which|that)\b.*$", "", t, flags=re.IGNORECASE
    ).strip()
    words = t.split()
    if not words or len(words) > max_words or len(t) > 60:
        return None
    return t


def sentences(text: str) -> Iterator[str]:
    text = _CODE_FENCE.sub(" ", text)
    for s in _SENT_SPLIT.split(text):
        s = s.strip()
        if 12 <= len(s) <= 400:
            yield s


def find_entities(sentence: str, known: set[str] | None = None) -> list[EntityRef]:
    """Capitalised terms that are not sentence-initial common words, plus known entities."""
    known = known or set()
    found: dict[str, EntityRef] = {}
    for m in _CAP_RUN.finditer(sentence):
        run = m.group(0).rstrip(".")
        words = run.split()
        while words and canonical(words[0]) in _STOP_CAPS:
            words.pop(0)
        while words and canonical(words[-1]) in _STOP_CAPS:
            words.pop()
        if not words:
            continue
        run = " ".join(words)
        c = canonical(run)
        mid_capital = any(ch.isupper() for ch in run[1:])
        sentence_initial = m.start() == 0 and len(words) == 1
        if len(c) < 2 or len(c) > 40:
            continue
        if sentence_initial and not mid_capital and c not in known:
            continue
        found.setdefault(c, EntityRef(run))
    for k in known:
        if re.search(rf"\b{re.escape(k)}\b", canonical(sentence)):
            found.setdefault(k, EntityRef(k))
    return list(found.values())


def _entity(name: str, type_: str = "concept") -> EntityRef:
    return EntityRef(name, type_)


def extract_from_message(
    text: str,
    *,
    message_id: str,
    ts: float | None,
    context_entity: str | None,
    known: set[str],
) -> tuple[list[Candidate], int]:
    """Return (candidates, dropped_secret_count) for one user message."""
    out: list[Candidate] = []
    dropped = 0
    for sent in sentences(text):
        if not _FIRST_PERSON.search(sent) and not _DEADLINE.search(sent):
            continue
        if contains_secret(sent):
            dropped += 1
            continue
        ents = find_entities(sent, known)

        def mk(category: str, *, sent: str = sent, **kw: object) -> Candidate:
            return Candidate(
                statement=sent, category=category, ref=message_id, quote=sent, ts=ts,
                sensitivity=sensitivity_of(sent), method="rule", **kw,  # type: ignore[arg-type]
            )  # fmt: skip

        def subject_entity(subj_phrase: str | None, ents: list[EntityRef] = ents) -> str | None:
            if subj_phrase:
                subj_c = canonical(subj_phrase)
                for e in ents + [EntityRef(k) for k in known]:
                    if (
                        canonical(e.name) in subj_c.split()
                        or canonical(e.name) in subj_c
                    ):
                        return canonical(e.name)
            for e in ents:
                if canonical(e.name) in known:
                    return canonical(e.name)
            return context_entity

        if (
            (m := _SWITCH.search(sent))
            and (obj := clean_object(m["obj"]))
            and not (
                m["verb"].lower() == "moved" and not m["old"]
            )  # "moved to Boston" is not a tool switch
        ):
            old = clean_object(m["old"]) if m["old"] else None
            subj = subject_entity(m["subj"])
            out.append(mk("decision", kind="episodic", subject=subj, predicate="uses", object=obj,
                          old_object=old, entities=ents + [_entity(obj)] + ([_entity(old)] if old else []),
                          confidence=0.9))  # fmt: skip
            continue
        if (m := _DECIDE.search(sent) or _INTENT.search(sent)) and (
            obj := clean_object(m["obj"])
        ):
            subj = subject_entity(m["subj"])
            is_intent = " decided" not in sent.lower() and "chose" not in sent.lower()
            asp = aspect_of(m["subj"], subj)
            out.append(mk("decision", kind="episodic", subject=subj, predicate=f"uses:{asp}" if asp else "decided",
                          object=obj, entities=ents + [_entity(obj)], confidence=0.7 if is_intent else 0.85))  # fmt: skip
        if m := _DEADLINE.search(sent):
            obj = clean_object(m["obj"], 8)
            if obj:
                out.append(mk("constraint", kind="episodic", subject=subject_entity(None), predicate="deadline",
                              object=obj, entities=ents, confidence=0.85))  # fmt: skip
        if m := _PREFER.search(sent):
            obj = clean_object(m["obj"])
            if obj and not re.match(
                r"^(to|it|that|this|when|how)\b", obj, re.IGNORECASE
            ):
                verb = m["verb"].lower()
                out.append(mk("preference", subject="user", predicate="prefers" if verb in _POSITIVE else "avoids",
                              object=obj, entities=ents + [_entity(obj)], confidence=0.8))  # fmt: skip
        if m := (_PROJECT_NAMED.search(sent) or _PROJECT_WORKING.search(sent)):
            obj = clean_object(m["obj"], 4)
            if obj and canonical(obj) not in _STOP_CAPS:
                out.append(mk("project", subject=canonical(obj), predicate="is_project", object=obj,
                              entities=[e for e in ents if canonical(e.name) != canonical(obj)] + [_entity(obj, "project")],
                              confidence=0.85))  # fmt: skip
        if m := _PROFILE.search(sent):
            obj = clean_object(m["obj"], 8)
            if obj:
                out.append(
                    mk(
                        "profile",
                        subject="user",
                        predicate="is",
                        object=obj,
                        entities=ents,
                        confidence=0.8,
                    )
                )
        if m := _SKILL.search(sent):
            obj = clean_object(m["obj"], 3)
            if obj:
                out.append(mk("skill", subject="user", predicate="uses_tool", object=obj,
                              entities=ents + [_entity(obj)], confidence=0.75))  # fmt: skip
    return out, dropped
