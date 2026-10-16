"""A labelled mini-corpus for the deterministic extractor: statements it must catch, and chatter/questions it must ignore.
Precision matters more than recall here (a wrong memory is worse than a missing one), so negatives are asserted strictly."""

from __future__ import annotations

import pytest

from membridge.memory.rules import extract_from_message

POSITIVE = [
    ("I'm using FastAPI for the backend.", "skill", "FastAPI"),
    ("I am currently learning Rust.", "interest", "Rust"),
    ("My tech stack is React, Node and MySQL.", "skill", "React, Node and MySQL"),
    ("My favorite editor is Neovim.", "preference", "Neovim"),
    ("My degree is Computer Science and Engineering.", "education", "Computer Science and Engineering"),
    ("I study at Anna University.", "education", "Anna University"),
    ("I work at Acme Robotics.", "profile", "Acme Robotics"),
    ("My name is Priya Raman.", "profile", "Priya Raman"),
    ("I decided to use PostgreSQL for Atlas storage.", "decision", "PostgreSQL"),
    ("I prefer tabs over spaces.", "preference", "tabs over spaces"),
]
NEGATIVE = [
    "How do I use React hooks?",
    "Can you write a script that uses Python?",
    "I'm using this function but it throws an error, can you fix it?",
    "What is my stack?",
    "Fix the bug in my login page",
    "The user is using a proxy.",
    "Please explain how Docker works.",
    "I am a bit confused about this.",
    "my code is below, please review it",
    "I'm working with a deadline tomorrow",
]


@pytest.mark.parametrize(("text", "category", "obj"), POSITIVE)
def test_known_statements_are_extracted(text: str, category: str, obj: str) -> None:
    cands, _ = extract_from_message(text, message_id="m", ts=1.0, context_entity=None, known=set())
    assert any(c.category == category and c.object == obj for c in cands), [(c.category, c.object) for c in cands]


@pytest.mark.parametrize("text", NEGATIVE)
def test_chatter_and_questions_yield_nothing(text: str) -> None:
    cands, _ = extract_from_message(text, message_id="m", ts=1.0, context_entity=None, known=set())
    assert cands == [], [(c.category, c.object) for c in cands]


def test_precision_and_recall_on_the_corpus() -> None:
    hit = sum(bool(extract_from_message(t, message_id="m", ts=1.0, context_entity=None, known=set())[0]) for t, _, _ in POSITIVE)
    false = sum(bool(extract_from_message(t, message_id="m", ts=1.0, context_entity=None, known=set())[0]) for t in NEGATIVE)
    assert hit == len(POSITIVE) and false == 0
