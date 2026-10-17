"""Does the system say "I have nothing" when it has nothing, without refusing what it knows?

The old gate was `top score >= 0.30`, but the top item's score is normalised against itself, so it could never fail: asking
for a favourite pizza topping returned "favourite editor". This is a measured gate over a synthetic persona: 30 questions
whose answer is stored and 30 whose answer is not. Thresholds are on the numbers, so a regression fails the build.
"""

from __future__ import annotations

import pytest

from membridge.memory.consolidate import upsert
from membridge.memory.models import Candidate, EntityRef
from membridge.retrieval.engine import Retriever
from membridge.store.sqlite import Store

T0 = 1_700_000_000.0
DAY = 86400.0

FACTS = [
    # (statement, category, subject, predicate, object, entities, day)
    ("My favourite editor is Neovim.", "preference", "user", "favourite_editor_is", "Neovim", ["Neovim"], 0),
    ("I work at Acme Robotics.", "profile", "user", "work_at", "Acme Robotics", ["Acme Robotics"], 1),
    ("I study at Northfield University.", "education", "user", "study_at", "Northfield University", ["Northfield University"], 2),
    ("I am currently learning Rust.", "interest", "user", "studies", "Rust", ["Rust"], 3),
    ("I use Python for data analysis.", "skill", "user", "uses_tool", "Python", ["Python"], 4),
    ("I decided to use PostgreSQL for Atlas storage.", "decision", "atlas", "uses:storage", "PostgreSQL", ["Atlas", "PostgreSQL"], 5),
    ("I switched Atlas from PostgreSQL to SQLite because the demo needs zero setup.", "decision", "atlas", "uses:storage", "SQLite", ["Atlas", "SQLite"], 20),
    ("The Atlas prototype deadline moved to 5 March.", "constraint", "atlas", "deadline", "5 March", ["Atlas"], 21),
    ("I prefer dark mode in every editor.", "preference", "user", "prefers", "dark mode", ["dark mode"], 6),
    ("I live in Lisbon.", "profile", "user", "live_at", "Lisbon", ["Lisbon"], 7),
    ("My name is Maria Silva.", "profile", "user", "name_is", "Maria Silva", [], 8),
    ("I am building Orbit, a scheduling app.", "project", "orbit", "is_project", "Orbit", ["Orbit"], 9),
    ("I use Docker to ship releases.", "skill", "user", "uses_tool", "Docker", ["Docker"], 10),
    ("I prefer tabs over spaces.", "preference", "user", "prefers", "tabs over spaces", [], 11),
    ("I read research papers about continual learning.", "interest", "user", "studies", "continual learning", [], 12),
    ("My degree is Computer Science.", "education", "user", "degree_is", "Computer Science", [], 13),
    ("Orbit uses Redis for caching.", "decision", "orbit", "uses:caching", "Redis", ["Orbit", "Redis"], 14),
    ("I decided to deploy Orbit on Fly.io.", "decision", "orbit", "uses:deployment", "Fly.io", ["Orbit", "Fly.io"], 15),
]

# (question, text that must appear in some returned memory)
ANSWERABLE = [
    ("What is my favourite editor?", "Neovim"),
    ("Which editor do I like best?", "Neovim"),
    ("Where do I work?", "Acme Robotics"),
    ("What company do I work for?", "Acme Robotics"),
    ("Where do I study?", "Northfield"),
    ("Which university am I at?", "Northfield"),
    ("What programming language am I learning?", "Rust"),
    ("What am I learning right now?", "Rust"),
    ("What do I use Python for?", "data analysis"),
    ("What database does Atlas use now?", "SQLite"),
    ("What does Atlas use for storage?", "SQLite"),
    ("Which database did Atlas use before SQLite?", "PostgreSQL"),
    ("When is the Atlas deadline?", "5 March"),
    ("Tell me about the Atlas prototype deadline", "5 March"),
    ("Do I prefer dark mode?", "dark mode"),
    ("Where do I live?", "Lisbon"),
    ("Which city do I live in?", "Lisbon"),
    ("What is my name?", "Maria Silva"),
    ("What am I building?", "Orbit"),
    ("What is Orbit?", "Orbit"),
    ("What do I use to ship releases?", "Docker"),
    ("Do I prefer tabs or spaces?", "tabs over spaces"),
    ("What research do I read about?", "continual learning"),
    ("What is my degree?", "Computer Science"),
    ("What does Orbit use for caching?", "Redis"),
    ("How is Orbit deployed?", "Fly.io"),
    ("Where do I deploy Orbit?", "Fly.io"),
    ("What did I decide about Orbit deployment?", "Fly.io"),
    ("What storage did I choose for Atlas?", "SQLite"),
    ("What tools do I use?", "Docker"),
    ("Which database did I pick for Atlas?", "SQLite"),  # added after the second review: "pick" is not "decide"
    ("Who employs me?", "Acme Robotics"),
    ("What is my job?", "Acme Robotics"),
]

UNANSWERABLE = [
    "What is my favourite pizza topping?",
    "What is my blood type?",
    "What is my pet's name?",
    "How much salary do I earn?",
    "What is my shoe size?",
    "Which car do I drive?",
    "When is my wedding anniversary?",
    "What is my favourite colour?",
    "Where was I born?",
    "What is my passport number?",
    "What is the name of my cat?",
    "Which gym do I go to?",
    "What is my favourite movie?",
    "Who is my best friend?",
    "What did I have for breakfast?",
    "What is my mother's maiden name?",
    "Which airline do I fly with?",
    "What is my favourite football team?",
    "What instruments do I play?",
    "What is my sister's phone number?",
    "How tall am I?",
    "What was the weather like on my birthday?",
    "Which restaurant do I like?",
    "What is my favourite band?",
    "What language did I speak as a child?",
    "What are my hobbies on weekends?",
    "What kind of coffee do I drink?",
    "Which Kubernetes version does Atlas run?",
    "What is the Orbit pricing plan?",
    "Which cloud provider hosts the Atlas demo?",
    "How much does Atlas cost per month?",  # added after the second review: names a known project, asks for an unknown thing
    "Who is the Atlas product manager?",
    "Where is the park?",
    "Rex",
]


@pytest.fixture
def persona(store: Store) -> Retriever:
    for stmt, cat, subj, pred, obj, ents, day in FACTS:
        upsert(store, Candidate(
            statement=stmt, category=cat, kind="episodic" if cat == "decision" else "semantic", subject=subj, predicate=pred, object=obj,
            entities=[EntityRef(e, "project" if e in ("Atlas", "Orbit") else "concept") for e in ents], ref=f"d-{day}", ref_kind="document",
            quote=stmt, ts=T0 + day * DAY, verification="user_stated"))  # fmt: skip
    store.conn.commit()
    return Retriever(store)


def test_nothing_is_returned_when_the_answer_is_not_stored(persona: Retriever) -> None:
    wrong = [q for q in UNANSWERABLE if persona.search(q)["sufficient_evidence"]]
    assert len(wrong) <= 3, f"answered {len(wrong)}/{len(UNANSWERABLE)} unanswerable questions: {wrong}"  # >= 90% abstention recall


def test_known_answers_are_still_given(persona: Retriever) -> None:
    missed = []
    for q, gold in ANSWERABLE:
        pkg = persona.search(q, limit=5)
        if not (pkg["sufficient_evidence"] and any(gold.lower() in m["statement"].lower() for m in pkg["memories"])):
            missed.append(q)
    assert len(missed) <= 3, f"missed {len(missed)}/{len(ANSWERABLE)}: {missed}"  # >= 90% answered


def test_abstention_precision_and_false_abstention_rate(persona: Retriever) -> None:
    abstain_on_unanswerable = sum(not persona.search(q)["sufficient_evidence"] for q in UNANSWERABLE)
    abstain_on_answerable = sum(not persona.search(q, limit=5)["sufficient_evidence"] for q, _ in ANSWERABLE)
    abstentions = abstain_on_unanswerable + abstain_on_answerable
    precision = abstain_on_unanswerable / abstentions if abstentions else 1.0
    assert precision >= 0.9, (abstain_on_unanswerable, abstain_on_answerable)
    assert abstain_on_answerable / len(ANSWERABLE) <= 0.1


def test_an_abstention_returns_no_memories_and_no_raw_messages(persona: Retriever) -> None:
    pkg = persona.search("What is my favourite pizza topping?")
    assert pkg["sufficient_evidence"] is False
    assert pkg["memories"] == [] and pkg["source_messages"] == [] and pkg["graph_paths"] == []
    assert pkg["note"]


def test_each_returned_memory_says_why_it_was_supported(persona: Retriever) -> None:
    pkg = persona.search("What is my favourite editor?")
    assert pkg["memories"] and all(any(w.startswith("support:") for w in m["why_selected"]) for m in pkg["memories"])
