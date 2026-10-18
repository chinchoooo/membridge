"""Demo of the v0.2 beats on a throw-away, synthetic store (no private data, no Docker, no model):

  1. it says "I don't know" instead of answering with something unrelated
  2. a ChatGPT memory dump is imported as *ChatGPT's belief*, never as "you said it", and cannot overrule what you said
  3. private memories are invisible to an assistant until you grant that one assistant access with your PIN
  4. an assistant can ask to forget something; only you can confirm, and the text is really gone

Run:  uv run python scripts/demo_trust.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from membridge.memory.consolidate import upsert  # noqa: E402
from membridge.memory.models import Candidate, EntityRef  # noqa: E402
from membridge.memory.seed_text import seed_from_text  # noqa: E402
from membridge.memory.service import MemoryService, ServiceError  # noqa: E402
from membridge.store.sqlite import Store  # noqa: E402

T0 = 1_700_000_000.0
DUMP = """\
FACT | work | Alex works at Globex Labs. | memory | high | ? | N
FACT | skill | Alex uses Haskell for compilers. | memory | high | ? | N
FACT | skill | Alex uses Go for services. | memory | high | ? | N
FACT | health | Alex takes sertraline 50 mg for anxiety. | memory | high | ? | Y
CHANGE | Editor | VS Code -> Neovim | ? | memory | high
"""


def banner(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    td = Path(tempfile.mkdtemp(prefix="membridge-trust-demo-"))
    store = Store(td / "demo.db")
    me = MemoryService(store, actor="me", human=True)  # you, at your own terminal / dashboard
    agent = MemoryService(store, actor="claude-code")  # an assistant connected over MCP

    # what you told us yourself
    for i, (text, pred, obj, cat) in enumerate([
        ("I work at Acme Robotics.", "work_at", "Acme Robotics", "profile"),
        ("My favourite editor is Neovim.", "favourite_editor_is", "Neovim", "preference"),
    ]):
        upsert(store, Candidate(statement=text, category=cat, subject="user", predicate=pred, object=obj, ref=f"you-{i}", ts=T0 + i,
                                entities=[EntityRef(obj)], verification="user_stated"))
    store.conn.commit()

    banner("1. It says 'I don't know' instead of guessing")
    for q in ("Where do I work?", "What is my favourite pizza topping?", "How much salary do I earn?"):
        r = agent.search_memory(q)
        print(f"  Q: {q}\n     -> " + (r["memories"][0]["statement"] if r["sufficient_evidence"] else f"(nothing stored) {r['note']}"))

    banner("2. A ChatGPT memory dump is ChatGPT's belief, not what you said")
    dump = td / "chatgpt-dump.txt"
    dump.write_text(DUMP, encoding="utf-8")
    print("  import:", seed_from_text(store, dump))
    for row in store.conn.execute("SELECT statement, verification, status FROM memories WHERE statement LIKE '%work% at%' ORDER BY verification DESC"):
        print(f"  [{row['verification']:15}] {row['status']:10} {row['statement']}")
    print("  -> Acme (you said it) stays current; Globex (ChatGPT's summary, no date) is kept as history, outranked.")

    banner("3. Private memories: invisible until YOU grant one assistant access, with a PIN")
    r = agent.search_memory("What do I take for anxiety?")
    print(f"  assistant asks what you take for anxiety -> sufficient_evidence={r['sufficient_evidence']}, memories={len(r['memories'])}")
    print(f"  notice (identical for every query, so it cannot reveal anything): {r['sensitive_notice'][:90]}...")
    try:
        agent.grant_sensitive("123456", "claude-code")
    except ServiceError:
        print("  assistant tries to grant itself access -> refused (only the user can grant access)")
    me.set_pin("246810")
    me.grant_sensitive("246810", "claude-code", minutes=10, uses=2)
    r = agent.search_memory("What do I take for anxiety?", include_sensitive=True)
    print(f"  after your PIN-created grant (10 minutes, 2 reads, this assistant only): {[m['statement'] for m in r['memories']]}")
    rows = store.conn.execute("SELECT action, actor FROM audit_log WHERE action LIKE 'sensitive_%' ORDER BY id")
    print("  audit log: " + "; ".join(f"{r['action']} by {r['actor']}" for r in rows))

    banner("4. An assistant can only ASK to forget; you confirm; the text is really gone")
    mid = me.save_memory("The door code for the office is purple-tangerine", "fact", "user_stated")["memory_id"]
    ask = agent.forget_memory(mid)
    print(f"  assistant: {ask['status']} (no token to pass back)")
    first = me.forget_memory(mid)
    me.forget_memory(mid, first["confirm_token"])
    store.conn.commit()
    store.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw = (td / "demo.db").read_bytes().lower()
    print(f"  you confirmed. 'tangerine' left in the database file: {b'tangerine' in raw}")
    print(f"  re-saving it is blocked: {me.save_memory('The door code for the office is purple-tangerine', 'fact', 'user_stated')['result']}")
    print(f"\nDemo store (delete freely): {td}")


if __name__ == "__main__":
    main()
