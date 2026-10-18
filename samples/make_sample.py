"""Compatibility shim: the synthetic export now lives in `membridge.demo_data` so the installed package can use it.

Run: uv run python samples/make_sample.py  -> samples/chatgpt_export/conversations.json
"""

from __future__ import annotations

import json
from pathlib import Path

from membridge.demo_data import T0, _msg, build_export, conv  # noqa: F401  (re-exported: tests and evals load this file by path)

if __name__ == "__main__":
    out = Path(__file__).parent / "chatgpt_export" / "conversations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_export(), indent=1), encoding="utf-8")
    print(f"wrote {out}")
