"""Terminal helpers shared by the CLI and the first-run wizard."""

from __future__ import annotations

import getpass
import sys


def hidden(prompt: str) -> str:
    """A hidden prompt that only a person at a terminal can answer: stdin must be a real TTY (getpass quietly falls back
    to reading a pipe on some platforms, which would let a program feed it)."""
    if not sys.stdin.isatty():
        raise EOFError("this needs an interactive terminal")
    return getpass.getpass(prompt)


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()
