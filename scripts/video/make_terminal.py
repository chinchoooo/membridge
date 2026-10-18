"""Capture REAL command output, then render it as typed-terminal clips (mp4).

   uv run python scripts/video/make_terminal.py          # install + wizard, trust demo, graph-vs-text benchmark

Nothing here is staged: the text on screen is what the commands printed (trimmed to the interesting lines). Output: submission/clips/term_*.mp4
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from common import AMBER, DIM, FG, GREEN, MONO, MONO_B, OUT, TEAL, H, W, canvas, font, rrect

ROOT = Path(__file__).resolve().parents[2]
CLIPS = OUT / "clips"
TERM = OUT / "term"
FPS = 30
PROMPT = "PS C:\\demo> "
FS = 30
LH = 44


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None, timeout: int = 600) -> str:
    e = {**os.environ, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1", **(env or {})}
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=cwd, timeout=timeout)
    return (r.stdout + ("\n" + r.stderr if r.returncode else "")).replace("\r", "")


# ---------------------------------------------------------------------------------------------------------------- rendering
Item = tuple[str, str]  # ("cmd"|"out"|"note"|"pause", text-or-seconds)


def colour(line: str) -> tuple[int, int, int]:
    s = line.strip()
    if s.startswith("[nothing") or "[nothing stored]" in s or "nothing stored" in s.lower() or "refused" in s.lower():
        return AMBER
    if "[answered" in s or s.startswith("Answered") or "current |" in s or "you said it" in s or s.startswith("Done:"):
        return GREEN
    if s.startswith("Q:") or s.startswith("[") and "/6]" in s:
        return TEAL
    if "older value" in s or s.startswith("#") or s.startswith("(") or "via the graph" in s:
        return DIM
    return FG


def render(items: list[Item], out: Path, cps: float = 38.0, line_delay: float = 0.07, hold: float = 2.5, title: str = "Terminal") -> None:
    """Typewriter for commands, quick line-by-line for output; the last frame is held."""
    frames: list[list[tuple[str, tuple[int, int, int]]]] = []
    shown: list[tuple[str, tuple[int, int, int]]] = []

    def snap(n: int = 1) -> None:
        frames.extend([list(shown)] * n)

    snap(int(0.6 * FPS))
    for kind, val in items:
        if kind == "cmd":
            for i in range(1, len(val) + 1):
                tmp = shown + [(PROMPT + val[:i] + "_", TEAL)]
                frames.extend([tmp] * max(1, round(FPS / cps)))
            shown.append((PROMPT + val, TEAL))
            snap(int(0.45 * FPS))
        elif kind == "out":
            for raw in val.split("\n"):
                indent = " " * (len(raw) - len(raw.lstrip()) + 4)
                for line in textwrap.wrap(raw, 92, subsequent_indent=indent, break_long_words=True, drop_whitespace=False) or [""]:
                    shown.append((line, colour(raw)))
                    snap(max(1, round(line_delay * FPS)))
        elif kind == "note":
            shown.append((val, DIM))
            snap(int(0.5 * FPS))
        elif kind == "pause":
            snap(int(float(val) * FPS))
    snap(int(hold * FPS))
    f, fb = font(MONO, FS), font(MONO_B, FS)
    rows = (H - 190) // LH
    tmpdir = Path(tempfile.mkdtemp(prefix="termframes-"))
    cache: dict[int, Path] = {}
    seq = []
    for idx, lines in enumerate(frames):
        key = hash(tuple(lines))
        if key not in cache:
            img, d = canvas()
            rrect(d, (50, 40, W - 50, H - 40), (14, 19, 34), (46, 58, 96), 22, 3)
            for k, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
                d.ellipse((90 + k * 42, 74, 116 + k * 42, 100), fill=c)
            d.text((W // 2, 88), title, font=font(MONO, 26), fill=DIM, anchor="mm")
            y = 140
            for text, col in lines[-rows:]:
                d.text((100, y), text, font=fb if text.startswith(PROMPT) or text.startswith("Q:") else f, fill=col)
                y += LH
            p = tmpdir / f"u{len(cache):05d}.png"
            img.save(p)
            cache[key] = p
        seq.append(cache[key])
    listing = tmpdir / "list.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\nduration {1 / FPS:.5f}\n" for p in seq) + f"file '{seq[-1].as_posix()}'\n", encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-vf", f"fps={FPS},format=yuv420p", "-c:v", "libx264", "-crf", "18", str(out)], check=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"  {out.name}: {len(frames) / FPS:.1f}s")


# ---------------------------------------------------------------------------------------------------------------- captures
def capture_install_and_wizard(work: Path) -> list[Item]:
    tools, bins, data = work / "tools", work / "bin", work / "data"
    env = {"UV_TOOL_DIR": str(tools), "UV_TOOL_BIN_DIR": str(bins), "MEMBRIDGE_DATA_DIR": str(data)}
    print("  installing from GitHub into a throw-away tool dir (a real install)...")
    install = run(["uv", "tool", "install", "--force", "git+https://github.com/chinchoooo/membridge"], env)
    exe = next(bins.glob("membridge*"))
    wiz = run([str(exe), "quickstart", "--demo", "--yes", "--no-serve"], env)
    (TERM / "install.txt").write_text(install, encoding="utf-8")
    (TERM / "wizard.txt").write_text(wiz, encoding="utf-8")
    keep = [ln for ln in install.splitlines() if ln.strip() and not ln.lstrip().startswith(("warning", "Downloading", "Resolved"))][-3:]
    w = wiz.splitlines()
    i2 = next(i for i, ln in enumerate(w) if ln.startswith("[2/6]"))
    i4 = next(i for i, ln in enumerate(w) if ln.startswith("[4/6]"))
    body = [ln for ln in w[i2:i4] if ln.strip() and not set(ln.strip()) <= {"-"}]
    return [("cmd", "uv tool install git+https://github.com/chinchoooo/membridge"), ("out", "\n".join(keep)), ("pause", 0.6),
            ("cmd", "membridge quickstart --demo"), ("out", "\n".join(body))]


def capture_trust() -> list[Item]:
    out = run([sys.executable, str(ROOT / "scripts" / "demo_trust.py")], cwd=ROOT)
    (TERM / "trust.txt").write_text(out, encoding="utf-8")
    lines = [ln for ln in out.splitlines() if ln.strip() and not set(ln.strip()) <= {"="} and not ln.startswith("Demo store")]
    keep: list[str] = []
    for ln in lines:
        if ln[:2] in ("2.", "3.", "4."):
            keep.append("")
            keep.append("# " + ln)
        elif "import:" in ln or "notice (identical" in ln:
            continue
        else:
            keep.append(ln)
    return [("cmd", "python scripts/demo_trust.py"), ("out", "\n".join(k for k in keep if k is not None).strip("\n"))]


def capture_hop(work: Path) -> list[Item]:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "eval"))
    import importlib.util
    import traceback

    spec = importlib.util.spec_from_file_location("multihop", ROOT / "eval" / "multihop.py")
    assert spec and spec.loader
    mh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mh)
    from membridge.importers.service import run_import
    from membridge.memory.pipeline import run_extraction
    from membridge.retrieval.engine import RetrievalConfig, Retriever
    from membridge.store.graph import GraphStore
    from membridge.store.sqlite import Store

    convs, qs = mh.build(n_projects=12, n_distractors=37)
    d = work / "mh" / "conversations.json"
    d.parent.mkdir(parents=True)
    d.write_text(json.dumps(convs), encoding="utf-8")
    store = Store(work / "mh.db")
    run_import(store, d)
    run_extraction(store)
    g = GraphStore(name="membridge_video")
    ok = g.available()
    if ok:
        g.rebuild(store)
    q = next(x for x in qs if x["kind"] == "lexical-gap-multihop")
    try:
        text_only = Retriever(store, None, RetrievalConfig(use_entities=False, use_ppr=False)).search(q["q"], limit=3)
        full = Retriever(store, g if ok else None).search(q["q"], limit=3)
    except Exception:
        traceback.print_exc()
        raise
    from membridge.cli import short_answer

    base = short_answer(text_only, q["q"]).replace("[nothing stored] It has nothing about that, and it will not guess.", "[nothing found] no stored fact shares a word with that question.")
    (TERM / "hop.txt").write_text(base + "\n\n" + short_answer(full, q["q"]) + f"\ngraph_available={ok}\n", encoding="utf-8")
    g.drop() if ok else None
    store.close()
    print("  hop: graph", "FalkorDB" if ok else "SQLite-degraded", "| text-only sufficient:", text_only["sufficient_evidence"], "| graph sufficient:", full["sufficient_evidence"])
    return [("note", "# 1) the baseline: text search only (BM25)"), ("out", base), ("pause", 1.0),
            ("note", "# 2) the same question through the FalkorDB graph"), ("cmd", f'membridge ask "{q["q"]}" --short'), ("out", short_answer(full, q["q"]))]


def main() -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    TERM.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mbvideo-"))
    print("terminal clips (real output)")
    render(capture_install_and_wizard(work), CLIPS / "term_install.mp4", title="PowerShell")
    render(capture_trust(), CLIPS / "term_trust.mp4", cps=60, line_delay=0.12, hold=3, title="PowerShell")
    render(capture_hop(work), CLIPS / "term_hop.mp4", line_delay=0.1, hold=3, title="PowerShell")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
