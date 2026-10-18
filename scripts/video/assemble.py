"""Assemble the demo video from generated cards/clips (+ your voice-over and live recording when present).

   uv run python scripts/video/assemble.py                 # silent preview -> submission/final/membridge_preview.mp4
   uv run python scripts/video/assemble.py --voice          # uses submission/inbox/voiceover.wav (+ live_chatgpt.mp4 for scene 3)
   --blur x:y:w:h                                          # blur a region of the live recording (default: ChatGPT's reply column)
   --live-speed 4                                          # speed factor for the waiting part of the live recording (start:end via --live-fast 30:90)

Scenes are cut where your voice-over has >= 1.5 s of silence. Each scene's visuals are stretched/held to fit its audio.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submission"
CARDS, CLIPS, INBOX, FINAL, TMP = SUB / "cards", SUB / "clips", SUB / "inbox", SUB / "final", SUB / "tmp"
W, H, FPS = 1920, 1080, 30
VF = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0b1020,setsar=1,fps={FPS},format=yuv420p"

# scene -> list of (file, preferred seconds | None = take what is left)
SCENES: dict[int, list[tuple[Path, float | None]]] = {
    1: [(CARDS / "01_title.png", 5), (CARDS / "02_problem.png", None)],
    2: [(CLIPS / "term_install.mp4", None), (CLIPS / "dash_hero.mp4", None)],
    3: [(INBOX / "live_chatgpt.mp4", None)],
    4: [(CARDS / "03_architecture.png", 14), (CLIPS / "dash_graph.mp4", None), (CARDS / "04_hop.png", 8), (CLIPS / "term_hop.mp4", None)],
    5: [(CLIPS / "term_trust.mp4", None), (CLIPS / "dash_private.mp4", None)],
    6: [(CARDS / "05_numbers.png", 12), (CARDS / "07_outro.png", None)],
}
PREVIEW_SECONDS = {1: 20, 2: 35, 3: 12, 4: 50, 5: 40, 6: 30}


def sh(*a: str) -> str:
    r = subprocess.run(a, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise SystemExit(f"ffmpeg failed: {' '.join(a)}\n{r.stderr[-1500:]}")
    return r.stderr + r.stdout


def duration(p: Path) -> float:
    raw = sh("ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)).strip().splitlines()[-1]
    try:
        return float(raw)
    except ValueError:  # some muxers carry no duration: decode and read the last timestamp
        times = re.findall(r"time=(\d+):(\d+):([\d.]+)", sh("ffmpeg", "-i", str(p), "-f", "null", "-"))
        h, m, s = times[-1]
        return int(h) * 3600 + int(m) * 60 + float(s)


def find_scenes(wav: Path) -> list[tuple[float, float]]:
    """Speech spans separated by >= 1.5 s of silence."""
    log = sh("ffmpeg", "-i", str(wav), "-af", "silencedetect=noise=-38dB:d=1.5", "-f", "null", "-")
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    total = duration(wav)
    spans, cur = [], 0.0
    for s, e in zip(starts, ends):
        if s - cur > 1.0:
            spans.append((cur, s + 0.4))
        cur = max(e - 0.4, 0.0)
    if total - cur > 1.0:
        spans.append((cur, total))
    return spans


def visual(src: Path, secs: float, out: Path, speed: float = 1.0, vf_extra: str = "") -> None:
    """One visual stretched / held / trimmed to `secs` seconds."""
    vf = (vf_extra + "," if vf_extra else "") + VF
    if src.suffix.lower() == ".png":
        sh("ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-t", f"{secs:.3f}", "-i", str(src), "-vf", f"{vf},fade=t=in:d=0.4", "-c:v", "libx264", "-crf", "18", str(out))
        return
    d = duration(src) / speed
    pad = max(secs - d, 0.0)
    sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", f"setpts=PTS/{speed},{vf},tpad=stop_mode=clone:stop_duration={pad:.3f},fade=t=in:d=0.3", "-t", f"{secs:.3f}", "-an", "-c:v", "libx264", "-crf", "18", str(out))


def plan(scene: int, secs: float) -> list[tuple[Path, float]]:
    items = [(p, s) for p, s in SCENES[scene] if p.exists()]
    if not items:
        return []
    fixed = sum(s for p, s in items if s is not None and p.suffix == ".png")
    free = [(p, s) for p, s in items if s is None or p.suffix != ".png" or True]
    # clips keep their natural length (capped), cards take their preferred time; the last item absorbs the remainder
    out: list[tuple[Path, float]] = []
    left = secs
    for i, (p, pref) in enumerate(items):
        natural = pref if pref is not None else (duration(p) + 0.8)
        take = left if i == len(items) - 1 else min(natural, left * 0.7)
        out.append((p, max(take, 1.0)))
        left -= take
    del fixed, free
    return out


def build_scene(n: int, secs: float, out: Path, blur: str | None, live_fast: tuple[float, float, float] | None) -> None:
    parts = []
    for i, (p, s) in enumerate(plan(n, secs)):
        seg = TMP / f"s{n}_{i}.mp4"
        extra = ""
        if p.name == "live_chatgpt.mp4" and blur:
            x, y, w, h = (int(v) for v in blur.split(":"))
            extra = f"split[a][b];[b]crop={w}:{h}:{x}:{y},boxblur=24:3[bl];[a][bl]overlay={x}:{y}"
        visual(p, s, seg, vf_extra=extra)
        parts.append(seg)
    lst = TMP / f"s{n}.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    sh("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out))
    _ = live_fast


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", action="store_true")
    ap.add_argument("--blur", default="960:150:960:800")
    ap.add_argument("--live-fast", default=None)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)

    wav = INBOX / "voiceover.wav"
    if a.voice and wav.exists():
        spans = find_scenes(wav)
        print("scene spans from the voice-over:", [(round(s, 1), round(e, 1)) for s, e in spans])
        voiced = [1, 2, 4, 5, 6]
        live = INBOX / "live_chatgpt.mp4"
        if len(spans) != len(voiced):
            raise SystemExit(f"expected {len(voiced)} voice-over scenes (1,2,4,5,6) separated by 2 s silences, found {len(spans)}. Re-record or adjust the silences.")
        order = [1, 2, 3, 4, 5, 6] if live.exists() else voiced
        parts, audio = [], []
        for n in order:
            seg = TMP / f"scene{n}.mp4"
            if n == 3:
                secs = duration(live)
                build_scene(3, secs, seg, a.blur, None)
                audio.append(("silence", secs))
            else:
                s, e = spans[voiced.index(n)]
                build_scene(n, e - s, seg, None, None)
                audio.append(("voice", (s, e)))
            parts.append(seg)
        lst = TMP / "all.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        silent = TMP / "all.mp4"
        sh("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(silent))
        # audio track: voice spans back to back, silence for the live scene (its own audio is kept if present)
        pieces = []
        for i, (k, v) in enumerate(audio):
            f = TMP / f"a{i}.wav"
            if k == "voice":
                s, e = v  # type: ignore[misc]
                sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ss", f"{s:.3f}", "-to", f"{e:.3f}", str(f))
            else:
                sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(live), "-vn", "-t", f"{v:.3f}", str(f))
            pieces.append(f)
        al = TMP / "alist.txt"
        al.write_text("".join(f"file '{p.as_posix()}'\n" for p in pieces), encoding="utf-8")
        mixed = TMP / "audio.wav"
        sh("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(al), "-af", "loudnorm=I=-16:TP=-1.5", str(mixed))
        out = FINAL / "membridge_demo.mp4"
        sh("ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(mixed), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out))
        print(f"wrote {out} ({duration(out):.0f}s)")
        return

    parts = []
    for n, secs in PREVIEW_SECONDS.items():
        seg = TMP / f"scene{n}.mp4"
        if not plan(n, secs):
            continue
        build_scene(n, secs, seg, a.blur, None)
        parts.append(seg)
    lst = TMP / "all.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    out = FINAL / "membridge_preview.mp4"
    sh("ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out))
    print(f"wrote {out} ({duration(out):.0f}s, silent preview; scene 3 appears once you record it)")


if __name__ == "__main__":
    main()
