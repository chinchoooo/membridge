"""Record the dashboard as video clips (real UI, driven by Playwright through the installed Edge).

   uv run --with playwright python scripts/video/capture_dashboard.py [--base http://127.0.0.1:8790] [--live https://membridge-sandbox.onrender.com]

Start the sandbox first:   MEMBRIDGE_PUBLIC_HOST=127.0.0.1 uv run membridge sandbox --port 8790    (with FalkorDB running to get the Graph tab)
Output: submission/clips/dash_*.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "submission" / "clips"

CURSOR_JS = """
(() => {
  localStorage.setItem('mb_present', '1');
  const mk = () => {
    if (document.getElementById('__cur')) return;
    const c = document.createElement('div'); c.id = '__cur';
    c.style.cssText = 'position:fixed;z-index:99999;pointer-events:none;width:26px;height:26px;border-radius:50%;background:rgba(53,208,186,.35);border:3px solid #35d0ba;left:-50px;top:-50px;transform:translate(-50%,-50%);transition:transform .08s,background .15s';
    document.documentElement.appendChild(c);
    addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    addEventListener('mousedown', () => { c.style.transform = 'translate(-50%,-50%) scale(.6)'; c.style.background = 'rgba(255,180,84,.6)'; }, true);
    addEventListener('mouseup', () => { c.style.transform = 'translate(-50%,-50%) scale(1)'; c.style.background = 'rgba(53,208,186,.35)'; }, true);
  };
  if (document.readyState !== 'loading') mk(); else addEventListener('DOMContentLoaded', mk);
})();
"""


def glide(pg: Page, x: float, y: float, steps: int = 28) -> None:
    pg.mouse.move(x, y, steps=steps)
    pg.wait_for_timeout(120)


def click(pg: Page, loc, pause: int = 350) -> None:  # type: ignore[no-untyped-def]
    loc.scroll_into_view_if_needed()
    b = loc.bounding_box()
    assert b, "element not visible"
    glide(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
    pg.wait_for_timeout(pause)
    pg.mouse.down()
    pg.wait_for_timeout(90)
    pg.mouse.up()


def record(pw, name: str, url: str, flow) -> Path | None:  # type: ignore[no-untyped-def]
    tmp = Path(tempfile.mkdtemp(prefix="dashrec-"))
    browser = pw.chromium.launch(channel="msedge", headless=True)
    ctx = browser.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(tmp), record_video_size={"width": 1920, "height": 1080})
    ctx.add_init_script(CURSOR_JS)
    pg = ctx.new_page()
    try:
        pg.goto(url)
        pg.wait_for_timeout(1500)
        flow(pg)
    finally:
        video = pg.video
        ctx.close()
        browser.close()
    src = Path(video.path()) if video else None
    if not src or not src.exists():
        return None
    out = CLIPS / f"dash_{name}.mp4"
    CLIPS.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "18", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  dash_{name}.mp4")
    return out


def flow_hero(pg: Page) -> None:
    click(pg, pg.get_by_role("button", name="Where do I work?"))
    pg.wait_for_timeout(3600)
    click(pg, pg.get_by_role("button", name="What is my favourite pizza topping?"))
    pg.wait_for_timeout(4200)


def flow_graph(pg: Page) -> None:
    pg.wait_for_timeout(3500)  # layout settles
    for x, y in ((900, 600), (1100, 520), (820, 460), (1000, 640), (960, 540)):
        glide(pg, x, y, 40)
        pg.wait_for_timeout(500)
    pg.wait_for_timeout(2500)


def flow_private(pg: Page) -> None:
    pg.wait_for_timeout(1800)  # shows the "N private hidden" chip
    click(pg, pg.get_by_label("show private").first if pg.get_by_label("show private").count() else pg.locator("label", has_text="show private").locator("input"))
    pg.wait_for_timeout(2800)  # the PIN-required message
    click(pg, pg.get_by_role("button", name="Maintenance"))
    pg.wait_for_timeout(900)
    box = pg.locator("div.filters", has_text="View private memories here")
    box.scroll_into_view_if_needed()
    pg.wait_for_timeout(700)
    inp = box.locator("input")
    click(pg, inp)
    pg.keyboard.type("123456", delay=140)
    pg.wait_for_timeout(500)
    click(pg, box.get_by_role("button", name="Unlock for 10 minutes"))
    pg.wait_for_timeout(2200)
    click(pg, pg.get_by_role("button", name="Memories"))
    pg.wait_for_timeout(900)
    click(pg, pg.locator("label", has_text="show private").locator("input"))
    pg.wait_for_timeout(3800)


def flow_live(pg: Page) -> None:
    pg.wait_for_timeout(1500)
    click(pg, pg.get_by_role("button", name="What database does Atlas use now?"))
    pg.wait_for_timeout(4200)


def wait_up(base: str, tries: int = 45) -> bool:
    for _ in range(tries):
        try:
            urllib.request.urlopen(base + "/api/ping", timeout=8)
            return True
        except Exception:
            time.sleep(3)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8790")
    ap.add_argument("--live", default="https://membridge-sandbox.onrender.com")
    a = ap.parse_args()
    atlas = json.load(urllib.request.urlopen(a.base + "/api/entities?q=Atlas"))["items"][0]["id"]
    print("dashboard clips (real UI)")
    with sync_playwright() as pw:
        record(pw, "hero", a.base + "/", flow_hero)
        record(pw, "graph", f"{a.base}/#graph/{atlas}", flow_graph)
        record(pw, "private", a.base + "/#memories", flow_private)
        if a.live and wait_up(a.live):
            record(pw, "live", a.live + "/", flow_live)
        else:
            print("  (live sandbox not reachable: skipped)")


if __name__ == "__main__":
    main()
