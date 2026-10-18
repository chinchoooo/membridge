"""Still cards, After Effects layers and the thumbnail.   uv run python scripts/video/make_cards.py [--tests N]"""

from __future__ import annotations

import argparse

from common import (
    AMBER,
    BG,
    BLUE,
    DIM,
    FG,
    GREEN,
    MONO,
    OUT,
    PANEL,
    PANEL2,
    RED,
    SANS,
    SANS_B,
    TEAL,
    H,
    W,
    arrow,
    canvas,
    font,
    paragraph,
    pill,
    rrect,
    text,
)

CARDS = OUT / "cards"
AE = OUT / "ae"


def title() -> None:
    img, d = canvas()
    text(d, (W // 2, 380), "MemBridge", font(SANS_B, 170), FG, "mm")
    text(d, (W // 2, 530), "Memory you own.", font(SANS, 64), TEAL, "mm")
    text(d, (W // 2, 610), "That knows what it doesn't know.", font(SANS, 64), FG, "mm")
    x = 560
    for s in ("local-first", "FalkorDB graph", "MCP", "private by default"):
        x, _ = pill(d, x, 740, s, font(SANS_B, 30), PANEL2, DIM)
        x += 16
    img.save(CARDS / "01_title.png")


def problem() -> None:
    img, d = canvas()
    text(d, (120, 90), "Typical AI memory", font(SANS_B, 66), FG)
    left = [("A list you can't read", "no source, no history, no way to see why it believes something"),
            ("Answers you never gave it", "it fills gaps with confident guesses"),
            ("Stuck with one provider", "switch assistants and it forgets you")]
    right = [("Memory you can inspect", "every fact links to the exact words it came from"),
             ("It says \"I don't know\"", "nothing stored means nothing returned: no guessing"),
             ("Yours, on your machine", "any assistant can use it, under rules you set")]
    for i, ((a, b), (c, e)) in enumerate(zip(left, right)):
        y = 250 + i * 245
        rrect(d, (120, y, 920, y + 205), PANEL, RED)
        text(d, (160, y + 28), "x  " + a, font(SANS_B, 42), RED)
        paragraph(d, 160, y + 98, b, font(SANS, 32), 720, DIM)
        arrow(d, (950, y + 102), (1000, y + 102), DIM, 6)
        rrect(d, (1030, y, 1800, y + 205), PANEL, TEAL)
        text(d, (1070, y + 28), c, font(SANS_B, 42), TEAL)
        paragraph(d, 1070, y + 98, e, font(SANS, 32), 700, FG)
    img.save(CARDS / "02_problem.png")


ARCH_BOXES = [
    ("Your history", ["ChatGPT / Claude", "exports", "ChatGPT memory dump"], BLUE),
    ("SQLite", ["source of truth", "messages + evidence", "history + audit log"], FG),
    ("FalkorDB", ["rebuildable graph", "Memory, Entity,", "Conversation, Message", "Cypher, full-text,", "vector, PageRank"], TEAL),
    ("Retrieval", ["text + graph hop", "PageRank", "evidence gate", "says \"I don't know\""], AMBER),
    ("Assistants", ["MCP (Claude Code)", "dashboard", "CLI"], GREEN),
]


def _arch_geometry() -> list[tuple[int, int, int, int]]:
    bw, gap, y0, h = 330, 45, 330, 440
    x0 = (W - (5 * bw + 4 * gap)) // 2
    return [(x0 + i * (bw + gap), y0, x0 + i * (bw + gap) + bw, y0 + h) for i in range(5)]


def architecture() -> None:
    boxes = _arch_geometry()
    img, d = canvas()
    text(d, (W // 2, 120), "How MemBridge works", font(SANS_B, 64), FG, "mm")
    text(d, (W // 2, 200), "SQLite is the truth. FalkorDB is the rebuildable graph that makes memory explainable.", font(SANS, 34), DIM, "mm")
    for (name, lines, col), b in zip(ARCH_BOXES, boxes):
        rrect(d, b, PANEL, col, 26, 5 if name == "FalkorDB" else 3)
        text(d, ((b[0] + b[2]) // 2, b[1] + 55), name, font(SANS_B, 42), col, "mm")
        y = b[1] + 120
        for ln in lines:
            text(d, ((b[0] + b[2]) // 2, y), ln, font(SANS, 26), FG, "mm")
            y += 46
    for i in range(4):
        arrow(d, (boxes[i][2] + 4, 550), (boxes[i + 1][0] - 4, 550), DIM, 7)
    cx = (boxes[2][0] + boxes[2][2]) // 2
    rrect(d, (boxes[1][0], 835, boxes[3][2], 975), PANEL2, TEAL, 20, 3)
    text(d, (cx, 880), "private memories are never projected into the graph", font(SANS_B, 30), TEAL, "mm")
    text(d, (cx, 935), "graph down? answers are identical, straight from SQLite", font(SANS, 28), DIM, "mm")
    img.save(CARDS / "03_architecture.png")
    # layered version for After Effects
    d0 = AE / "arch_layers"
    d0.mkdir(parents=True, exist_ok=True)
    for i, ((name, lines, col), b) in enumerate(zip(ARCH_BOXES, boxes)):
        li, ld = canvas(True)
        rrect(ld, b, PANEL, col, 26, 5 if name == "FalkorDB" else 3)
        text(ld, ((b[0] + b[2]) // 2, b[1] + 55), name, font(SANS_B, 42), col, "mm")
        y = b[1] + 120
        for ln in lines:
            text(ld, ((b[0] + b[2]) // 2, y), ln, font(SANS, 26), FG, "mm")
            y += 46
        li.save(d0 / f"box_{i + 1}_{name.split()[0].lower()}.png")
    for i in range(4):
        li, ld = canvas(True)
        arrow(ld, (boxes[i][2] + 4, 550), (boxes[i + 1][0] - 4, 550), DIM, 7)
        li.save(d0 / f"arrow_{i + 1}.png")
    bg, bd = canvas()
    text(bd, (W // 2, 120), "How MemBridge works", font(SANS_B, 64), FG, "mm")
    bg.save(d0 / "background_title.png")


HOP = [("Message", "\"The nightly ZX0 timeout keeps killing the Orion job\"", AMBER),
       ("Conversation", "Orion: nightly job", BLUE),
       ("Project", "Orion", TEAL),
       ("Decision", "I switched Orion from MySQL to Neo4j", GREEN)]


def hop() -> None:
    img, d = canvas()
    text(d, (W // 2, 100), "No shared words. The graph still finds it.", font(SANS_B, 60), FG, "mm")
    rrect(d, (360, 170, 1560, 262), PANEL2, AMBER, 46, 3)
    text(d, (W // 2, 216), "What did I change after the ZX0 timeout?", font(SANS_B, 44), AMBER, "mm")
    xs = [70, 540, 1010, 1480]
    for i, ((kind, label, col), x) in enumerate(zip(HOP, xs)):
        rrect(d, (x, 360, x + 370, 650), PANEL, col, 28, 4)
        text(d, (x + 185, 400), kind.upper(), font(SANS_B, 28), col, "mm")
        paragraph(d, x + 24, 450, label, font(SANS, 31), 322, FG)
        if i < 3:
            arrow(d, (x + 380, 505), (xs[i + 1] - 10, 505), col, 8)
    text(d, (W // 2, 730), "hops: message  >  conversation  >  project  >  decision", font(SANS, 34), DIM, "mm")
    for j, (label, val, col, frac) in enumerate([("Text search (BM25)", "0 / 12", RED, 0.0), ("FalkorDB graph", "12 / 12", TEAL, 1.0)]):
        y = 810 + j * 105
        text(d, (200, y + 32), label, font(SANS_B, 36), FG, "lm")
        rrect(d, (700, y, 1500, y + 64), PANEL, None, 32)
        if frac:
            rrect(d, (700, y, 700 + int(800 * frac), y + 64), col, None, 32)
        text(d, (1560, y + 32), val, font(SANS_B, 46), col, "lm")
    text(d, (W // 2, 1040), "48-question benchmark, 12 of them share no words with their answer (docs/EVAL.md)", font(SANS, 26), DIM, "mm")
    img.save(CARDS / "04_hop.png")
    d0 = AE / "hop_layers"
    d0.mkdir(parents=True, exist_ok=True)
    for i, ((kind, label, col), x) in enumerate(zip(HOP, xs)):
        li, ld = canvas(True)
        rrect(ld, (x, 360, x + 370, 650), PANEL, col, 28, 4)
        text(ld, (x + 185, 400), kind.upper(), font(SANS_B, 28), col, "mm")
        paragraph(ld, x + 24, 450, label, font(SANS, 31), 322, FG)
        li.save(d0 / f"node_{i + 1}_{kind.lower()}.png")
        if i < 3:
            li, ld = canvas(True)
            arrow(ld, (x + 380, 505), (xs[i + 1] - 10, 505), col, 8)
            li.save(d0 / f"edge_{i + 1}.png")
    li, ld = canvas(True)
    rrect(ld, (360, 170, 1560, 262), PANEL2, AMBER, 46, 3)
    text(ld, (W // 2, 216), "What did I change after the ZX0 timeout?", font(SANS_B, 44), AMBER, "mm")
    li.save(d0 / "question.png")


def numbers(tests: int) -> None:
    img, d = canvas()
    text(d, (W // 2, 110), "Measured, not claimed", font(SANS_B, 64), FG, "mm")
    tiles = [("12 / 12  vs  0 / 12", "multi-hop questions with no shared words: FalkorDB graph vs text search", TEAL),
             ("34 / 34", "unanswerable questions declined instead of guessed (synthetic test set)", AMBER),
             (f"{tests}", "automated tests, green on Linux and Windows in CI", GREEN),
             ("2", "independent security reviews by a separate model; every finding fixed", BLUE)]
    for i, (big, small, col) in enumerate(tiles):
        x, y = 120 + (i % 2) * 880, 230 + (i // 2) * 360
        rrect(d, (x, y, x + 820, y + 320), PANEL, col, 28, 4)
        text(d, (x + 40, y + 50), big, font(SANS_B, 84), col)
        paragraph(d, x + 40, y + 175, small, font(SANS, 34), 740, FG)
    text(d, (W // 2, 1000), "Not claimed: accuracy on real data (never measured). Benchmarks are synthetic.", font(SANS, 28), DIM, "mm")
    img.save(CARDS / "05_numbers.png")


def trust() -> None:
    img, d = canvas()
    text(d, (W // 2, 100), "Trustworthy by design", font(SANS_B, 64), FG, "mm")
    ladder = [("You verified it", GREEN), ("You said it", GREEN), ("An assistant wrote it", AMBER), ("An AI provider's summary of you", BLUE), ("A model's guess", DIM)]
    text(d, (120, 215), "Who vouches decides conflicts", font(SANS_B, 38), TEAL)
    for i, (s, col) in enumerate(ladder):
        y = 285 + i * 118
        rrect(d, (120 + i * 26, y, 860 + i * 26, y + 92), PANEL, col, 20, 3)
        text(d, (150 + i * 26, y + 46), f"{5 - i}   {s}", font(SANS_B, 34), col, "lm")
    items = [("Hidden by default", "health, money, family and ID facts never reach an assistant", TEAL),
             ("Your PIN, one assistant, minutes", "a grant is short, single-client and use-counted", AMBER),
             ("Assistants can only ask", "forget needs your confirmation; the text is erased from the file", GREEN)]
    for i, (a, b, col) in enumerate(items):
        y = 215 + i * 270
        rrect(d, (1000, y, 1800, y + 235), PANEL, col, 24, 4)
        text(d, (1036, y + 30), a, font(SANS_B, 38), col)
        paragraph(d, 1036, y + 100, b, font(SANS, 31), 730, FG)
    img.save(CARDS / "06_trust.png")


def outro() -> None:
    img, d = canvas()
    text(d, (W // 2, 170), "Try it now", font(SANS_B, 100), FG, "mm")
    rrect(d, (170, 290, 1750, 400), (6, 10, 22), TEAL, 18, 3)
    text(d, (W // 2, 345), "uvx --from git+https://github.com/chinchoooo/membridge membridge quickstart --demo", font(MONO, 31), TEAL, "mm")
    rows = [("Live sandbox (read-only, made-up data)", "membridge-sandbox.onrender.com", AMBER),
            ("Code", "github.com/chinchoooo/membridge", BLUE)]
    for i, (a, b, col) in enumerate(rows):
        y = 500 + i * 170
        text(d, (W // 2, y), a, font(SANS, 36), DIM, "mm")
        text(d, (W // 2, y + 70), b, font(SANS_B, 62), col, "mm")
    text(d, (W // 2, 930), "MemBridge: memory you own, that knows what it doesn't know.", font(SANS_B, 46), FG, "mm")
    text(d, (W // 2, 1010), "Built with FalkorDB, SQLite, MCP", font(SANS, 32), DIM, "mm")
    img.save(CARDS / "07_outro.png")


def thumbnail() -> None:
    img, d = canvas(size=(1280, 720))
    text(d, (70, 170), "MemBridge", font(SANS_B, 130), FG)
    text(d, (74, 330), "Memory you own.", font(SANS_B, 66), TEAL)
    text(d, (74, 410), "That knows what it doesn't know.", font(SANS_B, 66), FG)
    x = 74
    for s, col in (("FalkorDB graph", TEAL), ("local-first", PANEL2), ("MCP", PANEL2)):
        x, _ = pill(d, x, 560, s, font(SANS_B, 32), col, BG if col == TEAL else DIM)
        x += 14
    rrect(d, (880, 140, 1220, 330), PANEL, AMBER, 24, 4)
    text(d, (1050, 190), "Nothing stored", font(SANS_B, 34), AMBER, "mm")
    text(d, (1050, 245), "it will not guess", font(SANS, 30), FG, "mm")
    img.save(OUT / "thumbnail.png")


def overlays() -> None:
    AE.mkdir(parents=True, exist_ok=True)
    for name, label in (("lower_third_demo", "Live demo"), ("lower_third_live", "My real ChatGPT account")):
        li, ld = canvas(True)
        rrect(ld, (80, 880, 80 + 40 + int(ld.textlength(label, font=font(SANS_B, 44))) + 40, 970), (11, 16, 32, 230), TEAL, 14, 3)
        text(ld, (120, 925), label, font(SANS_B, 44), FG, "lm")
        li.save(AE / f"{name}.png")
    li, ld = canvas(True)
    rrect(ld, (460, 400, 1460, 640), (45, 33, 12, 245), AMBER, 28, 5)
    text(ld, (W // 2, 480), "Nothing stored about that", font(SANS_B, 66), AMBER, "mm")
    text(ld, (W // 2, 570), "so it will not guess.", font(SANS_B, 66), FG, "mm")
    li.save(AE / "callout_idontknow.png")
    for name, label, col in (("badge_falkordb", "FalkorDB", TEAL), ("badge_private", "private: hidden", AMBER)):
        li, ld = canvas(True)
        pill(ld, 100, 100, label, font(SANS_B, 44), col)
        li.crop((90, 90, 700, 220)).save(AE / f"{name}.png")


def svg_files() -> None:
    boxes = _arch_geometry()
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="#0b1020"/>']
    for (name, lines, col), b in zip(ARCH_BOXES, boxes):
        c = "#%02x%02x%02x" % col
        parts.append(f'<g id="{name.split()[0].lower()}"><rect x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" rx="26" fill="#141c36" stroke="{c}" stroke-width="4"/>'
                     f'<text x="{(b[0]+b[2])//2}" y="{b[1]+68}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="44" font-weight="700" fill="{c}">{name}</text>'
                     + "".join(f'<text x="{(b[0]+b[2])//2}" y="{b[1]+128+i*44}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="26" fill="#e8ecf8">{ln.replace("&", "&amp;").replace("<", "&lt;")}</text>' for i, ln in enumerate(lines)) + "</g>")
    for i in range(4):
        parts.append(f'<path id="arrow{i+1}" d="M{boxes[i][2]+4} 550 L{boxes[i+1][0]-4} 550" stroke="#8ea0c9" stroke-width="7" fill="none"/>')
    parts.append("</svg>")
    (AE / "arch_base.svg").write_text("".join(parts), encoding="utf-8")
    xs = [70, 540, 1010, 1480]
    hp = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="#0b1020"/>']
    for i, ((kind, label, col), x) in enumerate(zip(HOP, xs)):
        c = "#%02x%02x%02x" % col
        hp.append(f'<g id="node{i+1}"><rect x="{x}" y="360" width="370" height="290" rx="28" fill="#141c36" stroke="{c}" stroke-width="4"/><text x="{x+185}" y="410" text-anchor="middle" font-family="Segoe UI,Arial" font-size="28" font-weight="700" fill="{c}">{kind.upper()}</text></g>')
        if i < 3:
            hp.append(f'<path id="edge{i+1}" d="M{x+380} 505 L{xs[i+1]-10} 505" stroke="{c}" stroke-width="8" fill="none"/>')
    hp.append("</svg>")
    (AE / "hop_nodes.svg").write_text("".join(hp), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", type=int, default=357)
    a = ap.parse_args()
    CARDS.mkdir(parents=True, exist_ok=True)
    AE.mkdir(parents=True, exist_ok=True)
    for fn in (title, problem, architecture, hop, trust, outro, thumbnail, overlays, svg_files):
        fn()
    numbers(a.tests)
    print("wrote", sorted(p.name for p in CARDS.iterdir()), "+ AE layers in", AE)


if __name__ == "__main__":
    main()
