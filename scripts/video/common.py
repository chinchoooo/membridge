"""Shared look for the demo video assets: colours, fonts, small drawing helpers. Output goes to submission/ (git-ignored)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "submission"
W, H = 1920, 1080

BG = (11, 16, 32)
PANEL = (20, 28, 54)
PANEL2 = (28, 40, 76)
FG = (232, 236, 248)
DIM = (142, 160, 201)
TEAL = (53, 208, 186)
AMBER = (255, 180, 84)
RED = (255, 107, 107)
GREEN = (90, 208, 122)
BLUE = (110, 160, 255)

FONTS = Path("C:/Windows/Fonts")


def _first(names: list[str]) -> str:
    for n in names:
        if (FONTS / n).exists():
            return str(FONTS / n)
    raise FileNotFoundError(f"none of {names} in {FONTS}")


SANS = _first(["segoeui.ttf", "arial.ttf"])
SANS_B = _first(["segoeuib.ttf", "arialbd.ttf"])
MONO = _first(["consola.ttf", "lucon.ttf", "cour.ttf"])
MONO_B = _first(["consolab.ttf", "consola.ttf", "lucon.ttf", "cour.ttf"])


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def canvas(transparent: bool = False, size: tuple[int, int] = (W, H)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA" if transparent else "RGB", size, (0, 0, 0, 0) if transparent else BG)
    return img, ImageDraw.Draw(img)


def rrect(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, ...], outline: tuple[int, ...] | None = None, r: int = 22, w: int = 3) -> None:
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=w if outline else 0)


def text(d: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, f: ImageFont.FreeTypeFont, fill: tuple[int, ...] = FG, anchor: str = "la") -> None:
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def wrap(d: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    out: list[str] = []
    for para in s.split("\n"):
        line = ""
        for word in para.split():
            trial = (line + " " + word).strip()
            if d.textlength(trial, font=f) <= width:
                line = trial
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def paragraph(d: ImageDraw.ImageDraw, x: int, y: int, s: str, f: ImageFont.FreeTypeFont, width: int, fill: tuple[int, ...] = FG, gap: int = 10) -> int:
    for line in wrap(d, s, f, width):
        d.text((x, y), line, font=f, fill=fill)
        y += f.size + gap
    return y


def arrow(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], fill: tuple[int, ...] = DIM, w: int = 6) -> None:
    d.line([a, b], fill=fill, width=w)
    ax, ay = a
    bx, by = b
    import math

    ang = math.atan2(by - ay, bx - ax)
    for s in (-0.45, 0.45):
        d.line([b, (bx - 26 * math.cos(ang + s), by - 26 * math.sin(ang + s))], fill=fill, width=w)


def pill(d: ImageDraw.ImageDraw, x: int, y: int, s: str, f: ImageFont.FreeTypeFont, fill: tuple[int, ...], fg: tuple[int, ...] = BG, padx: int = 22, pady: int = 10) -> tuple[int, int]:
    tw = int(d.textlength(s, font=f))
    d.rounded_rectangle((x, y, x + tw + 2 * padx, y + f.size + 2 * pady), radius=(f.size + 2 * pady) // 2, fill=fill)
    d.text((x + padx, y + pady - 2), s, font=f, fill=fg)
    return x + tw + 2 * padx, y + f.size + 2 * pady
