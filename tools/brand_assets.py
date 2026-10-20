"""Render the README's branded media (universal-modder style) into docs/media/.

    python tools/brand_assets.py [--frames frames.json]

- banner.png: wordmark, "prompt2rig." in Geist Pixel, the one-line pitch, and a grid of real captures
- badges/*.png: pixel badges (notched void pill, magenta bullet, ash label, bone value) like universal-modder's
Fonts come from public/brand/fonts (Geist Pixel, Instrument Sans, Sometype Mono — OFL).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "public" / "brand"
FONTS = BRAND / "fonts"
OUT = ROOT / "docs" / "media"

VOID, COAL, COAL2, RULE = (14, 12, 16), (28, 25, 32), (37, 33, 42), (43, 38, 48)
ASH, ASH2, BONE, MISSING = (142, 135, 150), (93, 87, 101), (242, 236, 226), (255, 60, 210)
GEIST_EM = 1000 / 38  # Geist Pixel draws on a 38/1000 em grid: whole multiples of this stay pixel-crisp


def font(name: str, size: float, weight: int | None = None) -> ImageFont.FreeTypeFont:
    path = FONTS / f"{name}.woff2"
    ttf = FONTS / f"{name}.ttf"
    if not ttf.exists():  # Pillow can't read woff2: decompress once next to it (ignored by git)
        from fontTools.ttLib import TTFont

        f = TTFont(str(path)); f.flavor = None; f.save(str(ttf))
    f = ImageFont.truetype(str(ttf), round(size))
    if weight:
        try:
            axes = f.get_variation_axes()
            f.set_variation_by_axes([weight if (a["name"] if isinstance(a["name"], str) else a["name"].decode()).lower().startswith("wei") else a["default"] for a in axes])
        except Exception:
            pass
    return f


def svg_rects(path: Path, scale: int) -> Image.Image:
    """Rasterise the brand's crispEdges SVGs (unions of whole-unit rects) at an integer scale."""
    svg = path.read_text(encoding="utf-8")
    vb = [int(float(v)) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]
    img = Image.new("RGBA", (vb[2] * scale, vb[3] * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for fill, data in re.findall(r'<path fill="(#[0-9A-Fa-f]{6})" d="([^"]+)"', svg):
        rgb = tuple(int(fill[i:i + 2], 16) for i in (1, 3, 5))
        for x, y, w, h in re.findall(r"M(-?\d+) (-?\d+)h(-?\d+)v(-?\d+)h-?\d+z", data):
            x, y, w, h = map(int, (x, y, w, h))
            d.rectangle([x * scale, y * scale, (x + w) * scale - 1, (y + h) * scale - 1], fill=rgb)
    return img


def notched(w: int, h: int, n: int, color) -> Image.Image:
    """A pill with the brand's 1-unit notched corners."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([n, 0, w - n - 1, h - 1], fill=color)
    d.rectangle([0, n, w - 1, h - n - 1], fill=color)
    return img


def pixel_text(draw: ImageDraw.ImageDraw, xy, text, fnt, fill):
    draw.fontmode = "1"   # Geist Pixel without anti-aliasing
    draw.text(xy, text, font=fnt, fill=fill)
    draw.fontmode = "L"


# ---- badges ------------------------------------------------------------------------------------
def badge(label: str, value: str, icon: Image.Image | None = None) -> Image.Image:
    S = 2                                   # 2 px per brand unit (26-px-tall badge units → 52 px)
    mono = font("SometypeMono", 22, 400)
    h = 26 * S
    lw = mono.getlength(label + " "); vw = mono.getlength(value)
    lead = (icon.width + 10) if icon else 22
    w = int(14 + lead + lw + vw + 16)
    img = notched(w, h, S, VOID + (255,))
    d = ImageDraw.Draw(img)
    if icon:
        img.alpha_composite(icon, (12, (h - icon.height) // 2))
    else:
        d.rectangle([14, h // 2 - 4, 21, h // 2 + 3], fill=MISSING)
    x = 14 + lead
    d.text((x, h // 2), label, font=mono, fill=ASH, anchor="lm")
    d.text((x + lw, h // 2), value, font=mono, fill=BONE, anchor="lm")
    return img


X_ICON = ["10000001", "01000010", "00100100", "00011000", "00011000", "00100100", "01000010", "10000001"]


def pixel_icon(rows: list[str], unit: int, color) -> Image.Image:
    img = Image.new("RGBA", (len(rows[0]) * unit, len(rows) * unit), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c == "1":
                d.rectangle([x * unit, y * unit, (x + 1) * unit - 1, (y + 1) * unit - 1], fill=color)
    return img


def button(prefix: str, main: str, icon: Image.Image, unit: int = 4) -> Image.Image:
    """Pixel button matching the brand's 'join the discord' button: void pill 16 units tall with stepped corners,
    icon on the left, then the text in two tones ('prefix' ash, 'main' bone) in Geist Pixel.
    Rendered at `unit` px per brand unit; show it at height 32 (that's 2x for sharp retina display)."""
    h = 16 * unit
    title = font("GeistPixel", GEIST_EM * unit / 2)          # whole Geist cells: 2 px each at unit 4
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    pw = probe.textlength(prefix + " ", font=title)
    mw = probe.textlength(main, font=title)
    x0 = 3 * unit + icon.width + 3 * unit
    w = int(x0 + pw + mw + 4 * unit)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([unit, 0, w - unit - 1, h - 1], fill=VOID)        # stepped 1-unit corners
    d.rectangle([0, unit, w - 1, h - unit - 1], fill=VOID)
    img.alpha_composite(icon, (3 * unit, (h - icon.height) // 2))
    d.fontmode = "1"
    d.text((x0, h // 2), prefix, font=title, fill=ASH, anchor="lm")
    d.text((x0 + pw, h // 2), main, font=title, fill=BONE, anchor="lm")
    d.fontmode = "L"
    return img


def discord_icon(scale=2) -> Image.Image:
    full = svg_rects(BRAND / "discord.svg", scale)
    return full.crop((3 * scale, 3 * scale, 15 * scale, 12 * scale))


# ---- banner ------------------------------------------------------------------------------------
def banner(frames: list[dict]) -> Image.Image:
    W, H = 1280, 640
    img = Image.new("RGBA", (W, H), (19, 17, 22, 255))
    d = ImageDraw.Draw(img)
    # brand row: the universal-modder logo (mark + wordmark)
    logo = svg_rects(BRAND / "logo.svg", 3)
    img.alpha_composite(logo, (70, 70))
    # title
    title = font("GeistPixel", GEIST_EM * 3)
    pixel_text(d, (66, 176), "prompt2rig", title, BONE)
    tw = d.textlength("prompt2rig", font=title)
    d.rectangle([66 + tw + 6, 176 + 58, 66 + tw + 6 + 11, 176 + 69], fill=MISSING)   # the brand's magenta period
    # pitch (Instrument Sans, ash), wrapped to the left column
    sans = font("InstrumentSans", 21, 400)
    words = ("type a motion. a video model acts it out, your gpu lifts the 3d motion, fingers included, "
             "and any mixamo character performs it.").split()
    lines, cur = [], ""
    for wd in words:
        if d.textlength((cur + " " + wd).strip(), font=sans) > 470:
            lines.append(cur); cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines.append(cur)
    for i, ln in enumerate(lines):
        d.text((70, 300 + i * 32), ln, font=sans, fill=ASH)
    # footer lines (mono)
    mono = font("SometypeMono", 15, 400); monob = font("SometypeMono", 15, 700)
    d.text((70, 505), "apache-2.0", font=monob, fill=BONE)
    d.text((70 + d.textlength("apache-2.0  ", font=monob), 505), "github.com/blendi-remade/prompt2rig", font=mono, fill=ASH)
    d.text((70, 546), "a universal-modder tool · universalmodder.com", font=mono, fill=ASH)
    # grid of real captures, 2 x 3, with mono labels like the universal-modder banner
    x0, y0, tw_, th_, gap = 602, 52, 310, 174, 6
    lab = font("SometypeMono", 13, 400)
    for i, fr in enumerate(frames[:6]):
        cx, cy = x0 + (i % 2) * (tw_ + gap), y0 + (i // 2) * (th_ + gap + 1)
        tile = Image.open(fr["path"]).convert("RGB").resize((tw_, th_), Image.LANCZOS)
        img.paste(tile, (cx, cy))
        tlw = d.textlength(fr["label"], font=lab)
        d.rectangle([cx + 8, cy + th_ - 28, cx + 8 + tlw + 12, cy + th_ - 8], fill=VOID)
        d.text((cx + 14, cy + th_ - 18), fr["label"], font=lab, fill=BONE, anchor="lm")
    return img


def grab(video: str, t: float, out: Path) -> Path:
    # captures are 2560x1440 app recordings: crop the stage (video panel + character), 16:9
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", video, "-frames:v", "1",
                    "-vf", "crop=2220:1249:340:0", str(out)], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "tools" / "banner_frames.json"))
    a = ap.parse_args()
    (OUT / "badges").mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_frames"; tmp.mkdir(exist_ok=True)
    spec = json.loads(Path(a.frames).read_text(encoding="utf-8"))
    frames = [{"label": f["label"], "path": grab(str(ROOT / f["video"]), f["t"], tmp / f"{i}.png")} for i, f in enumerate(spec)]
    banner(frames).convert("RGB").save(OUT / "banner.png", optimize=True)
    for name, (label, value, icon) in {
        "video": ("video", "flux 3 max · h3 max turbo", None),
        "motion": ("motion", "gvhmr · wilor fingers", None),
        "license": ("license", "apache-2.0", None),
    }.items():
        badge(label, value, icon).save(OUT / "badges" / f"{name}.png", optimize=True)
    button("visit", "universalmodder.com", svg_rects(BRAND / "mark.svg", 4)).save(OUT / "button-website.png", optimize=True)
    button("follow", "@universalmodhq", pixel_icon(X_ICON, 4, BONE)).save(OUT / "button-x.png", optimize=True)
    svg_rects(BRAND / "discord.svg", 4).save(OUT / "button-discord.png", optimize=True)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
