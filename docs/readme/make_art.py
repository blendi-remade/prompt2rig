#!/usr/bin/env python3
"""The README's header and footer in Patch's brand (Lights Down), in Day and Evening for GitHub's light and dark modes.

  PATCH_BRAND_KIT=<kit folder> python docs/readme/make_art.py

A README is somewhere people read, so it is the world: black ink by day, white ink in the evening, the patch pink in
both. Drawn at 2x for sharp screens, on a transparent ground so it sits on GitHub's own page colour. Needs Patch's brand
kit (Patch Pixel, Funnel Sans and the logo masters), which isn't part of this repo: put it at ./brand-kit or point
PATCH_BRAND_KIT at it.
"""
import os
import re

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.environ.get('PATCH_BRAND_KIT', os.path.join(HERE, '..', '..', 'brand-kit'))
PINK = (245, 60, 209)
STYLES = {'day': ((0, 0, 0), (102, 102, 102)), 'evening': ((255, 255, 255), (153, 153, 153))}


def pixel(size):
    assert size % 9 == 0, 'Patch Pixel is set at multiples of 9 px'
    return ImageFont.truetype(os.path.join(KIT, '04 Type', 'Patch Pixel', 'PatchPixel-Regular.ttf'), size)


def funnel(size, weight=500):
    f = ImageFont.truetype(os.path.join(KIT, '04 Type', 'Funnel Sans', 'FunnelSans[wght].ttf'), size)
    f.set_variation_by_axes([weight])
    return f


def logo(name, ink, quiet):
    """A logo master's cells (10 units a cell, whole-cell rectangles), its ink swapped for this style's."""
    svg = open(os.path.join(KIT, '01 Logo', 'SVG', name + '.svg'), encoding='utf-8').read()
    rects = []
    for fill, d in re.findall(r'<path fill="#([0-9A-Fa-f]{6})"[^>]* d="([^"]+)"', svg):
        col = tuple(int(fill[i:i + 2], 16) for i in (0, 2, 4))
        col = PINK if col == PINK else quiet if col in ((153, 153, 153), (102, 102, 102)) else ink
        for x, y, w, h in re.findall(r'M(\d+) (\d+)h(\d+)v(\d+)h-\d+z', d):
            rects.append((int(x) // 10, int(y) // 10, int(w) // 10, int(h) // 10, col))
    return rects


def draw_rects(d, rects, x, y, c):
    for rx, ry, rw, rh, col in rects:
        d.rectangle([x + rx * c, y + ry * c, x + (rx + rw) * c - 1, y + (ry + rh) * c - 1], fill=col)


def header(style):
    ink, quiet = STYLES[style]
    img = Image.new('RGBA', (1600, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_rects(d, logo('patch-lockup-white', ink, quiet), 0, 0, 6)                 # 3 px cells on screen
    d.text((0, 254), 'prompt2rig.', font=pixel(144), fill=ink, anchor='ls')         # 8 px cells: the headline size
    d.text((0, 352), 'Type a motion, and any rigged character performs it.', font=funnel(52), fill=quiet, anchor='ls')
    img.save(os.path.join(HERE, f'header-{style}.png'), optimize=True)


def footer(style):
    ink, quiet = STYLES[style]
    img = Image.new('RGBA', (1000, 72), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    draw_rects(d, logo('patch-address-white', ink, quiet), 0, 0, 8)                # 4 px cells on screen
    d.text((460, 44), 'Mashups, mods and browser games.', font=funnel(30, 400), fill=quiet, anchor='ls')
    img.save(os.path.join(HERE, f'footer-{style}.png'), optimize=True)


if __name__ == '__main__':
    for s in STYLES:
        header(s)
        footer(s)
    print('wrote', sorted(f for f in os.listdir(HERE) if f.endswith('.png')))
