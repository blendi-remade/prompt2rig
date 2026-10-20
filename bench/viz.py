"""Side-by-side hand crops with each method's 2D skeleton drawn on top.

    python bench/viz.py bench/clips/h1.mp4 mediapipe wilor [--frames 20,50,80,110] [--out x.png]

One row per (frame, hand); columns: plain crop, then one per method.
"""

import argparse
import os

import cv2
import numpy as np

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
FINGER_COLORS = [(255, 255, 255), (0, 200, 255), (0, 255, 0), (255, 200, 0), (255, 0, 255), (0, 0, 255)]


def color(a, b):
    j = max(a, b)
    return FINGER_COLORS[0 if j == 0 else (j - 1) // 4 + 1]


def draw(img, kp, scale, off):
    for a, b in BONES:
        pa = tuple(int(v) for v in (kp[a] - off) * scale); pb = tuple(int(v) for v in (kp[b] - off) * scale)
        cv2.line(img, pa, pb, color(a, b), 2, cv2.LINE_AA)
    for p in kp:
        cv2.circle(img, tuple(int(v) for v in (p - off) * scale), 3, (20, 20, 20), -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("methods", nargs="+")
    ap.add_argument("--frames", default="20,50,80,110"); ap.add_argument("--out"); ap.add_argument("--size", type=int, default=220)
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.clip))[0]
    data = {m: np.load(os.path.join(os.path.dirname(__file__), "out", f"{name}_{m}.npz")) for m in a.methods}
    cap = cv2.VideoCapture(a.clip)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    S = a.size
    rows = []
    for t in [int(x) for x in a.frames.split(",")]:
        for h in ("R", "L"):
            pts = [data[m][f"{h}_kp2d"][t] for m in a.methods if data[m][f"{h}_valid"][t]]
            if not pts:
                continue
            allp = np.concatenate(pts)
            c = allp.mean(0); half = max(40.0, np.abs(allp - c).max() * 1.35)
            off = c - half; scale = S / (2 * half)
            x0, y0 = int(off[0]), int(off[1]); x1, y1 = int(c[0] + half), int(c[1] + half)
            src = cv2.copyMakeBorder(frames[t], 400, 400, 400, 400, cv2.BORDER_CONSTANT)
            crop = cv2.resize(src[y0 + 400:y1 + 400, x0 + 400:x1 + 400], (S, S))
            tiles = [crop.copy()]
            for m in a.methods:
                tile = crop.copy()
                if data[m][f"{h}_valid"][t]:
                    draw(tile, data[m][f"{h}_kp2d"][t], scale, off)
                else:
                    cv2.putText(tile, "missed", (8, S // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                tiles.append(tile)
            cv2.putText(tiles[0], f"{name} t{t} {h}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            rows.append(np.hstack(tiles))
    header = np.zeros((26, rows[0].shape[1], 3), np.uint8)
    for i, label in enumerate(["frame"] + a.methods):
        cv2.putText(header, label, (i * S + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    out = a.out or os.path.join(os.path.dirname(__file__), "out", f"{name}_viz.png")
    cv2.imwrite(out, np.vstack([header] + rows))
    print(out)


if __name__ == "__main__":
    main()
