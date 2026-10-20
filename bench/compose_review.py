"""Grid: source hand crop | character close-up (one column per method), per time and hand.

    python bench/compose_review.py --clip bench/clips/h3.mp4 --name h3 --times 0.83,2.08 --methods mediapipe,wilor
"""

import argparse
import os

import cv2
import numpy as np

HERE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--times", required=True); ap.add_argument("--methods", default="mediapipe")
    ap.add_argument("--kp", default="mediapipe", help="bench npz used to locate the hand in the source")
    ap.add_argument("--size", type=int, default=300)
    a = ap.parse_args()
    S = a.size
    cap = cv2.VideoCapture(a.clip); fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    kp = np.load(os.path.join(HERE, "out", f"{a.name}_{a.kp}.npz"))
    methods = a.methods.split(",")
    rows = []
    for t in [float(x) for x in a.times.split(",")]:
        fi = min(len(frames) - 1, int(round(t * fps)))
        for side, h in (("Right", "R"), ("Left", "L")):
            if kp[f"{h}_valid"][fi]:
                p = kp[f"{h}_kp2d"][fi]; c = p.mean(0); half = max(60, np.abs(p - c).max() * 1.5)
            else:
                continue
            src = cv2.copyMakeBorder(frames[fi], 600, 600, 600, 600, cv2.BORDER_CONSTANT)
            x0, y0, x1, y1 = (int(c[0] - half) + 600, int(c[1] - half) + 600, int(c[0] + half) + 600, int(c[1] + half) + 600)
            tiles = [cv2.resize(src[y0:y1, x0:x1], (S, S))]
            cv2.putText(tiles[0], f"{a.name} {t:.2f}s {side}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            for m in methods:
                path = os.path.join(HERE, "out", "review", f"{a.name}_{m}_{t:.2f}_{side}.png")
                img = cv2.imread(path)
                if img is None:
                    tiles.append(np.zeros((S, S, 3), np.uint8)); continue
                hh, ww = img.shape[:2]; r = int(min(hh, ww) * 0.42)
                cy, cx = hh // 2, ww // 2
                tiles.append(cv2.resize(img[cy - r:cy + r, cx - r:cx + r], (S, S)))
                cv2.putText(tiles[-1], m, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            rows.append(np.hstack(tiles))
    out = os.path.join(HERE, "out", f"{a.name}_review.png")
    cv2.imwrite(out, np.vstack(rows))
    print(out)


if __name__ == "__main__":
    main()
