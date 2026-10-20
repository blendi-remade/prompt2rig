"""Hand-pose bench: run a method over a clip and save per-frame keypoints for both hands.

    python bench/hands_bench.py --method mediapipe|wilor|wilor_body --clip bench/clips/h1.mp4

Saves bench/out/<clip>_<method>.npz with, per hand ("R" = subject's right, "L" = left):
  kp2d (T,21,2) image px · kp3d (T,21,3) hand-local meters · valid (T,) · score (T,)
and prints detection rate, 2D jitter and speed. Joint order (both methods): wrist, thumb 1-4,
index 5-8, middle 9-12, ring 13-16, pinky 17-20.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HANDS = ("R", "L")


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    cap.release()
    return frames, fps


def empty(T):
    return {h: dict(kp2d=np.zeros((T, 21, 2), np.float32), kp3d=np.zeros((T, 21, 3), np.float32),
                    valid=np.zeros(T, bool), score=np.zeros(T, np.float32)) for h in HANDS}


# ---- methods ---------------------------------------------------------------------------------
def run_mediapipe(frames, fps):
    import mediapipe as mp

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), "hand_landmarker.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)
    det = mp.tasks.vision.HandLandmarker.create_from_options(opts)
    T = len(frames); out = empty(T)
    h, w = frames[0].shape[:2]
    for t, f in enumerate(frames):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        res = det.detect_for_video(img, int(t * 1000 / fps))
        for lm, wl, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            # MediaPipe assumes a mirrored (selfie) image: swap labels for a normal camera.
            side = "L" if hd[0].category_name == "Right" else "R"
            if out[side]["valid"][t] and out[side]["score"][t] >= hd[0].score:
                continue
            out[side]["kp2d"][t] = [(p.x * w, p.y * h) for p in lm]
            out[side]["kp3d"][t] = [(p.x, p.y, p.z) for p in wl]
            out[side]["valid"][t] = True
            out[side]["score"][t] = hd[0].score
    return out


def wilor_pipe():
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    root = os.path.join(os.path.dirname(__file__), "..", "hq", "weights", "wilor")
    return WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16, verbose=False,
                                             wilor_pretrained_dir=root, WILOR_MINI_REPO_ID="__offline__")


def run_wilor(frames, fps):
    pipe = wilor_pipe()
    T = len(frames); out = empty(T)
    for t, f in enumerate(frames):
        rets = pipe.predict(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        for r in rets:
            if "wilor_preds" not in r:
                continue
            side = "R" if r["is_right"] > 0.5 else "L"
            p = r["wilor_preds"]
            out[side]["kp2d"][t] = p["pred_keypoints_2d"][0]
            out[side]["kp3d"][t] = p["pred_keypoints_3d"][0]
            out[side]["valid"][t] = True
            out[side]["score"][t] = 1.0
    return out


METHODS = {"mediapipe": run_mediapipe, "wilor": run_wilor}


# ---- left/right from the body: models' own handedness flips; body wrists don't ---------------
def body_wrists(clip, frames):
    """Per-frame (T,2,2) image px of the subject's [right, left] wrist from GVHMR's ViTPose cache."""
    import torch

    keys = dict(l.split()[:2] for l in open(os.path.join(os.path.dirname(clip), "keys.txt")))
    key = keys[os.path.splitext(os.path.basename(clip))[0]]
    kp = torch.load(os.path.join(os.path.dirname(__file__), "..", "hq", "cache", key, "preprocess", "vitpose.pt"), weights_only=False)
    kp = np.asarray(kp)[..., :2]                     # (Tv,17,2) on the 854px / 30fps working copy
    h, w = frames[0].shape[:2]
    s = max(h, w) / 854.0
    Tv, T = len(kp), len(frames)
    idx = np.clip(np.round(np.arange(T) * (Tv - 1) / max(1, T - 1)).astype(int), 0, Tv - 1)
    return np.stack([kp[idx, 10], kp[idx, 9]], 1) * s  # COCO: 10 = right wrist, 9 = left wrist


def relabel(out, wrists):
    T = len(wrists)
    hands = [(out[h]["kp2d"][t], out[h]["kp3d"][t], out[h]["score"][t]) if out[h]["valid"][t] else None
             for t in range(T) for h in HANDS]
    new = empty(T)
    for t in range(T):
        cands = [c for c in hands[2 * t:2 * t + 2] if c is not None]
        if not cands:
            continue
        d = lambda c, i: np.linalg.norm(c[0][0] - wrists[t, i])  # noqa: E731
        if len(cands) == 2:
            a, b = cands
            pairs = [(a, b)] if d(a, 0) + d(b, 1) <= d(a, 1) + d(b, 0) else [(b, a)]
            assign = list(zip(HANDS, pairs[0]))
        else:
            c = cands[0]
            assign = [(HANDS[0] if d(c, 0) <= d(c, 1) else HANDS[1], c)]
        for side, (k2, k3, sc) in assign:
            new[side]["kp2d"][t], new[side]["kp3d"][t], new[side]["score"][t] = k2, k3, sc
            new[side]["valid"][t] = True
    return new


# ---- metrics -----------------------------------------------------------------------------------
def metrics(out):
    rows = []
    for h in HANDS:
        v = out[h]["valid"]; k = out[h]["kp2d"]
        size = np.linalg.norm(k[:, 9] - k[:, 0], axis=-1) + 1e-6           # wrist → middle MCP
        ok = v[2:] & v[1:-1] & v[:-2]
        acc = np.linalg.norm(k[2:] - 2 * k[1:-1] + k[:-2], axis=-1).mean(-1) / size[1:-1]
        jit = float(np.median(acc[ok])) if ok.any() else float("nan")
        rows.append(f"{h}: detected {v.mean() * 100:5.1f}%  jitter {jit:.3f}")
    return "  |  ".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--clip", required=True)
    a = ap.parse_args()
    frames, fps = read_frames(a.clip)
    t0 = time.perf_counter()
    out = METHODS[a.method](frames, fps)
    dt = time.perf_counter() - t0
    try:
        out = relabel(out, body_wrists(a.clip, frames))
    except Exception as e:  # noqa: BLE001 — no body track: keep the model's own labels
        print("relabel skipped:", e)
    os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
    name = os.path.splitext(os.path.basename(a.clip))[0]
    np.savez_compressed(os.path.join(os.path.dirname(__file__), "out", f"{name}_{a.method}.npz"),
                        **{f"{h}_{k}": v for h in HANDS for k, v in out[h].items()}, fps=fps)
    print(f"{name} {a.method:10s} {len(frames)} frames in {dt:5.1f}s ({len(frames) / dt:5.1f} fps)  {metrics(out)}")


if __name__ == "__main__":
    sys.exit(main())
