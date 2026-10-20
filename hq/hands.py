"""Hand / finger tracking for the HQ pass.

Runs a hand model on the full-resolution source clip, assigns hands to the subject's wrists from
GVHMR's body keypoints (model handedness flips; the body doesn't), rotates keypoints from camera
into GVHMR's world frame, resamples to the motion frame rate, fills short gaps and smooths.

Output per hand ("R" / "L", the subject's own): 21 wrist-relative keypoints per frame in world axes
(Y up), plus a validity mask. Joint order: wrist, thumb 1-4, index 5-8, middle 9-12, ring 13-16,
pinky 17-20 (MediaPipe / OpenPose order — WiLoR uses the same).

Backends: "wilor" (WiLoR, needs MANO_RIGHT.pkl) and "mediapipe" (MediaPipe Hand Landmarker).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

HQ = Path(__file__).resolve().parent
WEIGHTS = HQ / "weights"
SIDES = ("R", "L")
MAX_GAP_S = 0.4      # interpolate missing stretches up to this long
SMOOTH_SIGMA = 1.2   # temporal Gaussian (frames at 30fps)


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return frames, fps


# ---- backends: frame list → per-frame list of (kp2d (21,2) px, kp3d (21,3) cam-axes meters, score) ----
class MediaPipeHands:
    name = "mediapipe"

    def __init__(self):
        import mediapipe as mp

        self.mp = mp
        self.opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(WEIGHTS / "hand_landmarker.task")),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3)

    def __call__(self, frames, fps):
        det = self.mp.tasks.vision.HandLandmarker.create_from_options(self.opts)  # fresh tracker per clip
        h, w = frames[0].shape[:2]
        out = []
        for t, f in enumerate(frames):
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            res = det.detect_for_video(img, int(t * 1000 / fps))
            out.append([(np.array([(p.x * w, p.y * h) for p in lm], np.float32),
                         np.array([(p.x, p.y, p.z) for p in wl], np.float32), hd[0].score)
                        for lm, wl, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness)])
        det.close()
        return out


class WiLoRHands:
    name = "wilor"

    def __init__(self):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

        root = WEIGHTS / "wilor"
        if not (root / "pretrained_models" / "MANO_RIGHT.pkl").exists():
            raise FileNotFoundError(f"WiLoR needs MANO_RIGHT.pkl in {root / 'pretrained_models'} (mano.is.tue.mpg.de)")
        # Every file is present locally, so the pipeline never reaches for its mirror repo.
        self.pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16, verbose=False,
                                                      wilor_pretrained_dir=str(root), WILOR_MINI_REPO_ID="__offline__")

    def __call__(self, frames, fps, det_batch=16, model_batch=32):
        """Batched WiLoR: same weights/maths as WiLoR-mini's predict(), ~3x faster than per-frame calls."""
        import torch
        from wilor_mini.utils import utils as wu

        P = self.pipe
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        H, W = rgb[0].shape[:2]
        # 1) hand boxes, batched and padded to a fixed batch shape
        dets = []
        for i in range(0, len(rgb), det_batch):
            chunk = rgb[i:i + det_batch]
            res = P.hand_detector(chunk + [chunk[-1]] * (det_batch - len(chunk)), conf=0.3, verbose=False, half=True)
            for r in res[:len(chunk)]:
                b = r.boxes
                dets.append(list(zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy())))
        # 2) 256px crops (left hands mirrored, as the model expects right hands)
        patches, meta = [], []
        for t, ds in enumerate(dets):
            for box, cls, conf in sorted(ds, key=lambda d: -d[2])[:4]:
                c = (box[:2] + box[2:]) / 2.0
                size = 2.5 * float((box[2:] - box[:2]).max())
                right = int(cls > 0.5)
                patch, _ = wu.generate_image_patch_cv2(rgb[t], c[0], c[1], size, size, 256, 256, right == 0, 1.0, 0,
                                                      border_mode=cv2.BORDER_CONSTANT)
                patches.append(patch); meta.append((t, c, size, right, float(conf)))
        out = [[] for _ in frames]
        if not patches:
            return out
        # 3) model, batched
        cams, kps = [], []
        with torch.no_grad():
            for i in range(0, len(patches), model_batch):
                x = torch.from_numpy(np.stack(patches[i:i + model_batch])).to(P.device, dtype=P.dtype)
                o = P.wilor_model(x)
                cams.append(o["pred_cam"].float().cpu().numpy()); kps.append(o["pred_keypoints_3d"].float().cpu().numpy())
        cams, kps = np.concatenate(cams), np.concatenate(kps)
        # 4) un-mirror left hands, place in the full frame, project to 2D (for assigning hands to wrists)
        focal = P.FOCAL_LENGTH / P.IMAGE_SIZE * max(W, H)
        for j, (t, c, size, right, conf) in enumerate(meta):
            k3 = kps[j].copy(); pc = cams[j:j + 1].copy()
            pc[:, 1] *= (2 * right - 1)
            if right == 0:
                k3[:, 0] = -k3[:, 0]
            cam_t = wu.cam_crop_to_full(pc, c[None], size, np.array([[W, H]], np.float32), focal)
            k2 = wu.perspective_projection(k3[None], translation=cam_t, focal_length=np.array([[focal, focal]]),
                                           camera_center=np.array([[W / 2.0, H / 2.0]]))[0]
            out[t].append((k2.astype(np.float32), k3.astype(np.float32), conf))
        return out


_BACKENDS: dict = {}


def backend(name: str):
    if name not in _BACKENDS:
        _BACKENDS[name] = {"mediapipe": MediaPipeHands, "wilor": WiLoRHands}[name]()
    return _BACKENDS[name]


# ---- post-processing ----------------------------------------------------------------------------
def assign_to_wrists(dets, wrists):
    """dets: per-frame candidates; wrists (T,2,2) px [right, left]. → {side: (kp3d (T,21,3), valid (T,))}."""
    T = len(dets)
    kp = {s: np.zeros((T, 21, 3), np.float32) for s in SIDES}
    valid = {s: np.zeros(T, bool) for s in SIDES}
    for t, cands in enumerate(dets):
        cands = sorted(cands, key=lambda c: -c[2])[:2]
        if not cands:
            continue
        dist = lambda c, i: float(np.linalg.norm(c[0][0] - wrists[t, i]))  # noqa: E731
        if len(cands) == 2:
            a, b = cands
            pairs = [(SIDES[0], a), (SIDES[1], b)] if dist(a, 0) + dist(b, 1) <= dist(a, 1) + dist(b, 0) \
                else [(SIDES[0], b), (SIDES[1], a)]
        else:
            c = cands[0]
            pairs = [(SIDES[0] if dist(c, 0) <= dist(c, 1) else SIDES[1], c)]
        # Reject detections far from the wrist they were assigned to (another person, a stray blob).
        forearm = np.linalg.norm(wrists[t, 0] - wrists[t, 1]) * 0.6 + 40
        for side, (k2, k3, _) in pairs:
            if np.linalg.norm(k2[0] - wrists[t, SIDES.index(side)]) < forearm:
                kp[side][t], valid[side][t] = k3, True
    return kp, valid


def resample(kp, valid, src_fps, n, fps):
    """Linear resample (T,21,3) from src_fps to n frames at fps; gaps longer than MAX_GAP_S stay invalid."""
    T = len(kp)
    t_src = np.arange(T) / src_fps
    t_dst = np.arange(n) / fps
    out = np.zeros((n, 21, 3), np.float32)
    ok = np.zeros(n, bool)
    vi = np.flatnonzero(valid)
    if len(vi) < 2:
        return out, ok
    flat = kp[vi].reshape(len(vi), -1)
    for j in range(flat.shape[1]):
        out.reshape(n, -1)[:, j] = np.interp(t_dst, t_src[vi], flat[:, j])
    # valid where the nearest valid source samples on both sides are close enough
    nxt = np.searchsorted(t_src[vi], t_dst)
    prev_t = t_src[vi][np.clip(nxt - 1, 0, len(vi) - 1)]
    next_t = t_src[vi][np.clip(nxt, 0, len(vi) - 1)]
    exact = np.isin(np.round(t_dst * src_fps).astype(int), vi)
    ok = exact | ((next_t - prev_t) <= MAX_GAP_S) & (t_dst >= t_src[vi][0]) & (t_dst <= t_src[vi][-1])
    return out, ok


def smooth(kp, ok, sigma=SMOOTH_SIGMA):
    """Gaussian smoothing within each valid run (no blending across gaps)."""
    if sigma <= 0:
        return kp
    r = int(3 * sigma)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    out = kp.copy()
    n = len(kp)
    i = 0
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        seg = kp[i:j]
        for t in range(len(seg)):
            lo, hi = max(0, t - r), min(len(seg), t + r + 1)
            ww = w[lo - t + r:hi - t + r]
            out[i + t] = (seg[lo:hi] * ww[:, None, None]).sum(0) / ww.sum()
        i = j
    return out


def aa_to_mat(aa):
    """(N,3) axis-angle → (N,3,3)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True) + 1e-9
    k = aa / th
    K = np.zeros((len(aa), 3, 3), np.float32)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3, dtype=np.float32)[None] + s * K + (1 - c) * (K @ K)


def detect(source_video: Path, method: str):
    """Run the hand model over the full-resolution source clip (independent of GVHMR — run it in parallel)."""
    frames, src_fps = read_frames(source_video)
    return backend(method)(frames, src_fps), src_fps, frames[0].shape[:2]


def finish(dets, src_fps, hw, vitpose_px, work_long, go_world, go_incam, n: int, fps: float):
    """Assign to wrists, rotate camera → world, resample to the motion frames, smooth."""
    T = len(dets)
    s = max(hw) / float(work_long)
    Tv = len(vitpose_px)
    idx = np.clip(np.round(np.arange(T) / src_fps * fps).astype(int), 0, Tv - 1)
    wrists = np.stack([vitpose_px[idx, 10, :2], vitpose_px[idx, 9, :2]], 1) * s   # [right, left] px
    kp_cam, valid = assign_to_wrists(dets, wrists)

    # camera → GVHMR world, per motion frame: R_wc = R_world(pelvis) · R_cam(pelvis)^T
    R_wc = aa_to_mat(np.asarray(go_world, np.float32)) @ np.transpose(aa_to_mat(np.asarray(go_incam, np.float32)), (0, 2, 1))
    out = {}
    for side in SIDES:
        k, ok = resample(kp_cam[side], valid[side], src_fps, n, fps)
        k = k - k[:, :1]                                       # wrist-relative
        k = np.einsum("tij,tkj->tki", R_wc[:n], k)              # into world axes
        k = smooth(k, ok)
        out[side] = {"kp": k.reshape(-1).round(4).tolist(), "valid": ok.astype(int).tolist(),
                     "coverage": round(float(ok.mean()), 3)}
    return out
