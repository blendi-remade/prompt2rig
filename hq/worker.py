"""Prompt2Rig HQ worker — keeps GVHMR warm on the GPU and turns a clip URL into SMPL motion.

POST /process {"video_url": "..."} → {n, fps, global_orient, body_pose, transl, leg_length, timings}
GET  /health                        → {ok, device, error?}

Run:  hq/.venv/Scripts/python hq/worker.py      (listens on 127.0.0.1:5178; setup in README)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path

HQ = Path(__file__).resolve().parent
os.environ.setdefault("GVHMR_CHECKPOINTS", str(HQ / "weights"))
os.environ.setdefault("GVHMR_BODY_MODELS", str(HQ / "body_models"))
CACHE = HQ / "cache"
SMPLX = Path(os.environ["GVHMR_BODY_MODELS"]) / "smplx" / "SMPLX_NEUTRAL.npz"
DETECTOR = os.environ.get("HQ_DETECTOR", "fast")
TRACKER = os.environ.get("HQ_TRACKER", "bytetrack.yaml")
POSE2D = os.environ.get("HQ_POSE2D", "vitpose")
# ViTPose flip-test runs every crop twice (mirrored) for a small accuracy gain.
POSE_FLIP = os.environ.get("HQ_POSE_FLIP", "0") == "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:  # let onnxruntime-gpu (RTMPose) use the CUDA/cuDNN DLLs that ship with torch
    import onnxruntime

    onnxruntime.preload_dlls()
except Exception:
    pass

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI()
state: dict = {"pipe": None, "error": None, "device": None}
# All GPU work runs on ONE thread: CUDA/cuBLAS/cuDNN handles are per-thread, so a fresh request
# thread would otherwise pay their setup (seconds) and miss the warm-up.
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")


def log(msg: str) -> None:
    print(f"[hq {time.strftime('%H:%M:%S')}] {msg}", flush=True)


FRAME_CACHE: dict[str, list] = {}


def fast_read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """Drop-in for gvhmr's PyAV reader: reuse the detector's decoded frames, resize with OpenCV.

    Same output contract (N, H, W, 3) RGB uint8; bicubic like ffmpeg's default scale filter.
    """
    import cv2

    frames = FRAME_CACHE.get(str(Path(video_path).resolve()))
    if frames is None:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
    frames = frames[start_frame:(None if end_frame == -1 else end_frame)]
    if scale != 1.0:
        h, w = frames[0].shape[:2]
        size = (int(w * scale), int(h * scale))
        frames = [cv2.resize(f, size, interpolation=cv2.INTER_CUBIC) for f in frames]
    return np.ascontiguousarray(np.stack(frames)[..., ::-1])


class FastSingleDetector:
    """Batched YOLO detection keeping the largest person per frame — replaces per-frame tracking.

    Generated clips contain exactly one person, so identity tracking buys nothing; batching the
    detector is ~5x faster than ultralytics' frame-by-frame `track()`.
    """

    _inst = None

    @classmethod
    def get(cls):
        cls._inst = cls._inst or cls()
        return cls._inst

    def __init__(self, ckpt: Path = HQ / "weights" / "yolo11m.pt", batch: int = 32):
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset

        ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.yolo, self.batch = YOLO(attempt_download_asset(str(ckpt))), batch

    def get_one_track(self, video_path) -> torch.Tensor:
        import cv2
        from gvhmr.utils.net_utils import moving_average_smooth

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        FRAME_CACHE.clear()
        FRAME_CACHE[str(Path(video_path).resolve())] = frames
        boxes = np.zeros((len(frames), 4), np.float32)
        found = np.zeros(len(frames), bool)
        for i in range(0, len(frames), self.batch):
            chunk = frames[i:i + self.batch]
            # Pad to a fixed batch size: every new batch shape costs a slow first CUDA call.
            pad = [chunk[-1]] * (self.batch - len(chunk))
            results = self.yolo.predict(chunk + pad, classes=[0], conf=0.3, half=True, device=0, verbose=False)[:len(chunk)]
            for j, r in enumerate(results):
                if len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    boxes[i + j], found[i + j] = xyxy[area.argmax()], True
        if not found.any():
            raise RuntimeError("no person detected in the clip")
        idx = np.arange(len(frames))
        for k in range(4):  # fill frames with no detection
            boxes[:, k] = np.interp(idx, idx[found], boxes[found, k])
        return moving_average_smooth(torch.from_numpy(boxes), window_size=5, dim=0)


def load_pipeline() -> None:
    if not SMPLX.exists():
        state["error"] = f"SMPL-X missing: put SMPLX_NEUTRAL.npz in {SMPLX.parent}"
        log(state["error"])
        return
    try:
        import gvhmr
        import gvhmr.cli.demo as demo

        if DETECTOR == "fast":
            demo.make_detector = lambda *a, **kw: FastSingleDetector.get()  # cfg still names a valid YOLO preset
            import gvhmr.utils.imgcrop as imgcrop

            imgcrop.read_video_np = fast_read_video_np

        t0 = time.perf_counter()
        state["pipe"] = gvhmr.pipeline("human-motion-recovery", device="cuda")
        state["device"] = torch.cuda.get_device_name(0)
        state["error"] = None
        log(f"GVHMR loaded on {state['device']} in {time.perf_counter() - t0:.1f}s")
        warmup()
    except Exception as e:  # noqa: BLE001
        state["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


def warmup() -> None:
    """Run one tiny clip so CUDA kernels / ONNX sessions are initialized before the first real request."""
    clip = CACHE / "_warmup.mp4"
    if not clip.exists():
        return
    try:
        t0 = time.perf_counter()
        shutil.rmtree(CACHE / "_warmup", ignore_errors=True)  # cached stages would skip the warm-up
        result, _ = recover(clip, CACHE / "_warmup")
        leg_length(result.smpl_params_world)
        if DETECTOR == "fast":  # first YOLO call per input shape is slow: cover both orientations
            det = FastSingleDetector.get()
            for h, w in ((480, 832), (832, 480), (470, 854), (854, 470)):  # H3 480P, FLUX 3 720p (downscaled)
                for n in (det.batch,):
                    det.yolo.predict([np.zeros((h, w, 3), np.uint8)] * n, classes=[0], half=True, device=0, verbose=False)
        log(f"warmup done in {time.perf_counter() - t0:.1f}s")
    except Exception as e:  # noqa: BLE001
        log(f"warmup failed: {e}")


def recover(video: Path, out_dir: Path):
    from motion2mixamorig.extract import _stabilize_pose_cache

    phases: dict[str, float] = {}
    last = {"name": None, "t": time.perf_counter()}

    def on_progress(_frac: float, desc: str) -> None:
        now = time.perf_counter()
        if desc != last["name"]:
            if last["name"]:
                phases[last["name"]] = phases.get(last["name"], 0) + now - last["t"]
            last.update(name=desc, t=now)

    # Generated clips are single-person, plain background, static camera: a mid-size YOLO with ByteTrack
    # (no camera-motion compensation) tracks as well as YOLOv8x+BoT-SORT at a fraction of the cost, and
    # ViTPose shares one decode+crop pass with the HMR2 feature backbone.
    kwargs = dict(static_camera=True, detector="yolo11m" if DETECTOR == "fast" else DETECTOR, pose2d=POSE2D, set_overrides=[f"++detector.tracker={TRACKER}"] + ([] if POSE_FLIP else ["++pose2d.flip_test=false"]),
                  output_dir=str(out_dir), output_root=str(out_dir.parent), progress=False, progress_callback=on_progress)
    t0 = time.perf_counter()
    result = state["pipe"](str(video), **kwargs)
    t1 = time.perf_counter()
    if last["name"]:
        phases[last["name"]] = phases.get(last["name"], 0) + t1 - last["t"]
    log("phases: " + "  ".join(f"{k} {v:.2f}s" for k, v in phases.items()))
    # RTMPose can swap left/right per frame when the subject turns; fix the 2D cache and redo the
    # (cheap) 3D lift once if anything changed — same trick Motion2MixamoRig uses.
    pose_cache = out_dir / "preprocess" / "vitpose.pt"
    if pose_cache.exists() and _stabilize_pose_cache(pose_cache):
        (out_dir / "hmr4d_results.pt").unlink(missing_ok=True)
        result = state["pipe"](str(video), **kwargs)
    return result, {"recover": t1 - t0, "stabilize": time.perf_counter() - t1}


_BODY: dict = {}


def leg_length(params) -> float:
    """Mean hip→ankle length (m) of the recovered body; models cached across calls.

    SMPL-X → SMPL joint regression adapted from Motion2MixamoRig (Apache-2.0), see NOTICE.
    """
    from einops import einsum
    from gvhmr import PROJ_ROOT
    from gvhmr.utils.smplx_utils import make_smplx

    if not _BODY:
        dev = torch.device("cuda")
        _BODY["smplx"] = make_smplx("supermotion").to(dev)
        _BODY["x2s"] = torch.load(PROJ_ROOT / "gvhmr/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to(dev)
        _BODY["jreg"] = torch.load(PROJ_ROOT / "gvhmr/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=False).to(dev)
    p = {k: v[:1].cuda() for k, v in params.items() if hasattr(v, "cuda")}
    with torch.no_grad():
        verts = _BODY["smplx"](**p).vertices
        verts = torch.stack([torch.matmul(_BODY["x2s"], v) for v in verts])
        j = einsum(_BODY["jreg"], verts, "j v, l v i -> l j i")[0].cpu().numpy()
    seg = lambda a, b: float(np.linalg.norm(j[a] - j[b]))  # noqa: E731
    return (seg(1, 4) + seg(4, 7) + seg(2, 5) + seg(5, 8)) / 2


class ProcessIn(BaseModel):
    video_url: str


@app.get("/health")
def health():
    return {"ok": state["pipe"] is not None, "device": state["device"], "error": state["error"]}


@app.post("/process")
def process(body: ProcessIn):
    if state["pipe"] is None:
        GPU.submit(load_pipeline).result()
        if state["pipe"] is None:
            raise HTTPException(503, state["error"] or "GVHMR not loaded")

    t0 = time.perf_counter()
    key = hashlib.sha1(body.video_url.encode()).hexdigest()[:12]
    out_dir = CACHE / key
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "input.mp4"
    if not video.exists():
        src = out_dir / "source.mp4"
        with urllib.request.urlopen(body.video_url, timeout=30) as r, open(src, "wb") as f:
            shutil.copyfileobj(r, f)
        # GVHMR runs at 30fps and would re-encode a 24fps clip with a slow preset; do it fast here.
        # It only ever sees 256px person crops, so cap the long side at 854px (H3 480P size): 720p+
        # sources (FLUX 3) otherwise cost ~2x in decode/crop for no accuracy gain.
        fit = "scale='if(gt(iw,ih),min(854,iw),-2)':'if(gt(iw,ih),-2,min(854,ih))'"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", f"fps=30,{fit}", "-an",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", str(video)], check=True)
    t_dl = time.perf_counter() - t0
    if not (CACHE / "_warmup.mp4").exists():  # reused to warm CUDA kernels on the next start
        shutil.copyfile(video, CACHE / "_warmup.mp4")

    to_np = lambda x: (x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)).astype(np.float32)  # noqa: E731
    def run():
        result, t = recover(video, out_dir)
        world = result.smpl_params_world
        t_ll = time.perf_counter()
        ll = leg_length(world)
        t["leg_length"] = time.perf_counter() - t_ll
        return result, t, world, ll, to_np(result.smpl_params_camera["global_orient"])[0]

    result, t, world, ll, incam_go0 = GPU.submit(run).result()

    go, bp, tr = to_np(world["global_orient"]), to_np(world["body_pose"]), to_np(world["transl"])
    n = int(go.shape[0])

    from motion2mixamorig.extract import playback_fps

    fps = playback_fps(video, n, fallback=30.0)
    total = time.perf_counter() - t0
    log(f"{key}: {n} frames @ {fps:.1f}fps  download {t_dl:.2f}s  gvhmr {t['recover']:.2f}s  "
        f"stabilize {t['stabilize']:.2f}s  legs {t['leg_length']:.2f}s  total {total:.2f}s")
    return {
        "n": n, "fps": fps, "leg_length": ll,
        "global_orient": go.reshape(-1).round(5).tolist(),
        "body_pose": bp.reshape(n, -1)[:, :63].reshape(-1).round(5).tolist(),
        "transl": tr.reshape(-1).round(5).tolist(),
        # Frame-0 orientation relative to the camera (OpenCV axes): lets the client face the character
        # the way the person faces the camera, so chained clips keep a consistent heading.
        "incam_global_orient0": incam_go0.round(5).tolist(),
        "elapsed": total, "timings": {"download": t_dl, **t},
    }


if __name__ == "__main__":
    import uvicorn

    CACHE.mkdir(exist_ok=True)
    GPU.submit(load_pipeline)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("HQ_PORT", 5178)), log_level="warning")
