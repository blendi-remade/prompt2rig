"""One-time setup for finger tracking.

    hq/.venv/Scripts/python hq/setup_hands.py path/to/mano_v1_2.zip

- extracts MANO_RIGHT.pkl from the official MANO download (mano.is.tue.mpg.de — registration, non-commercial)
  and re-saves it without chumpy objects (chumpy doesn't run on Python ≥ 3.11 / NumPy ≥ 1.24)
- downloads the WiLoR checkpoint, hand detector and MANO mean params from the WiLoR authors' Hugging Face Space
- downloads MediaPipe's hand landmarker (fallback backend)
"""

import inspect
import pickle
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HQ = Path(__file__).resolve().parent
WILOR = HQ / "weights" / "wilor" / "pretrained_models"
SPACE = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main"
FILES = {
    "detector.pt": f"{SPACE}/pretrained_models/detector.pt",
    "wilor_final.ckpt": f"{SPACE}/pretrained_models/wilor_final.ckpt",       # ~2.5 GB
    "mano_mean_params.npz": f"{SPACE}/mano_data/mano_mean_params.npz",
}
MP_TASK = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def fetch(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"  have {dst.name}")
        return
    print(f"  downloading {dst.name} …")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)


def convert_mano(zip_path: Path) -> None:
    out = WILOR / "MANO_RIGHT.pkl"
    raw = WILOR / "MANO_RIGHT.chumpy.pkl"
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("models/MANO_RIGHT.pkl"))
        raw.write_bytes(z.read(name))
    # chumpy shims: inspect.getargspec (removed in 3.11) and NumPy aliases (removed in 1.24)
    inspect.getargspec = inspect.getfullargspec
    for alias, t in (("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object), ("unicode", str), ("str", str)):
        if alias not in np.__dict__:
            setattr(np, alias, t)
    import chumpy

    data = pickle.load(open(raw, "rb"), encoding="latin1")
    data = {k: (np.array(v.r) if isinstance(v, chumpy.Ch) else v) for k, v in data.items()}
    pickle.dump(data, open(out, "wb"))
    print(f"  MANO_RIGHT.pkl converted ({len(data)} arrays)")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    WILOR.mkdir(parents=True, exist_ok=True)
    print("MANO:")
    convert_mano(Path(sys.argv[1]))
    print("WiLoR:")
    for name, url in FILES.items():
        fetch(url, WILOR / name)
    print("MediaPipe:")
    fetch(MP_TASK, HQ / "weights" / "hand_landmarker.task")
    print("done — restart hq/worker.py; it picks WiLoR automatically")
    return 0


if __name__ == "__main__":
    sys.exit(main())
