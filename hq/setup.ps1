# One-time setup for the HQ worker: Windows + NVIDIA GPU, needs `uv` and `ffmpeg` on PATH.
# Afterwards put SMPLX_NEUTRAL.npz in hq/body_models/smplx/ (see README).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

uv venv --python 3.11 .venv
$env:VIRTUAL_ENV = Join-Path $PSScriptRoot ".venv"

# CUDA build of PyTorch first, so gvhmr doesn't pull the CPU wheel from PyPI.
uv pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu126
# chumpy (legacy gvhmr dependency) breaks under PEP 517 build isolation.
uv pip install pip setuptools wheel "numpy>=1.26"
uv pip install --no-build-isolation "chumpy==0.70"
uv pip install -r requirements.txt
# Finger tracking model code (weights come from setup_hands.py)
uv pip install --no-deps "git+https://github.com/warmshao/WiLoR-mini"

# GVHMR / HMR2 / ViTPose checkpoints (~5 GB) into hq/weights.
$env:GVHMR_CHECKPOINTS = Join-Path $PSScriptRoot "weights"
$env:GVHMR_BODY_MODELS = Join-Path $PSScriptRoot "body_models"
& .venv\Scripts\gvhmr.exe download demo
New-Item -ItemType Directory -Force body_models\smplx | Out-Null
Write-Host "Done. Now add body_models\smplx\SMPLX_NEUTRAL.npz (https://smpl-x.is.tue.mpg.de/, v1.1)."
Write-Host "Optional fingers: download mano_v1_2.zip (https://mano.is.tue.mpg.de) and run: .venv\Scripts\python setup_hands.py <zip>"
