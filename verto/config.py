"""
verto/config.py
~~~~~~~~~~~~~~~
Central configuration for VERTO. All settings are read from environment
variables so they can be overridden in any environment without touching code.

Usage (override defaults):
    export VERTO_CHECKPOINT=/path/to/best_model.pt
    export VERTO_SPIDER_DIR=/path/to/spider
    export VERTO_DEVICE=cuda
    export VERTO_HOST=0.0.0.0
    export VERTO_PORT=8000
"""

import os
from pathlib import Path

# Project root — two levels up from this file (verto/config.py → VERTO/)
_ROOT = Path(__file__).resolve().parent.parent

# Path to the LoRA fine-tuned state-dict saved by the training script.
CHECKPOINT_PATH: str = os.environ.get(
    "VERTO_CHECKPOINT",
    str(_ROOT / "checkpoints" / "best_model.pt"),
)

# Root of the Spider dataset (must contain tables.json + database/).
SPIDER_DIR: str = os.environ.get(
    "VERTO_SPIDER_DIR",
    str(_ROOT / "data" / "spider"),
)

# Torch device. Leave blank for auto (CUDA → MPS → CPU).
DEVICE: str | None = os.environ.get("VERTO_DEVICE") or None

# Bind address for scripts/run_cloud.py (with uvicorn directly, pass --host and --port).
HOST: str = os.environ.get("VERTO_HOST", "0.0.0.0")
PORT: int = int(os.environ.get("VERTO_PORT", "8000"))

# Remote API Server URL for local CLI (e.g. "http://123.45.67.89:8000" or "https://xxx.ngrok-free.app")
SERVER_URL: str | None = os.environ.get("VERTO_SERVER_URL") or os.environ.get("VERTO_API_URL") or None

