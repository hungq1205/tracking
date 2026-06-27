"""
Scan Server — entry point.

Runs a FastAPI app (port 7861) with:
  - POST /api/upload   — receives video (required) + imu_data.csv (optional) from Android
  - GET  /api/uploads  — lists available upload scan IDs
  - /                  — Gradio UI for venue scanning (mounted at root)

Usage:
  cd scan_server
  python scan_server.py
"""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add server/ to sys.path so scan_server modules can import server/tools/*
_SERVER_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server")
)
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from gradio import mount_gradio_app

from da3_wrapper import DA3Estimator, DA3OnnxEstimator
from scan_session import ScanSessionManager
from scan_gui import create_scan_ui

GRADIO_PORT = int(os.getenv("SCAN_GRADIO_PORT", "7861"))
_DEVICE = os.getenv("SCAN_DEVICE", "cuda")

_BASE_DIR = Path(__file__).parent
UPLOAD_DIR = _BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

print(f"[SCAN SERVER] Loading DA3 model on {_DEVICE}…")
# estimator = DA3Estimator(model_id="depth-anything/da3-small", device=_DEVICE)
estimator = DA3OnnxEstimator(onnx_path="../DA3METRIC-LARGE.onnx", device=_DEVICE)
print("[SCAN SERVER] DA3 model ready.")

# ── Semantic Mapper (optional — requires OPENROUTER_API_KEY) ──────────────────
_semantic_mapper = None
_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
if _OPENROUTER_KEY:
    try:
        from tools.cloud_vlm import OpenRouterVLMClient
        from tools.detector import GroundingDINODetector
        from semantic_mapper import SemanticMapper

        print("[SCAN SERVER] Loading GroundingDINO for semantic mapping…")
        _detector = GroundingDINODetector()
        _vlm = OpenRouterVLMClient(api_key=_OPENROUTER_KEY)
        _semantic_mapper = SemanticMapper(_vlm, _detector)
        print("[SCAN SERVER] SemanticMapper ready.")
    except Exception as _sem_init_err:
        print(f"[SCAN SERVER] SemanticMapper init failed (semantic mapping disabled): {_sem_init_err}")
else:
    print("[SCAN SERVER] OPENROUTER_API_KEY not set — semantic mapping disabled.")

scan_manager = ScanSessionManager(estimator=estimator, semantic_mapper=_semantic_mapper)

# ── FastAPI app ────────────────────────────────────────────────────────────────

api = FastAPI(title="Scan Server")


@api.post("/api/upload")
async def upload_scan(
    video: UploadFile = File(...),
    imu: UploadFile = File(None),
):
    """Receive video.mp4 + optional imu_data.csv from Android client."""
    scan_id = uuid.uuid4().hex
    scan_dir = UPLOAD_DIR / scan_id
    scan_dir.mkdir(parents=True)

    video_path = scan_dir / "video.mp4"
    video_path.write_bytes(await video.read())

    imu_path = None
    if imu is not None:
        imu_path = scan_dir / "imu_data.csv"
        imu_path.write_bytes(await imu.read())

    return JSONResponse({
        "scan_id": scan_id,
        "video": str(video_path.relative_to(_BASE_DIR)),
        "imu": str(imu_path.relative_to(_BASE_DIR)) if imu_path else None,
    })


@api.get("/api/uploads")
async def list_uploads():
    """List all scan IDs available on the server."""
    scan_ids = [d.name for d in sorted(UPLOAD_DIR.iterdir()) if d.is_dir()]
    return JSONResponse({"scan_ids": scan_ids})


# ── Mount Gradio UI at root ────────────────────────────────────────────────────

gradio_app = create_scan_ui(scan_manager, upload_dir=str(UPLOAD_DIR))
app = mount_gradio_app(api, gradio_app, path="/")

if __name__ == "__main__":
    print(f"[SCAN SERVER] API + Gradio UI → http://0.0.0.0:{GRADIO_PORT}")
    print(f"[SCAN SERVER]   POST /api/upload   — video + optional IMU CSV")
    print(f"[SCAN SERVER]   GET  /api/uploads  — list scan IDs")
    print(f"[SCAN SERVER]   /                  — Gradio UI")
    uvicorn.run(app, host="0.0.0.0", port=GRADIO_PORT)
