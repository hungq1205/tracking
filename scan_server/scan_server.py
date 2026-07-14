"""
Scan Server — entry point.

Runs a FastAPI app (port 7861) with:
  - POST /api/upload   — receives a dataset.zip (images/ + imu.csv + camera.csv) from Android
  - GET  /api/uploads  — lists available upload scan IDs
  - /                  — Gradio UI for venue scanning (mounted at root)

Usage:
  cd scan_server
  python scan_server.py
"""

import argparse
import io
import os
import sys
import uuid
import zipfile
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--da3-model", choices=["torch", "onnx"], default="torch",
        help="Depth estimator backend for scan_session's main dense-depth pass, "
             "used for depth estimation regardless of pose source (IMU + VO or "
             "RTAB-Map). 'torch' (default) loads a DA3Estimator (depth-anything/"
             "da3-large by default). 'onnx' loads a DA3OnnxEstimator (DA3-METRIC "
             "ONNX, metric depth, CPU- or CUDA-runtime) — lighter/faster.",
    )
    parser.add_argument(
        "--da3-onnx-path", default=os.path.join(str(_BASE_DIR), "..", "DA3METRIC-LARGE.onnx"),
        help="Path to the DA3-METRIC ONNX file (only used with --da3-model onnx).",
    )
    return parser.parse_args()

# ── FastAPI app ────────────────────────────────────────────────────────────────

api = FastAPI(title="Scan Server")


@api.post("/api/upload")
async def upload_scan(dataset: UploadFile = File(...)):
    """
    Receive a dataset.zip from the Android client and extract it to
    uploads/<scan_id>/dataset/. Expected zip layout:
        images/000000000.jpg, images/000000001.jpg, ...
        imu.csv     (header: timestamp_ns,ax,ay,az,gx,gy,gz)
        camera.csv  (header: timestamp_ns,filename)
    """
    scan_id = uuid.uuid4().hex
    scan_dir = UPLOAD_DIR / scan_id
    dataset_dir = scan_dir / "dataset"
    dataset_dir.mkdir(parents=True)

    zip_bytes = await dataset.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            # Guard against zip-slip: reject entries that escape dataset_dir.
            dest = (dataset_dir / member).resolve()
            if not str(dest).startswith(str(dataset_dir.resolve())):
                raise HTTPException(status_code=400, detail=f"Unsafe zip entry: {member}")
        zf.extractall(dataset_dir)

    return JSONResponse({
        "scan_id": scan_id,
        "dataset": str(dataset_dir.relative_to(_BASE_DIR)),
    })


@api.get("/api/uploads")
async def list_uploads():
    """List all scan IDs available on the server."""
    scan_ids = [d.name for d in sorted(UPLOAD_DIR.iterdir()) if d.is_dir()]
    return JSONResponse({"scan_ids": scan_ids})


def _build_app(args: argparse.Namespace):
    """
    Load all models (DA3 estimators, GroundingDINO, Gemma via Gemini API),
    connect the optional RTAB-Map pose client, and mount the Gradio UI. Only
    called under `if __name__ == "__main__"` below — if qwen_vlm.py's local
    vLLM-backed Qwen3VLClient is ever swapped back in for GemmaVLMClient,
    note vLLM's LLM(...) spawns a worker subprocess
    (VLLM_WORKER_MULTIPROC_METHOD=spawn) that re-imports this script; Python
    sets that child's __name__ to "__mp_main__" rather than "__main__", so
    keeping all heavy loading behind this guard prevents the child from
    re-running it too (which would spawn its own worker, looping forever).
    """
    # Main dense-depth estimator — either the DA3 torch model or the lighter
    # DA3-METRIC ONNX model, used for depth estimation regardless of pose
    # source (IMU + VO or RTAB-Map).
    if args.da3_model == "onnx":
        print(f"[SCAN SERVER] Loading DA3 ONNX model from '{args.da3_onnx_path}' on {_DEVICE}…")
        estimator = DA3OnnxEstimator(onnx_path=args.da3_onnx_path, device=_DEVICE)
        print("[SCAN SERVER] DA3 ONNX model ready (metric depth).")
    else:
        _DA3_TORCH_MODEL_ID = os.getenv("SCAN_DA3_TORCH_MODEL_ID", "depth-anything/da3-large")
        print(f"[SCAN SERVER] Loading DA3 torch model '{_DA3_TORCH_MODEL_ID}' on {_DEVICE}…")
        estimator = DA3Estimator(model_id=_DA3_TORCH_MODEL_ID, device=_DEVICE)
        print("[SCAN SERVER] DA3 torch model ready.")

    # Semantic Mapper (Gemma 4 31B via Gemini API + GroundingDINO).
    # Local Qwen3-VL (qwen_vlm.py) is disabled for now — its 2B model was
    # prone to greedy-decoding repetition loops in grounding_dino_prompt
    # output (e.g. "bed frame" repeated ~150x); kept in the repo for
    # reference/future use, not deleted.
    _semantic_mapper = None
    try:
        from tools.detector import GroundingDINODetector
        from gemma_vlm import GemmaVLMClient
        from semantic_mapper import SemanticMapper

        print("[SCAN SERVER] Loading GroundingDINO for semantic mapping…")
        _detector = GroundingDINODetector()
        _GEMMA_MODEL_ID = os.getenv("SCAN_GEMMA_MODEL_ID", GemmaVLMClient.DEFAULT_MODEL)
        print(f"[SCAN SERVER] Using Gemma '{_GEMMA_MODEL_ID}' (Gemini API) for semantic mapping…")
        _vlm = GemmaVLMClient(model_id=_GEMMA_MODEL_ID, api_key=os.getenv("GEMINI_API_KEY", ""))
        _semantic_mapper = SemanticMapper(_vlm, _detector)
        print("[SCAN SERVER] SemanticMapper ready.")
    except Exception as _sem_init_err:
        print(f"[SCAN SERVER] SemanticMapper init failed (semantic mapping disabled): {_sem_init_err}")

    # RTAB-Map pose client (optional — requires RTABMAP_ADDR + a running
    # scan_server/rtabmap_docker/ container). Silently disabled if unset or
    # unreachable, same as the semantic mapper above. No calibration file
    # needed at all — RTAB-Map's RGB-D odometry only needs camera intrinsics
    # (sent per-frame), unlike the removed ORB-SLAM3 client this replaces.
    _rtabmap_client = None
    _RTABMAP_ADDR = os.getenv("RTABMAP_ADDR", "")
    if _RTABMAP_ADDR:
        try:
            from rtabmap_client import RtabmapPoseClient

            print(f"[SCAN SERVER] Connecting to RTAB-Map at {_RTABMAP_ADDR}…")
            _rtabmap_client = RtabmapPoseClient(_RTABMAP_ADDR)
            print("[SCAN SERVER] RTAB-Map pose client ready.")
        except Exception as _rtab_init_err:
            print(f"[SCAN SERVER] RTAB-Map client init failed (RTAB-Map pose source disabled): {_rtab_init_err}")
    else:
        print("[SCAN SERVER] RTABMAP_ADDR not set — RTAB-Map pose source disabled.")

    scan_manager = ScanSessionManager(
        estimator=estimator,
        rtabmap_client=_rtabmap_client, semantic_mapper=_semantic_mapper,
    )

    gradio_app = create_scan_ui(scan_manager, upload_dir=str(UPLOAD_DIR))
    return mount_gradio_app(api, gradio_app, path="/")


if __name__ == "__main__":
    app = _build_app(_parse_args())
    print(f"[SCAN SERVER] API + Gradio UI → http://0.0.0.0:{GRADIO_PORT}")
    print(f"[SCAN SERVER]   POST /api/upload   — dataset.zip (images/ + imu.csv + camera.csv)")
    print(f"[SCAN SERVER]   GET  /api/uploads  — list scan IDs")
    print(f"[SCAN SERVER]   /                  — Gradio UI")
    uvicorn.run(app, host="0.0.0.0", port=GRADIO_PORT)
