"""
Scan Server — entry point.

Launches scan_gui.py's Gradio UI (port 7861) — the venue-scanning /
offline live-mapping-pipeline debug tool. Video ingestion happens inside
the GUI itself now (scan_gui.py's "Load from Video" panel — upload a
video file, frames are extracted via OpenCV into uploads/<scan_id>/dataset/,
same images/+camera.csv layout as before), so there is no HTTP API left to
serve here — this file used to also run a FastAPI app (POST /api/upload,
GET /api/uploads) mounted alongside the Gradio UI via mount_gradio_app();
that layer is dropped entirely (not deprecated) now that nothing calls it
out-of-process. This is a plain Gradio launch.

Usage:
  cd scan_server
  python scan_server.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add server/ to sys.path so scan_server modules can import server/tools/*
_SERVER_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server")
)
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

# Add frame_extractor/ so "from tagging import FrameTagger" resolves —
# semantic_mapper.py's SemanticMapper is now built on top of frame_extractor's
# Gemini -> GroundingDINO-tiny pipeline (see semantic_mapper.py's module
# docstring) instead of the old Gemma VLM + full GroundingDINO design.
_FRAME_EXTRACTOR_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frame_extractor")
)
if _FRAME_EXTRACTOR_ROOT not in sys.path:
    sys.path.insert(0, _FRAME_EXTRACTOR_ROOT)

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
             "DA3METRIC-LARGE by default — see SCAN_DA3_TORCH_MODEL_ID). 'onnx' "
             "loads a DA3OnnxEstimator (DA3-METRIC ONNX, metric depth, CPU- or "
             "CUDA-runtime) — lighter/faster.",
    )
    parser.add_argument(
        "--da3-onnx-path", default=os.path.join(str(_BASE_DIR), "..", "DA3METRIC-LARGE.onnx"),
        help="Path to the DA3-METRIC ONNX file (only used with --da3-model onnx).",
    )
    return parser.parse_args()


def _build_ui(args: argparse.Namespace):
    """
    Load all models (DA3 estimators, Gemini + GroundingDINO-tiny for semantic
    mapping), connect the optional RTAB-Map pose client, and build the
    Gradio UI. Only called under `if __name__ == "__main__"` below.
    """
    # Main dense-depth estimator — either the DA3 torch model or the lighter
    # DA3-METRIC ONNX model, used for depth estimation regardless of pose
    # source (IMU + VO or RTAB-Map).
    if args.da3_model == "onnx":
        print(f"[SCAN SERVER] Loading DA3 ONNX model from '{args.da3_onnx_path}' on {_DEVICE}…")
        estimator = DA3OnnxEstimator(onnx_path=args.da3_onnx_path, device=_DEVICE)
        print("[SCAN SERVER] DA3 ONNX model ready (metric depth).")
    else:
        # Default matches grpc_server.py's own SCAN_DA3_TORCH_MODEL_ID fallback —
        # depth-anything/da3-large (the old default here) is a multi-view,
        # NOT monocular-metric model; using it caused a real RTAB-Map
        # total-tracking-failure incident, see CLAUDE.md's "DA3 model
        # default + per-frame processing" note.
        _DA3_TORCH_MODEL_ID = os.getenv("SCAN_DA3_TORCH_MODEL_ID", "depth-anything/DA3METRIC-LARGE")
        print(f"[SCAN SERVER] Loading DA3 torch model '{_DA3_TORCH_MODEL_ID}' on {_DEVICE}…")
        estimator = DA3Estimator(model_id=_DA3_TORCH_MODEL_ID, device=_DEVICE)
        print("[SCAN SERVER] DA3 torch model ready.")

    # Semantic Mapper — Gemini (open-set tagging via the Gemini API) ->
    # GroundingDINO-tiny (open-vocab detection, local), adapted wholesale
    # from frame_extractor/tagging.py + app.py (see semantic_mapper.py's
    # module docstring). This replaced the earlier Gemma-VLM-tag-then-
    # defer-GroundingDINO design entirely, and Gemini tagging (no local
    # checkpoint) replaced an intermediate RAM++-based tagging step (still true).
    _semantic_mapper = None
    try:
        from tagging import FrameTagger
        from semantic_mapper import SemanticMapper

        _GEMINI_TAG_MODEL_ID = os.getenv("GEMINI_TAGGING_MODEL_ID", FrameTagger.DEFAULT_GEMINI_MODEL)
        _GDINO_TAG_MODEL_ID = os.getenv("GDINO_TAGGING_MODEL_ID", "IDEA-Research/grounding-dino-base")
        print(f"[SCAN SERVER] Using Gemini '{_GEMINI_TAG_MODEL_ID}' + GroundingDINO-tiny "
              f"('{_GDINO_TAG_MODEL_ID}') for semantic mapping…")
        _frame_tagger = FrameTagger(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""), gemini_model_id=_GEMINI_TAG_MODEL_ID,
            gdino_model_id=_GDINO_TAG_MODEL_ID, device=_DEVICE,
            # Higher than FrameTagger's own default (0.3) — semantic mapping
            # landmarks feed navigation directly, so a shakier low-confidence
            # box is worse than just not tagging that object this frame.
            gdino_box_threshold=0.45,
        )
        _semantic_mapper = SemanticMapper(_frame_tagger)
        print("[SCAN SERVER] SemanticMapper ready.")
    except Exception as _sem_init_err:
        print(f"[SCAN SERVER] SemanticMapper init failed (semantic mapping disabled): {_sem_init_err}")

    # RTAB-Map pose client (optional — requires RTABMAP_ADDR + a running
    # scan_server/rtabmap_docker/ container). Silently disabled if unset or
    # unreachable, same as the semantic mapper above. No calibration file
    # needed at all — RTAB-Map's RGB-D odometry only needs camera intrinsics
    # (sent per-frame).
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

    return create_scan_ui(scan_manager, upload_dir=str(UPLOAD_DIR))


if __name__ == "__main__":
    demo = _build_ui(_parse_args())
    print(f"[SCAN SERVER] Gradio UI → http://0.0.0.0:{GRADIO_PORT}")
    demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT)
