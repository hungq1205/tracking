import warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import grpc
import os
from concurrent import futures

import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scan_server"))

import tracking_pb2_grpc
from services.activity_monitor import ActivityMonitor
from services.servicer import TrackingServiceServicer
from services.perception_servicer import PerceptionServiceServicer
from services.mapping_servicer import MappingServiceServicer  # imports stream_session — needs scan_server on sys.path first
from services.status_servicer import StatusServiceServicer
from server_gui import create_ui

from tools import GroundingDINODetector, RagStore, DummyRagStore
from tools.depth import DA3DepthDetector
from tools.embedder import DINOv2Embedder

gemini_api_key = os.getenv("GEMINI_API_KEY", "")

memory_base_dir = os.getenv("MEMORY_STORE_DIR", os.path.join(os.path.dirname(__file__), "data", "memory"))
rag_model_id = os.getenv("RAG_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")

maps_root_dir = os.path.join(os.path.dirname(__file__), "data", "maps")

# Fed by every servicer as real client RPCs land (see services/activity_monitor.py)
# — server_gui.py polls this to show live frames/results per RPC category,
# since there's no client-reported "mode" any more (see CLAUDE.md's
# "Client-Orchestrated Live Session" section).
activity_monitor = ActivityMonitor()

print("[SERVER] Initializing models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# detector = GroundingDINODetector()
# embedder = DINOv2Embedder()
detector, embedder = None, None

depth_detector = DA3DepthDetector(
    onnx_path=os.getenv("DA3_ONNX_PATH", "DA3METRIC-LARGE.onnx"),
    device=device,
)
print("[SERVER] Depth detector: DA3DepthDetector (Depth Anything 3 ONNX + VIO scale alignment)")
try:
    import time as _time
    import numpy as _np
    _t0 = _time.time()
    depth_detector.check_obstacle(_np.random.randint(0, 255, (480, 640, 3), dtype=_np.uint8))
    print(f"[SERVER] DA3 (ONNX, obstacle-check) warmed up in {_time.time() - _t0:.1f}s")
except Exception as e:
    print(f"[SERVER] DA3 (ONNX) warmup failed (non-fatal): {e}")

# rag_store = RagStore(base_dir=memory_base_dir, image_embedder=embedder, model_id=rag_model_id)
rag_store = DummyRagStore()

tts = None
try:
    from tools.tts import KokoroTTS
    # tts = KokoroTTS()
    print("[SERVER] KokoroTTS loaded — Synthesize RPC enabled")
except Exception as e:
    print(f"[SERVER] KokoroTTS load failed ({e}) — Synthesize RPC disabled")

servicer = TrackingServiceServicer(detector=detector, embedder=embedder, activity_monitor=activity_monitor)
status_servicer = StatusServiceServicer(activity_monitor=activity_monitor)

# Consolidated heavy-compute surface for the Android client's on-device
# Gemini Live tool-dispatch loop — see CLAUDE.md's "Client-Orchestrated
# Live Session" section.
perception_servicer = PerceptionServiceServicer(
    detector=detector,
    embedder=embedder,
    depth_detector=depth_detector,
    tts=tts,
    rag_store=rag_store,
    activity_monitor=activity_monitor,
)

# Live SLAM-style mapping for the Android client's guiding mode — RTAB-Map
# pose only (see CLAUDE.md's "Client-Orchestrated Live Session" section).
# A dedicated DA3 estimator is loaded here rather than sharing one across
# services — thread-safety of concurrent inference calls into one shared
# DA3 instance from two independent live gRPC streams hasn't been verified,
# so this trades some extra GPU memory for a correctness guarantee instead
# of assuming it's fine.
mapping_servicer = None
try:
    from da3_wrapper import DA3Estimator
    from scan_session import ScanSessionManager
    from rtabmap_client import RtabmapPoseClient

    _rtabmap_addr = os.getenv("RTABMAP_ADDR", "")
    if not _rtabmap_addr:
        print("[SERVER] RTABMAP_ADDR not set — MappingService disabled (RTAB-Map is the only supported pose source).")
    else:
        _mapping_rtabmap_client = RtabmapPoseClient(_rtabmap_addr)
        _mapping_da3_model_id = os.getenv("SCAN_DA3_TORCH_MODEL_ID", "depth-anything/DA3METRIC-LARGE")
        _mapping_estimator = DA3Estimator(model_id=_mapping_da3_model_id, device=device)

        # Warm up CUDA kernel selection/autotuning + memory allocator now, at
        # startup, instead of paying for it on the first real client batch —
        # observed directly in this project: the first few live
        # estimate_batch() calls took 25-53s each (cuDNN benchmarking a new
        # input shape, allocator growth), dropping to under 1s once warm.
        # Same mini_batch size (4) StreamingScanSession actually uses, so the
        # warmup exercises the exact shape/path real traffic will hit.
        try:
            import time as _time
            import numpy as _np
            _warmup_frames = [
                _np.random.randint(0, 255, (480, 640, 3), dtype=_np.uint8) for _ in range(4)
            ]
            _t0 = _time.time()
            _mapping_estimator.estimate_batch(_warmup_frames)
            print(f"[SERVER] DA3 (torch, mapping) warmed up in {_time.time() - _t0:.1f}s")
        except Exception as e:
            print(f"[SERVER] DA3 (torch, mapping) warmup failed (non-fatal, first real batch will pay the cost): {e}")

        _mapping_semantic_mapper = None
        try:
            from gemma_vlm import GemmaVLMClient
            from semantic_mapper import SemanticMapper
            _gemma_model_id = os.getenv("SCAN_GEMMA_MODEL_ID", GemmaVLMClient.DEFAULT_MODEL)
            _vlm = GemmaVLMClient(model_id=_gemma_model_id, api_key=gemini_api_key)
            _mapping_semantic_mapper = SemanticMapper(_vlm, detector)  # reuses the already-loaded GroundingDINO instance
            print(f"[SERVER] Mapping SemanticMapper ready (Gemma '{_gemma_model_id}').")
        except Exception as e:
            print(f"[SERVER] Mapping SemanticMapper init failed ({e}) — landmark extraction disabled for MappingService.")

        scan_manager = ScanSessionManager(
            estimator=_mapping_estimator,
            rtabmap_client=_mapping_rtabmap_client,
            semantic_mapper=_mapping_semantic_mapper,
        )
        mapping_servicer = MappingServiceServicer(scan_manager, maps_root_dir, activity_monitor=activity_monitor)
        print(f"[SERVER] MappingService ready (RTAB-Map at {_rtabmap_addr}).")
except Exception as e:
    print(f"[SERVER] MappingService init failed ({e}) — disabled.")


def _start_grpc_server(servicer_instance):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tracking_pb2_grpc.add_TrackingServiceServicer_to_server(servicer_instance, server)
    tracking_pb2_grpc.add_PerceptionServiceServicer_to_server(perception_servicer, server)
    tracking_pb2_grpc.add_StatusServiceServicer_to_server(status_servicer, server)
    if mapping_servicer is not None:
        tracking_pb2_grpc.add_MappingServiceServicer_to_server(mapping_servicer, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("[SERVER] gRPC server started on port 50051.")
    server.wait_for_termination()


if __name__ == "__main__":
    grpc_thread = futures.ThreadPoolExecutor(max_workers=1).submit(_start_grpc_server, servicer)
    app = create_ui(activity_monitor)
    print("[SERVER] Launching Gradio app...")
    app.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")
