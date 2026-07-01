import warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import grpc
import os
from concurrent import futures
import queue

import torch

import tracking_pb2_grpc
from map_service import MapServiceServicer
from services.servicer import TrackingServiceServicer
from server_gui import create_ui
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scan_server"))

from live_session import ToolsBundle, WalkingConfig
from tools import (
    GroundingDINODetector,
    OCRTool,
    JsonMemoryStore,
    RagStore,
    DummyRagStore,
    ObjectStore,
)
from tools.depth import SparseObstacleDetector
from tools.embedder import EfficientNetLiteEmbedder

memory_base_dir = os.getenv("MEMORY_STORE_DIR", os.path.join(os.path.dirname(__file__), "data", "memory"))
maps_root_dir = os.path.join(os.path.dirname(__file__), "data", "maps")
rag_model_id = os.getenv("RAG_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
rag_clip_model_id = os.getenv("RAG_CLIP_MODEL_ID", "clip-ViT-B-32")
ocr_server_url = os.getenv("OCR_SERVER_URL", "http://localhost:8100")
gemini_api_key = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6Lr2C7-ZasjbHtX-BC7QST_uR73D_sWOn-6D84SAzdQ4A")

gui_frame_queue = queue.Queue(maxsize=10)
print("[SERVER] Initializing models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

detector = GroundingDINODetector()
embedder = EfficientNetLiteEmbedder()

_depth_model = os.getenv("DEPTH_MODEL", "sparse")
if _depth_model == "stereo":
    from tools.depth import StereoDepthDetector
    depth_detector = StereoDepthDetector()
    print("[SERVER] Depth detector: StereoDepthDetector (plane sweep MVS, metric depth)")
elif _depth_model == "da3":
    from tools.depth import DA3DepthDetector
    depth_detector = DA3DepthDetector(os.getenv("DA3_MODEL_ID", "depth-anything/da3-large"))
    print("[SERVER] Depth detector: DA3DepthDetector (Depth Anything 3 + VIO scale alignment)")
else:
    depth_detector = SparseObstacleDetector()
    print("[SERVER] Depth detector: SparseObstacleDetector (ORB triangulation, relative depth)")

memory_store = JsonMemoryStore(base_dir=memory_base_dir)
object_store = ObjectStore(base_dir=memory_base_dir)
# rag_store = RagStore(base_dir=memory_base_dir, model_id=rag_model_id, clip_model_id=rag_clip_model_id)
rag_store = DummyRagStore()
ocr = OCRTool(url=ocr_server_url)

_da3_onnx_path = os.getenv("DA3_ONNX_PATH", os.path.join(os.path.dirname(__file__), "..", "DA3METRIC-LARGE.onnx"))
da3_onnx = None
try:
    from da3_wrapper import DA3OnnxEstimator
    if os.path.exists(_da3_onnx_path):
        da3_onnx = DA3OnnxEstimator(onnx_path=_da3_onnx_path, device=device)
        print(f"[SERVER] DA3 ONNX loaded from {_da3_onnx_path} — walking mode enabled")
    else:
        print(f"[SERVER] DA3 ONNX not found at {_da3_onnx_path} — walking mode disabled")
except Exception as e:
    print(f"[SERVER] DA3 ONNX load failed ({e}) — walking mode disabled")

walking_config = WalkingConfig()

tools_bundle = ToolsBundle(
    detector=detector,
    depth_detector=depth_detector,
    ocr=ocr,
    rag_store=rag_store,
    memory_store=memory_store,
    maps_root_dir=maps_root_dir,
    gemini_api_key=gemini_api_key,
    embedder=embedder,
    object_store=object_store,
    da3_onnx=da3_onnx,
    walking_config=walking_config,
)

servicer = TrackingServiceServicer(
    tools_bundle=tools_bundle,
    detector=detector,
    embedder=embedder,
    frame_queue=gui_frame_queue,
)


def _start_grpc_server(servicer_instance):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tracking_pb2_grpc.add_TrackingServiceServicer_to_server(servicer_instance, server)
    tracking_pb2_grpc.add_MapServiceServicer_to_server(MapServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("[SERVER] gRPC server started on port 50051.")
    server.wait_for_termination()


if __name__ == "__main__":
    grpc_thread = futures.ThreadPoolExecutor(max_workers=1).submit(_start_grpc_server, servicer)
    app = create_ui(gui_frame_queue, None, None, servicer)
    print("[SERVER] Launching Gradio app...")
    app.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")
