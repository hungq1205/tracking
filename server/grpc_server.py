import warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import grpc
import os
from concurrent import futures
import queue
import threading

import torch
import whisper
from kokoro import KPipeline

import tracking_pb2_grpc
from map_service import MapServiceServicer
from agents import TrackingAgent, ReadingAgent, MemoryAgent, NavigationAgent, InfoAgent
from orchestrator import Orchestrator
from services.servicer import TrackingServiceServicer
from server_gui import create_ui
from tools import (
    GroundingDINODetector,
    OCRTool,
    JsonMemoryStore,
    RagStore,
    KokoroTTS,
    WhisperASR,
    create_cloud_vlm_client,
)
from tools.depth import SparseObstacleDetector
from tools.intent_parser import GeneralIntentParser, ReadingIntentParser, TrackingIntentParser, NavigationIntentParser
from tools.embedder import EfficientNetLiteEmbedder

memory_base_dir = os.getenv("MEMORY_STORE_DIR", os.path.join(os.path.dirname(__file__), "data", "memory"))
maps_root_dir = os.path.join(os.path.dirname(__file__), "data", "maps")
rag_model_id = os.getenv("RAG_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
rag_clip_model_id = os.getenv("RAG_CLIP_MODEL_ID", "clip-ViT-B-32")
ocr_server_url = os.getenv("OCR_SERVER_URL", "http://localhost:8100")
cloud_vlm_vendor = os.getenv("CLOUD_VLM_VENDOR", "stub")
cloud_vlm_api_key = os.getenv("CLOUD_VLM_API_KEY", "")
gemini_api_key = os.getenv("GEMINI_API_KEY", cloud_vlm_api_key)

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

_vlm_key = gemini_api_key if cloud_vlm_vendor == "gemini" else cloud_vlm_api_key
cloud_vlm = create_cloud_vlm_client(vendor=cloud_vlm_vendor, api_key=_vlm_key)
print(f"[SERVER] Cloud VLM: vendor={cloud_vlm_vendor}.")

general_parser = GeneralIntentParser()
reading_parser = ReadingIntentParser()
tracking_parser = TrackingIntentParser()
navigation_parser = NavigationIntentParser()
memory_store = JsonMemoryStore(base_dir=memory_base_dir)
rag_store = RagStore(base_dir=memory_base_dir, model_id=rag_model_id, clip_model_id=rag_clip_model_id)
ocr = OCRTool(url=ocr_server_url)
asr_model = WhisperASR()
tts = KokoroTTS()

memory_agent = MemoryAgent(
    store=memory_store,
    rag_store=rag_store,
    detector=detector,
    cloud_vlm=cloud_vlm,
)
reading_agent = ReadingAgent(ocr=ocr, rag_store=rag_store)
tracking_agent = TrackingAgent(detector=detector)
navigation_agent = NavigationAgent(
    depth_detector=depth_detector,
    cloud_vlm=cloud_vlm,
    maps_root_dir=maps_root_dir,
)
info_agent = InfoAgent(cloud_vlm=cloud_vlm)

orchestrator = Orchestrator(
    agents=[tracking_agent, reading_agent, memory_agent, navigation_agent, info_agent],
    general_parser=general_parser,
    reading_parser=reading_parser,
    tracking_parser=tracking_parser,
    navigation_parser=navigation_parser,
    rag_store=rag_store,
)

servicer = TrackingServiceServicer(
    orchestrator=orchestrator,
    detector=detector,
    embedder=embedder,
    asr=asr_model,
    tts=tts,
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
    app = create_ui(gui_frame_queue, None, orchestrator, servicer)
    print("[SERVER] Launching Gradio app...")
    app.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")
