import grpc
import os
from concurrent import futures
import queue
import threading

import whisper
from kokoro import KPipeline

import tracking_pb2_grpc
from agents import TrackingAgent, ReadingAgent, MemoryAgent, InfoAgent
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
)
from tools.intent_parser import GeneralIntentParser, ReadingIntentParser, TrackingIntentParser
from tools.embedder import EfficientNetLiteEmbedder
from vlm_wrapper import GeminiLiveVLM

memory_base_dir = os.getenv("MEMORY_STORE_DIR", os.path.join(os.path.dirname(__file__), "data", "memory"))
ocr_server_url = os.getenv("OCR_SERVER_URL", "http://localhost:8100")
gemini_api_key = os.getenv("GEMINI_API_KEY", "")

gui_frame_queue = queue.Queue(maxsize=10)
print("[SERVER] Initializing models...")

detector = GroundingDINODetector()
embedder = EfficientNetLiteEmbedder()

streaming_vlm_instance = GeminiLiveVLM(api_key=gemini_api_key)
print("[SERVER] GeminiLiveVLM ready.")

general_parser = GeneralIntentParser()
reading_parser = ReadingIntentParser()
tracking_parser = TrackingIntentParser()
memory_store = JsonMemoryStore(base_dir=memory_base_dir)
rag_store = RagStore(base_dir=memory_base_dir)
ocr = OCRTool(url=ocr_server_url)
asr_model = WhisperASR()
tts = KokoroTTS()

vlm_lock = threading.Lock()
memory_agent = MemoryAgent(
    store=memory_store,
    rag_store=rag_store,
    detector=detector,
    vlm=streaming_vlm_instance,
    vlm_lock=vlm_lock,
)
reading_agent = ReadingAgent(ocr=ocr, rag_store=rag_store)
tracking_agent = TrackingAgent(detector=detector)
info_agent = InfoAgent(vlm=streaming_vlm_instance, vlm_lock=vlm_lock)

orchestrator = Orchestrator(
    agents=[tracking_agent, reading_agent, memory_agent, info_agent],
    general_parser=general_parser,
    reading_parser=reading_parser,
    tracking_parser=tracking_parser,
    rag_store=rag_store,
)

servicer = TrackingServiceServicer(
    orchestrator=orchestrator,
    detector=detector,
    embedder=embedder,
    asr=asr_model,
    tts=tts,
    streaming_vlm_instance=streaming_vlm_instance,
    frame_queue=gui_frame_queue,
)


def _start_grpc_server(servicer_instance):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tracking_pb2_grpc.add_TrackingServiceServicer_to_server(servicer_instance, server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("[SERVER] gRPC server started on port 50051.")
    server.wait_for_termination()


if __name__ == "__main__":
    grpc_thread = futures.ThreadPoolExecutor(max_workers=1).submit(_start_grpc_server, servicer)
    app = create_ui(gui_frame_queue, streaming_vlm_instance, orchestrator)
    print("[SERVER] Launching Gradio app...")
    app.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")
