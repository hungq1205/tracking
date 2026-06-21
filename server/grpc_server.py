import grpc
import os
from concurrent import futures
import queue
import threading

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline as hf_pipeline
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
from vlm_wrapper import HanLabStreamingVLM

vlm_model_path = os.getenv("VLM_MODEL_PATH", "/models/qwen/3B")
vlm_model_id = os.getenv("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
intent_parser_model_id = os.getenv("INTENT_PARSER_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
memory_base_dir = os.getenv("MEMORY_STORE_DIR", os.path.join(os.path.dirname(__file__), "data", "memory"))

gui_frame_queue = queue.Queue(maxsize=10)
print("[SERVER] Initializing heavy models on GPU/High-end CPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"

detector = GroundingDINODetector()
embedder = EfficientNetLiteEmbedder()

# Load VLM first so it gets full GPU VRAM priority before any other model
streaming_vlm_instance = None
if os.path.exists(vlm_model_path):
    model_to_load = vlm_model_path
    print(f"[SERVER] Loading VLM from local path: {model_to_load}")
else:
    model_to_load = vlm_model_id
    print(f"[SERVER] Local model not found at {vlm_model_path}, downloading from HuggingFace: {model_to_load}")
vlm_model = AutoModelForImageTextToText.from_pretrained(
    model_to_load,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
vlm_processor = AutoProcessor.from_pretrained(model_to_load, use_fast=True)
streaming_vlm_instance = HanLabStreamingVLM(model=vlm_model, processor=vlm_processor, device=device)
print("[SERVER] StreamingVLM Ready.")

# Intent parser on CPU to avoid VRAM contention with the VLM
intent_pipe = hf_pipeline("text-generation", model=intent_parser_model_id, device="cpu")
general_parser = GeneralIntentParser(pipe=intent_pipe)
reading_parser = ReadingIntentParser(pipe=intent_pipe)
tracking_parser = TrackingIntentParser(pipe=intent_pipe)
memory_store = JsonMemoryStore(base_dir=memory_base_dir)
rag_store = RagStore(base_dir=memory_base_dir)
ocr = OCRTool()
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
    app = create_ui(gui_frame_queue, streaming_vlm_instance)
    print("[SERVER] Launching Gradio app...")
    app.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")
