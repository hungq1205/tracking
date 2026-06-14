import grpc
import os
from concurrent import futures
import tracking_pb2
import tracking_pb2_grpc
from models import GroundingDINODetector, EfficientNetLiteEmbedder
import queue
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import whisper
from kokoro import KPipeline
from vlm_wrapper import HanLabStreamingVLM
from servicer import TrackingServiceServicer
from server_gui import create_ui

vlm_model_path = os.getenv("VLM_MODEL_PATH", "/models/qwen")

gui_frame_queue = queue.Queue(maxsize=10)
print("[SERVER] Initializing heavy models on GPU/High-end CPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = GroundingDINODetector()
embedder = EfficientNetLiteEmbedder()

# Initialize StreamingVLM components
streaming_vlm_instance = None
vlm_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
print("[SERVER] Initializing StreamingVLM - ${vlm_model_id}...")
vlm_model = AutoModelForImageTextToText.from_pretrained(
    vlm_model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
vlm_processor = AutoProcessor.from_pretrained(vlm_model_path)
streaming_vlm_instance = HanLabStreamingVLM(model=vlm_model, processor=vlm_processor, device=device)
print("[SERVER] StreamingVLM Ready.")
print("[SERVER] Initializing ASR (Whisper) & TTS (Kokoro)...")
asr_model = whisper.load_model("base")
tts_pipeline = KPipeline(lang_code='a')
print("[SERVER] ASR & TTS Ready.")

def _start_grpc_server(servicer_instance):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tracking_pb2_grpc.add_TrackingServiceServicer_to_server(servicer_instance, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("[SERVER] gRPC server started on port 50051.")
    server.wait_for_termination()

if __name__ == "__main__":
    servicer = TrackingServiceServicer(detector, embedder, asr_model, tts_pipeline, streaming_vlm_instance, gui_frame_queue)
    grpc_thread = futures.ThreadPoolExecutor(max_workers=1).submit(_start_grpc_server, servicer)
    app = create_ui(gui_frame_queue, streaming_vlm_instance)
    print("[SERVER] Launching Gradio app...")
    app.queue().launch(server_name="0.0.0.0", server_port=7860, theme="monochrome")