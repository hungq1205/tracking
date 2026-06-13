import cv2
import torch
import numpy as np
import logging
import os
import urllib.request
from PIL import Image
from typing import Tuple
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection, 
                          SiglipProcessor, SiglipModel, AutoImageProcessor, AutoModel)
from interfaces import Detection, IHandDetector, HandTrack

logger = logging.getLogger(__name__)

class GroundingDINODetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny"):
        print(f"[SERVER] Initializing Grounding DINO ({model_id})...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        # Load model in float32 initially. Autocast will handle mixed precision during inference.
        # Explicitly convert model to half() if on CUDA for memory efficiency.
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, low_cpu_mem_usage=True
        ).to(self.device)
        if self.device.type == "cuda":
            self.model.half() # Convert model parameters to float16

    def detect(self, frame: np.ndarray, prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        width, height = image.size
        clean_prompt = prompt.lower().strip()
        if not clean_prompt.endswith("."): clean_prompt += "."

        inputs = self.processor(images=image, text=clean_prompt, return_tensors="pt").to(self.device)
        if self.device.type == "cuda":
            for k, v in inputs.items():
                if torch.is_floating_point(v): inputs[k] = v.half() # Convert inputs to float16
        
        # Use autocast for mixed precision inference
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == "cuda" else torch.float32):
            with torch.no_grad():
                outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, 
            text_threshold=text_threshold, target_sizes=[(height, width)]
        )[0]
        
        if len(results["boxes"]) == 0:
            return Detection(box_xyxy=(0.0, 0.0, 0.0, 0.0), score=0.0)
            
        best_idx = int(torch.argmax(results["scores"]).item())
        box = results["boxes"][best_idx].cpu().numpy()
        return Detection(
            box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            score=float(results["scores"][best_idx].item()),
        )

class EfficientNetLiteEmbedder:
    # Placeholder for the embedder implementation used in your server
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embedding(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        # Logic for embedding generation...
        return torch.randn(1, 1280) # Mock

class YOLOHandDetector(IHandDetector):
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = YOLO("yolov8n.pt") 

    def detect(self, frame: np.ndarray) -> HandTrack:
        results = self._model.predict(frame, conf=0.4, verbose=False, device=self.device)[0]
        if len(results.boxes) > 0:
            box = results.boxes[0]
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            return HandTrack(
                box_xyxy=(x1, y1, x2, y2),
                center_xy=((x1 + x2) / 2, (y1 + y2) / 2),
                confidence=float(box.conf[0]),
                visible=True
            )
        h, w = frame.shape[:2]
        return HandTrack((0,0,0,0), (w/2, h/2), 0.0, False)