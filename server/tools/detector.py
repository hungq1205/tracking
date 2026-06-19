import cv2
import torch
import logging
from PIL import Image

from interfaces import Detection

logger = logging.getLogger(__name__)


class GroundingDINODetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny"):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        print(f"[SERVER] Initializing Grounding DINO ({model_id})...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, low_cpu_mem_usage=True
        ).to(self.device)
        if self.device.type == "cuda":
            self.model.half()

    def detect(self, frame, prompt: str, box_threshold: float = 0.35, text_threshold: float = 0.25):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        width, height = image.size
        clean_prompt = prompt.lower().strip()
        if not clean_prompt.endswith("."):
            clean_prompt += "."

        inputs = self.processor(images=image, text=clean_prompt, return_tensors="pt").to(self.device)
        if self.device.type == "cuda":
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ):
            with torch.no_grad():
                outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
        )[0]

        if len(results["boxes"]) == 0:
            return Detection(box_xyxy=(0.0, 0.0, 0.0, 0.0), score=0.0)

        best_idx = int(torch.argmax(results["scores"]).item())
        box = results["boxes"][best_idx].cpu().numpy()
        return Detection(
            box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            score=float(results["scores"][best_idx].item()),
        )
