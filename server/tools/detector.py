import cv2
import torch
import logging
from typing import List
from PIL import Image

from interfaces import Detection, IObjectDetector

logger = logging.getLogger(__name__)


def _post_process_grounded(processor, outputs, input_ids, box_threshold, text_threshold, target_sizes):
    """Layered backward-compat call, same style already used elsewhere in
    this codebase (e.g. rtabmap_client.py's node_id/inlier_fraction parsing)
    — transformers renamed post_process_grounded_object_detection's
    confidence-threshold kwarg from `box_threshold` (<=4.46.x) to
    `threshold` (>=5.x). Observed in practice: the installed version can
    disagree with what's actually loaded at runtime (multiple Python
    environments on the same machine), so this tries the current signature
    first and falls back instead of assuming one or the other."""
    try:
        return processor.post_process_grounded_object_detection(
            outputs, input_ids, threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=target_sizes,
        )
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs, input_ids, box_threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=target_sizes,
        )


class GroundingDINODetector(IObjectDetector):
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
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
        input_ids = inputs["input_ids"]
        if self.device.type == "cuda":
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ):
            with torch.no_grad():
                outputs = self.model(**inputs)

        results = _post_process_grounded(
            self.processor, outputs, input_ids, box_threshold, text_threshold, [(height, width)],
        )[0]

        if len(results["boxes"]) == 0:
            return Detection(box_xyxy=(0.0, 0.0, 0.0, 0.0), score=0.0)

        best_idx = int(torch.argmax(results["scores"]).item())
        box = results["boxes"][best_idx].cpu().numpy()
        return Detection(
            box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            score=float(results["scores"][best_idx].item()),
        )

    def detect_all(
        self,
        frame,
        prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> List:
        """Return all detections above threshold, each with its decoded text label."""
        from interfaces import LabeledDetection

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        width, height = image.size
        clean_prompt = prompt.lower().strip()
        if not clean_prompt.endswith("."):
            clean_prompt += "."

        inputs = self.processor(images=image, text=clean_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        if self.device.type == "cuda":
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ):
            with torch.no_grad():
                outputs = self.model(**inputs)

        results = _post_process_grounded(
            self.processor, outputs, input_ids, box_threshold, text_threshold, [(height, width)],
        )[0]

        if len(results["boxes"]) == 0:
            return []

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        # Same transformers<=4.46.x vs >=5.x rename _post_process_grounded()
        # above already works around for the call's kwarg name — the RETURN
        # dict's label key was renamed too ("labels" -> "text_labels"), and
        # this codebase has actually observed both in practice depending on
        # which environment's transformers ends up loaded at runtime.
        labels = results.get("text_labels", results.get("labels"))

        return [
            LabeledDetection(
                label=lbl.strip(),
                box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=float(sc),
            )
            for box, sc, lbl in zip(boxes, scores, labels)
        ]
