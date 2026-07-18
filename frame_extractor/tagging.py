"""
RAM++ (open-set image tagging) -> GroundingDINO tiny (open-vocab detection),
run on every frame extractor.py accepts as "new". RAM++ proposes what's
actually in a given frame with no fixed vocabulary needed up front; those
tags become GroundingDINO's text prompt for that same frame, so what gets
boxed tracks what RAM++ actually saw instead of one hardcoded prompt for
every frame.

Both models are loaded once (FrameTagger is meant to be constructed a single
time and reused) and run BATCHED across every frame accepted in one
extractor.py flush, mirroring the DA3 depth batching already there
(_flush_rtabmap_batch) — batching is most of the actual speed win here, since
per-call Python/CUDA-launch overhead is amortized across the batch instead of
paid once per frame.

VRAM/speed:
  - GroundingDINO: full `.half()` weights + fp16 autocast — the same,
    already-proven approach server/tools/detector.py uses for this exact
    model.
  - RAM++: fp16 autocast only, weights stay fp32 — its vendored BERT
    attention-mask arithmetic (additive -10000-scale masking before softmax)
    is more fp16-overflow-prone than a from-scratch model, so autocast
    (which runs softmax/reductions in fp32 internally) is used instead of a
    full weight conversion.
  - `torch.inference_mode()` (cheaper than `no_grad` — skips autograd
    version-counter bookkeeping entirely) around every forward pass.
  - Every frame is resized to each model's OWN preferred resolution before
    that model sees it, not whatever resolution the source video happens to
    be: RAM++ -> 384x384 (`ram.get_transform`, the size its
    ram_plus_swin_large_14m checkpoint was trained/calibrated at);
    GroundingDINO tiny -> its processor's own shortest_edge=800/
    longest_edge=1333 config (aspect-ratio-preserving resize, the
    checkpoint's own preferred input size) — a large extracted video frame
    is never handed to either model at native resolution.
"""
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from PIL import Image

def _with_article(tag: str) -> str:
    article = "an" if tag[:1] in "aeiou" else "a"
    return f"{article} {tag}"


def _build_prompt(tags: List[str]) -> str:
    # GroundingDINO's own convention: lowercase, article-prefixed, " . "-separated,
    # trailing " .", e.g. "a chair . a table . an apple ."
    if not tags:
        return "an object ."
    return " . ".join(_with_article(tag.lower()) for tag in tags) + " ."


@dataclass
class TagDetection:
    label: str
    box_xyxy: tuple
    score: float


@dataclass
class TagResult:
    tags: List[str]              # RAM++ tags for this frame, e.g. ["chair", "table", "door"]
    prompt: str                  # the exact dot-separated GroundingDINO text prompt built from `tags`
    detections: List[TagDetection]  # GroundingDINO boxes, prompted by the tags above


class FrameTagger:
    """RAM++ tagging -> GroundingDINO-tiny detection, batched across a list
    of frames. Construct once (loads both models onto `device`) and reuse
    across every accepted-frame batch."""

    def __init__(
        self,
        ram_checkpoint: str,
        gdino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        ram_image_size: int = 384,
        ram_tag_threshold: float = 0.68,
        gdino_box_threshold: float = 0.3,
        gdino_text_threshold: float = 0.25,
        max_tags_per_frame: int = 12,
    ):
        from ram import get_transform
        from ram.models import ram_plus
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_tags_per_frame = max_tags_per_frame
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold

        print(f"[frame_tagger] Loading RAM++ ({ram_checkpoint}, image_size={ram_image_size})...")
        self._ram_transform = get_transform(image_size=ram_image_size)
        self.ram_model = ram_plus(
            pretrained=ram_checkpoint, image_size=ram_image_size, vit="swin_l",
            threshold=ram_tag_threshold,
        )
        self.ram_model.eval().to(self.device)

        print(f"[frame_tagger] Loading GroundingDINO tiny ({gdino_model_id})...")
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_model_id, use_fast=True, local_files_only=True)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model_id, low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device)
        self._gdino_fp16 = self.device.type == "cuda"
        if self._gdino_fp16:
            self.gdino_model.half()

    def _tag_batch(self, rgbs: List[np.ndarray]) -> List[List[str]]:
        batch = torch.stack(
            [self._ram_transform(Image.fromarray(rgb)) for rgb in rgbs]
        ).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda",
        ):
            tag_strs, _tag_strs_chinese = self.ram_model.generate_tag(batch)
        out = []
        for s in tag_strs:
            tags = [t.strip() for t in s.split("|") if t.strip()]
            out.append(tags[: self.max_tags_per_frame])
        return out

    def _detect_batch(
        self, rgbs: List[np.ndarray], prompts: List[str],
    ) -> List[List[TagDetection]]:
        images = [Image.fromarray(rgb) for rgb in rgbs]
        inputs = self.gdino_processor(
            images=images, text=prompts, padding=True, return_tensors="pt",
        ).to(self.device)
        input_ids = inputs["input_ids"]
        if self._gdino_fp16:
            inputs = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self._gdino_fp16 else torch.float32,
            enabled=self.device.type == "cuda",
        ):
            outputs = self.gdino_model(**inputs)

        target_sizes = [(rgb.shape[0], rgb.shape[1]) for rgb in rgbs]
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs, input_ids,
            box_threshold=self.gdino_box_threshold,
            text_threshold=self.gdino_text_threshold,
            target_sizes=target_sizes,
        )

        out = []
        for r in results:
            boxes = r["boxes"].cpu().numpy()
            scores = r["scores"].cpu().numpy()
            labels = r["labels"]
            out.append([
                TagDetection(
                    label=lbl.strip(),
                    box_xyxy=tuple(float(v) for v in box),
                    score=float(sc),
                )
                for box, sc, lbl in zip(boxes, scores, labels)
            ])
        return out

    def tag_and_detect_batch(self, rgbs: List[np.ndarray]) -> List[TagResult]:
        """rgbs: list of HxWx3 uint8 RGB frames, any resolution (each model
        resizes to its own preferred size internally, see module docstring).
        Returns one TagResult per input frame, same order."""
        if not rgbs:
            return []
        tag_lists = self._tag_batch(rgbs)
        prompts = [_build_prompt(tags) for tags in tag_lists]
        detections = self._detect_batch(rgbs, prompts)
        return [
            TagResult(tags=t, prompt=p, detections=d)
            for t, p, d in zip(tag_lists, prompts, detections)
        ]


def draw_detections(rgb: np.ndarray, detections: List[TagDetection]) -> np.ndarray:
    """Draws GroundingDINO boxes + labels (yellow) on top of an already-
    ORB-keypoint-annotated frame from extractor.py's draw_keypoints."""
    import cv2

    vis = rgb.copy()
    for det in detections:
        x0, y0, x1, y1 = (int(round(v)) for v in det.box_xyxy)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 220, 0), 2, lineType=cv2.LINE_AA)
        caption = f"{det.label} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x0, max(0, y0 - th - 6)), (x0 + tw + 4, y0), (255, 220, 0), -1)
        cv2.putText(
            vis, caption, (x0 + 2, max(0, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA,
        )
    return vis
