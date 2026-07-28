"""
Gemini (gemini-3.1-flash-lite, open-set image tagging via the google-genai
SDK) -> GroundingDINO tiny (open-vocab detection), run on every frame
extractor.py accepts as "new". Gemini proposes what's actually in a given
frame with no fixed vocabulary needed up front; those tags become
GroundingDINO's text prompt for that same frame, so what gets boxed tracks
what Gemini actually saw instead of one hardcoded prompt for every frame.

Supersedes an earlier RAM++-based tagging step (open-set image tagging model,
~3GB checkpoint) — replaced because a RAM++ checkpoint isn't available in
every deployment of this pipeline (notably scan_server/'s SemanticMapper,
see that module's docstring); Gemini needs only an API key, no local
checkpoint. GroundingDINO tiny detection is UNCHANGED — still a local
transformers model, still prompted by whatever tags the tagging step
returns.

GroundingDINO is loaded once (FrameTagger is meant to be constructed a
single time and reused) and run BATCHED across every frame accepted in one
extractor.py flush, mirroring the DA3 depth batching already there
(_flush_rtabmap_batch) — batching amortizes per-call Python/CUDA-launch
overhead across the batch instead of paying it once per frame. Gemini
tagging is also batched into ONE multi-image API call per flush (same
multi-image-per-call convention gemma_vlm.py's GemmaVLMClient already
established for this codebase), not one call per frame.

VRAM/speed:
  - GroundingDINO: full `.half()` weights + fp16 autocast — the same,
    already-proven approach server/tools/detector.py uses for this exact
    model.
  - `torch.inference_mode()` (cheaper than `no_grad` — skips autograd
    version-counter bookkeeping entirely) around every GroundingDINO
    forward pass.
  - Every frame handed to GroundingDINO tiny is resized to its processor's
    own shortest_edge=800/longest_edge=1333 config (aspect-ratio-preserving,
    the checkpoint's own preferred input size), not native resolution.
    Gemini receives full-resolution PIL images directly — no local resize,
    same convention gemma_vlm.py's GemmaVLMClient uses.
"""
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from PIL import Image

# Multi-image batched tagging prompt — mirrors scan_server/semantic_mapper.py's
# earlier VLM tag prompt (before that module switched to this shared
# pipeline), adapted for open-vocab-DETECTION-prompt use rather than
# navigation-landmark use specifically.
_TAG_PROMPT_TEMPLATE = """\
You are analyzing {n} images, numbered 1 to {n} in order.

For EACH image, list the distinct LANDMARK objects visible in it that would \
help someone navigate the space or find something later — furniture, \
fixtures, appliances, signage, and large/distinctive containers (e.g. \
"chair", "desk", "tv", "bookshelf", "exit sign"). Tags can include brand \
names, colors, materials, or other descriptors if they are clearly visible \
(e.g. "red chair", "wooden desk", "samsung tv").

Do NOT tag: room surfaces (floor, wall, ceiling, door frame), loose or \
disposable items (papers, clothes, trash, cables), or anything too small \
or generic to be a useful landmark on its own. Only tag things a person \
could actually walk toward or reference as a fixed point in the room.

Respond with EXACTLY {n} lines, one per image, in the same order as the \
images. Each line must be a comma-separated list of short lowercase object \
names for that image only — no numbering, no extra commentary, no markdown. \
If an image has no notable landmarks, output an empty line for it.

Example (for 3 images):
red chair, wooden desk, floor lamp, computer moniter,
samsung tv, brown leather sofa, glass coffee table
exit sign, metal door

"""


def _parse_tag_response(text: str, n: int, max_tags: int) -> List[List[str]]:
    """Defensive parse: split on newlines, pad/truncate to exactly `n` lines
    if Gemini didn't follow the requested format, then split each line on
    commas. Never raises — worst case returns n empty lists."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("```")
        ).strip()

    lines = text.split("\n") if text else []
    if len(lines) < n:
        lines = lines + [""] * (n - len(lines))
    elif len(lines) > n:
        lines = lines[:n]

    return [
        [t.strip().lower() for t in line.split(",") if t.strip()][:max_tags]
        for line in lines
    ]


def _with_article(tag: str) -> str:
    article = "an" if tag[:1] in "aeiou" else "a"
    return f"{article} {tag}"


def _build_prompt(tags: List[str]) -> str:
    # GroundingDINO's own convention: lowercase, article-prefixed, " . "-separated,
    # trailing " .", e.g. "a chair . a table . an apple ."
    if not tags:
        return "an object ."
    return " . ".join(_with_article(tag.lower()) for tag in tags) + " ."


def _strip_leading_article(label: str) -> str:
    """post_process_grounded_object_detection's returned `label` is the
    matched TEXT SPAN decoded straight out of the prompt tokens (see
    _build_prompt's article-prefixed convention above) — so a detection for
    "a nightstand" comes back with the label literally "a nightstand", not
    "nightstand". Strips that leading "a "/"an " back off so Landmark.name
    (and anything downstream — resolve_landmark's substring match, spoken
    navigation destinations) deals with the plain noun, not the prompt
    artifact."""
    for article in ("an ", "a "):
        if label.lower().startswith(article):
            return label[len(article):]
    return label


@dataclass
class TagDetection:
    label: str
    box_xyxy: tuple
    score: float


@dataclass
class TagResult:
    tags: List[str]              # Gemini-proposed tags for this frame, e.g. ["chair", "table", "door"]
    prompt: str                  # the exact dot-separated GroundingDINO text prompt built from `tags`
    detections: List[TagDetection]  # GroundingDINO boxes, prompted by the tags above


class FrameTagger:
    """Gemini tagging -> GroundingDINO-tiny detection, batched across a list
    of frames. Construct once (loads the GroundingDINO model onto `device`
    and a Gemini API client) and reuse across every accepted-frame batch."""

    DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

    def __init__(
        self,
        gemini_api_key: str = "",
        gemini_model_id: str = DEFAULT_GEMINI_MODEL,
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        gdino_box_threshold: float = 0.3,
        gdino_text_threshold: float = 0.25,
        max_tags_per_frame: int = 12,
    ):
        from google import genai
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_tags_per_frame = max_tags_per_frame
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold

        print(f"[frame_tagger] Using Gemini '{gemini_model_id}' for tagging...")
        self._gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else genai.Client()
        self._gemini_model_id = gemini_model_id

        # A one-time, unavoidable ~30s cost lives inside the next call: this
        # model's custom CUDA deformable-attention kernel fails to JIT-
        # compile against this environment's torch/CUDA combination (a real,
        # already-known incompatibility — the vendored kernel source uses
        # Tensor.type(), removed/changed in newer PyTorch) and falls back to
        # the pure-PyTorch implementation. Not fixed here (patching vendored
        # transformers kernel source is out of scope) — see CLAUDE.md's
        # Frame Extractor section for the accepted tradeoff. A STALE lock
        # under ~/.cache/torch_extensions/*/MultiScaleDeformableAttention/
        # left behind by a previously killed process can make this HANG
        # indefinitely instead of just taking ~30s and failing — if this
        # call never returns, check for and remove that lock directory
        # before assuming there's a new bug here.
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
        """One batched Gemini call across every frame in this flush — same
        multi-image-per-call convention gemma_vlm.py's GemmaVLMClient uses
        elsewhere in this codebase."""
        from google.genai import types

        n = len(rgbs)
        prompt = _TAG_PROMPT_TEMPLATE.format(n=n)
        contents = [*(Image.fromarray(rgb) for rgb in rgbs), prompt]

        print(f"[frame_tagger] Sending {n} image(s) to Gemini '{self._gemini_model_id}' — prompt:\n{prompt}")

        response = self._gemini_client.models.generate_content(
            model=self._gemini_model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a direct, high-speed visual scene analysis assistant "
                    "for an open-vocabulary object detection pipeline. Follow the "
                    "requested output format exactly."
                ),
                temperature=0.1,
                response_mime_type="text/plain",
            ),
        )
        text = response.text.strip() if response.text else ""
        return _parse_tag_response(text, n, self.max_tags_per_frame)

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
                    label=_strip_leading_article(lbl.strip()),
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
