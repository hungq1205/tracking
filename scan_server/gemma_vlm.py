"""
GemmaVLMClient — Gemma 4 (31B) via the Gemini API (google-genai SDK).

Replaces the local Qwen3-VL client (qwen_vlm.py, kept for reference — its
2B model was prone to greedy-decoding repetition loops in the
grounding_dino_prompt output) for SemanticMapper's zone/area-aware
landmark-prompt generation. Unlike Qwen3VLClient, this takes multiple images
per call — SemanticMapper batches several sampled keyframes into one prompt
for multi-view context instead of judging landmarks off a single frame.
"""

import os
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image


class GemmaVLMClient:
    """
    Gemma 4 (31B) via Google's Gemini API. Implements the multi-image
    interface SemanticMapper expects: query(prompt, images=[...]) -> str.
    """

    DEFAULT_MODEL = "gemma-4-31b-it"

    def __init__(self, model_id: str = DEFAULT_MODEL, api_key: str = "") -> None:
        from google import genai

        resolved_key = api_key
        self._client = genai.Client(api_key=resolved_key) if resolved_key else genai.Client()
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def set_model_id(self, model_id: str) -> None:
        """Swap the model used for future query() calls — no client/model
        reload needed, this is just an API request parameter. Lets the Scan
        UI change models (e.g. while testing different Gemma sizes) without
        restarting scan_server.py."""
        self._model_id = model_id

    def query(self, prompt: str, images: List[np.ndarray]) -> str:
        from google.genai import types

        pil_images = [self._to_pil(img) for img in images]
        contents = [*pil_images, prompt]

        response = self._client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a direct, high-speed visual scene analysis assistant "
                    "for a navigation system. Give immediate, concise answers."
                ),
                # Deterministic JSON extraction, not creative generation — unlike
                # generic demo usage of this model, we want the same landmark
                # list every time for the same scene, not sampling variety.
                temperature=0.0,
                top_p=0.95,
                top_k=64,
            ),
        )
        return response.text.strip() if response.text else ""

    @staticmethod
    def _to_pil(frame_bgr: np.ndarray) -> Image.Image:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
