from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import cv2
import numpy as np


class CloudVLMClient(ABC):
    @abstractmethod
    def query(self, text: str, image: Optional[np.ndarray] = None) -> str:
        """General scene Q&A or description. image is BGR numpy array or None."""

    @abstractmethod
    def describe_obstacle(self, frame: np.ndarray, depth_info: str = "") -> str:
        """Return a short natural-language description of the nearest obstacle."""

    @abstractmethod
    def query_stream(self, text: str, image: Optional[np.ndarray] = None) -> Iterator[bytes]:
        """Stream raw PCM audio chunks (24 kHz, 16-bit LE, mono) from the VLM response."""


class StubVLMClient(CloudVLMClient):
    """Placeholder that returns fixed strings without any API call."""

    def query(self, text: str, image: Optional[np.ndarray] = None) -> str:
        return "[Cloud VLM stub] Scene description not available. Configure CLOUD_VLM_VENDOR."

    def describe_obstacle(self, frame: np.ndarray, depth_info: str = "") -> str:
        return "an obstacle"

    def query_stream(self, text: str, image: Optional[np.ndarray] = None) -> Iterator[bytes]:
        raise NotImplementedError("StubVLMClient does not support audio streaming")


class AnthropicVLMClient(CloudVLMClient):
    """
    Claude vision API client.
    Requires: pip install anthropic
    Env: CLOUD_VLM_API_KEY
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc
        self._client = _anthropic.Anthropic(api_key=api_key or os.getenv("CLOUD_VLM_API_KEY", ""))
        self._model = model

    def query(self, text: str, image: Optional[np.ndarray] = None) -> str:
        content = []
        if image is not None:
            content.append(self._image_block(image))
        content.append({"type": "text", "text": text})
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )
        return msg.content[0].text.strip()

    def describe_obstacle(self, frame: np.ndarray, depth_info: str = "") -> str:
        prompt = (
            "You are assisting a blind user who is walking. "
            "Look at this image and identify the NEAREST obstacle directly ahead. "
            "Reply with a single short phrase (5 words max), e.g. 'a glass door', "
            "'a person standing', 'a step down'. "
        )
        if depth_info:
            prompt += f"Depth sensor reads approximately {depth_info} ahead. "
        return self.query(prompt, frame)

    def query_stream(self, text: str, image: Optional[np.ndarray] = None) -> Iterator[bytes]:
        raise NotImplementedError("AnthropicVLMClient does not support audio streaming; use GeminiVLMClient")

    @staticmethod
    def _image_block(frame_bgr: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


class GeminiVLMClient(CloudVLMClient):
    """
    Gemini Flash VLM client with streaming audio output.
    Requires: pip install google-genai
    Env: GEMINI_API_KEY (or CLOUD_VLM_API_KEY as fallback)
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as exc:
            raise ImportError("Install google-genai: pip install google-genai") from exc
        self._genai = _genai
        self._types = _types
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("CLOUD_VLM_API_KEY", "")
        self._client = _genai.Client(api_key=resolved_key)
        self._model = model

    def query(self, text: str, image: Optional[np.ndarray] = None) -> str:
        contents = self._build_contents(text, image)
        response = self._client.models.generate_content(model=self._model, contents=contents)
        return response.text.strip() if response.text else ""

    def describe_obstacle(self, frame: np.ndarray, depth_info: str = "") -> str:
        prompt = (
            "You are assisting a blind user who is walking. "
            "Look at this image and identify the NEAREST obstacle directly ahead. "
            "Reply with a single short phrase (5 words max), e.g. 'a glass door', "
            "'a person standing', 'a step down'. "
        )
        if depth_info:
            prompt += f"Depth sensor reads approximately {depth_info} ahead. "
        return self.query(prompt, frame)

    def query_stream(self, text: str, image: Optional[np.ndarray] = None) -> Iterator[bytes]:
        """Stream raw PCM audio chunks (24 kHz, 16-bit LE, mono) from Gemini Flash."""
        contents = self._build_contents(text, image)
        response_stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=self._types.GenerateContentConfig(response_modalities=["AUDIO"]),
        )
        for chunk in response_stream:
            if not chunk.candidates:
                continue
            for part in chunk.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    yield part.inline_data.data

    def _build_contents(self, text: str, image: Optional[np.ndarray]) -> list:
        contents = []
        if image is not None:
            contents.append(self._types.Part.from_bytes(
                data=self._encode_image(image),
                mime_type="image/jpeg",
            ))
        contents.append(text)
        return contents

    @staticmethod
    def _encode_image(frame_bgr: np.ndarray) -> bytes:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()


class OpenRouterVLMClient(CloudVLMClient):
    """
    VLM client for OpenRouter API (used for offline semantic mapping).
    Env: OPENROUTER_API_KEY
    Default model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
    """

    DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    _SYSTEM_PROMPT = (
        "You are a scene understanding assistant for a navigation system designed for "
        "vision-impaired users. Your task is NOT to describe every visible object. "
        "Instead, identify only the static landmarks, facilities, and functional objects "
        "that are important for navigation within the specified zone/area. "
        "The generated object names will be passed directly to Grounding DINO for object grounding. "
        "Requirements: Prefer permanent landmarks over movable objects. "
        "Ignore people, animals, decorations, bags, clothing, food, shadows, reflections, "
        "and temporary clutter. Only include objects that are actually visible or can be "
        "confidently inferred from the image. Choose objects that are useful as navigation "
        "landmarks or destinations. Return only JSON."
    )

    def __init__(self, api_key: str = "", model: str = DEFAULT_MODEL):
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model = model

    def query(self, text: str, image: Optional[np.ndarray] = None) -> str:
        import requests as _requests
        content: list
        if image is not None:
            b64 = self._encode_image(image)
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": text},
            ]
        else:
            content = [{"type": "text", "text": text}]

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        resp = _requests.post(
            self.API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def describe_obstacle(self, frame: np.ndarray, depth_info: str = "") -> str:
        prompt = "Describe the nearest obstacle in 5 words or fewer."
        if depth_info:
            prompt += f" Depth sensor reads approximately {depth_info} ahead."
        return self.query(prompt, image=frame)

    def query_stream(self, text: str, image: Optional[np.ndarray] = None) -> Iterator[bytes]:
        raise NotImplementedError("OpenRouterVLMClient does not support audio streaming")

    @staticmethod
    def _encode_image(frame_bgr: np.ndarray) -> str:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode()


def create_cloud_vlm_client(vendor: str = "stub", **kwargs) -> CloudVLMClient:
    if vendor == "anthropic":
        return AnthropicVLMClient(**kwargs)
    if vendor == "gemini":
        return GeminiVLMClient(**kwargs)
    if vendor == "openrouter":
        return OpenRouterVLMClient(**kwargs)
    return StubVLMClient()
