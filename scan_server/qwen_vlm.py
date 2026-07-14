"""
Qwen3VLClient — local Qwen3-VL inference via vLLM.

Replaces the OpenRouter cloud VLM client for SemanticMapper's zone/area-aware
landmark-prompt generation (see semantic_mapper.py's SemanticMapper.extract_landmarks,
which calls `self._vlm.query(prompt, image=frame_bgr)`). Runs entirely locally —
no API key needed.
"""

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class Qwen3VLClient:
    """
    Local Qwen3-VL client backed by vLLM. Implements the same interface
    SemanticMapper expects: query(prompt, image=<BGR np.ndarray>) -> str.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Thinking-FP8",
        gpu_memory_utilization: float = 0.70,
        max_new_tokens: int = 1024,
        max_model_len: int = 8192,
    ) -> None:
        import torch
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._llm = LLM(
            model=model_id,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,
            tensor_parallel_size=torch.cuda.device_count(),
            # Model default (262144) needs ~28 GiB of KV cache; a single
            # image + short prompt per call needs nowhere near that.
            max_model_len=max_model_len,
            seed=0,
        )
        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            top_k=-1,
            stop_token_ids=[],
        )

    def query(self, prompt: str, image: Optional[np.ndarray] = None) -> str:
        from qwen_vl_utils import process_vision_info

        content = []
        if image is not None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            content.append({"type": "image", "image": Image.fromarray(rgb)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        vllm_input = {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        outputs = self._llm.generate([vllm_input], sampling_params=self._sampling_params)
        return outputs[0].outputs[0].text
