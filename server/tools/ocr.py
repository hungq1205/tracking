import tempfile
import os
import numpy as np
from typing import List

import cv2


class PaddleOCRVLTool:
    def __init__(self):
        from paddleocr import PaddleOCRVL

        print("[SERVER] Initializing PaddleOCR-VL-1.5...")
        self._pipeline = PaddleOCRVL(pipeline_version="v1.5")

    def read_blocks(self, frame: np.ndarray, direction: str = "ltr") -> List[str]:
        """Return one string per layout block, in reading order.

        direction is accepted for API compatibility; PaddleOCR-VL-1.5
        infers reading order automatically from the document structure.
        """
        del direction  # handled by the model internally
        if frame is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            from PIL import Image
            Image.fromarray(rgb).save(tmp_path)

            blocks: List[str] = []
            for res in self._pipeline.predict(tmp_path):
                blocks.extend(_extract_blocks(res))
        finally:
            os.unlink(tmp_path)

        return blocks

    def read_text(self, frame: np.ndarray, direction: str = "ltr") -> str:
        return " ".join(self.read_blocks(frame, direction))


def _extract_blocks(res) -> List[str]:
    """Pull ordered text blocks out of a PaddleOCRVL result object."""
    # res.res is the raw dict from the pipeline
    raw = getattr(res, "res", None)
    if raw is None:
        return []

    blocks: List[str] = []

    # layout_det_res contains blocks sorted in reading order by the pipeline
    for item in raw.get("layout_det_res", {}).get("input_path", []):
        text = (item.get("text") or "").strip()
        if text:
            blocks.append(text)

    # If layout path yields nothing, fall back to the overall OCR text in order
    if not blocks:
        ocr = raw.get("overall_ocr_res", {})
        for item in ocr.get("rec_texts", []):
            text = str(item).strip()
            if text:
                blocks.append(text)

    return blocks
