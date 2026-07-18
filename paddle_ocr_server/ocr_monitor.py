"""
OcrMonitor — thread-safe snapshot of the most recent /ocr request's pipeline
stages, fed by server.py's ocr() handler, polled by ocr_gui.py.

Single-slot ("last request only"), not a rolling log — a debug view of "what
did the OCR pipeline just do", mirroring the pattern server/services/
activity_monitor.py uses for the main gRPC server's dashboard.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np


class OcrMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot: Dict[str, Any] = {}

    def record(
        self,
        *,
        filename: str,
        image_bytes: int,
        original_rgb: Optional[np.ndarray],
        preprocessed_rgb: Optional[np.ndarray],
        processed_rgb: Optional[np.ndarray],
        raw_blocks: list,
        merged_blocks: list,
        final_text: str,
        error: str = "",
    ) -> None:
        with self._lock:
            self._snapshot = {
                "at": time.time(),
                "filename": filename,
                "image_bytes": image_bytes,
                "original_rgb": original_rgb,
                "preprocessed_rgb": preprocessed_rgb,
                "processed_rgb": processed_rgb,
                "raw_blocks": raw_blocks,
                "merged_blocks": merged_blocks,
                "final_text": final_text,
                "error": error,
            }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._snapshot)
