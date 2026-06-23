import numpy as np
import cv2
import requests
from typing import List


class DocLayoutRapidOCRTool:
    def __init__(self, url: str = "http://localhost:8100"):
        self._url = url
        print(f"[SERVER] OCR client ready → {self._url}")

    def read_blocks(self, frame: np.ndarray, direction: str = "ltr") -> List[str]:
        del direction
        if frame is None:
            return []

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return []

        try:
            resp = requests.post(
                f"{self._url}/ocr",
                files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("blocks", [])
        except Exception as e:
            print(f"[OCR] request failed: {e}")
            return []

    def beat_blocks(self, frame: np.ndarray, direction: str = "ltr") -> list:
        blocks = self.read_blocks(frame, direction)
        if not blocks:
            return []

        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    a, b = blocks[i], blocks[j]
                    ax1, ay1, ax2, ay2 = a["box"]
                    bx1, by1, bx2, by2 = b["box"]
                    x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                    w_a = ax2 - ax1
                    w_b = bx2 - bx1
                    if w_a > 0 and w_b > 0 and (x_overlap / w_a >= 0.7 or x_overlap / w_b >= 0.7):
                        first, second = (a, b) if ay1 <= by1 else (b, a)
                        merged = {
                            "text": first["text"] + " " + second["text"],
                            "box": [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)],
                            "score": round((a.get("score", 1.0) + b.get("score", 1.0)) / 2, 4),
                        }
                        blocks = [blocks[k] for k in range(len(blocks)) if k not in (i, j)]
                        blocks.append(merged)
                        changed = True
                        break
                if changed:
                    break

        reverse = direction == "rtl"
        return sorted(blocks, key=lambda b: b["box"][0], reverse=reverse)

    def read_text(self, frame: np.ndarray, direction: str = "ltr") -> str:
        return " ".join(self.read_blocks(frame, direction))
