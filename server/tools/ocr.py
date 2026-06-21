import numpy as np
from typing import List

import cv2


class DocLayoutRapidOCRTool:
    def __init__(self):
        # from doclayout_yolo import YOLOv10
        # from rapidocr_onnxruntime import RapidOCR
        from huggingface_hub import hf_hub_download

        print("[SERVER] Initializing DocLayout-YOLO + RapidOCR...")
        # model_path = hf_hub_download(
        #     repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        #     filename="doclayout_yolo_docstructbench_imgsz1024.pt",
        # )
        # self._layout = YOLOv10(model_path)
        # self._ocr = RapidOCR()
        print("[SERVER] DocLayout-YOLO + RapidOCR ready.")

    def read_blocks(self, frame: np.ndarray, direction: str = "ltr") -> List[str]:
        """Return one string per layout block, in reading order (top-to-bottom, left-to-right)."""
        del direction
        if frame is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        results = self._layout.predict(rgb, imgsz=1024, conf=0.2, verbose=False)
        raw_boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes else []
        boxes = sorted(raw_boxes, key=lambda b: (b[1], b[0]))

        blocks: List[str] = []
        for box in boxes:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            ocr_result, _ = self._ocr(rgb[y1:y2, x1:x2])
            if not ocr_result:
                continue

            text = " ".join(line[1] for line in ocr_result if line[1]).strip()
            if text:
                blocks.append(text)

        if not blocks:
            ocr_result, _ = self._ocr(rgb)
            if ocr_result:
                blocks = [line[1] for line in ocr_result if line[1] and line[1].strip()]

        return blocks

    def read_text(self, frame: np.ndarray, direction: str = "ltr") -> str:
        return " ".join(self.read_blocks(frame, direction))
