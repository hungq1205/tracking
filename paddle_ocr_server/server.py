import base64
import logging
from contextlib import asynccontextmanager
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="[OCR] %(message)s")
log = logging.getLogger(__name__)

_ocr = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ocr
    import paddle
    log.info("paddle device: %s", paddle.device.get_device())
    log.info("cuda available: %s", paddle.is_compiled_with_cuda())
    from paddleocr import PaddleOCR
    log.info("loading PaddleOCR (GPU)...")
    _ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device="gpu",
         enable_mkldnn=False,
    )
    log.info("models ready")
    yield


app = FastAPI(lifespan=lifespan)


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def _preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 1800:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _merge_into_paragraphs(blocks: list) -> list:
    if not blocks:
        return blocks

    heights = sorted(b["box"][3] - b["box"][1] for b in blocks)
    line_h = max(heights[len(heights) // 2], 1)

    def should_merge(a, b):
        ax1, ay1, ax2, ay2 = a["box"]
        bx1, by1, bx2, by2 = b["box"]
        v_gap = max(by1 - ay2, ay1 - by2)
        h_overlap = ax1 < bx2 and bx1 < ax2
        return v_gap <= line_h and h_overlap

    def do_merge(a, b):
        ax1, ay1, ax2, ay2 = a["box"]
        bx1, by1, bx2, by2 = b["box"]
        first, second = (a, b) if ay1 <= by1 else (b, a)
        return {
            "text": first["text"] + " " + second["text"],
            "box": [min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)],
            "score": round((a["score"] + b["score"]) / 2, 4),
        }

    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                if should_merge(blocks[i], blocks[j]):
                    merged = do_merge(blocks[i], blocks[j])
                    blocks = [b for k, b in enumerate(blocks) if k not in (i, j)]
                    blocks.append(merged)
                    changed = True
                    break
            if changed:
                break

    return sorted(blocks, key=lambda b: b["box"][1])


@app.post("/ocr")
async def ocr(image: UploadFile = File(...)):
    log.info("received image: %s (%s)", image.filename, image.content_type)
    data = await image.read()
    log.info("image size: %d bytes", len(data))
    try:
        img = _decode_image(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    preprocessed = _preprocess(img)
    results = list(_ocr.predict(preprocessed))

    # Extract the orientation/warp-corrected image that boxes reference
    processed_img = preprocessed
    for item in results:
        doc_prep = _get(item, "doc_preprocessor_res")
        if doc_prep is not None:
            out = _get(doc_prep, "output_img")
            if out is not None:
                processed_img = out
                break

    _, buf = cv2.imencode(".png", processed_img)
    img_b64 = base64.b64encode(buf.tobytes()).decode()

    blocks = []
    for item in results:
        texts = _get(item, "rec_texts") or []
        boxes = _get(item, "rec_boxes")
        scores = _get(item, "rec_scores") or []
        if boxes is None:
            continue
        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()
        for text, box, score in zip(texts, boxes, scores):
            text = text.strip()
            if not text:
                continue
            blocks.append({"text": text, "box": [round(v) for v in box], "score": round(float(score), 4)})

    blocks = _merge_into_paragraphs(blocks)
    log.info("returning %d blocks", len(blocks))
    return JSONResponse({"blocks": blocks, "image": img_b64})
