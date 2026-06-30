"""
Tracking tools: start_tracking, stop_tracking, get_object_from_memory.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any, Dict, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from live_session import LiveAPISession

_DETECTION_THRESHOLD = 0.3
_REFERENCE_DINO_THRESHOLD = 0.6  # min DINO score to capture reference for non-saved objects
_MEMORY_THRESHOLD = 0.5


def _encode_embedding(emb: np.ndarray) -> str:
    """Encode float32 numpy array as base64 string."""
    return base64.b64encode(emb.astype(np.float32).tobytes()).decode()


def _encode_image(image_bgr: np.ndarray) -> Optional[str]:
    """Encode BGR image as base64 JPEG string."""
    ok, jpeg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return base64.b64encode(jpeg.tobytes()).decode()


def _decode_frame(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


async def tool_start_tracking(
    session: "LiveAPISession", target: str, description: str = "", **_
) -> Dict[str, Any]:
    if session.latest_frame is None:
        return {"error": "No frame available. Make sure the camera is active."}
    try:
        frame = _decode_frame(session.latest_frame)
        if frame is None:
            return {"error": "Could not decode frame."}

        dino_query = description or target
        det = await asyncio.to_thread(session.tools.detector.detect, frame, dino_query)
        if det.score < _DETECTION_THRESHOLD:
            return {
                "found": False,
                "target": target,
                "message": f"Could not find '{target}' in the current view.",
            }

        session.state.mode = "tracking"
        session.state.tracking_target = target
        box = [round(v) for v in det.box_xyxy]

        # Reference embedding and image: prefer saved memory, fall back to current detection
        ref_embedding_b64: Optional[str] = None
        ref_image_b64: Optional[str] = None

        if session.tools.object_store is not None:
            saved = await asyncio.to_thread(session.tools.object_store.load, target)
            if saved is not None:
                if saved["embedding"] is not None:
                    ref_embedding_b64 = _encode_embedding(saved["embedding"])
                if saved["image"] is not None:
                    ref_image_b64 = _encode_image(saved["image"])

        # No saved reference: compute from current detection if confident enough
        if ref_embedding_b64 is None and det.score >= _REFERENCE_DINO_THRESHOLD:
            if session.tools.embedder is not None:
                try:
                    emb = await asyncio.to_thread(
                        session.tools.embedder.get_embedding, frame, det.box_xyxy
                    )
                    if emb is not None:
                        ref_embedding_b64 = _encode_embedding(emb.numpy())
                except Exception:
                    pass
            # Crop the detected object as reference image
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (int(max(0, v)) for v in det.box_xyxy)
            x2, y2 = min(x2, w), min(y2, h)
            if x2 > x1 and y2 > y1:
                ref_image_b64 = _encode_image(frame[y1:y2, x1:x2])

        detection = {
            "found": True,
            "target": target,
            "description": dino_query,
            "score": round(det.score, 3),
            "box_xyxy": box,
        }
        session.state.last_detection = detection

        try:
            session.state_update_q.put_nowait({
                "agent_state": "TRACKING",
                "agent_payload": json.dumps({
                    "target": target,
                    "description": dino_query,
                    "box_xyxy": box,
                    "ref_embedding": ref_embedding_b64,
                    "ref_image_b64": ref_image_b64,
                }),
            })
        except Exception:
            pass
        return detection
    except Exception as e:
        return {"error": str(e)}


async def tool_stop_tracking(session: "LiveAPISession", **_) -> Dict[str, Any]:
    target = session.state.tracking_target
    session.state.mode = "idle"
    session.state.tracking_target = ""
    session.state.last_detection = None
    try:
        session.state_update_q.put_nowait({"agent_state": "IDLE", "agent_payload": ""})
    except Exception:
        pass
    return {"status": "stopped", "was_tracking": target}


async def tool_get_object_from_memory(
    session: "LiveAPISession", query: str, **_
) -> Dict[str, Any]:
    try:
        results = []

        # Primary: object_store text search (no ML needed, works with DummyRagStore)
        if session.tools.object_store is not None:
            hits = await asyncio.to_thread(session.tools.object_store.search, query)
            for h in hits:
                if h["score"] >= _MEMORY_THRESHOLD:
                    results.append({
                        "label": h["label"],
                        "description": h["description"],
                        "confidence": round(h["score"], 3),
                    })

        # Secondary: rag_store semantic search (no-op when DummyRagStore)
        if not results:
            hits = await asyncio.to_thread(session.tools.rag_store.query_global, query, 3)
            filtered = [(t, l, s) for t, l, s in hits if s >= _MEMORY_THRESHOLD]
            results += [
                {"label": l, "description": t[:200], "confidence": round(s, 3)}
                for t, l, s in filtered
            ]

        if not results:
            return {"found": False, "query": query, "message": "No matching object found in memory."}
        return {"found": True, "results": results}
    except Exception as e:
        return {"error": str(e)}


HANDLERS = {
    "start_tracking": tool_start_tracking,
    "stop_tracking": tool_stop_tracking,
    "get_object_from_memory": tool_get_object_from_memory,
}
