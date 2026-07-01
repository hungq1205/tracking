"""
Memory tools: query_memory, save_memory, remember_object, list_memory_labels.
"""
from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from live_session import LiveAPISession

_MEMORY_THRESHOLD = 0.5
_DETECTION_THRESHOLD = 0.35


async def tool_query_memory(session: "LiveAPISession", question: str, **_) -> Dict[str, Any]:
    try:
        results = []

        # Search object store by label + description (no ML, always works)
        if session.tools.object_store is not None:
            obj_hits = await asyncio.to_thread(session.tools.object_store.search, question)
            for h in obj_hits:
                if h["score"] >= _MEMORY_THRESHOLD:
                    results.append({
                        "label": h["label"],
                        "text": h["description"],
                        "confidence": round(h["score"], 3),
                    })

        # Also search rag_store (no-op when DummyRagStore)
        rag_hits = await asyncio.to_thread(session.tools.rag_store.query_global, question, 3)
        for text, label, score in rag_hits:
            if score >= _MEMORY_THRESHOLD:
                results.append({"label": label, "text": text[:300], "confidence": round(score, 3)})

        if not results:
            return {"found": False, "question": question, "message": "Nothing relevant found in memory."}
        return {"found": True, "results": results}
    except Exception as e:
        return {"error": str(e)}


async def tool_save_memory(session: "LiveAPISession", label: str, note: str, **_) -> Dict[str, Any]:
    try:
        appended, _ = await asyncio.to_thread(
            session.tools.memory_store.append, label, note, "voice"
        )
        if appended:
            await asyncio.to_thread(
                session.tools.rag_store.add_text, label, appended, "voice"
            )
        return {"status": "saved", "label": label, "chars": len(note)}
    except Exception as e:
        return {"error": str(e)}


async def tool_remember_object(session: "LiveAPISession", label: str, description: str, **_) -> Dict[str, Any]:
    if session.latest_frame is None:
        return {"error": "No frame available. Point the camera at the object."}
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(session.latest_frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame."}

        # Detect and crop using the appearance description
        image_to_save = frame
        located = False
        box = None
        try:
            det = await asyncio.to_thread(session.tools.detector.detect, frame, description)
            if det.score > _DETECTION_THRESHOLD:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = (int(max(0, v)) for v in det.box_xyxy)
                x2, y2 = min(x2, w), min(y2, h)
                if x2 > x1 and y2 > y1:
                    image_to_save = frame[y1:y2, x1:x2]
                    box = det.box_xyxy
                    located = True
        except Exception:
            pass

        # Compute DINOv2 embedding of the crop
        embedding = None
        if session.tools.embedder is not None and box is not None:
            try:
                emb_tensor = await asyncio.to_thread(
                    session.tools.embedder.get_embedding, frame, box
                )
                if emb_tensor is not None:
                    embedding = emb_tensor.numpy().astype("float32")
            except Exception:
                pass

        # Save to object store (label, description, crop image, embedding)
        if session.tools.object_store is not None:
            await asyncio.to_thread(
                session.tools.object_store.save, label, description, image_to_save, embedding
            )

        # Also index in rag_store (no-op when DummyRagStore)
        combined_text = f"{label}. {description}"
        await asyncio.to_thread(
            session.tools.rag_store.add_object, label, image_to_save, combined_text
        )

        return {
            "status": "saved",
            "label": label,
            "description": description,
            "located_in_frame": located,
            "message": f"'{label}' has been saved to memory.",
        }
    except Exception as e:
        return {"error": str(e)}


async def tool_list_memory_labels(session: "LiveAPISession", **_) -> Dict[str, Any]:
    try:
        base_dir = session.tools.memory_store.base_dir
        labels = [
            f[:-5] for f in os.listdir(base_dir)
            if f.endswith(".json") and not f.startswith("_")
        ] if os.path.isdir(base_dir) else []
        return {"labels": sorted(labels), "count": len(labels)}
    except Exception as e:
        return {"error": str(e)}


HANDLERS = {
    "query_memory": tool_query_memory,
    "save_memory": tool_save_memory,
    "remember_object": tool_remember_object,
    "list_memory_labels": tool_list_memory_labels,
}
