"""
Reading tools: enter/exit reading mode, scan_current_view, get_reading_section,
flip_reading_direction.
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, Dict, List

import requests

if TYPE_CHECKING:
    from live_session import LiveAPISession

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_CHUNK_SIZE = 500  # chars per semantic chunk for get_reading_section


def _split_chunks(text: str, size: int = _CHUNK_SIZE) -> List[str]:
    sentences = _SENT_RE.split(text)
    chunks, current = [], []
    cur_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) > size and current:
            chunks.append(" ".join(current))
            current, cur_len = [], 0
        current.append(s)
        cur_len += len(s) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks or [text[:size]]


async def tool_enter_reading_mode(session: "LiveAPISession", label: str = "", **_) -> Dict[str, Any]:
    session.state.mode = "reading"
    session.state.reading_buffer = ""
    session.state.page_summaries = []
    session.state.reading_label = label or "reading"
    session.state.reading_direction = "ltr"
    session.state.last_ocr_at = 0.0
    return {
        "status": "reading_mode_active",
        "label": session.state.reading_label,
        "instruction": (
            "Reading mode is now active. The camera will passively accumulate text as the user "
            "moves the device over the document. Call scan_current_view() for an explicit capture. "
            "Use get_reading_section(query) to answer questions about scanned content."
        ),
    }


async def tool_scan_current_view(session: "LiveAPISession", **_) -> Dict[str, Any]:
    if session.latest_frame is None:
        return {"error": "No frame available. Point the camera at the text."}
    try:
        import cv2
        import numpy as np
        from tools.memory_store import filter_new_sentences

        nparr = np.frombuffer(session.latest_frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame."}

        try:
            text = await asyncio.to_thread(
                session.tools.ocr.read_text, frame, session.state.reading_direction
            )
        except requests.exceptions.ConnectionError as e:
            print(f"[OCR] connection failed: {e}")
            return {"error": "OCR service is unreachable. Text reading is unavailable right now."}
        except requests.exceptions.RequestException as e:
            print(f"[OCR] request failed: {e}")
            return {"error": f"OCR request failed: {e}"}

        if not text:
            return {"found": False, "message": "No text detected in current view."}

        new_text = filter_new_sentences(text, session.state.reading_buffer)
        if not new_text:
            return {"found": False, "message": "No new text (already scanned)."}

        session.state.reading_buffer = f"{session.state.reading_buffer}\n{new_text}".strip()

        # Build a short summary for this page
        words = new_text.split()
        summary = " ".join(words[:30]) + ("..." if len(words) > 30 else "")
        session.state.page_summaries.append(summary)

        return {
            "found": True,
            "new_text": new_text,
            "page_count": len(session.state.page_summaries),
            "total_chars": len(session.state.reading_buffer),
            "summary": summary,
        }
    except Exception as e:
        return {"error": str(e)}


async def tool_get_reading_section(session: "LiveAPISession", query: str, **_) -> Dict[str, Any]:
    if not session.state.reading_buffer:
        return {"error": "No text has been scanned yet. Use scan_current_view() first."}

    # Fast path: if buffer is small, return it directly
    if len(session.state.reading_buffer) < 1200:
        return {"text": session.state.reading_buffer, "source": "full_buffer"}

    # Semantic search over chunks
    try:
        chunks = _split_chunks(session.state.reading_buffer)
        results = await asyncio.to_thread(
            session.tools.rag_store.query_global, query, 3
        )
        # rag_store searches persisted memory; for in-session buffer do simple keyword fallback
        # Filter chunks containing query keywords
        q_lower = query.lower()
        keywords = [w for w in q_lower.split() if len(w) > 3]
        scored = []
        for chunk in chunks:
            c_lower = chunk.lower()
            score = sum(1 for kw in keywords if kw in c_lower)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(reverse=True)
        if scored:
            return {"text": "\n\n".join(c for _, c in scored[:2]), "source": "keyword_match"}

        # Fallback: first chunk
        return {"text": chunks[0] if chunks else "", "source": "first_chunk"}
    except Exception as e:
        return {"error": str(e)}


async def tool_read_aloud(session: "LiveAPISession", scope: str = "new", **_) -> Dict[str, Any]:
    if scope == "all":
        text = session.state.reading_buffer
        if not text:
            return {"error": "No text has been scanned yet."}
    else:
        scan_result = await tool_scan_current_view(session)
        if scan_result.get("error") or not scan_result.get("found"):
            return scan_result
        text = scan_result["new_text"]

    if session.tools.tts is None:
        return {"error": "Text-to-speech engine unavailable."}

    await asyncio.to_thread(session._speak_local, text)
    return {"status": "read_aloud", "scope": scope, "chars": len(text)}


async def tool_flip_reading_direction(session: "LiveAPISession", **_) -> Dict[str, Any]:
    session.state.reading_direction = "rtl" if session.state.reading_direction == "ltr" else "ltr"
    label = "right to left" if session.state.reading_direction == "rtl" else "left to right"
    return {"direction": session.state.reading_direction, "label": label}


async def tool_exit_reading_mode(session: "LiveAPISession", **_) -> Dict[str, Any]:
    label = session.state.reading_label
    char_count = len(session.state.reading_buffer)
    session.state.mode = "idle"
    session.state.reading_buffer = ""
    session.state.page_summaries = []
    session.state.reading_label = ""
    return {"status": "exited", "label": label, "chars_discarded": char_count}


HANDLERS = {
    "enter_reading_mode": tool_enter_reading_mode,
    "scan_current_view": tool_scan_current_view,
    "get_reading_section": tool_get_reading_section,
    "read_aloud": tool_read_aloud,
    "flip_reading_direction": tool_flip_reading_direction,
    "exit_reading_mode": tool_exit_reading_mode,
}
