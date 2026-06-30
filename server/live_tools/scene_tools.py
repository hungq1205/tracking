"""
Scene / vision tools: get_latest_frame, start/stop_vision_stream, run_detection, check_obstacle.
"""
from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from live_session import LiveAPISession

from google.genai import types


async def tool_get_latest_frame(session: "LiveAPISession", **_) -> Dict[str, Any]:
    if session.latest_frame is None:
        return {"error": "No frame available yet. Ask the user to point their camera."}
    await session._session.send_realtime_input(
        video=types.Blob(data=session.latest_frame, mime_type="image/jpeg")
    )
    return {"status": "frame_sent", "note": "You can now see the current camera view."}


async def tool_start_vision_stream(session: "LiveAPISession", reason: str = "", **_) -> Dict[str, Any]:
    if session.state.live_vision_active:
        return {"status": "already_active"}
    session.state.live_vision_active = True
    asyncio.get_event_loop().create_task(session._vision_stream_loop())
    return {"status": "started", "duration_s": 15, "reason": reason}


async def tool_stop_vision_stream(session: "LiveAPISession", **_) -> Dict[str, Any]:
    session.state.live_vision_active = False
    return {"status": "stopped"}


async def tool_run_detection(session: "LiveAPISession", object_description: str, **_) -> Dict[str, Any]:
    if session.latest_frame is None:
        return {"error": "No frame available. Call get_latest_frame() first."}
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(session.latest_frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame."}
        det = await asyncio.to_thread(session.tools.detector.detect, frame, object_description)
        if det.score < 0.3:
            return {"found": False, "object": object_description}
        return {
            "found": True,
            "object": object_description,
            "score": round(det.score, 3),
            "box_xyxy": [round(v) for v in det.box_xyxy],
        }
    except Exception as e:
        return {"error": str(e)}


async def tool_check_obstacle(session: "LiveAPISession", **_) -> Dict[str, Any]:
    if session.latest_frame is None:
        return {"error": "No frame available."}
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(session.latest_frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame."}
        is_obstacle, depth = await asyncio.to_thread(session.tools.depth_detector.check_obstacle, frame)
        return {"obstacle": is_obstacle, "depth_m": round(float(depth), 2)}
    except Exception as e:
        return {"error": str(e)}


HANDLERS = {
    "get_latest_frame": tool_get_latest_frame,
    "start_vision_stream": tool_start_vision_stream,
    "stop_vision_stream": tool_stop_vision_stream,
    "run_detection": tool_run_detection,
    "check_obstacle": tool_check_obstacle,
}
