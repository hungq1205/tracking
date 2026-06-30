"""
Walking tools: start_walking, stop_walking, quick_label_obstacle.

Walking mode does continuous free-walk pathway obstacle detection using
GroundingDINO + DA3 ONNX depth — no map required. Ticks are driven by
LiveAPISession._walking_tick at the rate set by WalkingConfig.detection_interval.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from live_session import LiveAPISession


async def tool_start_walking(session: "LiveAPISession", **_) -> Dict[str, Any]:
    if session.tools.da3_onnx is None:
        return {"error": "Walking mode unavailable: DA3 ONNX model not loaded (set DA3_ONNX_PATH)."}
    session.state.mode = "guiding"
    session.state.walking_obstacle_cache = []
    session.state.walking_last_detect_at = 0.0
    session.state.walking_last_obstacle_at = 0.0
    try:
        session.state_update_q.put_nowait({"agent_state": "WALKING", "agent_payload": ""})
    except Exception:
        pass
    cfg = session.tools.walking_config
    return {
        "status": "walking",
        "detection_interval_s": cfg.detection_interval if cfg else 2.0,
        "depth_threshold_m": cfg.depth_threshold_m if cfg else 3.0,
    }


async def tool_stop_walking(session: "LiveAPISession", **_) -> Dict[str, Any]:
    session.state.mode = "idle"
    session.state.walking_obstacle_cache = []
    try:
        session.state_update_q.put_nowait({"agent_state": "IDLE", "agent_payload": ""})
    except Exception:
        pass
    return {"status": "stopped"}


async def tool_quick_label_obstacle(session: "LiveAPISession", label: str, **_) -> Dict[str, Any]:
    _TTL = 6.0
    session.state.walking_obstacle_cache.append({
        "label": label,
        "expires_at": time.time() + _TTL,
    })
    return {"stored": label, "ttl_s": _TTL}


HANDLERS = {
    "start_walking": tool_start_walking,
    "stop_walking": tool_stop_walking,
    "quick_label_obstacle": tool_quick_label_obstacle,
}
