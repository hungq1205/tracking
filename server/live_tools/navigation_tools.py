"""
Navigation tools: start_navigation, stop_navigation, get_current_location.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from live_session import LiveAPISession


def _pick_location(maps_root_dir: str, preferred: str = "") -> Optional[str]:
    if preferred:
        candidate = os.path.join(maps_root_dir, preferred, "map_labels.json")
        if os.path.exists(candidate):
            return preferred
    try:
        entries = [
            d for d in os.listdir(maps_root_dir)
            if os.path.isdir(os.path.join(maps_root_dir, d))
            and os.path.exists(os.path.join(maps_root_dir, d, "map_labels.json"))
        ]
    except FileNotFoundError:
        return None
    return entries[0] if len(entries) == 1 else None


def _ensure_map_loaded(session: "LiveAPISession", location_id: str) -> bool:
    if location_id == session._current_location_id:
        return session._route_planner is not None

    map_dir = os.path.join(session.tools.maps_root_dir, location_id)
    labels_path = os.path.join(map_dir, "map_labels.json")
    if not os.path.exists(labels_path):
        print(f"[NavTools] map_labels.json not found: {labels_path}")
        session._route_planner = None
        session._localizer = None
        session._current_location_id = location_id
        return False

    from tools.route_planner import RoutePlanner
    from tools.localization import LocalizationEngine

    with open(labels_path) as f:
        data = json.load(f)
    session._route_planner = RoutePlanner(data.get("zones", []))
    session._localizer = LocalizationEngine(map_dir)
    session._current_location_id = location_id
    print(
        f"[NavTools] Map '{location_id}' loaded: "
        f"{len(session._route_planner.zones)} zones, "
        f"localizer={'ok' if session._localizer.available else 'no keyframes'}."
    )
    return True


async def tool_start_navigation(session: "LiveAPISession", destination: str, **_) -> Dict[str, Any]:
    location_id = _pick_location(session.tools.maps_root_dir, session.state.nav_location_id)
    if location_id is None:
        return {"error": "No map available. A venue must be scanned first."}

    ok = await asyncio.to_thread(_ensure_map_loaded, session, location_id)
    if not ok or session._route_planner is None:
        return {"error": "Failed to load map data."}

    if session._route_planner.find_zone(destination) is None:
        known = [z.label for z in session._route_planner.zones]
        return {
            "error": f"Unknown destination '{destination}'.",
            "known_zones": known,
        }

    # Try to localize start position
    start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if session.latest_frame is not None and session._localizer and session._localizer.available:
        try:
            import cv2
            nparr = np.frombuffer(session.latest_frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                loc = await asyncio.to_thread(session._localizer.localize, frame)
                if loc.position is not None:
                    start_pos = loc.position
                    session.state.nav_last_position = start_pos.tolist()
        except Exception as e:
            print(f"[NavTools] Localization failed: {e}")

    route = await asyncio.to_thread(session._route_planner.compute_route, start_pos, destination)

    import time
    session.state.mode = "guiding"
    session.state.nav_destination = destination
    session.state.nav_location_id = location_id
    session.state.nav_route = [z.label for z in route]
    session.state.nav_route_idx = 0
    session.state.nav_last_localize_at = time.time()
    # Shared walking obstacle state — reset so guiding starts fresh
    session.state.walking_obstacle_cache = []
    session.state.walking_last_detect_at = 0.0
    session.state.walking_last_obstacle_at = 0.0

    announcement = await asyncio.to_thread(
        session._route_planner.route_announcement, route, destination
    )
    try:
        session.state_update_q.put_nowait({
            "agent_state": "GUIDING",
            "agent_payload": json.dumps({
                "destination": destination,
                "route": session.state.nav_route,
            }),
        })
    except Exception:
        pass
    return {
        "status": "navigating",
        "destination": destination,
        "route": session.state.nav_route,
        "announcement": announcement,
    }


async def tool_stop_guiding(session: "LiveAPISession", **_) -> Dict[str, Any]:
    dest = session.state.nav_destination
    session.state.mode = "idle"
    session.state.nav_destination = ""
    session.state.nav_route = []
    session.state.nav_route_idx = 0
    session.state.nav_last_position = None
    session.state.walking_obstacle_cache = []
    try:
        session.state_update_q.put_nowait({"agent_state": "IDLE", "agent_payload": ""})
    except Exception:
        pass
    return {"status": "stopped", "was_guiding_to": dest}


async def tool_get_current_location(session: "LiveAPISession", **_) -> Dict[str, Any]:
    if session._localizer is None:
        return {"error": "No map loaded. Start navigation first."}
    if not session._localizer.available:
        return {"error": "No keyframes in map for localization."}
    if session.latest_frame is None:
        return {"error": "No frame available."}
    try:
        import cv2
        nparr = np.frombuffer(session.latest_frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame."}
        loc = await asyncio.to_thread(session._localizer.localize, frame)
        if loc.position is None:
            return {"localized": False, "message": "Could not determine position."}
        pos = loc.position.tolist()
        session.state.nav_last_position = pos

        # Find nearest zone
        nearest = None
        if session._route_planner:
            for zone in session._route_planner.zones:
                if zone.contains(np.array(pos, dtype=np.float32)):
                    nearest = zone.label
                    break
        return {"localized": True, "position": pos, "zone": nearest}
    except Exception as e:
        return {"error": str(e)}


HANDLERS = {
    "start_guiding": tool_start_navigation,
    "stop_guiding": tool_stop_guiding,
    "get_current_location": tool_get_current_location,
}
