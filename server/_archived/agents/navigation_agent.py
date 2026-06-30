from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import numpy as np

from agents.base import AgentRequest, AgentResult, BaseAgent
from domain.intents import Intent
from tools.localization import LocalizationEngine
from tools.route_planner import RoutePlanner, Zone

LOCALIZE_INTERVAL = 2.0     # seconds between full PnP localization attempts
DEPTH_CHECK_INTERVAL = 0.5  # seconds between depth/obstacle checks
OBSTACLE_COOLDOWN = 15.0    # seconds before re-alerting the same obstacle area


class NavigationAgent(BaseAgent):
    name = "navigation"

    def __init__(self, depth_detector, cloud_vlm, maps_root_dir: str):
        self.depth_detector = depth_detector
        self.cloud_vlm = cloud_vlm
        self.maps_root_dir = maps_root_dir

        self._localizer: Optional[LocalizationEngine] = None
        self._route_planner: Optional[RoutePlanner] = None
        self._current_location_id: Optional[str] = None

    # ---------------------------------------------------------------- public

    def handle(self, request: AgentRequest) -> AgentResult:
        if request.frame_tick:
            return self._handle_frame_tick(request)

        intent = request.intent.intent
        if intent in (Intent.START_NAVIGATION, Intent.SET_DESTINATION):
            return self._start_navigation(request)
        if intent == Intent.STOP_NAVIGATION:
            return self._stop_navigation(request)

        return AgentResult(agent_name=self.name, state="IDLE", reply_text="", speak=False)

    # ---------------------------------------------------------------- private: startup

    def _start_navigation(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        dest_label = (request.intent.target or "").strip()
        if not dest_label:
            return AgentResult(
                agent_name=self.name, state="ERROR",
                reply_text="Please say where you want to go.", speak=True,
            )

        location_id = self._pick_location(ctx)
        if location_id is None:
            return AgentResult(
                agent_name=self.name, state="ERROR",
                reply_text="No map is available for navigation. Please scan a venue first.",
                speak=True,
            )

        self._ensure_map_loaded(location_id)

        if self._route_planner is None:
            return AgentResult(
                agent_name=self.name, state="ERROR",
                reply_text="Could not load map data.", speak=True,
            )

        if self._route_planner.find_zone(dest_label) is None:
            known = ", ".join(z.label for z in self._route_planner.zones)
            return AgentResult(
                agent_name=self.name, state="ERROR",
                reply_text=f"I don't know where '{dest_label}' is. Known places: {known or 'none'}.",
                speak=True,
            )

        # Try to localize start position
        start_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if request.frame is not None and self._localizer is not None and self._localizer.available:
            loc = self._localizer.localize(request.frame)
            if loc.position is not None:
                start_pos = loc.position
                ctx.nav_last_position = start_pos.tolist()

        route: List[Zone] = self._route_planner.compute_route(start_pos, dest_label)

        # Persist route in context
        ctx.nav_state = "navigating"
        ctx.nav_destination = dest_label
        ctx.nav_location_id = location_id
        ctx.nav_route = [z.label for z in route]
        ctx.nav_route_idx = 0
        ctx.nav_last_obstacle_at = 0.0
        ctx.nav_last_depth_check_at = 0.0
        ctx.nav_last_localize_at = time.time()

        announcement = self._route_planner.route_announcement(route, dest_label)
        return AgentResult(
            agent_name=self.name,
            state="STARTED",
            payload={"destination": dest_label, "route": ctx.nav_route},
            reply_text=announcement,
            speak=True,
        )

    def _stop_navigation(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        ctx.nav_state = "idle"
        ctx.nav_destination = None
        ctx.nav_route = []
        ctx.nav_route_idx = 0
        ctx.nav_last_position = None
        return AgentResult(
            agent_name=self.name, state="STOPPED",
            reply_text="Navigation stopped.", speak=True,
        )

    # ---------------------------------------------------------------- private: frame tick

    def _handle_frame_tick(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        if ctx.nav_state != "navigating":
            return AgentResult(agent_name=self.name, state="IDLE", reply_text="", speak=False)

        frame = request.frame
        now = time.time()

        # 1. Periodically re-localize
        if frame is not None and self._localizer is not None:
            if now - ctx.nav_last_localize_at >= LOCALIZE_INTERVAL:
                loc = self._localizer.localize(frame)
                if loc.position is not None:
                    ctx.nav_last_position = loc.position.tolist()
                ctx.nav_last_localize_at = now

        # 2. Proximity / waypoint check (only if we have a known position)
        if ctx.nav_last_position is not None and self._route_planner is not None:
            pos = np.array(ctx.nav_last_position, dtype=np.float32)
            proximity_result = self._check_proximity(ctx, pos)
            if proximity_result is not None:
                return proximity_result

        # 3. Obstacle detection (throttled)
        if frame is not None and now - ctx.nav_last_depth_check_at >= DEPTH_CHECK_INTERVAL:
            ctx.nav_last_depth_check_at = now
            obstacle_result = self._check_obstacle(ctx, frame, now)
            if obstacle_result is not None:
                return obstacle_result

        return AgentResult(agent_name=self.name, state="NAVIGATING", reply_text="", speak=False)

    def _check_proximity(self, ctx, pos: np.ndarray) -> Optional[AgentResult]:
        route = ctx.nav_route
        if not route:
            return None

        # Check if arrived at destination
        dest_label = ctx.nav_destination
        dest_zone = self._route_planner.find_zone(dest_label) if dest_label else None
        if dest_zone and dest_zone.contains(pos):
            ctx.nav_state = "destination_reached"
            return AgentResult(
                agent_name=self.name,
                state="DESTINATION_REACHED",
                payload={"destination": dest_label},
                reply_text=f"You have arrived at {dest_label}.",
                speak=True,
            )

        # Check if passed through current intermediate waypoint
        idx = ctx.nav_route_idx
        if idx < len(route) - 1:
            wp_label = route[idx]
            wp_zone = self._route_planner.find_zone(wp_label)
            if wp_zone and wp_zone.contains(pos):
                ctx.nav_route_idx += 1
                next_label = route[ctx.nav_route_idx]
                return AgentResult(
                    agent_name=self.name,
                    state="WAYPOINT_REACHED",
                    payload={"waypoint": wp_label, "next": next_label},
                    reply_text=f"Passed {wp_label}. Heading to {next_label}.",
                    speak=True,
                )

        return None

    def _check_obstacle(self, ctx, frame: np.ndarray, now: float) -> Optional[AgentResult]:
        try:
            is_obstacle, min_depth = self.depth_detector.check_obstacle(frame)
        except Exception as e:
            print(f"[NavigationAgent] Depth check failed: {e}")
            return None

        if not is_obstacle:
            return None
        if now - ctx.nav_last_obstacle_at < OBSTACLE_COOLDOWN:
            return None

        depth_info = "very close" if min_depth < 0.15 else "nearby"
        try:
            description = self.cloud_vlm.describe_obstacle(frame, depth_info)
        except Exception as e:
            print(f"[NavigationAgent] Cloud VLM obstacle description failed: {e}")
            description = "an obstacle"

        ctx.nav_last_obstacle_at = now
        return AgentResult(
            agent_name=self.name,
            state="OBSTACLE_DETECTED",
            payload={"depth_m": min_depth, "description": description},
            reply_text=f"Caution: {description} ahead.",
            speak=True,
        )

    # ---------------------------------------------------------------- private: map loading

    def _pick_location(self, ctx) -> Optional[str]:
        if ctx.nav_location_id:
            return ctx.nav_location_id
        try:
            entries = [
                d for d in os.listdir(self.maps_root_dir)
                if os.path.isdir(os.path.join(self.maps_root_dir, d))
                and os.path.exists(os.path.join(self.maps_root_dir, d, "map_labels.json"))
            ]
        except FileNotFoundError:
            return None
        return entries[0] if len(entries) == 1 else None

    def _ensure_map_loaded(self, location_id: str) -> None:
        if location_id == self._current_location_id:
            return
        map_dir = os.path.join(self.maps_root_dir, location_id)
        labels_path = os.path.join(map_dir, "map_labels.json")
        if not os.path.exists(labels_path):
            print(f"[NavigationAgent] map_labels.json not found at {labels_path}")
            self._route_planner = None
            self._localizer = None
            self._current_location_id = location_id
            return

        with open(labels_path) as f:
            data = json.load(f)
        self._route_planner = RoutePlanner(data.get("zones", []))
        self._localizer = LocalizationEngine(map_dir)
        self._current_location_id = location_id
        print(
            f"[NavigationAgent] Loaded map '{location_id}': "
            f"{len(self._route_planner.zones)} zones, "
            f"localizer={'available' if self._localizer.available else 'no keyframes'}."
        )
