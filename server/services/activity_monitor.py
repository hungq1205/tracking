"""
ActivityMonitor — thread-safe "what did the server last receive" snapshot,
fed by each servicer as real client RPCs land, polled by server_gui.py.

There is no explicit client "mode" on the wire any more (no server-side
LiveSessionState — see CLAUDE.md's "Client-Orchestrated Live Session"
section): the server only ever sees individual RPCs. So instead of a mode
field, this tracks one snapshot per RPC category (tracking/perception/
mapping) plus which category most recently updated — server_gui.py uses
that to auto-select which tab to show, which is the closest equivalent to
"the server knows this is tracking activity and switches to it" without
inventing a mode concept the wire protocol doesn't have.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional


class ActivityMonitor:
    def __init__(self, max_log: int = 200):
        self._lock = threading.Lock()
        self.log: Deque[Dict[str, Any]] = deque(maxlen=max_log)
        self.tracking: Dict[str, Any] = {}
        self.perception: Dict[str, Any] = {}
        self.mapping: Dict[str, Any] = {}
        self.last_category: str = ""
        self.last_at: float = 0.0
        # Explicit client-reported mode (StatusService.ReportMode) — when
        # present, server_gui.py prefers this over last_category for tab
        # selection, since it's the client's own ground truth rather than
        # an inference from whichever RPC most recently happened to fire.
        self.client_mode: str = ""
        self.client_mode_target: str = ""
        self.client_mode_at: float = 0.0
        # Explicit client-reported HRTF beacon azimuth (StatusService.
        # ReportBeaconDirection) — the actual final steering angle after
        # goal-bias + EMA smoothing, all computed client-side, so the
        # server has no other way to know it. Dashboard-only, like
        # client_mode above; not logged to `log` (too frequent).
        self.beacon_azimuth_deg: float = 0.0
        self.beacon_muted: bool = True
        self.beacon_at: float = 0.0

    def _record(self, category: str, bucket: Dict[str, Any], fields: Dict[str, Any], log_text: str, log_entry: bool = True) -> None:
        with self._lock:
            # Merge, not clear+set — a bucket can be fed by several distinct
            # RPCs (e.g. mapping's UpdateMapping vs FindLandmark); clearing
            # would make a sticky field like occupancy_map (only set by
            # UpdateMapping) flicker away every time FindLandmark fires in
            # between. Stale keys from a differing op are harmless — every
            # *_status()/render helper in server_gui.py already branches on
            # the current op and only reads the fields relevant to it.
            bucket.update(fields)
            bucket["at"] = time.time()
            self.last_category = category
            self.last_at = bucket["at"]
            # log_entry=False (background per-tick polls: PerceptionService.
            # AnalyzeFrame's DEPTH obstacle-ahead check at ~2Hz, TRAVERSABILITY
            # local-avoidance fetch at ~2-3Hz) skips the shared Activity Log
            # deque entirely — the BUCKET itself still updates every call
            # (so the Perception tab's own frame/detail always reflects the
            # latest poll), only the log line is suppressed. Without this,
            # these two alone flood the maxlen=200 log fast enough to push
            # every genuinely rare event (walking/mapping updates, ~1Hz;
            # real ad-hoc DETECT/EMBED calls) out within well under a
            # minute — same "dashboard-only, no log spam" precedent this
            # codebase already uses for ReportBeaconDirection.
            if log_entry:
                self.log.appendleft({"at": bucket["at"], "category": category, "text": log_text})

    def record_tracking(self, log_text: str, **fields: Any) -> None:
        self._record("tracking", self.tracking, fields, log_text)

    def record_perception(self, log_text: str, log_entry: bool = True, **fields: Any) -> None:
        self._record("perception", self.perception, fields, log_text, log_entry=log_entry)

    def record_mapping(self, log_text: str, **fields: Any) -> None:
        self._record("mapping", self.mapping, fields, log_text)

    def record_client_mode(self, mode: str, target: str) -> None:
        with self._lock:
            self.client_mode = mode
            self.client_mode_target = target
            self.client_mode_at = time.time()
            self.log.appendleft({
                "at": self.client_mode_at, "category": "client",
                "text": f"mode -> '{mode}'" + (f" ({target})" if target else ""),
            })

    def record_beacon_direction(self, azimuth_deg: float, muted: bool) -> None:
        with self._lock:
            self.beacon_azimuth_deg = azimuth_deg
            self.beacon_muted = muted
            self.beacon_at = time.time()

    def reset(self) -> None:
        """Wipes every recorded bucket back to its __init__ state — called
        by StatusService.ResetSession when a fresh client connection starts,
        so server_gui.py's dashboard doesn't keep showing a previous
        connection's last frame/mode/beacon reading against a session that
        no longer exists server-side."""
        with self._lock:
            self.log.clear()
            self.tracking = {}
            self.perception = {}
            self.mapping = {}
            self.last_category = ""
            self.last_at = 0.0
            self.client_mode = ""
            self.client_mode_target = ""
            self.client_mode_at = 0.0
            self.beacon_azimuth_deg = 0.0
            self.beacon_muted = True
            self.beacon_at = 0.0

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tracking": dict(self.tracking),
                "perception": dict(self.perception),
                "mapping": dict(self.mapping),
                "last_category": self.last_category,
                "last_at": self.last_at,
                "client_mode": self.client_mode,
                "client_mode_target": self.client_mode_target,
                "client_mode_at": self.client_mode_at,
                "beacon_azimuth_deg": self.beacon_azimuth_deg,
                "beacon_muted": self.beacon_muted,
                "beacon_at": self.beacon_at,
                "log": list(self.log),
            }
