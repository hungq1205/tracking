"""
pixie_hrtf_server.py — standalone test harness validating the design behind
the future "Pixie" persistent-companion HRTF beacon (see this repo's
CLAUDE.md and test_module/pixie_hrtf_app/README.md) BEFORE it's built into
the real client/android/ app: does an RTAB-Map-derived, anchor-relative head
heading actually let a moving, spatialized pixie sound track a real head
turn correctly, and does the Android-side local drift compensation
(RotationTracker's ORB/Essential-matrix RANSAC) bridge the server round-trip
gap smoothly?

Talks to the test Android app (test_module/pixie_hrtf_app/android/) over a
plain WebSocket:

  Android -> server   {"type":"frame","ts_ns":<int>,"jpeg_b64":<str>}
  Android -> server   {"type":"reset"}
  server  -> Android  {"type":"heading","frame_ts_ns":<int>,"tracking_ok":<bool>,
                        "relative_heading_deg":<float>,
                        "pixie_azimuth_deg":<float>,"pixie_elevation_deg":<float>}
  server  -> Android  {"type":"error","message":<str>}

Reuses scan_server/rtabmap_client.py (RtabmapPoseClient, talking to the
already-existing tracking-rtabmap docker service — see
scan_server/rtabmap_docker/README.md) and scan_server/da3_wrapper.py (DA3
depth, RTAB-Map's RGB-D odometry needs a depth map same as everywhere else
in this codebase) via the same sys.path-bootstrap convention
server/grpc_server.py/mapping_servicer.py already use for those modules.

Anchor convention: the FIRST frame that yields a valid RTAB-Map pose
defines "straight ahead" (0 deg) AND the origin of the circle the pixie
moves on. "Reset anchor" (Gradio button or the Android app's own button,
which also sends {"type":"reset"}) clears it so the NEXT valid frame
re-anchors.

Pixie motion (PixieMotion, below) — requested directly by the user, second
design pass: the pixie autonomously loops MOVING (to a random point on the
circumference of a circle, radius/speed configurable, travelling ALONG the
circumference, never through the middle) then WAITING (wait_s, configurable)
then picking a new random target and repeating. Runs on its own background
thread, ticking independently of whether an Android client is even
connected — matches the eventual real app's "exists for the whole app
lifecycle" design. `pixie_azimuth_deg` on the wire is this motion's current
angle on the circle (0 = the anchor's original forward direction, + =
clockwise/right) — unchanged wire semantics from the first design pass,
just server-autonomous now instead of GUI-slider-set.

Deliberate simplification, called out directly: the circle is centered on
the ANCHOR's own position, and the user's OWN position is assumed to stay
there too (RTAB-Map's translation is read but never used) — this harness
only exercises HEAD ROTATION tracking (the actual thing being validated),
not walking. A user who physically walks during a session will see the
pixie's apparent bearing drift by exactly however far they moved, same
"known, accepted limitation" class as this whole harness's other
simplifications (see README).
"""

import asyncio
import base64
import json
import math
import os
import random
import sys
import threading
import time
import traceback

import cv2
import gradio as gr
import numpy as np
import websockets

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCAN_SERVER_DIR = os.path.join(_REPO_ROOT, "scan_server")
if _SCAN_SERVER_DIR not in sys.path:
    sys.path.insert(0, _SCAN_SERVER_DIR)

from da3_wrapper import build_estimator  # noqa: E402
from rtabmap_client import RtabmapPoseClient  # noqa: E402

WS_HOST = os.getenv("PIXIE_WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("PIXIE_WS_PORT", "8765"))
GRADIO_PORT = int(os.getenv("PIXIE_GRADIO_PORT", "7864"))
RTABMAP_ADDR = os.getenv("RTABMAP_ADDR", "tcp://localhost:5556")
DA3_ONNX_PATH = os.getenv("DA3_ONNX_PATH", os.path.join(_REPO_ROOT, "DA3METRIC-LARGE.onnx"))
DA3_DEVICE = os.getenv("PIXIE_DA3_DEVICE", "cpu")

_DEFAULT_RADIUS_M = 5.0
_DEFAULT_SPEED_MPS = 0.8  # moderate pace — the sine ease's average velocity runs ~64% of this peak
_DEFAULT_WAIT_S = 10.0
_DEFAULT_ELEVATION_DEG = -30.0
_MOTION_TICK_S = 0.05


def _pose_heading_rad(pose_mat: np.ndarray) -> float:
    """World-frame heading (floor-plane yaw) of the camera's own forward
    axis — direct duplicate of server/services/mapping_servicer.py's
    _pose_heading_rad (same camera-local X-right/Y-down/Z-forward
    convention this whole project uses throughout)."""
    forward_world = pose_mat[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return float(np.arctan2(forward_world[0], forward_world[2]))


def _wrap_deg(deg: float) -> float:
    d = deg % 360.0
    if d > 180.0:
        d -= 360.0
    return d


class PixieMotion:
    """Autonomous circular motion, running on its own daemon thread — see
    module docstring. Never touches RTAB-Map/depth/websockets directly,
    purely a angle-on-a-circle state machine; `snapshot()` is the only
    thing the WS handler / Gradio poller read from it.

    **Sine-eased motion, per direct request for realism**: each MOVING
    phase follows a raised-cosine angular-position profile —
    theta(t) = start + delta * (1 - cos(pi*t/T)) / 2 — whose derivative is
    a pure sine: zero velocity at the start and end of a move, peaking at
    the midpoint, instead of an instantly-on/instantly-off constant
    angular velocity. `speed_mps` is that profile's PEAK velocity (chosen
    so a move's duration works out to T = distance*pi/(2*peak), since
    integrating a sine of peak V over duration T covers V*T*(2/pi) of
    angular distance) — not an average. Direct Python port of the same
    change in the Android app's PixieMotion.kt, kept in sync deliberately.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.radius_m = _DEFAULT_RADIUS_M
        self.speed_mps = _DEFAULT_SPEED_MPS
        self.wait_s = _DEFAULT_WAIT_S
        self.elevation_deg = _DEFAULT_ELEVATION_DEG

        self._theta_deg = 0.0
        self._target_theta_deg = random.uniform(-180.0, 180.0)
        self._state = "MOVING"
        self._wait_until_ts = 0.0

        # Sine-eased move parameters — fixed once at the START of each
        # MOVING phase (not recomputed mid-move even if radius/speed
        # change live), so a move stays smooth and self-consistent.
        self._move_start_theta_deg = 0.0
        self._move_delta_deg = 0.0
        self._move_duration_s = 0.05
        self._move_elapsed_s = 0.0

        threading.Thread(target=self._run, daemon=True).start()

    def configure(self, radius_m=None, speed_mps=None, wait_s=None, elevation_deg=None):
        with self._lock:
            if radius_m is not None:
                self.radius_m = max(0.1, float(radius_m))
            if speed_mps is not None:
                self.speed_mps = max(0.01, float(speed_mps))
            if wait_s is not None:
                self.wait_s = max(0.0, float(wait_s))
            if elevation_deg is not None:
                self.elevation_deg = float(elevation_deg)

    def _begin_move(self):
        self._state = "MOVING"
        self._move_start_theta_deg = self._theta_deg
        self._move_delta_deg = _wrap_deg(self._target_theta_deg - self._theta_deg)
        peak_angular_speed_deg_s = math.degrees(self.speed_mps / max(self.radius_m, 1e-6))
        if peak_angular_speed_deg_s > 0.001:
            self._move_duration_s = max(
                0.05, abs(self._move_delta_deg) * math.pi / (2.0 * peak_angular_speed_deg_s)
            )
        else:
            self._move_duration_s = 0.05
        self._move_elapsed_s = 0.0

    def _run(self):
        last_ts = time.time()
        self._begin_move()
        while True:
            time.sleep(_MOTION_TICK_S)
            now = time.time()
            dt = now - last_ts
            last_ts = now
            with self._lock:
                if self._state == "WAITING":
                    if now >= self._wait_until_ts:
                        self._target_theta_deg = random.uniform(-180.0, 180.0)
                        self._begin_move()
                    continue
                # MOVING — travel ALONG the circumference (angle-space),
                # never a straight chord through the circle's interior.
                self._move_elapsed_s += dt
                if self._move_elapsed_s >= self._move_duration_s:
                    self._theta_deg = self._target_theta_deg
                    self._state = "WAITING"
                    self._wait_until_ts = now + self.wait_s
                else:
                    frac = min(1.0, self._move_elapsed_s / self._move_duration_s)
                    eased = (1.0 - math.cos(math.pi * frac)) / 2.0
                    self._theta_deg = _wrap_deg(self._move_start_theta_deg + self._move_delta_deg * eased)

    def snapshot(self) -> dict:
        with self._lock:
            remaining_wait_s = max(0.0, self._wait_until_ts - time.time()) if self._state == "WAITING" else 0.0
            return {
                "theta_deg": self._theta_deg,
                "elevation_deg": self.elevation_deg,
                "state": self._state,
                "target_theta_deg": self._target_theta_deg,
                "radius_m": self.radius_m,
                "speed_mps": self.speed_mps,
                "wait_s": self.wait_s,
                "remaining_wait_s": remaining_wait_s,
            }


class PixieState:
    """Shared, lock-protected state read by the WebSocket handler thread's
    event loop and written by both the Gradio UI thread and the handler.
    Head-tracking status only now — pixie position/motion lives in
    PixieMotion instead (see above)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.tracking_ok = False
        self.anchor_set = False
        self.relative_heading_deg = 0.0
        self.last_frame_rgb = None
        self.last_update_ts = 0.0
        self.client_connected = False

    def record_frame(self, tracking_ok, anchor_set, relative_heading_deg, frame_rgb):
        with self._lock:
            self.tracking_ok = tracking_ok
            self.anchor_set = anchor_set
            self.relative_heading_deg = relative_heading_deg
            self.last_frame_rgb = frame_rgb
            self.last_update_ts = time.time()

    def record_reset(self):
        with self._lock:
            self.tracking_ok = False
            self.anchor_set = False
            self.relative_heading_deg = 0.0

    def snapshot_status(self):
        with self._lock:
            return {
                "client_connected": self.client_connected,
                "tracking_ok": self.tracking_ok,
                "anchor_set": self.anchor_set,
                "relative_heading_deg": self.relative_heading_deg,
                "last_update_ts": self.last_update_ts,
                "last_frame_rgb": self.last_frame_rgb,
            }


class HeadTrackSession:
    """One RTAB-Map pose session + DA3 depth estimator + this harness's own
    anchor bookkeeping. RTAB-Map's own docker service only supports ONE
    active session at a time (see rtabmap_docker/README.md) — this class
    is deliberately a singleton (see `_session` below), matching that
    constraint rather than pretending multiple Android testers could use
    this server concurrently."""

    def __init__(self, rtabmap_addr: str, onnx_path: str, device: str):
        print(f"[pixie_hrtf_server] Loading DA3 estimator (onnx_path={onnx_path}, device={device})...")
        self.estimator = build_estimator(onnx_path=onnx_path, device=device)
        print(f"[pixie_hrtf_server] Connecting to RTAB-Map at {rtabmap_addr}...")
        self.rtabmap = RtabmapPoseClient(rtabmap_addr)
        self._anchor_heading_rad = None
        self._last_relative_deg = 0.0
        self._lock = threading.Lock()

    def process_frame(self, rgb: np.ndarray, ts_ns: int):
        """Runs DA3 depth + one RTAB-Map TRACK call — blocking/CPU-GPU
        work, always called via an executor from the async handler so it
        never stalls the asyncio event loop. Returns
        (tracking_ok, anchor_set, relative_heading_deg)."""
        with self._lock:
            depth_frame = self.estimator.estimate(rgb)
            tracked = self.rtabmap.track_batch([rgb], [depth_frame], [ts_ns])[0]
            if tracked.pose is None:
                return False, self._anchor_heading_rad is not None, self._last_relative_deg

            heading_rad = _pose_heading_rad(tracked.pose)
            if self._anchor_heading_rad is None:
                self._anchor_heading_rad = heading_rad
                print(f"[pixie_hrtf_server] Anchor set: heading_rad={heading_rad:.3f} "
                      f"({np.degrees(heading_rad):.1f} deg)")
            relative_rad = heading_rad - self._anchor_heading_rad
            self._last_relative_deg = _wrap_deg(np.degrees(relative_rad))
            return True, True, self._last_relative_deg

    def reset(self):
        with self._lock:
            self._anchor_heading_rad = None
            self._last_relative_deg = 0.0
            self.rtabmap.reset()
        print("[pixie_hrtf_server] Session reset — next tracked frame re-anchors.")


_state = PixieState()
_motion = PixieMotion()  # starts moving immediately, no client needed
_session: "HeadTrackSession | None" = None


def _get_session() -> HeadTrackSession:
    global _session
    if _session is None:
        _session = HeadTrackSession(RTABMAP_ADDR, DA3_ONNX_PATH, DA3_DEVICE)
    return _session


def _decode_frame(jpeg_b64: str):
    raw = base64.b64decode(jpeg_b64)
    nparr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


async def _ws_handler(websocket):
    print(f"[pixie_hrtf_server] Client connected: {websocket.remote_address}")
    _state.client_connected = True
    loop = asyncio.get_event_loop()
    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                continue
            msg_type = msg.get("type")

            if msg_type == "frame":
                rgb = _decode_frame(msg.get("jpeg_b64", ""))
                if rgb is None:
                    continue
                session = _get_session()
                tracking_ok, anchor_set, relative_heading_deg = await loop.run_in_executor(
                    None, session.process_frame, rgb, msg.get("ts_ns", 0)
                )
                _state.record_frame(tracking_ok, anchor_set, relative_heading_deg, rgb)
                motion = _motion.snapshot()
                response = {
                    "type": "heading",
                    "frame_ts_ns": msg.get("ts_ns", 0),
                    "tracking_ok": tracking_ok,
                    "relative_heading_deg": relative_heading_deg,
                    "pixie_azimuth_deg": motion["theta_deg"],
                    "pixie_elevation_deg": motion["elevation_deg"],
                }
                await websocket.send(json.dumps(response))

            elif msg_type == "reset":
                session = _get_session()
                await loop.run_in_executor(None, session.reset)
                _state.record_reset()
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        _state.client_connected = False
        print(f"[pixie_hrtf_server] Client disconnected: {websocket.remote_address}")


def _run_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _serve():
        async with websockets.serve(_ws_handler, WS_HOST, WS_PORT, max_size=10_000_000):
            print(f"[pixie_hrtf_server] WebSocket server listening on ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()  # run forever

    loop.run_until_complete(_serve())


def _build_ui() -> gr.Blocks:
    def _poll():
        snap = _state.snapshot_status()
        m = _motion.snapshot()
        status = (
            f"Android client connected: {snap['client_connected']}\n"
            f"Tracking OK: {snap['tracking_ok']}\n"
            f"Anchor set: {snap['anchor_set']}\n"
            f"Anchor-relative heading: {snap['relative_heading_deg']:.1f} deg\n"
            f"Pixie: {m['state']}  theta={m['theta_deg']:.1f} deg  "
            f"target={m['target_theta_deg']:.1f} deg"
        )
        if m["state"] == "WAITING":
            status += f"  (next move in {m['remaining_wait_s']:.1f}s)"
        status += f"\nelevation={m['elevation_deg']:.1f} deg  radius={m['radius_m']:.2f}m  speed={m['speed_mps']:.2f}m/s\n"
        if snap["last_update_ts"] > 0:
            status += f"Last frame: {time.time() - snap['last_update_ts']:.1f}s ago"
        else:
            status += "Last frame: never"
        return status, snap["last_frame_rgb"]

    def _on_radius(v):
        _motion.configure(radius_m=v)

    def _on_speed(v):
        _motion.configure(speed_mps=v)

    def _on_wait(v):
        _motion.configure(wait_s=v)

    def _on_elevation(v):
        _motion.configure(elevation_deg=v)

    def _on_reset():
        session = _get_session()
        threading.Thread(target=session.reset, daemon=True).start()
        _state.record_reset()
        return "Reset requested — next tracked frame re-anchors (pixie's circle re-centers on it too)."

    with gr.Blocks(title="Pixie HRTF head-tracking test") as demo:
        gr.Markdown(
            "## Pixie HRTF head-tracking test\n"
            "The pixie moves on its own — it loops between flying to a "
            "random point on the circle's circumference (travelling along "
            "the arc, never straight through the middle) and waiting there "
            "— configure the circle below. 0 deg is the direction the "
            "Android app was first pointed when it connected. Put on "
            "headphones and confirm the pixie's ping tone quiets down "
            "exactly when you turn to face it, and gets louder the more "
            "you're turned away, with no noticeable lag between server "
            "updates."
        )
        with gr.Row():
            radius_slider = gr.Slider(
                minimum=0.5, maximum=10.0, step=0.1, value=_DEFAULT_RADIUS_M,
                label="Circle radius (m)",
            )
            speed_slider = gr.Slider(
                minimum=0.1, maximum=3.0, step=0.1, value=_DEFAULT_SPEED_MPS,
                label="Travel speed (m/s)",
            )
        with gr.Row():
            wait_input = gr.Number(value=_DEFAULT_WAIT_S, label="Wait at each point (s)", precision=1)
            elevation_slider = gr.Slider(
                minimum=-90, maximum=90, step=1, value=_DEFAULT_ELEVATION_DEG,
                label="Pixie elevation (deg, + = up, fixed — not part of the circle motion)",
            )
        radius_slider.change(_on_radius, inputs=[radius_slider], outputs=[])
        speed_slider.change(_on_speed, inputs=[speed_slider], outputs=[])
        wait_input.change(_on_wait, inputs=[wait_input], outputs=[])
        elevation_slider.change(_on_elevation, inputs=[elevation_slider], outputs=[])

        reset_button = gr.Button("Reset anchor + RTAB-Map session")
        reset_status = gr.Textbox(label="", show_label=False, interactive=False)
        reset_button.click(_on_reset, inputs=[], outputs=[reset_status])

        with gr.Row():
            status_box = gr.Textbox(label="Live status", lines=8, interactive=False)
            frame_image = gr.Image(label="Last received frame", type="numpy")

        timer = gr.Timer(value=0.2)
        timer.tick(fn=_poll, inputs=[], outputs=[status_box, frame_image])

    return demo


if __name__ == "__main__":
    threading.Thread(target=_run_ws_server, daemon=True).start()
    ui = _build_ui()
    ui.launch(server_name="0.0.0.0", server_port=GRADIO_PORT)
