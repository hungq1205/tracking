"""
Desktop client for the Vision Assistant — uses VoiceChatStream gRPC.

Streams mic audio (16 kHz PCM) + camera frames (JPEG) to the server.
Server runs a Gemini Live session and streams 24 kHz PCM audio back.

Local tracking overlay (ORB + GroundingDINO re-ID) and hand detection
run in parallel using the same server's DetectObject / GetEmbedding RPCs.

Use headphones to avoid microphone echo.
"""
from __future__ import annotations

import base64
import queue
import threading
import time
from typing import Optional

import cv2
import grpc
import numpy as np
import sounddevice as sd
import torch
import gradio as gr

import json

from proto import tracking_pb2, tracking_pb2_grpc
from rpc_client.grpc_client import RemoteGroundingDINO, RemoteEmbedder
from core.local_models import LocalHandDetector
from core.object_tracker import GPUVIOAnchorBackend
from core.guidance_engine import GuidanceEngine
from core.renderer import GUIRenderer

AUDIO_IN_RATE  = 16_000
AUDIO_IN_CHUNK = 1024        # ~64 ms per chunk
AUDIO_OUT_RATE = 24_000
VIDEO_FPS      = 5           # camera frames per second sent to server
JPEG_QUALITY   = 70

_EMB_DIM = 1280  # EfficientNetLite embedding dimension


# ── Local tracking ───────────────────────────────────────────────────────────

class LocalTracker:
    """
    ORB-based object tracker + hand detector running entirely on the client.
    Uses server's DetectObject / GetEmbedding RPCs for initialization and re-ID
    but does local ORB tracking between keyframes.
    """

    def __init__(self):
        self._backend: Optional[GPUVIOAnchorBackend] = None
        self._hand: Optional[LocalHandDetector] = None
        self._engine: Optional[GuidanceEngine] = None
        self._target: str = ""
        self._description: str = ""
        self._ref_embedding: Optional[torch.Tensor] = None
        self._ref_image_bgr: Optional[np.ndarray] = None  # BGR, for display
        self._initialized: bool = False
        self._last_init_at: float = 0.0
        self._active: bool = False
        self._last_fps_t: float = time.perf_counter()
        self.last_state = None        # GuidanceState for GUIRenderer
        self.status: str = ""

    def setup(self, server_addr: str) -> None:
        detector = RemoteGroundingDINO(server_addr)
        embedder = RemoteEmbedder(server_addr)
        self._backend = GPUVIOAnchorBackend(
            detector=detector,
            embedder=embedder,
            nfeatures=800,
            renewal_interval=2.0,
        )
        self._hand = LocalHandDetector()
        self._engine = GuidanceEngine(
            object_tracker=self._backend,
            hand_detector=self._hand,
        )

    @property
    def reid_threshold(self) -> float:
        if self._backend:
            return self._backend.reid_threshold
        return 0.4

    @reid_threshold.setter
    def reid_threshold(self, value: float) -> None:
        if self._backend:
            self._backend.reid_threshold = float(value)

    @property
    def ref_image(self) -> Optional[np.ndarray]:
        """Return reference image as RGB ndarray for Gradio display."""
        img = None
        if self._backend and self._backend._ref_image is not None:
            img = self._backend._ref_image
        elif self._ref_image_bgr is not None:
            img = self._ref_image_bgr
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def start(
        self,
        target: str,
        description: str = "",
        ref_embedding: Optional[torch.Tensor] = None,
        ref_image_bgr: Optional[np.ndarray] = None,
    ) -> None:
        if not self._backend:
            return
        self._target = target
        self._description = description or target
        self._ref_embedding = ref_embedding
        self._ref_image_bgr = ref_image_bgr
        self._initialized = False
        self._last_init_at = 0.0
        self._active = True
        self.last_state = None
        self.status = f"Searching for '{target}'..."

    def stop(self) -> None:
        self._active = False
        self._initialized = False
        self._ref_embedding = None
        self._ref_image_bgr = None
        self.last_state = None
        self.status = ""
        if self._backend:
            try:
                self._backend.stop()
                # clear ref image on backend too
                self._backend._ref_image = None
            except Exception:
                pass

    def update(self, frame: np.ndarray) -> None:
        if not self._active or not self._engine:
            return
        now_t = time.perf_counter()
        fps = 1.0 / max(now_t - self._last_fps_t, 1e-6)
        self._last_fps_t = now_t

        if not self._initialized:
            now = time.time()
            if now - self._last_init_at >= 1.0:
                self._last_init_at = now
                try:
                    obj = self._backend.initialize(
                        frame,
                        self._target,
                        description=self._description,
                        ref_embedding=self._ref_embedding,
                        ref_image=self._ref_image_bgr,
                    )
                    if obj.visible:
                        self._initialized = True
                        self.status = f"Tracking '{self._target}'"
                except Exception as e:
                    self.status = f"Init error: {e}"
        else:
            try:
                self.last_state = self._engine.update(frame, fps=fps)
                if self.last_state:
                    s = self.last_state.object_track
                    self.status = (
                        f"Tracking '{self._target}'  "
                        f"Conf: {s.confidence:.2f}  "
                        f"{self.last_state.instruction}"
                    )
            except Exception as e:
                self.status = f"Track error: {e}"
                self._initialized = False


# ── Gemini Live stream client ─────────────────────────────────────────────────

class LiveStreamClient:
    """
    Manages one VoiceChatStream gRPC connection with four daemon threads:
      _cam_loop   — reads camera/file, stores latest_frame, enqueues JPEG chunks
      _mic_loop   — reads mic at 16 kHz, enqueues int16 PCM chunks
      _grpc_loop  — opens VoiceChatStream, forwards chunks in, receives PCM out
      _play_loop  — drains play queue into 24 kHz sounddevice OutputStream
    """

    def __init__(self, tracker: "LocalTracker"):
        self._tracker = tracker
        self._stop = threading.Event()
        self._send_q: queue.Queue = queue.Queue(maxsize=300)
        self._play_q: queue.Queue = queue.Queue(maxsize=300)
        self.latest_frame: Optional[np.ndarray] = None  # BGR
        self.status: str = "Idle"
        self.is_running: bool = False

    def start(self, server_addr: str, video_source) -> None:
        self._stop.clear()
        self.status = "Connecting..."
        self.is_running = True
        for target, args in [
            (self._cam_loop,  (video_source,)),
            (self._mic_loop,  ()),
            (self._play_loop, ()),
            (self._grpc_loop, (server_addr,)),
        ]:
            threading.Thread(target=target, args=args, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self.is_running = False
        self.status = "Disconnected"

    def _cam_loop(self, video_source) -> None:
        cap = cv2.VideoCapture(video_source)
        send_interval = 1.0 / VIDEO_FPS
        last_sent = 0.0

        is_file = isinstance(video_source, str)
        native_fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0.0
        frame_interval = 1.0 / native_fps if native_fps > 0 else 0.0
        last_frame_t = 0.0

        while not self._stop.is_set():
            if is_file and frame_interval > 0:
                now = time.time()
                wait = frame_interval - (now - last_frame_t)
                if wait > 0:
                    time.sleep(wait)
                last_frame_t = time.time()

            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    last_frame_t = 0.0
                    continue
                break
            self.latest_frame = frame
            now = time.time()
            if now - last_sent >= send_interval:
                ok, jpeg = cv2.imencode(".jpg", frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    try:
                        self._send_q.put_nowait(
                            tracking_pb2.VoiceChatChunk(video_frame=jpeg.tobytes())
                        )
                    except queue.Full:
                        pass
                last_sent = now
            if not is_file:
                time.sleep(0.01)
        cap.release()

    def _mic_loop(self) -> None:
        def _cb(indata, _frames, _time_info, _status):
            if self._stop.is_set():
                return
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            try:
                self._send_q.put_nowait(
                    tracking_pb2.VoiceChatChunk(audio_chunk=pcm)
                )
            except queue.Full:
                pass

        with sd.InputStream(samplerate=AUDIO_IN_RATE, channels=1, dtype="float32",
                            blocksize=AUDIO_IN_CHUNK, callback=_cb):
            while not self._stop.is_set():
                time.sleep(0.05)

    def _play_loop(self) -> None:
        stream = sd.RawOutputStream(
            samplerate=AUDIO_OUT_RATE, channels=1, dtype="int16"
        )
        stream.start()
        try:
            while not self._stop.is_set():
                try:
                    pcm = self._play_q.get(timeout=0.1)
                    stream.write(pcm)
                except queue.Empty:
                    continue
        finally:
            stream.stop()
            stream.close()

    def _handle_state(self, agent_state: str, agent_payload: str) -> None:
        """React to state-only AudioChunk notifications from the server."""
        if agent_state == "TRACKING":
            try:
                payload = json.loads(agent_payload) if agent_payload else {}
            except Exception:
                payload = {}
            target = payload.get("target", "")
            description = payload.get("description", target)
            ref_embedding: Optional[torch.Tensor] = None
            ref_image_bgr: Optional[np.ndarray] = None

            # Decode reference embedding (float32 bytes, base64)
            emb_b64 = payload.get("ref_embedding")
            if emb_b64:
                try:
                    raw = base64.b64decode(emb_b64)
                    arr = np.frombuffer(raw, dtype=np.float32).reshape(1, _EMB_DIM).copy()
                    ref_embedding = torch.from_numpy(arr)
                except Exception:
                    pass

            # Decode reference image (JPEG bytes, base64)
            img_b64 = payload.get("ref_image_b64")
            if img_b64:
                try:
                    raw = base64.b64decode(img_b64)
                    nparr = np.frombuffer(raw, dtype=np.uint8)
                    ref_image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    pass

            if target:
                print(f"[CLIENT] Server started tracking: '{target}' (desc: '{description}')", flush=True)
                self._tracker.start(target, description, ref_embedding, ref_image_bgr)
                self.status = f"Tracking '{target}' (server-initiated)"
        elif agent_state == "IDLE":
            if self._tracker._active:
                print("[CLIENT] Server stopped tracking.", flush=True)
                self._tracker.stop()
                self.status = "Listening..."

    def _request_iter(self):
        while not self._stop.is_set():
            try:
                yield self._send_q.get(timeout=0.1)
            except queue.Empty:
                continue

    def _grpc_loop(self, server_addr: str) -> None:
        channel = None
        try:
            channel = grpc.insecure_channel(server_addr)
            stub = tracking_pb2_grpc.TrackingServiceStub(channel)
            responses = stub.VoiceChatStream(self._request_iter())
            self.status = "Connected — Listening..."
            for chunk in responses:
                if self._stop.is_set():
                    break
                if chunk.pcm_data:
                    self.status = "AI responding..."
                    try:
                        self._play_q.put(chunk.pcm_data, timeout=0.2)
                    except queue.Full:
                        pass
                elif chunk.agent_state:
                    self._handle_state(chunk.agent_state, chunk.agent_payload)
                else:
                    self.status = "Listening..."
        except grpc.RpcError as e:
            self.status = f"gRPC error: {e.code().name}"
        except Exception as e:
            self.status = f"Error: {e}"
        finally:
            self.is_running = False
            if self.status in ("Connected — Listening...", "AI responding...", "Listening..."):
                self.status = "Disconnected"
            if channel:
                try:
                    channel.close()
                except Exception:
                    pass


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    tracker = LocalTracker()
    stream_client = LiveStreamClient(tracker)

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _connect(server_ip: str, port: str, source: str, video_path: str):
        addr = f"{server_ip.strip()}:{port.strip()}"
        src = 0 if source == "Webcam" else (video_path.strip() or 0)
        tracker.setup(addr)
        stream_client.start(addr, src)
        return "Connecting..."

    def _disconnect():
        stream_client.stop()
        tracker.stop()
        return "Disconnected."

    def _start_tracking(prompt: str):
        if not prompt.strip():
            return "Enter a tracking target first."
        tracker.start(prompt.strip())
        return f"Searching for '{prompt.strip()}'..."

    def _stop_tracking():
        tracker.stop()
        return "Tracking stopped."

    def _set_threshold(value: float):
        tracker.reid_threshold = value

    def _frame_stream():
        """Streaming generator: runs at ~20 fps, yields annotated frame + ref image + status."""
        while True:
            frame = stream_client.latest_frame
            if frame is not None:
                tracker.update(frame)
                if tracker._active and tracker.last_state is not None:
                    vis = GUIRenderer.render(frame, tracker.last_state)
                else:
                    vis = frame
                rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            else:
                rgb = None

            ref_img = tracker.ref_image  # RGB or None

            live_status = stream_client.status
            track_status = tracker.status
            combined = live_status
            if track_status:
                combined = f"{live_status}\n{track_status}"

            yield rgb, ref_img, combined
            time.sleep(0.05)

    # ── Layout ────────────────────────────────────────────────────────────

    with gr.Blocks(title="Vision Assistant — Desktop") as app:
        gr.Markdown(
            "## Vision Assistant — Desktop Client\n"
            "Voice + camera stream to Gemini Live.  "
            "**Use headphones** to avoid echo."
        )

        with gr.Row():
            # Left: controls
            with gr.Column(scale=1):
                gr.Markdown("### Connection")
                ui_server_ip = gr.Textbox(value="127.0.0.1", label="Server IP")
                ui_port      = gr.Textbox(value="50051",     label="gRPC Port")

                ui_source = gr.Radio(
                    ["Webcam", "Video File"], value="Webcam", label="Video Source"
                )
                ui_video_path = gr.Textbox(
                    placeholder="/path/to/video.mp4",
                    label="Video File Path",
                    visible=False,
                )
                ui_source.change(
                    fn=lambda s: gr.update(visible=(s == "Video File")),
                    inputs=[ui_source],
                    outputs=[ui_video_path],
                )

                btn_connect    = gr.Button("Connect & Stream", variant="primary")
                btn_disconnect = gr.Button("Disconnect",       variant="stop")

                gr.Markdown("### Local Object Tracking")
                ui_track_prompt = gr.Textbox(
                    placeholder="e.g. bottle, person, keys...",
                    label="Tracking Target",
                )
                with gr.Row():
                    btn_track_start = gr.Button("Start Tracking", variant="secondary")
                    btn_track_stop  = gr.Button("Stop Tracking")

                ui_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                    label="Re-ID Threshold",
                    info="Minimum embedding similarity to accept a detection as the same object",
                )

                gr.Markdown("### Reference Object")
                ui_ref_image = gr.Image(
                    label="Reference Image",
                    type="numpy",
                    height=160,
                    interactive=False,
                )

                ui_status = gr.Textbox(
                    label="Status", lines=3, interactive=False
                )

            # Right: camera feed with overlay
            with gr.Column(scale=2):
                ui_frame = gr.Image(label="Camera Feed", type="numpy")

        # ── Events ────────────────────────────────────────────────────────

        stream_event = btn_connect.click(
            fn=_connect,
            inputs=[ui_server_ip, ui_port, ui_source, ui_video_path],
            outputs=[ui_status],
        ).then(
            fn=_frame_stream,
            inputs=[],
            outputs=[ui_frame, ui_ref_image, ui_status],
        )

        btn_disconnect.click(
            fn=_disconnect,
            inputs=[],
            outputs=[ui_status],
            cancels=[stream_event],
        )

        btn_track_start.click(
            fn=_start_tracking,
            inputs=[ui_track_prompt],
            outputs=[ui_status],
        )

        btn_track_stop.click(
            fn=_stop_tracking,
            inputs=[],
            outputs=[ui_status],
        )

        ui_threshold.change(
            fn=_set_threshold,
            inputs=[ui_threshold],
            outputs=[],
        )

    return app


if __name__ == "__main__":
    build_ui().queue().launch(
        server_name="0.0.0.0", server_port=7861, theme=gr.themes.Monochrome()
    )
