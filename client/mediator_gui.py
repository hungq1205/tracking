import cv2
import json
import os
import tempfile
import threading
import time
from concurrent import futures
from typing import Optional

import grpc
import gradio as gr
import numpy as np

from proto import tracking_pb2
from proto import tracking_pb2_grpc
from rpc_client.grpc_client import RemoteGroundingDINO, RemoteEmbedder
from core.local_models import LocalHandDetector
from core.object_tracker import GPUVIOAnchorBackend
from core.guidance_engine import GuidanceEngine
from core.renderer import GUIRenderer
from core.interfaces import GuidanceState


class MediatorServicer(tracking_pb2_grpc.MediatorServiceServicer):
    def __init__(self, main_server_addr: str, nfeatures: int = 800, renewal_interval: float = 1.5, server_stream_fps: float = 3.0):
        self.remote_detector = RemoteGroundingDINO(main_server_addr)
        self.server_stream_fps = server_stream_fps
        self._last_server_send: float = 0.0
        self.remote_embedder = RemoteEmbedder(main_server_addr)
        self.local_hand = LocalHandDetector()
        self.backend = GPUVIOAnchorBackend(
            detector=self.remote_detector,
            embedder=self.remote_embedder,
            nfeatures=nfeatures,
            renewal_interval=renewal_interval,
        )
        self.engine = GuidanceEngine(object_tracker=self.backend, hand_detector=self.local_hand)

        # Tracking state — mirrors DesktopVideoStreamGUI
        self.initialized: bool = False
        self.tracking_active: bool = False
        self.auto_target: Optional[str] = None
        self.effective_prompt: str = ""
        self.last_init_attempt: float = 0.0
        self._agent_state: str = ""
        self._reading_timer: Optional[threading.Timer] = None
        self._pending_audio: Optional[str] = None

        # Gradio-shared state, lock-protected
        self._lock = threading.Lock()
        self._init_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_vis_frame: Optional[np.ndarray] = None
        self.latest_status: str = "Mediator: Awaiting Pi connection..."
        self.latest_guidance: Optional[GuidanceState] = None
        self.chat_history: list = []
        self._last_frame_time: float = time.perf_counter()
        self._fps: float = 0.0

    # ── gRPC: Pi → Mediator ──────────────────────────────────────────────────

    def StreamFrameWithGuidance(self, request, context):
        img_array = np.frombuffer(request.image_data, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return tracking_pb2.GuidanceFrameResponse(success=False, instruction="DECODE_ERROR")

        now = time.perf_counter()
        with self._lock:
            elapsed = now - self._last_frame_time
            self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self._last_frame_time = now
            self.latest_frame = frame.copy()

        # Forward frame to main server at throttled rate (default 3 FPS)
        if (now - self._last_server_send) >= 1.0 / self.server_stream_fps:
            self.remote_detector.send_gui_frame(frame)
            self._last_server_send = now

        # Consume deferred target set by Chat/VoiceChat agent response
        with self._lock:
            if self.auto_target:
                print(f"[MEDIATOR] Orchestrator triggered tracking for: {self.auto_target}")
                self.effective_prompt = self.auto_target
                self.auto_target = None
                self.initialized = False
                self.tracking_active = True

        if not self.tracking_active:
            vis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self.latest_vis_frame = vis
                self.latest_status = "Idle — awaiting voice/chat intent"
            return tracking_pb2.GuidanceFrameResponse(
                success=True, instruction="IDLE", tracking_status="IDLE"
            )

        # Initialization path (throttled, protected against concurrent gRPC threads)
        if not self.initialized:
            with self._init_lock:
                if not self.initialized:
                    if (now - self.last_init_attempt) >= 1.0:
                        self.last_init_attempt = now
                        print(f"[MEDIATOR] Attempting init for: {self.effective_prompt}")
                        obj_track = self.backend.initialize(frame, self.effective_prompt)
                        if obj_track.visible:
                            self.initialized = True
                        else:
                            print(f"[MEDIATOR] Not found: {self.effective_prompt}")
                            with self._lock:
                                self.latest_status = f"Searching for '{self.effective_prompt}'..."
                            return tracking_pb2.GuidanceFrameResponse(
                                success=True,
                                instruction="TARGET_OR_HAND_LOST",
                                tracking_status="SEARCHING",
                            )
                    else:
                        return tracking_pb2.GuidanceFrameResponse(
                            success=True,
                            instruction="TARGET_OR_HAND_LOST",
                            tracking_status="SEARCHING",
                        )

        fps = self._fps
        state = self.engine.update(frame, fps=fps)
        vis_frame = GUIRenderer.render(frame, state)
        vis_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)

        active_anchors = (
            len(state.object_track.debug.get("anchor_pts", []))
            if "anchor_pts" in state.object_track.debug
            else state.object_track.debug.get("total_anchors", 0)
        )
        status = (
            f"Status: {state.object_track.status}\n"
            f"FPS: {state.fps:.1f} | Conf: {state.object_track.confidence:.2f} | Anchors: {active_anchors}\n"
            f"Instruction: {state.instruction}"
        )

        with self._lock:
            self.latest_vis_frame = vis_rgb
            self.latest_guidance = state
            self.latest_status = status

        obj = state.object_track
        hand = state.hand_track
        return tracking_pb2.GuidanceFrameResponse(
            success=True,
            instruction=state.instruction,
            tracking_status=obj.status,
            object_confidence=obj.confidence,
            object_box_xyxy=list(obj.box_xyxy) if obj.visible else [],
            hand_box_xyxy=list(hand.box_xyxy) if hand.visible else [],
            delta_x=state.delta_x,
            delta_y=state.delta_y,
            distance_px=state.distance_px,
        )

    def Chat(self, request, context):
        frame = self._get_latest_frame()
        res = self.remote_detector.chat(frame, request.message)
        self._handle_agent_response(res)
        self._update_chat_history(request.message, res.response)
        if res.audio_response and self._agent_state == "READING_ALOUD":
            self._schedule_reading_continue(res.audio_response)
        return res

    def VoiceChat(self, request, context):
        res = self.remote_detector.voice_chat(request.audio_data)
        self._handle_agent_response(res)
        self._update_chat_history("[voice]", res.response)
        if res.audio_response and self._agent_state == "READING_ALOUD":
            self._schedule_reading_continue(res.audio_response)
        return res

    # ── agent response wiring (mirrors DesktopVideoStreamGUI) ────────────────

    def _handle_agent_response(self, res) -> None:
        state = res.agent_state
        self._agent_state = state
        try:
            payload = json.loads(res.agent_payload) if res.agent_payload else {}
        except (json.JSONDecodeError, ValueError):
            payload = {}

        if state == "INITIALIZING":
            with self._lock:
                self.auto_target = payload.get("target", "")
        elif state == "STOPPED":
            with self._lock:
                self.tracking_active = False
                self.initialized = False
        elif state in ("DONE_READING",):
            self._cancel_reading_timer()

    def _agent_status_text(self, res) -> str:
        state = res.agent_state
        try:
            payload = json.loads(res.agent_payload) if res.agent_payload else {}
        except Exception:
            payload = {}
        if state == "SCANNING":
            return f"Scanning... {payload.get('char_count', 0)} chars"
        if state == "READING_ALOUD":
            return f"Reading sentence {payload.get('sentence_index', 0) + 1}/{payload.get('total', 0)}"
        if state == "PAUSED":
            return f"Paused at {payload.get('sentence_index', 0) + 1}/{payload.get('total', 0)}"
        if state == "DONE_READING":
            return "Finished reading"
        return f"Agent: {state}"

    # ── reading auto-continue (mirrors DesktopVideoStreamGUI lines 127-162) ──

    def _schedule_reading_continue(self, audio_bytes: bytes) -> None:
        self._cancel_reading_timer()
        if self._agent_state != "READING_ALOUD" or not audio_bytes:
            return
        audio_data_bytes = max(0, len(audio_bytes) - 44)
        duration_sec = audio_data_bytes / (24000 * 2)
        delay = max(0.1, duration_sec + 0.3)
        self._reading_timer = threading.Timer(delay, self._auto_continue_reading)
        self._reading_timer.daemon = True
        self._reading_timer.start()

    def _cancel_reading_timer(self) -> None:
        if self._reading_timer and self._reading_timer.is_alive():
            self._reading_timer.cancel()
        self._reading_timer = None

    def _auto_continue_reading(self) -> None:
        if self._agent_state != "READING_ALOUD":
            return
        try:
            frame = self._get_latest_frame()
            res = self.remote_detector.chat(frame, "continue reading")
            self._handle_agent_response(res)
            if res.audio_response:
                audio_path = os.path.join(tempfile.gettempdir(), "mediator_auto_continue.wav")
                with open(audio_path, "wb") as f:
                    f.write(res.audio_response)
                self._pending_audio = audio_path
            if self._agent_state == "READING_ALOUD":
                self._schedule_reading_continue(res.audio_response or b"")
        except Exception as e:
            print(f"[MEDIATOR] Auto-continue error: {e}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def _update_chat_history(self, user_msg: str, bot_msg: str) -> None:
        with self._lock:
            self.chat_history.append((user_msg, bot_msg))

    def _poll_pending_audio(self) -> Optional[str]:
        if self._pending_audio:
            path = self._pending_audio
            self._pending_audio = None
            return path
        return None

    def _poll_pending_audio_stream(self):
        while True:
            time.sleep(0.5)
            yield self._poll_pending_audio()


class MediatorGUI:
    def __init__(self, main_server_ip: str, nfeatures: int = 800, renewal_interval: float = 1.5, server_stream_fps: float = 3.0):
        self.servicer = MediatorServicer(
            main_server_addr=f"{main_server_ip}:50051",
            nfeatures=nfeatures,
            renewal_interval=renewal_interval,
            server_stream_fps=server_stream_fps,
        )

    def _start_grpc_server(self) -> None:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        tracking_pb2_grpc.add_MediatorServiceServicer_to_server(self.servicer, server)
        server.add_insecure_port("[::]:50052")
        server.start()
        print("[MEDIATOR] gRPC server started on port 50052")
        server.wait_for_termination()

    def _poll_dashboard_state(self):
        with self.servicer._lock:
            vis = self.servicer.latest_vis_frame
            status = self.servicer.latest_status
            agent = self.servicer._agent_state or "—"
            pi_status = f"Pi frames FPS: {self.servicer._fps:.1f}"
        return vis, status, agent, pi_status

    def _handle_operator_chat(self, message: str, history: list):
        history = history or []
        if not message:
            return history, None
        frame = self.servicer._get_latest_frame()
        if frame is None:
            history.append((message, "No frames from Pi yet."))
            return history, None
        res = self.servicer.remote_detector.chat(frame, message)
        self.servicer._handle_agent_response(res)
        audio_path = None
        if res.audio_response:
            audio_path = os.path.join(tempfile.gettempdir(), "mediator_operator.wav")
            with open(audio_path, "wb") as f:
                f.write(res.audio_response)
            if self.servicer._agent_state == "READING_ALOUD":
                self.servicer._schedule_reading_continue(res.audio_response)
        history.append((message, res.response))
        return history, audio_path

    def _build_gradio_app(self) -> gr.Blocks:
        with gr.Blocks(title="Mediator Dashboard") as app:
            gr.Markdown("## Mediator — Pi Edge Client Monitor")

            with gr.Row():
                with gr.Column(scale=1):
                    ui_pi_status = gr.Textbox(
                        label="Pi Connection", value="Waiting...", interactive=False
                    )
                    ui_agent_state = gr.Textbox(label="Agent State", interactive=False)
                    ui_status = gr.Textbox(
                        label="Tracking Status",
                        value="Mediator: Awaiting Pi connection...",
                        lines=4,
                        interactive=False,
                    )

                with gr.Column(scale=2):
                    ui_image = gr.Image(label="Processed View (from Pi)")

                    ui_chatbot = gr.Chatbot(label="Agent Conversation")
                    with gr.Row():
                        ui_chat_input = gr.Textbox(
                            placeholder="Send chat to main server...",
                            label=None,
                            scale=4,
                            container=False,
                        )
                        btn_chat = gr.Button("Ask", scale=1)
                    ui_audio_playback = gr.Audio(visible=False, autoplay=True)

            timer = gr.Timer(value=0.5)
            timer.tick(
                fn=self._poll_dashboard_state,
                outputs=[ui_image, ui_status, ui_agent_state, ui_pi_status],
            )

            btn_chat.click(
                fn=self._handle_operator_chat,
                inputs=[ui_chat_input, ui_chatbot],
                outputs=[ui_chatbot, ui_audio_playback],
            )

            app.load(
                fn=self.servicer._poll_pending_audio_stream,
                outputs=[ui_audio_playback],
            )

        return app

    def run(self) -> None:
        grpc_thread = threading.Thread(target=self._start_grpc_server, daemon=True)
        grpc_thread.start()
        print("[MEDIATOR] Launching Gradio dashboard on port 7862")
        self._build_gradio_app().queue().launch(
            server_name="0.0.0.0", server_port=7862, theme=gr.themes.Monochrome()
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mediator GUI — bridges Pi client and main server")
    parser.add_argument("--server-ip", default="127.0.0.1", help="Main server IP (default: 127.0.0.1)")
    parser.add_argument("--nfeatures", type=int, default=800, help="ORB anchor count")
    parser.add_argument("--renewal-interval", type=float, default=1.5, help="Tracker renewal interval (s)")
    parser.add_argument("--server-stream-fps", type=float, default=3.0, help="FPS at which frames are forwarded to main server (default: 3)")
    args = parser.parse_args()

    gui = MediatorGUI(
        main_server_ip=args.server_ip,
        nfeatures=args.nfeatures,
        renewal_interval=args.renewal_interval,
        server_stream_fps=args.server_stream_fps,
    )
    gui.run()
