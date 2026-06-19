import cv2
import json
import time
import threading
import gradio as gr
import os
import tempfile
from typing import Dict, Any, Generator, Optional
from concurrent.futures import ThreadPoolExecutor

from rpc_client.grpc_client import RemoteGroundingDINO, RemoteEmbedder
from core.local_models import LocalHandDetector
from core.object_tracker import GPUVIOAnchorBackend
from core.guidance_engine import GuidanceEngine
from core.helpers import CameraEmulator
from core.renderer import GUIRenderer
from core.interfaces import IClientApp

global_executor = ThreadPoolExecutor(max_workers=4)


class DesktopVideoStreamGUI(IClientApp):
    def __init__(self):
        self.remote_detector = None
        self.remote_embedder = None
        self.local_hand = None
        self.backend = None
        self.engine = None
        self.emulator = None
        self.last_gui_send = 0
        self.initialized = False
        self.server_addr = None
        self.stream_to_server_flag = False
        self.latest_frame = None
        self.auto_target = None
        self.tracking_active = False

        self._agent_state: str = ""
        self._reading_timer: Optional[threading.Timer] = None
        self._pending_audio: Optional[str] = None

    def _initialize_backend_services(self, server_ip, nfeatures, renewal_interval):
        self.server_addr = f"{server_ip}:50051"
        print(f"[DESKTOP GUI] Connecting to Server at {self.server_addr}...")
        self.remote_detector = RemoteGroundingDINO(self.server_addr)
        self.remote_embedder = RemoteEmbedder(self.server_addr)
        self.local_hand = LocalHandDetector()

        self.backend = GPUVIOAnchorBackend(
            detector=self.remote_detector,
            embedder=self.remote_embedder,
            nfeatures=int(nfeatures),
            renewal_interval=float(renewal_interval),
        )
        self.engine = GuidanceEngine(object_tracker=self.backend, hand_detector=self.local_hand)

    def _handle_tracking_logic(self, frame, text_prompt, fps, last_init_attempt):
        if not self.initialized:
            now = time.time()
            if (now - last_init_attempt) >= 1.0:
                print(f"[DESKTOP GUI] Attempting initialization for: {text_prompt}")
                obj_track = self.backend.initialize(frame, text_prompt)
                if obj_track.visible:
                    self.initialized = True
                    return self.engine._build_state(obj_track, self.local_hand.detect(frame), fps=fps), now
                print(f"[DESKTOP GUI] Not found: {text_prompt}")
                return None, now
            return None, last_init_attempt

        return self.engine.update(frame, fps=fps), last_init_attempt

    def _maybe_stream_to_server(self, frame, stream_fps):
        if self.stream_to_server_flag:
            now = time.time()
            if (now - self.last_gui_send) > (1.0 / float(stream_fps)):
                self.remote_detector.send_gui_frame(frame)
                self.last_gui_send = now

    def _handle_agent_response(self, res) -> None:
        state = res.agent_state
        self._agent_state = state

        try:
            payload = json.loads(res.agent_payload) if res.agent_payload else {}
        except (json.JSONDecodeError, ValueError):
            payload = {}

        if state == "INITIALIZING":
            self.auto_target = payload.get("target", "")
        elif state == "STOPPED":
            self.tracking_active = False
            self.initialized = False
        elif state in ("DONE_READING", "STOPPED"):
            self._cancel_reading_timer()

    def _agent_status_text(self, res) -> str:
        state = res.agent_state
        try:
            payload = json.loads(res.agent_payload) if res.agent_payload else {}
        except Exception:
            payload = {}

        if state == "SCANNING":
            char_count = payload.get("char_count", 0)
            label = payload.get("label", "")
            return f"Status: Scanning... {char_count} characters saved" + (f" [{label}]" if label else "")
        if state == "READING_ALOUD":
            idx = payload.get("sentence_index", 0)
            total = payload.get("total", 0)
            return f"Status: Reading sentence {idx + 1} of {total}"
        if state == "PAUSED":
            idx = payload.get("sentence_index", 0)
            total = payload.get("total", 0)
            return f"Status: Paused at sentence {idx + 1} of {total}"
        if state == "DONE_READING":
            return "Status: Finished reading"
        if state == "OBJECT_SAVED":
            label = payload.get("label", "")
            return f"Status: Object saved to memory '{label}'"
        if state == "SAVE_STARTED":
            label = payload.get("label", "")
            return f"Status: Ready to scan for '{label}'"
        return f"Status: {state}"

    # ── reading auto-continue ─────────────────────────────────────────────────

    def _schedule_reading_continue(self, audio_bytes: bytes) -> None:
        self._cancel_reading_timer()
        if self._agent_state != "READING_ALOUD":
            return
        if not audio_bytes:
            return
        # Kokoro: 24kHz 16-bit PCM WAV; header is 44 bytes
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
        if self.remote_detector is None:
            return
        try:
            res = self.remote_detector.chat(self.latest_frame, "continue reading")
            self._handle_agent_response(res)
            if res.audio_response:
                audio_path = os.path.join(tempfile.gettempdir(), "auto_continue.wav")
                with open(audio_path, "wb") as f:
                    f.write(res.audio_response)
                self._pending_audio = audio_path
            if self._agent_state == "READING_ALOUD":
                self._schedule_reading_continue(res.audio_response or b"")
        except Exception as e:
            print(f"[DESKTOP GUI] Auto-continue error: {e}")

    def _poll_pending_audio(self):
        if self._pending_audio:
            path = self._pending_audio
            self._pending_audio = None
            return path
        return None

    def _poll_pending_audio_stream(self):
        while True:
            time.sleep(0.5)
            yield self._poll_pending_audio()

    # ── experiment generator ──────────────────────────────────────────────────

    def _run_experiment_generator(self, video_file, server_ip, stream_to_server_toggle, nfeatures, renewal_interval, stream_fps) -> Generator[Dict[str, Any], None, None]:
        if video_file is None:
            yield {ui_status: "Status: Please upload a video file."}
            return

        self._initialize_backend_services(server_ip, nfeatures, renewal_interval)
        self.stream_to_server_flag = stream_to_server_toggle
        self.emulator = CameraEmulator(video_file, target_fps=15)

        self.initialized = False
        self.last_gui_send = 0
        last_init_attempt = 0
        last_loop_time = time.perf_counter()
        self.tracking_active = False
        effective_prompt = ""

        for frame, _ in self.emulator.stream():
            self.latest_frame = frame.copy()
            t_start = time.perf_counter()

            if self.auto_target:
                print(f"[DESKTOP GUI] Orchestrator triggered tracking for: {self.auto_target}")
                effective_prompt = self.auto_target
                self.auto_target = None
                self.initialized = False
                self.tracking_active = True

            current_time = time.perf_counter()
            fps = 1.0 / (current_time - last_loop_time) if current_time > last_loop_time else 0.0
            last_loop_time = current_time

            self._maybe_stream_to_server(frame, stream_fps)

            if not self.tracking_active:
                yield {
                    ui_image: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ui_status: "Status: Idle (Awaiting Voice/Chat Intent)",
                }
                continue

            state, last_init_attempt = self._handle_tracking_logic(frame, effective_prompt, fps, last_init_attempt)

            if state is None:
                yield {
                    ui_image: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ui_status: f"Status: Searching for '{effective_prompt}'...\n(Retrying every 1s)",
                }
                continue

            latency = (time.perf_counter() - t_start) * 1000
            vis_frame = GUIRenderer.render(frame, state)
            active_anchors = (
                len(state.object_track.debug.get("anchor_pts", []))
                if "anchor_pts" in state.object_track.debug
                else state.object_track.debug.get("total_anchors", 0)
            )
            status_text = (
                f"Status: {state.object_track.status}\n"
                f"FPS: {state.fps:.1f} | Conf: {state.object_track.confidence:.2f} | Anchors: {active_anchors}\n"
                f"Latency: {latency:.1f}ms\nInstruction: {state.instruction}"
            )
            yield {ui_image: cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), ui_status: status_text}

    # ── chat handlers ─────────────────────────────────────────────────────────

    def _handle_chat(self, message, history):
        history = history or []
        message = message or ""
        if self.remote_detector is None:
            return history + [["User", "Error: Click 'Run' first."]], None
        if not self.stream_to_server_flag:
            return history + [["User", "Error: Enable 'Stream to Server'."]], None
        if self.latest_frame is None:
            return history + [["User", "Error: No frames."]], None

        res = self.remote_detector.chat(self.latest_frame, message)
        self._handle_agent_response(res)

        audio_path = None
        if res.audio_response:
            audio_path = os.path.join(tempfile.gettempdir(), "response.wav")
            with open(audio_path, "wb") as f:
                f.write(res.audio_response)
            if self._agent_state == "READING_ALOUD":
                self._schedule_reading_continue(res.audio_response)

        history.append((message, res.response))
        return history, audio_path

    def _handle_voice_chat(self, audio_path, history):
        history = history or []
        if audio_path is None:
            return history, None
        if self.remote_detector is None:
            return history + [["System", "Error: Click 'Run' first."]], None
        if not self.stream_to_server_flag:
            return history + [["System", "Error: Enable 'Stream to Server'."]], None
        if self.latest_frame is None:
            return history + [["System", "Error: No frames."]], None

        with open(audio_path, "rb") as f:
            audio_data = f.read()

        res = self.remote_detector.voice_chat(audio_data)
        self._handle_agent_response(res)

        out_audio = None
        if res.audio_response:
            out_audio = os.path.join(tempfile.gettempdir(), "voice_response.wav")
            with open(out_audio, "wb") as f:
                f.write(res.audio_response)
            if self._agent_state == "READING_ALOUD":
                self._schedule_reading_continue(res.audio_response)

        history.append((None, res.response))
        return history, out_audio

    # ── UI ────────────────────────────────────────────────────────────────────

    def run(self, video_source: str, prompt: str, server_ip: str, stream_to_server: bool):
        global ui_image, ui_status, ui_chatbot
        ui_image = gr.Image(label="Processed GPU Output Pipeline View")
        ui_status = gr.Textbox(label="Framework Metrics & Connection Status", lines=3, interactive=False)
        ui_chatbot = gr.Chatbot(label="VLM Conversation History")

        with gr.Blocks(title="Desktop Object Tracker Client") as app:
            gr.Markdown("## Reactive Intent-Orchestrated Tracker")

            with gr.Row():
                with gr.Column(scale=1):
                    ui_video = gr.Video(label="Video Input Stream")
                    ui_server_ip = gr.Textbox(value=server_ip, label="gRPC Server IP")

                    with gr.Accordion("Engine Settings", open=False):
                        ui_nfeatures = gr.Slider(minimum=100, maximum=3000, step=100, value=800, label="ORB Anchors")
                        ui_renewal = gr.Slider(minimum=0.5, maximum=10.0, step=0.5, value=1.5, label="Renewal Interval (s)")
                        ui_stream_toggle = gr.Checkbox(value=stream_to_server, label="Stream to Server GUI")
                        ui_stream_fps = gr.Slider(minimum=1, maximum=30, step=1, value=3, label="Server Stream FPS")

                    btn_start = gr.Button("Run", variant="primary")
                    btn_stop = gr.Button("Stop", variant="stop")
                with gr.Column(scale=2):
                    ui_image.render()

                    with gr.Group():
                        ui_chatbot.render()
                        with gr.Row(equal_height=True):
                            ui_chat_input = gr.Textbox(
                                placeholder="Ask something about the current view...",
                                label=None,
                                scale=4,
                                container=False,
                            )
                            ui_audio_input = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                show_label=False,
                                scale=3,
                                container=False,
                            )
                            btn_chat = gr.Button("Ask", variant="secondary", scale=1)
                        ui_status.render()
                        ui_audio_playback = gr.Audio(visible=False, autoplay=True)

            run_event = btn_start.click(
                fn=self._run_experiment_generator,
                inputs=[ui_video, ui_server_ip, ui_stream_toggle, ui_nfeatures, ui_renewal, ui_stream_fps],
                outputs=[ui_image, ui_status],
            )
            btn_stop.click(fn=None, cancels=[run_event])

            btn_chat.click(
                fn=self._handle_chat,
                inputs=[ui_chat_input, ui_chatbot],
                outputs=[ui_chatbot, ui_audio_playback],
            )

            ui_audio_input.stop_recording(
                fn=self._handle_voice_chat,
                inputs=[ui_audio_input, ui_chatbot],
                outputs=[ui_chatbot, ui_audio_playback],
            )

            # Poll for auto-continue audio pushed from background reading timer
            app.load(
                fn=self._poll_pending_audio_stream,
                outputs=[ui_audio_playback],
            )

        print("[DESKTOP GUI] Launching Gradio app...")
        app.queue().launch(server_name="0.0.0.0", server_port=7861, theme=gr.themes.Monochrome())


if __name__ == "__main__":
    client_app = DesktopVideoStreamGUI()
    client_app.run(video_source="test_video.mp4", prompt="Calculator", server_ip="127.0.0.1", stream_to_server=True)
