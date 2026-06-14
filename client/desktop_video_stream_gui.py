import cv2
import time
import gradio as gr
import os
import tempfile
from typing import Dict, Any, Generator
from concurrent.futures import ThreadPoolExecutor

from rpc_client.grpc_client import RemoteGroundingDINO, RemoteEmbedder
from core.local_models import LocalHandDetector
from core.object_tracker import GPUVIOAnchorBackend
from core.guidance_engine import GuidanceEngine
from core.helpers import CameraEmulator
from core.renderer import GUIRenderer
from core.interfaces import IClientApp

# Global executor for background renewal tasks (if any, though not directly used here for renewal)
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

    def _initialize_backend_services(self, server_ip, nfeatures, renewal_interval):
        """Modularized setup for gRPC proxies and tracking engines."""
        self.server_addr = f"{server_ip}:50051"
        print(f"[DESKTOP GUI] Connecting to Server at {self.server_addr}...")
        self.remote_detector = RemoteGroundingDINO(self.server_addr)
        self.remote_embedder = RemoteEmbedder(self.server_addr)
        self.local_hand = LocalHandDetector()
        
        self.backend = GPUVIOAnchorBackend(
            detector=self.remote_detector, 
            embedder=self.remote_embedder,
            nfeatures=int(nfeatures),
            renewal_interval=float(renewal_interval)
        )
        self.engine = GuidanceEngine(object_tracker=self.backend, hand_detector=self.local_hand)

    def _handle_tracking_logic(self, frame, text_prompt, fps, last_init_attempt):
        """Handles object initialization with a 1s retry interval or active tracking updates."""
        if not self.initialized:
            now = time.time()
            if (now - last_init_attempt) >= 1.0:
                print(f"[DESKTOP GUI] Attempting initialization for: {text_prompt}")
                obj_track = self.backend.initialize(frame, text_prompt)
                if obj_track.visible:
                    self.initialized = True
                    return self.engine._build_state(obj_track, self.local_hand.detect(frame), fps=fps), now
                print(f"[DESKTOP GUI] Not found: {text_prompt}")
                return None, now  # Failed, trigger retry in next second
            return None, last_init_attempt  # Waiting for 1s window
        
        # Active tracking
        return self.engine.update(frame, fps=fps), last_init_attempt

    def _maybe_stream_to_server(self, frame, stream_fps):
        """Throttled frame streaming to the remote VLM server."""
        if self.stream_to_server_flag:
            now = time.time()
            if (now - self.last_gui_send) > (1.0 / float(stream_fps)):
                self.remote_detector.send_gui_frame(frame)
                self.last_gui_send = now

    def _run_experiment_generator(self, video_file, text_prompt, server_ip, workflow_mode, stream_to_server_toggle, nfeatures, renewal_interval, stream_fps) -> Generator[Dict[str, Any], None, None]:
        if video_file is None:
            yield {
                ui_status: "Status: Please upload a video file."
            }
            return

        self._initialize_backend_services(server_ip, nfeatures, renewal_interval)
        self.stream_to_server_flag = stream_to_server_toggle
        self.emulator = CameraEmulator(video_file, target_fps=15)
        
        self.initialized = False
        self.last_gui_send = 0
        last_init_attempt = 0
        last_loop_time = time.perf_counter()

        for frame, _ in self.emulator.stream():
            self.latest_frame = frame.copy()
            t_start = time.perf_counter()
            
            # Calculate real-time FPS
            current_time = time.perf_counter()
            fps = 1.0 / (current_time - last_loop_time) if current_time > last_loop_time else 0.0
            last_loop_time = current_time

            self._maybe_stream_to_server(frame, stream_fps)

            # Mode Logic
            if workflow_mode == "Chat Only":
                yield {
                    ui_image: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ui_status: "Status: Chat Mode Active (Tracking Bypassed)"
                }
                continue

            # Process Tracking (Unified or Tracking Only)
            state, last_init_attempt = self._handle_tracking_logic(frame, text_prompt, fps, last_init_attempt)
            
            if state is None:
                yield {
                    ui_image: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ui_status: f"Status: Searching for '{text_prompt}'...\n(Retrying every 1s)"
                }
                continue

            # Final visualization for tracking modes
            latency = (time.perf_counter() - t_start) * 1000
            vis_frame = GUIRenderer.render(frame, state)
            active_anchors = len(state.object_track.debug.get("anchor_pts", [])) if "anchor_pts" in state.object_track.debug else state.object_track.debug.get("total_anchors", 0)
            status_text = (f"Status: {state.object_track.status}\n"
                           f"FPS: {state.fps:.1f} | Conf: {state.object_track.confidence:.2f} | Anchors: {active_anchors}\n"
                           f"Latency: {latency:.1f}ms\nInstruction: {state.instruction}")

            yield {ui_image: cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB), ui_status: status_text}

    def _handle_chat(self, message, history):
        history = history or ""
        message = message or ""
        if self.remote_detector is None:
            return history + "\nError: Client services not initialized. Click 'Run' first.", None
        if not self.stream_to_server_flag:
            return history + "\nError: VLM Chat requires 'Stream to Server' to be enabled.", None
        if self.latest_frame is None:
            return history + "\nError: No video frames captured yet.", None
            
        res = self.remote_detector.chat(self.latest_frame, message)
        new_history = history + f"\nUser: {message}\nAssistant: {res.response}"
        
        audio_path = None
        if res.audio_response:
            audio_path = os.path.join(tempfile.gettempdir(), "response.wav")
            with open(audio_path, "wb") as f:
                f.write(res.audio_response)
        return new_history, audio_path

    def _handle_voice_chat(self, audio_path, history):
        history = history or ""
        if audio_path is None:
            return history, None
        if self.remote_detector is None:
            return history + "\nError: Client services not initialized. Click 'Run' first.", None
        if not self.stream_to_server_flag:
            return history + "\nError: VLM Chat requires 'Stream to Server' to be enabled.", None
        if self.latest_frame is None:
            return history + "\nError: No video frames captured yet.", None
            
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        res = self.remote_detector.voice_chat(audio_data)
        new_history = history + f"\n{res.response}" # res.response already contains transcribed text
        
        out_audio = None
        if res.audio_response:
            out_audio = os.path.join(tempfile.gettempdir(), "voice_response.wav")
            with open(out_audio, "wb") as f:
                f.write(res.audio_response)
        return new_history, out_audio

    def run(self, video_source: str, prompt: str, server_ip: str, stream_to_server: bool):
        # Gradio UI components (defined globally for access within _run_experiment_generator)
        global ui_image, ui_status, ui_chat_output
        ui_image = gr.Image(label="Processed GPU Output Pipeline View")
        ui_status = gr.Textbox(label="Framework Metrics & Connection Status", lines=3, interactive=False)
        ui_chat_output = gr.Textbox(label="Assistant Response", interactive=False)

        with gr.Blocks(title="Desktop Object Tracker Client") as app:
            gr.Markdown("## Desktop Object Tracker Client - Video Stream")
            gr.Markdown("Upload a video file to simulate a camera feed and track objects.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    ui_video = gr.Video(label="Video Input Stream")
                    ui_workflow = gr.Radio(
                        choices=["Tracking Only", "Chat Only", "Unified"], 
                        value="Unified", 
                        label="Workflow Mode"
                    )
                    ui_server_ip = gr.Textbox(value=server_ip, label="gRPC Server IP")
                    
                    with gr.Accordion("Object Tracking Params", open=True):
                        ui_prompt = gr.Textbox(value=prompt, label="Target Object Prompt")
                        ui_nfeatures = gr.Slider(minimum=100, maximum=3000, step=100, value=800, label="ORB Anchors")
                        ui_renewal = gr.Slider(minimum=0.5, maximum=10.0, step=0.5, value=1.5, label="Renewal Interval (s)")
                    
                    with gr.Accordion("VLM Chat / Streaming Params", open=False):
                        ui_stream_toggle = gr.Checkbox(value=stream_to_server, label="Stream to Server GUI")
                        ui_stream_fps = gr.Slider(minimum=1, maximum=30, step=1, value=5, label="Server Stream FPS")
                    
                    btn_start = gr.Button("Run", variant="primary")
                    btn_stop = gr.Button("Stop", variant="stop")
                with gr.Column(scale=2):
                    ui_image.render()
                    ui_status.render()
                    
                    with gr.Group():
                        gr.Markdown("### Multimodal Chat")
                        with gr.Row(equal_height=True):
                            ui_chat_input = gr.Textbox(placeholder="Ask something about the current view...", label=None, scale=4, container=False)
                            ui_audio_input = gr.Audio(sources=["microphone"], type="filepath", show_label=False, scale=3, container=False)
                            btn_chat = gr.Button("Ask", variant="secondary", scale=1)
                        ui_chat_output.render()
                        ui_audio_playback = gr.Audio(visible=False, autoplay=True)

            run_event = btn_start.click(
                fn=self._run_experiment_generator,
                inputs=[ui_video, ui_prompt, ui_server_ip, ui_workflow, ui_stream_toggle, ui_nfeatures, ui_renewal, ui_stream_fps],
                outputs=[ui_image, ui_status]
            )
            btn_stop.click(fn=None, cancels=[run_event])
            
            btn_chat.click(
                fn=self._handle_chat,
                inputs=[ui_chat_input, ui_chat_output],
                outputs=[ui_chat_output, ui_audio_playback]
            )

            ui_audio_input.stop_recording(
                fn=self._handle_voice_chat,
                inputs=[ui_audio_input, ui_chat_output],
                outputs=[ui_chat_output, ui_audio_playback]
            )

        print("[DESKTOP GUI] Launching Gradio app...")
        app.queue().launch(server_name="0.0.0.0", server_port=7861, theme=gr.themes.Monochrome())

if __name__ == "__main__":
    client_app = DesktopVideoStreamGUI()
    client_app.run(video_source="test_video.mp4", prompt="Calculator", server_ip="127.0.0.1", stream_to_server=True)