# -*- coding: utf-8 -*-
import asyncio
import io
import threading
import traceback
import queue as queue_module

import cv2
import numpy as np
import pyaudio
import PIL.Image
import gradio as gr
from google import genai
from google.genai import types


API_KEY = "AQ.Ab8RN6JuCu7Hvy66o6kwRJIWqsfRjFHpscpbRdWGvXySO5MFkg"

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

LIVE_MODEL = "gemini-3.1-flash-live-preview"

client = genai.Client(api_key=API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a real-time assistant for a visually impaired person. "
    "You listen to the user's speech and call the appropriate intent function based on what they want to do. "
    "For obstacle warnings when video is active, keep them extremely short (under 4 words). "
    "Always call an intent function when you detect the user wants to perform an action or change state."
    "If the user intention or parameters are unclear, ask a clarifying question. "
)

# ── Intent function declarations ──────────────────────────────────────
INTENT_TOOLS = [
    {"function_declarations": [
        {
            "name": "start_tracking",
            "description": "User wants to track or find a specific object",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "The object or thing to track/find"},
                },
                "required": ["target"],
            },
        },
        {
            "name": "stop_tracking",
            "description": "User wants to stop tracking the current object",
        },
        {
            "name": "start_reading",
            "description": "User wants to enter reading mode",
        },
        {
            "name": "scan_page",
            "description": "User wants to scan or capture the current page/document",
        },
        {
            "name": "read_aloud",
            "description": "User wants the scanned content to be read aloud",
        },
        {
            "name": "pause_reading",
            "description": "User wants to pause the reading",
        },
        {
            "name": "continue_reading",
            "description": "User wants to continue or resume reading",
        },
        {
            "name": "back_sentence",
            "description": "User wants to go back to the previous sentence",
        },
        {
            "name": "forward_sentence",
            "description": "User wants to skip to the next sentence",
        },
        {
            "name": "flip_reading_direction",
            "description": "User wants to flip/toggle the reading direction (left-to-right or right-to-left)",
        },
        {
            "name": "read_again",
            "description": "User wants to start reading from the beginning again",
        },
        {
            "name": "stop_reading",
            "description": "User wants to exit reading mode",
        },
        {
            "name": "remember_object",
            "description": "User wants to remember/label what is currently in view",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "The name or label to assign to the object"},
                },
                "required": ["label"],
            },
        },
        {
            "name": "save_memory",
            "description": "User wants to scan the current view and save it as a named memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "The memory label/name to save under"},
                },
                "required": ["label"],
            },
        },
        {
            "name": "start_navigation",
            "description": "User wants to start navigating to a destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "The target destination or zone"},
                },
                "required": ["destination"],
            },
        },
        {
            "name": "set_destination",
            "description": "User wants to change the navigation destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "The new destination or zone"},
                },
                "required": ["destination"],
            },
        },
        {
            "name": "stop_navigation",
            "description": "User wants to stop/cancel navigation",
        },
        {
            "name": "ask_info",
            "description": "User is asking a general question about the scene or environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question"},
                },
                "required": ["question"],
            },
        },
    ]}
]

REALTIME_PROMPT = (
    "This is the user's POV. "
    "Is there any obstacle, hazard, or changing terrain directly ahead within 3 feet? "
    "Skip warnings for objects that are not in the path or are far away "
    "If YES, warn the user in under 4 words. (e.g. 'step down ahead', 'pothole ahead', 'person walking by', 'wet floor ahead'"
    "If NO, REMEMBER to output empty spacebar of '.' ONLY, NOTHING ELSE."
)

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    tools=INTENT_TOOLS,
    system_instruction=types.Content(parts=[types.Part(text=SYSTEM_INSTRUCTION)]),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

pya = pyaudio.PyAudio()
transcript_queue = queue_module.Queue()
frame_queue = queue_module.Queue(maxsize=2)
_stop_live = threading.Event()


def _encode_frame(frame_bgr, portrait: bool) -> PIL.Image.Image:
    if portrait:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(rgb)
    img.thumbnail([1024, 1024])
    return img


class AudioVideoLoop:
    def __init__(self, video_path: str | None = None, portrait: bool = False):
        self.video_path = video_path
        self.portrait = portrait
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.model_speaking = False
        self.SPEECH_RMS_THRESHOLD = 800

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        try:
            while True:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                if self.model_speaking:
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    rms = np.sqrt(np.mean(samples ** 2))
                    if rms < self.SPEECH_RMS_THRESHOLD:
                        continue
                payload = {"data": data, "mime_type": "audio/pcm;rate=16000"}
                try:
                    self.out_queue.put_nowait(payload)
                except asyncio.QueueFull:
                    _ = self.out_queue.get_nowait()
                    self.out_queue.put_nowait(payload)
        except asyncio.CancelledError:
            pass
        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE, output=True,
        )
        try:
            while True:
                bytestream = await self.audio_in_queue.get()
                self.model_speaking = True
                await asyncio.to_thread(stream.write, bytestream)
                if self.audio_in_queue.empty():
                    self.model_speaking = False
        except asyncio.CancelledError:
            pass
        finally:
            self.model_speaking = False
            if stream:
                stream.stop_stream()
                stream.close()

    async def receive_audio(self):
        gemini_buf = []
        try:
            while True:
                async for response in self.session.receive():
                    # Handle intent function calls
                    if response.tool_call:
                        function_responses = []
                        for fc in response.tool_call.function_calls:
                            params = dict(fc.args) if fc.args else {}
                            print(f"\n[INTENT] name={fc.name!r}  params={params}")
                            transcript_queue.put({
                                "role": "assistant",
                                "content": f"[Intent: {fc.name}] {params}",
                            })
                            function_responses.append(types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response={"result": "ok"},
                            ))
                        await self.session.send_tool_response(function_responses=function_responses)
                        continue

                    server_content = response.server_content
                    if server_content is None:
                        continue
                    if server_content.interrupted:
                        while not self.audio_in_queue.empty():
                            self.audio_in_queue.get_nowait()
                        self.model_speaking = False
                        if gemini_buf:
                            gemini_buf.clear()
                    if server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if part.inline_data:
                                self.audio_in_queue.put_nowait(part.inline_data.data)
                    if server_content.input_transcription:
                        transcript_queue.put({"role": "user", "content": server_content.input_transcription.text})
                    if server_content.output_transcription:
                        gemini_buf.append(server_content.output_transcription.text)
                    if server_content.turn_complete:
                        if gemini_buf:
                            transcript_queue.put({"role": "assistant", "content": "".join(gemini_buf)})
                            gemini_buf.clear()
        except asyncio.CancelledError:
            pass

    async def send_realtime(self):
        try:
            while True:
                msg = await self.out_queue.get()
                await self.session.send_realtime_input(
                    audio=types.Blob(data=msg["data"], mime_type=msg["mime_type"])
                )
        except asyncio.CancelledError:
            pass

    async def send_video_frames(self):
        if not self.video_path:
            return
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = int(total_frames / fps)
        try:
            for sec in range(max(duration_s, 1)):
                if _stop_live.is_set():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps * 3))
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                pil_img = _encode_frame(frame_bgr, self.portrait)
                # push to display queue (drop old frame if full)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue_module.Empty:
                        pass
                frame_queue.put_nowait(pil_img)
                # frame every second; prompt every 3 seconds
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG")
                await self.session.send_realtime_input(
                    video=types.Blob(data=buf.getvalue(), mime_type="image/jpeg")
                )
                await self.session.send_realtime_input(text=REALTIME_PROMPT)
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            pass
        finally:
            cap.release()

    async def _wait_for_stop(self):
        while not _stop_live.is_set():
            await asyncio.sleep(0.2)
        raise asyncio.CancelledError("Stop requested")

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=LIVE_MODEL, config=LIVE_CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self.send_video_frames())
                tg.create_task(self._wait_for_stop())

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)


# ── Session management ────────────────────────────────────────────────

_live_thread = None


def start_live(video_path, portrait):
    global _live_thread
    _stop_live.clear()
    while not frame_queue.empty():
        frame_queue.get_nowait()
    loop = asyncio.new_event_loop()
    al = AudioVideoLoop(video_path=video_path, portrait=portrait)

    def _run():
        loop.run_until_complete(al.run())

    _live_thread = threading.Thread(target=_run, daemon=True)
    _live_thread.start()
    return gr.update(interactive=False), gr.update(interactive=True)


def stop_live():
    _stop_live.set()
    return gr.update(interactive=True), gr.update(interactive=False)


def poll_updates(history, _frame):
    history = list(history)
    while not transcript_queue.empty():
        history.append(transcript_queue.get_nowait())
    new_frame = None
    while not frame_queue.empty():
        new_frame = frame_queue.get_nowait()
    return history, new_frame if new_frame is not None else _frame


def clear_all():
    _stop_live.set()
    while not transcript_queue.empty():
        transcript_queue.get_nowait()
    while not frame_queue.empty():
        frame_queue.get_nowait()
    return (
        None, None, [],
        gr.update(interactive=True), gr.update(interactive=False),
    )


# ── UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Gemini Navigation Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🦯 Gemini Navigation Assistant")

    with gr.Row(equal_height=True):
        # Left panel
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", height=300)
            portrait_cb = gr.Checkbox(label="Portrait video (rotate 90°)", value=False)
            current_frame = gr.Image(label="Current Frame", height=340, interactive=False)
            clear_btn = gr.Button("🗑 Clear")

        # Right panel
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Transcript", height=540)
            with gr.Row():
                start_btn = gr.Button("🎤 Start", variant="primary")
                stop_btn = gr.Button("⏹ Stop", variant="stop", interactive=False)
            gr.Markdown("*Audio via system mic & speakers. Frames sent at 1 fps to Gemini Live.*")

    timer = gr.Timer(1.0)

    start_btn.click(
        fn=start_live,
        inputs=[video_input, portrait_cb],
        outputs=[start_btn, stop_btn],
    )
    stop_btn.click(fn=stop_live, outputs=[start_btn, stop_btn])

    clear_btn.click(
        fn=clear_all,
        outputs=[video_input, current_frame, chatbot, start_btn, stop_btn],
    )

    timer.tick(
        fn=poll_updates,
        inputs=[chatbot, current_frame],
        outputs=[chatbot, current_frame],
    )


if __name__ == "__main__":
    demo.launch(share=False)
