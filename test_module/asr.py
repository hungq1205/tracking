# pip install gradio openai-whisper sounddevice soundfile numpy

import io
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import gradio as gr
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


# =========================
# Whisper
# =========================

print("Loading Whisper...")
model = whisper.load_model("base.en")


_PROMPTS = {
    "general": (
        "reading mode, enter reading mode, start reading mode, "
        "read aloud, read this, read it out loud, read it, "
        "remember this is, remember this as, remember that is, remember that as, remember this, remember, "
        "scan and remember my, scan then remember, scan remember, "
        "start tracking me to the, start navigate me to the, "
        "object"
    ),
    "tracking": (
        "stop, stop tracking, quit, exit, cancel, "
        "track the, find the, navigate to the, switch to the, "
        "object"
    ),
    "reading": (
        "scan, scan this page, capture, capture this, read aloud, read it aloud, read it, read, "
        "pause, stop, wait, hold on, "
        "continue, keep going, go on, resume, "
        "go back, repeat last, backward, "
        "skip ahead, next sentence, forward, "
        "read again, start over, "
        "quit reading, exit reading, flip direction, "
        "page, book, paper"
    ),
}


def transcribe(audio_path, mode="general"):
    if not audio_path:
        return ""

    result = model.transcribe(
        audio_path,
        language="en",
        initial_prompt=_PROMPTS.get(mode, ""),
    )

    return result["text"].strip()


def transcribe_bytes(audio_bytes: bytes, mode="general") -> str:
    buf = io.BytesIO(audio_bytes)
    audio_array, sr = sf.read(buf, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]
    if sr != 16000:
        import resampy
        audio_array = resampy.resample(audio_array, sr, 16000)
    result = model.transcribe(
        audio_array,
        language="en",
        initial_prompt=_PROMPTS.get(mode, ""),
    )
    return result["text"].strip()


# =========================
# VAD
# =========================

SAMPLE_RATE = 16000
VAD_CHUNK = 512          # ~32 ms per chunk
VAD_THRESHOLD = 0.01     # RMS threshold for speech
VAD_SILENCE_CHUNKS = 47  # ~1500 ms silence before submit
VAD_MIN_SPEECH_CHUNKS = 5  # ~160 ms minimum to avoid noise triggers


class VoiceActivityDetector:
    """Always-on VAD: listens, detects speech by RMS, fires callback on silence."""

    def __init__(self, on_audio: Callable[[bytes], None]):
        self._on_audio = on_audio
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        buffer: list = []
        silent_chunks = 0
        speech_chunks = 0
        in_speech = False

        def callback(indata, _frames, _time_info, _status):
            nonlocal buffer, silent_chunks, speech_chunks, in_speech
            chunk = indata[:, 0].copy()
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= VAD_THRESHOLD:
                buffer.append(chunk)
                speech_chunks += 1
                silent_chunks = 0
                if not in_speech and speech_chunks >= VAD_MIN_SPEECH_CHUNKS:
                    in_speech = True
            else:
                if in_speech:
                    buffer.append(chunk)
                    silent_chunks += 1
                    if silent_chunks >= VAD_SILENCE_CHUNKS:
                        self._submit(buffer[:])
                        buffer.clear()
                        silent_chunks = 0
                        speech_chunks = 0
                        in_speech = False
                else:
                    speech_chunks = max(0, speech_chunks - 1)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=VAD_CHUNK,
            callback=callback,
        ):
            while not self._stop_event.is_set():
                time.sleep(0.05)

    def _submit(self, chunks: list):
        audio = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        self._on_audio(buf.getvalue())


# =========================
# Intent Models
# =========================

class Intent(Enum):
    START_READING = "START_READING"
    READ_SCREEN = "READ_SCREEN"

    REMEMBER_OBJECT = "REMEMBER_OBJECT"
    SAVE_MEMORY = "SAVE_MEMORY"

    START_TRACKING = "START_TRACKING"

    STOP_READING = "STOP_READING"
    FLIP_READING_DIRECTION = "FLIP_READING_DIRECTION"

    READ_AGAIN = "READ_AGAIN"
    BACK_SENTENCE = "BACK_SENTENCE"
    FORWARD_SENTENCE = "FORWARD_SENTENCE"

    PAUSE_READING = "PAUSE_READING"
    CONTINUE_READING = "CONTINUE_READING"

    READ_ALOUD = "READ_ALOUD"
    SCAN_PAGE = "SCAN_PAGE"

    STOP_TRACKING = "STOP_TRACKING"

    INFO = "INFO"


@dataclass
class ParsedIntent:
    intent: Intent
    label: str | None = None
    target: str | None = None
    question: str | None = None


# =========================
# Intent Helpers
# =========================

def _strip_punct(text):
    return re.sub(r"[^\w\s]", "", text).strip()


def _match(patterns, text):
    for pat in patterns:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            return m
    return None


def _extract(m, group=1):
    try:
        return (m.group(group) or "").strip()
    except:
        return ""


# =========================
# Parser
# =========================

class IntentParser:

    START_READING = [
        r"reading\s+mode",
        r"enter\s+reading(?:\s+mode)?",
        r"start\s+reading(?:\s+mode)?",
    ]

    READ_SCREEN = [
        r"read\s+it\s+out\s+loud",
        r"read\s+this\s+out\s+loud",
        r"read\s+out\s+loud",
        r"read\s+this",
        r"read\s+it",
        r"read\s+aloud",
    ]

    REMEMBER = [
        r"remember\s+that\s+is\s+(.+)",
        r"remember\s+that\s+as\s+(.+)",
        r"remember\s+this\s+is\s+(.+)",
        r"remember\s+this\s+as\s+(.+)",
        r"remember\s+this\s+(.+)",
        r"remember\s+(.+)",
    ]

    SAVE = [
        r"scan\s+and\s+remember\s+my\s+(.+)",
        r"scan\s+then\s+remember\s+(.+)",
        r"scan\s+remember\s+(.+)",
    ]

    TRACK = [
        r"start\s+(?:tracking|track)(?:\s+me)(?:\s+to)?\s+(?:the\s+)?(.+)",
        r"start\s+navigating?\s+(?:me\s+)?to\s+(?:the\s+)?(.+)",
    ]

    STOP_TRACK = [
        r"stop(?:\s+tracking)?",
        r"quit",
        r"exit",
        r"cancel",
    ]

    def parse(self, text):

        original = text.strip()
        t = _strip_punct(original).lower()

        if _match(self.START_READING, t):
            return ParsedIntent(Intent.START_READING)

        if _match(self.READ_SCREEN, t):
            return ParsedIntent(Intent.READ_SCREEN)

        m = _match(self.REMEMBER, t)
        if m:
            return ParsedIntent(
                Intent.REMEMBER_OBJECT,
                label=_extract(m)
            )

        m = _match(self.SAVE, t)
        if m:
            return ParsedIntent(
                Intent.SAVE_MEMORY,
                label=_extract(m).replace(" ", "_")
            )

        m = _match(self.TRACK, t)
        if m:
            return ParsedIntent(
                Intent.START_TRACKING,
                target=_extract(m)
            )

        if _match(self.STOP_TRACK, t):
            return ParsedIntent(Intent.STOP_TRACKING)

        return ParsedIntent(
            Intent.INFO,
            question=original
        )


parser = IntentParser()


# =========================
# Manuals
# =========================

_MANUALS = {
    "general": """**General mode commands**
- `reading mode` / `enter reading` / `start reading` → START_READING
- `read this` / `read it` / `read aloud` / `read it out loud` → READ_SCREEN
- `remember this <label>` / `remember this as <label>` / `remember this is <label>` → REMEMBER_OBJECT
- `remember that is <label>` / `remember that as <label>` / `remember <label>` → REMEMBER_OBJECT
- `scan and remember my <label>` / `scan then remember <label>` / `scan remember <label>` → SAVE_MEMORY
- `start tracking me to <object>` / `start navigating to <object>` → START_TRACKING
- anything else → INFO""",

    "tracking": """**Tracking mode commands**
- `stop` / `stop tracking` / `quit` / `exit` / `cancel` → STOP_TRACKING
- `track <object>` / `find <object>` / `navigate to <object>` → START_TRACKING
- `switch to <object>` / `track <object> instead` → START_TRACKING
- anything else → INFO""",

    "reading": """**Reading mode commands**
- `scan` / `scan this` / `scan this page` / `capture` / `capture this` → SCAN_PAGE
- `read` / `read it` / `read aloud` → READ_ALOUD
- `pause` / `stop` / `wait` / `hold on` → PAUSE_READING
- `continue` / `resume` / `keep going` / `go on` → CONTINUE_READING
- `go back` / `backward` / `repeat last` → BACK_SENTENCE
- `skip` / `skip ahead` / `next sentence` / `forward` → FORWARD_SENTENCE
- `read again` / `start over` → READ_AGAIN
- `flip direction` → FLIP_READING_DIRECTION
- `quit reading` / `exit reading` → STOP_READING
- anything else → INFO""",
}


# =========================
# Pipeline
# =========================

def run(audio, mode):

    text = transcribe(audio, mode=mode)

    if not text:
        return "", ""

    parsed = parser.parse(text)

    result = (
        f"intent={parsed.intent.value}\n"
        f"label={parsed.label}\n"
        f"target={parsed.target}\n"
        f"question={parsed.question}"
    )

    return text, result


# =========================
# VAD state
# =========================

_pending_result: Optional[tuple] = None  # (transcript, parsed)
_current_mode = "general"


def _on_vad_audio(audio_bytes: bytes):
    global _pending_result
    text = transcribe_bytes(audio_bytes, mode=_current_mode)
    if not text:
        return
    parsed = parser.parse(text)
    result = (
        f"intent={parsed.intent.value}\n"
        f"label={parsed.label}\n"
        f"target={parsed.target}\n"
        f"question={parsed.question}"
    )
    _pending_result = (text, result)


vad = VoiceActivityDetector(on_audio=_on_vad_audio)
vad.start()


def _poll_vad():
    global _pending_result
    while True:
        time.sleep(0.3)
        result = _pending_result
        if result is not None:
            _pending_result = None
            yield result[0], result[1]
        else:
            yield gr.update(), gr.update()


# =========================
# UI
# =========================

with gr.Blocks() as app:

    gr.Markdown("# Whisper Intent Test")
    gr.Markdown("*Microphone is always open — speak naturally, results appear after silence.*")

    mode = gr.Radio(
        choices=["general", "tracking", "reading"],
        value="general",
        label="Mode (context biasing)",
    )

    transcript = gr.Textbox(
        label="Transcript"
    )

    parsed = gr.Textbox(
        label="Parsed Intent",
        lines=6
    )

    manual = gr.Markdown(value=_MANUALS["general"])

    def _on_mode_change(m):
        global _current_mode
        _current_mode = m
        return _MANUALS[m]

    mode.change(fn=_on_mode_change, inputs=mode, outputs=manual)

    app.load(fn=_poll_vad, outputs=[transcript, parsed])

app.launch()
