import tempfile
import os

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


class WhisperASR:
    def __init__(self, model_name: str = "base"):
        import whisper

        print(f"[SERVER] Initializing Whisper ASR ({model_name})...")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_data: bytes, mode: str = "general") -> str:
        if not audio_data:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        try:
            result = self.model.transcribe(
                tmp_path,
                language="en",
                initial_prompt=_PROMPTS.get(mode, ""),
            )
            return result["text"].strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
