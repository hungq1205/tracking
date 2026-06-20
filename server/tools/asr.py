import tempfile
import os


class WhisperASR:
    def __init__(self, model_name: str = "base"):
        from faster_whisper import WhisperModel

        print(f"[SERVER] Initializing Whisper ASR ({model_name})...")
        self.model = WhisperModel(model_name, device="cuda", compute_type="float16")

    def transcribe(self, audio_data: bytes) -> str:
        if not audio_data:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        try:
            segments, _ = self.model.transcribe(tmp_path)
            return " ".join(segment.text for segment in segments).strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
