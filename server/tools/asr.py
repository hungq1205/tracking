import tempfile
import os


class WhisperASR:
    def __init__(self, model_name: str = "base"):
        import whisper

        print(f"[SERVER] Initializing Whisper ASR ({model_name})...")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_data: bytes) -> str:
        if not audio_data:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        try:
            result = self.model.transcribe(tmp_path)
            return result["text"].strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
