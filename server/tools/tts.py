import io

import numpy as np
import soundfile as sf


class KokoroTTS:
    def __init__(self, lang_code: str = "a", gpu_lock=None, device: str = "cpu"):
        from kokoro import KPipeline

        print(f"[SERVER] Initializing Kokoro TTS on {device}...")
        self.pipeline = KPipeline(lang_code=lang_code, device=device)
        self.gpu_lock = gpu_lock

    def synthesize(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> bytes:
        if not text:
            return b""
        audio_out = []
        with self.gpu_lock:
            for _, _, audio in self.pipeline(text, voice=voice, speed=speed):
                audio_out.append(audio)
        if not audio_out:
            return b""
        full_audio = np.concatenate(audio_out)
        byte_io = io.BytesIO()
        sf.write(byte_io, full_audio, 24000, format="WAV")
        return byte_io.getvalue()
