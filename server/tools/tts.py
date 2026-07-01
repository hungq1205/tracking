from typing import Iterator

import numpy as np


class KokoroTTS:
    SAMPLE_RATE = 24000

    def __init__(self, lang_code: str = "a"):
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code=lang_code)

    def synthesize_pcm_chunks(
        self, text: str, voice: str = "af_heart", speed: float = 1.0
    ) -> Iterator[bytes]:
        """Yield raw 16-bit LE mono PCM chunks at 24 kHz, matching AudioChunk.pcm_data format."""
        if not text:
            return
        for _, _, audio in self.pipeline(text, voice=voice, speed=speed):
            pcm16 = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm16 * 32767).astype(np.int16)
            yield pcm16.tobytes()
