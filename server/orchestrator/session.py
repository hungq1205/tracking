from dataclasses import dataclass, field
from typing import List, Literal, Optional

ReadingState = Literal["idle", "scanning", "reading_aloud", "paused"]
@dataclass
class SessionContext:
    active_agent: Optional[str] = None
    active_label: Optional[str] = None
    last_frame_ocr_at: float = 0.0
    last_intent_at: float = 0.0

    reading_state: ReadingState = "idle"
    reading_direction: str = "ltr"  # "ltr" | "rtl"

    scan_buffer: str = ""
    scan_buffer_char_count: int = 0
    memory_text_cache: str = ""  # snapshot of label's persisted text at scan start

    read_sentences: List[str] = field(default_factory=list)
    read_position: int = 0
