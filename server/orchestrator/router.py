import time
from typing import Dict, Optional

from domain.intents import (
    MEMORY_INTENTS,
    READING_INTENTS,
    TRACKING_INTENTS,
    Intent,
    ParsedIntent,
)
from orchestrator.session import SessionContext


class RuleRouter:
    OCR_INTERVAL_SEC = 1.5

    def __init__(self, agents_by_name: Dict[str, object]):
        self.agents_by_name = agents_by_name

    def select(
        self,
        intent: ParsedIntent,
        context: SessionContext,
        frame_tick: bool = False,
        now: Optional[float] = None,
    ) -> str:
        now = now if now is not None else time.time()

        # Explicit intent routing — always takes priority
        if intent.intent in TRACKING_INTENTS:
            return "tracking"
        if intent.intent in MEMORY_INTENTS:
            return "memory"
        if intent.intent in READING_INTENTS:
            return "reading"

        # Frame ticks: only route to reading when actively scanning
        if frame_tick:
            if context.reading_state == "scanning":
                if now - context.last_frame_ocr_at >= self.OCR_INTERVAL_SEC:
                    return "reading"
            return ""

        # Conversational fallback during active reading (user asks a question mid-session)
        if context.reading_state in ("scanning", "reading_aloud", "paused"):
            if intent.intent == Intent.INFO:
                return "info"
            return "reading"

        return "info"
