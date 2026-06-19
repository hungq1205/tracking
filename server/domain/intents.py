from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class Intent(str, Enum):
    # Tracking
    START_TRACKING = "START_TRACKING"
    STOP_TRACKING = "STOP_TRACKING"
    # Reading session (mode entry/exit + within-mode navigation)
    START_READING = "START_READING"
    STOP_READING = "STOP_READING"
    SCAN_PAGE = "SCAN_PAGE"
    READ_ALOUD = "READ_ALOUD"
    PAUSE_READING = "PAUSE_READING"
    CONTINUE_READING = "CONTINUE_READING"
    BACK_SENTENCE = "BACK_SENTENCE"
    FORWARD_SENTENCE = "FORWARD_SENTENCE"
    READ_AGAIN = "READ_AGAIN"
    # One-shot read (no mode entry)
    READ_SCREEN = "READ_SCREEN"
    # In-session reading direction toggle
    FLIP_READING_DIRECTION = "FLIP_READING_DIRECTION"
    # Memory
    SAVE_MEMORY = "SAVE_MEMORY"
    READ_MEMORY = "READ_MEMORY"
    REMEMBER_OBJECT = "REMEMBER_OBJECT"
    # General
    INFO = "INFO"


TRACKING_INTENTS = {Intent.START_TRACKING, Intent.STOP_TRACKING}

MEMORY_INTENTS = {Intent.SAVE_MEMORY, Intent.READ_MEMORY, Intent.REMEMBER_OBJECT}

# Intents that route to the reading agent.
# Navigation intents are included here because ReadingIntentParser can emit them
# and the router needs to forward them to the reading agent.
READING_INTENTS = {
    Intent.START_READING,
    Intent.STOP_READING,
    Intent.SCAN_PAGE,
    Intent.READ_ALOUD,
    Intent.PAUSE_READING,
    Intent.CONTINUE_READING,
    Intent.BACK_SENTENCE,
    Intent.FORWARD_SENTENCE,
    Intent.READ_AGAIN,
    Intent.READ_SCREEN,
    Intent.FLIP_READING_DIRECTION,
}


@dataclass
class ParsedIntent:
    intent: Intent = Intent.INFO
    target: str = ""
    label: str = ""
    question: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParsedIntent":
        intent_str = data.get("intent", "INFO")
        try:
            intent = Intent(intent_str)
        except ValueError:
            intent = Intent.INFO
        return cls(
            intent=intent,
            target=data.get("target", "") or "",
            label=data.get("label", "") or "",
            question=data.get("question", "") or "",
            raw=data,
        )
