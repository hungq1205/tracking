from __future__ import annotations

import re
from typing import List, Optional

from domain.intents import Intent, ParsedIntent
from interfaces import IIntentParser


def _strip_punct(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip()


def _match(patterns: List[str], text: str) -> Optional[re.Match]:
    for pat in patterns:
        m = re.fullmatch(pat, text, re.IGNORECASE)
        if m:
            return m
    return None


def _extract(m: re.Match, group: int = 1) -> str:
    try:
        return (m.group(group) or "").strip()
    except IndexError:
        return ""


class GeneralIntentParser(IIntentParser):
    _START_READING = [
        r"reading\s+mode",
        r"enter\s+reading(?:\s+mode)?",
        r"start\s+reading(?:\s+mode)?",
    ]
    _READ_SCREEN = [
        r"read\s+it\s+out\s+loud",
        r"read\s+this",
        r"read\s+aloud",
        r"read\s+it",
    ]
    _REMEMBER_OBJECT = [
        r"remember\s+that\s+is\s+(.+)",
        r"remember\s+that\s+as\s+(.+)",
        r"remember\s+this\s+is\s+(.+)",
        r"remember\s+this\s+as\s+(.+)",
        r"remember\s+this\s+(.+)",
        r"remember\s+(.+)",
    ]
    _SAVE_MEMORY = [
        r"scan\s+and\s+remember\s+my\s+(.+)",
        r"scan\s+then\s+remember\s+(.+)",
        r"scan\s+remember\s+(.+)",
    ]
    _START_TRACKING = [
        r"start\s+(?:tracking|track)(?:\s+me)(?:\s+to)?\s+(?:the\s+)?(.+)",
        r"start\s+navigating?\s+(?:me\s+)?to\s+(?:the\s+)?(.+)",
    ]

    def parse(self, text: str) -> ParsedIntent:
        original = text.strip()
        t = _strip_punct(original)

        if _match(self._START_READING, t):
            return ParsedIntent(intent=Intent.START_READING)

        if _match(self._READ_SCREEN, t):
            return ParsedIntent(intent=Intent.READ_SCREEN)

        m = _match(self._REMEMBER_OBJECT, t)
        if m:
            return ParsedIntent(intent=Intent.REMEMBER_OBJECT, label=_extract(m, 1))

        m = _match(self._SAVE_MEMORY, t)
        if m:
            label = _extract(m, 1).replace(" ", "_")
            return ParsedIntent(intent=Intent.SAVE_MEMORY, label=label)

        m = _match(self._START_TRACKING, t)
        if m:
            return ParsedIntent(intent=Intent.START_TRACKING, target=_extract(m, 1))

        return ParsedIntent(intent=Intent.INFO, question=original)


class ReadingIntentParser(IIntentParser):
    _STOP_READING = [
        r"quit\s+reading(?:\s+mode)?",
        r"exit\s+reading(?:\s+mode)?",
    ]
    _FLIP = [
        r"flip\s+direction",
    ]
    _READ_AGAIN = [
        r"read\s+again",
        r"start\s+over",
    ]
    _BACK = [
        r"go\s+back",
        r"repeat\s+last",
        r"backward",
    ]
    _FORWARD = [
        r"skip(?:\s+ahead)?",
        r"next\s+sentence",
        r"forward",
    ]
    _PAUSE = [
        r"stop",
        r"pause",
        r"wait",
        r"hold\s+on",
    ]
    _CONTINUE = [
        r"continue",
        r"keep\s+going",
        r"go\s+on",
        r"resume",
    ]
    _READ_ALOUD = [
        r"read(?:\s+it)?(?:\s+aloud)?",
    ]
    _SCAN = [
        r"scan(?:\s+this)?(?:\s+page)?",
        r"capture(?:\s+this)?",
    ]

    def parse(self, text: str) -> ParsedIntent:
        original = text.strip()
        t = _strip_punct(original)

        if _match(self._STOP_READING, t):
            return ParsedIntent(intent=Intent.STOP_READING)
        if _match(self._FLIP, t):
            return ParsedIntent(intent=Intent.FLIP_READING_DIRECTION)
        if _match(self._READ_AGAIN, t):
            return ParsedIntent(intent=Intent.READ_AGAIN)
        if _match(self._BACK, t):
            return ParsedIntent(intent=Intent.BACK_SENTENCE)
        if _match(self._FORWARD, t):
            return ParsedIntent(intent=Intent.FORWARD_SENTENCE)
        if _match(self._PAUSE, t):
            return ParsedIntent(intent=Intent.PAUSE_READING)
        if _match(self._CONTINUE, t):
            return ParsedIntent(intent=Intent.CONTINUE_READING)
        if _match(self._READ_ALOUD, t):
            return ParsedIntent(intent=Intent.READ_ALOUD)
        if _match(self._SCAN, t):
            return ParsedIntent(intent=Intent.SCAN_PAGE)

        return ParsedIntent(intent=Intent.INFO, question=original)


class TrackingIntentParser(IIntentParser):
    _STOP = [
        r"stop(?:\s+tracking)?",
        r"quit",
        r"exit",
        r"cancel",
    ]
    _START = [
        r"(?:track(?:\s+to)?|find|navigate\s+to|switch\s+to)\s+(?:the\s+)?(.+?)(?:\s+instead)?$",
    ]

    def parse(self, text: str) -> ParsedIntent:
        original = text.strip()
        t = _strip_punct(original)

        if _match(self._STOP, t):
            return ParsedIntent(intent=Intent.STOP_TRACKING)

        m = _match(self._START, t)
        if m:
            return ParsedIntent(intent=Intent.START_TRACKING, target=_extract(m, 1))

        return ParsedIntent(intent=Intent.INFO, question=original)


# Legacy alias
IntentParser = GeneralIntentParser
