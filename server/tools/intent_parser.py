import json
import re

from domain.intents import ParsedIntent


_GENERAL_PROMPT = (
    "You are an intent parser for an assistive vision robot for vision-impaired users. "
    "Extract intent and fields as JSON. Only output a JSON object, nothing else.\n"
    "Intents:\n"
    "- START_TRACKING: user wants to track/find an object. Fields: target\n"
    "- STOP_TRACKING: user wants to stop tracking. Phrases: 'stop tracking', 'quit', 'exit'\n"
    "- START_READING: user wants to enter a reading session. Fields: label (optional). Phrases: 'enter reading', 'reading mode'\n"
    "- READ_SCREEN: user wants to read text on screen right now, one-shot, no session. "
    "Phrases: 'read it out loud', 'read what's on screen', 'read this'\n"
    "- REMEMBER_OBJECT: user identifies an object they want remembered. "
    "Extract the object name as label. Phrases: 'this is my phone', 'this is the AC remote', 'remember this as my wallet'\n"
    "- SAVE_MEMORY: user wants to save scanned text to persistent memory. Fields: label\n"
    "- INFO: general conversation, questions about the scene, questions about saved items, or anything else\n"
    "Examples:\n"
    "'Find the backpack' -> {\"intent\": \"START_TRACKING\", \"target\": \"backpack\"}\n"
    "'reading mode' -> {\"intent\": \"START_READING\"}\n"
    "'read it out loud' -> {\"intent\": \"READ_SCREEN\"}\n"
    "'this is my phone' -> {\"intent\": \"REMEMBER_OBJECT\", \"label\": \"my phone\"}\n"
    "'this is the AC remote' -> {\"intent\": \"REMEMBER_OBJECT\", \"label\": \"AC remote\"}\n"
    "'remember this as my medicine' -> {\"intent\": \"REMEMBER_OBJECT\", \"label\": \"medicine\"}\n"
    "'save this as my grocery list' -> {\"intent\": \"SAVE_MEMORY\", \"label\": \"grocery_list\"}\n"
    "'what do I need to buy?' -> {\"intent\": \"INFO\"}\n"
    "'what is on my shopping list?' -> {\"intent\": \"INFO\"}\n"
    "'what do you see?' -> {\"intent\": \"INFO\"}\n"
)

_READING_PROMPT = (
    "You are an intent parser for an assistive reading mode. "
    "The user is currently in a reading session. Extract intent as JSON. Only output a JSON object.\n"
    "Intents:\n"
    "- SCAN_PAGE: scan current page/screen into buffer. Phrases: 'scan this', 'scan this page', 'capture this'\n"
    "- READ_ALOUD: read the accumulated buffer aloud. Phrases: 'read it', 'read aloud', 'read what you have'\n"
    "- PAUSE_READING: pause playback. Phrases: 'pause', 'stop for now', 'wait', 'hold on'\n"
    "- CONTINUE_READING: resume playback. Phrases: 'continue', 'keep going', 'next', 'go on', 'resume'\n"
    "- BACK_SENTENCE: go back one sentence. Phrases: 'go back', 'back', 'back a sentence', 'repeat last'\n"
    "- FORWARD_SENTENCE: skip to next sentence. Phrases: 'skip', 'next sentence', 'forward', 'skip ahead'\n"
    "- READ_AGAIN: restart from beginning. Phrases: 'read again', 'from the beginning', 'restart', 'start over'\n"
    "- FLIP_READING_DIRECTION: change column reading order. Phrases: 'flip direction', 'switch direction'\n"
    "- STOP_READING: exit reading mode entirely. Phrases: 'quit reading', 'exit reading mode', 'done'\n"
    "- INFO: a question or comment unrelated to reading navigation\n"
    "Examples:\n"
    "'scan this page' -> {\"intent\": \"SCAN_PAGE\"}\n"
    "'read it' -> {\"intent\": \"READ_ALOUD\"}\n"
    "'read it aloud' -> {\"intent\": \"READ_ALOUD\"}\n"
    "'pause' -> {\"intent\": \"PAUSE_READING\"}\n"
    "'continue' -> {\"intent\": \"CONTINUE_READING\"}\n"
    "'go back' -> {\"intent\": \"BACK_SENTENCE\"}\n"
    "'skip ahead' -> {\"intent\": \"FORWARD_SENTENCE\"}\n"
    "'read again' -> {\"intent\": \"READ_AGAIN\"}\n"
    "'flip direction' -> {\"intent\": \"FLIP_READING_DIRECTION\"}\n"
    "'exit reading mode' -> {\"intent\": \"STOP_READING\"}\n"
    "'what page is this?' -> {\"intent\": \"INFO\"}\n"
)

_TRACKING_PROMPT = (
    "You are an intent parser for an assistive tracking mode. "
    "The user is currently tracking an object. Extract intent as JSON. Only output a JSON object.\n"
    "Intents:\n"
    "- STOP_TRACKING: stop tracking entirely. Phrases: 'stop', 'quit', 'exit', 'found it', 'never mind', 'stop tracking', 'cancel'\n"
    "- START_TRACKING: switch to track a different object. Fields: target. Phrases: 'track the X instead', 'find the X', 'switch to X'\n"
    "- INFO: questions about tracking state or the scene. Phrases: 'where is it?', 'am I close?', 'how far?'\n"
    "Examples:\n"
    "'stop' -> {\"intent\": \"STOP_TRACKING\"}\n"
    "'found it' -> {\"intent\": \"STOP_TRACKING\"}\n"
    "'never mind' -> {\"intent\": \"STOP_TRACKING\"}\n"
    "'track the phone instead' -> {\"intent\": \"START_TRACKING\", \"target\": \"phone\"}\n"
    "'find the keys' -> {\"intent\": \"START_TRACKING\", \"target\": \"keys\"}\n"
    "'am I close?' -> {\"intent\": \"INFO\"}\n"
    "'where is it?' -> {\"intent\": \"INFO\"}\n"
)


class GeneralIntentParser:
    """Used when not in any active mode (idle)."""

    def __init__(self, pipe):
        self.pipe = pipe

    def parse(self, text: str) -> ParsedIntent:
        return _run_parse(self.pipe, _GENERAL_PROMPT, text)


class ReadingIntentParser:
    """Used when reading_state != 'idle'. Handles navigation and STOP_READING only."""

    def __init__(self, pipe):
        self.pipe = pipe

    def parse(self, text: str) -> ParsedIntent:
        return _run_parse(self.pipe, _READING_PROMPT, text)


class TrackingIntentParser:
    """Used when active_agent == 'tracking'. Handles STOP_TRACKING, retarget, and INFO."""

    def __init__(self, pipe):
        self.pipe = pipe

    def parse(self, text: str) -> ParsedIntent:
        return _run_parse(self.pipe, _TRACKING_PROMPT, text)


def _run_parse(pipe, system_prompt: str, text: str) -> ParsedIntent:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    out = pipe(messages, max_new_tokens=80, do_sample=False)[0]["generated_text"][-1]["content"]
    try:
        match = re.search(r"\{.*\}", out, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"intent": "INFO"}
        return ParsedIntent.from_dict(data)
    except Exception:
        return ParsedIntent()


# Legacy alias
IntentParser = GeneralIntentParser
