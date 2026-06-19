import re
from typing import List, Optional

from agents.base import AgentRequest, AgentResult, BaseAgent
from domain.intents import Intent
from tools.memory_store import filter_new_sentences


_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_SENTENCE_WORDS = 50


def _split_sentences(text: str) -> List[str]:
    raw = _SENT_RE.split(text.strip())
    sentences: List[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # Long fragment with no punctuation: split at word boundary
        words = s.split()
        while len(words) > _MAX_SENTENCE_WORDS:
            sentences.append(" ".join(words[:_MAX_SENTENCE_WORDS]))
            words = words[_MAX_SENTENCE_WORDS:]
        part = " ".join(words)
        if sentences and len(sentences[-1].split()) < 6:
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    return sentences or [text.strip()]



class ReadingAgent(BaseAgent):
    name = "reading"

    def __init__(self, ocr, rag_store=None):
        self.ocr = ocr
        self.rag_store = rag_store

    def handle(self, request: AgentRequest) -> AgentResult:
        intent = request.intent.intent
        ctx = request.context

        if intent == Intent.READ_SCREEN:
            return self._read_screen(request)
        if intent == Intent.START_READING:
            return self._start(request)
        if intent == Intent.STOP_READING:
            return self._stop(request)
        if intent == Intent.SCAN_PAGE:
            return self._scan_page(request)
        if intent == Intent.READ_ALOUD:
            return self._read_aloud(request)
        if intent == Intent.PAUSE_READING:
            return self._pause(request)
        if intent == Intent.CONTINUE_READING:
            return self._continue(request)
        if intent == Intent.BACK_SENTENCE:
            return self._navigate(request, -1)
        if intent == Intent.FORWARD_SENTENCE:
            return self._navigate(request, +1)
        if intent == Intent.READ_AGAIN:
            return self._read_again(request)
        if intent == Intent.FLIP_READING_DIRECTION:
            return self._flip_direction(request)

        # Frame tick
        if request.frame_tick and ctx.reading_state == "scanning":
            return self._frame_tick(request)

        return AgentResult(agent_name=self.name, state="IDLE", reply_text="", speak=False)

    # ── state transitions ─────────────────────────────────────────────────────

    def _read_screen(self, request: AgentRequest) -> AgentResult:
        """One-shot OCR+TTS. Does not enter reading mode or change any state."""
        if request.frame is None:
            return AgentResult(
                agent_name=self.name, state="IDLE",
                reply_text="No frame available to read.", speak=True,
            )
        direction = request.context.reading_direction
        text = self.ocr.read_text(request.frame, direction=direction)
        if not text:
            return AgentResult(
                agent_name=self.name, state="IDLE",
                reply_text="No text found on screen.", speak=True,
            )
        return AgentResult(
            agent_name=self.name,
            state="SCREEN_READ",
            reply_text=text,
            speak=True,
        )

    def _start(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        label = request.intent.label or ctx.active_label or "reading"
        ctx.reading_state = "scanning"
        ctx.scan_buffer = ""
        ctx.scan_buffer_char_count = 0
        ctx.active_label = label
        ctx.read_sentences = []
        ctx.read_position = 0
        ctx.memory_text_cache = ""
        if self.rag_store and label:
            try:
                ctx.memory_text_cache = self.rag_store.get_full_text(label) or ""
            except Exception:
                pass
        return AgentResult(
            agent_name=self.name,
            state="STARTED",
            payload={"label": label},
            reply_text=f"Reading mode active. Label: '{label}'.",
            speak=True,
        )

    def _stop(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        label = ctx.active_label
        self._clear_state(ctx)
        return AgentResult(
            agent_name=self.name,
            state="STOPPED",
            payload={"label": label},
            reply_text="Reading stopped.",
            speak=True,
        )

    def _scan_page(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        if ctx.reading_state != "scanning":
            # Auto-start in scan mode then scan immediately
            self._start(request)
        return self._do_ocr_and_append(request, speak=False, state="SCANNING")

    def _frame_tick(self, request: AgentRequest) -> AgentResult:
        return self._do_ocr_and_append(request, speak=False, state="SCANNING")

    def _do_ocr_and_append(self, request: AgentRequest, speak: bool, state: str) -> AgentResult:
        ctx = request.context
        if request.frame is None:
            return AgentResult(agent_name=self.name, state=state, reply_text="", speak=False)

        blocks = self.ocr.read_blocks(request.frame, direction=ctx.reading_direction)
        if not blocks:
            return AgentResult(agent_name=self.name, state=state, reply_text="", speak=False)

        # existing text = current session buffer + persisted memory loaded at scan start
        existing = ctx.scan_buffer + "\n" + ctx.memory_text_cache

        new_parts = []
        for block in blocks:
            filtered = filter_new_sentences(block, existing)
            if filtered:
                new_parts.append(filtered)
                existing += "\n" + filtered  # later blocks see earlier ones within this tick

        if not new_parts:
            return AgentResult(agent_name=self.name, state=state, reply_text="", speak=False)

        new_text = "\n".join(new_parts)
        ctx.scan_buffer = f"{ctx.scan_buffer}\n{new_text}".strip()
        ctx.scan_buffer_char_count = len(ctx.scan_buffer)

        return AgentResult(
            agent_name=self.name,
            state=state,
            payload={"char_count": ctx.scan_buffer_char_count, "label": ctx.active_label},
            reply_text=new_text if speak else "",
            speak=speak and bool(new_text),
        )

    def _read_aloud(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        text = ctx.scan_buffer
        if not text:
            return AgentResult(
                agent_name=self.name,
                state="EMPTY",
                reply_text="No text has been scanned yet.",
                speak=True,
            )
        ctx.read_sentences = _split_sentences(text)
        ctx.read_position = 0
        ctx.reading_state = "reading_aloud"
        return self._speak_current(ctx)

    def _pause(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        ctx.reading_state = "paused"
        return AgentResult(
            agent_name=self.name,
            state="PAUSED",
            payload={"sentence_index": ctx.read_position, "total": len(ctx.read_sentences)},
            reply_text="",
            speak=False,
        )

    def _continue(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        if ctx.reading_state == "paused":
            ctx.reading_state = "reading_aloud"
        if ctx.read_position >= len(ctx.read_sentences):
            self._clear_state(ctx)
            return AgentResult(
                agent_name=self.name,
                state="DONE_READING",
                payload={"finished": True},
                reply_text="",
                speak=False,
            )
        return self._speak_current(ctx)

    def _navigate(self, request: AgentRequest, delta: int) -> AgentResult:
        ctx = request.context
        ctx.read_position = max(0, min(len(ctx.read_sentences) - 1, ctx.read_position + delta))
        if ctx.reading_state not in ("reading_aloud", "paused"):
            ctx.reading_state = "reading_aloud"
        return self._speak_current(ctx)

    def _read_again(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        if not ctx.read_sentences:
            # Re-split from buffer if we have content
            if ctx.scan_buffer:
                ctx.read_sentences = _split_sentences(ctx.scan_buffer)
            else:
                return AgentResult(
                    agent_name=self.name,
                    state="EMPTY",
                    reply_text="Nothing to read again.",
                    speak=True,
                )
        ctx.read_position = 0
        ctx.reading_state = "reading_aloud"
        return self._speak_current(ctx)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _speak_current(self, ctx) -> AgentResult:
        pos = ctx.read_position
        sentence = ctx.read_sentences[pos]
        ctx.read_position = pos + 1
        return AgentResult(
            agent_name=self.name,
            state="READING_ALOUD",
            payload={"sentence_index": pos, "total": len(ctx.read_sentences)},
            reply_text=sentence,
            speak=True,
        )

    def _flip_direction(self, request: AgentRequest) -> AgentResult:
        ctx = request.context
        ctx.reading_direction = "rtl" if ctx.reading_direction == "ltr" else "ltr"
        label = "right to left" if ctx.reading_direction == "rtl" else "left to right"
        return AgentResult(
            agent_name=self.name,
            state="DIRECTION_SET",
            payload={"direction": ctx.reading_direction},
            reply_text=f"Reading direction set to {label}.",
            speak=True,
        )

    def _flush_buffer(self, ctx) -> None:
        if ctx.scan_buffer and ctx.active_label and self.rag_store is not None:
            try:
                self.rag_store.add_text(ctx.active_label, ctx.scan_buffer, source="reading_session")
            except Exception as e:
                print(f"[ReadingAgent] Failed to flush buffer to RAG: {e}")

    def _clear_state(self, ctx) -> None:
        ctx.reading_state = "idle"
        ctx.scan_buffer = ""
        ctx.scan_buffer_char_count = 0
        ctx.memory_text_cache = ""
        ctx.read_sentences = []
        ctx.read_position = 0
        ctx.active_agent = None
