import re
import io
import queue
import threading
import asyncio
from google import genai
from google.genai import types

_READING_CTX_PATTERN = re.compile(
    r'<<<READING_CONTEXT_START>>>\n.*?\n<<<READING_CONTEXT_END>>>\n',
    re.DOTALL,
)

_MODEL = "gemini-3.1-flash-live-preview"


class GeminiLiveVLM:
    """
    Adapts the Gemini Live bidirectional API to the existing synchronous VLM interface.
    Runs a persistent async session in a background daemon thread.
    """

    def __init__(self, api_key: str, long_term_memory: str = ""):
        self._api_key = api_key
        self._long_term_memory = long_term_memory
        self._base_instruction = (
            "Short noun phrases only, comma-separated. No full sentences. "
            "Example: chair left, bottle on table, socks floor, person walking."
        )

        # Public attributes expected by callers
        self.chunk_index = 0
        self.full_conversation_history = []
        self.max_new_tokens = 20  # kept for update_params compatibility

        # Cross-thread frame delivery
        self._frame_queue = queue.Queue()

        # Proactive background responses from video stream
        self._bg_response_buffer = []
        self._response_lock = threading.Lock()

        # Pending blocking chat() call: (text, threading.Event, list[str])
        self._pending_query = None
        self._pending_query_lock = threading.Lock()

        # Async session handle and readiness gate
        self._session = None
        self._session_ready = threading.Event()

        # Active asyncio Task for the current session (used for cancellation)
        self._session_task = None

        # Background event loop running in daemon thread
        self._loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._bg_thread.start()

        asyncio.run_coroutine_threadsafe(self._launch_session(), self._loop)

    # ── system instruction ─────────────────────────────────────────────────────

    @property
    def _system_instruction(self) -> str:
        parts = [self._base_instruction]
        if self._long_term_memory:
            parts.append(f"[LONG-TERM MEMORY / CONTEXT]:\n{self._long_term_memory}")
        return "\n".join(parts)

    # ── async internals ────────────────────────────────────────────────────────

    async def _launch_session(self):
        self._session_task = asyncio.create_task(self._run_session())

    async def _run_session(self):
        client = genai.Client(api_key=self._api_key)
        config = types.LiveConnectConfig(
            response_modalities=[types.LiveModality.TEXT],
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=self._system_instruction)]
            ),
        )
        try:
            async with client.aio.live.connect(model=_MODEL, config=config) as session:
                self._session = session
                self._session_ready.set()
                print(f"[GeminiLiveVLM] Session open.")
                await asyncio.gather(
                    self._video_sender(session),
                    self._response_receiver(session),
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[GeminiLiveVLM] Session error: {e}")
        finally:
            self._session = None
            self._session_ready.clear()
            print("[GeminiLiveVLM] Session closed.")

    async def _video_sender(self, session):
        while True:
            frames = []
            try:
                while True:
                    frames.append(self._frame_queue.get_nowait())
            except queue.Empty:
                pass

            for pil_image in frames:
                buf = io.BytesIO()
                pil_image.save(buf, format="JPEG")
                await session.send_realtime_input(
                    media_chunks=[types.Blob(data=buf.getvalue(), mime_type="image/jpeg")]
                )

            await asyncio.sleep(0.05)

    async def _response_receiver(self, session):
        current_turn_parts = []
        async for response in session.receive():
            if not response.server_content:
                continue

            if response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.text:
                        current_turn_parts.append(part.text)

            if response.server_content.turn_complete and current_turn_parts:
                complete_text = "".join(current_turn_parts).strip()
                current_turn_parts = []

                if not complete_text:
                    continue

                with self._pending_query_lock:
                    pending = self._pending_query

                if pending is not None and not pending[1].is_set():
                    _, event, holder = pending
                    holder.append(complete_text)
                    event.set()
                else:
                    with self._response_lock:
                        self._bg_response_buffer.append(complete_text)

    async def _reconnect(self):
        if self._session_task and not self._session_task.done():
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass
        self._session_task = asyncio.create_task(self._run_session())

    # ── public interface ───────────────────────────────────────────────────────

    def strip_reading_context(self):
        modified = False

        cleaned = _READING_CTX_PATTERN.sub("", self._long_term_memory)
        if cleaned != self._long_term_memory:
            self._long_term_memory = cleaned
            modified = True

        for turn in self.full_conversation_history:
            if turn.get("role") != "user":
                continue
            content = turn["content"]
            if isinstance(content, str) and "<<<READING_CONTEXT_START>>>" in content:
                turn["content"] = _READING_CTX_PATTERN.sub("", content)
                modified = True

        if modified:
            print("[GeminiLiveVLM] Stripped reading context. Reconnecting with updated system instruction.")
            asyncio.run_coroutine_threadsafe(self._reconnect(), self._loop)

    def reset(self):
        self.chunk_index = 0
        self.full_conversation_history = []

        with self._response_lock:
            self._bg_response_buffer.clear()

        with self._pending_query_lock:
            self._pending_query = None

        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        asyncio.run_coroutine_threadsafe(self._reconnect(), self._loop)
        print("[GeminiLiveVLM] State cleared. Session reconnecting.")

    def update_params(self, params: dict):
        if "base_instruction" in params:
            self._base_instruction = params["base_instruction"]
        if "max_new_tokens" in params:
            self.max_new_tokens = params["max_new_tokens"]
        # text_round / visual_round are Qwen-specific; accepted and ignored

    def push_frame(self, pil_image):
        self._frame_queue.put(pil_image)

    def chat(self, query: str) -> str:
        if not self._session_ready.wait(timeout=15):
            return "[GeminiLiveVLM] Session not ready"

        event = threading.Event()
        holder = []
        with self._pending_query_lock:
            self._pending_query = (query, event, holder)

        try:
            asyncio.run_coroutine_threadsafe(
                self._session.send_realtime_input(text=query), self._loop
            ).result(timeout=5)
        except Exception as e:
            with self._pending_query_lock:
                self._pending_query = None
            return f"[GeminiLiveVLM] Send error: {e}"

        if not event.wait(timeout=30):
            with self._pending_query_lock:
                self._pending_query = None
            return "[GeminiLiveVLM] Response timeout"

        with self._pending_query_lock:
            self._pending_query = None

        response = holder[0] if holder else ""
        self.full_conversation_history.append({"role": "user", "content": query})
        self.full_conversation_history.append({"role": "assistant", "content": response})
        return response

    def process_video_step(self, query=None) -> str:
        if query:
            return self.chat(query)

        with self._response_lock:
            if not self._bg_response_buffer:
                self.chunk_index += 1
                return ""
            response = " ".join(self._bg_response_buffer)
            self._bg_response_buffer.clear()

        self.full_conversation_history.append({"role": "assistant", "content": response})
        self.chunk_index += 1
        return response
