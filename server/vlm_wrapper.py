import io
import queue
import threading
import asyncio
from google import genai
from google.genai import types

_MODEL = "gemini-3.1-flash-live-preview"


class GeminiLiveVLM:
    """
    Persistent Gemini Live session. Video frames stream continuously via push_frame();
    RAG and reading context are injected asynchronously via inject_context();
    blocking text queries go through chat().
    """

    def __init__(self, api_key: str, long_term_memory: str = ""):
        self._api_key = api_key
        self._long_term_memory = long_term_memory
        self._base_instruction = (
            "You assist a vision-impaired user. What you see is the user's POV. "
            "Short noun phrases, comma-separated. No full sentences. "
            "Example: bottle on table, person walking, phone on desk."
        )

        self._frame_queue = queue.Queue()

        # Pending blocking chat() call: (text, threading.Event, list[str], is_cancelled_flag)
        self._pending_query = None
        self._pending_query_lock = threading.Lock()

        self._session = None
        self._session_ready = threading.Event()
        self._session_task = None

        self._loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._bg_thread.start()

        asyncio.run_coroutine_threadsafe(self._launch_session(), self._loop)

    @property
    def _system_instruction(self) -> str:
        parts = [self._base_instruction]
        if self._long_term_memory:
            parts.append(f"[LONG-TERM MEMORY / CONTEXT]:\n{self._long_term_memory}")
        return "\n".join(parts)

    async def _launch_session(self):
        self._session_task = asyncio.create_task(self._run_session())

    async def _run_session(self):
        client = genai.Client(api_key=self._api_key)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=self._system_instruction)]
            ),
        )
        try:
            async with client.aio.live.connect(model=_MODEL, config=config) as session:
                self._session = session
                self._session_ready.set()
                print("[GeminiLiveVLM] Session open.")
                await asyncio.gather(
                    self._video_sender(session),
                    self._response_receiver(session),
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[GeminiLiveVLM] Session error: {e}")
            self._session = None
            self._session_ready.clear()
            print("[GeminiLiveVLM] Session closed. Reconnecting in 3s...")
            await asyncio.sleep(3)
            self._session_task = asyncio.create_task(self._run_session())
            return
        finally:
            self._session = None
            self._session_ready.clear()
            print("[GeminiLiveVLM] Session closed.")

    async def _video_sender(self, session):
        while True:
            # Drain queue and keep only the latest frame (API caps at 1 FPS)
            latest = None
            try:
                while True:
                    latest = self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            if latest is not None:
                buf = io.BytesIO()
                latest.save(buf, format="JPEG")
                await session.send_realtime_input(
                    video=types.Blob(data=buf.getvalue(), mime_type="image/jpeg")
                )

            await asyncio.sleep(1.0)

    async def _response_receiver(self, session):
        current_turn_parts = []
        async for response in session.receive():
            if not response.server_content:
                continue

            # In AUDIO modality, text arrives only via output_transcription
            trans = getattr(response.server_content, "output_transcription", None)
            if trans and getattr(trans, "text", None):
                current_turn_parts.append(trans.text)

            if response.server_content.turn_complete:
                complete_text = "".join(current_turn_parts).strip()
                current_turn_parts = []

                with self._pending_query_lock:
                    if self._pending_query and not self._pending_query[1].is_set():
                        query_text, event, holder, is_cancelled = self._pending_query
                        if not is_cancelled[0]:
                            holder.append(complete_text)
                            event.set()
                        self._pending_query = None
                        continue

                if not complete_text:
                    continue
                print(f"[GeminiLiveVLM] Proactive: {complete_text}")

    async def _reconnect(self):
        if self._session_task and not self._session_task.done():
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass
        self._session_task = asyncio.create_task(self._run_session())

    def reset(self):
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

    def push_frame(self, pil_image):
        self._frame_queue.put(pil_image)

    def chat(self, query: str) -> str:
        if not self._session_ready.wait(timeout=15):
            return "[GeminiLiveVLM] Session not ready"

        event = threading.Event()
        holder = []
        is_cancelled = [False]  # Use a list for mutable access in the tuple
        with self._pending_query_lock:
            self._pending_query = (query, event, holder, is_cancelled)

        try:
            asyncio.run_coroutine_threadsafe(
                self._session.send_realtime_input(text=query),
                self._loop,
            ).result(timeout=5)
        except Exception as e:
            with self._pending_query_lock:
                self._pending_query = None
            return f"[GeminiLiveVLM] Send error: {e}"

        if not event.wait(timeout=30):
            # Timed out. Mark the query as cancelled.
            # The receiver will see this and discard the late response if it arrives.
            with self._pending_query_lock:
                if self._pending_query and self._pending_query[0] == query:
                    is_cancelled[0] = True
            return "[GeminiLiveVLM] Response timeout"

        # Success! The event was set by the receiver.
        return holder[0] if holder else ""
