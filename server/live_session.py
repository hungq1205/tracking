"""
LiveAPISession — Gemini Live WebSocket session per user.

Replaces: Orchestrator + all intent parsers + CloudVLMClient + Whisper ASR + Kokoro TTS.

Thread model:
  Sync gRPC thread  ──queues──►  Async Gemini Live loop (background daemon thread)
  Output PCM chunks land in self._output_q (thread-safe Queue) for gRPC to yield.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

LIVE_MODEL = "gemini-3.1-flash-live-preview"
_SENTINEL = object()  # signals end-of-stream in output queue

# Intervals (seconds) for background frame processing
_OCR_INTERVAL = 1.5
_DEPTH_INTERVAL = 0.5
_OBSTACLE_COOLDOWN = 15.0
_LOCALIZE_INTERVAL = 2.0
_WALKING_DETECT_COOLDOWN = 10.0  # seconds between obstacle alerts in walking mode


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkingConfig:
    detection_interval: float = 2.0   # seconds between DINO runs (0.5 fps default)
    inner_rect_fraction: float = 0.5  # fraction of frame WIDTH for center detection zone
    depth_threshold_m: float = 3.0    # alert if obstacle closer than this (metres)


@dataclass
class LiveSessionState:
    mode: str = "idle"  # idle | reading | tracking | guiding

    # Vision stream
    live_vision_active: bool = False

    # Reading
    reading_buffer: str = ""
    page_summaries: List[str] = field(default_factory=list)
    reading_direction: str = "ltr"
    last_ocr_at: float = 0.0
    reading_label: str = ""

    # Tracking
    tracking_target: str = ""
    last_detection: Optional[Dict] = None

    # Guiding — shared by free-walk (no route) and routed guiding (with route)
    nav_destination: str = ""
    nav_location_id: str = ""
    nav_route: List[str] = field(default_factory=list)
    nav_route_idx: int = 0
    nav_last_position: Optional[List[float]] = None
    nav_last_localize_at: float = 0.0

    # Obstacle detection (used in guiding mode regardless of whether route is set)
    walking_obstacle_cache: List[Dict] = field(default_factory=list)  # [{label, expires_at}]
    walking_last_detect_at: float = 0.0
    walking_last_obstacle_at: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ToolsBundle
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolsBundle:
    detector: Any
    depth_detector: Any
    ocr: Any
    rag_store: Any
    memory_store: Any
    maps_root_dir: str
    gemini_api_key: str
    embedder: Any = None          # EfficientNetLiteEmbedder
    object_store: Any = None      # ObjectStore
    da3_onnx: Any = None          # DA3OnnxEstimator; None = walking mode disabled
    walking_config: Any = None    # WalkingConfig instance shared across sessions


# ──────────────────────────────────────────────────────────────────────────────
# LiveAPISession
# ──────────────────────────────────────────────────────────────────────────────

class LiveAPISession:
    """
    One Gemini Live WebSocket session per connected Android client.

    Sync callers (gRPC thread):
      session.start_sync()                  — connect; blocks until ready
      session.send_audio_sync(pcm: bytes)   — forward mic PCM to Gemini
      session.receive_frame_sync(jpeg: bytes) — store frame + trigger ticks
      session.read_pcm_sync(timeout) -> bytes | None  — blocking read of output
      session.close_sync()                  — graceful teardown
    """

    def __init__(self, tools: ToolsBundle):
        self.tools = tools
        self.state = LiveSessionState()

        # Latest JPEG frame from Android (updated from gRPC thread, read in async loop)
        self.latest_frame: Optional[bytes] = None

        # Thread-safe output queue: PCM bytes or _SENTINEL
        self._output_q: queue.Queue = queue.Queue()

        # Async internals (set up in _run)
        self._input_q: Optional[asyncio.Queue] = None
        self._session: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._done_event = threading.Event()

        # Navigation helpers (lazy loaded when start_guiding is called)
        self._route_planner = None
        self._localizer = None
        self._current_location_id: Optional[str] = None

        # GUI display frames — updated after each guiding tick
        self.last_guide_dino_frame: Optional[Any] = None   # BGR ndarray with DINO boxes
        self.last_guide_depth_image: Optional[Any] = None  # RGB ndarray, colorized DA3 depth

        # GUI display logs (bounded lists)
        self.conversation_log: List[Dict[str, Any]] = []   # {role, text, at}
        self.context_injections: List[Dict[str, Any]] = [] # {text, at}
        self._MAX_LOG = 100

        # State updates pushed by tools → consumed by servicer to send AudioChunk metadata
        self.state_update_q: queue.Queue = queue.Queue(maxsize=20)

    # ── Public sync API ───────────────────────────────────────────────────────

    def start_sync(self) -> None:
        """Connect to Gemini Live API. Blocks until the session is established."""
        self._loop = asyncio.new_event_loop()
        ready_event = threading.Event()
        error_holder: list = []

        def _run_loop():
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run(ready_event))
            except Exception as exc:
                error_holder.append(exc)
                ready_event.set()
            finally:
                self._done_event.set()
                self._output_q.put(_SENTINEL)

        self._loop_thread = threading.Thread(target=_run_loop, daemon=True, name="gemini-live")
        self._loop_thread.start()
        ready_event.wait(timeout=15)
        if error_holder:
            raise error_holder[0]

    def send_audio_sync(self, pcm: bytes) -> None:
        """Queue 16 kHz PCM chunk for forwarding to Gemini."""
        if self._loop and self._input_q and not self._done_event.is_set():
            self._loop.call_soon_threadsafe(self._input_q.put_nowait, ("audio", pcm))

    def send_audio_end_sync(self) -> None:
        """Signal end of user's audio turn (PTT released) — flushes Gemini's audio buffer."""
        if self._loop and self._input_q and not self._done_event.is_set():
            self._loop.call_soon_threadsafe(self._input_q.put_nowait, ("audio_end", None))

    def receive_frame_sync(self, jpeg: bytes) -> None:
        """Store JPEG frame and schedule background processing ticks."""
        self.latest_frame = jpeg
        if self._loop and self._input_q and not self._done_event.is_set():
            self._loop.call_soon_threadsafe(self._input_q.put_nowait, ("frame", jpeg))

    def read_pcm_sync(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Blocking read of next 24 kHz PCM chunk from Gemini.
        Returns None when the session has ended (sentinel received).
        Raises queue.Empty on timeout.
        """
        item = self._output_q.get(timeout=timeout)
        if item is _SENTINEL:
            self._output_q.put(_SENTINEL)  # re-enqueue so future callers also see it
            return None
        return item

    def is_done(self) -> bool:
        return self._done_event.is_set()

    def close_sync(self) -> None:
        """Signal the async loop to close and wait for it."""
        if self._loop and self._input_q and not self._done_event.is_set():
            self._loop.call_soon_threadsafe(self._input_q.put_nowait, ("close", None))
        self._done_event.wait(timeout=8)

    # ── Async core ────────────────────────────────────────────────────────────

    async def _run(self, ready_event: threading.Event) -> None:
        from live_tools.tool_declarations import TOOL_DECLARATIONS, SYSTEM_PROMPT

        self._input_q = asyncio.Queue()
        client = genai.Client(api_key=self.tools.gemini_api_key)

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            tools=TOOL_DECLARATIONS,
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
                )
            ),
        )

        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            self._session = session
            ready_event.set()
            print("[LiveSession] Connected to Gemini Live API.", flush=True)

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._send_loop(), name="send")
                    tg.create_task(self._recv_loop(), name="recv")
            except* asyncio.CancelledError:
                pass
            except* Exception as eg:
                for exc in eg.exceptions:
                    print(f"[LiveSession] Error: {exc}", flush=True)

        print("[LiveSession] Gemini Live session closed.", flush=True)

    async def _send_loop(self) -> None:
        """Drain input queue and forward to Gemini."""
        while True:
            kind, data = await self._input_q.get()
            if kind == "close":
                try:
                    await self._session.send_realtime_input(audio_stream_end=True)
                except Exception:
                    pass
                break
            if kind == "audio":
                try:
                    await self._session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                    )
                except Exception as e:
                    print(f"[LiveSession] send_audio error (ignored): {e}", flush=True)
            elif kind == "audio_end":
                try:
                    await self._session.send_realtime_input(audio_stream_end=True)
                    print("[LiveSession] audio_stream_end sent", flush=True)
                except Exception as e:
                    print(f"[LiveSession] send_audio_end error (ignored): {e}", flush=True)
            elif kind == "frame":
                # Frames are stored as self.latest_frame; run background ticks here
                await self._on_frame_tick(data)

    async def _recv_loop(self) -> None:
        """Receive Gemini responses and dispatch tool calls."""
        while True:
            async for response in self._session.receive():
                content = response.server_content
                if content:
                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.inline_data:
                                self._output_q.put(part.inline_data.data)

                    if content.interrupted:
                        while not self._output_q.empty():
                            try:
                                self._output_q.get_nowait()
                            except queue.Empty:
                                break

                    if content.input_transcription and content.input_transcription.text:
                        t = content.input_transcription.text.strip()
                        print(f"[LiveSession] User: {t}", flush=True)
                        if t:
                            self.conversation_log.append({"role": "user", "text": t, "at": time.time()})
                            if len(self.conversation_log) > self._MAX_LOG:
                                self.conversation_log = self.conversation_log[-self._MAX_LOG:]

                    if content.output_transcription and content.output_transcription.text:
                        t = content.output_transcription.text.strip()
                        if t:
                            if self.conversation_log and self.conversation_log[-1]["role"] == "assistant":
                                self.conversation_log[-1]["text"] += " " + t
                            else:
                                self.conversation_log.append({"role": "assistant", "text": t, "at": time.time()})
                            if len(self.conversation_log) > self._MAX_LOG:
                                self.conversation_log = self.conversation_log[-self._MAX_LOG:]

                if response.tool_call:
                    responses = []
                    for fn_call in response.tool_call.function_calls:
                        result = await self._dispatch_tool(fn_call.name, dict(fn_call.args or {}))
                        result_str = str(result)[:100]
                        print(f"[LiveSession] Tool {fn_call.name} → {result_str}", flush=True)
                        args_str = ", ".join(f"{k}={v}" for k, v in (fn_call.args or {}).items())
                        self.context_injections.append({
                            "text": f"[TOOL] {fn_call.name}({args_str}) → {result_str}",
                            "at": time.time(),
                        })
                        if len(self.context_injections) > 50:
                            self.context_injections = self.context_injections[-50:]
                        responses.append(types.FunctionResponse(
                            id=fn_call.id,
                            name=fn_call.name,
                            response={"result": result},
                        ))
                    await self._session.send_tool_response(function_responses=responses)

    # ── Context injection helper ──────────────────────────────────────────────

    async def _inject_system(self, text: str) -> None:
        """Send a [SYSTEM] text to Gemini and record it in context_injections for the GUI."""
        entry = {"text": text, "at": time.time()}
        self.context_injections.append(entry)
        if len(self.context_injections) > 50:
            self.context_injections = self.context_injections[-50:]
        if self._session:
            await self._session.send_realtime_input(text=text)

    # ── Background frame ticks ────────────────────────────────────────────────

    async def _on_frame_tick(self, jpeg: bytes) -> None:
        now = time.time()

        # Reading: passive OCR accumulation
        if self.state.mode == "reading" and now - self.state.last_ocr_at > _OCR_INTERVAL:
            self.state.last_ocr_at = now
            asyncio.get_event_loop().create_task(self._ocr_tick(jpeg))

        # Guiding (free-walk or routed): DINO+DA3 obstacle detection every detection_interval
        if self.state.mode == "guiding":
            cfg = self.tools.walking_config
            if cfg and now - self.state.walking_last_detect_at > cfg.detection_interval:
                self.state.walking_last_detect_at = now
                asyncio.get_event_loop().create_task(self._walking_tick(jpeg, now))
            # Localization only runs when a route/destination is active
            if self.state.nav_route and now - self.state.nav_last_localize_at > _LOCALIZE_INTERVAL:
                self.state.nav_last_localize_at = now
                asyncio.get_event_loop().create_task(self._localize_tick(jpeg))

        # Vision stream
        if self.state.live_vision_active:
            # Vision stream loop handles its own timing; nothing to do here
            pass

    async def _ocr_tick(self, jpeg: bytes) -> None:
        try:
            import cv2, numpy as np
            from tools.memory_store import filter_new_sentences

            nparr = np.frombuffer(jpeg, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            text = await asyncio.to_thread(
                self.tools.ocr.read_text, frame, self.state.reading_direction
            )
            if not text:
                return
            new = filter_new_sentences(text, self.state.reading_buffer)
            if new:
                self.state.reading_buffer = f"{self.state.reading_buffer}\n{new}".strip()
        except Exception as e:
            print(f"[LiveSession] OCR tick error: {e}", flush=True)

    async def _localize_tick(self, jpeg: bytes) -> None:
        try:
            if not self._localizer or not self._localizer.available:
                return
            import cv2, numpy as np
            nparr = np.frombuffer(jpeg, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            loc = await asyncio.to_thread(self._localizer.localize, frame)
            if loc.position is not None:
                self.state.nav_last_position = loc.position.tolist()
                await self._check_proximity()
        except Exception as e:
            print(f"[LiveSession] Localize tick error: {e}", flush=True)

    async def _check_proximity(self) -> None:
        if not self.state.nav_last_position or not self._route_planner:
            return
        import numpy as np
        pos = np.array(self.state.nav_last_position, dtype=np.float32)

        dest_zone = self._route_planner.find_zone(self.state.nav_destination) if self.state.nav_destination else None
        if dest_zone and dest_zone.contains(pos):
            dest = self.state.nav_destination
            self.state.mode = "idle"
            self.state.nav_destination = ""
            await self._inject_system(f"[SYSTEM] User has arrived at {dest}. Announce arrival warmly.")
            return

        route = self.state.nav_route
        idx = self.state.nav_route_idx
        if idx < len(route) - 1:
            wp_zone = self._route_planner.find_zone(route[idx])
            if wp_zone and wp_zone.contains(pos):
                self.state.nav_route_idx += 1
                next_label = route[self.state.nav_route_idx]
                await self._inject_system(
                    f"[SYSTEM] Passed {route[idx]}, now heading to {next_label}. Announce this."
                )

    async def _walking_tick(self, jpeg: bytes, now: float) -> None:
        if self.tools.da3_onnx is None or self.tools.walking_config is None:
            return
        try:
            import cv2, numpy as np
            cfg = self.tools.walking_config

            # Expire stale obstacle labels (6 s TTL)
            self.state.walking_obstacle_cache = [
                e for e in self.state.walking_obstacle_cache if e["expires_at"] > now
            ]

            nparr = np.frombuffer(jpeg, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Build DINO prompt including unexpired known obstacle labels
            known = [e["label"] for e in self.state.walking_obstacle_cache]
            prompt = "a thing" + (". " + ". ".join(known) if known else "")

            # Run DA3 ONNX for depth map and DINO for detections concurrently
            depth_result, detections = await asyncio.gather(
                asyncio.to_thread(self.tools.da3_onnx.estimate, rgb),
                asyncio.to_thread(self.tools.detector.detect_all, frame_bgr, prompt),
            )
            depth_map = depth_result.depth_map  # HxW float32

            # Inner detection zone: centre strip of the frame
            margin = int(w * (1.0 - cfg.inner_rect_fraction) / 2)
            rx0, rx1 = margin, w - margin

            # ── Obstacle check ────────────────────────────────────────────────
            for det in detections:
                if "a thing" not in det.label.lower():
                    continue
                x1, y1, x2, y2 = det.box_xyxy
                # Only consider boxes centred inside the inner rect
                if (x1 + x2) / 2 < rx0 or (x1 + x2) / 2 > rx1:
                    continue
                # Average depth over bottom half of box (quarters 3+4 = 50-100% of height)
                bh = y2 - y1
                yi0 = max(0, int(y1 + bh * 0.5))
                yi1 = min(h - 1, int(y2))
                xi0, xi1 = max(0, int(x1)), min(w - 1, int(x2))
                roi = depth_map[yi0:yi1, xi0:xi1]
                if roi.size == 0:
                    continue
                avg_depth = float(np.mean(roi))
                if avg_depth < cfg.depth_threshold_m:
                    if now - self.state.walking_last_obstacle_at > _WALKING_DETECT_COOLDOWN:
                        self.state.walking_last_obstacle_at = now
                        ignore_note = (f" Ignore {', '.join(known)}." if known else "")
                        await self._inject_system(
                            f"[SYSTEM] Obstacle ~{avg_depth:.1f}m ahead on path. "
                            f"Briefly warn user and call quick_label_obstacle(label) with a short label.{ignore_note}"
                        )
                    break  # one alert per tick

            # ── Build GUI display frames ──────────────────────────────────────
            vis = frame_bgr.copy()
            cv2.rectangle(vis, (rx0, 0), (rx1, h - 1), (0, 200, 80), 2)
            for det in detections:
                x1, y1, x2, y2 = map(int, det.box_xyxy)
                in_zone = rx0 <= (x1 + x2) / 2 <= rx1
                is_thing = "a thing" in det.label.lower()
                color = (0, 200, 255) if (is_thing and in_zone) else (80, 80, 80)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis, f"{det.label} {det.score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )
            self.last_guide_dino_frame = vis  # BGR, converted to RGB in GUI

            d_min, d_max = float(depth_map.min()), float(depth_map.max())
            d_norm = ((depth_map - d_min) / (d_max - d_min + 1e-6) * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_PLASMA)
            self.last_guide_depth_image = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"[LiveSession] Walking tick error: {e}", flush=True)

    async def _vision_stream_loop(self) -> None:
        """Send frames at 1 fps for up to 15 s while live_vision_active is True."""
        deadline = time.time() + 15
        while self.state.live_vision_active and time.time() < deadline:
            if self.latest_frame and self._session:
                try:
                    await self._session.send_realtime_input(
                        video=types.Blob(data=self.latest_frame, mime_type="image/jpeg")
                    )
                except Exception:
                    break
            await asyncio.sleep(1.0)
        self.state.live_vision_active = False

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    async def _dispatch_tool(self, name: str, args: Dict[str, Any]) -> Any:
        from live_tools.scene_tools import HANDLERS as SCENE
        from live_tools.reading_tools import HANDLERS as READING
        from live_tools.tracking_tools import HANDLERS as TRACKING
        from live_tools.memory_tools import HANDLERS as MEMORY
        from live_tools.navigation_tools import HANDLERS as NAV
        from live_tools.walking_tools import HANDLERS as WALKING

        ALL = {**SCENE, **READING, **TRACKING, **MEMORY, **NAV, **WALKING}
        handler = ALL.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return await handler(self, **args)
        except Exception as exc:
            print(f"[LiveSession] Tool {name} raised: {exc}", flush=True)
            return {"error": str(exc)}
