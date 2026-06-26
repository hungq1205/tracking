import io
import struct
import tempfile
import time
import traceback

import grpc
import cv2
import numpy as np
import soundfile as sf
from PIL import Image

import tracking_pb2
import tracking_pb2_grpc
from domain.intents import Intent
from services.proto_converters import agent_result_to_chat_response
from tools.cloud_vlm import GeminiVLMClient

try:
    from vio import IMUPreintegrator, VIOEstimator
    _VIO_AVAILABLE = True
except ImportError:
    _VIO_AVAILABLE = False


class TrackingServiceServicer(tracking_pb2_grpc.TrackingServiceServicer):
    def __init__(
        self,
        orchestrator,
        detector,
        embedder,
        asr,
        tts,
        frame_queue,
    ):
        self.orchestrator = orchestrator
        self.detector = detector
        self.embedder = embedder
        self.asr = asr
        self.tts = tts
        self.frame_queue = frame_queue
        self.latest_frame = None
        self.chat_log: list[tuple[str, str]] = []
        self.last_nav_result: dict | None = None

    def _log_chat(self, user_text: str, bot_text: str) -> None:
        self.chat_log.append((user_text, bot_text))
        if len(self.chat_log) > 200:
            self.chat_log = self.chat_log[-200:]

    def _decode_image(self, data):
        if not data:
            return None
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def _push_frame(self, frame):
        if frame is not None:
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

    def _generate_tts(self, text):
        return self.tts.synthesize(text)

    def DetectObject(self, request, context):
        print(f"[SERVER] DetectObject: prompt='{request.prompt}'", flush=True)
        if self.latest_frame is None:
            return tracking_pb2.DetectionResponse()
        frame = self.latest_frame.copy()
        det = self.detector.detect(frame, request.prompt)
        if det.score > 0:
            x1, y1, x2, y2 = map(int, det.box_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{request.prompt}: {det.score:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        print(f"[SERVER] DetectObject → score={det.score:.2f}", flush=True)
        return tracking_pb2.DetectionResponse(box_xyxy=list(det.box_xyxy), score=det.score)

    def GetEmbedding(self, request, context):
        print(f"[SERVER] GetEmbedding: box={request.box_xyxy}", flush=True)
        if self.latest_frame is None:
            return tracking_pb2.EmbeddingResponse(embedding=[])
        emb = self.embedder.get_embedding(self.latest_frame, tuple(request.box_xyxy))
        if emb is not None:
            embedding_list = emb.detach().cpu().numpy().flatten().tolist()
            return tracking_pb2.EmbeddingResponse(embedding=embedding_list)
        return tracking_pb2.EmbeddingResponse(embedding=[])

    def _handle_chat_message(self, user_text: str, prefix: str = "") -> tracking_pb2.ChatResponse:
        if self.latest_frame is None and not user_text:
            return tracking_pb2.ChatResponse(response="Error: No video frames received yet.")
        try:
            result = self.orchestrator.orchestrate(user_text, self.latest_frame)
            audio_bytes = self._generate_tts(result.reply_text) if result.speak and result.reply_text else b""
            if prefix:
                result = type(result)(
                    agent_name=result.agent_name,
                    state=result.state,
                    payload=result.payload,
                    reply_text=f"{prefix}{result.reply_text}",
                    speak=result.speak,
                )
            print(f"[SERVER] Chat → agent={result.agent_name} state={result.state}", flush=True)
            resp = agent_result_to_chat_response(result, audio_bytes)
            self._log_chat(user_text, result.reply_text)
            return resp
        except Exception as e:
            traceback.print_exc()
            return tracking_pb2.ChatResponse(response=f"Error: {str(e)}")

    def Chat(self, request, context):
        print(f"[SERVER] Chat: '{request.message}'", flush=True)
        return self._handle_chat_message(request.message or "")

    def _asr_mode(self) -> str:
        ctx = self.orchestrator.context
        if ctx.reading_state != "idle":
            return "reading"
        if ctx.active_agent == "tracking":
            return "tracking"
        return "general"

    def VoiceChat(self, request, context):
        print(f"[SERVER] VoiceChat: audio_len={len(request.audio_data)}", flush=True)
        if not request.audio_data:
            return tracking_pb2.ChatResponse(response="Error: No audio received.")
        user_text = self.asr.transcribe(request.audio_data, mode=self._asr_mode())
        if not user_text:
            return tracking_pb2.ChatResponse(response="[ASR] Could not understand audio.")
        return self._handle_chat_message(user_text, prefix=f"[Voice: {user_text}]\n")

    def _cloud_vlm(self):
        info = self.orchestrator.agents_by_name.get("info")
        return info.cloud_vlm if info else None

    def _build_wav_from_pcm(self, pcm: bytes, sample_rate: int = 16000, channels: int = 1, bits: int = 16) -> bytes:
        data_size = len(pcm)
        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE",
            b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits,
            b"data", data_size,
        )
        return header + pcm

    def _build_info_prompt(self, user_text: str) -> str:
        ctx = self.orchestrator.context
        prompt = user_text
        if ctx.reading_state != "idle" and ctx.scan_buffer:
            prompt = f"<<<READING_CONTEXT_START>>>\n{ctx.scan_buffer}\n<<<READING_CONTEXT_END>>>\n{prompt}"
        if self.orchestrator.rag_store:
            try:
                hits = self.orchestrator.rag_store.query_global(user_text, top_k=3)
                if hits:
                    rag_ctx = "\n".join(f"[{lbl}] {text}" for text, lbl, _ in hits)
                    prompt = f"Relevant saved memory:\n{rag_ctx}\n\nUser: {prompt}"
            except Exception as e:
                print(f"[VoiceChatStream] RAG lookup failed: {e}")
        return prompt

    def _ensure_vio(self, ctx) -> None:
        """Lazily initialize VIO components in the session context."""
        if not _VIO_AVAILABLE:
            return
        if ctx.vio_estimator is None:
            ctx.vio_estimator = VIOEstimator()
            ctx.imu_preintegrator = IMUPreintegrator()

    def VoiceChatStream(self, request_iterator, context):
        audio_buf = bytearray()
        latest_frame_bytes = None
        ctx = self.orchestrator.context

        for chunk in request_iterator:
            which = chunk.WhichOneof("payload")
            if which == "audio_chunk":
                audio_buf.extend(chunk.audio_chunk)
            elif which == "video_frame":
                latest_frame_bytes = chunk.video_frame
            elif which == "imu_frame":
                imu = chunk.imu_frame
                self._ensure_vio(ctx)
                if ctx.imu_preintegrator is not None:
                    ctx.imu_preintegrator.add_sample(
                        imu.timestamp_ns,
                        np.array([imu.accel_x, imu.accel_y, imu.accel_z], dtype=np.float64),
                        np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z], dtype=np.float64),
                    )

        if not audio_buf:
            return

        frame = self._decode_image(latest_frame_bytes) if latest_frame_bytes else self.latest_frame
        if frame is not None:
            self.latest_frame = frame

        # VIO pose update: flush pre-integrated IMU + identity visual to get current pose
        if (
            ctx.vio_estimator is not None
            and ctx.imu_preintegrator is not None
            and ctx.imu_preintegrator.is_initialized
        ):
            import numpy as _np
            prev_pose = ctx.current_pose if ctx.current_pose is not None else _np.eye(4)
            pim = ctx.imu_preintegrator.get_and_reset()
            ctx.current_pose = ctx.vio_estimator.add_keyframe(
                timestamp_ns=0,
                visual_rel_pose=_np.eye(4),
                pim=pim,
                prev_world_pose=prev_pose,
            )

        wav_bytes = self._build_wav_from_pcm(bytes(audio_buf))
        user_text = self.asr.transcribe(wav_bytes, mode=self._asr_mode())
        print(f"[VoiceChatStream] ASR: '{user_text}'", flush=True)
        if not user_text:
            return

        intent = self.orchestrator.parse_intent(user_text)
        vlm = self._cloud_vlm()

        if intent.intent == Intent.INFO and isinstance(vlm, GeminiVLMClient):
            prompt = self._build_info_prompt(user_text)
            print(f"[VoiceChatStream] INFO → Gemini stream", flush=True)
            try:
                for pcm_chunk in vlm.query_stream(prompt, frame):
                    if pcm_chunk:
                        yield tracking_pb2.AudioChunk(pcm_data=pcm_chunk)
                self._log_chat(user_text, "[streamed audio response]")
            except Exception as e:
                traceback.print_exc()
                print(f"[VoiceChatStream] Gemini error: {e}", flush=True)
        else:
            try:
                chat_resp = self._handle_chat_message(user_text, prefix=f"[Voice: {user_text}]\n")
                wav = chat_resp.audio_response
                pcm = wav[44:] if len(wav) > 44 else wav
                if pcm:
                    yield tracking_pb2.AudioChunk(pcm_data=pcm)
            except Exception as e:
                traceback.print_exc()
                print(f"[VoiceChatStream] fallback error: {e}", flush=True)

    def StreamFrame(self, request, context):
        try:
            frame = self._decode_image(request.image_data)
            if frame is not None:
                h, w = frame.shape[:2]
                self.latest_frame = frame
                self._push_frame(frame)

                # Reading frame tick (OCR accumulation)
                reading_result = self.orchestrator.on_frame_tick(frame)
                if reading_result and reading_result.reply_text and reading_result.speak:
                    audio = self._generate_tts(reading_result.reply_text)
                    return tracking_pb2.FrameResponse(success=True, audio_response=audio)

                # Navigation frame tick (obstacle detection + proximity)
                nav_result = self.orchestrator.on_nav_tick(frame)
                if nav_result and nav_result.state == "OBSTACLE_DETECTED":
                    self.last_nav_result = {
                        "description": nav_result.payload.get("description", ""),
                        "depth_m": nav_result.payload.get("depth_m", 0.0),
                        "at": time.time(),
                    }
                if nav_result and nav_result.reply_text and nav_result.speak:
                    print(f"[StreamFrame] Nav: {nav_result.state} — {nav_result.reply_text[:80]}", flush=True)
                    audio = self._generate_tts(nav_result.reply_text)
                    return tracking_pb2.FrameResponse(success=True, audio_response=audio)

            return tracking_pb2.FrameResponse(success=True)
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tracking_pb2.FrameResponse(success=False)
