import io
import queue
import tempfile
import threading
import time
import traceback

import grpc
import cv2
import numpy as np
import soundfile as sf
from PIL import Image

import tracking_pb2
import tracking_pb2_grpc
from services.proto_converters import agent_result_to_chat_response


class TrackingServiceServicer(tracking_pb2_grpc.TrackingServiceServicer):
    def __init__(
        self,
        orchestrator,
        detector,
        embedder,
        asr,
        tts,
        streaming_vlm_instance,
        frame_queue,
        conversation_queue=None,
    ):
        self.orchestrator = orchestrator
        self.detector = detector
        self.embedder = embedder
        self.asr = asr
        self.tts = tts
        self.streaming_vlm_instance = streaming_vlm_instance
        self.frame_queue = frame_queue
        self.conversation_queue = conversation_queue
        self.latest_frame = None
        self.last_frame_tick_time = time.time()
        self.vlm_lock = threading.Lock()

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
            except queue.Full:
                pass

    def _generate_tts(self, text):
        return self.tts.synthesize(text)

    def DetectObject(self, request, context):
        print(f"[SERVER] Received DetectObject request: prompt='{request.prompt}'", flush=True)
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
        print(f"[SERVER] Sending DetectObject response: score={det.score:.2f}", flush=True)
        return tracking_pb2.DetectionResponse(box_xyxy=list(det.box_xyxy), score=det.score)

    def GetEmbedding(self, request, context):
        print(f"[SERVER] Received GetEmbedding request: box={request.box_xyxy}", flush=True)
        if self.latest_frame is None:
            return tracking_pb2.EmbeddingResponse(embedding=[])
        emb = self.embedder.get_embedding(self.latest_frame, tuple(request.box_xyxy))
        if emb is not None:
            embedding_list = emb.detach().cpu().numpy().flatten().tolist()
            print(f"[SERVER] Sending GetEmbedding response: embedding_len={len(embedding_list)}", flush=True)
            return tracking_pb2.EmbeddingResponse(embedding=embedding_list)
        print("[SERVER] Sending GetEmbedding response: no embedding found", flush=True)
        return tracking_pb2.EmbeddingResponse(embedding=[])

    def _handle_chat_message(
        self, user_text: str, prefix: str = "", asr_ms: float = 0.0
    ) -> tracking_pb2.ChatResponse:
        if self.streaming_vlm_instance is None:
            return tracking_pb2.ChatResponse(response="Error: StreamingVLM not initialized.")
        if self.latest_frame is None and not user_text:
            return tracking_pb2.ChatResponse(response="Error: No video frames received by server yet.")

        try:
            timings: dict = {}
            if asr_ms:
                timings["asr_ms"] = asr_ms

            t0 = time.time()
            with self.vlm_lock:
                result = self.orchestrator.orchestrate(user_text, self.latest_frame, timings=timings)
            timings["orchestrate_ms"] = (time.time() - t0) * 1000

            t0 = time.time()
            audio_bytes = self._generate_tts(result.reply_text) if result.speak and result.reply_text else b""
            if result.speak and result.reply_text:
                timings["tts_ms"] = (time.time() - t0) * 1000

            reply = f"{prefix}{result.reply_text}" if prefix else result.reply_text
            result = type(result)(
                agent_name=result.agent_name,
                state=result.state,
                payload=result.payload,
                reply_text=reply,
                speak=result.speak,
            )

            parts = []
            if "asr_ms" in timings:
                parts.append(f"asr={timings['asr_ms']:.0f}ms")
            if "intent_parse_ms" in timings:
                parts.append(f"intent={timings['intent_parse_ms']:.0f}ms")
            if "rag_ms" in timings:
                parts.append(f"rag={timings['rag_ms']:.0f}ms")
            if "agent_ms" in timings:
                agent_label = timings.get("agent_name", "agent")
                parts.append(f"{agent_label}={timings['agent_ms']:.0f}ms")
            if "tts_ms" in timings:
                parts.append(f"tts={timings['tts_ms']:.0f}ms")
            parts.append(f"total={timings['orchestrate_ms'] + timings.get('asr_ms', 0) + timings.get('tts_ms', 0):.0f}ms")
            timing_str = "  ".join(parts)
            print(f"[SERVER] Chat result: agent={result.agent_name} state={result.state}  [{timing_str}]", flush=True)
            if self.conversation_queue is not None:
                try:
                    self.conversation_queue.put_nowait({"user": user_text, "assistant": result.reply_text})
                except queue.Full:
                    pass
            return agent_result_to_chat_response(result, audio_bytes)
        except Exception as e:
            traceback.print_exc()
            return tracking_pb2.ChatResponse(response=f"Error: {str(e)}")

    def Chat(self, request, context):
        print(f"[SERVER] Received Chat request: message='{request.message}'", flush=True)
        return self._handle_chat_message(request.message or "")

    def _asr_mode(self) -> str:
        ctx = self.orchestrator.context
        if ctx.reading_state != "idle":
            return "reading"
        if ctx.active_agent == "tracking":
            return "tracking"
        return "general"

    def VoiceChat(self, request, context):
        print(f"[SERVER] Received VoiceChat request: audio_len={len(request.audio_data)}", flush=True)
        if not request.audio_data:
            return tracking_pb2.ChatResponse(response="Error: Setup issues.")
        t0 = time.time()
        user_text = self.asr.transcribe(request.audio_data, mode=self._asr_mode())
        asr_ms = (time.time() - t0) * 1000
        if not user_text:
            return tracking_pb2.ChatResponse(response="[ASR] Could not understand audio.")
        return self._handle_chat_message(user_text, prefix=f"[Voice: {user_text}]\n", asr_ms=asr_ms)

    def StreamFrame(self, request, context):
        try:
            frame = self._decode_image(request.image_data)
            if frame is not None:
                self.latest_frame = frame
                if self.streaming_vlm_instance:
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.streaming_vlm_instance.push_frame(pil_image)

                now = time.time()
                if now - self.last_frame_tick_time >= 1.0:
                    reading_result = self.orchestrator.on_frame_tick(frame)
                    self.last_frame_tick_time = now
                    if reading_result and reading_result.reply_text and reading_result.speak:
                        print(
                            f"[SERVER] Reading (frame tick): {reading_result.reply_text[:120]}",
                            flush=True,
                        )

                self._push_frame(frame)
            return tracking_pb2.FrameResponse(success=True)
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tracking_pb2.FrameResponse(success=False)
