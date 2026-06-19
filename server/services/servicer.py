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
    ):
        self.orchestrator = orchestrator
        self.detector = detector
        self.embedder = embedder
        self.asr = asr
        self.tts = tts
        self.streaming_vlm_instance = streaming_vlm_instance
        self.frame_queue = frame_queue
        self.latest_frame = None
        self.last_vlm_step_time = time.time()
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

    def _handle_chat_message(self, user_text: str, prefix: str = "") -> tracking_pb2.ChatResponse:
        if self.streaming_vlm_instance is None:
            return tracking_pb2.ChatResponse(response="Error: StreamingVLM not initialized.")
        if self.latest_frame is None and not user_text:
            return tracking_pb2.ChatResponse(response="Error: No video frames received by server yet.")

        try:
            with self.vlm_lock:
                result = self.orchestrator.orchestrate(user_text, self.latest_frame)
                if result.agent_name == "info":
                    self.last_vlm_step_time = time.time()

            audio_bytes = self._generate_tts(result.reply_text) if result.speak and result.reply_text else b""
            reply = f"{prefix}{result.reply_text}" if prefix else result.reply_text
            result = type(result)(
                agent_name=result.agent_name,
                state=result.state,
                payload=result.payload,
                reply_text=reply,
                speak=result.speak,
            )
            print(f"[SERVER] Chat result: agent={result.agent_name} state={result.state}", flush=True)
            return agent_result_to_chat_response(result, audio_bytes)
        except Exception as e:
            traceback.print_exc()
            return tracking_pb2.ChatResponse(response=f"Error: {str(e)}")

    def Chat(self, request, context):
        print(f"[SERVER] Received Chat request: message='{request.message}'", flush=True)
        return self._handle_chat_message(request.message or "")

    def VoiceChat(self, request, context):
        print(f"[SERVER] Received VoiceChat request: audio_len={len(request.audio_data)}", flush=True)
        if not request.audio_data:
            return tracking_pb2.ChatResponse(response="Error: Setup issues.")
        user_text = self.asr.transcribe(request.audio_data)
        if not user_text:
            return tracking_pb2.ChatResponse(response="[ASR] Could not understand audio.")
        response = self._handle_chat_message(user_text, prefix=f"[Voice: {user_text}]\n")
        return response

    def StreamFrame(self, request, context):
        try:
            frame = self._decode_image(request.image_data)
            if frame is not None:
                self.latest_frame = frame
                if self.streaming_vlm_instance:
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.streaming_vlm_instance.push_frame(pil_image)

                    now = time.time()
                    if now - self.last_vlm_step_time >= 1.0:
                        start_time = float(self.streaming_vlm_instance.chunk_index)
                        end_time = start_time + 1.0
                        with self.vlm_lock:
                            reply = self.streaming_vlm_instance.process_video_step()
                        reading_result = self.orchestrator.on_frame_tick(frame)
                        if reply:
                            clean_reply = reply[:-4] if reply.endswith(" ...") else reply
                            hms_start = time.strftime("%H:%M:%S", time.gmtime(int(start_time)))
                            hms_end = time.strftime("%H:%M:%S", time.gmtime(int(end_time)))
                            print(
                                f"Time={hms_start}-{hms_end}: \033[1m\033[34m{clean_reply}\033[0m",
                                flush=True,
                            )
                        if reading_result and reading_result.reply_text and reading_result.speak:
                            print(
                                f"[SERVER] Reading (frame tick): {reading_result.reply_text[:120]}",
                                flush=True,
                            )
                        self.last_vlm_step_time = now

                self._push_frame(frame)
            return tracking_pb2.FrameResponse(success=True)
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tracking_pb2.FrameResponse(success=False)
