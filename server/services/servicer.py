import queue as _queue
import traceback

import cv2
import grpc
import numpy as np

import tracking_pb2
import tracking_pb2_grpc
from live_session import LiveAPISession, ToolsBundle


class TrackingServiceServicer(tracking_pb2_grpc.TrackingServiceServicer):
    def __init__(
        self,
        tools_bundle: ToolsBundle,
        detector,
        embedder,
        frame_queue,
    ):
        self.tools_bundle = tools_bundle
        self.detector = detector
        self.embedder = embedder
        self.frame_queue = frame_queue
        self.latest_frame = None
        self.current_session: "LiveAPISession | None" = None

    # ── Helpers ──────────────────────────────────────────────────────────────

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

    # ── Simple RPCs (unchanged behaviour) ────────────────────────────────────

    def DetectObject(self, request, context):
        print(f"[SERVER] DetectObject: prompt='{request.prompt}'", flush=True)
        if self.latest_frame is None:
            return tracking_pb2.DetectionResponse()
        frame = self.latest_frame.copy()
        det = self.detector.detect(frame, request.prompt)
        if det.score > 0:
            x1, y1, x2, y2 = map(int, det.box_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"[SERVER] DetectObject → score={det.score:.2f}", flush=True)
        return tracking_pb2.DetectionResponse(box_xyxy=list(det.box_xyxy), score=det.score)

    def GetEmbedding(self, request, context):
        print(f"[SERVER] GetEmbedding: box={request.box_xyxy}", flush=True)
        if self.latest_frame is None:
            return tracking_pb2.EmbeddingResponse(embedding=[])
        emb = self.embedder.get_embedding(self.latest_frame, tuple(request.box_xyxy))
        if emb is not None:
            return tracking_pb2.EmbeddingResponse(
                embedding=emb.detach().cpu().numpy().flatten().tolist()
            )
        return tracking_pb2.EmbeddingResponse(embedding=[])

    def Chat(self, request, context):
        """Simple text stub — uses one-shot Gemini generate (not Live)."""
        print(f"[SERVER] Chat: '{request.message}'", flush=True)
        try:
            from google import genai as _genai
            client = _genai.Client(api_key=self.tools_bundle.gemini_api_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=request.message or "",
            )
            reply = resp.text or ""
            return tracking_pb2.ChatResponse(response=reply)
        except Exception as e:
            return tracking_pb2.ChatResponse(response=f"Error: {e}")

    def VoiceChat(self, request, context):
        """Simple voice stub — transcribe then Chat."""
        print(f"[SERVER] VoiceChat: audio_len={len(request.audio_data)}", flush=True)
        return tracking_pb2.ChatResponse(response="[VoiceChat] Use VoiceChatStream for full voice interaction.")

    def StreamFrame(self, request, context):
        """Store latest frame for DetectObject / GetEmbedding. Ticks moved to LiveAPISession."""
        try:
            frame = self._decode_image(request.image_data)
            if frame is not None:
                self.latest_frame = frame
                self._push_frame(frame)
            return tracking_pb2.FrameResponse(success=True)
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tracking_pb2.FrameResponse(success=False)

    # ── VoiceChatStream — primary real-time path ──────────────────────────────

    def VoiceChatStream(self, request_iterator, context):
        """
        Bidirectional streaming RPC.

        Receives VoiceChatChunk stream (audio PCM + video JPEG + IMU) from Android.
        Creates a LiveAPISession, forwards audio/video to Gemini Live, and yields
        24 kHz PCM AudioChunk responses back to the client.
        """
        session = LiveAPISession(tools=self.tools_bundle)
        try:
            session.start_sync()
        except Exception as e:
            print(f"[VoiceChatStream] Failed to connect to Gemini Live: {e}", flush=True)
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details(f"Gemini Live unavailable: {e}")
            return

        self.current_session = session

        # Background thread reads from Android and feeds the session
        def _feed_session():
            try:
                for chunk in request_iterator:
                    which = chunk.WhichOneof("payload")
                    if which == "audio_chunk":
                        pcm = bytes(chunk.audio_chunk)
                        if pcm:
                            session.send_audio_sync(pcm)
                        else:
                            # Empty chunk = PTT released → flush Gemini's audio buffer
                            session.send_audio_end_sync()
                    elif which == "video_frame":
                        jpeg = bytes(chunk.video_frame)
                        self.latest_frame = self._decode_image(jpeg)  # keep for DetectObject
                        session.receive_frame_sync(jpeg)
                    # IMU frames ignored in Live path (VIO not used in Live session)
            except Exception as exc:
                print(f"[VoiceChatStream] Feed error: {exc}", flush=True)
            finally:
                session.close_sync()

        import threading
        feed_thread = threading.Thread(target=_feed_session, daemon=True, name="gRPC-feed")
        feed_thread.start()

        # Yield PCM audio chunks and state-update notifications to the client
        try:
            while not session.is_done():
                try:
                    pcm = session.read_pcm_sync(timeout=0.05)
                    if pcm is None:
                        break
                    if pcm:
                        yield tracking_pb2.AudioChunk(pcm_data=pcm)
                except _queue.Empty:
                    pass
                # Forward any pending state updates as metadata-only chunks
                try:
                    while True:
                        update = session.state_update_q.get_nowait()
                        yield tracking_pb2.AudioChunk(
                            agent_state=update.get("agent_state", ""),
                            agent_payload=update.get("agent_payload", ""),
                        )
                except _queue.Empty:
                    pass
        except Exception as e:
            print(f"[VoiceChatStream] Yield error: {e}", flush=True)
        finally:
            feed_thread.join(timeout=3)
            if self.current_session is session:
                self.current_session = None
            print("[VoiceChatStream] Session ended.", flush=True)
