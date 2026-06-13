import grpc
import cv2
import numpy as np
import io
import tempfile
import soundfile as sf
import queue
import time
from PIL import Image
import tracking_pb2
import tracking_pb2_grpc

class TrackingServiceServicer(tracking_pb2_grpc.TrackingServiceServicer):
    def __init__(self, detector, embedder, asr_model, tts_pipeline, streaming_vlm_instance, frame_queue):
        self.detector = detector
        self.embedder = embedder
        self.asr_model = asr_model
        self.tts_pipeline = tts_pipeline
        self.streaming_vlm_instance = streaming_vlm_instance
        self.frame_queue = frame_queue
        self.latest_frame = None
        self.last_vlm_step_time = time.time()

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
        audio_out = []
        for _, _, audio in self.tts_pipeline(text, voice='af_heart', speed=1):
            audio_out.append(audio)
        if not audio_out: return b""
        full_audio = np.concatenate(audio_out)
        byte_io = io.BytesIO()
        sf.write(byte_io, full_audio, 24000, format='WAV')
        return byte_io.getvalue()

    def DetectObject(self, request, context):
        if self.latest_frame is None:
            return tracking_pb2.DetectionResponse()
        frame = self.latest_frame.copy()
        det = self.detector.detect(frame, request.prompt)
        if det.score > 0:
            x1, y1, x2, y2 = map(int, det.box_xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{request.prompt}: {det.score:.2f}", (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return tracking_pb2.DetectionResponse(box_xyxy=list(det.box_xyxy), score=det.score)

    def GetEmbedding(self, request, context):
        if self.latest_frame is None:
            return tracking_pb2.EmbeddingResponse(embedding=[])
        emb = self.embedder.get_embedding(self.latest_frame, tuple(request.box_xyxy))
        if emb is not None:
            embedding_list = emb.detach().cpu().numpy().flatten().tolist()
            return tracking_pb2.EmbeddingResponse(embedding=embedding_list)
        return tracking_pb2.EmbeddingResponse(embedding=[])

    def Chat(self, request, context):
        if self.streaming_vlm_instance is None:
            return tracking_pb2.ChatResponse(response="Error: StreamingVLM not initialized.")
        try:
            reply = self.streaming_vlm_instance.chat(request.message)
            # Reset the background timer since we just forced an inference step
            self.last_vlm_step_time = time.time()
            audio_bytes = self._generate_tts(reply)
            return tracking_pb2.ChatResponse(response=reply, audio_response=audio_bytes)
        except Exception as e:
            return tracking_pb2.ChatResponse(response=f"Error: {str(e)}")

    def VoiceChat(self, request, context):
        if self.streaming_vlm_instance is None or not request.audio_data:
            return tracking_pb2.ChatResponse(response="Error: Setup issues.")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(request.audio_data)
            tmp_path = tmp.name
        try:
            result = self.asr_model.transcribe(tmp_path)
            user_text = result["text"].strip()
            if not user_text:
                return tracking_pb2.ChatResponse(response="[ASR] Could not understand audio.")
            reply = self.streaming_vlm_instance.chat(user_text)
            self.last_vlm_step_time = time.time()
            audio_bytes = self._generate_tts(reply)
            return tracking_pb2.ChatResponse(response=f"[Voice: {user_text}]\n{reply}", audio_response=audio_bytes)
        finally:
            import os
            if os.path.exists(tmp_path): os.remove(tmp_path)

    def StreamFrame(self, request, context):
        frame = self._decode_image(request.image_data)
        if frame is not None:
            self.latest_frame = frame
            if self.streaming_vlm_instance:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.streaming_vlm_instance.push_frame(pil_image)

                # Mimic inference.py: Once 1 second of frames is "pulled" (buffered),
                # run the model ONCE per chunk to generate memory.
                now = time.time()
                if now - self.last_vlm_step_time >= 1.0:
                    # Replicate inference.py logging style and handle chunking
                    start_time = float(self.streaming_vlm_instance.chunk_index)
                    end_time = start_time + 1.0
                    
                    reply = self.streaming_vlm_instance.process_video_step()
                    if reply:
                        # Clean the suffix for logging, same as inference.py
                        clean_reply = reply[:-4] if reply.endswith(" ...") else reply
                        hms_start = time.strftime('%H:%M:%S', time.gmtime(int(start_time)))
                        hms_end = time.strftime('%H:%M:%S', time.gmtime(int(end_time)))
                        print(f"Time={hms_start}-{hms_end}: \033[1m\033[34m{clean_reply}\033[0m")
                    
                    self.last_vlm_step_time = now

            self._push_frame(frame)
        return tracking_pb2.FrameResponse(success=True)