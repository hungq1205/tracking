import grpc
import numpy as np
import cv2
import torch
from proto import tracking_pb2
from proto import tracking_pb2_grpc
from core.interfaces import Detection, HandTrack

class RemoteTrackingClient:
    def __init__(self, server_address='localhost:50051'):
        self.channel = grpc.insecure_channel(server_address)
        self.track_stub = tracking_pb2_grpc.TrackingServiceStub(self.channel)
        self.last_scale = 1.0 # Current scaling factor from the most recent send_gui_frame

    def _encode_image_with_scale(self, frame):
        # Optimize: Resize before encoding if frame is too large for the Pi Zero 2's upload bandwidth
        h, w = frame.shape[:2]
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return img_encoded.tobytes(), scale

    def detect(self, frame, prompt, *args, **kwargs):
        # Server uses the previously streamed frame; frame argument is kept for API compatibility
        request = tracking_pb2.DetectionRequest(prompt=prompt)
        try:
            res = self.track_stub.DetectObject(request)
            # Scale the bounding box back to original image dimensions
            scaled_box = tuple(coord / self.last_scale for coord in res.box_xyxy)
            return Detection(box_xyxy=scaled_box, score=res.score)
        except grpc.RpcError as e:
            print(f"[EDGE] Detection RPC failed: {e}")
            return Detection(box_xyxy=(0,0,0,0), score=0.0)

    def get_embedding(self, frame, box):
        # Scale the request box down to match the resized image
        scaled_box = [coord * self.last_scale for coord in box]
        request = tracking_pb2.EmbeddingRequest(
            box_xyxy=scaled_box
        )
        try:
            res = self.track_stub.GetEmbedding(request)
            if not res.embedding:
                return None
            return torch.tensor(res.embedding).unsqueeze(0)
        except grpc.RpcError as e:
            print(f"[EDGE] Embedding RPC failed: {e}")
            return None

    def chat(self, frame, message):
        request = tracking_pb2.ChatRequest(message=message)
        try:
            res = self.track_stub.Chat(request)
            return res
        except grpc.RpcError as e:
            return tracking_pb2.ChatResponse(response=f"Chat RPC failed: {e}")

    def voice_chat(self, audio_data):
        request = tracking_pb2.VoiceChatRequest(audio_data=audio_data)
        try:
            res = self.track_stub.VoiceChat(request)
            return res
        except grpc.RpcError as e:
            return tracking_pb2.ChatResponse(response=f"Voice Chat RPC failed: {e}")

    def send_gui_frame(self, frame):
        """Send rendered frame to server via the dedicated GUI stream RPC."""
        image_data, scale = self._encode_image_with_scale(frame)
        self.last_scale = scale # Update scale for subsequent stateful RPCs
        request = tracking_pb2.FrameRequest(image_data=image_data)
        try:
            res = self.track_stub.StreamFrame(request)
        except grpc.RpcError as e:
            print(f"[EDGE] Gui frame RPC failed: {e}")
            pass

# Wrapper classes to maintain compatibility with existing interfaces
class RemoteGroundingDINO(RemoteTrackingClient): pass
class RemoteEmbedder(RemoteTrackingClient): pass