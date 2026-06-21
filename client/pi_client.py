import time
from typing import Optional

import cv2
import grpc
import numpy as np

from proto import tracking_pb2
from proto import tracking_pb2_grpc


class PiMediatorClient:
    """
    Thin Raspberry Pi client. Captures webcam frames, sends them to the Mediator
    via gRPC, and receives guidance instructions back. Runs no local algorithms.
    """

    def __init__(
        self,
        mediator_ip: str,
        camera_index: int = 0,
        jpeg_quality: int = 60,
    ):
        self.mediator_addr = f"{mediator_ip}:50052"
        self.camera_index = camera_index
        self.jpeg_quality = jpeg_quality
        self.channel = grpc.insecure_channel(self.mediator_addr)
        self.stub = tracking_pb2_grpc.MediatorServiceStub(self.channel)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        _, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        return encoded.tobytes()

    def _on_guidance(self, response) -> None:
        """
        Called after each frame. On a headless Pi, drive a buzzer/LED here.
        Override or subclass for custom feedback hardware.
        """
        print(
            f"[PI] status={response.tracking_status} "
            f"instruction={response.instruction} "
            f"conf={response.object_confidence:.2f}"
        )

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"[PI] Cannot open camera {self.camera_index}")

        print(f"[PI] Connecting to Mediator at {self.mediator_addr} (camera-native rate)")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[PI] Camera read failed, retrying...")
                time.sleep(0.5)
                continue

            request = tracking_pb2.FrameRequest(image_data=self._encode_frame(frame))
            try:
                response = self.stub.StreamFrameWithGuidance(request)
                self._on_guidance(response)
            except grpc.RpcError as e:
                print(f"[PI] gRPC error: {e.code()} — {e.details()}")

        cap.release()

    def send_text_chat(self, message: str) -> Optional[object]:
        """Send a text command to the main server via the mediator proxy."""
        request = tracking_pb2.ChatRequest(message=message)
        try:
            return self.stub.Chat(request)
        except grpc.RpcError as e:
            print(f"[PI] Chat RPC error: {e}")
            return None

    def send_voice_chat(self, audio_data: bytes) -> Optional[object]:
        """Send raw audio to the main server via the mediator proxy."""
        request = tracking_pb2.VoiceChatRequest(audio_data=audio_data)
        try:
            return self.stub.VoiceChat(request)
        except grpc.RpcError as e:
            print(f"[PI] VoiceChat RPC error: {e}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pi thin client — sends frames to Mediator")
    parser.add_argument("--mediator-ip", default="127.0.0.1", help="Mediator IP (default: 127.0.0.1)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--quality", type=int, default=60, help="JPEG quality 1-100 (default: 60)")
    args = parser.parse_args()

    client = PiMediatorClient(
        mediator_ip=args.mediator_ip,
        camera_index=args.camera,
        jpeg_quality=args.quality,
    )
    client.run()
