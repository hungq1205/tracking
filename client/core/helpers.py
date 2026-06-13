import time
import cv2
import numpy as np
from typing import Tuple, Generator
from concurrent.futures import ThreadPoolExecutor

# Global executor for background renewal tasks
global_executor = ThreadPoolExecutor(max_workers=1)

def clamp_box_xyxy(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    return (max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3]))

def box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def to_gray(frame: np.ndarray) -> np.ndarray:
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

class CameraEmulator:
    """
    Simulates a live wearable camera feed. Emits frames strictly based on 
    their original video timestamps.
    """
    def __init__(self, video_path: str, target_fps: int = 30):
        self.video_path = video_path
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps

    def stream(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        while True:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.video_path}")

            start_time = time.perf_counter()
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                expected_elapsed = frame_idx * self.frame_delay
                actual_elapsed = time.perf_counter() - start_time

                if actual_elapsed > expected_elapsed + self.frame_delay:
                    frame_idx += 1
                    continue

                sleep_time = expected_elapsed - actual_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                yield frame, time.perf_counter() - start_time
                frame_idx += 1

            cap.release()