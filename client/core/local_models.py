import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from .interfaces import IHandDetector, HandTrack

class LocalHandDetector(IHandDetector):
    """
    Optimized Hand Detector using MediaPipe Tasks API.
    """
    def __init__(self, model_path: str = 'hand_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> HandTrack:
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.detector.detect(mp_image)

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            
            # Extract bounding box from landmarks
            x_coords = [lm.x for lm in hand_lms]
            y_coords = [lm.y for lm in hand_lms]
            
            x1, y1 = min(x_coords) * w, min(y_coords) * h
            x2, y2 = max(x_coords) * w, max(y_coords) * h
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            return HandTrack(
                box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                center_xy=(float(cx), float(cy)),
                confidence=0.8, # MediaPipe doesn't provide a per-frame score easily
                visible=True,
                landmarks=[(lm.x * w, lm.y * h) for lm in hand_lms]
            )

        return HandTrack((0,0,0,0), (w/2, h/2), 0.0, False)