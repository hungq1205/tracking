import math
import numpy as np
from .interfaces import IObjectTracker, IHandDetector, GuidanceState, ObjectTrack, HandTrack

class GuidanceEngine:
    def __init__(self, object_tracker: IObjectTracker, hand_detector: IHandDetector):
        self.object_tracker = object_tracker
        self.hand_detector = hand_detector

    def update(self, frame: np.ndarray, fps: float = 0.0) -> GuidanceState:
        obj = self.object_tracker.update(frame)
        hand = self.hand_detector.detect(frame)
        return self._build_state(obj, hand, fps)

    def _build_state(self, obj: ObjectTrack, hand: HandTrack, fps: float = 0.0) -> GuidanceState:
        if not obj.visible or not hand.visible:
            return GuidanceState(
                object_track=obj,
                hand_track=hand,
                delta_x=0.0,
                delta_y=0.0,
                distance_px=0.0,
                instruction="TARGET_OR_HAND_LOST",
                fps=fps
            )

        dx = obj.center_xy[0] - hand.center_xy[0]
        dy = obj.center_xy[1] - hand.center_xy[1]
        dist = math.sqrt(dx*dx + dy*dy)

        return GuidanceState(
            object_track=obj,
            hand_track=hand,
            delta_x=dx,
            delta_y=dy,
            distance_px=dist,
            instruction=self._compute_instruction(dx, dy),
            fps=fps
        )

    def _compute_instruction(self, dx: float, dy: float) -> str:
        threshold = 40
        if abs(dx) < threshold and abs(dy) < threshold:
            return "ON_TARGET"

        instructions = []
        if dx > threshold: instructions.append("MOVE RIGHT")
        elif dx < -threshold: instructions.append("MOVE LEFT")

        if dy > threshold: instructions.append("MOVE DOWN")
        elif dy < -threshold: instructions.append("MOVE UP")

        return " + ".join(instructions)