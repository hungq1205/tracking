import cv2
import math
import numpy as np
from .interfaces import GuidanceState

class GUIRenderer:
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),      # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)            # Palm
    ]

    @staticmethod
    def draw_dashed_line(img, p1, p2, color=(255, 255, 255), thickness=1, dash_length=10):
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        if dist == 0: return
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start_pt = (int(p1[0] + (p2[0]-p1[0]) * i / dashes), int(p1[1] + (p2[1]-p1[1]) * i / dashes))
            end_pt = (int(p1[0] + (p2[0]-p1[0]) * (i + 0.5) / dashes), int(p1[1] + (p2[1]-p1[1]) * (i + 0.5) / dashes))
            cv2.line(img, start_pt, end_pt, color, thickness)

    @classmethod
    def render(cls, frame: np.ndarray, state: GuidanceState) -> np.ndarray:
        vis_frame = frame.copy()
        obj = state.object_track
        hand = state.hand_track

        # 1. Draw Anchor Points (Environment)
        if obj.debug and "anchor_pts" in obj.debug:
            for pt in obj.debug["anchor_pts"]:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 2, (255, 255, 0), -1)

        # 2. Draw Object
        if obj.visible or obj.status == "LOST":
            x1, y1, x2, y2 = map(int, obj.box_xyxy)
            color = (0, 255, 0) if obj.status == "TRACKING" else (0, 165, 255)
            if obj.status == "LOST": color = (0, 0, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(vis_frame, (int(obj.center_xy[0]), int(obj.center_xy[1])), 6, (0, 0, 255), -1)

        # 3. Draw Hand
        if hand.visible:
            # Draw skeletal connections
            if len(hand.landmarks) == 21:
                for connection in cls.HAND_CONNECTIONS:
                    pt1 = (int(hand.landmarks[connection[0]][0]), int(hand.landmarks[connection[0]][1]))
                    pt2 = (int(hand.landmarks[connection[1]][0]), int(hand.landmarks[connection[1]][1]))
                    cv2.line(vis_frame, pt1, pt2, (0, 255, 0), 1)
            
            # Display hand landmarks
            for pt in hand.landmarks:
                cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
            cv2.circle(vis_frame, (int(hand.center_xy[0]), int(hand.center_xy[1])), 6, (255, 0, 0), -1)

        # 4. Draw Guidance
        if obj.visible and hand.visible:
            cls.draw_dashed_line(vis_frame, hand.center_xy, obj.center_xy)

        # 5. Draw Instruction Overlay
        cv2.putText(vis_frame, state.instruction, (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # 6. Draw Metrics Overlay (FPS, Confidence, Active Anchors)
        active_anchors = len(obj.debug.get("anchor_pts", [])) if "anchor_pts" in obj.debug else obj.debug.get("total_anchors", 0)
        metrics = [
            f"FPS: {state.fps:.1f}",
            f"Conf: {obj.confidence:.2f}",
            f"Anchors: {active_anchors}"
        ]
        for i, text in enumerate(metrics):
            cv2.putText(vis_frame, text, (30, 85 + i * 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return vis_frame