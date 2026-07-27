"""
Standalone offline Gradio tool: upload a video + a target object label,
get back a video with the tracked object's bounding box (yellow) and
detected hand bounding box(es) (cyan) drawn on every frame.

This is a Python port of the two Android/Kotlin modules it's based on:
  - client/android/.../tracking/TrackingBackend.kt (ORB + homography
    object tracking, periodic GroundingDINO re-identify)
  - client/android/.../tracking/HandTracker.kt (MediaPipe HandLandmarker)

The initial detection + periodic renewal reuse this repo's own
GroundingDINODetector (server/tools/detector.py) instead of a gRPC call,
since this tool runs everything in-process, offline, on a whole video.

Run: server/.venv/bin/python test_module/tracking_preview_app/app.py
(needs server/.venv's mediapipe/torch/transformers/gradio/opencv deps)
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np
import gradio as gr

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SERVER_DIR = os.path.join(_REPO_ROOT, "server")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from tools.detector import GroundingDINODetector  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402

_HAND_MODEL_PATH = os.path.join(_REPO_ROOT, "client", "hand_landmarker.task")

# --- Tracking tuning, mirrors TrackingBackend.kt's own constants ---
ORB_NFEATURES = 800
RENEWAL_INTERVAL_S = 4.0
INIT_CONFIDENCE_MIN = 0.45
RENEWAL_CONFIDENCE_MIN = 0.2
EMA_ALPHA = 0.4
MIN_MATCHES = 10

_detector_cache: dict[str, GroundingDINODetector] = {}


def _get_detector(model_id: str = "IDEA-Research/grounding-dino-base") -> GroundingDINODetector:
    if model_id not in _detector_cache:
        _detector_cache[model_id] = GroundingDINODetector(model_id=model_id)
    return _detector_cache[model_id]


def _clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(x1, w)),
        max(0.0, min(y1, h)),
        max(0.0, min(x2, w)),
        max(0.0, min(y2, h)),
    )


def _boxes_overlap(a, b):
    if a is None or b is None:
        return False
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class ObjectTracker:
    """Python port of TrackingBackend.kt — ORB + homography tracking with
    periodic GroundingDINO renewal, single physical target only."""

    def __init__(self, detector: GroundingDINODetector, prompt: str):
        self.detector = detector
        self.prompt = prompt
        self.orb = cv2.ORB_create(ORB_NFEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.active = False
        self.ref_kp = None
        self.ref_desc = None
        self.ref_center = None  # (cx, cy) in the reference frame
        self.init_w = 0.0
        self.init_h = 0.0

        self.last_box = None
        self.smooth_cx = 0.0
        self.smooth_cy = 0.0
        self.last_renewal_t = 0.0

    def initialize(self, frame_bgr) -> bool:
        h, w = frame_bgr.shape[:2]
        det = self.detector.detect(frame_bgr, self.prompt)
        if det.score < INIT_CONFIDENCE_MIN:
            return False
        box = _clamp_box(det.box_xyxy, w, h)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kp) == 0:
            return False

        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        self.ref_kp = kp
        self.ref_desc = desc
        self.ref_center = (cx, cy)
        self.init_w = box[2] - box[0]
        self.init_h = box[3] - box[1]
        self.last_box = box
        self.smooth_cx, self.smooth_cy = cx, cy
        self.active = True
        self.last_renewal_t = time.monotonic()
        return True

    def update(self, frame_bgr, hand_box=None):
        """Returns (box_xyxy, visible: bool)."""
        if not self.active:
            return None, False
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kp) == 0:
            return self.last_box, False

        matches = self.matcher.match(self.ref_desc, desc)
        if len(matches) < MIN_MATCHES:
            return self.last_box, False

        src_pts = np.float32([self.ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None:
            return self.last_box, False

        ref_center_h = np.array([[self.ref_center[0]], [self.ref_center[1]], [1.0]])
        transformed = homography @ ref_center_h
        wv = transformed[2, 0]
        if wv == 0:
            return self.last_box, False
        raw_cx = transformed[0, 0] / wv
        raw_cy = transformed[1, 0] / wv

        self.smooth_cx = EMA_ALPHA * raw_cx + (1 - EMA_ALPHA) * self.smooth_cx
        self.smooth_cy = EMA_ALPHA * raw_cy + (1 - EMA_ALPHA) * self.smooth_cy
        box = _clamp_box(
            (
                self.smooth_cx - self.init_w / 2,
                self.smooth_cy - self.init_h / 2,
                self.smooth_cx + self.init_w / 2,
                self.smooth_cy + self.init_h / 2,
            ),
            w, h,
        )
        self.last_box = box

        now = time.monotonic()
        if now - self.last_renewal_t > RENEWAL_INTERVAL_S and not _boxes_overlap(box, hand_box):
            self.last_renewal_t = now
            self._renewal(frame_bgr, kp, desc)

        return box, True

    def _renewal(self, frame_bgr, kp, desc):
        """Re-identify against the SAME already-tracked object (spatial
        consistency required), same guard TrackingBackend.kt's renewal()
        uses — never lets the reference jump to an unrelated detection."""
        h, w = frame_bgr.shape[:2]
        det = self.detector.detect(frame_bgr, self.prompt)
        if det.score < RENEWAL_CONFIDENCE_MIN:
            return
        raw_box = det.box_xyxy
        if not _boxes_overlap(self.last_box, raw_box):
            return
        box = _clamp_box(raw_box, w, h)
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        self.ref_kp = kp
        self.ref_desc = desc
        self.ref_center = (cx, cy)
        self.init_w = box[2] - box[0]
        self.init_h = box[3] - box[1]
        self.last_box = box
        self.smooth_cx, self.smooth_cy = cx, cy


def _hand_boxes_from_result(result, w, h):
    boxes = []
    for hand_landmarks in result.hand_landmarks:
        xs = [lm.x * w for lm in hand_landmarks]
        ys = [lm.y * h for lm in hand_landmarks]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def _draw_box(frame, box, color, label):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def process_video(video_path, target_label, gdino_model_id, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Please upload a video.")
    if not target_label or not target_label.strip():
        raise gr.Error("Please enter a target object label.")

    progress(0, desc="Loading models...")
    detector = _get_detector(gdino_model_id.strip() or "IDEA-Research/grounding-dino-base")

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    tracker = ObjectTracker(detector, target_label.strip())
    frame_idx = 0
    initialized = False
    status_msg = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            timestamp_ms = int((frame_idx / fps) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_boxes = _hand_boxes_from_result(hand_result, w, h)
            primary_hand_box = hand_boxes[0] if hand_boxes else None

            if not initialized:
                progress(frame_idx / max(total_frames, 1), desc=f"Detecting '{target_label}'...")
                initialized = tracker.initialize(frame)
                if not initialized and frame_idx > int(fps * 5):
                    # Give up trying to init after ~5s of no confident detection,
                    # still emit hand boxes on every frame so far/after.
                    status_msg = "target never confidently detected"

            obj_box, obj_visible = (None, False)
            if initialized:
                obj_box, obj_visible = tracker.update(frame, primary_hand_box)

            if obj_box is not None:
                color = (0, 215, 255) if obj_visible else (0, 120, 180)  # BGR yellow / dim yellow
                _draw_box(frame, obj_box, color, target_label if obj_visible else f"{target_label} (lost)")

            for hb in hand_boxes:
                _draw_box(frame, hb, (255, 220, 0), "hand")  # BGR cyan-ish

            writer.write(frame)

            if total_frames > 0 and frame_idx % 5 == 0:
                progress(min(frame_idx / total_frames, 0.99), desc=f"Processing frame {frame_idx}/{total_frames}")
    finally:
        cap.release()
        writer.release()
        hand_landmarker.close()

    progress(1.0, desc="Done")
    note = "" if initialized else f" (warning: {status_msg or 'target was never detected'})"
    return out_path, f"Processed {frame_idx} frames at {fps:.1f} fps.{note}"


with gr.Blocks(title="Hand + Object Tracking Preview") as demo:
    gr.Markdown(
        "# Hand + Object Tracking Preview\n"
        "Upload a video and a target object label. Reuses this repo's "
        "GroundingDINO detector + an ORB/homography tracker (ported from "
        "`TrackingBackend.kt`) for the object, and MediaPipe's "
        "HandLandmarker (ported from `HandTracker.kt`) for hands."
    )
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input video")
            target_in = gr.Textbox(label="Target object label", placeholder="e.g. water bottle")
            model_in = gr.Textbox(
                label="GroundingDINO model id",
                value="IDEA-Research/grounding-dino-base",
            )
            run_btn = gr.Button("Process video", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Output video (object=yellow, hand=cyan)")
            status_out = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=process_video,
        inputs=[video_in, target_in, model_in],
        outputs=[video_out, status_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("TRACKING_PREVIEW_PORT", "7864")))
