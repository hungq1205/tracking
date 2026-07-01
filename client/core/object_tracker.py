import cv2
import numpy as np
import threading
import time
import torch
from typing import Tuple, Optional, Any
from .interfaces import IObjectTracker, ObjectTrack
from .helpers import clamp_box_xyxy, box_center, to_gray

_DEFAULT_REID_THRESHOLD = 0.4
_REFERENCE_DINO_THRESHOLD = 0.6  # min DINO score to capture a reference for non-saved objects


class GPUVIOAnchorBackend(IObjectTracker):
    """
    Lightweight tracking backend for Edge Devices.
    Runs ORB feature detection and Homography locally.
    Re-identifies using DINOv2 embeddings every 2 seconds.
    """
    def __init__(
        self,
        detector: Any,
        embedder: Any,
        nfeatures: int = 1000,
        max_total_anchors: int = 2000,
        renewal_interval: float = 2.0,
        reid_threshold: float = _DEFAULT_REID_THRESHOLD,
    ):
        self.detector = detector
        self.embedder = embedder
        self.lock = threading.Lock()
        self._orb = cv2.ORB_create(nfeatures=nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_total_anchors = max_total_anchors

        self.renewal_interval = renewal_interval
        self.reid_threshold = reid_threshold
        self.last_renewal_time = 0
        self._renewal_running = False
        self._active = False
        self.prompt = None
        self._description: Optional[str] = None  # appearance description for DINO

        self._ref_kp = None
        self._ref_desc = None
        self._ref_center_cpu = None
        self._init_wh = (0, 0)
        self._last_box = None
        self._ref_crop_emb = None   # reference DINOv2 embedding (torch.Tensor)
        self._ref_image: Optional[np.ndarray] = None  # reference crop (BGR) for display
        self._last_H = None

    def stop(self):
        self._active = False

    def initialize(
        self,
        first_frame_bgr: np.ndarray,
        prompt: str,
        description: Optional[str] = None,
        ref_embedding: Optional[torch.Tensor] = None,
        ref_image: Optional[np.ndarray] = None,
    ) -> ObjectTrack:
        self._active = True
        self.prompt = prompt
        self._description = description or prompt

        if ref_image is not None:
            self._ref_image = ref_image

        # Detect using appearance description for better accuracy
        det = self.detector.detect(first_frame_bgr, self._description)

        if det.score == 0.0:
            return ObjectTrack((0, 0, 0, 0), (0, 0), 0.0, False, "DETECTION_FAILED")

        h, w = first_frame_bgr.shape[:2]
        self._last_box = clamp_box_xyxy(det.box_xyxy, w, h)
        self._init_wh = (self._last_box[2] - self._last_box[0], self._last_box[3] - self._last_box[1])

        cx, cy = box_center(self._last_box)
        self._ref_center_cpu = np.array([[cx, cy, 1.0]], dtype=np.float32).T

        gray = to_gray(first_frame_bgr)
        kp, desc = self._orb.detectAndCompute(gray, None)

        if ref_embedding is not None:
            # Saved object: use the pre-computed reference embedding
            self._ref_crop_emb = ref_embedding
        elif det.score >= _REFERENCE_DINO_THRESHOLD:
            # Non-saved, confident detection: compute fresh embedding as reference
            self._ref_crop_emb = self.embedder.get_embedding(first_frame_bgr, self._last_box)
            # Capture reference image from the current detection crop
            if self._ref_image is None:
                x1, y1, x2, y2 = (int(max(0, v)) for v in self._last_box)
                x2, y2 = min(x2, w), min(y2, h)
                if x2 > x1 and y2 > y1:
                    self._ref_image = first_frame_bgr[y1:y2, x1:x2].copy()
        else:
            self._ref_crop_emb = self.embedder.get_embedding(first_frame_bgr, self._last_box)

        self._ref_kp = list(kp)
        self._ref_desc = np.array(desc)
        self.last_renewal_time = time.time()

        return ObjectTrack(
            box_xyxy=self._last_box,
            center_xy=box_center(self._last_box),
            confidence=det.score,
            visible=True,
            status="INITIALIZED",
            debug={"total_anchors": len(self._ref_kp)}
        )

    def _renewal_worker(self, renewal_frame: np.ndarray):
        """
        Runs every 2 s: re-detects with DINO (using description), compares embedding
        similarity to reference. Updates ORB anchors only if similarity >= reid_threshold.
        """
        try:
            if not self._active:
                return
            dino_query = self._description or self.prompt
            det = self.detector.detect(renewal_frame, dino_query)
            print(
                f"[TRACKER] Renewal DINO '{dino_query}' → score={det.score:.3f}",
                flush=True,
            )
            if det.score < 0.2:
                print("[TRACKER] Renewal skipped: DINO score too low", flush=True)
                return

            new_emb = self.embedder.get_embedding(renewal_frame, det.box_xyxy)
            if new_emb is None or self._ref_crop_emb is None:
                print("[TRACKER] Renewal skipped: missing embedding", flush=True)
                return

            similarity = torch.nn.functional.cosine_similarity(
                self._ref_crop_emb, new_emb
            ).item()
            accepted = similarity > self.reid_threshold
            print(
                f"[TRACKER] Re-ID similarity={similarity:.3f}  threshold={self.reid_threshold:.2f}  "
                f"→ {'ACCEPTED — anchors updated' if accepted else 'REJECTED'}",
                flush=True,
            )

            if accepted:
                gray = to_gray(renewal_frame)
                kp, desc = self._orb.detectAndCompute(gray, None)
                if desc is not None:
                    with self.lock:
                        self._last_box = det.box_xyxy
                        self._init_wh = (
                            self._last_box[2] - self._last_box[0],
                            self._last_box[3] - self._last_box[1],
                        )
                        cx, cy = box_center(self._last_box)
                        self._ref_center_cpu = np.array([[cx, cy, 1.0]], dtype=np.float32).T
                        self._ref_kp = list(kp)
                        self._ref_desc = np.array(desc)
        finally:
            self.last_renewal_time = time.time()
            self._renewal_running = False

    def update(self, frame_bgr: np.ndarray) -> ObjectTrack:
        if self.prompt and not self._renewal_running and (
            time.time() - self.last_renewal_time > self.renewal_interval
        ):
            self._renewal_running = True
            threading.Thread(
                target=self._renewal_worker, args=(frame_bgr.copy(),), daemon=True
            ).start()

        with self.lock:
            if self._ref_desc is None:
                return ObjectTrack((0, 0, 0, 0), (0, 0), 0, False, "NOT_INIT")

            gray = to_gray(frame_bgr)
            kp, desc = self._orb.detectAndCompute(gray, None)

            if desc is None or len(kp) < 10:
                return ObjectTrack(self._last_box, box_center(self._last_box), 0.0, False, "LOST")

            matches = self._matcher.match(self._ref_desc, desc)
            if len(matches) < 15:
                return ObjectTrack(self._last_box, box_center(self._last_box), 0.0, False, "LOST")

            src_pts = np.float32([self._ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            self._last_H = H
            if H is None:
                return ObjectTrack(self._last_box, box_center(self._last_box), 0.0, False, "LOST")

            transformed_homogeneous = np.dot(H, self._ref_center_cpu)
            new_center = (transformed_homogeneous[:2] / transformed_homogeneous[2]).flatten()

            cw, ch = self._init_wh
            new_box = (
                float(new_center[0] - cw / 2),
                float(new_center[1] - ch / 2),
                float(new_center[0] + cw / 2),
                float(new_center[1] + ch / 2),
            )

            h, w = frame_bgr.shape[:2]
            self._last_box = clamp_box_xyxy(new_box, w, h)

            anchor_pts = dst_pts[inliers.ravel() == 1].reshape(-1, 2).tolist()

            return ObjectTrack(
                box_xyxy=self._last_box,
                center_xy=tuple(new_center),
                confidence=float(np.mean(inliers)),
                visible=True,
                status="TRACKING",
                debug={"anchor_pts": anchor_pts}
            )
