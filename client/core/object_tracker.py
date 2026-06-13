import cv2
import numpy as np
import threading
import time
import torch
from typing import Tuple, Optional, Any
from .interfaces import IObjectTracker, ObjectTrack
from .helpers import clamp_box_xyxy, box_center, to_gray

class GPUVIOAnchorBackend(IObjectTracker):
    """
    Lightweight tracking backend for Edge Devices. 
    Runs ORB feature detection and Homography locally.
    """
    def __init__(self, detector: Any, embedder: Any, nfeatures: int = 1000, max_total_anchors: int = 2000, renewal_interval: float = 1.5):
        self.detector = detector
        self.embedder = embedder
        self.lock = threading.Lock()
        self._orb = cv2.ORB_create(nfeatures=nfeatures)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_total_anchors = max_total_anchors

        self.renewal_interval = renewal_interval
        self.last_renewal_time = 0
        self._renewal_running = False
        self.prompt = None
        
        self._ref_kp = None
        self._ref_desc = None
        self._ref_center_cpu = None  
        self._init_wh = (0, 0)
        self._last_box = None
        self._ref_crop_emb = None
        self._last_H = None

    def initialize(self, first_frame_bgr: np.ndarray, prompt: str) -> ObjectTrack:
        self.prompt = prompt
        # Calls RemoteGroundingDINO via gRPC
        det = self.detector.detect(first_frame_bgr, prompt)
        
        if det.score == 0.0:
            return ObjectTrack((0,0,0,0), (0,0), 0.0, False, "DETECTION_FAILED")

        h, w = first_frame_bgr.shape[:2]
        self._last_box = clamp_box_xyxy(det.box_xyxy, w, h)
        self._init_wh = (self._last_box[2] - self._last_box[0], self._last_box[3] - self._last_box[1])
        
        cx, cy = box_center(self._last_box)
        self._ref_center_cpu = np.array([[cx, cy, 1.0]], dtype=np.float32).T

        gray = to_gray(first_frame_bgr)
        kp, desc = self._orb.detectAndCompute(gray, None)
        
        # Calls RemoteEmbedder via gRPC
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
        Validates tracking against server detection using the frame captured at request time.
        Replaces anchors if the object is confirmed via embedding similarity.
        """
        try:
            det = self.detector.detect(renewal_frame, self.prompt)
            if det.score < 0.2: return

            new_emb = self.embedder.get_embedding(renewal_frame, det.box_xyxy)
            if new_emb is None or self._ref_crop_emb is None: return

            # Similarity check (> 0.75) against initial embedding
            similarity = torch.nn.functional.cosine_similarity(self._ref_crop_emb, new_emb).item()
            
            if similarity > 0.75:
                gray = to_gray(renewal_frame)
                kp, desc = self._orb.detectAndCompute(gray, None)
                if desc is not None:
                    with self.lock:
                        self._last_box = det.box_xyxy
                        self._init_wh = (self._last_box[2] - self._last_box[0], self._last_box[3] - self._last_box[1])
                        cx, cy = box_center(self._last_box)
                        self._ref_center_cpu = np.array([[cx, cy, 1.0]], dtype=np.float32).T
                        self._ref_kp = list(kp)
                        self._ref_desc = np.array(desc)
        finally:
            self.last_renewal_time = time.time()
            self._renewal_running = False

    def update(self, frame_bgr: np.ndarray) -> ObjectTrack:
        # Trigger periodic renewal in background
        if self.prompt and not self._renewal_running and (time.time() - self.last_renewal_time > self.renewal_interval):
            self._renewal_running = True
            threading.Thread(target=self._renewal_worker, args=(frame_bgr.copy(),), daemon=True).start()

        with self.lock:
            if self._ref_desc is None: return ObjectTrack((0,0,0,0),(0,0),0,False,"NOT_INIT")

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
            
            # Extract anchor points for the renderer
            anchor_pts = dst_pts[inliers.ravel() == 1].reshape(-1, 2).tolist()

            return ObjectTrack(
                box_xyxy=self._last_box,
                center_xy=tuple(new_center),
                confidence=float(np.mean(inliers)),
                visible=True,
                status="TRACKING",
                debug={"anchor_pts": anchor_pts}
            )