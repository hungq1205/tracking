from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class LocalizationResult:
    position: Optional[np.ndarray]  # [x, y, z] in map world space, or None
    confidence: float               # 0.0 – 1.0
    source: str                     # "keyframe_match" | "vo_dead_reckoning" | "none"


class LocalizationEngine:
    """
    Localizes the camera against ORB keyframes saved during offline scan.

    Keyframes index at maps/{location_id}/keyframes/index.json:
      { "camera_K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
        "keyframes": [{"id":0, "file":"kf000000.npz", "pose_c2w":[[4x4]]},...] }

    Each .npz: descriptors (Nxd uint8), keypoints_2d (Nx2 float32), points_3d (Nx3 float32 world).
    """

    ORB_FEATURES = 2000
    MIN_MATCHES = 8       # minimum good descriptor matches to attempt PnP
    MIN_INLIERS = 6       # minimum PnP RANSAC inliers to accept the pose
    RATIO_THRESH = 0.75   # Lowe's ratio test threshold

    def __init__(self, map_dir: str):
        self.orb = cv2.ORB_create(nfeatures=self.ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._keyframes: List[dict] = []    # {descriptors, points_3d, pose_c2w}
        self._camera_K: Optional[np.ndarray] = None
        self._last_position: Optional[np.ndarray] = None
        self._load(map_dir)

    def _load(self, map_dir: str) -> None:
        index_path = os.path.join(map_dir, "keyframes", "index.json")
        if not os.path.exists(index_path):
            print(f"[Localization] No keyframe index at {index_path} — localization disabled.")
            return
        with open(index_path) as f:
            index = json.load(f)
        self._camera_K = np.array(index["camera_K"], dtype=np.float32)
        kf_dir = os.path.join(map_dir, "keyframes")
        loaded = 0
        for entry in index.get("keyframes", []):
            npz_path = os.path.join(kf_dir, entry["file"])
            if not os.path.exists(npz_path):
                continue
            npz = np.load(npz_path)
            self._keyframes.append({
                "descriptors": npz["descriptors"],
                "keypoints_2d": npz["keypoints_2d"],
                "points_3d": npz["points_3d"],
                "pose_c2w": np.array(entry["pose_c2w"], dtype=np.float64),
            })
            loaded += 1
        print(f"[Localization] Loaded {loaded} keyframes from {map_dir}")

    @property
    def available(self) -> bool:
        return len(self._keyframes) > 0 and self._camera_K is not None

    def localize(self, frame_bgr: np.ndarray) -> LocalizationResult:
        if not self.available:
            return LocalizationResult(self._last_position, 0.0, "none")

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None or len(kp) < self.MIN_MATCHES:
            return LocalizationResult(self._last_position, 0.1, "vo_dead_reckoning")

        # Find the keyframe with most good matches (ratio-test filtered)
        best_good: List = []
        best_kf: Optional[dict] = None
        for kf in self._keyframes:
            matches = self.matcher.knnMatch(des, kf["descriptors"], k=2)
            good = [m for m, n in matches if m.distance < self.RATIO_THRESH * n.distance]
            if len(good) > len(best_good):
                best_good = good
                best_kf = kf

        if best_kf is None or len(best_good) < self.MIN_MATCHES:
            return LocalizationResult(self._last_position, 0.1, "vo_dead_reckoning")

        # Build 2D↔3D correspondences for PnP
        pts_2d = np.array([kp[m.queryIdx].pt for m in best_good], dtype=np.float32)
        pts_3d = best_kf["points_3d"][[m.trainIdx for m in best_good]]

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self._camera_K, None,
            iterationsCount=200,
            reprojectionError=8.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or inliers is None or len(inliers) < self.MIN_INLIERS:
            return LocalizationResult(self._last_position, 0.15, "vo_dead_reckoning")

        R, _ = cv2.Rodrigues(rvec)
        # Camera position in world coords: C = -R^T * t
        position = (-R.T @ tvec).flatten().astype(np.float32)
        confidence = min(1.0, len(inliers) / 30.0)
        self._last_position = position
        return LocalizationResult(position, confidence, "keyframe_match")
