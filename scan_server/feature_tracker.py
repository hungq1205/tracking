"""
Feature-based pose estimator for V-SLAM.

Geometry (dense depth) and localization (sparse features) are separated:
- Depth Anything provides dense depth → all pixels become 3D geometry
- This module tracks ORB features and estimates camera pose via PnP
  (depth at each keypoint lifts 2D matches to 3D-to-2D correspondences)

When depth is unavailable at a keypoint, falls back to Essential Matrix.
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class PoseFrame:
    pose: np.ndarray          # 4×4 float64 camera-to-world transform
    rel_pose: np.ndarray      # 4×4 relative pose from previous frame (prev→curr delta)
    keypoints: list           # cv2.KeyPoint list (current frame)
    descriptors: np.ndarray   # ORB descriptors (current frame)
    depth_map: np.ndarray     # HxW float32, used to lift keypoints to 3D


class FeatureTracker:
    """
    Tracks ORB features across frames and estimates absolute camera pose.

    Uses depth at matched keypoints for PnP (RGB-D mode).
    Falls back to Essential Matrix when depth is sparse or unavailable.
    """

    def __init__(
        self,
        n_features: int = 2000,
        min_matches: int = 8,
        pnp_min_inliers: int = 6,
    ):
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._min_matches = min_matches
        self._pnp_min_inliers = pnp_min_inliers

        self._world_pose = np.eye(4, dtype=np.float64)  # camera-to-world
        self._prev: Optional[PoseFrame] = None

    # ------------------------------------------------------------------ public

    def track(
        self,
        rgb: np.ndarray,
        depth_map: np.ndarray,
        K: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process one RGB frame + its depth map.

        Args:
            rgb:       HxWx3 uint8
            depth_map: HxW float32, metric depth in metres
            K:         3x3 intrinsic matrix; estimated from image size if None

        Returns:
            (world_pose, rel_pose) — both 4×4 float64.
            world_pose: camera-to-world absolute pose.
            rel_pose:   relative pose from previous frame (identity on first frame).
        """
        h, w = rgb.shape[:2]
        if K is None:
            K = _estimate_K(h, w)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kps, descs = self._orb.detectAndCompute(gray, None)

        identity = np.eye(4, dtype=np.float64)

        if descs is None or len(kps) < self._min_matches:
            frame = PoseFrame(
                pose=self._world_pose.copy(),
                rel_pose=identity,
                keypoints=kps or [],
                descriptors=descs if descs is not None else np.empty((0, 32), dtype=np.uint8),
                depth_map=depth_map,
            )
            self._prev = frame
            return self._world_pose.copy(), identity

        if self._prev is None or len(self._prev.keypoints) < self._min_matches:
            frame = PoseFrame(
                pose=self._world_pose.copy(),
                rel_pose=identity,
                keypoints=kps,
                descriptors=descs,
                depth_map=depth_map,
            )
            self._prev = frame
            return self._world_pose.copy(), identity

        # Match against previous frame
        matches = self._matcher.knnMatch(descs, self._prev.descriptors, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < self._min_matches:
            frame = PoseFrame(
                pose=self._world_pose.copy(),
                rel_pose=identity,
                keypoints=kps,
                descriptors=descs,
                depth_map=depth_map,
            )
            self._prev = frame
            return self._world_pose.copy(), identity

        pts_curr = np.float32([kps[m.queryIdx].pt for m in good])
        pts_prev = np.float32([self._prev.keypoints[m.trainIdx].pt for m in good])

        rel = self._estimate_relative_pose(
            pts_curr, pts_prev, kps, good, depth_map, K
        )

        # world_pose_prev × rel → current camera in world
        self._world_pose = self._prev.pose @ rel

        frame = PoseFrame(
            pose=self._world_pose.copy(),
            rel_pose=rel.copy(),
            keypoints=kps,
            descriptors=descs,
            depth_map=depth_map,
        )
        self._prev = frame
        return self._world_pose.copy(), rel.copy()

    def reset(self):
        self._world_pose = np.eye(4, dtype=np.float64)
        self._prev = None

    # --------------------------------------------------------------- private

    def _estimate_relative_pose(
        self,
        pts_curr: np.ndarray,
        pts_prev: np.ndarray,
        kps_curr,
        good_matches,
        depth_map: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate relative pose (prev → curr).

        Prefers PnP: lifts prev-frame keypoints to 3D using prev depth,
        then solves 3D-to-2D correspondence with curr 2D points.
        Falls back to Essential Matrix if too few points have valid depth.
        """
        h, w = depth_map.shape[:2]
        prev_depth = self._prev.depth_map

        obj_pts = []
        img_pts = []
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        for i, m in enumerate(good_matches):
            px, py = pts_prev[i]
            pi, pj = int(round(py)), int(round(px))
            if 0 <= pi < prev_depth.shape[0] and 0 <= pj < prev_depth.shape[1]:
                d = float(prev_depth[pi, pj])
                if 0.1 < d < 10.0:
                    X = (px - cx) * d / fx
                    Y = (py - cy) * d / fy
                    obj_pts.append([X, Y, d])
                    img_pts.append(pts_curr[i])

        if len(obj_pts) >= self._pnp_min_inliers:
            obj_arr = np.array(obj_pts, dtype=np.float64)
            img_arr = np.array(img_pts, dtype=np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_arr, img_arr, K.astype(np.float64), None,
                iterationsCount=200, reprojectionError=2.0, confidence=0.99,
            )
            if ok and inliers is not None and len(inliers) >= self._pnp_min_inliers:
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = tvec.ravel()
                # T is world→camera; invert to get camera→world delta
                return np.linalg.inv(T)

        # Fallback: Essential Matrix
        # recoverPose returns t as a unit vector — scale-ambiguous.
        # We recover metric scale by looking at the depth at inlier keypoints:
        # the median scene depth at matched features is a good proxy for how far
        # the camera moved (valid for small-baseline / forward-facing motion).
        E, e_mask = cv2.findEssentialMat(
            pts_curr, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return np.eye(4, dtype=np.float64)

        _, R, t, pose_mask = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=e_mask)

        # Collect depth values at inlier correspondences
        inlier_depths: list = []
        valid_inliers = (pose_mask.ravel() > 0) if pose_mask is not None else np.ones(len(pts_prev), dtype=bool)
        for i, (px, py) in enumerate(pts_prev):
            if not valid_inliers[i]:
                continue
            pi, pj = int(round(py)), int(round(px))
            if 0 <= pi < prev_depth.shape[0] and 0 <= pj < prev_depth.shape[1]:
                d = float(prev_depth[pi, pj])
                if 0.1 < d < 10.0:
                    inlier_depths.append(d)

        # Median depth → approximate metric scale for translation direction
        scale = float(np.median(inlier_depths)) if len(inlier_depths) >= 3 else 0.3

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.ravel() * scale
        return np.linalg.inv(T)


# ------------------------------------------------------------------ helpers

def _estimate_K(h: int, w: int) -> np.ndarray:
    """Pinhole estimate when no calibration is available."""
    f = max(w, h) * 0.8
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
