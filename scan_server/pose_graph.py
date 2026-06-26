"""
Keyframe-based pose graph for VI-SLAM.

Nodes: Keyframe — camera pose (4x4 c2w) + ORB descriptors + 3D map points
Edges:
  - Odometry:      consecutive keyframe relative transform from FeatureTracker
  - Loop closure:  relative transform detected by ORB matching + PnP verification

Optimization: scipy.optimize.least_squares (Levenberg-Marquardt) on all keyframe
poses simultaneously, with the first pose fixed as an anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class Keyframe:
    id: int                      # sequential index in PoseGraph.keyframes
    frame_idx: int               # absolute frame counter from ScanSession
    pose: np.ndarray             # 4x4 float64 camera-to-world
    descriptors: np.ndarray      # Nxd uint8 ORB descriptors
    keypoints_2d: np.ndarray     # Nx2 float32 image coordinates
    points_3d: np.ndarray        # Nx3 float32 world-space 3D points


@dataclass
class Edge:
    src: int                     # source keyframe id
    dst: int                     # destination keyframe id
    T_rel: np.ndarray            # 4x4 relative transform src→dst
    weight: float = 1.0          # 1.0 odometry, 5.0 loop closure


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _pose_to_6(T: np.ndarray) -> np.ndarray:
    """4x4 c2w → [tx, ty, tz, rx, ry, rz] (axis-angle rotation)."""
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def _6_to_pose(v: np.ndarray) -> np.ndarray:
    """[tx, ty, tz, rx, ry, rz] → 4x4 c2w."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = v[:3]
    T[:3, :3] = Rotation.from_rotvec(v[3:]).as_matrix()
    return T


# ── Pose Graph ──────────────────────────────────────────────────────────────────


class PoseGraph:
    LOOP_MIN_MATCHES = 15
    LOOP_MIN_INLIERS = 10
    LOOP_MIN_FRAME_GAP = 30   # frames between the current and the candidate keyframe
    LOOP_MAX_SEARCH = 50      # only look at the last N keyframes for O(N) cost
    LOOP_WEIGHT = 5.0

    def __init__(self) -> None:
        self.keyframes: List[Keyframe] = []
        self.edges: List[Edge] = []
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # ── public API ────────────────────────────────────────────────────────────

    def add_keyframe(
        self,
        frame_idx: int,
        pose: np.ndarray,
        descriptors: np.ndarray,
        keypoints_2d: np.ndarray,
        points_3d: np.ndarray,
    ) -> int:
        """Append a new keyframe; returns its id (= index in self.keyframes)."""
        kf_id = len(self.keyframes)
        self.keyframes.append(
            Keyframe(
                id=kf_id,
                frame_idx=frame_idx,
                pose=pose.copy(),
                descriptors=descriptors.copy(),
                keypoints_2d=keypoints_2d.copy(),
                points_3d=points_3d.copy(),
            )
        )
        return kf_id

    def add_odometry_edge(self, src_id: int, dst_id: int, T_rel: np.ndarray) -> None:
        self.edges.append(Edge(src=src_id, dst=dst_id, T_rel=T_rel.copy(), weight=1.0))

    def detect_loop(
        self,
        descriptors: np.ndarray,
        keypoints_2d: np.ndarray,
        points_3d: np.ndarray,
        K: np.ndarray,
    ) -> Optional[int]:
        """
        Search past keyframes for a loop closure candidate.

        Matches current-frame ORB descriptors against stored keyframes, verifies
        geometry with PnP, and returns the matched keyframe id or None.
        """
        n = len(self.keyframes)
        if n < 2:
            return None

        current_kf = self.keyframes[-1]
        search_end = n - 1   # exclude the most recently added keyframe (itself)
        search_start = max(0, search_end - self.LOOP_MAX_SEARCH)

        best_id = None
        best_inliers = self.LOOP_MIN_INLIERS - 1  # need to beat this

        for i in range(search_start, search_end):
            kf = self.keyframes[i]
            frame_gap = current_kf.frame_idx - kf.frame_idx
            if frame_gap < self.LOOP_MIN_FRAME_GAP:
                continue
            if len(kf.descriptors) < self.LOOP_MIN_MATCHES:
                continue

            try:
                raw_matches = self._matcher.knnMatch(descriptors, kf.descriptors, k=2)
            except Exception:
                continue

            good = [
                m for m, n2 in raw_matches
                if m.distance < 0.75 * n2.distance
            ]
            if len(good) < self.LOOP_MIN_MATCHES:
                continue

            # PnP verification: world points from past kf ↔ 2D points in current frame
            obj_pts = np.array(
                [kf.points_3d[m.trainIdx] for m in good], dtype=np.float64
            )
            img_pts = np.array(
                [keypoints_2d[m.queryIdx] for m in good], dtype=np.float64
            )

            try:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K.astype(np.float64), None,
                    iterationsCount=200, reprojectionError=2.0, confidence=0.99,
                )
            except Exception:
                continue

            if ok and inliers is not None and len(inliers) > best_inliers:
                best_inliers = len(inliers)
                best_id = i

        if best_id is not None:
            print(
                f"[PoseGraph] Loop closure: kf{current_kf.id} ↔ kf{best_id} "
                f"({best_inliers} inliers)"
            )
        return best_id

    def add_loop_edge(self, src_id: int, dst_id: int, T_rel: np.ndarray) -> None:
        self.edges.append(
            Edge(src=src_id, dst=dst_id, T_rel=T_rel.copy(), weight=self.LOOP_WEIGHT)
        )

    def optimize(self) -> dict[int, np.ndarray]:
        """
        Optimize all keyframe poses (Levenberg-Marquardt via scipy).

        First keyframe is fixed as the anchor.
        Returns {keyframe_id: optimized 4x4 pose}.
        """
        n = len(self.keyframes)
        if n < 2 or not self.edges:
            return {kf.id: kf.pose for kf in self.keyframes}

        anchor_pose = self.keyframes[0].pose
        # Initial 6-DOF vector for keyframes 1..N-1
        x0 = np.concatenate([_pose_to_6(kf.pose) for kf in self.keyframes[1:]])

        def residuals(x: np.ndarray) -> np.ndarray:
            poses = [anchor_pose] + [
                _6_to_pose(x[i * 6:(i + 1) * 6]) for i in range(n - 1)
            ]
            res = []
            for e in self.edges:
                if e.src >= len(poses) or e.dst >= len(poses):
                    continue
                T_pred = np.linalg.inv(poses[e.src]) @ poses[e.dst]
                dT = np.linalg.inv(e.T_rel) @ T_pred
                t_err = dT[:3, 3]
                try:
                    r_err = Rotation.from_matrix(dT[:3, :3]).as_rotvec()
                except Exception:
                    r_err = np.zeros(3)
                err = np.concatenate([t_err, r_err]) * e.weight
                res.extend(err.tolist())
            return np.array(res)

        try:
            result = least_squares(residuals, x0, method="lm", max_nfev=500)
            x_opt = result.x
            print(
                f"[PoseGraph] Optimized {n} keyframes, {len(self.edges)} edges "
                f"(cost: {result.cost:.4f})"
            )
        except Exception as exc:
            print(f"[PoseGraph] Optimization failed ({exc}), keeping original poses.")
            x_opt = x0

        opt = {self.keyframes[0].id: anchor_pose}
        for i, kf in enumerate(self.keyframes[1:]):
            opt[kf.id] = _6_to_pose(x_opt[i * 6:(i + 1) * 6])
        return opt
