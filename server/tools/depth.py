import sys
import os
import cv2
import numpy as np
from typing import Optional

OBSTACLE_RELATIVE_THRESHOLD = 0.35  # flag when corridor depth < 35% of scene median
CORRIDOR_FRACTION = 1 / 3
STEREO_OBSTACLE_THRESHOLD_M = 1.5   # metres — obstacle alert for StereoDepthDetector


def _estimate_K(h: int, w: int) -> np.ndarray:
    f = max(w, h) * 0.8
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


class SparseObstacleDetector:
    """
    Obstacle detection via ORB feature triangulation between consecutive frames.

    No neural network required — uses Essential Matrix + triangulation to get
    scale-relative depth.  Returns a relative depth ratio instead of metres;
    the NavigationAgent threshold is calibrated accordingly.
    """

    def __init__(self, n_features: int = 1000):
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_kps = None
        self._prev_descs = None

    def check_obstacle(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Returns (obstacle_present, relative_depth).

        relative_depth is corridor_depth / median_scene_depth.
        Values < OBSTACLE_RELATIVE_THRESHOLD indicate an obstacle in the corridor.
        Returns (False, 1.0) when there is insufficient data for a decision.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = self._orb.detectAndCompute(gray, None)

        if descs is None or len(kps) < 10 or self._prev_descs is None:
            self._prev_kps, self._prev_descs = kps, descs
            return False, 1.0

        h, w = gray.shape
        K = _estimate_K(h, w)

        matches = self._matcher.knnMatch(descs, self._prev_descs, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 8:
            self._prev_kps, self._prev_descs = kps, descs
            return False, 1.0

        pts_curr = np.float32([kps[m.queryIdx].pt for m in good])
        pts_prev = np.float32([self._prev_kps[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            self._prev_kps, self._prev_descs = kps, descs
            return False, 1.0

        _, R, t, pose_mask = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=mask)
        inlier = pose_mask.ravel() > 0
        if inlier.sum() < 5:
            self._prev_kps, self._prev_descs = kps, descs
            return False, 1.0

        # Triangulate (unit translation → scale-relative depth)
        P1 = K @ np.eye(3, 4)
        P2 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, pts_prev[inlier].T, pts_curr[inlier].T)
        pts3d = (pts4d[:3] / pts4d[3]).T
        depths = pts3d[:, 2]
        valid = (depths > 0.01) & (depths < 1e4)

        self._prev_kps, self._prev_descs = kps, descs

        if valid.sum() < 5:
            return False, 1.0

        depths = depths[valid]
        pts_u = pts_curr[inlier][valid]
        median_depth = float(np.median(depths))

        cx_start = int(w * (0.5 - CORRIDOR_FRACTION / 2))
        cx_end = int(w * (0.5 + CORRIDOR_FRACTION / 2))
        in_corridor = (pts_u[:, 0] >= cx_start) & (pts_u[:, 0] < cx_end)

        if in_corridor.sum() < 3:
            return False, 1.0

        corridor_p10 = float(np.percentile(depths[in_corridor], 10))
        relative = corridor_p10 / max(median_depth, 1e-6)
        return relative < OBSTACLE_RELATIVE_THRESHOLD, relative


class StereoDepthDetector:
    """
    Obstacle detection via 2-frame plane sweep stereo with metric depth.

    Uses ORB + Essential Matrix for relative pose, then plane_sweep_stereo
    from scan_server/mvs.py for dense metric depth.  Imported lazily so the
    server does not need scan_server on the Python path unless DEPTH_MODEL=stereo.

    Returns (obstacle_present, min_depth_metres) — metric scale from E-matrix
    baseline estimation (normalized to expected walking speed).
    """

    N_DEPTHS = 32  # fewer hypotheses for real-time speed

    def __init__(self, n_features: int = 1000):
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_kps = None
        self._prev_descs = None
        self._prev_pose: Optional[np.ndarray] = None

        # Lazy import — scan_server must be on sys.path
        scan_root = os.path.join(os.path.dirname(__file__), "..", "..", "scan_server")
        if scan_root not in sys.path:
            sys.path.insert(0, os.path.abspath(scan_root))
        from mvs import plane_sweep_stereo as _pss  # noqa: F401
        self._plane_sweep = _pss

    def check_obstacle(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Returns (obstacle_present, min_depth_metres).

        Falls back to (False, 1.0) when there is insufficient data.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = self._orb.detectAndCompute(gray, None)

        if descs is None or len(kps) < 10 or self._prev_descs is None:
            self._prev_frame = frame_bgr.copy()
            self._prev_kps, self._prev_descs = kps, descs
            return False, 1.0

        h, w = gray.shape
        K = _estimate_K(h, w)

        matches = self._matcher.knnMatch(descs, self._prev_descs, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 8:
            self._prev_frame = frame_bgr.copy()
            self._prev_kps, self._prev_descs = kps, descs
            return False, 1.0

        pts_curr = np.float32([kps[m.queryIdx].pt for m in good])
        pts_prev = np.float32([self._prev_kps[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            self._prev_frame = frame_bgr.copy()
            self._prev_kps, self._prev_descs = kps, descs
            return False, 1.0

        _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=mask)

        # Build camera-to-world poses: prev = identity, curr = [R|t] relative
        pose_prev = np.eye(4, dtype=np.float64)
        pose_curr = np.eye(4, dtype=np.float64)
        pose_curr[:3, :3] = R
        pose_curr[:3, 3] = t.ravel()

        # Convert BGR→RGB for MVS
        ref_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src_rgb = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2RGB)

        try:
            depth_map = self._plane_sweep(
                ref_rgb, [src_rgb], K, pose_curr, [pose_prev],
                depth_min=0.2, depth_max=8.0, n_depths=self.N_DEPTHS,
            )
        except Exception:
            self._prev_frame = frame_bgr.copy()
            self._prev_kps, self._prev_descs = kps, descs
            return False, 1.0

        self._prev_frame = frame_bgr.copy()
        self._prev_kps, self._prev_descs = kps, descs

        if depth_map is None or depth_map.max() == 0:
            return False, 1.0

        cx_start = int(w * (0.5 - CORRIDOR_FRACTION / 2))
        cx_end = int(w * (0.5 + CORRIDOR_FRACTION / 2))
        corridor_depth = depth_map[:, cx_start:cx_end]
        valid = corridor_depth > 0.1
        if not valid.any():
            return False, 1.0

        min_depth = float(np.percentile(corridor_depth[valid], 10))
        return min_depth < STEREO_OBSTACLE_THRESHOLD_M, min_depth


class DA3DepthDetector:
    """
    Metric obstacle detection via Depth Anything 3 + sparse ORB scale alignment.

    Per frame:
      1. ORB match + Essential Matrix → triangulate sparse metric anchors
      2. DA3 inference → dense relative depth
      3. RANSAC align_metric_depth() → metric depth map
      4. Corridor 10th-percentile depth → obstacle decision
    """

    OBSTACLE_THRESHOLD_M = 1.5

    def __init__(self, model_id: str = "depth-anything/da3-large"):
        scan_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "scan_server")
        )
        if scan_root not in sys.path:
            sys.path.insert(0, scan_root)

        from da3_wrapper import build_estimator
        from mvs import align_metric_depth as _align_fn

        self._da3 = build_estimator(prefer_da3=True, model_id=model_id)
        self._align = _align_fn
        self._orb = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_kps = None
        self._prev_descs = None
        self._last_scale: Optional[tuple] = None

    def check_obstacle(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Returns (obstacle_present, min_depth_metres).

        Falls back to (False, 1.0) when scale cannot be estimated.
        """
        h_bgr, w_bgr = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, descs = self._orb.detectAndCompute(gray, None)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        K = _estimate_K(h_bgr, w_bgr)

        da3_frame = self._da3.estimate(rgb)
        da3_rel = da3_frame.depth_map  # HxW float32, relative depth

        # Try to build sparse metric anchors from ORB triangulation
        if descs is not None and self._prev_descs is not None and len(kps) >= 10:
            matches = self._matcher.knnMatch(descs, self._prev_descs, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= 8:
                pts_curr = np.float32([kps[m.queryIdx].pt for m in good])
                pts_prev = np.float32([self._prev_kps[m.trainIdx].pt for m in good])
                E, e_mask = cv2.findEssentialMat(
                    pts_curr, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                if E is not None:
                    _, R, t, p_mask = cv2.recoverPose(E, pts_curr, pts_prev, K, mask=e_mask)
                    inlier = p_mask.ravel() > 0
                    if inlier.sum() >= 5:
                        P1 = K @ np.eye(3, 4)
                        P2 = K @ np.hstack([R, t])
                        pts4d = cv2.triangulatePoints(
                            P1, P2, pts_prev[inlier].T, pts_curr[inlier].T
                        )
                        pts3d_cam = (pts4d[:3] / pts4d[3]).T
                        depth_valid = (pts3d_cam[:, 2] > 0.1) & (pts3d_cam[:, 2] < 8.0)
                        if depth_valid.sum() >= 5:
                            # Alignment in current-frame camera space (pose_c2w = identity)
                            result = self._align(
                                da3_rel,
                                pts3d_cam[depth_valid],
                                pts_curr[inlier][depth_valid].astype(np.float64),
                                np.eye(4, dtype=np.float64),
                                K,
                            )
                            if result is not None:
                                self._last_scale = (result[0], result[1])

        self._prev_frame = frame_bgr.copy()
        self._prev_kps, self._prev_descs = kps, descs

        if self._last_scale is None:
            return False, 1.0

        s, t = self._last_scale
        depth_metric = np.clip(s * da3_rel.astype(np.float64) + t, 0.0, 8.0).astype(np.float32)

        cx_start = int(w_bgr * (0.5 - CORRIDOR_FRACTION / 2))
        cx_end = int(w_bgr * (0.5 + CORRIDOR_FRACTION / 2))
        corridor = depth_metric[:, cx_start:cx_end]
        valid_c = corridor > 0.1
        if not valid_c.any():
            return False, 1.0

        min_depth = float(np.percentile(corridor[valid_c], 10))
        return min_depth < self.OBSTACLE_THRESHOLD_M, min_depth
