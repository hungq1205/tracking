"""
Feature-based pose estimator for V-SLAM.

Geometry (dense depth) and localization (sparse features) are separated:
- Depth Anything provides dense depth → all pixels become 3D geometry
- This module tracks ORB features and estimates camera pose via PnP
  (depth at each keypoint lifts 2D matches to 3D-to-2D correspondences)

When depth is unavailable at a keypoint, falls back to Essential Matrix.

Every PnP-solved frame also gets an independent depth-consistency check
(_triangulate_depth_agreement): the same PnP-inlier 2D correspondences are
re-triangulated via two-view geometry (no dense depth involved), the
resulting 3D points are transformed into the CURRENT frame's own camera
frame, and compared per point against DA3's dense depth at the CURRENT
frame's own pixels. This catches DA3 batches whose depth is internally
coherent-but-wrong (e.g. a warped wall from motion blur or low texture) —
PnP alone can't detect that, since it only fits pose to 2D reprojection
error, not absolute depth correctness. Deliberately checks CURR's own depth,
not prev's, so the verdict lands on the same frame whose back-projection it
gates — checking prev's depth here (prev was already validated the call
before, as its own "curr") would attribute the verdict to the wrong frame,
a one-call lag that was caught and fixed during testing (see git history).
The aggregate metric is the FRACTION of inlier points whose per-point
relative error exceeds _depth_point_err_thresh, not the median — a coherent
warp often only covers part of the frame (e.g. one wall out of the whole
view), so a plain median can hide under the well-behaved majority;
fraction-bad catches a large-but-partial disagreement a median would average
away. Results land in last_depth_trustworthy/last_depth_agree_err
(=frac_bad)/last_depth_agree_n; ScanSession.process_frames_batch uses
last_depth_trustworthy to skip fusing a frame's dense point cloud into the
permanent map when it disagrees.
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
        depth_point_err_thresh: float = 0.30,
        depth_frac_bad_thresh: float = 0.30,
    ):
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._min_matches = min_matches
        self._pnp_min_inliers = pnp_min_inliers
        # Dense-depth consistency check (_triangulate_depth_agreement): a
        # point counts as "bad" if its triangulated-vs-DA3 relative error
        # exceeds depth_point_err_thresh; the frame is flagged untrustworthy
        # if more than depth_frac_bad_thresh of checked points are bad.
        self._depth_point_err_thresh = depth_point_err_thresh
        self._depth_frac_bad_thresh = depth_frac_bad_thresh

        self._world_pose = np.eye(4, dtype=np.float64)  # camera-to-world
        self._prev: Optional[PoseFrame] = None

        # Set by _estimate_relative_pose every call — whether THIS frame's
        # dense DA3 depth agreed with an independent geometric check, and
        # the stats behind that verdict (n=0 means "not enough inliers to
        # evaluate", in which case trustworthy defaults True rather than
        # penalizing frames we simply couldn't check).
        self.last_depth_trustworthy: bool = True
        self.last_depth_agree_err: Optional[float] = None
        self.last_depth_agree_n: int = 0

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

        # Reset per-call — overwritten below only when we actually have
        # enough inliers to evaluate; otherwise stays "trustworthy" so a
        # frame we simply couldn't check is never penalized.
        self.last_depth_trustworthy = True
        self.last_depth_agree_err = None
        self.last_depth_agree_n = 0

        obj_pts = []
        img_pts = []
        prev_pts_2d = []
        curr_depth_pts = []  # DA3 depth AT CURR's own pixels — see docstring
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
                    prev_pts_2d.append([px, py])
                    # Sampled independently of the obj_pts/PnP gate above
                    # (which only requires PREV's depth to be valid) so
                    # curr-depth validity never changes which points feed
                    # PnP/pose estimation — only which are usable for the
                    # post-hoc consistency check below.
                    cu, cvv = pts_curr[i]
                    ci, cj = int(round(cvv)), int(round(cu))
                    curr_d = (
                        float(depth_map[ci, cj])
                        if 0 <= ci < h and 0 <= cj < w else -1.0
                    )
                    curr_depth_pts.append(curr_d if 0.1 < curr_d < 10.0 else np.nan)

        if len(obj_pts) >= self._pnp_min_inliers:
            obj_arr = np.array(obj_pts, dtype=np.float64)
            img_arr = np.array(img_pts, dtype=np.float64)
            prev_arr = np.array(prev_pts_2d, dtype=np.float64)
            curr_depth_arr = np.array(curr_depth_pts, dtype=np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_arr, img_arr, K.astype(np.float64), None,
                iterationsCount=200, reprojectionError=2.0, confidence=0.99,
            )
            if ok and inliers is not None and len(inliers) >= self._pnp_min_inliers:
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = tvec.ravel()

                idx = inliers.ravel()
                frac_bad, n = self._triangulate_depth_agreement(
                    prev_arr[idx], img_arr[idx], curr_depth_arr[idx], T, K.astype(np.float64)
                )
                self.last_depth_agree_err = frac_bad
                self.last_depth_agree_n = n
                self.last_depth_trustworthy = (
                    n < self._pnp_min_inliers or frac_bad <= self._depth_frac_bad_thresh
                )

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

    def _triangulate_depth_agreement(
        self,
        pts_prev_inliers: np.ndarray,
        pts_curr_inliers: np.ndarray,
        da3_depth_curr_inliers: np.ndarray,
        T: np.ndarray,
        K: np.ndarray,
    ) -> tuple[float, int]:
        """
        Independent sanity check on the CURRENT frame's OWN DA3 depth:
        triangulate the PnP-inlier 2D correspondences via two-view geometry
        (using the just-solved prev->curr pose T, camera-independent of any
        dense depth map), transform the resulting points into CURR's own
        camera frame, and compare against DA3's dense depth at CURR's own
        pixels (da3_depth_curr_inliers).

        Deliberately checks CURR's depth, not prev's: prev's depth was
        already validated the call before (when prev was itself "curr"), so
        checking it again here would both duplicate that verdict AND, worse,
        attribute it to the wrong frame — a caller gating "should frame i's
        dense cloud be fused" needs frame i's own verdict, evaluated at the
        same call where frame i is "curr", not lagged one frame behind.

        A batch whose DA3 depth is internally coherent-but-wrong (e.g. a
        warped wall from motion blur or low texture) tends to disagree with
        its own inlier geometry even though PnP still finds a
        locally-consistent pose, since PnP only fits camera pose to 2D
        reprojection error, not to absolute depth.

        Returns (fraction_of_points_bad, n_points_evaluated) — a point is
        "bad" if its relative error exceeds self._depth_point_err_thresh.
        Fraction, not median: a coherent warp often only covers PART of the
        frame (e.g. one wall out of the whole view), so the well-behaved
        majority would pull a median back under any reasonable threshold and
        hide exactly the failure mode this check exists to catch.
        """
        P1 = K @ np.eye(3, 4, dtype=np.float64)
        P2 = K @ T[:3, :]
        pts4d = cv2.triangulatePoints(
            P1, P2,
            pts_prev_inliers.T.astype(np.float64),
            pts_curr_inliers.T.astype(np.float64),
        )
        w_h = pts4d[3]
        valid_w = np.abs(w_h) > 1e-9
        pts3d_prev = np.zeros((3, pts4d.shape[1]))
        pts3d_prev[:, valid_w] = pts4d[:3, valid_w] / w_h[valid_w]
        # T is prev_cam -> curr_cam (see track()'s docstring on rel/T
        # convention) — apply it to move the triangulated point (currently
        # in prev-cam coords) into curr-cam coords.
        pts3d_curr = (T[:3, :3] @ pts3d_prev) + T[:3, 3:4]
        z_curr = pts3d_curr[2]
        valid = valid_w & (z_curr > 0.05) & np.isfinite(da3_depth_curr_inliers)
        if valid.sum() == 0:
            return 0.0, 0
        rel_err = (
            np.abs(z_curr[valid] - da3_depth_curr_inliers[valid])
            / da3_depth_curr_inliers[valid]
        )
        frac_bad = float(np.mean(rel_err > self._depth_point_err_thresh))
        return frac_bad, int(valid.sum())


# ------------------------------------------------------------------ helpers

def _estimate_K(h: int, w: int) -> np.ndarray:
    """Pinhole estimate when no calibration is available."""
    f = max(w, h) * 0.8
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
