"""
ORB-descriptor-match + Essential-Matrix-RANSAC frame novelty gate, plus a
hard variance-of-Laplacian blur-reject threshold — ported from
frame_extractor/extractor.py's OrbNoveltyGate/_sharpness_score/extract_new_
frames() (see that module's docstring for the full algorithm rationale:
mirrors how RTAB-Map's own visual odometry judges novelty — descriptor match
against every previously-accepted frame, not just the last one, so panning
back to an earlier view is correctly recognized as already-seen, followed by
Essential Matrix + RANSAC geometric verification so only RANSAC inliers count
as "already seen").

Duplicated here rather than imported from frame_extractor/ — server/ and
scan_server/ (and frame_extractor/, a separate standalone tool) are
independently deployed processes/environments, the same reason
live_path_planner.py duplicates server/tools/grid_path_planner.py instead of
importing it (see that module's docstring).

Two structural differences from frame_extractor/extractor.py's original:
  - `evaluate()`'s reference match/RANSAC loop is split out into
    `_match_against_references()`, and a new `evaluate_with_keypoints()`
    entry point skips ORB detection entirely — scan_session.py's live
    pipeline already runs ORB detection once per frame via
    FeatureTracker.track() (for IMU+VO/VO pose estimation) and reuses those
    same keypoints/descriptors here rather than paying for a second
    independent detection pass.
  - `decide_accept()` is a new small helper that mirrors
    frame_extractor/extractor.py's extract_new_frames() loop body's tested
    accept/reject boolean algebra exactly (novelty fraction/count + rotation
    guard + blur reject) — that logic lives inline in the CLI tool's loop,
    not inside OrbNoveltyGate itself, so it's copied here rather than
    re-derived.
"""

from typing import Optional

import cv2
import numpy as np

from timing_utils import timed

NEW_FEATURE_COLOR = (0, 255, 0)   # green, RGB — unused here, kept for parity with extractor.py
OLD_FEATURE_COLOR = (255, 0, 0)   # red, RGB


def _gpu_available() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2, "cuda_ORB")
    except Exception:
        return False


def _sharpness_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian — a standard, cheap blur proxy: a sharp
    image has strong high-frequency edge response, a blurry one doesn't."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _estimate_K(h: int, w: int) -> np.ndarray:
    f = max(w, h) * 0.8
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _rotation_deg(R: np.ndarray) -> float:
    """Angle (degrees) of rotation matrix R, via the standard trace formula."""
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


class OrbNoveltyGate:
    """Keeps every accepted frame's own keypoints+descriptors (not a merged
    pool) so each candidate can be geometrically verified — via descriptor
    matching + Essential Matrix RANSAC, same as real visual odometry —
    against its single best-matching accepted frame."""

    def __init__(self, n_features: int = 1500, ratio: float = 0.75,
                 min_raw_matches: int = 20, ransac_threshold_px: float = 3.0,
                 use_gpu: bool = True):
        self._ratio = ratio
        self._min_raw_matches = min_raw_matches
        self._ransac_threshold_px = ransac_threshold_px
        self._references: list = []  # [{"keypoints": [...], "descriptors": Nx32 uint8}, ...]

        self._use_gpu = use_gpu and _gpu_available()
        if self._use_gpu:
            self._orb_gpu = cv2.cuda_ORB.create(nfeatures=n_features)
        else:
            self._orb = cv2.ORB_create(nfeatures=n_features)
        # Matching/RANSAC always run on CPU — findEssentialMat/recoverPose
        # have no CUDA path in stock opencv-python(-contrib); only ORB
        # detection benefits from GPU here.
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def _detect(self, gray: np.ndarray):
        with timed("orb_novelty_detect"):
            if self._use_gpu:
                gpu_gray = cv2.cuda_GpuMat()
                gpu_gray.upload(gray)
                gpu_kp, gpu_desc = self._orb_gpu.detectAndComputeAsync(gpu_gray, None)
                keypoints = self._orb_gpu.convert(gpu_kp)
                descriptors = gpu_desc.download() if gpu_desc is not None else None
            else:
                keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def evaluate(self, gray: np.ndarray, min_new_fraction: Optional[float] = None,
                 min_new_count: Optional[int] = None) -> tuple:
        """Detects ORB features in `gray` then evaluates novelty against the
        accepted-reference set — see evaluate_with_keypoints() for the
        return shape and the early-exit rationale. Use this when no other
        caller has already run ORB detection on this frame; otherwise prefer
        evaluate_with_keypoints() to avoid a redundant detection pass."""
        keypoints, descriptors = self._detect(gray)
        return self._evaluate_from_keypoints(keypoints, descriptors, gray.shape[:2],
                                              min_new_fraction, min_new_count)

    def evaluate_with_keypoints(self, keypoints, descriptors: np.ndarray, image_shape: tuple,
                                 min_new_fraction: Optional[float] = None,
                                 min_new_count: Optional[int] = None) -> tuple:
        """Same as evaluate(), but skips ORB detection — takes
        already-computed `keypoints`/`descriptors` (e.g. from
        FeatureTracker.track()'s own per-frame ORB pass) directly.
        `image_shape` is (h, w), needed to estimate K for the Essential
        Matrix step."""
        return self._evaluate_from_keypoints(keypoints, descriptors, image_shape,
                                              min_new_fraction, min_new_count)

    def _evaluate_from_keypoints(self, keypoints, descriptors, image_shape: tuple,
                                  min_new_fraction: Optional[float],
                                  min_new_count: Optional[int]) -> tuple:
        """Returns (keypoints, descriptors, new_mask, new_count, total_count,
        new_fraction, best_match_inliers, best_rotation_deg, best_translation_px).
        new_mask is a bool array aligned with keypoints/descriptors — True
        where that keypoint was NOT a RANSAC inlier against the
        best-matching accepted frame. best_rotation_deg/best_translation_px
        describe the recovered relative pose (rotation angle in degrees;
        translation as the median pixel displacement of inlier
        correspondences, since monocular translation direction has no
        metric scale) against that same best-matching reference — both are
        0.0 when there is no matching reference (best_match_inliers == 0).

        If `min_new_fraction`/`min_new_count` are given, the reference loop
        below exits early once the running best-match inlier count already
        guarantees rejection (more references can only ADD inliers to the
        best match, never remove them, so once the frame is provably too
        "old" to pass, scanning the rest of the references is wasted work —
        this doesn't change which frames get accepted, only how fast a
        frame that was always going to be rejected gets rejected)."""
        if descriptors is None or len(descriptors) == 0:
            return keypoints or [], descriptors, np.zeros(0, dtype=bool), 0, 0, 0.0, 0, 0.0, 0.0

        if not self._references:
            new_mask = np.ones(len(descriptors), dtype=bool)
            return keypoints, descriptors, new_mask, len(descriptors), len(descriptors), 1.0, 0, 0.0, 0.0

        h, w = image_shape
        K = _estimate_K(h, w)
        old_mask = np.zeros(len(descriptors), dtype=bool)  # union of best reference's RANSAC inliers
        best_inlier_count = 0
        best_rotation_deg = 0.0
        best_translation_px = 0.0
        total = len(descriptors)

        reject_inlier_threshold = None
        if min_new_fraction is not None or min_new_count is not None:
            limits = []
            if min_new_fraction is not None:
                limits.append(total * (1.0 - min_new_fraction))
            if min_new_count is not None:
                limits.append(total - min_new_count)
            reject_inlier_threshold = min(limits)

        # This loop's cost grows with the number of accepted reference
        # frames (every candidate is matched+RANSAC'd against ALL of them)
        # — on a long session with many accepted frames this, not ORB
        # detection, is typically the actual bottleneck.
        with timed("orb_novelty_match_ransac"):
            for ref in self._references:
                if reject_inlier_threshold is not None and best_inlier_count > reject_inlier_threshold:
                    break  # already provably rejected — no need to check remaining references
                matches = self._matcher.knnMatch(descriptors, ref["descriptors"], k=2)
                good = [m[0] for m in matches if len(m) == 2 and m[0].distance < self._ratio * m[1].distance]
                if len(good) < self._min_raw_matches:
                    continue  # not even enough raw candidate matches — clearly not the same view

                pts_cur = np.float32([keypoints[m.queryIdx].pt for m in good])
                pts_ref = np.float32([ref["keypoints"][m.trainIdx].pt for m in good])
                E, mask = cv2.findEssentialMat(
                    pts_cur, pts_ref, K, method=cv2.RANSAC,
                    prob=0.999, threshold=self._ransac_threshold_px,
                )
                if E is None or mask is None:
                    continue
                inlier_mask = mask.ravel().astype(bool)
                inlier_count = int(inlier_mask.sum())
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    frame_mask = np.zeros(len(descriptors), dtype=bool)
                    for m, is_inlier in zip(good, inlier_mask):
                        if is_inlier:
                            frame_mask[m.queryIdx] = True
                    old_mask = frame_mask

                    inlier_pts_cur = pts_cur[inlier_mask]
                    inlier_pts_ref = pts_ref[inlier_mask]
                    best_translation_px = float(np.median(
                        np.linalg.norm(inlier_pts_cur - inlier_pts_ref, axis=1)
                    ))
                    _, R, _t, _ = cv2.recoverPose(E, inlier_pts_cur, inlier_pts_ref, K)
                    best_rotation_deg = _rotation_deg(R)

        new_mask = ~old_mask
        new_count = int(new_mask.sum())
        return (
            keypoints, descriptors, new_mask, new_count, total,
            (new_count / total if total else 0.0), best_inlier_count,
            best_rotation_deg, best_translation_px,
        )

    def accept(self, keypoints, descriptors: np.ndarray) -> None:
        if descriptors is None or len(descriptors) == 0:
            return
        self._references.append({"keypoints": keypoints, "descriptors": descriptors})


def decide_accept(new_fraction: float, new_count: int, best_match_inliers: int,
                   rotation_deg: float, sharpness: float,
                   min_new_fraction: float, min_new_count: int,
                   min_rotation_deg: float, min_sharpness: float) -> bool:
    """Mirrors frame_extractor/extractor.py's extract_new_frames() loop body
    exactly (that file's ~L495-511) — copied rather than re-derived so the
    tested accept/reject boolean algebra stays identical between the offline
    tool and this live-pipeline port. `min_sharpness` of 0 disables blur
    gating entirely (matches extract_new_frames()'s own convention)."""
    pose_ok = best_match_inliers == 0 or rotation_deg >= min_rotation_deg
    candidate = new_fraction >= min_new_fraction and new_count >= min_new_count and pose_ok
    if candidate and min_sharpness > 0 and sharpness < min_sharpness:
        candidate = False  # too blurry — wait for a clearer view of this region
    return candidate
