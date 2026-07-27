"""
New-frame extraction from a video, deciding novelty the way RTAB-Map's own
visual odometry actually decides whether two views show the same physical
points — not by comparing raw ORB keypoints directly (see CLAUDE.md's 3D
Scanning Pipeline / Client-Orchestrated Live Session sections for the
underlying pipeline this borrows from):

  1. ORB detects keypoints+descriptors independently per frame (detections
     are NOT stable frame to frame — corners appear/disappear/shift even
     for a near-static view; comparing raw keypoint sets directly is
     meaningless).
  2. Descriptors are matched against each previously-accepted frame
     (ratio test) — most detections won't have a plausible match at all.
  3. The surviving matches are geometrically verified with an Essential
     Matrix + RANSAC (same as feature_tracker.py/depth.py's
     SparseObstacleDetector elsewhere in this repo) — only matches
     consistent with ONE real camera transform between the two views
     survive as inliers; ambiguous/repetitive-texture false matches get
     rejected here even if their descriptors looked similar.
  4. A frame counts as "new" based on how many of its keypoints were NOT
     RANSAC-inliers against its best-matching previously-accepted frame —
     i.e. not explained by any camera motion from something already seen.

Checked against every accepted frame (keeping the best-matching one), not
just the last, so panning back to an earlier view is correctly recognized
as already-seen and not re-accepted.

Each accepted frame is returned with its ORB keypoints drawn on it — green
for keypoints not explained by its best-matching accepted frame ("new"),
red for RANSAC-verified inlier matches ("old").

RTAB-Map is still used for pose/node-id (attached to each NewFrame as
metadata, e.g. for later 3D reconstruction) but no longer gates acceptance.
Set `rtabmap_addr=None` to skip RTAB-Map entirely and run on ORB novelty
alone. When RTAB-Map is enabled, DA3 depth for accepted frames is computed
in batches (`da3_batch_size`, default 32) via DA3Estimator.estimate_batch's
multi-view joint inference rather than one frame at a time, then each
frame's depth is fed to RTAB-Map sequentially (RTAB-Map's own tracking is
inherently per-frame/chronological — only the depth estimation batches).

Optionally (`frame_tagger`, see tagging.py), every accepted frame also gets
Gemini open-set tagging -> GroundingDINO-tiny detection, batched independently
of the RTAB-Map/DA3 batching above (`tag_batch_size`) since tagging has no
chronological dependency. GroundingDINO's boxes get drawn on the frame too
(yellow, on top of the ORB keypoints), and both the raw tag list and
detections are attached to each NewFrame.
"""
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

_SCAN_SERVER_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "scan_server"))
if _SCAN_SERVER_DIR not in sys.path:
    sys.path.insert(0, _SCAN_SERVER_DIR)

from da3_wrapper import build_estimator  # noqa: E402
from rtabmap_client import RtabmapPoseClient  # noqa: E402

from tagging import FrameTagger, TagDetection, draw_detections  # noqa: E402

NEW_FEATURE_COLOR = (0, 255, 0)   # green, RGB
OLD_FEATURE_COLOR = (255, 0, 0)   # red, RGB


class _Timers:
    """Accumulates wall-clock time per named stage across the whole run —
    printed periodically so a slow run's actual bottleneck (ORB detection?
    the per-reference match+RANSAC loop? DA3? RTAB-Map?) is visible instead
    of guessed at."""

    def __init__(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def track(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - start
            self.counts[name] += 1

    def report(self, prefix: str = "") -> str:
        total = sum(self.totals.values()) or 1e-9
        lines = [f"{prefix}[timing] total={total:.2f}s"]
        for name, t in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            n = self.counts[name]
            lines.append(
                f"{prefix}  {name}: {t:.2f}s ({t / total:.0%}, n={n}, avg={1000 * t / max(n, 1):.1f}ms)"
            )
        return "\n".join(lines)


def _gpu_available() -> bool:
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2, "cuda_ORB")
    except Exception:
        return False


@dataclass
class NewFrame:
    frame_idx: int           # index into the ORIGINAL video (not the sampled sequence)
    timestamp_s: float
    node_id: int              # -1 if RTAB-Map wasn't used or didn't make this a graph node
    new_feature_count: int    # ORB keypoints NOT explained by the best-matching accepted frame
    new_feature_fraction: float
    best_match_inliers: int   # RANSAC inlier count against the best-matching accepted frame (0 if none)
    pose_rotation_deg: float  # recovered rotation vs. best-matching accepted frame (0.0 if no match)
    pose_translation_px: float  # median inlier pixel displacement vs. best-matching accepted frame (0.0 if no match)
    image_rgb: np.ndarray    # HxW x3 uint8, ORB keypoints (green=new/red=old) + tag_detections boxes drawn on it
    sharpness: float = 0.0   # variance of Laplacian (blur metric); 0.0 if min_sharpness gating was disabled
    tags: List[str] = field(default_factory=list)                    # Gemini-proposed tags (empty if frame_tagger unset)
    tag_prompt: str = ""                                              # exact GroundingDINO text prompt built from `tags`
    tag_detections: List[TagDetection] = field(default_factory=list)  # GroundingDINO boxes prompted by `tags`


def draw_keypoints(rgb: np.ndarray, keypoints, new_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    vis = rgb.copy()
    for kp, is_new in zip(keypoints, new_mask):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        color = NEW_FEATURE_COLOR if is_new else OLD_FEATURE_COLOR
        cv2.circle(vis, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return vis


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
                 use_gpu: bool = True, timers: Optional[_Timers] = None):
        self._ratio = ratio
        self._min_raw_matches = min_raw_matches
        self._ransac_threshold_px = ransac_threshold_px
        self._references: list = []  # [{"keypoints": [...], "descriptors": Nx32 uint8}, ...]
        self._timers = timers or _Timers()

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
        with self._timers.track("orb_detect"):
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
        keypoints, descriptors = self._detect(gray)
        if descriptors is None or len(descriptors) == 0:
            return keypoints or [], descriptors, np.zeros(0, dtype=bool), 0, 0, 0.0, 0, 0.0, 0.0

        if not self._references:
            new_mask = np.ones(len(descriptors), dtype=bool)
            return keypoints, descriptors, new_mask, len(descriptors), len(descriptors), 1.0, 0, 0.0, 0.0

        h, w = gray.shape[:2]
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
        # — on a long video with many accepted frames this, not ORB
        # detection, is typically the actual bottleneck. See "match_ransac"
        # in the printed timing report.
        with self._timers.track("match_ransac"):
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


def _flush_rtabmap_batch(pending: list, estimator, client, timers: _Timers) -> List[NewFrame]:
    """Runs DA3 depth for `pending` (a list of dicts) as ONE batched
    estimate_batch() call (falls back to per-frame estimate() if the
    estimator doesn't support batching, e.g. the ONNX backend), then feeds
    each frame's depth to RTAB-Map sequentially (chronological order,
    required by RTAB-Map's own tracking) via a single track_batch() call."""
    if not pending:
        return []
    rgbs = [p["rgb"] for p in pending]
    with timers.track("da3_depth"):
        if hasattr(estimator, "estimate_batch"):
            depth_frames = estimator.estimate_batch(rgbs)
        else:
            depth_frames = [estimator.estimate(rgb) for rgb in rgbs]

    ts_list = [p["ts_ns"] for p in pending]
    with timers.track("rtabmap_track"):
        tracked_list = client.track_batch(rgbs, depth_frames, ts_list)

    out = []
    for p, tracked in zip(pending, tracked_list):
        out.append(NewFrame(
            frame_idx=p["frame_idx"],
            timestamp_s=p["timestamp_s"],
            node_id=tracked.node_id,
            new_feature_count=p["new_feature_count"],
            new_feature_fraction=p["new_feature_fraction"],
            best_match_inliers=p["best_match_inliers"],
            pose_rotation_deg=p["pose_rotation_deg"],
            pose_translation_px=p["pose_translation_px"],
            image_rgb=p["annotated_rgb"],
            sharpness=p["sharpness"],
        ))
    return out


def _flush_tag_batch(pending: list, frame_tagger: FrameTagger, timers: _Timers) -> None:
    """Runs Gemini tagging + GroundingDINO detection as ONE batched call
    across every already-built NewFrame in `pending` (a list of {"rgb",
    "frame_ref"} dicts), then mutates each NewFrame in place (tags,
    tag_detections, and image_rgb — the GroundingDINO boxes get drawn on
    top of the existing ORB-keypoint annotation). Decoupled from
    `_flush_rtabmap_batch`'s own batch size — tagging has no chronological
    dependency (unlike RTAB-Map's tracking) and its own VRAM/speed profile,
    so it accumulates and flushes independently, on every path a frame gets
    accepted through (RTAB-Map-batched or not)."""
    if not pending:
        return
    rgbs = [p["rgb"] for p in pending]
    with timers.track("ram_gdino_tag"):
        tag_results = frame_tagger.tag_and_detect_batch(rgbs)
    for p, result in zip(pending, tag_results):
        nf = p["frame_ref"]
        nf.tags = result.tags
        nf.tag_prompt = result.prompt
        nf.tag_detections = result.detections
        nf.image_rgb = draw_detections(nf.image_rgb, result.detections)


def extract_new_frames(
    video_path: str,
    sample_fps: float = 5.0,
    min_new_fraction: float = 0.95,
    min_new_count: int = 80,
    orb_features: int = 1500,
    min_raw_matches: int = 20,
    ransac_threshold_px: float = 3.0,
    min_rotation_deg: float = 2.0,
    min_sharpness: float = 0.0,
    rtabmap_addr: Optional[str] = "tcp://localhost:5556",
    da3_batch_size: int = 32,
    da3_model: str = "torch",
    da3_onnx_path: Optional[str] = "DA3METRIC-LARGE.onnx",
    da3_model_id: str = "depth-anything/da3-large",
    device: str = "cuda",
    use_gpu_orb: bool = True,
    frame_tagger: Optional[FrameTagger] = None,
    tag_batch_size: int = 8,
    timing_report_every: int = 100,
    progress_cb=None,
) -> List[NewFrame]:
    """
    Reads `video_path`, subsamples to ~`sample_fps`, and keeps a frame only
    if BOTH: its fraction of keypoints NOT RANSAC-verified against its
    best-matching accepted frame is >= `min_new_fraction`, AND the raw count
    of such keypoints is >= `min_new_count` (the count guard avoids
    accepting a low-texture frame purely because a handful of noisy
    features didn't happen to match). Raise either threshold if a small pan
    is still producing too many frames; lower them if real new views are
    getting skipped. `min_raw_matches` is the minimum ratio-test descriptor
    matches required before even attempting RANSAC against a reference
    frame (below this, it's clearly not the same view — skip straight to
    "no match"); `ransac_threshold_px` is the Essential Matrix RANSAC
    reprojection-error tolerance (looser = more forgiving of motion
    blur/rolling shutter, but lets more false matches through as inliers).

    A frame that otherwise passes the ORB novelty thresholds is still
    rejected unless the recovered relative pose against its best-matching
    reference shows at least `min_rotation_deg` of rotation — this guards
    against small jitter/noise producing enough unmatched keypoints to look
    "new" when the camera hasn't actually panned/rotated. Only applies when a
    best-matching reference with a recoverable pose exists (best_match_inliers
    > 0); a frame with zero reference matches (nothing to compare pose
    against) is accepted on ORB novelty alone. (Translation isn't gated on —
    monocular translation has no metric scale, so a fixed pixel threshold is
    an unreliable proxy; `pose_translation_px` is still reported on each
    NewFrame for reference.)

    `min_sharpness`: 0 (default) disables blur gating entirely. Above 0, a
    frame that otherwise passes every other check is still rejected — not
    accepted as a reference, not returned — if its variance-of-Laplacian
    blur score is below this threshold (`_sharpness_score`); the region it
    would have covered stays "unexplored" until a sharper view of it comes
    along. Laplacian variance scales with resolution/content, so this needs
    tuning per video rather than one universal default.

    `rtabmap_addr`: if set, accepted frames get pose/node_id metadata from
    RTAB-Map (informational only — no longer used to decide acceptance),
    with DA3 depth computed `da3_batch_size` frames at a time. Pass None to
    skip RTAB-Map/DA3 entirely and run on ORB novelty alone (much faster, no
    depth model or RTAB-Map service needed).

    `use_gpu_orb`: if a CUDA-enabled OpenCV build is available
    (`cv2.cuda.getCudaEnabledDeviceCount() > 0` and `cv2.cuda_ORB` exists),
    runs ORB keypoint detection on the GPU instead of CPU — silently falls
    back to CPU otherwise. Only ORB detection is GPU-accelerated; descriptor
    matching + Essential Matrix RANSAC have no CUDA path in stock
    opencv-python(-contrib) and always run on CPU. On a long video with many
    accepted reference frames, that CPU match+RANSAC loop (cost grows with
    reference count) is typically the actual bottleneck, not ORB detection —
    see the printed timing report (`timing_report_every`, 0 to disable
    periodic reports; a final report always prints at the end) to see which
    stage dominates for your video.

    `frame_tagger`: an already-constructed `tagging.FrameTagger` (Gemini
    open-set tagging -> GroundingDINO-tiny detection, see tagging.py's module
    docstring), or None to skip tagging entirely. Passed in already-built
    rather than constructed from flat args here (unlike `estimator`/`client`
    above) because loading GroundingDINO tiny still costs a few seconds —
    callers (e.g. app.py) should build one FrameTagger once and reuse it
    across every extract_new_frames() call instead of paying that cost per
    video. `tag_batch_size` batches accepted frames for the Gemini call /
    GroundingDINO forward pass — independent of `da3_batch_size`/RTAB-Map's
    own batching, since tagging has no chronological dependency and a
    different VRAM profile.

    `progress_cb(done, total)`, if given, is called after each sampled frame.
    """
    timers = _Timers()
    estimator = None
    client = None
    if rtabmap_addr:
        if da3_model == "onnx":
            if not da3_onnx_path:
                raise ValueError("da3_onnx_path is required when da3_model='onnx'")
            estimator = build_estimator(onnx_path=da3_onnx_path, device=device)
        else:
            estimator = build_estimator(model_id=da3_model_id, device=device)
        client = RtabmapPoseClient(rtabmap_addr)
        client.reset()

    gate = OrbNoveltyGate(
        n_features=orb_features,
        min_raw_matches=min_raw_matches,
        ransac_threshold_px=ransac_threshold_px,
        use_gpu=use_gpu_orb,
        timers=timers,
    )
    print(f"[frame_extractor] GPU ORB detection: {'enabled' if gate._use_gpu else 'disabled (CPU fallback)'}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(native_fps / sample_fps)) if sample_fps > 0 else 1
    sampled_indices = list(range(0, max(total_frames, 1), step))

    results: List[NewFrame] = []
    pending: list = []  # only used when RTAB-Map is enabled, batched by da3_batch_size
    tag_pending: list = []  # only used when frame_tagger is set, batched by tag_batch_size
    sampled_set = set(sampled_indices)
    try:
        done = 0
        frame_idx = 0
        # Sequential grab()+retrieve() instead of cap.set(CAP_PROP_POS_FRAMES)
        # per sampled frame: a random seek forces most codecs to decode
        # forward from the nearest earlier keyframe every single call, which
        # measured as the single biggest cost in this pipeline (~75ms/frame,
        # more than match_ransac). grab() just advances the demuxer/decoder
        # without doing the (expensive) full frame decode+color-convert that
        # retrieve() does, so skipped frames are cheap and only sampled ones
        # pay the full decode cost.
        while True:
            with timers.track("frame_read"):
                ok = cap.grab()
            if not ok:
                break
            if frame_idx not in sampled_set:
                frame_idx += 1
                continue
            with timers.track("frame_read"):
                ok, frame_bgr = cap.retrieve()
            if not ok:
                break
            with timers.track("cvt_color"):
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            (
                keypoints, descriptors, new_mask, new_count, _total_count,
                new_fraction, best_inliers, rotation_deg, translation_px,
            ) = gate.evaluate(gray, min_new_fraction=min_new_fraction, min_new_count=min_new_count)

            pose_ok = best_inliers == 0 or rotation_deg >= min_rotation_deg

            candidate = (descriptors is not None and new_fraction >= min_new_fraction
                         and new_count >= min_new_count and pose_ok)

            sharpness = 0.0
            if candidate and min_sharpness > 0:
                with timers.track("sharpness"):
                    sharpness = _sharpness_score(gray)
                if sharpness < min_sharpness:
                    candidate = False  # too blurry — wait for a clearer view of this region

            if candidate:
                gate.accept(keypoints, descriptors)
                with timers.track("annotate"):
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    annotated = draw_keypoints(rgb, keypoints, new_mask)

                if client is not None:
                    pending.append({
                        "frame_idx": frame_idx,
                        "timestamp_s": frame_idx / native_fps,
                        "ts_ns": int((frame_idx / native_fps) * 1e9),
                        "rgb": rgb,
                        "annotated_rgb": annotated,
                        "new_feature_count": new_count,
                        "new_feature_fraction": new_fraction,
                        "best_match_inliers": best_inliers,
                        "pose_rotation_deg": rotation_deg,
                        "pose_translation_px": translation_px,
                        "sharpness": sharpness,
                    })
                    if len(pending) >= da3_batch_size:
                        flushed_rgbs = [p["rgb"] for p in pending]
                        flushed = _flush_rtabmap_batch(pending, estimator, client, timers)
                        results.extend(flushed)
                        if frame_tagger is not None:
                            tag_pending.extend(
                                {"rgb": rgb2, "frame_ref": nf2}
                                for rgb2, nf2 in zip(flushed_rgbs, flushed)
                            )
                        pending = []
                else:
                    nf = NewFrame(
                        frame_idx=frame_idx,
                        timestamp_s=frame_idx / native_fps,
                        node_id=-1,
                        new_feature_count=new_count,
                        new_feature_fraction=new_fraction,
                        best_match_inliers=best_inliers,
                        pose_rotation_deg=rotation_deg,
                        pose_translation_px=translation_px,
                        image_rgb=annotated,
                        sharpness=sharpness,
                    )
                    results.append(nf)
                    if frame_tagger is not None:
                        tag_pending.append({"rgb": rgb, "frame_ref": nf})

                if frame_tagger is not None and len(tag_pending) >= tag_batch_size:
                    _flush_tag_batch(tag_pending, frame_tagger, timers)
                    tag_pending = []

            done += 1
            frame_idx += 1

            if progress_cb is not None:
                progress_cb(done, len(sampled_indices))

            if timing_report_every and done % timing_report_every == 0:
                print(timers.report(prefix=f"[{done}/{len(sampled_indices)}] "))

            if done >= len(sampled_indices):
                break

        if pending:
            flushed_rgbs = [p["rgb"] for p in pending]
            flushed = _flush_rtabmap_batch(pending, estimator, client, timers)
            results.extend(flushed)
            if frame_tagger is not None:
                tag_pending.extend(
                    {"rgb": rgb2, "frame_ref": nf2} for rgb2, nf2 in zip(flushed_rgbs, flushed)
                )
        if tag_pending:
            _flush_tag_batch(tag_pending, frame_tagger, timers)
    finally:
        cap.release()

    print(timers.report(prefix="[final] "))

    return results
