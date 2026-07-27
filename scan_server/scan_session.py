"""
ScanSession — per-location scanning state.

Pipeline (called per dataset segment):
  1. DA3 batch depth estimation  → metric depth + intrinsics per frame
  2. FeatureTracker VO           → incremental camera pose per frame
  3. Pose graph keyframe mgmt    → every KEYFRAME_INTERVAL frames
     • loop closure detection via ORB matching + PnP
     • add odometry / loop edges
  4. Pose graph optimization     → scipy LM on all keyframe poses
  5. Dense back-projection       → Open3D colored point cloud (voxel-fused)
  6. Occupancy map update        → Bresenham ray casting on X-Z grid
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import numpy as np
import open3d as o3d
import pandas as pd

from da3_wrapper import BaseDepthEstimator
from feature_tracker import FeatureTracker
from map_exporter import export_map
from occupancy_map import OccupancyMap
from orb_novelty_gate import OrbNoveltyGate, _sharpness_score, decide_accept
from pose_graph import PoseGraph
from semantic_mapper import Landmark, SemanticMapper
from timing_utils import timed
from zone_labeler import Zone, ZoneLabeler

if TYPE_CHECKING:
    # Optional dependency (pyzmq) — only imported for type checking so a
    # server without RTAB-Map configured never needs pyzmq installed.
    from rtabmap_client import RtabmapPoseClient

_MAPS_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "data", "maps"))

KEYFRAME_INTERVAL = 10
MIN_KF_VALID_PTS = 10
VOXEL_SIZE = 0.02

# "Completely lost, no sign of recovery" thresholds — once RTAB-Map reports
# zero pose for this many consecutive SECONDS (not frames — frame cadence
# is client-controlled, so a time threshold stays meaningful regardless of
# it), the session is reset (RTAB-Map + local grid wiped, same as a fresh
# stream) rather than continuing to reuse a stale last-known pose
# indefinitely. Two separate thresholds, requested directly by the user —
# WALKING (SessionMode.WALKING, no destination) stays fast: its local grid
# is short-lived and cheap to rebuild, so a quick reset-and-restart beats
# limping along on stale state while F2M tries to recover. GUIDING
# (SessionMode.GUIDING, a real destination + route in progress) gets a much
# more forgiving threshold instead — resetting a route mid-navigation over
# a brief tracking hiccup is far more disruptive than for ambient walking,
# so it's worth waiting longer for F2M to relocalize on its own before
# giving up and starting over. See process_frames_batch's `pure_walking`/
# `walking_lite` params and the RTAB-Map branch below for where each is
# actually applied (`walking_lite` is true for BOTH modes; `pure_walking`
# distinguishes WALKING from GUIDING within that).
PURE_WALKING_LOST_RESET_S = 1.25
GUIDING_LOST_RESET_S = 2.0

# Statistical Outlier Removal (Open3D) — run after every voxel downsample to
# strip stray/noisy points whose average neighbor distance is far outside the
# cloud's typical density (see ScanSession.process_frames_batch, Step 3).
# Defaults only — GUI-adjustable per scan via the Outlier Removal accordion.
SOR_NB_NEIGHBORS = 10
SOR_STD_RATIO = 2.25

# Voxel size for the point-cloud → voxelization → occupancy-map pipeline
# (voxelize_cloud, below) — shared by scan_gui.py's Voxelization tab display
# and ScanSession.process_frames_batch's Occupancy Map step, so both stages
# always reflect the exact same voxels. Default only — GUI-adjustable via the
# Voxelization tab's voxel size slider.
DEFAULT_VOXEL_SIZE = 0.05
MAX_VOXELS = 20_000  # coarsen voxel_size rather than exceed this many voxels

# MAX_VOXELS's auto-coarsening exists to bound a ONE-SHOT whole-cloud
# voxelize_cloud() call's render cost (a GLB mesh with too many boxes).
# The incremental per-batch/per-node occupancy feed (process_frames_batch's
# Step 4, _rtabmap_process_nodes) must NEVER let that kick in: each batch's
# own point count varies independently of total scan size, so if even one
# batch happened to exceed MAX_VOXELS at the configured occupancy_voxel_size,
# voxelize_cloud() would silently return a DIFFERENT (coarser) actual vsize
# for just that call — and since _merge_voxels() resets its whole
# accumulator on any vsize change (a real voxel-size change genuinely isn't
# mergeable with the old grid), that single oversized batch would silently
# wipe every previously-accumulated voxel, corrupting both the Occupancy Map
# feed's resolution and the Voxelization display (observed as holes/a
# disconnected fragment — whatever survived after the last accidental
# reset). OccupancyMap.update() already subsamples internally
# (MAX_CLOUD_SAMPLE) if a single call gets too many points, so there's no
# cost reason to coarsen here either.
_NO_COARSEN_MAX_VOXELS = 10 ** 9

# Fixed voxel-grid anchor used by every voxelize_cloud() call (see below) —
# Open3D's create_from_point_cloud() derives its grid origin from whichever
# point subset is passed in, so voxelizing "this batch's new points" and
# voxelizing "the whole accumulated cloud" land on slightly different bin
# boundaries (verified empirically: origins differ by ~1e-5 for two point
# sets covering the same physical region — enough that a batch's voxel
# centers are NOT a subset of the whole cloud's voxel centers). Anchoring
# every call to the same fixed min/max bound instead makes any subset's
# voxelization an exact subset of any other's at the same voxel_size — a
# real precondition for ScanSession's single incremental voxel accumulator
# (see _merge_voxels) to be correct. Range is generous for any realistic
# indoor scan; Open3D's VoxelGrid only materializes occupied voxels, so a
# larger-than-needed bound costs nothing extra.
_VOXEL_GRID_MIN_BOUND = np.array([-1000.0, -1000.0, -1000.0])
_VOXEL_GRID_MAX_BOUND = np.array([1000.0, 1000.0, 1000.0])

# ── GPU acceleration (Open3D tensor API) ────────────────────────────────────
# Open3D's legacy o3d.geometry.PointCloud API (used throughout this file) is
# CPU-only; o3d.t.geometry.PointCloud (tensor-based) supports CUDA for
# voxel_down_sample/remove_statistical_outliers. Benchmarked against
# realistic depth-camera-shaped point distributions (clustered surface
# patches, not uniform noise — uniform noise is an adversarial worst case
# for the GPU's KNN-based outlier search and makes GPU look artificially
# slow): voxel_down_sample is consistently faster on GPU once the cloud is
# large (3x+ at ~500k points), but the legacy<->tensor round-trip transfer
# has fixed overhead (~100-200ms) that isn't worth paying for small clouds —
# below _GPU_ACCEL_MIN_POINTS, plain CPU is as fast or faster once that
# transfer cost is counted. Every call through _voxel_down_sample_accel/
# _remove_statistical_outlier_accel logs which path it took and how long via
# timing_utils.timed, so this threshold's accuracy against real data (not
# just this benchmark) is always visible in the console, not just assumed.
#
# Known, verified quirk: the GPU path is NOT run-to-run deterministic —
# calling voxel_down_sample twice on the IDENTICAL input can yield the same
# point count but different exact positions each time (confirmed directly;
# almost certainly non-associative floating-point reduction order in
# parallel voxel-centroid averaging). Harmless for this pipeline's purposes
# (sub-voxel jitter doesn't change Live Points/Voxelization display or
# Occupancy Map classification, given the Bayesian design's own noise
# tolerance) but means point-for-point equality is NOT a valid correctness
# check across two separate GPU calls on the same data, even ignoring the
# separate CPU-vs-GPU numerical difference noted below — count/bbox-level
# comparison is the right invariant to test instead.
_CUDA_AVAILABLE = False
try:
    _CUDA_AVAILABLE = o3d.core.cuda.is_available() and o3d.core.cuda.device_count() > 0
except Exception:
    _CUDA_AVAILABLE = False
print(
    f"[ScanSession] Open3D CUDA acceleration "
    f"{'available — used for large point-cloud voxel/outlier ops' if _CUDA_AVAILABLE else 'NOT available — all point-cloud ops run on CPU'}."
)

_GPU_ACCEL_MIN_POINTS = 50_000
_CUDA_DEVICE = o3d.core.Device("CUDA:0") if _CUDA_AVAILABLE else None


def _voxel_down_sample_accel(cloud: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """Voxel-downsamples on GPU when CUDA is available and the cloud is large
    enough for that to actually pay off (see module-level GPU acceleration
    note) — otherwise identical legacy CPU call. Numerically equivalent
    either way, just routed differently."""
    if not _CUDA_AVAILABLE or len(cloud.points) < _GPU_ACCEL_MIN_POINTS:
        with timed(f"voxel_down_sample CPU ({len(cloud.points)} pts)"):
            return cloud.voxel_down_sample(voxel_size)
    with timed(f"voxel_down_sample GPU ({len(cloud.points)} pts)"):
        tcloud = o3d.t.geometry.PointCloud.from_legacy(cloud, device=_CUDA_DEVICE)
        tcloud = tcloud.voxel_down_sample(voxel_size)
        return tcloud.to_legacy()


def _remove_statistical_outlier_accel_masked(
    cloud: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Like _remove_statistical_outlier_accel, but also returns a boolean
    keep-mask (True = survived) aligned to the INPUT cloud's own point order
    — lets a caller that concatenated several sources into one cloud before
    calling this (see _rtabmap_process_nodes' batched cleanup) split the
    result back into per-source buckets afterward. Legacy SOR returns a
    sorted index list; the tensor/GPU API returns a boolean mask directly —
    normalized to a mask either way so callers don't care which path ran."""
    n = len(cloud.points)
    if not _CUDA_AVAILABLE or n < _GPU_ACCEL_MIN_POINTS:
        with timed(f"remove_statistical_outlier CPU ({n} pts)"):
            cleaned, ind = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            mask = np.zeros(n, dtype=bool)
            mask[ind] = True
            return cleaned, mask
    with timed(f"remove_statistical_outlier GPU ({n} pts)"):
        tcloud = o3d.t.geometry.PointCloud.from_legacy(cloud, device=_CUDA_DEVICE)
        tcleaned, tmask = tcloud.remove_statistical_outliers(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return tcleaned.to_legacy(), tmask.cpu().numpy()


def _remove_statistical_outlier_accel(
    cloud: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float
) -> o3d.geometry.PointCloud:
    """Statistical outlier removal on GPU when CUDA is available and the
    cloud is large enough (see _voxel_down_sample_accel) — otherwise
    identical legacy CPU call."""
    cleaned, _ = _remove_statistical_outlier_accel_masked(cloud, nb_neighbors, std_ratio)
    return cleaned


# ── IMU integrator ─────────────────────────────────────────────────────────────

# Android's SensorManager reports accel/gyro in a fixed body frame tied to the
# phone's NATIVE (portrait) orientation — it does NOT rotate with how the
# phone is physically held (see ImuSensor.kt). If a dataset was recorded with
# the phone actually held in landscape, the saved images (CameraManager.kt
# rotates each frame to match imu_orientation before writing it) end up in a
# different frame than the raw IMU samples. These matrices re-express every
# accel/gyro sample in the same target frame the images were rotated into —
# same convention as Android's own SensorManager.remapCoordinateSystem for
# Surface.ROTATION_90 / ROTATION_270. Rotation only (no translation concept
# applies to accel/gyro vectors); Z (out-of-screen) is unaffected by an
# in-plane 90° turn.
IMU_ORIENTATIONS = ("portrait", "landscape-left", "landscape-right")

_IMU_ORIENTATION_MATS = {
    "portrait": np.eye(3, dtype=np.float64),
    # top of phone rotated to point left: x'=-y, y'=x
    "landscape-left": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    # top of phone rotated to point right: x'=y, y'=-x
    "landscape-right": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


class ImuIntegrator:
    """
    Dead-reckoning pose integrator from Android IMU CSV.
    CSV format (header row): timestamp_ns,ax,ay,az,gx,gy,gz
    Integrates gyroscope for orientation and double-integrates debiased
    acceleration for translation.  Used to pre-compute all camera poses for
    a segment before depth estimation, so point-cloud back-projection uses
    globally consistent poses rather than per-batch VO estimates.

    `orientation` (one of IMU_ORIENTATIONS) rotates every accel/gyro sample
    into the frame matching how the phone was actually held during this
    recording — see IMU_ORIENTATIONS' comment above. Applied once, in place,
    before integration; "portrait" (the default) is a no-op.
    """

    def __init__(self, csv_path: str, orientation: str = "portrait") -> None:
        df = pd.read_csv(csv_path).rename(columns={"timestamp_ns": "ts"})
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        self._ts: np.ndarray = df["ts"].to_numpy(dtype=np.float64)
        accel = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
        gyro = df[["gx", "gy", "gz"]].to_numpy(dtype=np.float64)
        R = _IMU_ORIENTATION_MATS.get(orientation, _IMU_ORIENTATION_MATS["portrait"])
        self._accel: np.ndarray = accel @ R.T
        self._gyro: np.ndarray = gyro @ R.T
        self._poses: np.ndarray = self._integrate()
        duration = (self._ts[-1] - self._ts[0]) * 1e-9
        print(f"[ImuIntegrator] {len(self._ts):,} samples  {duration:.1f} s  "
              f"start={self._ts[0]:.0f} ns  orientation={orientation}")

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def start_ns(self) -> float:
        return float(self._ts[0])

    @property
    def end_ns(self) -> float:
        return float(self._ts[-1])

    def pose_at(self, timestamp_ns: float) -> np.ndarray:
        """Interpolated 4×4 c2w pose at an arbitrary nanosecond timestamp."""
        idx = int(np.searchsorted(self._ts, timestamp_ns))
        if idx <= 0:
            return self._poses[0].copy()
        if idx >= len(self._poses):
            return self._poses[-1].copy()

        t0, t1 = self._ts[idx - 1], self._ts[idx]
        alpha = float((timestamp_ns - t0) / (t1 - t0)) if t1 != t0 else 0.0

        p0, p1 = self._poses[idx - 1], self._poses[idx]
        t_lerp = p0[:3, 3] * (1.0 - alpha) + p1[:3, 3] * alpha

        R_lerp = p0[:3, :3] * (1.0 - alpha) + p1[:3, :3] * alpha
        U, _, Vt = np.linalg.svd(R_lerp)
        R_lerp = U @ Vt  # re-orthogonalise

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_lerp
        pose[:3, 3] = t_lerp
        return pose

    # ── private ───────────────────────────────────────────────────────────────

    def _integrate(self) -> np.ndarray:
        n = len(self._ts)
        poses = np.empty((n, 4, 4), dtype=np.float64)
        poses[0] = np.eye(4)

        # Gravity estimate from first stationary period (~first 5 % of data, max 200 samples)
        stat_n = min(max(5, n // 20), 200)
        g_body = self._accel[:stat_n].mean(axis=0)
        g_norm = np.linalg.norm(g_body)
        if g_norm > 0.5:
            g_body = g_body / g_norm * 9.81  # normalise to standard gravity

        R = np.eye(3, dtype=np.float64)
        vel = np.zeros(3, dtype=np.float64)
        pos = np.zeros(3, dtype=np.float64)

        for i in range(1, n):
            dt = (self._ts[i] - self._ts[i - 1]) * 1e-9
            if dt <= 0.0 or dt > 0.5:          # skip bad / gap timestamps
                poses[i] = poses[i - 1]
                continue

            # Rotation via Rodrigues (body-frame gyro → incremental rotation)
            omega = self._gyro[i - 1]
            angle = np.linalg.norm(omega) * dt
            if angle > 1e-8:
                ax = omega / np.linalg.norm(omega)
                K = np.array([[0, -ax[2], ax[1]],
                               [ax[2], 0, -ax[0]],
                               [-ax[1], ax[0], 0]], dtype=np.float64)
                dR = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
                R = R @ dR

            # Translation is intentionally NOT integrated here.
            # Double-integrating accelerometer gives O(t²) drift that dominates
            # any actual motion within seconds (walking step peaks alone are
            # 10–20 m/s²).  Translation is supplied by FeatureTracker VO which
            # uses metric depth (DA3) for scale — see process_frames_batch fusion.
            poses[i, :3, :3] = R
            poses[i, :3, 3] = 0.0          # rotation-only pose
            poses[i, 3] = [0.0, 0.0, 0.0, 1.0]

        return poses


class IncrementalImuIntegrator:
    """
    Streaming counterpart to ImuIntegrator: integrates one IMU sample at a
    time instead of a whole pre-recorded CSV. Used by StreamingScanSession
    (stream_session.py) so a genuinely live IMU source — where the total
    session length isn't known up front — can still produce dead-reckoning
    poses. `ImuIntegrator` itself is untouched; it stays the batch-mode
    loader for scan_gui.py's Scan tab.

    Gravity/bias init: ImuIntegrator uses "first 5% of data, max 200 samples"
    (_integrate, above) because it knows the whole file's length in advance.
    A stream doesn't, so this uses a fixed time window instead — the first
    `stationary_s` seconds of samples — after which the running gravity
    estimate is frozen and every subsequent sample is integrated immediately.
    Same orientation remap + rotation-only Rodrigues integration as
    ImuIntegrator._integrate(), just called per-sample instead of in one pass.
    """

    def __init__(self, orientation: str = "portrait", stationary_s: float = 1.0) -> None:
        self._R = _IMU_ORIENTATION_MATS.get(orientation, _IMU_ORIENTATION_MATS["portrait"])
        self._stationary_ns = stationary_s * 1e9

        self._init_accel: List[np.ndarray] = []  # buffered during the stationary window
        self._init_done = False

        self._ts: List[float] = []
        self._poses: List[np.ndarray] = []

        self._R_cur = np.eye(3, dtype=np.float64)
        self._last_ts: Optional[float] = None
        self._last_gyro: Optional[np.ndarray] = None
        self._t0: Optional[float] = None

    @property
    def start_ns(self) -> Optional[float]:
        return self._ts[0] if self._ts else None

    @property
    def end_ns(self) -> Optional[float]:
        return self._ts[-1] if self._ts else None

    def push(self, ts_ns: float, ax: float, ay: float, az: float,
              gx: float, gy: float, gz: float) -> None:
        """Feed one raw IMU sample (Android's native-portrait body frame,
        same convention as ImuIntegrator/imu.csv). Rotates it into
        `orientation`'s frame, then either buffers it for gravity init or
        integrates it immediately onto the running pose."""
        accel = self._R @ np.array([ax, ay, az], dtype=np.float64)
        gyro = self._R @ np.array([gx, gy, gz], dtype=np.float64)

        if self._t0 is None:
            self._t0 = ts_ns

        if not self._init_done:
            self._init_accel.append(accel)
            if ts_ns - self._t0 < self._stationary_ns:
                # Still in the stationary window — no pose yet for this sample.
                self._last_ts = ts_ns
                self._last_gyro = gyro
                if not self._ts:
                    self._ts.append(ts_ns)
                    self._poses.append(np.eye(4, dtype=np.float64))
                return
            self._init_done = True
            # Gravity estimate frozen from the stationary window — mirrors
            # ImuIntegrator._integrate()'s normalization (unused directly by
            # rotation-only integration, kept for parity/future translation use).
            g_body = np.mean(self._init_accel, axis=0)
            g_norm = np.linalg.norm(g_body)
            if g_norm > 0.5:
                g_body = g_body / g_norm * 9.81
            self._init_accel = []

        if self._last_ts is None:
            self._last_ts = ts_ns
            self._last_gyro = gyro
            self._ts.append(ts_ns)
            self._poses.append(np.eye(4, dtype=np.float64))
            return

        dt = (ts_ns - self._last_ts) * 1e-9
        if 0.0 < dt <= 0.5:
            omega = self._last_gyro
            angle = np.linalg.norm(omega) * dt
            if angle > 1e-8:
                axis = omega / np.linalg.norm(omega)
                K = np.array([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]], dtype=np.float64)
                dR = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
                self._R_cur = self._R_cur @ dR

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self._R_cur
        self._ts.append(ts_ns)
        self._poses.append(pose)
        self._last_ts = ts_ns
        self._last_gyro = gyro

    def pose_at(self, timestamp_ns: float) -> np.ndarray:
        """Interpolated 4×4 c2w pose — same lerp+re-orthogonalize behavior as
        ImuIntegrator.pose_at(), against whatever samples have arrived so far."""
        if not self._ts:
            return np.eye(4, dtype=np.float64)
        ts_arr = np.asarray(self._ts, dtype=np.float64)
        idx = int(np.searchsorted(ts_arr, timestamp_ns))
        if idx <= 0:
            return self._poses[0].copy()
        if idx >= len(self._poses):
            return self._poses[-1].copy()

        t0, t1 = ts_arr[idx - 1], ts_arr[idx]
        alpha = float((timestamp_ns - t0) / (t1 - t0)) if t1 != t0 else 0.0

        p0, p1 = self._poses[idx - 1], self._poses[idx]
        t_lerp = p0[:3, 3] * (1.0 - alpha) + p1[:3, 3] * alpha

        R_lerp = p0[:3, :3] * (1.0 - alpha) + p1[:3, :3] * alpha
        U, _, Vt = np.linalg.svd(R_lerp)
        R_lerp = U @ Vt

        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_lerp
        pose[:3, 3] = t_lerp
        return pose


# ── Module-level helpers ───────────────────────────────────────────────────────


def _estimate_K(h: int, w: int) -> np.ndarray:
    f = max(w, h) * 0.8
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


def _remap_poses(poses: List[np.ndarray], perm: Optional[np.ndarray]) -> List[np.ndarray]:
    """
    Apply a sensor-axis permutation/reflection P (3×3 signed permutation,
    P^-1 == P.T) to each pose's ROTATION only: P @ R @ P.T — matching
    scan_gui.py's _perm_matrix / the Frame Explorer's legacy post-hoc remap
    convention exactly, just applied globally before processing instead of
    only in that preview.

    Translation is deliberately left untouched: this setting exists to fix a
    mislabeled *sensor* axis convention (e.g. depth/IMU rig reporting "up" on
    the wrong local axis), which is a per-frame orientation problem, not a
    claim that the VO/IMU-estimated world-space trajectory shape is wrong.
    Permuting translation too was tried and made tracking visibly worse (a
    wobble in the previously-flat axis) — it reshapes an already-correct
    trajectory using an operation that was only ever meant to relabel local
    orientation, so don't reintroduce it.
    """
    if perm is None:
        return poses
    out = []
    for p in poses:
        q = p.copy()
        q[:3, :3] = perm @ p[:3, :3] @ perm.T
        out.append(q)
    return out


def voxelize_cloud(
    cloud: o3d.geometry.PointCloud, voxel_size: float = DEFAULT_VOXEL_SIZE,
    max_voxels: int = MAX_VOXELS,
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    """
    Voxelize a point cloud (Open3D VoxelGrid) and return (centers, colors,
    actual_voxel_size) — one representative point (+ averaged color) per
    occupied voxel, at the voxel's grid-cell center (not the mean of its
    contained points).

    Shared by scan_gui.py's Voxelization tab (_voxels_to_glb, display) and
    ScanSession.process_frames_batch's Occupancy Map step, so "point cloud →
    voxelization → occupancy map" is one consistent pipeline: both stages see
    the exact same voxels for a given voxel_size, not two independently
    computed approximations. Uses a fixed grid anchor (_VOXEL_GRID_MIN_BOUND/
    _MAX_BOUND) rather than Open3D's default per-call bounding-box origin, so
    voxelizing any subset of a cloud's points is guaranteed to produce voxel
    centers that are an exact subset of voxelizing the whole cloud at the
    same voxel_size — required for ScanSession._merge_voxels' incremental
    accumulation to be correct.

    Coarsens voxel_size (cube-root scaling) rather than truncate when the
    occupied-voxel count would exceed max_voxels — keeps the whole cloud
    represented, just chunkier. actual_voxel_size reflects this — callers
    that render voxels as boxes (e.g. _voxels_to_glb) need it, not the
    originally-requested voxel_size, to size boxes correctly.

    Returns (centers, None, voxel_size) with an empty centers array if cloud
    is empty.

    No GPU path: Open3D has no tensor/CUDA equivalent of VoxelGrid (its only
    CUDA voxel structure, VoxelBlockGrid, is a TSDF-style volumetric grid for
    3D reconstruction, not a drop-in replacement for "one point per occupied
    voxel"), so this stays CPU-only — timed like everything else so that's
    visible rather than assumed. The grid_index -> world-space center
    conversion IS vectorized (numpy arithmetic against the voxel grid's own
    origin/voxel_size) instead of calling get_voxel_center_coordinate() in a
    per-voxel Python loop, which matters once voxel counts climb into the
    thousands.
    """
    if len(cloud.points) == 0:
        return np.zeros((0, 3), dtype=np.float32), None, float(voxel_size)

    with timed(f"voxelize_cloud ({len(cloud.points)} pts, voxel_size={voxel_size:.3f})"):
        vsize = max(float(voxel_size), 1e-3)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            cloud, vsize, _VOXEL_GRID_MIN_BOUND, _VOXEL_GRID_MAX_BOUND
        )
        voxels = voxel_grid.get_voxels()
        if len(voxels) > max_voxels:
            scale = (len(voxels) / max_voxels) ** (1.0 / 3.0)
            vsize *= scale
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
                cloud, vsize, _VOXEL_GRID_MIN_BOUND, _VOXEL_GRID_MAX_BOUND
            )
            voxels = voxel_grid.get_voxels()

        if not voxels:
            return np.zeros((0, 3), dtype=np.float32), None, vsize

        grid_indices = np.array([v.grid_index for v in voxels], dtype=np.float64)
        centers = (voxel_grid.origin + (grid_indices + 0.5) * voxel_grid.voxel_size).astype(np.float32)
        colors = np.array([v.color for v in voxels], dtype=np.float32)
        return centers, colors, vsize


def _voxel_majority_flags(
    points: np.ndarray, flags: np.ndarray, centers: np.ndarray, vsize: float,
) -> np.ndarray:
    """
    Aggregates a per-point boolean (RTAB-Map's own is_ground, see
    rtabmap_server.cc's segment_ground_flags()) into a per-VOXEL majority
    vote, aligned with voxelize_cloud()'s own returned `centers` — Open3D's
    VoxelGrid exposes no per-voxel source-point membership directly, so this
    independently recomputes each point's grid index using the exact same
    fixed anchor (_VOXEL_GRID_MIN_BOUND) and voxel size voxelize_cloud()
    itself uses (create_from_point_cloud_within_bounds's own `origin` is
    guaranteed to equal that anchor exactly, not a per-call bounding-box
    derived one — see voxelize_cloud's docstring), so the same point always
    buckets into the same voxel index either way.

    A tie (exactly half ground, half obstacle in a voxel) resolves to
    obstacle — the safer default for navigation. A voxel with no points at
    all mapped to it here (shouldn't happen — centers come FROM these same
    points — but defended anyway) also defaults to obstacle.
    """
    n_centers = len(centers)
    if len(points) == 0 or n_centers == 0:
        return np.zeros(n_centers, dtype=bool)

    def _grid_keys(idx: np.ndarray) -> np.ndarray:
        # Packs a (ix, iy, iz) grid index into one int64 key for fast
        # grouping. BASE comfortably covers this project's real grid extent
        # (_VOXEL_GRID_MIN_BOUND/_MAX_BOUND span 2000m at even a fine 0.02m
        # voxel size -> 100,000 cells/axis, well under BASE).
        BASE = 300_000
        return idx[:, 0] * BASE * BASE + idx[:, 1] * BASE + idx[:, 2]

    pt_idx = np.floor((points - _VOXEL_GRID_MIN_BOUND) / vsize).astype(np.int64)
    pt_keys = _grid_keys(pt_idx)
    center_idx = np.round((centers - _VOXEL_GRID_MIN_BOUND) / vsize - 0.5).astype(np.int64)
    center_keys = _grid_keys(center_idx)

    order = np.argsort(pt_keys)
    sorted_keys = pt_keys[order]
    sorted_flags = np.asarray(flags, dtype=np.int64)[order]
    unique_keys, start_idx, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_flags, start_idx)
    majority = sums * 2 > counts

    key_to_majority = dict(zip(unique_keys.tolist(), majority.tolist()))
    return np.array([key_to_majority.get(k, False) for k in center_keys.tolist()], dtype=bool)


def _zone_contains_point_xz(zone: Zone, x: float, z: float) -> bool:
    return zone.bbox_min[0] <= x <= zone.bbox_max[0] and zone.bbox_min[2] <= z <= zone.bbox_max[2]


def _zone_center_dist_xz(zone: Zone, x: float, z: float) -> float:
    cx = (zone.bbox_min[0] + zone.bbox_max[0]) / 2.0
    cz = (zone.bbox_min[2] + zone.bbox_max[2]) / 2.0
    return ((x - cx) ** 2 + (z - cz) ** 2) ** 0.5


def _back_project_kps(
    keypoints: list,
    depth_map: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
) -> np.ndarray:
    """
    Back-project ORB keypoints to world-space 3D coords.
    Returns Nx3 float32; rows with invalid depth are set to [0,0,0].
    """
    h, w = depth_map.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    pts = []
    for kp in keypoints:
        u, v = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= v < h and 0 <= u < w:
            d = float(depth_map[v, u])
            if 0.1 < d < 8.0:
                X = (u - cx) * d / fx
                Y = (v - cy) * d / fy
                pw = c2w @ np.array([X, Y, d, 1.0], dtype=np.float64)
                pts.append(pw[:3].astype(np.float32))
                continue
        pts.append(np.zeros(3, dtype=np.float32))
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


def _back_project_frame(
    rgb: np.ndarray,
    depth_frame,
    pose: np.ndarray,
    max_depth: float = 5.0,
) -> Optional[tuple]:
    """
    Dense back-projection: all valid pixels → world-space points + RGB colors.
    Returns (Nx3 world_pts float64, Nx3 colors float64) or None.
    Uses per-pixel rays when available, falls back to K-matrix back-projection.
    """
    depth = depth_frame.depth_map
    rays = depth_frame.rays
    mask = (depth > 0.1) & (depth < max_depth)
    if mask.sum() < 100:
        return None

    if rays is not None:
        pts_cam = (rays[mask] * depth[mask, np.newaxis]).astype(np.float64)
    else:
        h, w = depth.shape
        K = (depth_frame.intrinsics.astype(np.float64)
             if depth_frame.intrinsics is not None
             else _estimate_K(h, w))
        ys, xs = np.where(mask)
        zs = depth[mask].astype(np.float64)
        x3 = (xs - K[0, 2]) * zs / K[0, 0]
        y3 = (ys - K[1, 2]) * zs / K[1, 1]
        pts_cam = np.stack([x3, y3, zs], axis=-1)

    ones = np.ones((len(pts_cam), 1), dtype=np.float64)
    pts_h = np.hstack([pts_cam, ones])
    pts_world = (pose @ pts_h.T).T[:, :3]
    colors = rgb[mask].astype(np.float64) / 255.0
    return pts_world, colors


# Default novelty-gate thresholds — GUI/configure_novelty_gate-overridable,
# same "constructor default + live-tunable" pattern as OccupancyMap's own
# constants (see ScanSession.configure_novelty_gate /
# ScanSessionManager.configure_novelty_gate_defaults).
DEFAULT_MIN_NEW_FRACTION = 0.85
DEFAULT_MIN_NEW_COUNT = 60
DEFAULT_MIN_ROTATION_DEG = 2.0
DEFAULT_MIN_SHARPNESS = 0.0  # 0 disables blur-reject entirely


@dataclass
class _PendingTagFrame:
    """One novelty+blur-gated accepted frame, buffered in
    ScanSession._tag_pending until a full SemanticMapper.IMAGES_PER_PROMPT
    batch is ready for the Gemini -> GroundingDINO-tiny tag+detect+
    backproject pipeline (see semantic_mapper.py / frame_extractor/
    tagging.py) — short-lived, unlike the old design's session-long
    _frame_store, since detection now runs immediately rather than being
    deferred to an on-demand query (see ScanSession.resolve_landmark())."""
    frame_bgr: np.ndarray        # HxWx3 uint8
    depth_map: np.ndarray        # float32 HxW, metres
    world_pose: np.ndarray       # 4x4 float64, camera-to-world
    K: np.ndarray                 # 3x3 float64 intrinsics
    frame_idx: int


# ── ScanSession ────────────────────────────────────────────────────────────────


class ScanSession:
    """
    Holds all mutable state for one scanning session (one location_id).
    Thread-safe: process_frames_batch may be called from a worker thread
    while set_label / export are called from another.
    """

    def __init__(
        self,
        location_id: str,
        estimator: BaseDepthEstimator,
        rtabmap_client: Optional["RtabmapPoseClient"] = None,
        semantic_mapper: Optional[SemanticMapper] = None,
        zone_type: str = "",
        occupancy_params: Optional[dict] = None,
        novelty_params: Optional[dict] = None,
    ) -> None:
        self.location_id = location_id
        self.estimator = estimator
        self.rtabmap_client = rtabmap_client
        self.semantic_mapper = semantic_mapper
        self.zone_type = zone_type
        self.labeler = ZoneLabeler()

        # Novelty+blur gate (see orb_novelty_gate.py) — same "constructor
        # default + live-tunable, remembered for reset_cloud()" pattern as
        # _occupancy_params/OccupancyMap below. Frames failing this gate are
        # excluded from cloud/TSDF fusion, the frame store, and VLM tagging —
        # see process_frames_batch's Step 3.
        self._novelty_params: dict = dict(novelty_params or {})
        self._min_sharpness: float = self._novelty_params.get("min_sharpness", DEFAULT_MIN_SHARPNESS)
        self._min_new_fraction: float = self._novelty_params.get("min_new_fraction", DEFAULT_MIN_NEW_FRACTION)
        self._min_new_count: int = self._novelty_params.get("min_new_count", DEFAULT_MIN_NEW_COUNT)
        self._min_rotation_deg: float = self._novelty_params.get("min_rotation_deg", DEFAULT_MIN_ROTATION_DEG)
        self.novelty_gate = OrbNoveltyGate(**self._gate_ctor_kwargs())

        # Buffer of accepted frames awaiting a full SemanticMapper.
        # IMAGES_PER_PROMPT batch for the eager tag+detect+backproject
        # pipeline — see _PendingTagFrame's docstring. Landmarks resolved
        # from a flushed batch land directly in self._raw_landmarks (below);
        # nothing about a frame itself is kept around afterward.
        self._tag_pending: List[_PendingTagFrame] = []

        self.tracker = FeatureTracker(n_features=2000)
        self.pose_graph = PoseGraph()
        # Bayesian/height tuning knobs (occupancy_map.OccupancyMap.set_params'
        # keys) — kept here so reset_cloud()'s fresh OccupancyMap re-applies
        # whatever the GUI last configured instead of silently reverting to
        # class defaults. See scan_gui.py's "Occupancy Map Settings" accordion.
        self._occupancy_params: dict = dict(occupancy_params or {})
        self.occupancy_map = OccupancyMap(resolution=0.05, **self._occupancy_params)
        # self._cloud is NOT kept live/up to date during scanning anymore —
        # it's built lazily, only on demand (see ensure_cloud_built()), by
        # Reload (Live Points), Voxelize, Export, or finalize_voxel_and_
        # occupancy(). Occupancy Map generation never needed the full
        # accumulated cloud (only each batch's own new_cloud — see
        # process_frames_batch's Step 3/4), so re-running voxel_down_sample +
        # remove_statistical_outlier over the ENTIRE ever-growing cloud on
        # every single batch was pure waste unless something was actually
        # going to look at self._cloud that run.
        self._cloud = o3d.geometry.PointCloud()
        # Each batch's own already-cleaned (voxelized + outlier-removed)
        # point cloud, buffered raw — ensure_cloud_built() merges these into
        # self._cloud on demand. Kept per-batch (not concatenated eagerly)
        # so buffering itself stays cheap; only the on-demand merge pays for
        # one more voxel pass across everything.
        self._raw_cloud_batches: List[o3d.geometry.PointCloud] = []
        self._raw_point_count: int = 0  # live running total — cheap progress indicator, no full build needed
        # How many of self._raw_cloud_batches are already folded into
        # self._cloud_raw_accum — lets ensure_cloud_built() only concatenate
        # NEW batches on repeat calls (e.g. once per streamed chunk when the
        # Live Points checkbox is on) instead of re-concatenating the whole
        # history every time. self._cloud_raw_accum holds the UNdownsampled
        # concatenation of every batch so far — kept separate from
        # self._cloud (the actual voxel-downsampled display cloud) so
        # re-voxel-downsampling always starts from genuinely raw points,
        # never from an already-computed centroid re-averaged as if it were
        # a single point (which would silently under-weight old, densely-
        # sampled voxels vs. new ones). The final voxel_down_sample pass is
        # still O(current total size) each call regardless — Open3D has no
        # incremental voxel-merge primitive — this only avoids redundant
        # re-concatenation of already-merged batches.
        self._merged_batch_count: int = 0
        self._cloud_raw_accum = o3d.geometry.PointCloud()
        self._last_sor_nb_neighbors: int = SOR_NB_NEIGHBORS
        self._last_sor_std_ratio: float = SOR_STD_RATIO

        # TSDF fusion — non-RTAB-Map (IMU+VO/VO) pose sources only. Unlike
        # _raw_cloud_batches (first observation wins, never revised), TSDF
        # averages every trusted frame's contribution to the same surface,
        # denoising residual noise the depth-consistency gate doesn't reject
        # outright (see feature_tracker.py's _triangulate_depth_agreement —
        # that gate only rejects OBVIOUSLY bad frames; a frame that passes
        # still gets fused here with no revision otherwise). Deliberately NOT
        # used for RTAB-Map mode: RTAB-Map's own server-side reconstruction
        # already gets retroactively re-transformed by its CURRENT
        # graph-corrected pose on every loop-closure resync (see
        # _rtabmap_full_resync) — a client-side TSDF volume integrated at
        # pose-at-time-of-integration would reintroduce that exact staleness
        # with no correction path, regressing a problem RTAB-Map mode's
        # design already solves. ensure_cloud_built() branches on
        # _session_uses_rtabmap to pick the right source.
        self._tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=VOXEL_SIZE, sdf_trunc=VOXEL_SIZE * 4,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self._tsdf_integrated_count: int = 0
        self._tsdf_cloud_cache: Optional[o3d.geometry.PointCloud] = None
        self._tsdf_extracted_at_count: int = -1
        self._session_uses_rtabmap: bool = False

        self.latest_frame_rgb: Optional[np.ndarray] = None
        self.last_frames_rgb: list = []
        self.last_depth_frames: list = []
        self.last_frame_poses: list = []
        self.last_trajectory: np.ndarray = np.zeros((0, 3), dtype=np.float32)

        # The ONE canonical voxelization result — populated incrementally by
        # _merge_voxels() every time process_frames_batch/_rtabmap_process_
        # nodes voxelizes a batch/node's new points for the Occupancy Map
        # feed (see that method), and read directly by scan_gui.py's
        # Voxelization tab / Live Reconstruction view — there is no second,
        # independently-computed voxelization anywhere else in the live
        # pipeline. _voxel_dict is keyed by grid index at self.last_voxel_size
        # (see _merge_voxels) so repeated observations of the same physical
        # voxel update in place instead of duplicating; a change in voxel
        # size (e.g. the GUI's on-demand "Voxelize" at a different size)
        # resets it, since indices from two different sizes aren't comparable.
        self._voxel_dict: dict = {}
        self.last_voxel_centers: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.last_voxel_colors: Optional[np.ndarray] = None
        self.last_voxel_size: float = DEFAULT_VOXEL_SIZE

        # Every batch's camera positions (post axis-remap), accumulated across
        # the WHOLE session — fed to occupancy_map.update() in one shot by
        # finalize_voxel_and_occupancy() so its eye-height estimate (median
        # camera Y) reflects the complete walkthrough, not an early subset.
        self._all_trajectory: List[np.ndarray] = []

        self._frame_count: int = 0
        self._camera_K: Optional[list] = None
        self._imu: Optional[ImuIntegrator] = None
        self._lock = threading.Lock()

        # Fallback pose for a use_rtabmap_pose frame that timed out (tracking
        # lost / node lagging) — reused rather than dropping the frame.
        self._rtabmap_last_pose: Optional[np.ndarray] = None

        # Highest RTAB-Map node id already pulled via rtabmap_client.get_cloud()
        # — lets each batch request only nodes reconstructed since last time
        # (see process_frames_batch's RTAB-Map Step 3 branch), instead of
        # re-fetching/re-voxelizing the whole session's cloud every batch.
        # Reset to 0 (pull everything again) whenever a batch's TRACK replies
        # report a loop closure — RTAB-Map's OWN corrected poses for
        # previously-pulled nodes may have just shifted, so those nodes' cached
        # clouds/occupancy-map contributions are stale and must be rebuilt from
        # a fresh full pull (see _rtabmap_full_resync()).
        self._rtabmap_last_pulled_node_id: int = 0

        # RTAB-Map node ids whose originating frame failed the depth-
        # consistency check (see feature_tracker.py's
        # _triangulate_depth_agreement, reused here as a side channel — see
        # process_frames_batch's RTAB-Map branch) — filtered out in
        # _rtabmap_process_nodes() so a frame with bad DA3 depth never gets
        # its RTAB-Map-reconstructed geometry fused into the map, same intent
        # as the IMU+VO/VO path's per-frame back-projection skip, just
        # attributed via node_id instead since RTAB-Map's own reconstruction
        # is an opaque server-side process with no other way to correlate a
        # bad frame with the node it produced.
        self._rtabmap_untrusted_node_ids: set = set()

        # Per-node SOR (statistical outlier removal) keep-mask cache, keyed
        # by node_id — a full resync (_rtabmap_full_resync, on loop closure)
        # re-pulls EVERY node's cloud, re-transformed by its new corrected
        # pose, but a rigid transform (rotation+translation) preserves every
        # pairwise Euclidean distance — so SOR's outlier decision for a given
        # node is IDENTICAL before and after a pose correction; only the
        # points' world-space position changes, not which ones are outliers.
        # Recomputing SOR for every already-seen node on every resync was
        # pure redundant work (measured: ~7.8s for a 1.2M-point resync,
        # almost entirely nodes seen before). _rtabmap_process_nodes() now
        # only runs SOR for node ids not yet in this cache; a cached mask is
        # applied directly to the freshly re-posed points instead. Falls
        # back to recomputing for a node if its cached mask's length doesn't
        # match the freshly-pulled point count (shouldn't normally happen —
        # a node's own stored SensorData never changes — but defends against
        # it rather than misapplying a wrong-length mask).
        self._rtabmap_sor_keep_mask: Dict[int, np.ndarray] = {}

        # Per-node depth confidence (see occupancy_map.py's update()
        # docstring) — populated from the same side-channel FeatureTracker
        # depth-consistency check that builds _rtabmap_untrusted_node_ids
        # (process_frames_batch's RTAB-Map branch), just continuous instead
        # of binary: 1 - frac_bad, or 1.0 if the check couldn't run (too few
        # PnP inliers). Read by _rtabmap_process_nodes() per node when
        # calling occupancy_map.update().
        self._rtabmap_node_confidence: Dict[int, float] = {}

        # Diagnostics for the most recent process_frames_batch() call — read by
        # scan_gui.py to surface actual pose source + RTAB-Map tracking health
        # in the Gradio log panel (server-console prints alone are easy to miss).
        self.last_pose_source: str = "none"
        self.last_rtabmap_lost: int = 0
        self.last_rtabmap_total: int = 0
        self.last_rtabmap_loop_closure: bool = False

        # pure_walking (SessionMode.WALKING) only — see process_frames_batch's
        # pure_walking param and CLAUDE.md's walking-mode local-map note.
        # Wall-clock time.time() (not frame_timestamps_ns — a different
        # clock domain, and we want a threshold that's meaningful even if
        # frame_timestamps_ns is ever absent) at which the CURRENT unbroken
        # streak of "RTAB-Map reported no pose" began; None while tracked.
        # See PURE_WALKING_LOST_RESET_S.
        self._walking_lost_streak_start: Optional[float] = None
        # True only on the process_frames_batch() call whose processing
        # just triggered a pure_walking reset — read (and implicitly
        # consumed, since it's overwritten at the top of every call) by
        # mapping_servicer.py to set MappingUpdate.reset_occurred.
        self.last_reset_occurred: bool = False

        # Depth-consistency confidence behind the most recent occupancy_map.update()
        # call (batch-level for IMU+VO/VO, last-processed-node for RTAB-Map) — read
        # by MappingService (server/services/mapping_servicer.py) to report per-update
        # confidence to the client without recomputing it from depth_checks.
        self.last_batch_confidence: float = 1.0

        # Session-wide semantic landmark accumulation (raw, unmerged). Persists
        # across all segments/zones — global overlap merging happens once, at
        # export time, via finalize_landmarks().
        self._raw_landmarks: List[Landmark] = []
        self._current_area_name: str = ""

    # ── private ───────────────────────────────────────────────────────────────

    def _gate_ctor_kwargs(self) -> dict:
        """Pulls only the OrbNoveltyGate-constructor-relevant subset out of
        _novelty_params (n_features/ratio/min_raw_matches/
        ransac_threshold_px/use_gpu) — the rest (min_sharpness/
        min_new_fraction/min_new_count/min_rotation_deg) are runtime args to
        evaluate_with_keypoints()/decide_accept(), not constructor args."""
        keys = ("n_features", "ratio", "min_raw_matches", "ransac_threshold_px", "use_gpu")
        return {k: self._novelty_params[k] for k in keys if k in self._novelty_params}

    def _evaluate_novelty(self, rgb: np.ndarray) -> tuple:
        """IMU+VO/VO branches only — call immediately after
        self.tracker.track(rgb, ...) so self.tracker._prev holds THIS
        frame's already-detected ORB keypoints/descriptors (reused here, no
        redundant second detection pass — see orb_novelty_gate.py's module
        docstring). Returns (accepted, sharpness, reject_reason); calls
        self.novelty_gate.accept(...) when accepted."""
        prev = self.tracker._prev
        sharpness = _sharpness_score(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        (
            _keypoints, _descriptors, _new_mask, new_count, _total,
            new_fraction, best_match_inliers, rotation_deg, _translation_px,
        ) = self.novelty_gate.evaluate_with_keypoints(
            prev.keypoints, prev.descriptors, rgb.shape[:2],
            self._min_new_fraction, self._min_new_count,
        )
        accepted = decide_accept(
            new_fraction, new_count, best_match_inliers, rotation_deg, sharpness,
            self._min_new_fraction, self._min_new_count, self._min_rotation_deg,
            self._min_sharpness,
        )
        if accepted:
            self.novelty_gate.accept(prev.keypoints, prev.descriptors)
            reject_reason = ""
        elif new_fraction < self._min_new_fraction or new_count < self._min_new_count:
            reject_reason = f"not novel enough (new_fraction={new_fraction:.2f} < {self._min_new_fraction:.2f})"
        elif best_match_inliers and rotation_deg < self._min_rotation_deg:
            reject_reason = f"insufficient rotation ({rotation_deg:.1f}° < {self._min_rotation_deg:.1f}°)"
        else:
            reject_reason = f"too blurry (sharpness={sharpness:.0f} < min_sharpness={self._min_sharpness:.0f})"
        return accepted, sharpness, reject_reason

    def _flush_tag_pending(self) -> None:
        """Runs SemanticMapper.tag_and_backproject_batch() across whatever's
        currently buffered in self._tag_pending (a full IMAGES_PER_PROMPT
        batch, or a partial leftover at finalize time) and appends every
        resolved Landmark straight into self._raw_landmarks — immediate,
        not deferred (see semantic_mapper.py's module docstring for why this
        replaced the old VLM-tag-then-defer-GroundingDINO design). No-op if
        semantic_mapper isn't configured or nothing's pending."""
        if not self._tag_pending or self.semantic_mapper is None:
            return
        batch = self._tag_pending
        self._tag_pending = []
        try:
            landmark_lists = self.semantic_mapper.tag_and_backproject_batch(
                [pf.frame_bgr for pf in batch],
                [pf.depth_map for pf in batch],
                [pf.world_pose for pf in batch],
                [pf.K for pf in batch],
                [pf.frame_idx for pf in batch],
            )
        except Exception as e:
            print(f"[ScanSession:{self.location_id}] tag_and_backproject_batch failed: {e}")
            return
        for landmarks in landmark_lists:
            self._raw_landmarks.extend(landmarks)

    # ── public ────────────────────────────────────────────────────────────────

    def set_imu_file(self, csv_path: str, orientation: str = "portrait") -> None:
        """Load an IMU CSV and prepare the integrator for this session.
        `orientation` — see IMU_ORIENTATIONS / ImuIntegrator docstring."""
        self._imu = ImuIntegrator(csv_path, orientation=orientation)

    def compute_segment_poses(
        self, frame_timestamps_ns: List[float]
    ) -> Optional[List[np.ndarray]]:
        """
        Pre-compute one 4×4 c2w pose per frame using IMU dead-reckoning, from
        each frame's actual capture timestamp (camera.csv — same clock domain
        as imu.csv, see ImuSensor.kt). Call this *before* process_frames_batch
        so all poses are ready before the depth-estimation pass begins.
        Returns None when no IMU data is loaded.
        """
        if self._imu is None:
            return None
        poses = [self._imu.pose_at(ts_ns) for ts_ns in frame_timestamps_ns]
        print(
            f"[ScanSession:{self.location_id}] Pre-computed {len(poses)} IMU poses "
            f"from frame timestamps"
        )
        return poses

    def process_frames_batch(
        self,
        frames_rgb: list,
        imu_poses: Optional[List[np.ndarray]] = None,
        use_rtabmap_pose: bool = False,
        frame_timestamps_ns: Optional[List[float]] = None,
        axis_perm: Optional[np.ndarray] = None,
        sor_nb_neighbors: int = SOR_NB_NEIGHBORS,
        sor_std_ratio: float = SOR_STD_RATIO,
        occupancy_voxel_size: float = DEFAULT_VOXEL_SIZE,
        walking_lite: bool = False,
        pure_walking: bool = False,
    ) -> tuple[int, list[float], float]:
        """
        Run depth estimation and dense back-projection on a mini-batch.

        When `imu_poses` is provided (pre-computed by compute_segment_poses),
        those poses are used directly — FeatureTracker VO and pose-graph
        optimisation are skipped.  This implements the two-pass strategy:
          Pass 1 (caller): compute_segment_poses → all IMU poses for the segment
          Pass 2 (here):   depth estimation + back-projection per mini-batch

        Without `imu_poses`, falls back to the original FeatureTracker VO +
        pose-graph optimisation pipeline.

        `use_rtabmap_pose` routes poses through self.rtabmap_client instead of
        IMU+VO/VO. Unlike the removed ORB-SLAM3 path, no raw IMU is sent at
        all: RTAB-Map's RGB-D visual odometry + loop closure only needs
        camera intrinsics + a depth map, both already available from Step 1's
        depth_frames, sidestepping the unreliable camera-IMU calibration
        ORB-SLAM3 depended on. Depth still comes from self.estimator either
        way; RTAB-Map replaces pose only, and — like ORB-SLAM3 before it —
        needs no cross-batch anchoring since its own map/pose graph is
        continuous for the whole session.

        `axis_perm` (3×3 signed permutation, see scan_gui.py's _perm_matrix) is
        applied to every pose's ROTATION ONLY (see _remap_poses) after pose
        computation/chaining — internal cross-batch anchors
        (self._rtabmap_last_pose) stay in the original, un-remapped frame so
        their own math is unaffected; only the world-space output (point
        cloud, occupancy map, semantic-mapper world_pose, trajectory, exposed
        cam_pos) reflects the remap, applied consistently across every pose
        source. Translation is intentionally untouched — see _remap_poses'
        docstring for why.

        This accumulates the point cloud (self._raw_cloud_batches) and
        per-batch trajectory (self._all_trajectory), AND progressively
        updates the Occupancy Map once per call (or once per RTAB-Map node,
        see Step 3b) — see occupancy_map.py's "Progressive, Bayesian,
        SLAM-style Occupancy Map" docs for why building it live like this,
        rather than once at the end, is safe. The Occupancy Map is fully
        INFERRED FROM THE VOXELIZATION, not a separately-computed
        approximation of it: this batch/node's own new points are run
        through voxelize_cloud() (the exact same function scan_gui.py's
        Voxelization view calls) at `occupancy_voxel_size`, and the
        resulting voxel centers — not the raw or fine-voxel-downsampled
        points — are what feeds OccupancyMap.update(). `occupancy_voxel_size`
        matches whatever the GUI's Voxelization view is currently showing
        (scan_gui.py's voxel_size_input), so occupancy cells correspond
        exactly to what's displayed there — same voxels, same coordinates,
        not just a similarly-sized approximation. This is still strictly
        per-call/per-node — never a re-feed of accumulated history, which
        would double-count old points and break the Bayesian log-odds
        design (see occupancy_map.py).

        `walking_lite` (only meaningful together with `use_rtabmap_pose`):
        skips semantic tagging (_PendingTagFrame/_tag_pending)
        AND skips Step 3b's RTAB-Map get_cloud()/SOR/server-side-voxelize
        pull entirely — the two most expensive parts of a batch (observed
        15s+ for get_cloud, 12s+ for SOR on a real session; VLM tagging is
        pointless work outside an explicit scan pass anyway). Instead,
        Step 3's LOCAL back-projection path (normally IMU+VO/VO-only) also
        runs for RTAB-Map-posed frames, feeding the exact same generic
        Step 4 occupancy update every other pose source already uses — RTAB-
        Map's pose is still authoritative, only its OWN heavier server-side
        reconstruction is bypassed. No TSDF fusion either (that's for
        denoised Live Points/PLY-export quality, not needed for a live
        occupancy grid). Intended for walking/guiding — modes that need a
        live, responsive occupancy grid but not a persisted, semantically-
        tagged reconstruction (that's what an explicit scan pass is for; see
        CLAUDE.md's "Client-Orchestrated Live Session" walking-mode note).

        `pure_walking` (only meaningful together with `walking_lite`/
        `use_rtabmap_pose`; SessionMode.WALKING specifically, not GUIDING):
        gets the same live Step 4 occupancy_map.update()/_merge_voxels() as
        GUIDING — walking's beacon now steers along a client-side
        LocalPathPlanner route through this grid toward a synthetic
        "straight ahead" target (see CLAUDE.md's walking-mode local-map
        note) — but the grid is NEVER persisted (mapping_servicer.py skips
        finalize/snapshot-save for pure_walking streams). Both WALKING and
        GUIDING now enable the consecutive-tracking-loss reset check (see
        the RTAB-Map branch below), just at different thresholds —
        `PURE_WALKING_LOST_RESET_S` (0.5s) for WALKING, `GUIDING_LOST_RESET_S`
        (2.0s) for GUIDING, since resetting a route mid-navigation is far
        more disruptive than resetting walking's short-lived, cheap-to-
        rebuild local grid — see those constants' own comments. SCAN
        (`walking_lite=False`) never resets on tracking loss at all.

        Returns (point_count, cam_pos, infer_ms).
        """
        with self._lock:
            self.last_reset_occurred = False

            # ── Step 0: blur pre-check — skip DA3/pose entirely for a batch
            # that's not worth it ────────────────────────────────────────────
            # The novelty+blur gate below (Step 3) only vetoes FUSION — DA3
            # depth estimation and RTAB-Map/VO pose computation already ran
            # for every frame by the time it fires, regardless of outcome.
            # Real cost data showed DA3 (600-1200ms) + RTAB-Map pose
            # (300-1200ms) paid on every single mini-batch even when every
            # frame in it turned out unusable — pure waste for a frame that
            # was already known-blurry before either call started. This gate
            # runs first and, for a batch with nothing worth keeping, skips
            # Step 1 onward completely rather than paying for depth/pose work
            # whose result gets thrown away a few lines later anyway.
            # min_sharpness<=0 (DEFAULT_MIN_SHARPNESS) disables this
            # entirely — same "0 = off" convention the novelty gate itself
            # already uses, so a session that never calls
            # configure_novelty_gate() sees byte-identical behavior to
            # before this pre-check existed.
            frame_sharpness: list = [None] * len(frames_rgb)
            if self._min_sharpness > 0:
                frame_sharpness = [
                    _sharpness_score(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)) for f in frames_rgb
                ]
                keep_idx = [i for i, s in enumerate(frame_sharpness) if s >= self._min_sharpness]
                if not keep_idx:
                    print(
                        f"[blur-precheck] SKIPPED entire batch — {len(frames_rgb)}/{len(frames_rgb)} "
                        f"frame(s) below min_sharpness={self._min_sharpness:.0f} "
                        f"(scores={[f'{s:.0f}' for s in frame_sharpness]}); no DA3/pose work done."
                    )
                    cam_pos = (
                        self.last_frame_poses[-1][:3, 3].tolist()
                        if self.last_frame_poses else [0.0, 0.0, 0.0]
                    )
                    return self._raw_point_count, cam_pos, 0.0
                if len(keep_idx) < len(frames_rgb):
                    print(
                        f"[blur-precheck] dropped {len(frames_rgb) - len(keep_idx)}/{len(frames_rgb)} "
                        f"frame(s) below min_sharpness={self._min_sharpness:.0f} before DA3/pose."
                    )
                    frames_rgb = [frames_rgb[i] for i in keep_idx]
                    frame_sharpness = [frame_sharpness[i] for i in keep_idx]
                    if imu_poses is not None:
                        imu_poses = [imu_poses[i] for i in keep_idx]
                    if frame_timestamps_ns is not None:
                        frame_timestamps_ns = [frame_timestamps_ns[i] for i in keep_idx]

            # ── Step 1: depth estimation ──────────────────────────────────────
            active_estimator = self.estimator
            _t0 = time.perf_counter()
            if hasattr(active_estimator, "estimate_batch"):
                depth_frames = active_estimator.estimate_batch(frames_rgb)
            else:
                depth_frames = [active_estimator.estimate(f) for f in frames_rgb]
            infer_ms = (time.perf_counter() - _t0) * 1000
            _estimator_device = getattr(active_estimator, "device", "unknown")
            if walking_lite:
                # Debug prints trimmed to walking/guiding only — confirmed
                # with the user; scan mode's own console output was getting
                # too noisy to read through during live debugging.
                print(
                    f"[timing] depth estimation ({len(frames_rgb)} frames, "
                    f"{type(active_estimator).__name__}, device={_estimator_device}): {infer_ms:.1f} ms"
                )

            # Cache intrinsics from first available depth frame
            if self._camera_K is None:
                for df in depth_frames:
                    if df.intrinsics is not None:
                        self._camera_K = df.intrinsics.tolist()
                        break

            # ── Step 2: Camera poses ───────────────────────────────────────────
            _t0_pose = time.perf_counter()
            _lost_track_should_reset = False
            if use_rtabmap_pose and self.rtabmap_client is None:
                print(
                    f"[ScanSession:{self.location_id}] RTAB-Map poses requested but no "
                    f"connected client — falling back to VO."
                )
                use_rtabmap_pose = False
            self._session_uses_rtabmap = use_rtabmap_pose

            # Per-frame (trustworthy, frac_bad, n_checked) from FeatureTracker's
            # independent depth-consistency check — see feature_tracker.py's
            # _triangulate_depth_agreement. Defaults to "trustworthy" for
            # RTAB-Map: this list stays unused there (Step 3's back-projection
            # loop below only reads it when NOT use_rtabmap_pose, since RTAB-
            # Map's own reconstruction doesn't use this project's DA3 back-
            # projection at all) — the same underlying check still runs for
            # RTAB-Map frames, just through a separate node_id-based veto (see
            # the use_rtabmap_pose branch below and _rtabmap_process_nodes),
            # since RTAB-Map's server-side reconstruction has no per-frame
            # cloud for this list to gate directly. Gets overwritten with real
            # per-frame results in the imu_poses/VO branches below.
            depth_checks: list = [(True, None, 0)] * len(frames_rgb)

            # Per-frame (accepted, sharpness) from the novelty+blur gate (see
            # orb_novelty_gate.py) — populated per pose-source branch below,
            # aligned with frames_rgb order. Consumed by Step 3 to gate cloud/
            # TSDF fusion, frame-store admission, and VLM tagging (all three
            # from one decision, computed once per frame — see Step 3).
            novelty_flags: list = []

            if use_rtabmap_pose:
                # RTAB-Map's own odometry+map/pose-graph is continuous for the
                # whole session (unlike the DA3 path, no per-batch anchoring
                # needed) — no IMU passthrough at all, unlike the removed
                # ORB-SLAM3 branch: this only ever reads depth_frames (already
                # computed in Step 1 above) and frame_timestamps_ns.
                tracked = self.rtabmap_client.track_batch(
                    frames_rgb, depth_frames, frame_timestamps_ns or [0.0] * len(frames_rgb)
                )
                n_lost = sum(t.pose is None for t in tracked)
                self.last_rtabmap_lost = n_lost
                self.last_rtabmap_total = len(tracked)
                self.last_rtabmap_loop_closure = any(t.loop_closure for t in tracked)
                if n_lost:
                    print(
                        f"[ScanSession:{self.location_id}] RTAB-Map tracking LOST on "
                        f"{n_lost}/{len(tracked)} frames this batch — reusing last known pose."
                    )

                # Consecutive-loss streak (WALKING + GUIDING only — SCAN
                # never resets on tracking loss, its reconstruction is too
                # valuable to nuke over a brief hiccup). Time-based, not
                # frame-count, since frame arrival cadence is client-
                # controlled. Any single tracked frame this batch clears
                # the streak entirely — "no sign of recovery" means
                # UNBROKEN loss, not merely frequent loss. Threshold
                # differs by mode — see PURE_WALKING_LOST_RESET_S/
                # GUIDING_LOST_RESET_S's own comment for why. This is the
                # ONGOING in-session reset (real, sustained RTAB-Map
                # tracking loss mid-walk/guide) — separate from, and
                # unaffected by, the scan->walking/guiding STREAM-OPEN
                # reset skip in StreamingScanSession.__init__ (see its own
                # comment for why that one specifically is suppressed).
                if walking_lite:
                    _lost_reset_threshold_s = PURE_WALKING_LOST_RESET_S if pure_walking else GUIDING_LOST_RESET_S
                    _now_wall = time.time()
                    for t in tracked:
                        if t.pose is None:
                            if self._walking_lost_streak_start is None:
                                self._walking_lost_streak_start = _now_wall
                            elif _now_wall - self._walking_lost_streak_start >= _lost_reset_threshold_s:
                                _lost_track_should_reset = True
                        else:
                            self._walking_lost_streak_start = None

                # Depth-consistency check, reused here as a side channel:
                # self.tracker's own returned pose is discarded (RTAB-Map's
                # own pose above is authoritative for this mode) — only
                # last_depth_trustworthy is read. reset_cloud() already resets
                # self.tracker per fresh StreamingScanSession, so its internal
                # VO state never mixes across pose-source choices within one
                # session. Unlike IMU+VO/VO (which just skips back-projecting
                # a bad frame locally), RTAB-Map's reconstruction happens
                # server-side with no client-side view of which pixels fed
                # which points — so instead we track which RTAB-Map NODE (if
                # any) each frame became, via TrackedFrame.node_id (see
                # rtabmap_server.cc), and veto pulling that node's geometry
                # later in _rtabmap_process_nodes().
                for _i, (rgb, df, t) in enumerate(zip(frames_rgb, depth_frames, tracked)):
                    self.tracker.track(rgb, df.depth_map, df.intrinsics)
                    # Populated here (previously left at the hardcoded "trustworthy"
                    # default above) so walking_lite's local back-projection path
                    # below can gate on the SAME per-frame depth-consistency check
                    # IMU+VO/VO already use, instead of trusting every frame blindly.
                    depth_checks[_i] = (
                        self.tracker.last_depth_trustworthy,
                        self.tracker.last_depth_agree_err,
                        self.tracker.last_depth_agree_n,
                    )
                    if t.node_id >= 0:
                        err = self.tracker.last_depth_agree_err
                        self._rtabmap_node_confidence[t.node_id] = 1.0 - err if err is not None else 1.0
                    if not self.tracker.last_depth_trustworthy and t.node_id >= 0:
                        # State only, no print — this is scan-only-meaningful
                        # bookkeeping (only _rtabmap_process_nodes/get_cloud
                        # ever reads _rtabmap_untrusted_node_ids, and that
                        # path is skipped entirely for walking_lite), trimmed
                        # from console output per the walking-only debug
                        # log policy above.
                        self._rtabmap_untrusted_node_ids.add(t.node_id)

                    # Novelty+blur gate (RTAB-Map branch): reuses RTAB-Map's
                    # OWN already-computed frame-to-map registration
                    # inlier_fraction (rtabmap_server.cc/rtabmap_client.py) as
                    # the novelty signal, instead of a redundant Python-side
                    # ORB pass — see orb_novelty_gate.py's module docstring
                    # for why new_node_id isn't used for this instead. No
                    # RTAB-Map-native equivalent of min_new_count exists (the
                    # wire only carries a ratio, not a raw match count) — the
                    # threshold itself is passed as new_count so that check
                    # is a deliberate no-op for this branch (see CLAUDE.md).
                    sharpness = (
                        frame_sharpness[_i] if frame_sharpness[_i] is not None
                        else _sharpness_score(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
                    )
                    if t.pose is None:
                        accepted = False  # tracking lost — can't judge novelty this frame
                        reject_reason = "RTAB-Map tracking lost this frame"
                    elif walking_lite:
                        # Walking/guiding never discover landmarks or persist
                        # geometry the way a scan pass does (walking_lite
                        # already skips VLM tagging/get_cloud entirely — see
                        # this method's own docstring) — "is this view novel
                        # compared to earlier ones" is a scan-only question,
                        # since there's no frame store/reconstruction here for
                        # novelty to protect against re-covering. Blur
                        # filtering is disabled live-path-wide (self._min_sharpness
                        # stays at DEFAULT_MIN_SHARPNESS=0.0, confirmed with
                        # the user — see mapping_servicer.py's UpdateMapping),
                        # so this reduces to "accept whenever RTAB-Map
                        # actually produced a pose" — the `<= 0` branch is
                        # what makes that true; still respects a future
                        # nonzero threshold if configure_novelty_gate() is
                        # ever called again.
                        accepted = self._min_sharpness <= 0 or sharpness >= self._min_sharpness
                        reject_reason = (
                            "" if accepted
                            else f"too blurry (sharpness={sharpness:.0f} < min_sharpness={self._min_sharpness:.0f})"
                        )
                    else:
                        # Was: RTAB-Map's own inlier_fraction-based novelty
                        # signal (1.0 - t.inlier_fraction), independent of
                        # ORB. Reverted per a real, confirmed problem: that
                        # signal comes from RTAB-Map's frame-to-LOCAL-MAP
                        # registration, which stays robustly high across
                        # genuinely new viewpoints within the same room (F2M
                        # tracking is DESIGNED to keep registering well) —
                        # so `new_fraction >= min_new_fraction` (0.85, tuned
                        # for the ORB check below) almost never fired after
                        # the first frame, in real scans confirmed to
                        # process barely 1 frame's worth of tags where
                        # frame_extractor's own ORB gate found 8-9 on
                        # identical footage. self.tracker.track() already ran
                        # for THIS frame just above (the depth-consistency
                        # side channel), so self.tracker._prev holds its
                        # keypoints/descriptors — reusing the exact same
                        # OrbNoveltyGate/_evaluate_novelty() the IMU+VO/VO
                        # branch (and frame_extractor itself) uses costs
                        # nothing extra and decouples tagging admission from
                        # RTAB-Map's own (separately throttled — see
                        # rtabmap_server.cc's Rtabmap/DetectionRate) node
                        # creation entirely.
                        accepted, sharpness, reject_reason = self._evaluate_novelty(rgb)
                    novelty_flags.append((accepted, sharpness, reject_reason))
                raw_poses = []
                for t in tracked:
                    p = t.pose
                    if p is None:
                        p = (
                            self._rtabmap_last_pose.copy()
                            if self._rtabmap_last_pose is not None
                            else np.eye(4, dtype=np.float64)
                        )
                    raw_poses.append(p)
                    self._rtabmap_last_pose = p
                for rgb in frames_rgb:
                    self._frame_count += 1
                    self.latest_frame_rgb = rgb
            elif imu_poses is not None:
                # IMU+VO fusion: FeatureTracker provides metric translation;
                # IMU gyro integration provides rotation (gyro is reliable;
                # accelerometer double-integration drifts catastrophically for
                # walking/dynamic motion so translation is NOT taken from IMU).
                raw_poses = []
                depth_checks = []
                for rgb, df, imu_pose in zip(frames_rgb, depth_frames, imu_poses):
                    K = df.intrinsics
                    tracker_pose, _ = self.tracker.track(rgb, df.depth_map, K)
                    depth_checks.append((
                        self.tracker.last_depth_trustworthy,
                        self.tracker.last_depth_agree_err,
                        self.tracker.last_depth_agree_n,
                    ))
                    novelty_flags.append(self._evaluate_novelty(rgb))
                    # Replace VO rotation with IMU rotation, keep VO translation
                    fused = tracker_pose.copy()
                    fused[:3, :3] = imu_pose[:3, :3]
                    raw_poses.append(fused)
                self._frame_count += len(frames_rgb)
                self.latest_frame_rgb = frames_rgb[-1]
            else:
                # VO path: FeatureTracker + pose graph
                raw_poses = []
                depth_checks = []
                _da3_vo_anchor: Optional[np.ndarray] = None

                for rgb, df in zip(frames_rgb, depth_frames):
                    K = df.intrinsics
                    tracker_pose, _ = self.tracker.track(rgb, df.depth_map, K)
                    depth_checks.append((
                        self.tracker.last_depth_trustworthy,
                        self.tracker.last_depth_agree_err,
                        self.tracker.last_depth_agree_n,
                    ))
                    novelty_flags.append(self._evaluate_novelty(rgb))

                    if df.camera_pose is not None:
                        if _da3_vo_anchor is None:
                            _da3_vo_anchor = tracker_pose @ np.linalg.inv(df.camera_pose)
                        pose = _da3_vo_anchor @ df.camera_pose
                    else:
                        pose = tracker_pose

                    raw_poses.append(pose)
                    self._frame_count += 1
                    self.latest_frame_rgb = rgb

                    if self._frame_count % KEYFRAME_INTERVAL != 0:
                        continue
                    prev = self.tracker._prev
                    if prev.descriptors is None or len(prev.keypoints) < MIN_KF_VALID_PTS:
                        continue
                    K_eff = K if K is not None else _estimate_K(*rgb.shape[:2])
                    pts_3d = _back_project_kps(prev.keypoints, df.depth_map, K_eff, pose)
                    valid = pts_3d[:, 2] > 0.1 if len(pts_3d) > 0 else np.array([], bool)
                    if valid.sum() < MIN_KF_VALID_PTS:
                        continue
                    kps_arr = np.float32([kp.pt for kp in prev.keypoints])
                    kf_id = self.pose_graph.add_keyframe(
                        self._frame_count, pose,
                        prev.descriptors[valid], kps_arr[valid], pts_3d[valid],
                    )
                    if len(self.pose_graph.keyframes) > 1:
                        prev_pose = self.pose_graph.keyframes[-2].pose
                        self.pose_graph.add_odometry_edge(
                            kf_id - 1, kf_id, np.linalg.inv(prev_pose) @ pose
                        )
                    if K is not None and len(self.pose_graph.keyframes) > 1:
                        match_id = self.pose_graph.detect_loop(
                            prev.descriptors[valid], kps_arr[valid], pts_3d[valid], K
                        )
                        if match_id is not None:
                            m_pose = self.pose_graph.keyframes[match_id].pose
                            self.pose_graph.add_loop_edge(
                                match_id, kf_id, np.linalg.inv(m_pose) @ pose
                            )

                # Pose graph optimisation (VO path only)
                opt_poses = self.pose_graph.optimize()
                kf_pose_map = {
                    kf.frame_idx: opt_poses.get(kf.id, kf.pose)
                    for kf in self.pose_graph.keyframes
                }
                base_idx = self._frame_count - len(frames_rgb)
                raw_poses = [
                    kf_pose_map.get(base_idx + i, raw_poses[i])
                    for i in range(len(raw_poses))
                ]

            _pose_ms = (time.perf_counter() - _t0_pose) * 1000
            _pose_src_label = (
                "RTAB-Map" if use_rtabmap_pose
                else "IMU+VO" if imu_poses is not None else "VO"
            )
            if walking_lite:
                print(f"[timing] pose computation ({len(frames_rgb)} frames, {_pose_src_label}): {_pose_ms:.1f} ms")

            # World-space output remap (see process_frames_batch docstring) —
            # applied after all pose sources/anchoring above, before anything
            # downstream (back-projection, occupancy map, semantic mapper,
            # exposed trajectory/cam_pos) touches raw_poses.
            raw_poses = _remap_poses(raw_poses, axis_perm)

            # Pure-walking total-tracking-loss reset (see the streak
            # bookkeeping in Step 2's RTAB-Map branch above) — fires here,
            # right after Step 2 finishes and before Step 3 would otherwise
            # back-project THIS frame's (untracked, garbage) pose into
            # anything. _reset_cloud_locked() is reset_cloud()'s body minus
            # the self._lock acquisition (we're already holding it) — see
            # that method's own docstring for why reset_cloud() itself
            # can't be called from here. Unlike reset_cloud()'s usual
            # convention, self.rtabmap_client.reset() is called here
            # WITHOUT releasing self._lock first — restructuring this
            # method to drop the lock mid-body for one rare branch wasn't
            # worth it; this only holds the lock slightly longer during a
            # network round trip on an already-rare event (total tracking
            # loss), not a hot path.
            if _lost_track_should_reset:
                _mode_name = "pure_walking" if pure_walking else "guiding"
                _threshold_used = PURE_WALKING_LOST_RESET_S if pure_walking else GUIDING_LOST_RESET_S
                print(
                    f"[ScanSession:{self.location_id}] {_mode_name}: RTAB-Map tracking lost "
                    f"for >= {_threshold_used:.1f}s with no recovery — resetting "
                    f"session to a fresh state."
                )
                self._reset_cloud_locked()
                if self.rtabmap_client is not None:
                    self.rtabmap_client.reset()
                self.last_reset_occurred = True
                return self._raw_point_count, [0.0, 0.0, 0.0], infer_ms

            # ── Step 3: Dense back-projection + voxel fusion ──────────────────
            # RTAB-Map mode does NOT back-project here at all — its own
            # reconstructed surface is pulled separately below (Step 3b, via
            # rtabmap_client.get_cloud()), sourced from RTAB-Map's own
            # cloudRGBFromSensorData + its CURRENT graph-corrected poses
            # rather than this project's DA3-depth back-projection. Eager
            # tag+detect+backproject below still runs for every pose
            # source — independent of which cloud-reconstruction path is
            # active (see _PendingTagFrame/SemanticMapper.tag_and_backproject_batch).
            new_cloud = o3d.geometry.PointCloud()
            _bp_counts = []
            _t0_bp = time.perf_counter()

            _ts_slice = (
                frame_timestamps_ns if frame_timestamps_ns is not None
                else [None] * len(frames_rgb)
            )
            for rgb, df, pose, ts_ns, depth_check, novelty_flag in zip(
                frames_rgb, depth_frames, raw_poses, _ts_slice, depth_checks, novelty_flags
            ):
                accepted, sharpness, reject_reason = novelty_flag

                # Local back-projection: normally IMU+VO/VO only — RTAB-Map mode's
                # own server-side reconstruction (Step 3b below) replaces this. But
                # walking_lite explicitly wants OUT of that heavier path (get_cloud/
                # SOR/server-side-voxelize measured 15s+/12s+ on a real session) and
                # back-projects locally here too, same as IMU+VO/VO, using RTAB-Map's
                # pose (still authoritative) + this frame's own already-computed DA3
                # depth. See process_frames_batch's walking_lite docstring note.
                #
                # Fusion itself is unconditional — neither the depth-consistency
                # check (depth_check) nor the novelty+blur gate (accepted) blocks
                # it. Both checks still run (needed elsewhere: depth_check backs
                # the RTAB-Map untrusted-node veto below, accepted still gates
                # frame-store/VLM-tagging admission just below this loop, which is
                # the gate's actual purpose — deciding what the semantic mapper
                # sees, not what gets fused into the occupancy grid), they just no
                # longer withhold a frame's own geometry from the live map:
                # occupancy_map.py's Bayesian scheme is already self-correcting
                # against occasional bad frames, and this project doesn't need
                # scan-grade accuracy enough to pay for dropping otherwise-good
                # pose/geometry over it.
                if not use_rtabmap_pose or walking_lite:
                    result = _back_project_frame(rgb, df, pose, max_depth=10.0)
                    if result is None:
                        _bp_counts.append(0)
                    else:
                        pts, cols = result
                        _bp_counts.append(len(pts))
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        pcd.colors = o3d.utility.Vector3dVector(cols)
                        new_cloud += pcd

                    # TSDF fusion (denoised Live Points/PLY) — non-RTAB-Map
                    # sessions only, unchanged reasoning (see __init__'s
                    # _tsdf_volume comment: RTAB-Map's own reconstruction
                    # already gets corrected on loop closure, a client-side
                    # TSDF integrated at pose-at-time would reintroduce that
                    # staleness). walking_lite doesn't pull get_cloud() at
                    # all, but has no need for PLY-quality denoising either
                    # — a live occupancy grid tolerates raw back-projected
                    # points fine (occupancy_map.py's Bayesian scheme is
                    # explicitly self-correcting), so this stays off for it too.
                    if not use_rtabmap_pose:
                        h_f, w_f = df.depth_map.shape[:2]
                        K_f = df.intrinsics if df.intrinsics is not None else _estimate_K(h_f, w_f)
                        intrinsic = o3d.camera.PinholeCameraIntrinsic(
                            w_f, h_f, float(K_f[0, 0]), float(K_f[1, 1]),
                            float(K_f[0, 2]), float(K_f[1, 2]),
                        )
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            o3d.geometry.Image(np.ascontiguousarray(rgb)),
                            o3d.geometry.Image(np.ascontiguousarray(df.depth_map, dtype=np.float32)),
                            depth_scale=1.0, depth_trunc=10.0,
                            convert_rgb_to_intensity=False,
                        )
                        self._tsdf_volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
                        self._tsdf_integrated_count += 1

                # Eager tag+detect+backproject — only novelty+blur-gated
                # frames are buffered for it (see _PendingTagFrame's
                # docstring and semantic_mapper.py's module docstring for
                # why GroundingDINO-tiny now runs immediately, batched, per
                # accepted frame, rather than deferred to an on-demand
                # query). No dependency on use_rtabmap_pose — this runs for
                # every pose source identically. walking_lite is the one
                # exception: semantic tagging is an explicit-scan-only
                # concept (see CLAUDE.md's walking-mode note) — walking/
                # guiding rely on FindLandmark's persisted-snapshot fallback
                # for whatever a prior scan already tagged, rather than
                # re-discovering landmarks live.
                if accepted and self.semantic_mapper is not None and not walking_lite:
                    K_eff = (
                        df.intrinsics if df.intrinsics is not None
                        else _estimate_K(*rgb.shape[:2])
                    )
                    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    self._tag_pending.append(_PendingTagFrame(
                        frame_bgr=frame_bgr,
                        depth_map=df.depth_map,
                        world_pose=pose,
                        K=K_eff,
                        frame_idx=self._frame_count,
                    ))
                    if len(self._tag_pending) >= SemanticMapper.IMAGES_PER_PROMPT:
                        self._flush_tag_pending()

            _bp_ms = (time.perf_counter() - _t0_bp) * 1000
            if walking_lite:
                print(
                    f"[timing] back-projection + frame-store tagging "
                    f"({len(frames_rgb)} frames, {sum(_bp_counts):,} raw pts): {_bp_ms:.1f} ms"
                )

            self.last_pose_source = (
                "RTAB-Map" if use_rtabmap_pose
                else "IMU+VO" if imu_poses is not None
                else "VO"
            )

            # ── Step 3b: RTAB-Map's own surface reconstruction ─────────────────
            # Pulls whatever RTAB-Map has newly reconstructed since the last
            # pull (or, after a loop closure, EVERYTHING — see
            # _rtabmap_full_resync) via rtabmap_client.get_cloud(): each node's
            # own cloudRGBFromSensorData reconstruction, voxelized server-side,
            # transformed by RTAB-Map's CURRENT graph-corrected pose. This is
            # the "reconstruct the surface the RTAB-Map way" path — it
            # populates self._raw_cloud_batches / self.occupancy_map exactly
            # like Step 3/Step 4 do for the other pose sources, just sourced
            # differently, one node at a time so the Occupancy Map's ray
            # casting keeps its "one representative camera position per call"
            # invariant (see occupancy_map.py's sunburst-artifact note).
            # Skipped entirely for walking_lite — Step 3 above already fed the
            # occupancy map via local back-projection, and this pull (get_cloud
            # + SOR + server-side voxelize) is the single most expensive part
            # of a batch for exactly the quality (clean, persisted, loop-
            # closure-correctable reconstruction) walking/guiding don't need.
            if use_rtabmap_pose and self.rtabmap_client is not None and not walking_lite:
                if self.last_rtabmap_loop_closure:
                    with timed("RTAB-Map full resync (get_cloud since=0)"):
                        self._rtabmap_full_resync(sor_nb_neighbors, sor_std_ratio, occupancy_voxel_size)
                else:
                    with timed("RTAB-Map incremental pull (get_cloud)"):
                        self._rtabmap_pull_new_nodes(sor_nb_neighbors, sor_std_ratio, occupancy_voxel_size)

            # Per-batch cleanup (voxel downsample + outlier removal), computed
            # ONCE per batch and reused for BOTH the Occupancy Map update
            # below and the lazily-rebuilt full point cloud (see
            # ensure_cloud_built()). self._cloud itself is NOT accumulated/
            # re-cleaned here anymore — see its field comment in __init__ for
            # why (this used to re-run voxel_down_sample +
            # remove_statistical_outlier over the ENTIRE ever-growing cloud
            # on every batch, which neither the Occupancy Map nor an unwatched
            # Live Points/Voxelization tab ever needed).
            new_cloud_clean = new_cloud
            if len(new_cloud_clean.points) > 0:
                new_cloud_clean = _voxel_down_sample_accel(new_cloud_clean, VOXEL_SIZE)
                if len(new_cloud_clean.points) > sor_nb_neighbors:
                    # Statistical Outlier Removal — strip points whose average
                    # distance to their nb_neighbors nearest neighbors is more
                    # than std_ratio standard deviations above the cloud's mean
                    # (adaptive to local density, no fixed distance threshold).
                    # nb_neighbors/std_ratio are GUI-adjustable (Outlier Removal
                    # accordion) — below nb_neighbors+1 points, neighbor stats
                    # aren't meaningful, so skip rather than error.
                    new_cloud_clean = _remove_statistical_outlier_accel(
                        new_cloud_clean, sor_nb_neighbors, sor_std_ratio
                    )
                self._raw_cloud_batches.append(new_cloud_clean)
                self._raw_point_count += len(new_cloud_clean.points)
            self._last_sor_nb_neighbors = sor_nb_neighbors
            self._last_sor_std_ratio = sor_std_ratio

            # ── Step 4: track trajectory + progressively update the Occupancy Map ──
            # The Occupancy Map builds up live, one batch at a time — see
            # OccupancyMap.update()'s docstring for why this is safe (cumulative
            # ground-plane estimate) — same as a SLAM system's occupancy grid
            # filling in progressively as it explores, rather than waiting for
            # a finished scan. It only ever needs this batch's own cleaned
            # points, never the full accumulated cloud. Fed directly from
            # voxelize_cloud() at occupancy_voxel_size, run on just this
            # batch's own new points, then merged into self._voxel_dict via
            # _merge_voxels() — the SAME accumulator scan_gui.py's
            # Voxelization view reads directly (see _merge_voxels' docstring)
            # — there is exactly one voxelization computed here, not a
            # separate approximation for display.
            traj = np.array([p[:3, 3] for p in raw_poses], dtype=np.float32)
            self.last_trajectory = traj
            self._all_trajectory.append(traj)

            # pure_walking (SessionMode.WALKING) ALSO gets a live occupancy
            # grid now, same as GUIDING — confirmed with the user: walking's
            # HRTF beacon steers along an actual planned path through this
            # grid (LocalPathPlanner, client-side) toward a synthetic
            # "straight ahead" target, not a single per-frame direction
            # pick. Never persisted (see the pure_walking finally-block
            # skip in mapping_servicer.py) and reset far more aggressively
            # on tracking loss (PURE_WALKING_LOST_RESET_S) than a scan/
            # guiding session would want — see CLAUDE.md's walking-mode
            # local-map note.
            if len(new_cloud_clean.points) > 0:
                occ_centers, occ_colors, occ_vsize = voxelize_cloud(
                    new_cloud_clean, voxel_size=occupancy_voxel_size, max_voxels=_NO_COARSEN_MAX_VOXELS
                )
                if len(occ_centers) > 0:
                    # Batch-level depth confidence (see occupancy_map.py's
                    # update() docstring) — mean of (1 - frac_bad) across
                    # this batch's own frames (depth_checks, from Step 2),
                    # skipping frames that couldn't be checked (too few PnP
                    # inliers) rather than penalizing missing data. Defaults
                    # to full confidence if NO frame had a usable check.
                    # Still computed (and still surfaced via
                    # last_batch_confidence, e.g. server_gui.py's dashboard
                    # text) for pure_walking too — only the WEIGHT actually
                    # fed into the occupancy grid is forced to full (1.0)
                    # for pure_walking, confirmed with the user: walking
                    # needs an immediately-usable grid for navigation, not
                    # scan-grade caution about a momentarily-uncertain
                    # frame — a real obstacle should register at full
                    # strength right away rather than needing several
                    # confirming hits to reach the same belief a
                    # full-confidence hit would in one.
                    _confs = [1.0 - fb for (_, fb, _) in depth_checks if fb is not None]
                    batch_confidence = sum(_confs) / len(_confs) if _confs else 1.0
                    self.last_batch_confidence = batch_confidence
                    update_confidence = 1.0 if pure_walking else batch_confidence
                    with timed(f"occupancy_map.update ({len(occ_centers)} voxel centers)"):
                        self.occupancy_map.update(traj, occ_centers, confidence=update_confidence)
                    self._merge_voxels(occ_centers, occ_colors, occ_vsize)

            self.last_frames_rgb = list(frames_rgb)
            self.last_depth_frames = list(depth_frames)
            self.last_frame_poses = list(raw_poses)

            cam_pos = raw_poses[-1][:3, 3].tolist() if raw_poses else [0.0, 0.0, 0.0]
            return self._raw_point_count, cam_pos, infer_ms

    def _rtabmap_process_nodes(
        self, nodes, sor_nb_neighbors: int, sor_std_ratio: float,
        occupancy_voxel_size: float,
    ) -> None:
        """
        Shared by _rtabmap_pull_new_nodes (incremental) and
        _rtabmap_full_resync (loop-closure) — both just fetch a different
        `nodes` list and delegate here.

        SOR is BATCHED across every node into ONE combined cloud + ONE
        remove_statistical_outlier call, instead of one small call per node.
        Each node's own cloud (a few thousand-tens of thousands of points,
        per real RTAB-Map sessions) individually stayed below
        _GPU_ACCEL_MIN_POINTS, so a resync with dozens of nodes did that many
        separate CPU-only SOR calls whose cost added up — batched, the
        combined cloud is large enough to actually engage the GPU path, and
        the fixed per-call overhead (Python loop, Open3D call dispatch) is
        paid once instead of per node. The combined SOR's keep-mask is then
        sliced back into per-node segments (see
        _remove_statistical_outlier_accel_masked) so each node still gets
        its own cleaned cloud — this only changes how the CLEANING step is
        batched, not the outlier statistics' spatial scope in any way that
        matters in practice (nodes are spatially distinct camera positions,
        so a global KNN search still only finds each point's own nearby
        neighbors, same as a per-node search would).

        voxelize_cloud() + occupancy_map.update() stay PER NODE, not
        batched — the Occupancy Map's ray casting needs one call per
        representative camera position; batching those into one call for
        many nodes at once would reproduce the sunburst artifact documented
        in occupancy_map.py.

        Nodes flagged untrustworthy (self._rtabmap_untrusted_node_ids — see
        process_frames_batch's RTAB-Map branch) are dropped before fusion.
        _rtabmap_last_pulled_node_id still advances past them (computed from
        the full, unfiltered `nodes` list below) so a dropped node is never
        re-requested on every subsequent incremental pull.
        """
        if nodes:
            self._rtabmap_last_pulled_node_id = max(
                self._rtabmap_last_pulled_node_id, max(n.node_id for n in nodes)
            )
        if self._rtabmap_untrusted_node_ids:
            _kept = [n for n in nodes if n.node_id not in self._rtabmap_untrusted_node_ids]
            _n_dropped = len(nodes) - len(_kept)
            if _n_dropped:
                print(
                    f"[depth-consistency] dropped {_n_dropped}/{len(nodes)} RTAB-Map "
                    f"node(s) previously flagged untrustworthy — not fused into map."
                )
            nodes = _kept
        if not nodes:
            return

        # Split into nodes whose SOR keep-mask is already cached (from an
        # earlier pull of this same node_id — see self._rtabmap_sor_keep_
        # mask's field comment for why a pose-only change never invalidates
        # it) vs nodes that need SOR computed fresh. Only the latter pay the
        # SOR cost; a full resync of a mostly-already-seen map is the common
        # case this optimizes (measured: ~7.8s -> near-zero for an all-cached
        # resync).
        # ground_by_id mirrors cleaned_by_id exactly — each node's own
        # is_ground array (RTAB-Map's native ground/obstacle segmentation,
        # see ReconstructedNode.is_ground), masked by the SAME SOR keep-mask
        # applied to that node's points/colors below, so the three arrays
        # stay index-aligned all the way through to voxelize_cloud().
        cleaned_by_id: Dict[int, o3d.geometry.PointCloud] = {}
        ground_by_id: Dict[int, np.ndarray] = {}
        uncached_nodes = []
        for node in nodes:
            cached_mask = self._rtabmap_sor_keep_mask.get(node.node_id)
            if cached_mask is not None and len(cached_mask) == len(node.points):
                nc = o3d.geometry.PointCloud()
                if cached_mask.any():
                    nc.points = o3d.utility.Vector3dVector(
                        node.points[cached_mask].astype(np.float64)
                    )
                    nc.colors = o3d.utility.Vector3dVector(
                        node.colors[cached_mask].astype(np.float64) / 255.0
                    )
                    ground_by_id[node.node_id] = node.is_ground[cached_mask]
                else:
                    ground_by_id[node.node_id] = np.zeros(0, dtype=bool)
                cleaned_by_id[node.node_id] = nc
            else:
                uncached_nodes.append(node)

        if uncached_nodes:
            raw_clouds: List[o3d.geometry.PointCloud] = []
            for node in uncached_nodes:
                c = o3d.geometry.PointCloud()
                if len(node.points) > 0:
                    c.points = o3d.utility.Vector3dVector(node.points.astype(np.float64))
                    c.colors = o3d.utility.Vector3dVector(node.colors.astype(np.float64) / 255.0)
                raw_clouds.append(c)

            sizes = [len(c.points) for c in raw_clouds]
            total = sum(sizes)
            if total > sor_nb_neighbors:
                with timed(f"RTAB-Map batched cloud concat ({total} pts, {len(uncached_nodes)} nodes)"):
                    combined = o3d.geometry.PointCloud()
                    for c in raw_clouds:
                        combined += c
                _, keep_mask = _remove_statistical_outlier_accel_masked(
                    combined, sor_nb_neighbors, sor_std_ratio
                )
                combined_pts = np.asarray(combined.points)
                combined_cols = np.asarray(combined.colors)
                combined_ground = np.concatenate(
                    [node.is_ground for node in uncached_nodes]
                ) if any(len(node.is_ground) for node in uncached_nodes) else np.zeros(0, dtype=bool)
                offset = 0
                for node, size in zip(uncached_nodes, sizes):
                    node_mask = keep_mask[offset:offset + size]
                    seg = slice(offset, offset + size)
                    offset += size
                    self._rtabmap_sor_keep_mask[node.node_id] = node_mask
                    nc = o3d.geometry.PointCloud()
                    if node_mask.any():
                        nc.points = o3d.utility.Vector3dVector(combined_pts[seg][node_mask])
                        nc.colors = o3d.utility.Vector3dVector(combined_cols[seg][node_mask])
                        ground_by_id[node.node_id] = combined_ground[seg][node_mask]
                    else:
                        ground_by_id[node.node_id] = np.zeros(0, dtype=bool)
                    cleaned_by_id[node.node_id] = nc
            else:
                # Too few points for SOR to be meaningful — keep as-is, and
                # still cache an all-True mask so a later resync of this
                # same node (once it's part of a large-enough batch) is
                # consistent rather than silently re-deciding differently.
                for node, c in zip(uncached_nodes, raw_clouds):
                    self._rtabmap_sor_keep_mask[node.node_id] = np.ones(len(node.points), dtype=bool)
                    cleaned_by_id[node.node_id] = c
                    ground_by_id[node.node_id] = node.is_ground

        cleaned_clouds = [cleaned_by_id[node.node_id] for node in nodes]
        cleaned_grounds = [ground_by_id[node.node_id] for node in nodes]

        _voxelize_ms = 0.0
        _occ_update_ms = 0.0
        for node, cloud, node_ground in zip(nodes, cleaned_clouds, cleaned_grounds):
            if len(cloud.points) > 0:
                self._raw_cloud_batches.append(cloud)
                self._raw_point_count += len(cloud.points)
                traj = node.pose[:3, 3].reshape(1, 3).astype(np.float32)
                _t0 = time.perf_counter()
                occ_centers, occ_colors, occ_vsize = voxelize_cloud(
                    cloud, voxel_size=occupancy_voxel_size, max_voxels=_NO_COARSEN_MAX_VOXELS
                )
                _voxelize_ms += (time.perf_counter() - _t0) * 1000
                if len(occ_centers) > 0:
                    _t0 = time.perf_counter()
                    node_confidence = self._rtabmap_node_confidence.get(node.node_id, 1.0)
                    self.last_batch_confidence = node_confidence
                    occ_is_ground = None
                    if len(node_ground) == len(np.asarray(cloud.points)):
                        occ_is_ground = _voxel_majority_flags(
                            np.asarray(cloud.points), node_ground, occ_centers, occ_vsize
                        )
                    self.occupancy_map.update(
                        traj, occ_centers, confidence=node_confidence,
                        point_is_ground=occ_is_ground,
                    )
                    _occ_update_ms += (time.perf_counter() - _t0) * 1000
                    self._merge_voxels(occ_centers, occ_colors, occ_vsize)
        print(
            f"[timing] RTAB-Map per-node voxelize_cloud x{len(nodes)} nodes "
            f"(excl. its own internal timing above): {_voxelize_ms:.1f} ms"
        )
        print(f"[timing] RTAB-Map per-node occupancy_map.update x{len(nodes)} nodes: {_occ_update_ms:.1f} ms")

    def _rtabmap_pull_new_nodes(
        self, sor_nb_neighbors: int, sor_std_ratio: float,
        occupancy_voxel_size: float = DEFAULT_VOXEL_SIZE,
    ) -> None:
        """Incremental path: pull only nodes reconstructed since the last
        pull (self._rtabmap_last_pulled_node_id), append each to
        self._raw_cloud_batches, and feed the Occupancy Map one node at a
        time (that node's own pose as a 1-point trajectory + its own cloud,
        run through voxelize_cloud() at occupancy_voxel_size — the Occupancy
        Map is fully inferred from the voxelization, see process_frames_batch's
        docstring) — same per-call granularity Step 4 uses for the other pose
        sources."""
        nodes = self.rtabmap_client.get_cloud(
            since_node_id=self._rtabmap_last_pulled_node_id,
            voxel_size=VOXEL_SIZE, max_depth=10.0,
        )
        if not nodes:
            return
        self._rtabmap_process_nodes(nodes, sor_nb_neighbors, sor_std_ratio, occupancy_voxel_size)

    def _rtabmap_full_resync(
        self, sor_nb_neighbors: int, sor_std_ratio: float,
        occupancy_voxel_size: float = DEFAULT_VOXEL_SIZE,
    ) -> None:
        """Loop-closure path: a TRACK reply this batch reported a loop
        closure, meaning RTAB-Map's graph-corrected poses for PREVIOUSLY
        pulled nodes may have just shifted — every node this session has
        already contributed to self._raw_cloud_batches/self.occupancy_map is
        now potentially stale (built at a pose that's since been corrected).
        Discards all of it and rebuilds from a fresh since_node_id=0 pull,
        replaying occupancy_map.update() once per node in ascending id order
        (never one call for the whole history at once — see
        occupancy_map.py's documented sunburst-artifact bug from doing
        exactly that)."""
        nodes = self.rtabmap_client.get_cloud(
            since_node_id=0, voxel_size=VOXEL_SIZE, max_depth=10.0,
        )
        if nodes is None:
            print(f"[ScanSession:{self.location_id}] RTAB-Map loop-closure resync: "
                  f"get_cloud() request failed — will retry next batch.")
            return
        self._raw_cloud_batches = []
        self._raw_point_count = 0
        self._merged_batch_count = 0
        self._cloud_raw_accum = o3d.geometry.PointCloud()
        self.occupancy_map.reset()
        self._voxel_dict = {}
        self.last_voxel_centers = np.zeros((0, 3), dtype=np.float32)
        self.last_voxel_colors = None
        self._rtabmap_last_pulled_node_id = 0
        self._rtabmap_process_nodes(nodes, sor_nb_neighbors, sor_std_ratio, occupancy_voxel_size)
        print(f"[ScanSession:{self.location_id}] RTAB-Map loop closure — resynced "
              f"{len(nodes)} nodes, {self._raw_point_count} points.")

    def _merge_voxels(self, centers: np.ndarray, colors: Optional[np.ndarray], vsize: float) -> None:
        """
        The ONE place self.last_voxel_centers/last_voxel_colors/last_voxel_size
        get written — merges a newly-voxelized batch/node's centers into
        self._voxel_dict (keyed by grid index at vsize, via voxelize_cloud's
        fixed grid anchor — see _VOXEL_GRID_MIN_BOUND/_MAX_BOUND — so the same
        physical voxel observed by two different batches maps to the same
        key and updates in place rather than duplicating). Called from the
        Occupancy Map feed in process_frames_batch/_rtabmap_process_nodes
        (incremental, per new batch/node) and from scan_gui.py's on-demand
        "Voxelize" button (a full recompute over the whole cloud, typically
        at a different voxel_size) — either way, this is the only path that
        updates the accumulator, so there is exactly one voxelization the
        Occupancy Map and the Voxelization tab both read from.

        A voxel_size change resets the accumulator first — indices computed
        at one size aren't meaningful at another, so merging across a size
        change would produce a nonsensical mix of two different grids.
        """
        if abs(vsize - self.last_voxel_size) > 1e-9:
            self._voxel_dict = {}
            self.last_voxel_size = vsize
        for i in range(len(centers)):
            key = tuple(np.round(centers[i] / vsize).astype(np.int64))
            color = colors[i] if colors is not None else None
            self._voxel_dict[key] = (centers[i], color)
        if not self._voxel_dict:
            self.last_voxel_centers = np.zeros((0, 3), dtype=np.float32)
            self.last_voxel_colors = None
            return
        self.last_voxel_centers = np.array([v[0] for v in self._voxel_dict.values()], dtype=np.float32)
        if colors is not None:
            self.last_voxel_colors = np.array(
                [v[1] if v[1] is not None else (0.6, 0.6, 0.6) for v in self._voxel_dict.values()],
                dtype=np.float32,
            )

    def ensure_cloud_built(self) -> o3d.geometry.PointCloud:
        """
        For non-RTAB-Map sessions with any TSDF-integrated frames: returns
        the TSDF-fused cloud (self._tsdf_volume.extract_point_cloud()),
        re-extracted only when new frames were integrated since the last
        call (self._tsdf_extracted_at_count) — denoised vs. the raw
        first-observation-wins accumulation below, see __init__'s
        _tsdf_volume comment for why. RTAB-Map sessions (no TSDF volume
        populated — see that comment) fall through to the original path.

        Original path — lazily merge every batch's already-cleaned
        (voxelized + outlier-removed, see process_frames_batch's Step 3)
        point cloud into self._cloud. Historically only called on demand
        (Reload on the Live Points tab, Voxelize, Export,
        finalize_voxel_and_occupancy); now also called once per processed
        chunk during Scan/Simulated Live Stream when the "Live Points" GUI
        checkbox is on (see scan_gui.py), so this needs to stay cheap on
        repeat calls with only a little new data each time — see
        self._merged_batch_count's field comment for how.

        Each batch was already voxel-downsampled + outlier-removed at batch
        time, so this just needs one more voxel pass to merge adjacent
        batches' voxels into a single consistent grid, not a full outlier-
        removal re-run over everything. That final voxel_down_sample pass is
        still O(current total accumulated size) every call — Open3D has no
        incremental voxel-merge primitive, so this can't be made fully O(new
        data) — the GUI checkbox is the actual cost control for a large scan,
        not this method. (TSDF extraction has the same O(total volume size)
        per-call cost, for the same reason — no incremental extraction
        primitive either.)
        """
        with self._lock:
            if not self._session_uses_rtabmap and self._tsdf_integrated_count > 0:
                if self._tsdf_extracted_at_count != self._tsdf_integrated_count:
                    self._tsdf_cloud_cache = self._tsdf_volume.extract_point_cloud()
                    self._tsdf_extracted_at_count = self._tsdf_integrated_count
                return self._tsdf_cloud_cache

            new_batches = self._raw_cloud_batches[self._merged_batch_count:]
            if not new_batches:
                return self._cloud
            for c in new_batches:
                self._cloud_raw_accum += c
            self._merged_batch_count = len(self._raw_cloud_batches)
            if len(self._cloud_raw_accum.points) > 0:
                self._cloud = _voxel_down_sample_accel(self._cloud_raw_accum, VOXEL_SIZE)
            return self._cloud

    def finalize_voxel_and_occupancy(self) -> None:
        """
        Run once after every batch/segment for this Scan click has already
        been processed.

        Does NOT touch the Occupancy Map anymore — that now builds up
        progressively, once per batch, inside process_frames_batch (see
        OccupancyMap.update()'s docstring). Calling occupancy_map.reset() +
        update() here once, with the ENTIRE session's concatenated
        trajectory and full voxel cloud, actively broke the ray-casting free-
        space fill: OccupancyMap.update() picks a single representative
        camera position per call (trajectory[-1], a reasonable approximation
        within one small batch) and ray-casts every touched cell from that
        one point — fine per-batch, but with the whole session passed in at
        once this cast thousands of rays from a single final camera position
        to every cell in the map, producing a sunburst/hatch artifact instead
        of a clean grid. The progressive per-batch updates already build a
        correct grid on their own; this method no longer needs to touch it.

        Does NOT re-voxelize either, anymore — self.last_voxel_centers/
        colors/voxel_size are already the single, incrementally-accumulated
        result of every batch's own voxelize_cloud() call (see
        process_frames_batch/_rtabmap_process_nodes + _merge_voxels); a
        second whole-cloud voxelize_cloud() call here would just be a second,
        independently-computed approximation of the same data — exactly what
        _merge_voxels exists to avoid. Only ensure_cloud_built() runs, since
        Live Points/Export still need self._cloud built up.
        """
        self.ensure_cloud_built()

    def reset_cloud(self) -> None:
        """Clear accumulated point cloud, poses, and occupancy map. Keeps zone labels and IMU."""
        with self._lock:
            self._reset_cloud_locked()
        if self.rtabmap_client is not None:
            self.rtabmap_client.reset()
        print(f"[ScanSession:{self.location_id}] Cloud cleared.")

    def _reset_cloud_locked(self) -> None:
        """Body of reset_cloud() minus self._lock acquisition and the
        rtabmap_client.reset() wire call — for a caller that already holds
        self._lock (process_frames_batch's pure_walking total-tracking-loss
        branch). threading.Lock() isn't reentrant, so reset_cloud() itself
        can't be called from inside process_frames_batch's own `with
        self._lock:` block without deadlocking; that caller is responsible
        for calling self.rtabmap_client.reset() itself afterward, same as
        reset_cloud() does."""
        self._cloud = o3d.geometry.PointCloud()
        self._raw_cloud_batches = []
        self._raw_point_count = 0
        self._merged_batch_count = 0
        self._cloud_raw_accum = o3d.geometry.PointCloud()
        self._tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=VOXEL_SIZE, sdf_trunc=VOXEL_SIZE * 4,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        self._tsdf_integrated_count = 0
        self._tsdf_cloud_cache = None
        self._tsdf_extracted_at_count = -1
        self._session_uses_rtabmap = False
        self.pose_graph = PoseGraph()
        self.occupancy_map = OccupancyMap(resolution=0.05, **self._occupancy_params)
        self.tracker.reset()
        self.last_frame_poses = []
        self.last_frames_rgb = []
        self.last_depth_frames = []
        self.last_trajectory = np.zeros((0, 3), dtype=np.float32)
        self._voxel_dict = {}
        self.last_voxel_centers = np.zeros((0, 3), dtype=np.float32)
        self.last_voxel_colors = None
        self.last_voxel_size = DEFAULT_VOXEL_SIZE
        self._all_trajectory = []
        self._frame_count = 0
        self._rtabmap_last_pose = None
        self._rtabmap_last_pulled_node_id = 0
        self._rtabmap_untrusted_node_ids = set()
        self._rtabmap_sor_keep_mask = {}
        self._rtabmap_node_confidence = {}
        self.last_rtabmap_loop_closure = False
        self._raw_landmarks = []
        self.novelty_gate = OrbNoveltyGate(**self._gate_ctor_kwargs())
        self._tag_pending = []
        self._walking_lost_streak_start = None

    def configure_occupancy_map(self, **kwargs) -> None:
        """
        Apply new Bayesian/height tuning knobs (see OccupancyMap.set_params
        for accepted keys) to the CURRENT occupancy map in place — keeps
        accumulated cell beliefs, only affects future update() calls — and
        remembers them for the next reset_cloud()/fresh session so they
        aren't silently lost. Called from scan_gui.py's "Occupancy Map
        Settings" accordion.
        """
        self._occupancy_params.update({k: v for k, v in kwargs.items() if v is not None})
        self.occupancy_map.set_params(**kwargs)

    def configure_novelty_gate(self, **kwargs) -> None:
        """
        Apply new novelty/blur-gate tuning knobs (min_sharpness/
        min_new_fraction/min_new_count/min_rotation_deg, plus OrbNoveltyGate
        constructor knobs n_features/ratio/min_raw_matches/
        ransac_threshold_px/use_gpu) and remember them for the next
        reset_cloud()/fresh session, same pattern as configure_occupancy_map.

        One real deviation from that pattern, worth noting: OpenCV gives no
        way to reconfigure cv2.ORB_create's nfeatures post-construction, so
        changing a constructor-relevant knob recreates self.novelty_gate —
        which also discards its accumulated reference-frame set (unlike
        OccupancyMap.set_params(), which mutates in place with no such side
        effect).
        """
        clean = {k: v for k, v in kwargs.items() if v is not None}
        self._novelty_params.update(clean)
        self._min_sharpness = self._novelty_params.get("min_sharpness", self._min_sharpness)
        self._min_new_fraction = self._novelty_params.get("min_new_fraction", self._min_new_fraction)
        self._min_new_count = self._novelty_params.get("min_new_count", self._min_new_count)
        self._min_rotation_deg = self._novelty_params.get("min_rotation_deg", self._min_rotation_deg)
        ctor_keys = ("n_features", "ratio", "min_raw_matches", "ransac_threshold_px", "use_gpu")
        if any(k in clean for k in ctor_keys):
            self.novelty_gate = OrbNoveltyGate(**self._gate_ctor_kwargs())

    def set_label(self, label: str, radius: float = 1.5) -> Zone:
        """Creates a Zone AABB of the given radius around the current camera position."""
        with self._lock:
            pos = self.tracker._world_pose[:3, 3]
            bbox_min = (pos - radius).tolist()
            bbox_max = (pos + radius).tolist()
            zone = Zone(label=label, bbox_min=bbox_min, bbox_max=bbox_max)
            self.labeler.zones.append(zone)
        print(f"[ScanSession:{self.location_id}] Zone '{label}' at {pos.tolist()}")
        return zone

    def set_label_from_positions(
        self, label: str, positions, margin: float = 1.5
    ) -> Zone:
        """
        Create a zone AABB covering all camera positions visited during a segment,
        expanded outward by `margin` metres in each axis. Landmarks are assigned
        later, once for the whole session, by finalize_landmarks() at export time
        (with a nearest-zone fallback for landmarks outside every AABB — real
        furniture is often several metres from the walked path, so margin alone
        isn't a full fix, just reduces how often that fallback has to fire and
        makes multi-zone misassignment less likely).
        """
        with self._lock:
            arr = np.array(positions, dtype=np.float32)
            bbox_min = (arr.min(axis=0) - margin).tolist()
            bbox_max = (arr.max(axis=0) + margin).tolist()
            zone = Zone(label=label, bbox_min=bbox_min, bbox_max=bbox_max)
            self.labeler.zones.append(zone)

        print(f"[ScanSession:{self.location_id}] Zone '{label}' from path: {bbox_min} → {bbox_max}")
        return zone

    def resolve_landmark(self, query: str) -> Optional[Landmark]:
        """
        Landmark lookup by name — GroundingDINO-tiny now runs immediately on
        every accepted frame (see _flush_tag_pending / semantic_mapper.py's
        frame_extractor-adapted pipeline), so there's no on-demand detection
        left to defer here. This just searches whatever's already been
        resolved into self._raw_landmarks (case-insensitive substring match,
        either direction — same matching convention the old deferred design
        used), returning the highest-confidence hit. Returns None if nothing
        matches, or if this session has no semantic_mapper configured.

        Real, accepted trade-off vs. the old deferred design: a landmark
        Gemini never tagged in ANY accepted frame is never found here at all
        (there's no more open-vocabulary "scan every frame for this exact
        query" fallback) — see semantic_mapper.py's module docstring.
        """
        if self.semantic_mapper is None:
            return None
        query_norm = query.strip().lower()
        if not query_norm:
            return None
        with self._lock:
            candidates = [
                lm for lm in self._raw_landmarks
                if query_norm in lm.name.lower() or lm.name.lower() in query_norm
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda lm: lm.confidence)

    def _finalize_raw_landmarks(self) -> List[Landmark]:
        """
        Finalize-time export helper, shared by finalize_landmarks()/
        finalize_landmarks_flat(): flushes any leftover partial tag+detect
        batch, then runs a final cluster_landmarks() pass over everything
        resolved so far (self._raw_landmarks) — landmarks are already
        resolved live now (see _flush_tag_pending), so this only needs to
        catch near-duplicate resolutions from synonym tags (e.g. "chair" vs
        "office chair" landing at nearly the same spot), not run any new
        detection. Must NOT be called while the caller already holds
        self._lock — this acquires it internally.
        """
        if self.semantic_mapper is None:
            return []
        with self._lock:
            self._flush_tag_pending()
            raw_snapshot = list(self._raw_landmarks)
        if not raw_snapshot:
            return []
        return self.semantic_mapper.cluster_landmarks(raw_snapshot)

    def finalize_landmarks(self) -> None:
        """
        Run once, at export time: cluster every landmark resolved so far
        across the whole session (see _finalize_raw_landmarks), then assign
        each merged Landmark to a Zone — first choice is whichever Zone's
        X-Z AABB
        footprint contains its centroid, but zone AABBs are built from the
        camera's walked path (+ a small margin, see set_label_from_positions)
        which is often much smaller than the room itself, so a real,
        correctly-detected landmark (e.g. furniture against a wall, a few
        metres from the path) can legitimately fall outside every zone's
        strict AABB. Falling back to the nearest zone by centroid distance
        instead of dropping it as "unassigned" is what actually gets
        landmarks to show up in that common case.
        """
        if self.semantic_mapper is None:
            print(f"[ScanSession:{self.location_id}] finalize_landmarks: no semantic_mapper configured, skipping.")
            return
        merged = self._finalize_raw_landmarks()
        with self._lock:
            print(f"[ScanSession:{self.location_id}] finalize_landmarks: {len(merged)} resolved landmarks, {len(self.labeler.zones)} zones.")
            for z in self.labeler.zones:
                print(f"  zone '{z.label}': bbox_min={z.bbox_min} bbox_max={z.bbox_max}")
            if not merged:
                print(f"[ScanSession:{self.location_id}] finalize_landmarks: no landmarks resolved — nothing to assign. "
                      f"(Check earlier logs for VLM/detector failures.)")
                return
            for zone in self.labeler.zones:
                zone.landmarks = []
            contained, nearest_fallback, truly_unassigned = 0, 0, 0
            for lm in merged:
                target = next(
                    (z for z in self.labeler.zones if _zone_contains_point_xz(z, lm.x, lm.z)),
                    None,
                )
                if target is not None:
                    target.landmarks.append(lm)
                    contained += 1
                    print(f"  '{lm.name}' at ({lm.x:.2f}, {lm.z:.2f}) -> zone '{target.label}' (contained)")
                elif self.labeler.zones:
                    target = min(
                        self.labeler.zones,
                        key=lambda z: _zone_center_dist_xz(z, lm.x, lm.z),
                    )
                    target.landmarks.append(lm)
                    nearest_fallback += 1
                    print(f"  '{lm.name}' at ({lm.x:.2f}, {lm.z:.2f}) -> zone '{target.label}' (nearest fallback, outside AABB)")
                else:
                    truly_unassigned += 1
                    print(f"  '{lm.name}' at ({lm.x:.2f}, {lm.z:.2f}) -> UNASSIGNED (no zones exist)")
            n_merged, n_zones = len(merged), len(self.labeler.zones)
        print(
            f"[ScanSession:{self.location_id}] finalize_landmarks: {n_merged} merged landmarks → "
            f"{n_zones} zones ({contained} contained, {nearest_fallback} nearest-fallback, {truly_unassigned} unassigned)"
        )

    def finalize_landmarks_flat(self) -> List[Landmark]:
        """
        Zone-free counterpart to finalize_landmarks(), for MappingService
        (server/services/mapping_servicer.py) — named zones/labels are being
        dropped from the live-guiding architecture (see CLAUDE.md's
        "Client-Orchestrated Live Session" section): navigation now targets
        landmarks/functional objects directly, not zone containers, so
        there's no AABB to assign a merged landmark into any more. Uses the
        same _finalize_raw_landmarks() finalize-time clustering
        finalize_landmarks() does, just returns the flat list instead of
        bucketing it into self.labeler.zones.
        """
        if self.semantic_mapper is None:
            print(f"[ScanSession:{self.location_id}] finalize_landmarks_flat: no semantic_mapper configured, skipping.")
            return []
        merged = self._finalize_raw_landmarks()
        if not merged:
            print(f"[ScanSession:{self.location_id}] finalize_landmarks_flat: no landmarks resolved.")
        else:
            print(f"[ScanSession:{self.location_id}] finalize_landmarks_flat: {len(merged)} merged landmarks.")
        return merged

    def preview_landmarks(self) -> None:
        """
        Non-destructive, unclustered zone.landmarks assignment for LIVE display
        during scanning — call after each mini-batch, before rendering
        live_cloud_plot/occupancy_plot, so markers show up without waiting for
        Export. Recomputes zone.landmarks from ALL of _raw_landmarks every
        call (cheap: nearest-zone grouping, no overlap-merge clustering) and
        does NOT clear _raw_landmarks or touch clustering state — the
        authoritative, deduplicated version still only happens once at export
        via finalize_landmarks().
        """
        with self._lock:
            if not self.labeler.zones or not self._raw_landmarks:
                return
            for zone in self.labeler.zones:
                zone.landmarks = []
            for lm in self._raw_landmarks:
                target = next(
                    (z for z in self.labeler.zones if _zone_contains_point_xz(z, lm.x, lm.z)),
                    None,
                )
                if target is None:
                    target = min(
                        self.labeler.zones,
                        key=lambda z: _zone_center_dist_xz(z, lm.x, lm.z),
                    )
                target.landmarks.append(lm)

    def export(self) -> str:
        """Export point cloud + zone labels + ORB keyframes for online localization."""
        self.finalize_landmarks()
        self.ensure_cloud_built()  # self._cloud is lazy — must be built before export reads it
        with self._lock:
            cloud = self._cloud
            zones = list(self.labeler.zones)
            keyframes = list(self.pose_graph.keyframes)
            camera_K = self._camera_K

        out_dir = export_map(
            cloud, zones, self.location_id,
            zone_type=self.zone_type,
            occupancy_map=self.occupancy_map,
        )

        if keyframes and camera_K is not None:
            kf_dir = os.path.join(out_dir, "keyframes")
            os.makedirs(kf_dir, exist_ok=True)
            kf_index = []
            for kf in keyframes:
                fname = f"kf{kf.id:06d}.npz"
                np.savez_compressed(
                    os.path.join(kf_dir, fname),
                    descriptors=kf.descriptors,
                    keypoints_2d=kf.keypoints_2d,
                    points_3d=kf.points_3d,
                )
                kf_index.append(
                    {"id": kf.id, "file": fname, "pose_c2w": kf.pose.tolist()}
                )
            with open(os.path.join(kf_dir, "index.json"), "w") as f:
                json.dump({"camera_K": camera_K, "keyframes": kf_index}, f, indent=2)
            print(
                f"[ScanSession:{self.location_id}] Saved {len(kf_index)} keyframes "
                f"→ {kf_dir}"
            )

        return out_dir

    @property
    def zones(self) -> List[Zone]:
        with self._lock:
            return list(self.labeler.zones)

    @property
    def camera_position(self) -> List[float]:
        with self._lock:
            return self.tracker._world_pose[:3, 3].tolist()


# ── ScanSessionManager ────────────────────────────────────────────────────────


class ScanSessionManager:
    """Thread-safe registry mapping location_id → ScanSession."""

    def __init__(
        self,
        estimator: BaseDepthEstimator,
        rtabmap_client: Optional["RtabmapPoseClient"] = None,
        semantic_mapper: Optional[SemanticMapper] = None,
    ) -> None:
        self._estimator = estimator
        self._rtabmap_client = rtabmap_client
        self._semantic_mapper = semantic_mapper
        self._sessions: dict[str, ScanSession] = {}
        self._lock = threading.Lock()
        # Last-applied Occupancy Map tuning knobs (OccupancyMap.set_params'
        # keys) — carried into every NEW session so the GUI's "Occupancy Map
        # Settings" accordion sticks across location switches, not just the
        # currently-open one. See configure_occupancy_defaults().
        self._occupancy_defaults: dict = {}
        # Same pattern for the novelty+blur gate (ScanSession.
        # configure_novelty_gate) — see configure_novelty_gate_defaults().
        self._novelty_defaults: dict = {}

    @property
    def rtabmap_pose_available(self) -> bool:
        return self._rtabmap_client is not None and self._rtabmap_client.connected

    @property
    def semantic_mapper_available(self) -> bool:
        return self._semantic_mapper is not None

    @property
    def semantic_mapper_model_id(self) -> str:
        """Gemini (tagging) + GroundingDINO-tiny (detection) — see
        frame_extractor/tagging.py's FrameTagger. Unlike the old Gemma-VLM
        design, this pipeline has no single swappable "model id"; both are
        configured once at server startup (see scan_server.py/
        grpc_server.py's GEMINI_TAGGING_MODEL_ID/GDINO_TAGGING_MODEL_ID env
        vars) and aren't hot-swappable from the GUI."""
        if self._semantic_mapper is None:
            return ""
        tagger = self._semantic_mapper._tagger
        gemini_id = getattr(tagger, "_gemini_model_id", "gemini")
        return f"Gemini '{gemini_id}' + GroundingDINO-tiny"

    def get_or_create(self, location_id: str, zone_type: str = "") -> ScanSession:
        with self._lock:
            if location_id not in self._sessions:
                print(f"[ScanSessionManager] Creating session for '{location_id}'")
                self._sessions[location_id] = ScanSession(
                    location_id,
                    self._estimator,
                    rtabmap_client=self._rtabmap_client,
                    semantic_mapper=self._semantic_mapper,
                    zone_type=zone_type,
                    occupancy_params=self._occupancy_defaults,
                    novelty_params=self._novelty_defaults,
                )
            elif zone_type:
                self._sessions[location_id].zone_type = zone_type
            return self._sessions[location_id]

    def get(self, location_id: str) -> Optional[ScanSession]:
        with self._lock:
            return self._sessions.get(location_id)

    def reset_all(self) -> None:
        """Drops every accumulated ScanSession (all location_ids) and resets
        the shared RTAB-Map docker session — called by StatusService.
        ResetSession when a fresh client connection starts, so a new
        connection never silently resumes a stale map/pose left behind by
        whatever the previous connection was doing. A location's next
        get_or_create() call after this rebuilds a genuinely fresh
        ScanSession, same as the very first time that location_id was ever
        seen. RTAB-Map itself only ever holds one active SLAM session at a
        time (see rtabmap_docker/README.md's "single active session"
        limitation) — resetting it here matters even for location_ids not
        being dropped, since RTAB-Map's own memory doesn't distinguish
        between them at all."""
        with self._lock:
            n_dropped = len(self._sessions)
            self._sessions.clear()
        if self._rtabmap_client is not None:
            self._rtabmap_client.reset()
        print(f"[ScanSessionManager] reset_all: dropped {n_dropped} session(s), RTAB-Map reset.")

    def configure_occupancy_defaults(self, **kwargs) -> None:
        """
        Remember these Occupancy Map tuning knobs as the default for every
        FUTURE session, and apply them immediately to every session that
        already exists (in place — see ScanSession.configure_occupancy_map).
        """
        clean = {k: v for k, v in kwargs.items() if v is not None}
        self._occupancy_defaults.update(clean)
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.configure_occupancy_map(**clean)

    def configure_novelty_gate_defaults(self, **kwargs) -> None:
        """
        Remember these novelty+blur gate tuning knobs as the default for
        every FUTURE session, and apply them immediately to every session
        that already exists (in place — see ScanSession.configure_novelty_gate).
        """
        clean = {k: v for k, v in kwargs.items() if v is not None}
        self._novelty_defaults.update(clean)
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.configure_novelty_gate(**clean)
