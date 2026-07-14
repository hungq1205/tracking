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
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np
import open3d as o3d
import pandas as pd

from da3_wrapper import BaseDepthEstimator
from feature_tracker import FeatureTracker
from map_exporter import export_map
from occupancy_map import OccupancyMap
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

# Statistical Outlier Removal (Open3D) — run after every voxel downsample to
# strip stray/noisy points whose average neighbor distance is far outside the
# cloud's typical density (see ScanSession.process_frames_batch, Step 3).
# Defaults only — GUI-adjustable per scan via the Outlier Removal accordion.
SOR_NB_NEIGHBORS = 20
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
    ) -> None:
        self.location_id = location_id
        self.estimator = estimator
        self.rtabmap_client = rtabmap_client
        self.semantic_mapper = semantic_mapper
        self.zone_type = zone_type
        self.labeler = ZoneLabeler()

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

        # Diagnostics for the most recent process_frames_batch() call — read by
        # scan_gui.py to surface actual pose source + RTAB-Map tracking health
        # in the Gradio log panel (server-console prints alone are easy to miss).
        self.last_pose_source: str = "none"
        self.last_rtabmap_lost: int = 0
        self.last_rtabmap_total: int = 0
        self.last_rtabmap_loop_closure: bool = False

        # Session-wide semantic landmark accumulation (raw, unmerged). Persists
        # across all segments/zones — global overlap merging happens once, at
        # export time, via finalize_landmarks().
        self._raw_landmarks: List[Landmark] = []
        self._current_area_name: str = ""

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

        Returns (point_count, cam_pos, infer_ms).
        """
        with self._lock:
            # ── Step 1: depth estimation ──────────────────────────────────────
            active_estimator = self.estimator
            _t0 = time.perf_counter()
            if hasattr(active_estimator, "estimate_batch"):
                depth_frames = active_estimator.estimate_batch(frames_rgb)
            else:
                depth_frames = [active_estimator.estimate(f) for f in frames_rgb]
            infer_ms = (time.perf_counter() - _t0) * 1000
            _estimator_device = getattr(active_estimator, "device", "unknown")
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
            if use_rtabmap_pose and self.rtabmap_client is None:
                print(
                    f"[ScanSession:{self.location_id}] RTAB-Map poses requested but no "
                    f"connected client — falling back to VO."
                )
                use_rtabmap_pose = False

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
                for rgb, df, t in zip(frames_rgb, depth_frames, tracked):
                    self.tracker.track(rgb, df.depth_map, df.intrinsics)
                    if not self.tracker.last_depth_trustworthy and t.node_id >= 0:
                        self._rtabmap_untrusted_node_ids.add(t.node_id)
                        print(
                            f"[depth-consistency] RTAB-Map node {t.node_id} flagged "
                            f"untrustworthy (frac_bad={self.tracker.last_depth_agree_err:.2f}, "
                            f"n={self.tracker.last_depth_agree_n}); its geometry will be "
                            f"skipped when pulled via get_cloud()."
                        )
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
            print(f"[timing] pose computation ({len(frames_rgb)} frames, {_pose_src_label}): {_pose_ms:.1f} ms")

            # World-space output remap (see process_frames_batch docstring) —
            # applied after all pose sources/anchoring above, before anything
            # downstream (back-projection, occupancy map, semantic mapper,
            # exposed trajectory/cam_pos) touches raw_poses.
            raw_poses = _remap_poses(raw_poses, axis_perm)

            # ── Step 3: Dense back-projection + voxel fusion ──────────────────
            # RTAB-Map mode does NOT back-project here at all — its own
            # reconstructed surface is pulled separately below (Step 3b, via
            # rtabmap_client.get_cloud()), sourced from RTAB-Map's own
            # cloudRGBFromSensorData + its CURRENT graph-corrected poses
            # rather than this project's DA3-depth back-projection. Semantic
            # landmark extraction below still runs for every pose source —
            # it's independent of which cloud-reconstruction path is active.
            new_cloud = o3d.geometry.PointCloud()
            _bp_counts = []
            _t0_bp = time.perf_counter()

            _ts_slice = (
                frame_timestamps_ns if frame_timestamps_ns is not None
                else [None] * len(frames_rgb)
            )
            _n_depth_rejected = 0
            for rgb, df, pose, ts_ns, depth_check in zip(
                frames_rgb, depth_frames, raw_poses, _ts_slice, depth_checks
            ):
                if not use_rtabmap_pose:
                    trustworthy, agree_err, agree_n = depth_check
                    if not trustworthy:
                        _bp_counts.append(0)
                        _n_depth_rejected += 1
                        print(
                            f"[depth-consistency] frame REJECTED — dense DA3 depth "
                            f"disagreed with independently triangulated sparse points "
                            f"(frac_bad={agree_err:.2f}, n={agree_n}, "
                            f"thresh={self.tracker._depth_frac_bad_thresh:.2f}); "
                            f"not fused into map (pose/semantic extraction unaffected)."
                        )
                    else:
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

                # Semantic landmark extraction — every raw frame is offered to
                # the mapper, which internally keeps only the sharpest frame
                # per ~1s window of actual capture time (SemanticMapper.consider_frame).
                if self.semantic_mapper is not None and self._current_area_name:
                    K_eff = (
                        df.intrinsics if df.intrinsics is not None
                        else _estimate_K(*rgb.shape[:2])
                    )
                    try:
                        new_lms = self.semantic_mapper.consider_frame(
                            frame_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            depth_map=df.depth_map,
                            world_pose=pose,
                            K=K_eff,
                            zone_type=self.zone_type,
                            area_name=self._current_area_name,
                            frame_idx=self._frame_count,
                            timestamp_ns=int(ts_ns) if ts_ns is not None else None,
                        )
                        self._raw_landmarks.extend(new_lms)
                    except Exception as _sem_err:
                        print(f"[ScanSession] Semantic extraction error: {_sem_err}")

            _bp_ms = (time.perf_counter() - _t0_bp) * 1000
            print(
                f"[timing] back-projection + semantic extraction "
                f"({len(frames_rgb)} frames, {sum(_bp_counts):,} raw pts, "
                f"{_n_depth_rejected} depth-rejected): {_bp_ms:.1f} ms"
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
            if use_rtabmap_pose and self.rtabmap_client is not None:
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

            if len(new_cloud_clean.points) > 0:
                occ_centers, occ_colors, occ_vsize = voxelize_cloud(
                    new_cloud_clean, voxel_size=occupancy_voxel_size, max_voxels=_NO_COARSEN_MAX_VOXELS
                )
                if len(occ_centers) > 0:
                    with timed(f"occupancy_map.update ({len(occ_centers)} voxel centers)"):
                        self.occupancy_map.update(traj, occ_centers)
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

        raw_clouds: List[o3d.geometry.PointCloud] = []
        for node in nodes:
            c = o3d.geometry.PointCloud()
            if len(node.points) > 0:
                c.points = o3d.utility.Vector3dVector(node.points.astype(np.float64))
                c.colors = o3d.utility.Vector3dVector(node.colors.astype(np.float64) / 255.0)
            raw_clouds.append(c)

        sizes = [len(c.points) for c in raw_clouds]
        total = sum(sizes)
        if total > sor_nb_neighbors:
            with timed(f"RTAB-Map batched cloud concat ({total} pts, {len(nodes)} nodes)"):
                combined = o3d.geometry.PointCloud()
                for c in raw_clouds:
                    combined += c
            _, keep_mask = _remove_statistical_outlier_accel_masked(
                combined, sor_nb_neighbors, sor_std_ratio
            )
            combined_pts = np.asarray(combined.points)
            combined_cols = np.asarray(combined.colors)
            cleaned_clouds = []
            offset = 0
            for size in sizes:
                node_mask = keep_mask[offset:offset + size]
                offset += size
                nc = o3d.geometry.PointCloud()
                if node_mask.any():
                    seg = slice(offset - size, offset)
                    nc.points = o3d.utility.Vector3dVector(combined_pts[seg][node_mask])
                    nc.colors = o3d.utility.Vector3dVector(combined_cols[seg][node_mask])
                cleaned_clouds.append(nc)
        else:
            cleaned_clouds = raw_clouds

        _voxelize_ms = 0.0
        _occ_update_ms = 0.0
        for node, cloud in zip(nodes, cleaned_clouds):
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
                    self.occupancy_map.update(traj, occ_centers)
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
        Lazily merge every batch's already-cleaned (voxelized + outlier-
        removed, see process_frames_batch's Step 3) point cloud into
        self._cloud. Historically only called on demand (Reload on the Live
        Points tab, Voxelize, Export, finalize_voxel_and_occupancy); now also
        called once per processed chunk during Scan/Simulated Live Stream
        when the "Live Points" GUI checkbox is on (see scan_gui.py), so this
        needs to stay cheap on repeat calls with only a little new data each
        time — see self._merged_batch_count's field comment for how.

        Each batch was already voxel-downsampled + outlier-removed at batch
        time, so this just needs one more voxel pass to merge adjacent
        batches' voxels into a single consistent grid, not a full outlier-
        removal re-run over everything. That final voxel_down_sample pass is
        still O(current total accumulated size) every call — Open3D has no
        incremental voxel-merge primitive, so this can't be made fully O(new
        data) — the GUI checkbox is the actual cost control for a large scan,
        not this method.
        """
        with self._lock:
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
            self._cloud = o3d.geometry.PointCloud()
            self._raw_cloud_batches = []
            self._raw_point_count = 0
            self._merged_batch_count = 0
            self._cloud_raw_accum = o3d.geometry.PointCloud()
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
            self.last_rtabmap_loop_closure = False
            self._raw_landmarks = []
        if self.rtabmap_client is not None:
            self.rtabmap_client.reset()
        print(f"[ScanSession:{self.location_id}] Cloud cleared.")

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

    def finalize_landmarks(self) -> None:
        """
        Run once, at export time: globally merge every raw landmark accumulated
        across the whole session (all segments/zones), then assign each merged
        Landmark to a Zone — first choice is whichever Zone's X-Z AABB
        footprint contains its centroid, but zone AABBs are built from the
        camera's walked path (+ a small margin, see set_label_from_positions)
        which is often much smaller than the room itself, so a real,
        correctly-detected landmark (e.g. furniture against a wall, a few
        metres from the path) can legitimately fall outside every zone's
        strict AABB. Falling back to the nearest zone by centroid distance
        instead of dropping it as "unassigned" is what actually gets
        landmarks to show up in that common case.
        """
        with self._lock:
            if self.semantic_mapper is None:
                print(f"[ScanSession:{self.location_id}] finalize_landmarks: no semantic_mapper configured, skipping.")
                return
            # Flush any partial batch (< IMAGES_PER_PROMPT frames) still
            # buffered in the VLM-batching path — otherwise the last few
            # sampled frames of the session are silently dropped.
            flushed = self.semantic_mapper.flush()
            if flushed:
                print(f"[ScanSession:{self.location_id}] finalize_landmarks: flushed {len(flushed)} landmarks from a partial VLM batch.")
                self._raw_landmarks.extend(flushed)
            print(f"[ScanSession:{self.location_id}] finalize_landmarks: {len(self._raw_landmarks)} raw landmarks accumulated, {len(self.labeler.zones)} zones.")
            for z in self.labeler.zones:
                print(f"  zone '{z.label}': bbox_min={z.bbox_min} bbox_max={z.bbox_max}")
            if not self._raw_landmarks:
                print(f"[ScanSession:{self.location_id}] finalize_landmarks: no raw landmarks — nothing to assign. "
                      f"(Check earlier logs for VLM/detector failures in extract_landmarks.)")
                return
            merged = self.semantic_mapper.cluster_landmarks(list(self._raw_landmarks))
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
            self._raw_landmarks = []
            n_merged, n_zones = len(merged), len(self.labeler.zones)
        print(
            f"[ScanSession:{self.location_id}] finalize_landmarks: {n_merged} merged landmarks → "
            f"{n_zones} zones ({contained} contained, {nearest_fallback} nearest-fallback, {truly_unassigned} unassigned)"
        )

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

    @property
    def rtabmap_pose_available(self) -> bool:
        return self._rtabmap_client is not None and self._rtabmap_client.connected

    @property
    def semantic_mapper_available(self) -> bool:
        return self._semantic_mapper is not None

    @property
    def semantic_mapper_model_id(self) -> str:
        if self._semantic_mapper is None:
            return ""
        return getattr(self._semantic_mapper._vlm, "model_id", "")

    def set_semantic_mapper_model(self, model_id: str) -> None:
        """Swap the VLM model used by the shared SemanticMapper for future
        calls — lets the Scan UI change models without restarting scan_server.py."""
        if self._semantic_mapper is None:
            return
        self._semantic_mapper._vlm.set_model_id(model_id)

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
                )
            elif zone_type:
                self._sessions[location_id].zone_type = zone_type
            return self._sessions[location_id]

    def get(self, location_id: str) -> Optional[ScanSession]:
        with self._lock:
            return self._sessions.get(location_id)

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
