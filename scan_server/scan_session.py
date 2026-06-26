"""
ScanSession — per-location scanning state.

Pipeline (called per video segment):
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
from typing import List, Optional

import cv2
import numpy as np
import open3d as o3d
import pandas as pd

from da3_wrapper import BaseDepthEstimator
from feature_tracker import FeatureTracker
from map_exporter import export_map
from occupancy_map import OccupancyMap
from pose_graph import PoseGraph
from zone_labeler import Zone, ZoneLabeler

_MAPS_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "data", "maps"))

KEYFRAME_INTERVAL = 10
MIN_KF_VALID_PTS = 10
VOXEL_SIZE = 0.02


# ── IMU integrator ─────────────────────────────────────────────────────────────


class ImuIntegrator:
    """
    Dead-reckoning pose integrator from Android IMU CSV.
    CSV format (no header): timestamp_ns,ax,ay,az,gx,gy,gz
    Integrates gyroscope for orientation and double-integrates debiased
    acceleration for translation.  Used to pre-compute all camera poses for
    a segment before depth estimation, so point-cloud back-projection uses
    globally consistent poses rather than per-batch VO estimates.
    """

    _COLS = ["ts", "ax", "ay", "az", "gx", "gy", "gz"]

    def __init__(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path, header=None, names=self._COLS)
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        self._ts: np.ndarray = df["ts"].to_numpy(dtype=np.float64)
        self._accel: np.ndarray = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
        self._gyro: np.ndarray = df[["gx", "gy", "gz"]].to_numpy(dtype=np.float64)
        self._poses: np.ndarray = self._integrate()
        duration = (self._ts[-1] - self._ts[0]) * 1e-9
        print(f"[ImuIntegrator] {len(self._ts):,} samples  {duration:.1f} s  "
              f"start={self._ts[0]:.0f} ns")

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


# ── Module-level helpers ───────────────────────────────────────────────────────


def _estimate_K(h: int, w: int) -> np.ndarray:
    f = max(w, h) * 0.8
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


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

    def __init__(self, location_id: str, estimator: BaseDepthEstimator) -> None:
        self.location_id = location_id
        self.estimator = estimator
        self.labeler = ZoneLabeler()

        self.tracker = FeatureTracker(n_features=2000)
        self.pose_graph = PoseGraph()
        self.occupancy_map = OccupancyMap(resolution=0.05)
        self._cloud = o3d.geometry.PointCloud()

        self.latest_frame_rgb: Optional[np.ndarray] = None
        self.last_frames_rgb: list = []
        self.last_depth_frames: list = []
        self.last_frame_poses: list = []
        self.last_trajectory: np.ndarray = np.zeros((0, 3), dtype=np.float32)

        self._frame_count: int = 0
        self._camera_K: Optional[list] = None
        self._imu: Optional[ImuIntegrator] = None
        self._lock = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────────

    def set_imu_file(self, csv_path: str) -> None:
        """Load an IMU CSV and prepare the integrator for this session."""
        self._imu = ImuIntegrator(csv_path)

    def compute_segment_poses(
        self,
        n_frames: int,
        start_s: float,
        video_fps: float,
    ) -> Optional[List[np.ndarray]]:
        """
        Pre-compute one 4×4 c2w pose per video frame using IMU dead-reckoning.
        Call this *before* process_frames_batch so all poses are ready before
        the depth-estimation pass begins.
        Returns None when no IMU data is loaded.
        """
        if self._imu is None:
            return None
        poses = []
        for i in range(n_frames):
            frame_time_s = start_s + i / max(video_fps, 1.0)
            ts_ns = self._imu.start_ns + frame_time_s * 1e9
            poses.append(self._imu.pose_at(ts_ns))
        print(
            f"[ScanSession:{self.location_id}] Pre-computed {n_frames} IMU poses "
            f"(seg start={start_s:.1f} s, fps={video_fps:.1f})"
        )
        return poses

    def process_frames_batch(
        self,
        frames_rgb: list,
        imu_poses: Optional[List[np.ndarray]] = None,
        use_da3_pose: bool = False,
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

        Returns (point_count, cam_pos, infer_ms).
        """
        with self._lock:
            # ── Step 1: DA3 depth estimation ──────────────────────────────────
            _t0 = time.perf_counter()
            if hasattr(self.estimator, "estimate_batch"):
                depth_frames = self.estimator.estimate_batch(frames_rgb)
            else:
                depth_frames = [self.estimator.estimate(f) for f in frames_rgb]
            infer_ms = (time.perf_counter() - _t0) * 1000

            # Cache intrinsics from first available depth frame
            if self._camera_K is None:
                for df in depth_frames:
                    if df.intrinsics is not None:
                        self._camera_K = df.intrinsics.tolist()
                        break

            # ── Step 2: Camera poses ───────────────────────────────────────────
            if use_da3_pose:
                # DA3 multi-view camera pose path.
                # df.camera_pose is c2w in the DA3 model's coordinate frame.
                # Anchor the first pose to the current VO world frame so
                # successive batches stay consistent; fall back to VO per-frame
                # if the estimator doesn't produce camera_pose (e.g. ONNX model).
                raw_poses: list[np.ndarray] = []
                da3_anchor: Optional[np.ndarray] = None
                for rgb, df in zip(frames_rgb, depth_frames):
                    if df.camera_pose is not None:
                        if da3_anchor is None:
                            K = df.intrinsics
                            vo_seed, _ = self.tracker.track(rgb, df.depth_map, K)
                            da3_anchor = vo_seed @ np.linalg.inv(df.camera_pose)
                        pose = da3_anchor @ df.camera_pose
                    else:
                        # ONNX model — no camera_pose; fall back to VO
                        K = df.intrinsics
                        pose, _ = self.tracker.track(rgb, df.depth_map, K)
                    raw_poses.append(pose)
                    self._frame_count += 1
                    self.latest_frame_rgb = rgb
            elif imu_poses is not None:
                # IMU+VO fusion: FeatureTracker provides metric translation;
                # IMU gyro integration provides rotation (gyro is reliable;
                # accelerometer double-integration drifts catastrophically for
                # walking/dynamic motion so translation is NOT taken from IMU).
                raw_poses = []
                for rgb, df, imu_pose in zip(frames_rgb, depth_frames, imu_poses):
                    K = df.intrinsics
                    tracker_pose, _ = self.tracker.track(rgb, df.depth_map, K)
                    # Replace VO rotation with IMU rotation, keep VO translation
                    fused = tracker_pose.copy()
                    fused[:3, :3] = imu_pose[:3, :3]
                    raw_poses.append(fused)
                self._frame_count += len(frames_rgb)
                self.latest_frame_rgb = frames_rgb[-1]
            else:
                # VO path: FeatureTracker + pose graph
                raw_poses = []
                _da3_vo_anchor: Optional[np.ndarray] = None

                for rgb, df in zip(frames_rgb, depth_frames):
                    K = df.intrinsics
                    tracker_pose, _ = self.tracker.track(rgb, df.depth_map, K)

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

            # ── Step 3: Dense back-projection + voxel fusion ──────────────────
            new_cloud = o3d.geometry.PointCloud()
            _bp_counts = []

            for rgb, df, pose in zip(frames_rgb, depth_frames, raw_poses):
                result = _back_project_frame(rgb, df, pose, max_depth=10.0)
                if result is None:
                    _bp_counts.append(0)
                    continue
                pts, cols = result
                _bp_counts.append(len(pts))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)
                new_cloud += pcd

            print(
                f"  [batch] infer {infer_ms:.0f} ms ({infer_ms/len(frames_rgb):.0f} ms/f)  "
                f"back-project pts/frame: {_bp_counts}  "
                f"pose_src={'DA3' if use_da3_pose else ('IMU+VO' if imu_poses is not None else 'VO')}"
            )

            self._cloud += new_cloud
            if len(self._cloud.points) > 0:
                self._cloud = self._cloud.voxel_down_sample(VOXEL_SIZE)

            # ── Step 4: Occupancy map ──────────────────────────────────────────
            traj = np.array([p[:3, 3] for p in raw_poses], dtype=np.float32)
            self.last_trajectory = traj
            self.occupancy_map.update(traj, np.asarray(self._cloud.points))

            self.last_frames_rgb = list(frames_rgb)
            self.last_depth_frames = list(depth_frames)
            self.last_frame_poses = list(raw_poses)

            cam_pos = raw_poses[-1][:3, 3].tolist() if raw_poses else [0.0, 0.0, 0.0]
            return len(self._cloud.points), cam_pos, infer_ms

    def reset_cloud(self) -> None:
        """Clear accumulated point cloud, poses, and occupancy map. Keeps zone labels and IMU."""
        with self._lock:
            self._cloud = o3d.geometry.PointCloud()
            self.pose_graph = PoseGraph()
            self.occupancy_map = OccupancyMap(resolution=0.05)
            self.tracker.reset()
            self.last_frame_poses = []
            self.last_frames_rgb = []
            self.last_depth_frames = []
            self.last_trajectory = np.zeros((0, 3), dtype=np.float32)
            self._frame_count = 0
        print(f"[ScanSession:{self.location_id}] Cloud cleared.")

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
        self, label: str, positions, margin: float = 0.3
    ) -> Zone:
        """
        Create a zone AABB covering all camera positions visited during a segment,
        expanded outward by `margin` metres in each axis.
        """
        with self._lock:
            arr = np.array(positions, dtype=np.float32)
            bbox_min = (arr.min(axis=0) - margin).tolist()
            bbox_max = (arr.max(axis=0) + margin).tolist()
            zone = Zone(label=label, bbox_min=bbox_min, bbox_max=bbox_max)
            self.labeler.zones.append(zone)
        print(
            f"[ScanSession:{self.location_id}] Zone '{label}' from path: "
            f"{bbox_min} → {bbox_max}"
        )
        return zone

    def export(self) -> str:
        """Export point cloud + zone labels + ORB keyframes for online localization."""
        with self._lock:
            cloud = self._cloud
            zones = list(self.labeler.zones)
            keyframes = list(self.pose_graph.keyframes)
            camera_K = self._camera_K

        out_dir = export_map(cloud, zones, self.location_id)

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

    def __init__(self, estimator: BaseDepthEstimator) -> None:
        self._estimator = estimator
        self._sessions: dict[str, ScanSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, location_id: str) -> ScanSession:
        with self._lock:
            if location_id not in self._sessions:
                print(f"[ScanSessionManager] Creating session for '{location_id}'")
                self._sessions[location_id] = ScanSession(location_id, self._estimator)
            return self._sessions[location_id]

    def get(self, location_id: str) -> Optional[ScanSession]:
        with self._lock:
            return self._sessions.get(location_id)
