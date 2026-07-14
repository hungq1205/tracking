"""
StreamingScanSession — push-based, incremental counterpart to ScanSession's
batch orchestration (previously only available via scan_gui.py's
_run_local_scan, which requires a finished on-disk dataset + a pre-declared
Segment Table before it can start).

Call sequence for one scanning session (frame/IMU arrival order doesn't need
to be interleaved in any particular way — push whichever arrives):
    session = StreamingScanSession(scan_manager, location_id, pose_src=...)
    session.start_zone("kitchen")       # optional — zone-less scans are fine
    session.push_imu(ts_ns, ax, ay, az, gx, gy, gz)   # as samples arrive
    session.push_frame(rgb, ts_ns)                    # as frames arrive
    ...
    session.end_zone()
    session.start_zone("hallway")
    ...
    session.finish()                    # flush + finalize + export

Today's caller is stream_simulator.py, replaying an already-uploaded dataset
frame-by-frame/IMU-by-IMU to prove this interface end-to-end. A later, real
live source (Android gRPC stream) is meant to drive this exact same class
through the exact same four calls — nothing here should need to change for
that; only the driver (what calls push_frame/push_imu, and when) changes.

RTAB-Map pose mode is fully supported here, unlike the old ORB-SLAM3 mode it
replaced — RTAB-Map's RGB-D odometry needs no raw IMU sample retention at
all, so there's no streaming-specific limitation to fall back from.

Only two pose sources are supported: "IMU + VO" and "RTAB-Map" — "Auto",
"VO only", and "DA3 poses" were removed (DA3 poses' 5-frame sliding-window
pose stitching added complexity without being needed once RTAB-Map covers
the no-calibration-needed case; "VO only"/"Auto" collapsed into "IMU + VO",
which already falls back to VO-only behavior on its own when no imu.csv is
present).
"""

from typing import List, Optional

import numpy as np

from scan_session import (
    DEFAULT_VOXEL_SIZE,
    IncrementalImuIntegrator,
    ScanSession,
    ScanSessionManager,
)


def resolve_pose_flags(pose_src: str, rtabmap_available: bool) -> tuple:
    """(use_imu, use_rtabmap) for a given Pose source radio value — only
    "IMU + VO" and "RTAB-Map" are supported. RTAB-Map needs no retained raw
    IMU samples (see IncrementalImuIntegrator); falls back to IMU + VO if no
    RTAB-Map client is connected."""
    use_rtabmap = pose_src == "RTAB-Map" and rtabmap_available
    use_imu = not use_rtabmap
    return use_imu, use_rtabmap


class StreamingScanSession:
    def __init__(
        self,
        scan_manager: ScanSessionManager,
        location_id: str,
        pose_src: str = "IMU + VO",
        imu_orientation: str = "portrait",
        zone_type: str = "",
        axis_perm: Optional[np.ndarray] = None,
        mini_batch: int = 4,
        sor_nb_neighbors: int = 20,
        sor_std_ratio: float = 2.25,
        imu_stationary_s: float = 1.0,
        occupancy_voxel_size: float = DEFAULT_VOXEL_SIZE,
    ) -> None:
        self.session: ScanSession = scan_manager.get_or_create(location_id, zone_type=zone_type)
        # ScanSessionManager.get_or_create() returns the SAME ScanSession
        # across separate StreamingScanSession instantiations for the same
        # location_id — without this, starting a new stream (Simulated or
        # Manual) would silently resume on top of whatever point cloud/voxel/
        # occupancy data an earlier, unrelated run already accumulated for
        # this location, making it look like the new run already "saw"
        # frames it was never fed. A live camera source has no such leftover
        # state, so every new StreamingScanSession starts from a genuinely
        # empty reconstruction, same as the real thing would.
        self.session.reset_cloud()
        self._use_imu, self._use_rtabmap = resolve_pose_flags(
            pose_src,
            rtabmap_available=getattr(scan_manager, "rtabmap_pose_available", False),
        )
        self._imu_orientation = imu_orientation
        self._axis_perm = axis_perm
        self._mini_batch = max(1, int(mini_batch))
        self._sor_nb_neighbors = sor_nb_neighbors
        self._sor_std_ratio = sor_std_ratio
        self._imu_stationary_s = imu_stationary_s
        # Additional voxel-downsample applied to each chunk's new points
        # before feeding the Occupancy Map (on top of process_frames_batch's
        # existing fine VOXEL_SIZE cleanup) — matches the GUI's Voxelization
        # view (scan_gui.py's voxel_size_input) so occupancy cells correspond
        # to what's displayed there. See ScanSession.process_frames_batch's
        # docstring.
        self._occupancy_voxel_size = occupancy_voxel_size

        self._imu: Optional[IncrementalImuIntegrator] = None
        self._frame_buffer: List[tuple] = []  # (rgb, ts_ns)

        self._zone_name: str = ""
        self._zone_active: bool = False
        self._zone_positions: List[list] = []

        self.last_result: Optional[dict] = None

    # ── ingestion ────────────────────────────────────────────────────────────

    def push_imu(self, ts_ns: float, ax: float, ay: float, az: float,
                  gx: float, gy: float, gz: float) -> None:
        if not self._use_imu:
            return
        if self._imu is None:
            self._imu = IncrementalImuIntegrator(
                orientation=self._imu_orientation, stationary_s=self._imu_stationary_s
            )
        self._imu.push(ts_ns, ax, ay, az, gx, gy, gz)

    def push_frame(self, rgb: np.ndarray, ts_ns: float) -> Optional[dict]:
        """Buffers frames into mini_batch-sized chunks, calling
        ScanSession.process_frames_batch once a chunk is ready. Returns a
        result dict when a chunk was just processed, else None (still
        buffering)."""
        self._frame_buffer.append((rgb, ts_ns))
        if len(self._frame_buffer) < self._mini_batch:
            return None
        chunk = self._frame_buffer
        self._frame_buffer = []
        return self._process_chunk(chunk)

    def flush(self) -> Optional[dict]:
        """Process whatever's left in the frame buffer as a final, possibly
        short chunk. Call before finish() so the last few buffered frames
        (fewer than mini_batch) aren't silently dropped."""
        if not self._frame_buffer:
            return None
        chunk = self._frame_buffer
        self._frame_buffer = []
        return self._process_chunk(chunk)

    def _process_chunk(self, chunk: List[tuple]) -> dict:
        frames = [c[0] for c in chunk]
        ts_list = [c[1] for c in chunk]
        imu_poses = (
            [self._imu.pose_at(ts) for ts in ts_list]
            if self._use_imu and self._imu is not None
            else None
        )
        point_count, cam_pos, infer_ms = self.session.process_frames_batch(
            frames,
            imu_poses=imu_poses,
            use_rtabmap_pose=self._use_rtabmap,
            frame_timestamps_ns=ts_list,
            axis_perm=self._axis_perm,
            sor_nb_neighbors=self._sor_nb_neighbors,
            sor_std_ratio=self._sor_std_ratio,
            occupancy_voxel_size=self._occupancy_voxel_size,
        )
        if self._zone_active and len(self.session.last_trajectory) > 0:
            self._zone_positions.extend(self.session.last_trajectory.tolist())

        result = {
            "point_count": point_count,
            "cam_pos": cam_pos,
            "infer_ms": infer_ms,
            "pose_source": self.session.last_pose_source,
            "n_frames": len(frames),
        }
        self.last_result = result
        return result

    # ── zone signaling (live replacement for the pre-declared Segment Table) ──

    def start_zone(self, name: str) -> None:
        self.end_zone()  # close out any zone still open
        self.session._current_area_name = name
        self._zone_name = name
        self._zone_positions = []
        self._zone_active = bool(name)

    def end_zone(self) -> None:
        if self._zone_active and self._zone_name and self._zone_positions:
            self.session.set_label_from_positions(
                self._zone_name, self._zone_positions, margin=1.5
            )
            self.session.preview_landmarks()
        self._zone_active = False
        self._zone_name = ""
        self._zone_positions = []

    # ── lifecycle ────────────────────────────────────────────────────────────

    def finish(self) -> str:
        """self.session.last_voxel_centers is already the single,
        incrementally-accumulated voxelization from every processed
        batch/node (see ScanSession._merge_voxels) — no voxel_size param
        needed here anymore, nothing left to re-voxelize."""
        self.flush()
        self.end_zone()
        self.session.finalize_voxel_and_occupancy()
        return self.session.export()
