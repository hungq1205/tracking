"""
RtabmapPoseClient — bridges scan_server to the standalone RTAB-Map pose +
surface-reconstruction service (scan_server/rtabmap_docker/) over a single
ZeroMQ REQ/REP socket. No ROS on either end. See
rtabmap_docker/src/rtabmap_server.cc for the wire protocol this mirrors
exactly (struct formats below must stay in lockstep with the C++
`TrackRequestHeader`/`GetCloudRequestHeader`/reply layouts).

Replaces the old OrbSlam3PoseClient (scan_server/orbslam3_docker/, removed) —
unlike ORB-SLAM3's mono-inertial protocol, there is NO IMU payload anywhere in
this client: RTAB-Map's RGB-D visual odometry only needs camera intrinsics +
a depth map (this project's own DA3-ONNX estimated depth, not a real sensor),
sidestepping ORB-SLAM3's unreliable camera-IMU calibration entirely.

get_cloud() pulls RTAB-Map's OWN reconstructed surface — each node's stored
SensorData re-projected (util3d::cloudRGBFromSensorData), voxelized, and
transformed by RTAB-Map's CURRENT graph-corrected pose (getLocalOptimizedPoses)
— server-side, not scan_session.py's own DA3-depth back-projection. Since
poses used are always the LATEST corrected ones, re-pulling after a loop
closure gives every already-reconstructed node's cloud its corrected
position — see track_batch's `loop_closure` flag, which signals exactly when
that's worth doing.
"""

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import zmq

if TYPE_CHECKING:
    from da3_wrapper import DepthFrame

_CMD_TRACK = 0x01
_CMD_RESET = 0x02
_CMD_PING = 0x03
_CMD_GET_CLOUD = 0x04
_STATUS_OK = 0x00
_STATUS_LOST = 0x01
_STATUS_RESET_OK = 0x02
_STATUS_PONG = 0x03

_HEADER_FMT = "<qiidddd"      # timestamp_ns, width, height, fx, fy, cx, cy — no n_imu
_POSE_FMT = "<7d"             # tx, ty, tz, qx, qy, qz, qw
_TRACK_TAIL_FMT = "<Bi"       # loop_closure_flag, new_node_id (-1 = no new node)
_GET_CLOUD_REQ_FMT = "<iff"   # since_node_id, voxel_size, max_depth
_NODE_HDR_FMT = "<i7di"       # node_id, pose(7d), point_count


@dataclass
class TrackedFrame:
    """One TRACK reply — pose is None on LOST/timeout (see track_batch)."""
    pose: Optional[np.ndarray]
    # True if THIS frame's processing closed a loop AND the resulting map
    # correction was significant (see rtabmap_server.cc's
    # SIGNIFICANT_CORRECTION_* thresholds — RGBD/ProximityBySpace accepts a
    # loop closure against any spatially-nearby node, which fires almost
    # every frame during slow, close-range scanning even though the
    # correction is negligible; the server only sets this flag when a
    # correction actually moved enough to matter). When true, the graph's
    # earlier poses may have just shifted, so a normal incremental
    # get_cloud(since=last_pulled_id) pull is stale; the caller should
    # instead do a full get_cloud(since_node_id=0) resync and rebuild from
    # scratch.
    loop_closure: bool = False
    # id of the RTAB-Map node THIS frame became, or -1 if it didn't become a
    # new node (not every processed frame becomes one — RTAB-Map skips nodes
    # for insufficient displacement). Lets scan_session.py correlate its own
    # per-frame depth-consistency check (see feature_tracker.py) with the
    # SPECIFIC reconstructed node a bad frame produced, so that node can be
    # excluded when later pulled via get_cloud() — see rtabmap_server.cc's
    # wire-protocol comment for why this can't be determined any other way.
    node_id: int = -1


@dataclass
class ReconstructedNode:
    """One node's own reconstructed cloud from get_cloud()."""
    node_id: int
    pose: np.ndarray            # 4x4 c2w, world frame — RTAB-Map's CURRENT corrected pose
    points: np.ndarray          # Nx3 float32, world frame
    colors: np.ndarray          # Nx3 uint8, RGB


def _pose_bytes_to_c2w(tx: float, ty: float, tz: float,
                        qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    T[:3, 3] = [tx, ty, tz]
    return T


def _estimate_K(h: int, w: int) -> np.ndarray:
    f = max(w, h) * 0.8
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)


class RtabmapPoseClient:
    """
    One instance == one running RTAB-Map session (own Odometry + Rtabmap
    objects server-side — see the container's "single active session"
    limitation in rtabmap_docker/README.md).

    track_batch() sends one TRACK request per frame (RGB + depth) and blocks
    for the reply — REQ/REP is strictly request-then-reply, matching the
    synchronous per-frame style every other pose source in scan_session.py
    already uses (e.g. FeatureTracker.track()).
    """

    def __init__(self, addr: str, timeout: float = 2.0) -> None:
        self.addr = addr
        self.timeout = timeout
        self._ctx = zmq.Context.instance()
        self._sock = self._connect()

        try:
            self._sock.send(bytes([_CMD_PING]))
            reply = self._sock.recv()
        except zmq.error.ZMQError as exc:
            raise ConnectionError(f"Could not reach RTAB-Map server at {addr}: {exc}") from exc
        if not reply or reply[0] != _STATUS_PONG:
            raise ConnectionError(f"Unexpected PING reply from RTAB-Map server at {addr}")
        print(f"[RtabmapPoseClient] Connected to {addr}")

    def _connect(self) -> "zmq.Socket":
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.addr)
        return sock

    @property
    def connected(self) -> bool:
        return self._sock is not None

    # ── public ────────────────────────────────────────────────────────────────

    def track_batch(
        self,
        frames_rgb: list,
        depth_frames: List["DepthFrame"],
        frame_timestamps_ns: List[float],
    ) -> List[TrackedFrame]:
        """
        frames_rgb / depth_frames / frame_timestamps_ns: same length, strict
        chronological order. depth_frames[i].depth_map must be metres,
        same HxW as frames_rgb[i]; depth_frames[i].intrinsics used as this
        frame's camera model (falls back to _estimate_K if None — shouldn't
        happen since Step 1 of process_frames_batch always computes it, but
        defends against sending garbage over the wire).

        Returns one TrackedFrame per input frame (pose=None on tracking lost
        or a request timeout, e.g. the server process lagging or unreachable).
        """
        results: List[TrackedFrame] = []

        for rgb, df, ts_ns in zip(frames_rgb, depth_frames, frame_timestamps_ns):
            h, w = rgb.shape[:2]
            K = df.intrinsics if df.intrinsics is not None else _estimate_K(h, w)
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

            rgb_bytes = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
            depth_bytes = np.ascontiguousarray(df.depth_map, dtype=np.float32).tobytes()
            req = (
                bytes([_CMD_TRACK])
                + struct.pack(_HEADER_FMT, int(ts_ns), w, h, fx, fy, cx, cy)
                + rgb_bytes
                + depth_bytes
            )
            results.append(self._track_request(req, ts_ns))

        return results

    def get_cloud(
        self,
        since_node_id: int = 0,
        voxel_size: float = 0.02,
        max_depth: float = 0.0,
    ) -> Optional[List[ReconstructedNode]]:
        """
        Pulls RTAB-Map's own reconstructed surface for every node with
        id > since_node_id, using its CURRENT graph-corrected poses — see
        module docstring. voxel_size=0 disables server-side voxel downsample
        (matches scan_session.py's VOXEL_SIZE convention when set explicitly);
        max_depth=0 means no limit. Returns None on a request failure
        (timeout/disconnect) — caller should treat like a skipped batch, same
        as track_batch's per-frame None.
        """
        req = bytes([_CMD_GET_CLOUD]) + struct.pack(_GET_CLOUD_REQ_FMT, since_node_id, voxel_size, max_depth)
        try:
            self._sock.send(req)
            reply = self._sock.recv()
        except zmq.error.ZMQError as exc:
            print(f"[RtabmapPoseClient] get_cloud() failed ({exc}) — reconnecting.")
            self._reconnect()
            return None

        if not reply or reply[0] != _STATUS_OK:
            return None

        (node_count,) = struct.unpack("<i", reply[1:5])
        offset = 5
        node_hdr_size = struct.calcsize(_NODE_HDR_FMT)
        nodes: List[ReconstructedNode] = []
        for _ in range(node_count):
            node_id, tx, ty, tz, qx, qy, qz, qw, point_count = struct.unpack(
                _NODE_HDR_FMT, reply[offset:offset + node_hdr_size]
            )
            offset += node_hdr_size
            xyz_bytes = point_count * 3 * 4
            rgb_bytes = point_count * 3
            if point_count > 0:
                points = np.frombuffer(reply[offset:offset + xyz_bytes], dtype=np.float32).reshape(-1, 3)
                offset += xyz_bytes
                colors = np.frombuffer(reply[offset:offset + rgb_bytes], dtype=np.uint8).reshape(-1, 3)
                offset += rgb_bytes
            else:
                points = np.zeros((0, 3), dtype=np.float32)
                colors = np.zeros((0, 3), dtype=np.uint8)
            nodes.append(ReconstructedNode(
                node_id=node_id,
                pose=_pose_bytes_to_c2w(tx, ty, tz, qx, qy, qz, qw),
                points=points,
                colors=colors,
            ))
        return nodes

    def reset(self) -> None:
        """Sends CMD_RESET (fully reinits Odometry + Rtabmap server-side);
        reconnects on failure."""
        try:
            self._sock.send(bytes([_CMD_RESET]))
            self._sock.recv()
        except zmq.error.ZMQError as exc:
            print(f"[RtabmapPoseClient] reset() failed: {exc} — reconnecting.")
            self._reconnect()

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    # ── private ───────────────────────────────────────────────────────────────

    def _track_request(self, req: bytes, ts_ns: float) -> TrackedFrame:
        try:
            self._sock.send(req)
            reply = self._sock.recv()
        except zmq.error.ZMQError as exc:
            print(
                f"[RtabmapPoseClient] Request at t={ts_ns:.0f} ns failed ({exc}) "
                f"— reconnecting."
            )
            self._reconnect()
            return TrackedFrame(pose=None)

        if not reply or reply[0] != _STATUS_OK:
            return TrackedFrame(pose=None)
        pose_size = struct.calcsize(_POSE_FMT)
        pose = _pose_bytes_to_c2w(*struct.unpack(_POSE_FMT, reply[1:1 + pose_size]))
        tail_size = struct.calcsize(_TRACK_TAIL_FMT)
        tail_off = 1 + pose_size
        if len(reply) >= tail_off + tail_size:
            loop_closure_flag, node_id = struct.unpack(
                _TRACK_TAIL_FMT, reply[tail_off:tail_off + tail_size]
            )
            loop_closure = bool(loop_closure_flag)
        else:
            # Older server build without the node_id field — degrade
            # gracefully rather than crash (loop_closure alone, no node
            # veto capability, same as before this field existed).
            loop_closure = bool(reply[tail_off]) if len(reply) > tail_off else False
            node_id = -1
        return TrackedFrame(pose=pose, loop_closure=loop_closure, node_id=node_id)

    def _reconnect(self) -> None:
        # A timed-out/errored REQ socket is stuck mid-transaction (REQ enforces
        # strict send/recv alternation) — recreate it rather than trying to
        # recover the existing one.
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
        self._sock = self._connect()
