"""
stream_simulator — replays an already-recorded dataset/ folder
(images/ + camera.csv + imu.csv, the same format scan_gui.py's batch Scan tab
reads) as if it were arriving live, frame-by-frame and IMU-sample-by-sample,
driving a StreamingScanSession through its public push_frame/push_imu/
start_zone/end_zone API only.

This exists to prove the streaming interface (stream_session.py) end-to-end
before a real live source (Android gRPC stream) exists — see CLAUDE.md's
3D Scanning Pipeline section. A future real-time driver is expected to call
the exact same StreamingScanSession methods this file calls; only the event
source changes (a socket instead of a sorted replay of on-disk files).

The Segment Table (start_s, end_s, zone_name — the same input the batch tab's
Gradio UI takes) has no live equivalent yet for a real operator, so this
simulator fires start_zone/end_zone at the moments its boundaries are crossed
during replay — standing in for a real operator's button presses.
"""

import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from stream_session import StreamingScanSession


def _read_camera_index(dataset_path: str) -> List[Tuple[int, str]]:
    base = Path(dataset_path)
    csv_path = base / "camera.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    rows = sorted(zip(df["timestamp_ns"].tolist(), df["filename"].tolist()), key=lambda r: r[0])
    return [(int(ts), str(base / "images" / fname)) for ts, fname in rows]


def _read_imu_samples(dataset_path: str) -> List[tuple]:
    csv_path = Path(dataset_path) / "imu.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path).rename(columns={"timestamp_ns": "ts"})
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return list(
        zip(
            df["ts"].astype(np.int64).tolist(),
            df["ax"].tolist(), df["ay"].tolist(), df["az"].tolist(),
            df["gx"].tolist(), df["gy"].tolist(), df["gz"].tolist(),
        )
    )


def _subsample_interval(frame_ts: List[int], fps_val: float) -> int:
    if len(frame_ts) < 2:
        return 1
    deltas = np.diff(frame_ts)
    dataset_fps = float(1e9 / np.median(deltas)) if len(deltas) else fps_val
    return max(1, round(dataset_fps / max(fps_val, 0.1)))


def _correct_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    rotation = rotation % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _resize_frame(frame: np.ndarray, max_dim: int) -> np.ndarray:
    if max_dim <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def build_event_timeline(
    dataset_path: str, fps_val: float = 5.0, native_fps: bool = False,
) -> dict:
    """
    Reads dataset_path's camera.csv/imu.csv, subsamples frames to ~fps_val
    using the dataset's actual frame rate, and returns one timestamp-sorted
    event timeline: {"events": [("frame"|"imu", ts_ns, payload), ...],
    "t0": int, "t1": int, "total_ns": int} (plus "error" if camera.csv is
    missing). Shared by replay_dataset() (auto-driven) and
    ManualDatasetReplayer (single-step-driven) so both replay modes see the
    identical sampled frame set.

    native_fps=True skips fps subsampling entirely (interval=1, every
    recorded frame included) — set by callers when RTAB-Map pose mode is
    active. RTAB-Map's own frame-to-frame visual odometry needs enough
    overlap between consecutive TRACKED frames to match features; skipping
    frames via fps subsampling increases inter-frame motion, which measurably
    increases its tracking-lost rate (verified: ~10% lost at fps=10 on a real
    recording, climbing to ~70% at fps=0.5) — every lost frame contributes no
    node to the reconstruction at all. VO/IMU+VO/DA3 poses tolerate sparser
    frame rates fine (see 3D Scanning Pipeline), so they keep normal
    subsampling; only RTAB-Map needs this override.
    """
    camera_index = _read_camera_index(dataset_path)
    imu_samples = _read_imu_samples(dataset_path)
    if not camera_index:
        return {"error": f"No camera.csv found under {dataset_path}.",
                "events": [], "t0": 0, "t1": 0, "total_ns": 1}

    frame_ts = [ts for ts, _ in camera_index]
    interval = 1 if native_fps else _subsample_interval(frame_ts, fps_val)
    sampled = [(ts, path) for i, (ts, path) in enumerate(camera_index) if i % interval == 0]
    # i % interval == 0 always includes frame 0 but has no guarantee of
    # landing on the LAST frame — at a coarse interval (low fps_val), the
    # gap between the last sampled index and the true end grows (up to
    # interval-1 frames), so replay/manual stepping would reach
    # has_more() == False well before the recording's actual end. Always
    # include the dataset's true last frame so nothing at the tail is ever
    # silently dropped, regardless of fps_val.
    if sampled and sampled[-1][0] != camera_index[-1][0]:
        sampled.append(camera_index[-1])

    t0 = camera_index[0][0]
    t1 = camera_index[-1][0]
    total_ns = max(1, t1 - t0)

    events: List[tuple] = [("frame", ts, path) for ts, path in sampled]
    events += [("imu", s[0], s) for s in imu_samples]
    events.sort(key=lambda e: e[1])
    return {"events": events, "t0": t0, "t1": t1, "total_ns": total_ns}


def replay_dataset(
    stream: StreamingScanSession,
    dataset_path: str,
    segments: List[tuple],
    fps_val: float = 5.0,
    max_dim: int = 0,
    extra_rotation: int = 0,
    realtime: bool = False,
    speed: float = 1.0,
) -> Generator[dict, None, None]:
    """
    Replays dataset_path's frames + IMU samples in timestamp order, firing
    start_zone/end_zone at each segment's [start_s, end_s) boundary (in the
    dataset's own clock, t0-relative — same convention scan_gui.py's batch
    tab uses). Yields a progress dict after every event:
        {"kind": "imu"|"frame"|"zone_start"|"zone_end", "ts_ns": int,
         "progress": float in [0,1], "result": <push_frame's return or None>,
         "zone": <name, for zone events>}

    `realtime` paces playback to wall-clock time scaled by `speed` (2.0 = 2x
    real time); default is max-speed replay for fast iteration.
    """
    timeline = build_event_timeline(dataset_path, fps_val, native_fps=stream._use_rtabmap)
    if timeline.get("error"):
        yield {"kind": "error", "message": timeline["error"]}
        return
    events = timeline["events"]
    t0, t1, total_ns = timeline["t0"], timeline["t1"], timeline["total_ns"]

    # Segment boundaries in absolute ns, sorted by start — walked with a
    # single pointer as the replay's timestamp advances (one pass, not a
    # per-event rescan).
    seg_bounds = sorted(
        (
            (t0 + start_s * 1e9, t0 + end_s * 1e9, zone_name)
            for start_s, end_s, zone_name in segments
        ),
        key=lambda s: s[0],
    )
    seg_idx = 0
    active_zone: Optional[str] = None
    active_end_ns: Optional[float] = None
    wall_start = time.monotonic()
    replay_start_ns = events[0][1] if events else t0

    for kind, ts_ns, payload in events:
        if active_zone is not None and ts_ns >= active_end_ns:
            stream.end_zone()
            yield {"kind": "zone_end", "ts_ns": ts_ns, "zone": active_zone,
                   "progress": (ts_ns - t0) / total_ns, "result": None}
            active_zone, active_end_ns = None, None

        while seg_idx < len(seg_bounds) and seg_bounds[seg_idx][0] <= ts_ns:
            start_ns, end_ns, zone_name = seg_bounds[seg_idx]
            seg_idx += 1
            if ts_ns >= end_ns:
                continue  # this segment's window has already fully passed
            if active_zone is not None:
                stream.end_zone()
                yield {"kind": "zone_end", "ts_ns": ts_ns, "zone": active_zone,
                       "progress": (ts_ns - t0) / total_ns, "result": None}
            if zone_name:
                stream.start_zone(zone_name)
                active_zone, active_end_ns = zone_name, end_ns
                yield {"kind": "zone_start", "ts_ns": ts_ns, "zone": zone_name,
                       "progress": (ts_ns - t0) / total_ns, "result": None}
            else:
                active_zone, active_end_ns = None, None

        if realtime:
            target_wall = wall_start + (ts_ns - replay_start_ns) * 1e-9 / max(speed, 1e-3)
            delay = target_wall - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        if kind == "imu":
            ts, ax, ay, az, gx, gy, gz = payload
            stream.push_imu(ts, ax, ay, az, gx, gy, gz)
            yield {"kind": "imu", "ts_ns": ts_ns, "progress": (ts_ns - t0) / total_ns, "result": None}
        else:
            frame = cv2.imread(payload)
            if frame is None:
                continue
            frame = _correct_rotation(frame, extra_rotation)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = _resize_frame(rgb, max_dim)
            result = stream.push_frame(rgb, ts_ns)
            yield {"kind": "frame", "ts_ns": ts_ns, "progress": (ts_ns - t0) / total_ns, "result": result}

    if active_zone is not None:
        stream.end_zone()
        yield {"kind": "zone_end", "ts_ns": t1, "zone": active_zone, "progress": 1.0, "result": None}


class ManualDatasetReplayer:
    """
    Single-step counterpart to replay_dataset(): instead of an auto-driven
    generator, exposes peek_next_frame_preview()/has_more()/step() so a GUI
    button can feed exactly one frame — plus any IMU samples/zone boundaries
    that fall before it in the recorded timeline — into a
    StreamingScanSession per click. Lets the GUI preview the upcoming frame
    before it's fed. Shares build_event_timeline()'s dataset reading/sampling
    with replay_dataset() so both replay modes see the identical sampled
    frame set; only the driver (an auto loop vs. one call per click) differs
    — same relationship this whole module already has to a future real
    Android live source (see module docstring).
    """

    def __init__(
        self,
        stream: StreamingScanSession,
        dataset_path: str,
        segments: List[tuple],
        fps_val: float = 5.0,
        max_dim: int = 0,
        extra_rotation: int = 0,
    ) -> None:
        self.stream = stream
        self.max_dim = max_dim
        self.extra_rotation = extra_rotation

        timeline = build_event_timeline(dataset_path, fps_val, native_fps=stream._use_rtabmap)
        self.error: Optional[str] = timeline.get("error")
        self.events: List[tuple] = timeline["events"]
        self.t0: int = timeline["t0"]
        self.t1: int = timeline["t1"]
        self.total_ns: int = timeline["total_ns"]
        self.idx = 0
        # scan_gui.py's _manual_stream_feed's skip-if-unchanged Live
        # Points/Voxelization optimization — same role as the other two
        # replay modes' local `_last_render_point_count`, just living on the
        # replayer instance since it must survive across separate per-click
        # Gradio callback invocations rather than one generator's closure.
        self.last_render_point_count = -1

        self.seg_bounds = sorted(
            (
                (self.t0 + start_s * 1e9, self.t0 + end_s * 1e9, zone_name)
                for start_s, end_s, zone_name in segments
            ),
            key=lambda s: s[0],
        )
        self.seg_idx = 0
        self.active_zone: Optional[str] = None
        self.active_end_ns: Optional[float] = None

    def has_more(self) -> bool:
        return any(kind == "frame" for kind, _, _ in self.events[self.idx:])

    def peek_next_frame_path(self) -> Optional[str]:
        for kind, _, payload in self.events[self.idx:]:
            if kind == "frame":
                return payload
        return None

    def peek_next_frame_preview(self) -> Optional[np.ndarray]:
        """Loads (without consuming) the next not-yet-fed frame, for the GUI
        to show as an "about to be fed" preview — same rotation/resize this
        frame will actually get once step() reaches it."""
        path = self.peek_next_frame_path()
        if path is None:
            return None
        frame = cv2.imread(path)
        if frame is None:
            return None
        frame = _correct_rotation(frame, self.extra_rotation)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return _resize_frame(rgb, self.max_dim)

    def step(self) -> dict:
        """
        Applies every IMU sample / zone-boundary crossing up to the next
        not-yet-fed frame, then pushes exactly that one frame. Returns
        {"kind": "frame", "ts_ns", "progress", "result", "zone_events": [...]}
        for the frame just fed, or {"kind": "done", "zone_events": [...]}
        once no frame events remain (any still-open zone is closed here,
        mirroring replay_dataset()'s own end-of-replay cleanup).
        """
        zone_events: List[dict] = []
        while self.idx < len(self.events):
            kind, ts_ns, payload = self.events[self.idx]

            if self.active_zone is not None and ts_ns >= self.active_end_ns:
                self.stream.end_zone()
                zone_events.append({"kind": "zone_end", "zone": self.active_zone,
                                     "progress": (ts_ns - self.t0) / self.total_ns})
                self.active_zone, self.active_end_ns = None, None

            while self.seg_idx < len(self.seg_bounds) and self.seg_bounds[self.seg_idx][0] <= ts_ns:
                start_ns, end_ns, zone_name = self.seg_bounds[self.seg_idx]
                self.seg_idx += 1
                if ts_ns >= end_ns:
                    continue  # this segment's window has already fully passed
                if self.active_zone is not None:
                    self.stream.end_zone()
                    zone_events.append({"kind": "zone_end", "zone": self.active_zone,
                                         "progress": (ts_ns - self.t0) / self.total_ns})
                if zone_name:
                    self.stream.start_zone(zone_name)
                    self.active_zone, self.active_end_ns = zone_name, end_ns
                    zone_events.append({"kind": "zone_start", "zone": zone_name,
                                         "progress": (ts_ns - self.t0) / self.total_ns})
                else:
                    self.active_zone, self.active_end_ns = None, None

            if kind == "imu":
                ts, ax, ay, az, gx, gy, gz = payload
                self.stream.push_imu(ts, ax, ay, az, gx, gy, gz)
                self.idx += 1
                continue

            # kind == "frame" — feed exactly this one, then stop.
            self.idx += 1
            frame = cv2.imread(payload)
            if frame is None:
                continue
            frame = _correct_rotation(frame, self.extra_rotation)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = _resize_frame(rgb, self.max_dim)
            result = self.stream.push_frame(rgb, ts_ns)
            return {
                "kind": "frame", "ts_ns": ts_ns,
                "progress": (ts_ns - self.t0) / self.total_ns,
                "result": result, "zone_events": zone_events,
            }

        if self.active_zone is not None:
            self.stream.end_zone()
            zone_events.append({"kind": "zone_end", "zone": self.active_zone, "progress": 1.0})
            self.active_zone = None
        return {"kind": "done", "zone_events": zone_events}
