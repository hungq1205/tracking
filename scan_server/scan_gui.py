"""
Scan Server GUI — 3 tabs:
  1. Live Points   — interactive 3D model viewer (PLY), updated after each batch
  2. Depth Metric  — per-frame depth map + click-to-measure
  3. Occupancy Map — 2D top-down grid (free / unknown / occupied)

Workflow:
  1. Upload a video + set Location ID
  2. Fill Segment Table: start_s | end_s | zone_name
  3. Click Scan → SLAM processes each segment in order
  4. Click Export Map → saves PLY + JSON + ORB keyframes
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import cv2
import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")

from scan_css import SCAN_CSS, SCAN_DESCRIPTION_HTML, SCAN_HEADER_HTML, get_scan_theme

_DEFAULT_SEGMENTS = pd.DataFrame(
    {"start_s": [0.0], "end_s": [0.0], "area_name": [""]}
)


# ── Depth helpers ──────────────────────────────────────────────────────────────


def _colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    import matplotlib.pyplot as plt
    valid = depth_map[depth_map > 0]
    if len(valid) == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    d_min, d_max = float(valid.min()), float(valid.max())
    norm = np.clip(1.0 - (depth_map - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
    return (plt.get_cmap("plasma")(norm)[:, :, :3] * 255).astype(np.uint8)


def _build_depth_data(
    frames_rgb: list,
    depth_frames: list,
    frame_poses: Optional[List] = None,
) -> Optional[Dict]:
    if not frames_rgb or not depth_frames:
        return None
    return {
        i: {
            "image": rgb.astype(np.uint8),
            "depth": df.depth_map,
            "rays": df.rays,
            "intrinsics": df.intrinsics,
            "depth_vis": _colorize_depth(df.depth_map),
            "pose": frame_poses[i] if frame_poses and i < len(frame_poses) else df.camera_pose,
        }
        for i, (rgb, df) in enumerate(zip(frames_rgb, depth_frames))
    }


# ── Zone / landmark rendering helpers ─────────────────────────────────────────

_ZONE_RGBA = [
    (255,  80,  80, 220),   # red
    ( 80, 210,  80, 220),   # green
    ( 80, 130, 255, 220),   # blue
    (255, 200,  50, 220),   # yellow
    (200,  80, 200, 220),   # magenta
    ( 50, 210, 210, 220),   # cyan
    (255, 140,   0, 220),   # orange
    (160, 100, 200, 220),   # purple
]

def _zone_color_css(idx: int) -> str:
    r, g, b, a = _ZONE_RGBA[idx % len(_ZONE_RGBA)]
    return f"rgba({r},{g},{b},{a/255:.2f})"

def _make_stick(p1, p2, radius: float = 0.03, color=(255, 80, 80, 220)):
    """Return a thin box mesh along the p1→p2 edge, or None on failure."""
    try:
        import trimesh
        d = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
        length = float(np.linalg.norm(d))
        if length < 1e-4:
            return None
        mid = (np.asarray(p1) + np.asarray(p2)) * 0.5
        d_norm = d / length
        z_axis = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_axis, d_norm)
        cross_len = float(np.linalg.norm(cross))
        dot_val = float(np.dot(z_axis, d_norm))
        if cross_len > 1e-6:
            angle = np.arctan2(cross_len, dot_val)
            R = trimesh.transformations.rotation_matrix(angle, cross / cross_len)
        elif dot_val < 0:
            R = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
        else:
            R = np.eye(4)
        T = np.eye(4)
        T[:3, 3] = mid
        box = trimesh.creation.box(extents=[radius * 2, radius * 2, length])
        box.apply_transform(T @ R)
        box.visual.face_colors = np.array(color, dtype=np.uint8)
        return box
    except Exception:
        return None


# ── Point cloud → Model3D ──────────────────────────────────────────────────────


def _cloud_to_glb(cloud_or_pts, zones=None, ground_y=None) -> Optional[str]:
    """
    Export point cloud to a temp GLB file for gr.Model3D.
    Uses trimesh (same as DA3 app): PointCloud → Scene → .glb
    Sets an initial top-down camera so the 3D viewer opens from above.
    """
    try:
        import trimesh

        if hasattr(cloud_or_pts, "points"):
            pts = np.asarray(cloud_or_pts.points, dtype=np.float32)
            has_color = cloud_or_pts.has_colors()
            colors_f = np.asarray(cloud_or_pts.colors) if has_color else None
        else:
            pts_raw, colors_raw = cloud_or_pts
            pts = np.asarray(pts_raw, dtype=np.float32)
            colors_f = np.asarray(colors_raw) if colors_raw is not None else None
            has_color = colors_f is not None and len(colors_f) == len(pts)

        if len(pts) == 0:
            return None

        # trimesh expects RGBA uint8 colors
        if has_color and len(colors_f) == len(pts):
            rgb8 = (np.clip(colors_f, 0.0, 1.0) * 255).astype(np.uint8)
            alpha = np.full((len(pts), 1), 255, dtype=np.uint8)
            rgba = np.hstack([rgb8, alpha])
        else:
            rgba = np.full((len(pts), 4), [180, 180, 180, 255], dtype=np.uint8)

        pc = trimesh.points.PointCloud(vertices=pts, colors=rgba)
        scene = trimesh.Scene()
        scene.add_geometry(pc)

        # ── Zone bounding boxes + landmark spheres ────────────────────────────
        if zones:
            # Floor Y: use provided ground_y or estimate from point cloud centroid
            gy = float(ground_y) if ground_y is not None else float(pts[:, 1].max())
            for z_idx, zone in enumerate(zones):
                color = _ZONE_RGBA[z_idx % len(_ZONE_RGBA)]
                mn = np.array(zone.bbox_min, dtype=np.float64)
                mx = np.array(zone.bbox_max, dtype=np.float64)

                # Draw the floor rectangle of the AABB as 4 edge sticks
                floor_y = gy
                fc = np.array([
                    [mn[0], floor_y, mn[2]],
                    [mx[0], floor_y, mn[2]],
                    [mx[0], floor_y, mx[2]],
                    [mn[0], floor_y, mx[2]],
                ])
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
                    stick = _make_stick(fc[a], fc[b], radius=0.04, color=color)
                    if stick:
                        scene.add_geometry(stick)

                # Landmark spheres placed at floor level
                for lm in getattr(zone, "landmarks", []):
                    sph = trimesh.creation.icosphere(subdivisions=1, radius=0.18)
                    sph.apply_translation([float(lm.x), floor_y, float(lm.z)])
                    sph.visual.face_colors = np.array(color, dtype=np.uint8)
                    scene.add_geometry(sph)

        # Top-down initial camera: camera placed above centroid looking down (-Y).
        # In trimesh camera space: +X=right, +Y=up, -Z=forward (toward scene).
        # For top-down: camera -Z (forward) aligns with world -Y (down),
        #   so camera +Z column = world +Y = [0,1,0]
        #       camera +X column = world +X = [1,0,0]
        #       camera +Y column = world -Z = [0,0,-1]  (right-hand cross product)
        centroid = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        view_dist = float(max(extent)) * 1.5 + 1.5

        cam_R = np.array([
            [1.0,  0.0,  0.0],
            [0.0,  0.0, -1.0],
            [0.0,  1.0,  0.0],
        ], dtype=np.float64)
        cam_T = np.eye(4, dtype=np.float64)
        cam_T[:3, :3] = cam_R
        cam_T[:3, 3] = [centroid[0], centroid[1] + view_dist, centroid[2]]
        scene.camera_transform = cam_T

        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        tmp.close()
        scene.export(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[cloud_to_glb] EXCEPTION: {e}")
        return None


# ── Frame-explorer helpers ─────────────────────────────────────────────────────


def _euler_from_R(R: np.ndarray):
    """ZYX Euler angles (roll, pitch, yaw) in degrees from a 3×3 rotation matrix."""
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
    return roll, pitch, yaw


# Axis label → (column_index, sign) for building permutation matrix
_AXIS_MAP = {"+X": (0, 1), "+Y": (1, 1), "+Z": (2, 1),
             "-X": (0,-1), "-Y": (1,-1), "-Z": (2,-1)}


def _perm_matrix(roll_src: str, pitch_src: str, yaw_src: str) -> np.ndarray:
    """
    Build a 3×3 signed permutation matrix P such that the remapped rotation is
    P @ R @ P.T, and Euler angles of the result give (roll, pitch, yaw) drawn
    from the selected source axes.
    """
    P = np.zeros((3, 3), dtype=np.float64)
    for row, src in enumerate([roll_src, pitch_src, yaw_src]):
        col, sign = _AXIS_MAP.get(src, (row, 1))
        P[row, col] = sign
    return P


def _back_project_frames(all_frames: list, start: int, end: int,
                         max_depth: float = 10.0, perm: Optional[np.ndarray] = None):
    """
    Back-project a subset of stored frames to (Nx3 pts, Nx3 colors) without
    re-running depth estimation.  Returns (pts, colors) or None if empty.
    """
    pts_list, col_list = [], []
    for rgb, df, pose in all_frames[start:end + 1]:
        depth = df.depth_map
        rays  = df.rays
        mask  = (depth > 0.1) & (depth < max_depth)
        if mask.sum() < 10:
            continue
        if rays is not None:
            pts_cam = (rays[mask] * depth[mask, np.newaxis]).astype(np.float64)
        else:
            h, w = depth.shape
            fx = fy = max(w, h) * 0.8
            cx, cy = w / 2.0, h / 2.0
            if df.intrinsics is not None:
                K = df.intrinsics.astype(np.float64)
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            ys, xs = np.where(mask)
            zs = depth[mask].astype(np.float64)
            pts_cam = np.stack([(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs], axis=-1)
        # Apply axis permutation to the pose rotation before back-projecting
        p_use = pose.copy()
        if perm is not None:
            p_use[:3, :3] = perm @ pose[:3, :3] @ perm.T
        ones = np.ones((len(pts_cam), 1), dtype=np.float64)
        pts_world = (p_use @ np.hstack([pts_cam, ones]).T).T[:, :3]
        pts_list.append(pts_world.astype(np.float32))
        col_list.append(rgb[mask].astype(np.float32) / 255.0)
    if not pts_list:
        return None
    return np.vstack(pts_list), np.vstack(col_list)


def _format_poses(all_frames: list, start: int, end: int,
                  perm: Optional[np.ndarray] = None) -> str:
    """Return a human-readable pose table for the selected frame range."""
    if not all_frames:
        return "No frames stored yet — run Scan first."
    n = len(all_frames)
    start = max(0, min(start, n - 1))
    end   = max(start, min(end, n - 1))
    lines = [f"Frames {start + 1} – {end + 1}  (total stored: {n})\n"]
    for i in range(start, end + 1):
        _, _, pose = all_frames[i]
        pos = pose[:3, 3]
        R = pose[:3, :3]
        if perm is not None:
            R = perm @ R @ perm.T
        roll, pitch, yaw = _euler_from_R(R)
        lines.append(
            f"  Frame {i + 1:>4d} | "
            f"pos  x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}  |  "
            f"rot  roll={roll:+.1f}°  pitch={pitch:+.1f}°  yaw={yaw:+.1f}°"
        )
        if end - start == 0:
            lines.append("")
            lines.append("  4×4 pose matrix (camera-to-world):")
            for row in pose:
                lines.append("    " + "  ".join(f"{v:+.6f}" for v in row))
    return "\n".join(lines)


# ── UI factory ─────────────────────────────────────────────────────────────────


def create_scan_ui(scan_manager, upload_dir: Optional[str] = None) -> gr.Blocks:

    _upload_dir = Path(upload_dir) if upload_dir else None

    # ── helpers ────────────────────────────────────────────────────────────────

    def _list_uploads() -> List[str]:
        if _upload_dir is None or not _upload_dir.exists():
            return []
        return sorted(p.name for p in _upload_dir.iterdir() if p.is_dir())

    def _load_upload(scan_id: Optional[str]):
        if not scan_id or _upload_dir is None:
            return None, None
        base = _upload_dir / scan_id
        video = str(base / "video.mp4") if (base / "video.mp4").exists() else None
        imu = str(base / "imu_data.csv") if (base / "imu_data.csv").exists() else None
        return video, imu

    def _correct_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
        rotation = rotation % 360
        if rotation == 90:   return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:  return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _resize_frame(frame: np.ndarray, max_dim: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = max_dim / max(h, w)
        if scale >= 1.0:
            return frame
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _collect_frames_in_range(
        video_path: str, start_s: float, end_s: float, fps_val: float,
        max_dim: int = 0, extra_rotation: int = 0,
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = (int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) + extra_rotation) % 360
        interval = max(1, int(video_fps / max(fps_val, 0.1)))
        start_frame = max(0, int(start_s * video_fps))
        end_frame = min(int(end_s * video_fps) if end_s > 0 else total, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames, idx = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) > end_frame:
                break
            if idx % interval == 0:
                frame = _correct_rotation(frame, rotation)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if max_dim > 0:
                    rgb = _resize_frame(rgb, max_dim)
                frames.append(rgb)
            idx += 1
        cap.release()
        return frames

    def _parse_segments(df) -> List[tuple]:
        if df is None or (hasattr(df, "empty") and df.empty):
            return [(0.0, 9999.0, "")]
        segments = []
        for _, row in df.iterrows():
            try:
                start_s = float(row.get("start_s") or 0.0)
                end_s = float(row.get("end_s") or 9999.0)
                # Support both new "area_name" and old "zone_name" column names
                zone = str(
                    row.get("area_name") or row.get("zone_name") or ""
                ).strip()
                if end_s > start_s:
                    segments.append((start_s, end_s, zone))
            except (ValueError, TypeError):
                continue
        return segments or [(0.0, 9999.0, "")]

    def _get_depth_view(depth_data, idx: int):
        if not depth_data:
            return None, None
        keys = list(depth_data.keys())
        idx = max(0, min(idx, len(keys) - 1))
        d = depth_data[keys[idx]]
        return d["image"], d["depth_vis"]

    def _navigate_depth(depth_data, selector: str, direction: int):
        if not depth_data:
            return "View 1", None, None, []
        n = len(depth_data)
        try:
            cur = int(selector.split()[1]) - 1
        except Exception:
            cur = 0
        new_idx = (cur + direction) % n
        rgb, dvis = _get_depth_view(depth_data, new_idx)
        return f"View {new_idx + 1}", rgb, dvis, []

    def _update_depth_selector(depth_data, selector: str):
        if not depth_data or not selector:
            return None, None, []
        try:
            idx = int(selector.split()[1]) - 1
        except Exception:
            idx = 0
        rgb, dvis = _get_depth_view(depth_data, idx)
        return rgb, dvis, []

    def _do_measure(depth_data, measure_points, selector: str, evt: gr.SelectData):
        if not depth_data:
            return None, [], "No depth data."
        try:
            idx = int(selector.split()[1]) - 1
        except Exception:
            idx = 0
        keys = list(depth_data.keys())
        idx = max(0, min(idx, len(keys) - 1))
        view = depth_data[keys[idx]]
        image = view["image"].copy().astype(np.uint8)
        depth = view["depth"]
        rays = view.get("rays")
        intrinsics = view.get("intrinsics")
        point = (int(evt.index[0]), int(evt.index[1]))
        measure_points = list(measure_points) + [point]
        for p in measure_points:
            if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
                cv2.circle(image, p, radius=6, color=(255, 50, 50), thickness=2)
        text_lines = []
        for i, p in enumerate(measure_points):
            if 0 <= p[1] < depth.shape[0] and 0 <= p[0] < depth.shape[1]:
                d = float(depth[p[1], p[0]])
                text_lines.append(f"- **P{i+1}** ({p[0]}, {p[1]}): **{d:.3f} m**")
        if len(measure_points) == 2:
            p1, p2 = measure_points
            cv2.line(image, p1, p2, color=(255, 50, 50), thickness=2)
            if (0 <= p1[1] < depth.shape[0] and 0 <= p1[0] < depth.shape[1]
                    and 0 <= p2[1] < depth.shape[0] and 0 <= p2[0] < depth.shape[1]):
                d3 = None
                if rays is not None:
                    d3 = float(np.linalg.norm(
                        rays[p1[1], p1[0]] * depth[p1[1], p1[0]]
                        - rays[p2[1], p2[0]] * depth[p2[1], p2[0]]
                    ))
                elif intrinsics is not None:
                    K = intrinsics
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    d1 = float(depth[p1[1], p1[0]])
                    d2 = float(depth[p2[1], p2[0]])
                    pt1 = np.array([(p1[0] - cx) * d1 / fx, (p1[1] - cy) * d1 / fy, d1])
                    pt2 = np.array([(p2[0] - cx) * d2 / fx, (p2[1] - cy) * d2 / fy, d2])
                    d3 = float(np.linalg.norm(pt1 - pt2))
                if d3 is not None:
                    text_lines.append(f"- **3D Distance: {d3:.3f} m**")
                else:
                    text_lines.append("- **3D Distance: unavailable (no rays or intrinsics)**")
            measure_points = []
        return image, measure_points, "\n".join(text_lines)

    # ── event handlers ─────────────────────────────────────────────────────────

    def _handle_video_upload(video_path: Optional[str], fps_val: float, extra_rotation: int = 0):
        if not video_path:
            return None, "Upload a video to preview.", _DEFAULT_SEGMENTS
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rotation = (int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) + extra_rotation) % 360
        duration = total / video_fps if video_fps > 0 else 0.0
        interval = max(1, int(video_fps / max(fps_val, 0.1)))
        frames, idx = [], 0
        while cap.isOpened() and len(frames) < 16:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frame = _correct_rotation(frame, rotation)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()
        approx = max(1, int(total / max(interval, 1)))
        msg = f"Video: {duration:.1f}s — ~{approx} sampled frames at {fps_val} FPS"
        default_df = pd.DataFrame(
            {"start_s": [0.0], "end_s": [round(duration, 1)], "area_name": [""]}
        )
        return frames, msg, default_df

    def _run_local_scan(
        video_path: Optional[str],
        imu_file_path: Optional[str],
        fps_val: float,
        batch_size: int,
        location_id: str,
        segments_df,
        resolution: str,
        pose_src: str = "Auto",
        extra_rotation: int = 0,
        zone_type: str = "",
    ) -> Generator[Dict[str, Any], None, None]:
        if not video_path:
            yield {log_output: "Upload a video first."}
            return

        max_dim = int(resolution.split("×")[0]) if resolution != "Original" else 0
        location_id = (location_id or "").strip() or "default"
        zone_type = (zone_type or "").strip()
        segments = _parse_segments(segments_df)
        session = scan_manager.get_or_create(location_id, zone_type=zone_type)

        # Resolve effective pose source
        _use_imu  = pose_src in ("Auto", "IMU + VO")
        _use_da3  = pose_src == "DA3 poses"
        # _use_vo   = pose_src in ("VO only",) — default when neither flag set

        # Load IMU data (only if an IMU-based mode is requested)
        if _use_imu and imu_file_path and Path(imu_file_path).exists():
            yield {log_output: f"Loading IMU data from {Path(imu_file_path).name}…"}
            try:
                session.set_imu_file(imu_file_path)
                yield {log_output: f"[{pose_src}] IMU data loaded — poses pre-computed before depth estimation."}
            except Exception as e:
                yield {log_output: f"IMU load failed ({e}), falling back to VO."}
                _use_imu = False
        elif _use_imu and pose_src == "IMU + VO":
            yield {log_output: "IMU + VO selected but no IMU file provided — falling back to VO."}
            _use_imu = False
        elif _use_da3:
            yield {log_output: "[DA3 poses] Using DA3 model camera_pose — requires PyTorch DA3Estimator (not ONNX)."}
        else:
            yield {log_output: f"[{pose_src}] Using FeatureTracker VO for pose estimation."}

        # Read video FPS for IMU timestamp mapping
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps_val
        cap.release()

        MINI_BATCH = max(1, int(batch_size))
        depth_accum: list = []
        all_frames_accum: list = []   # (rgb, DepthFrame, pose) for every processed frame
        total_infer_ms = 0.0
        total_frames_processed = 0

        for seg_i, (start_s, end_s, zone_name) in enumerate(segments, 1):
            label = f"'{zone_name}'" if zone_name else "(unlabelled)"
            yield {log_output: f"Segment {seg_i}/{len(segments)} {label}: "
                               f"extracting frames {start_s:.1f}s – {end_s:.1f}s…"}

            frames_rgb = _collect_frames_in_range(video_path, start_s, end_s, fps_val, max_dim, extra_rotation)
            if not frames_rgb:
                yield {log_output: f"Segment {seg_i}: no frames in range, skipping."}
                continue

            # Set area context for semantic landmark accumulation
            session._current_area_name = zone_name
            session._raw_landmarks = []

            n_total = len(frames_rgb)
            segment_positions: list = []
            point_count, cam_pos = 0, [0.0, 0.0, 0.0]
            seg_infer_ms = 0.0

            # ── Pass 1: pre-compute all IMU poses for this segment ─────────────
            # Use sampling FPS (fps_val), not video FPS — extracted frame i is at
            # start_s + i/fps_val seconds, not start_s + i/video_fps.
            seg_imu_poses: Optional[List] = (
                session.compute_segment_poses(n_total, start_s, fps_val)
                if _use_imu else None
            )
            if seg_imu_poses is not None:
                yield {log_output: (
                    f"Seg {seg_i}/{len(segments)} {label} — "
                    f"{n_total} IMU poses computed, starting depth estimation…"
                )}

            # ── Pass 2: depth estimation + back-projection in mini-batches ─────
            for chunk_start in range(0, n_total, MINI_BATCH):
                chunk = frames_rgb[chunk_start:chunk_start + MINI_BATCH]
                chunk_poses = (
                    seg_imu_poses[chunk_start:chunk_start + len(chunk)]
                    if seg_imu_poses is not None else None
                )
                f_lo = chunk_start + 1
                f_hi = chunk_start + len(chunk)

                yield {
                    log_output: (
                        f"Seg {seg_i}/{len(segments)} {label} — "
                        f"frames {f_lo}–{f_hi} / {n_total}…"
                    )
                }

                point_count, cam_pos, batch_infer_ms = session.process_frames_batch(
                    chunk, imu_poses=chunk_poses, use_da3_pose=_use_da3
                )
                seg_infer_ms += batch_infer_ms
                total_infer_ms += batch_infer_ms
                total_frames_processed += len(chunk)

                if session.last_frames_rgb and session.last_depth_frames:
                    mid = len(session.last_frames_rgb) // 2
                    depth_accum.append((
                        session.last_frames_rgb[mid],
                        session.last_depth_frames[mid],
                        session.last_frame_poses[mid] if session.last_frame_poses else None,
                    ))
                    # Accumulate every frame for the frame explorer (skip if no pose)
                    poses_for_accum = session.last_frame_poses or []
                    for rgb_f, df_f, pose_f in zip(
                        session.last_frames_rgb,
                        session.last_depth_frames,
                        poses_for_accum,
                    ):
                        if pose_f is not None:
                            all_frames_accum.append((rgb_f, df_f, pose_f))

                if len(session.last_trajectory) > 0:
                    segment_positions.extend(session.last_trajectory.tolist())

                pos_str = f"x={cam_pos[0]:.2f}  y={cam_pos[1]:.2f}  z={cam_pos[2]:.2f}"

                _cur_zones = session.zones  # zones completed so far
                _cur_gy = session.occupancy_map._ground_y
                yield {
                    live_cloud_plot: _cloud_to_glb(session._cloud, zones=_cur_zones, ground_y=_cur_gy),
                    occupancy_plot:  session.occupancy_map.render_plotly(zones=_cur_zones),
                    scan_status:     (
                        f"Seg {seg_i}/{len(segments)} "
                        f"[{f_lo}–{f_hi}/{n_total}] | {point_count:,} pts | "
                        f"batch {batch_infer_ms:.0f} ms "
                        f"({batch_infer_ms/len(chunk):.0f} ms/f)"
                    ),
                    scan_position:   pos_str,
                    log_output:      (
                        f"Seg {seg_i}/{len(segments)} {label} — "
                        f"frames {f_hi}/{n_total} | {point_count:,} pts | "
                        f"batch {batch_infer_ms:.0f} ms "
                        f"({batch_infer_ms/len(chunk):.0f} ms/f) | "
                        f"total {total_infer_ms/1000:.1f} s"
                    ),
                }

            if zone_name and segment_positions:
                session.set_label_from_positions(zone_name, segment_positions, margin=0.3)
                # Refresh displays immediately so the new area label/landmarks appear
                _seg_zones = session.zones
                _seg_gy = session.occupancy_map._ground_y
                yield {
                    live_cloud_plot: _cloud_to_glb(session._cloud, zones=_seg_zones, ground_y=_seg_gy),
                    occupancy_plot:  session.occupancy_map.render_plotly(zones=_seg_zones),
                }

        accum_rgb   = [x[0] for x in depth_accum]
        accum_df    = [x[1] for x in depth_accum]
        accum_poses = [x[2] for x in depth_accum]
        depth_data = _build_depth_data(accum_rgb, accum_df, frame_poses=accum_poses)
        n_views = len(depth_data) if depth_data else 0
        depth_choices = [f"View {i+1}" for i in range(n_views)] if n_views else ["View 1"]
        first_rgb, first_dvis = _get_depth_view(depth_data, 0) if depth_data else (None, None)
        zone_names = ", ".join(z.label for z in session.zones) if session.zones else "none"

        n_stored = len(all_frames_accum)
        slider_max = max(0, n_stored - 1)
        yield {
            scan_status:          (
                f"Done | {len(session._cloud.points):,} pts | Zones: {zone_names} | "
                f"total infer {total_infer_ms/1000:.1f} s"
            ),
            scan_position:        f"x={cam_pos[0]:.2f}  y={cam_pos[1]:.2f}  z={cam_pos[2]:.2f}",
            log_output:           (
                f"All {len(segments)} segment(s) complete. "
                f"{n_stored} frames stored for Frame Explorer. "
                f"Total inference: {total_infer_ms/1000:.1f} s "
                f"({total_infer_ms / max(total_frames_processed, 1):.0f} ms/f avg)"
            ),
            live_cloud_plot:      _cloud_to_glb(session._cloud, zones=session.zones, ground_y=session.occupancy_map._ground_y),
            occupancy_plot:       session.occupancy_map.render_plotly(zones=session.zones),
            depth_data_state:     depth_data,
            depth_view_selector:  gr.Dropdown(choices=depth_choices, value=depth_choices[0]),
            depth_rgb_image:      first_rgb,
            depth_vis_image:      first_dvis,
            measure_points_state: [],
            depth_measure_text:   "",
            all_frames_state:     all_frames_accum,
            frame_start_slider:   gr.Slider(minimum=0, maximum=slider_max, value=0, step=1),
            frame_end_slider:     gr.Slider(minimum=0, maximum=slider_max, value=slider_max, step=1),
            frame_pose_text:      _format_poses(all_frames_accum, 0, slider_max),
        }

    def _export_map(location_id: str):
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is None:
            return "No active session. Run Scan first."
        out_dir = session.export()
        n_pts = len(session._cloud.points)
        n_zones = len(session.zones)
        return f"Exported → `{out_dir}`  ({n_pts:,} pts, {n_zones} zones)"

    def _clear_cloud(location_id: str):
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is not None:
            session.reset_cloud()
        return None, go.Figure()

    # ── layout ─────────────────────────────────────────────────────────────────

    with gr.Blocks(theme=get_scan_theme(), css=SCAN_CSS, title="Scan Server") as app:

        gr.HTML(SCAN_HEADER_HTML)
        gr.HTML(SCAN_DESCRIPTION_HTML)

        depth_data_state    = gr.State(value=None)
        measure_points_state = gr.State(value=[])
        all_frames_state    = gr.State(value=[])   # list of (rgb, DepthFrame, pose 4×4)
        video_rotation_state = gr.State(value=0)   # extra rotation in degrees (0/90/180/270)

        with gr.Row():

            # ── Left: inputs ─────────────────────────────────────────────────
            with gr.Column(scale=2):
                input_video = gr.Video(label="Upload Video", interactive=True)
                with gr.Row():
                    rotate_video_btn = gr.Button("Rotate Video 90°", size="sm", scale=1)
                    video_rotation_display = gr.Textbox(
                        value="Rotation: 0°", label="", interactive=False, scale=2,
                        container=False,
                    )
                imu_file_input = gr.File(
                    label="IMU data (imu_data.csv — optional)",
                    file_types=[".csv"],
                    type="filepath",
                )
                pose_src_radio = gr.Radio(
                    choices=["Auto", "IMU + VO", "VO only", "DA3 poses"],
                    value="Auto",
                    label="Pose source",
                    info=(
                        "Auto = IMU+VO if file loaded, else VO.  "
                        "DA3 poses requires the PyTorch DA3Estimator (not ONNX)."
                    ),
                    interactive=True,
                )
                with gr.Row():
                    s_fps = gr.Slider(
                        minimum=0.1, maximum=10, value=5, step=0.1,
                        label="Sampling FPS", interactive=True,
                    )
                    batch_size_input = gr.Slider(
                        minimum=1, maximum=32, value=4, step=1,
                        label="Batch Size (frames)", interactive=True,
                    )
                with gr.Row():
                    location_id_input = gr.Textbox(
                        label="Location ID", placeholder="e.g. home-floor-1",
                        value="default",
                    )
                with gr.Row():
                    resolution_input = gr.Dropdown(
                        choices=["320×240 (~2 GB)", "480×360 (~3 GB)",
                                 "640×480 (~4 GB)", "Original"],
                        value="480×360 (~3 GB)",
                        label="DA3 Input Resolution (VRAM estimate)",
                        interactive=True,
                    )

                with gr.Accordion("Load from Android Upload", open=True):
                    with gr.Row():
                        upload_dropdown = gr.Dropdown(
                            choices=_list_uploads(),
                            label="Past Uploads",
                            interactive=True,
                            scale=4,
                        )
                        refresh_uploads_btn = gr.Button("Refresh", size="sm", scale=1)
                    load_upload_btn = gr.Button("Load Selected", variant="secondary", size="sm")

                zone_type_input = gr.Textbox(
                    label="Venue Type (optional)",
                    placeholder='e.g. "hospital", "supermarket", "home" — leave blank if unknown',
                    value="",
                )
                gr.Markdown(
                    "**Segment Table** — one row per area. "
                    "`start_s` / `end_s` in seconds. "
                    "Leave `area_name` blank for unlabelled sections."
                )
                segment_table = gr.Dataframe(
                    value=_DEFAULT_SEGMENTS,
                    headers=["start_s", "end_s", "area_name"],
                    datatype=["number", "number", "str"],
                    row_count=(1, "dynamic"),
                    col_count=(3, "fixed"),
                    interactive=True,
                    label="Area Segments",
                )

                frame_gallery = gr.Gallery(
                    label="Frame Preview", columns=4, height="200px",
                    object_fit="contain", interactive=False,
                )

            # ── Right: viewer ────────────────────────────────────────────────
            with gr.Column(scale=4):
                log_output = gr.Markdown(
                    "Upload a video, fill the segment table, then click **Scan**."
                )

                with gr.Tabs():

                    with gr.Tab("Live Points"):
                        with gr.Row():
                            clear_cloud_btn = gr.Button(
                                "Clear Cloud", variant="stop", size="sm", scale=0,
                            )
                        live_cloud_plot = gr.Model3D(
                            height=480,
                            zoom_speed=0.5,
                            pan_speed=0.5,
                            clear_color=[0.05, 0.05, 0.05, 1.0],
                            label="3D Point Cloud — updates each batch; or render a frame selection below",
                        )

                        with gr.Accordion("Frame Explorer", open=False):
                            gr.Markdown(
                                "Select a frame range and re-render the 3D cloud from "
                                "only those frames — **no reprocessing needed**."
                            )
                            with gr.Row():
                                frame_start_slider = gr.Slider(
                                    minimum=0, maximum=1, step=1, value=0,
                                    label="Start Frame", interactive=True, scale=3,
                                )
                                frame_end_slider = gr.Slider(
                                    minimum=0, maximum=1, step=1, value=0,
                                    label="End Frame", interactive=True, scale=3,
                                )
                                render_frames_btn = gr.Button(
                                    "Render", variant="secondary", size="sm", scale=1,
                                )
                            with gr.Row():
                                gr.Markdown(
                                    "**Axis remap** — if horizontal pan shows as wrong angle, "
                                    "swap axes here. Default: Roll←+X, Pitch←+Y, Yaw←+Z.",
                                    scale=3,
                                )
                            _AXIS_CHOICES = ["+X", "+Y", "+Z", "-X", "-Y", "-Z"]
                            with gr.Row():
                                roll_src_dd = gr.Dropdown(
                                    choices=_AXIS_CHOICES, value="+X",
                                    label="Roll ← axis", scale=1, interactive=True,
                                )
                                pitch_src_dd = gr.Dropdown(
                                    choices=_AXIS_CHOICES, value="+Y",
                                    label="Pitch ← axis", scale=1, interactive=True,
                                )
                                yaw_src_dd = gr.Dropdown(
                                    choices=_AXIS_CHOICES, value="+Z",
                                    label="Yaw ← axis", scale=1, interactive=True,
                                )
                            frame_pose_text = gr.Textbox(
                                label="Camera Pose(s)",
                                lines=6,
                                interactive=False,
                                placeholder="Run Scan, then select frames here.",
                            )

                    with gr.Tab("Depth Metric"):
                        with gr.Row():
                            prev_depth_btn = gr.Button("◀ Prev", size="sm", scale=1)
                            depth_view_selector = gr.Dropdown(
                                choices=["View 1"], value="View 1",
                                label="Select View", scale=2, interactive=True,
                            )
                            next_depth_btn = gr.Button("Next ▶", size="sm", scale=1)
                        with gr.Row():
                            depth_rgb_image = gr.Image(
                                type="numpy",
                                label="RGB — click two points to measure",
                                format="png", interactive=False, sources=[],
                                scale=1, height=350,
                            )
                            depth_vis_image = gr.Image(
                                type="numpy",
                                label="Metric Depth (near=bright, far=dark)",
                                format="png", interactive=False, sources=[],
                                scale=1, height=350,
                            )
                        gr.Markdown(
                            "Click **two points** on the RGB image to measure 3D distance."
                        )
                        depth_measure_text = gr.Markdown("")

                    with gr.Tab("Occupancy Map"):
                        occupancy_plot = gr.Plot(
                            label="Occupancy Map (top-down X-Z, updates each segment)"
                        )

                with gr.Row():
                    scan_btn = gr.Button("Scan", variant="primary", scale=3)
                    export_btn = gr.Button("Export Map", variant="secondary", scale=1)

        with gr.Row():
            scan_status = gr.Textbox(
                label="Status", value="Waiting for video upload…", interactive=False,
            )
            scan_position = gr.Textbox(
                label="Last Camera Position (m)", value="x=0.00  y=0.00  z=0.00",
                interactive=False,
            )

        export_log = gr.Markdown("")

        # ── event wiring ───────────────────────────────────────────────────────

        def _rotate_video(rotation, video_path, fps_val):
            new_rot = (rotation + 90) % 360
            gallery, msg, segs = _handle_video_upload(video_path, fps_val, new_rot)
            return new_rot, f"Rotation: {new_rot}°", gallery, msg, segs

        rotate_video_btn.click(
            fn=_rotate_video,
            inputs=[video_rotation_state, input_video, s_fps],
            outputs=[video_rotation_state, video_rotation_display, frame_gallery, log_output, segment_table],
        )

        input_video.change(
            fn=_handle_video_upload,
            inputs=[input_video, s_fps, video_rotation_state],
            outputs=[frame_gallery, log_output, segment_table],
        )

        scan_btn.click(
            fn=_run_local_scan,
            inputs=[input_video, imu_file_input, s_fps, batch_size_input,
                    location_id_input, segment_table, resolution_input,
                    pose_src_radio, video_rotation_state, zone_type_input],
            outputs=[
                live_cloud_plot, occupancy_plot,
                scan_status, scan_position, log_output,
                depth_data_state, depth_view_selector,
                depth_rgb_image, depth_vis_image,
                measure_points_state, depth_measure_text,
                all_frames_state, frame_start_slider, frame_end_slider,
                frame_pose_text,
            ],
        )

        export_btn.click(
            fn=_export_map,
            inputs=[location_id_input],
            outputs=[export_log],
        )

        prev_depth_btn.click(
            fn=lambda data, sel: _navigate_depth(data, sel, -1),
            inputs=[depth_data_state, depth_view_selector],
            outputs=[depth_view_selector, depth_rgb_image, depth_vis_image,
                     measure_points_state],
        )
        next_depth_btn.click(
            fn=lambda data, sel: _navigate_depth(data, sel, 1),
            inputs=[depth_data_state, depth_view_selector],
            outputs=[depth_view_selector, depth_rgb_image, depth_vis_image,
                     measure_points_state],
        )
        depth_view_selector.change(
            fn=_update_depth_selector,
            inputs=[depth_data_state, depth_view_selector],
            outputs=[depth_rgb_image, depth_vis_image, measure_points_state],
        )
        depth_rgb_image.select(
            fn=_do_measure,
            inputs=[depth_data_state, measure_points_state, depth_view_selector],
            outputs=[depth_rgb_image, measure_points_state, depth_measure_text],
        )

        clear_cloud_btn.click(
            fn=_clear_cloud,
            inputs=[location_id_input],
            outputs=[live_cloud_plot, occupancy_plot],
        )

        # ── Frame Explorer ─────────────────────────────────────────────────────

        def _render_selected(all_frames, start, end, roll_s, pitch_s, yaw_s):
            if not all_frames:
                return None, "No frames stored yet — run Scan first."
            s = max(0, int(min(start, end)))
            e = min(len(all_frames) - 1, int(max(start, end)))
            P = _perm_matrix(roll_s, pitch_s, yaw_s)
            result = _back_project_frames(all_frames, s, e, perm=P)
            if result is None:
                return None, f"No valid depth pixels in frames {s+1}–{e+1}."
            pts, cols = result
            glb = _cloud_to_glb((pts, cols))
            return glb, f"Rendered frames {s+1}–{e+1} | {len(pts):,} pts"

        def _update_pose_display(all_frames, start, end, roll_s, pitch_s, yaw_s):
            if not all_frames:
                return "No frames stored yet."
            s = max(0, int(min(start, end)))
            e = min(len(all_frames) - 1, int(max(start, end)))
            P = _perm_matrix(roll_s, pitch_s, yaw_s)
            return _format_poses(all_frames, s, e, perm=P)

        _axis_dd_inputs = [roll_src_dd, pitch_src_dd, yaw_src_dd]

        render_frames_btn.click(
            fn=_render_selected,
            inputs=[all_frames_state, frame_start_slider, frame_end_slider] + _axis_dd_inputs,
            outputs=[live_cloud_plot, scan_status],
        )

        for _slider in [frame_start_slider, frame_end_slider]:
            _slider.change(
                fn=_update_pose_display,
                inputs=[all_frames_state, frame_start_slider, frame_end_slider] + _axis_dd_inputs,
                outputs=[frame_pose_text],
            )

        for _dd in _axis_dd_inputs:
            _dd.change(
                fn=_update_pose_display,
                inputs=[all_frames_state, frame_start_slider, frame_end_slider] + _axis_dd_inputs,
                outputs=[frame_pose_text],
            )

        refresh_uploads_btn.click(
            fn=lambda: gr.Dropdown(choices=_list_uploads()),
            inputs=[],
            outputs=[upload_dropdown],
        )

        def _do_load_upload(scan_id):
            video, imu = _load_upload(scan_id)
            return video, imu  # imu may be None — gr.File accepts None to clear

        load_upload_btn.click(
            fn=_do_load_upload,
            inputs=[upload_dropdown],
            outputs=[input_video, imu_file_input],
        )

    return app
