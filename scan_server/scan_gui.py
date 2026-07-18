"""
Scan Server GUI — tabs:
  1. Live Reconstruction — Live Points + Voxelization (side by side) and
                      Occupancy Map (below), all three rebuilding
                      progressively, once per processed chunk, as a scan
                      runs (see ScanSession.process_frames_batch /
                      OccupancyMap.update()) — not deferred to the end.
                      Each view toggleable via its own checkbox.
  2. Depth Metric  — per-frame depth map + click-to-measure
  3. Detections    — GroundingDINO boxes + VLM response debug view

Workflow (there is no batch "Scan" button — every run replays the dataset
frame-by-frame through StreamingScanSession, see stream_session.py /
stream_simulator.py):
  1. Point at a dataset folder (images/ + imu.csv + camera.csv) + set Location ID
  2. Fill Segment Table: start_s | end_s | zone_name
  3. Click "Simulated Live Stream" (auto, whole dataset) OR click "Start /
     Reset Manual Stream" then "Feed Next Frame" repeatedly (one frame per
     click, with a preview of the frame about to be fed)
  4. Click Export Map (or let a stream finish, which auto-exports) — saves
     PLY + JSON + ORB keyframes

Dataset folder layout (see camera.csv/imu.csv headers for exact columns):
    dataset/
        images/000000000.jpg, 000000001.jpg, ...
        imu.csv       timestamp_ns,ax,ay,az,gx,gy,gz
        camera.csv    timestamp_ns,filename
"""

import math
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
from scan_session import voxelize_cloud, DEFAULT_VOXEL_SIZE, MAX_VOXELS
from timing_utils import timed
from stream_session import StreamingScanSession
from stream_simulator import replay_dataset, ManualDatasetReplayer
from live_path_planner import LiveGridPathPlanner

# Dropdown label -> ImuIntegrator's `orientation` key (see scan_session.IMU_ORIENTATIONS).
# Only the raw imu.csv values are rotated; images are handled separately by the
# existing "Rotate Images 90°" button/video_rotation_state.
_IMU_ORIENTATION_LABELS = {
    "Portrait (native)": "portrait",
    "Landscape — left (top of phone left)": "landscape-left",
    "Landscape — right (top of phone right)": "landscape-right",
}

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


# ── Detection debug view (GroundingDINO boxes + Qwen VL response) ─────────────


def _draw_detections(frame_bgr: np.ndarray, detections: list) -> np.ndarray:
    """Draw GroundingDINO boxes + label/score on a BGR frame; return RGB uint8."""
    img = frame_bgr.copy()
    for det in detections:
        x0, y0, x1, y1 = [int(round(v)) for v in det.box_xyxy]
        cv2.rectangle(img, (x0, y0), (x1, y1), (50, 220, 50), 2)
        label = f"{det.label} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty0 = max(0, y0 - th - 6)
        cv2.rectangle(img, (x0, ty0), (x0 + tw + 4, y0), (50, 220, 50), -1)
        cv2.putText(img, label, (x0 + 2, max(th + 2, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _build_detection_view(session):
    """Return (boxed_rgb_image_or_None, status_markdown) for the Detections tab."""
    if session is None or session.semantic_mapper is None:
        return None, "Semantic mapping is disabled on this server."
    sm = session.semantic_mapper
    frame_bgr = sm.last_frame_bgr
    if frame_bgr is None:
        return None, "No frame processed yet — run Scan with a labelled area."
    image = _draw_detections(frame_bgr, sm.last_detections)
    lines = [f"**{len(sm.last_detections)} detection(s)**"]
    if sm.last_error:
        lines.append(f"⚠️ {sm.last_error}")
    lines.append("\n**Qwen VL response:**\n")
    lines.append(f"```\n{sm.last_vlm_response or '(empty)'}\n```")
    return image, "\n".join(lines)


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
    """Times the render (GLB export via trimesh — CPU-only, no GPU path in
    this library) and delegates to _cloud_to_glb_impl."""
    try:
        n_pts = len(cloud_or_pts.points) if hasattr(cloud_or_pts, "points") else len(cloud_or_pts[0])
    except Exception:
        n_pts = "?"
    with timed(f"_cloud_to_glb render ({n_pts} pts)"):
        return _cloud_to_glb_impl(cloud_or_pts, zones=zones, ground_y=ground_y)


def _cloud_to_glb_impl(cloud_or_pts, zones=None, ground_y=None) -> Optional[str]:
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

        _add_zone_overlays(scene, zones, pts, ground_y)
        _set_top_down_camera(scene, pts)

        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        tmp.close()
        scene.export(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[cloud_to_glb] EXCEPTION: {e}")
        return None


def _add_zone_overlays(scene, zones, pts: np.ndarray, ground_y: Optional[float]) -> None:
    """Draw zone AABB floor outlines + landmark spheres into a trimesh Scene.
    Shared by _cloud_to_glb (raw point cloud) and _voxels_to_glb (voxel blocks)."""
    if not zones:
        return
    import trimesh
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


def _set_top_down_camera(scene, pts: np.ndarray) -> None:
    """Top-down initial camera: camera placed above centroid looking down (-Y).
    In trimesh camera space: +X=right, +Y=up, -Z=forward (toward scene).
    For top-down: camera -Z (forward) aligns with world -Y (down),
      so camera +Z column = world +Y = [0,1,0]
          camera +X column = world +X = [1,0,0]
          camera +Y column = world -Z = [0,0,-1]  (right-hand cross product)"""
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


# ── Point cloud → voxel blocks → Model3D ───────────────────────────────────────


def _voxel_centers_to_glb(
    centers: np.ndarray, colors: Optional[np.ndarray], voxel_size: float,
    zones=None, ground_y=None,
) -> Optional[str]:
    """Times the render (per-voxel trimesh box construction — CPU-only) and
    delegates to _voxel_centers_to_glb_impl."""
    with timed(f"_voxel_centers_to_glb render ({len(centers)} voxels)"):
        return _voxel_centers_to_glb_impl(centers, colors, voxel_size, zones=zones, ground_y=ground_y)


def _voxel_centers_to_glb_impl(
    centers: np.ndarray, colors: Optional[np.ndarray], voxel_size: float,
    zones=None, ground_y=None,
) -> Optional[str]:
    """
    Render already-voxelized centers+colors (e.g. session.last_voxel_centers/
    colors, cached once per batch by ScanSession.process_frames_batch) as
    solid cube blocks — the box-building half of _voxels_to_glb, factored out
    so the Scan loop's per-batch auto-refresh can reuse the exact same voxels
    that just fed the Occupancy Map, instead of re-running voxelize_cloud.

    Builds ONE combined mesh directly via vectorized numpy broadcasting of a
    single reference box's vertex/face template, instead of calling
    trimesh.creation.box() once per voxel and concatenating — profiled at
    seconds (scales linearly with voxel count, each call constructing a full
    separate trimesh object) for tens of thousands of voxels; this is the
    same final geometry (the template IS a real trimesh.creation.box() call,
    just reused instead of rebuilt), just constructed in one shot.
    """
    if len(centers) == 0:
        return None
    try:
        import trimesh

        template = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size])
        template_verts = np.asarray(template.vertices, dtype=np.float64)  # (V, 3)
        template_faces = np.asarray(template.faces, dtype=np.int64)      # (F, 3)
        n_verts_per_box = len(template_verts)
        n_faces_per_box = len(template_faces)

        n = len(centers)
        centers64 = np.asarray(centers, dtype=np.float64)
        # (n, V, 3) = (n, 1, 3) centers broadcast against (1, V, 3) template
        all_verts = (centers64[:, None, :] + template_verts[None, :, :]).reshape(-1, 3)
        # Each box's faces reference its OWN 8 vertices — offset the shared
        # template's indices by i*n_verts_per_box per box.
        face_offsets = (np.arange(n, dtype=np.int64) * n_verts_per_box)[:, None, None]
        all_faces = (template_faces[None, :, :] + face_offsets).reshape(-1, 3)

        # process=False: skip trimesh's default post-load processing (merge
        # duplicate vertices etc.) — unnecessary here (every vertex is
        # already correctly placed) and itself scales with mesh size.
        mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)

        if colors is not None:
            rgb8 = np.clip(np.asarray(colors) * 255, 0, 255).astype(np.uint8)
        else:
            rgb8 = np.full((n, 3), [180, 180, 180], dtype=np.uint8)
        alpha = np.full((n, 1), 255, dtype=np.uint8)
        rgba = np.hstack([rgb8, alpha])                       # (n, 4)
        mesh.visual.face_colors = np.repeat(rgba, n_faces_per_box, axis=0)  # (n*F, 4)

        scene = trimesh.Scene()
        scene.add_geometry(mesh)

        _add_zone_overlays(scene, zones, centers, ground_y)
        _set_top_down_camera(scene, centers)

        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        tmp.close()
        scene.export(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[voxel_centers_to_glb] EXCEPTION: {e}")
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


_AXIS_CHOICES = ["+X", "+Y", "+Z", "-X", "-Y", "-Z"]

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
            return None
        base = _upload_dir / scan_id / "dataset"
        return str(base) if base.exists() else None

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

    def _read_camera_index(dataset_path: str) -> List[tuple]:
        """Return [(timestamp_ns, abs_image_path), ...] sorted by timestamp,
        read from dataset_path/camera.csv (header: timestamp_ns,filename)."""
        base = Path(dataset_path)
        csv_path = base / "camera.csv"
        if not csv_path.exists():
            return []
        df = pd.read_csv(csv_path)
        rows = sorted(zip(df["timestamp_ns"].tolist(), df["filename"].tolist()), key=lambda r: r[0])
        return [(int(ts), str(base / "images" / fname)) for ts, fname in rows]

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

    def _handle_dataset_change(dataset_path: Optional[str], fps_val: float, extra_rotation: int = 0):
        if not dataset_path or not Path(dataset_path).exists():
            return None, "Enter/select a dataset folder (images/ + camera.csv) to preview.", _DEFAULT_SEGMENTS
        index = _read_camera_index(dataset_path)
        if not index:
            return None, f"No camera.csv found under {dataset_path}.", _DEFAULT_SEGMENTS
        t0, t1 = index[0][0], index[-1][0]
        duration = (t1 - t0) / 1e9
        deltas = np.diff([ts for ts, _ in index])
        dataset_fps = float(1e9 / np.median(deltas)) if len(deltas) else fps_val
        interval = max(1, round(dataset_fps / max(fps_val, 0.1)))

        preview = []
        for i, (_, path) in enumerate(index):
            if i % interval != 0:
                continue
            frame = cv2.imread(path)
            if frame is None:
                continue
            frame = _correct_rotation(frame, extra_rotation)
            preview.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(preview) >= 16:
                break

        approx = max(1, len(index) // max(interval, 1))
        msg = f"Dataset: {duration:.1f}s — {len(index)} frames, ~{approx} sampled at {fps_val} FPS"
        default_df = pd.DataFrame(
            {"start_s": [0.0], "end_s": [round(duration, 1)], "area_name": [""]}
        )
        return preview, msg, default_df

    def _run_simulated_stream(
        dataset_path: Optional[str],
        fps_val: float,
        batch_size: int,
        location_id: str,
        segments_df,
        resolution: str,
        pose_src: str,
        extra_rotation: int,
        imu_orientation: str,
        zone_type: str,
        axis_roll: str,
        axis_pitch: str,
        axis_yaw: str,
        sor_neighbors: int,
        sor_std_ratio: float,
        voxel_size: float,
        realtime: bool,
        occ_obstacle_min_h: float = 0.10,
        occ_step_over_max_h: float = 0.40,
        occ_obstacle_max_h: float = 2.20,
        occ_logodds_hit: float = 0.85,
        occ_logodds_miss: float = 0.40,
        occ_logodds_occ_thresh: float = 1.0,
        occ_logodds_free_thresh: float = -1.0,
        occ_height_ewma_alpha: float = 0.30,
        occ_enable_ray_casting: bool = True,
        occ_enable_bayesian: bool = True,
        show_live_points: bool = True,
        show_voxelization: bool = True,
        show_occupancy: bool = True,
        nav_state_value: Optional[dict] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Simulated-live counterpart to _run_local_scan: instead of reading a
        finished camera.csv/imu.csv all at once and slicing it into
        pre-declared segments, this replays the same dataset folder
        frame-by-frame / IMU-sample-by-sample through StreamingScanSession's
        push API (stream_session.py) via stream_simulator.replay_dataset —
        proving the streaming interface a real Android live source will
        drive later, without needing that source to exist yet.

        Live Points -> Voxelization -> Occupancy Map all rebuild after every
        processed chunk now, each gated by its own checkbox (see the Live
        Reconstruction tab) — the Occupancy Map's own update() already ran
        inside StreamingScanSession._process_chunk, fed from this chunk's
        points additionally voxel-downsampled at `voxel_size` (see
        ScanSession.process_frames_batch's docstring); Live Points/
        Voxelization are rebuilt here for display only when checked, since
        both cost grows with the total scan size (no incremental voxel-merge
        primitive in Open3D) — unlike the Occupancy Map's update.
        """
        if not dataset_path or not Path(dataset_path).exists():
            yield {log_output: "Select a dataset folder first."}
            return

        nav_state_value = nav_state_value or dict(target=None, route=None, confirmed=None, computed_at=-1)

        axis_perm = _perm_matrix(axis_roll, axis_pitch, axis_yaw)
        max_dim = int(resolution.split("×")[0]) if resolution != "Original" else 0
        location_id = (location_id or "").strip() or "default"
        zone_type = (zone_type or "").strip()
        segments = _parse_segments(segments_df)
        imu_orient_key = _IMU_ORIENTATION_LABELS.get(imu_orientation, "portrait")

        stream = StreamingScanSession(
            scan_manager, location_id, pose_src=pose_src,
            imu_orientation=imu_orient_key, zone_type=zone_type,
            axis_perm=axis_perm, mini_batch=int(batch_size),
            sor_nb_neighbors=int(sor_neighbors), sor_std_ratio=float(sor_std_ratio),
            occupancy_voxel_size=float(voxel_size),
        )
        session = stream.session
        # See _run_local_scan's identical call — applied fresh at the start
        # of this run, only affects FUTURE update() calls.
        session.configure_occupancy_map(
            obstacle_min_h=occ_obstacle_min_h,
            step_over_max_h=occ_step_over_max_h,
            obstacle_max_h=occ_obstacle_max_h,
            logodds_hit=occ_logodds_hit,
            logodds_miss=occ_logodds_miss,
            logodds_occupied_thresh=occ_logodds_occ_thresh,
            logodds_free_thresh=occ_logodds_free_thresh,
            height_ewma_alpha=occ_height_ewma_alpha,
            enable_ray_casting=occ_enable_ray_casting,
            enable_bayesian=occ_enable_bayesian,
        )

        yield {log_output: f"[Simulated Live Stream] Replaying `{dataset_path}` "
                           f"as a live frame+IMU source…"}

        n_chunks = 0
        # See _run_local_scan's identical variable — skip rebuilding Live
        # Points/Voxelization on a chunk that added zero new points.
        _last_render_point_count = -1
        for event in replay_dataset(
            stream, dataset_path, segments, fps_val=fps_val,
            max_dim=max_dim, extra_rotation=extra_rotation, realtime=bool(realtime),
        ):
            if event["kind"] == "error":
                yield {log_output: event["message"]}
                return
            if event["kind"] in ("zone_start", "zone_end"):
                yield {log_output: f"[{event['kind']}] '{event['zone']}' "
                                   f"@ {event['progress']*100:.0f}% replayed"}
                continue

            result = event.get("result")
            if result is None:
                continue
            n_chunks += 1
            cam_pos = result["cam_pos"]
            pos_str = f"x={cam_pos[0]:.2f}  y={cam_pos[1]:.2f}  z={cam_pos[2]:.2f}"
            src_tag = f"[{result['pose_source']}]"
            session.preview_landmarks()
            det_image, det_text = _build_detection_view(session)
            _yield: Dict[Any, Any] = {
                detection_image: det_image,
                detection_text: det_text,
                scan_status: (
                    f"[Live] chunk {n_chunks} | {result['point_count']:,} pts | "
                    f"{event['progress']*100:.0f}% replayed | "
                    f"batch {result['infer_ms']:.0f} ms "
                    f"({result['infer_ms']/result['n_frames']:.0f} ms/f)"
                ),
                scan_position: f"{src_tag}  {pos_str}",
                log_output: (
                    f"{src_tag} {pos_str} | {result['point_count']:,} pts | "
                    f"{event['progress']*100:.0f}% replayed"
                ),
            }
            # Live navigation preview — "constantly extend the navigation...
            # as new data comes in": if a destination is set, recompute the
            # route whenever the occupancy grid actually changed since the
            # last computation (_update_count, not every single chunk —
            # avoids redundant recomputation on a chunk that added nothing).
            if (
                nav_state_value.get("target") is not None
                and session.occupancy_map._update_count != nav_state_value.get("computed_at")
            ):
                route, confirmed, reached_exactly, nav_status_text = _nav_route_full_and_status(
                    session, nav_state_value["target"], nav_state_value.get("min_clearance", 0.0)
                )
                nav_state_value = dict(
                    nav_state_value, route=route, confirmed=confirmed,
                    reached_exactly=reached_exactly,
                    computed_at=session.occupancy_map._update_count,
                )
                _yield[nav_status] = nav_status_text
                _yield[nav_state] = nav_state_value

            if show_occupancy:
                _yield[occupancy_plot] = session.occupancy_map.render_plotly(
                    zones=session.zones,
                    route=nav_state_value.get("route"), route_confirmed=nav_state_value.get("confirmed"),
                )
                _yield[confidence_plot] = session.occupancy_map.render_confidence_plotly(
                    zones=session.zones,
                    route=nav_state_value.get("route"), route_confirmed=nav_state_value.get("confirmed"),
                )
            # session.last_voxel_centers is already the single, incrementally-
            # accumulated voxelization the Occupancy Map feed itself computed
            # this chunk (see ScanSession._merge_voxels) — no separate
            # voxelize_cloud() call needed here, just render what's there.
            # Still skip the mesh rebuild when nothing changed, same as before.
            if (show_live_points or show_voxelization) and session._raw_point_count != _last_render_point_count:
                if show_live_points:
                    cloud = session.ensure_cloud_built()
                    _yield[live_cloud_plot] = _cloud_to_glb(
                        cloud, zones=session.zones, ground_y=session.occupancy_map._ground_y
                    )
                if show_voxelization:
                    _yield[voxel_plot] = _voxel_centers_to_glb(
                        session.last_voxel_centers, session.last_voxel_colors, session.last_voxel_size,
                        zones=session.zones, ground_y=session.occupancy_map._ground_y,
                    )
                _last_render_point_count = session._raw_point_count
            yield _yield

        yield {log_output: f"Replay complete ({session._raw_point_count:,} pts) — exporting…"}
        stream.finish()

        zones = session.zones
        zone_names = ", ".join(z.label for z in zones) if zones else "none"
        _final_yield: Dict[Any, Any] = {
            scan_status: f"Done | {len(session._cloud.points):,} pts | Zones: {zone_names}",
            log_output: f"Simulated live stream finished and exported map for '{location_id}'.",
        }
        if show_occupancy:
            _final_yield[occupancy_plot] = session.occupancy_map.render_plotly(
                zones=zones, route=nav_state_value.get("route"), route_confirmed=nav_state_value.get("confirmed"))
            _final_yield[confidence_plot] = session.occupancy_map.render_confidence_plotly(
                zones=zones, route=nav_state_value.get("route"), route_confirmed=nav_state_value.get("confirmed"))
        if show_live_points:
            _final_yield[live_cloud_plot] = _cloud_to_glb(
                session._cloud, zones=zones, ground_y=session.occupancy_map._ground_y
            )
        if show_voxelization:
            _final_yield[voxel_plot] = _voxel_centers_to_glb(
                session.last_voxel_centers, session.last_voxel_colors, session.last_voxel_size,
                zones=zones, ground_y=session.occupancy_map._ground_y,
            )
        yield _final_yield

    def _manual_stream_start(
        dataset_path: Optional[str],
        fps_val: float,
        batch_size: int,
        location_id: str,
        segments_df,
        resolution: str,
        pose_src: str,
        extra_rotation: int,
        imu_orientation: str,
        zone_type: str,
        axis_roll: str,
        axis_pitch: str,
        axis_yaw: str,
        sor_neighbors: int,
        sor_std_ratio: float,
        voxel_size: float,
        occ_obstacle_min_h: float,
        occ_step_over_max_h: float,
        occ_obstacle_max_h: float,
        occ_logodds_hit: float,
        occ_logodds_miss: float,
        occ_logodds_occ_thresh: float,
        occ_logodds_free_thresh: float,
        occ_height_ewma_alpha: float,
        occ_enable_ray_casting: bool,
        occ_enable_bayesian: bool,
    ):
        """
        (Re)initializes a manual, single-step replay of dataset_path — same
        setup as _run_simulated_stream (StreamingScanSession + Occupancy Map
        Settings applied fresh), but hands control back to the GUI after
        building the first frame preview instead of auto-looping through
        replay_dataset(). Each subsequent "Feed Next Frame" click drives one
        ManualDatasetReplayer.step() (see stream_simulator.py).
        """
        if not dataset_path or not Path(dataset_path).exists():
            return None, None, gr.update(interactive=False), "Select a dataset folder first."

        axis_perm = _perm_matrix(axis_roll, axis_pitch, axis_yaw)
        max_dim = int(resolution.split("×")[0]) if resolution != "Original" else 0
        location_id = (location_id or "").strip() or "default"
        zone_type = (zone_type or "").strip()
        segments = _parse_segments(segments_df)
        imu_orient_key = _IMU_ORIENTATION_LABELS.get(imu_orientation, "portrait")

        stream = StreamingScanSession(
            scan_manager, location_id, pose_src=pose_src,
            imu_orientation=imu_orient_key, zone_type=zone_type,
            axis_perm=axis_perm, mini_batch=int(batch_size),
            sor_nb_neighbors=int(sor_neighbors), sor_std_ratio=float(sor_std_ratio),
            occupancy_voxel_size=float(voxel_size),
        )
        # See _run_simulated_stream's identical call — applied fresh at the
        # start of this run, only affects FUTURE update() calls.
        stream.session.configure_occupancy_map(
            obstacle_min_h=occ_obstacle_min_h,
            step_over_max_h=occ_step_over_max_h,
            obstacle_max_h=occ_obstacle_max_h,
            logodds_hit=occ_logodds_hit,
            logodds_miss=occ_logodds_miss,
            logodds_occupied_thresh=occ_logodds_occ_thresh,
            logodds_free_thresh=occ_logodds_free_thresh,
            height_ewma_alpha=occ_height_ewma_alpha,
            enable_ray_casting=occ_enable_ray_casting,
            enable_bayesian=occ_enable_bayesian,
        )

        replayer = ManualDatasetReplayer(
            stream, dataset_path, segments, fps_val=fps_val,
            max_dim=max_dim, extra_rotation=extra_rotation,
        )
        if replayer.error:
            return None, None, gr.update(interactive=False), replayer.error
        if not replayer.has_more():
            return None, None, gr.update(interactive=False), "No frames found in this dataset."

        preview = replayer.peek_next_frame_preview()
        return (
            replayer, preview, gr.update(interactive=True),
            f"[Manual Stream] Ready — '{dataset_path}' loaded. "
            f"Preview shows the first frame; click 'Feed Next Frame' to begin.",
        )

    def _manual_stream_feed(
        replayer: Optional["ManualDatasetReplayer"],
        show_live_points: bool,
        show_voxelization: bool,
        show_occupancy: bool,
    ):
        """
        One "Feed Next Frame" click == one ManualDatasetReplayer.step() —
        applies any IMU samples/zone boundaries preceding the next frame,
        then pushes that single frame through the same
        StreamingScanSession/process_frames_batch pipeline
        _run_simulated_stream uses (just one frame per click instead of an
        auto loop). Auto-finalizes + exports once no frames remain, exactly
        like _run_simulated_stream's own end-of-replay step.
        """
        if replayer is None:
            return (
                replayer, None, gr.update(interactive=False),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                "Click 'Start / Reset Manual Stream' first.",
            )

        event = replayer.step()
        session = replayer.stream.session
        zone_log = " ".join(
            f"[{e['kind']}] '{e.get('zone', '')}'" for e in event.get("zone_events", [])
        )

        if event["kind"] == "done":
            replayer.stream.finish()
            zones = session.zones
            zone_names = ", ".join(z.label for z in zones) if zones else "none"
            gy = session.occupancy_map._ground_y
            occ = session.occupancy_map.render_plotly(zones=zones) if show_occupancy else gr.update()
            conf = session.occupancy_map.render_confidence_plotly(zones=zones) if show_occupancy else gr.update()
            lp = _cloud_to_glb(session._cloud, zones=zones, ground_y=gy) if show_live_points else gr.update()
            vp = _voxel_centers_to_glb(
                session.last_voxel_centers, session.last_voxel_colors, session.last_voxel_size,
                zones=zones, ground_y=gy,
            ) if show_voxelization else gr.update()
            log = f"Manual stream finished and exported map for '{session.location_id}'."
            return (
                replayer, None, gr.update(interactive=False),
                lp, vp, occ, conf,
                gr.update(), gr.update(),
                f"Done | {len(session._cloud.points):,} pts | Zones: {zone_names}",
                gr.update(),
                f"{zone_log} {log}" if zone_log else log,
            )

        preview = replayer.peek_next_frame_preview()
        has_more = replayer.has_more()
        result = event.get("result")

        if result is None:
            # Still buffering into a mini-batch/DA3 window — no new render yet.
            log = (zone_log or
                   f"Frame buffered ({event['progress']*100:.0f}% replayed) — "
                   f"waiting for a full chunk.")
            return (
                replayer, preview, gr.update(interactive=has_more),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                log,
            )

        cam_pos = result["cam_pos"]
        pos_str = f"x={cam_pos[0]:.2f}  y={cam_pos[1]:.2f}  z={cam_pos[2]:.2f}"
        src_tag = f"[{result['pose_source']}]"
        session.preview_landmarks()
        det_image, det_text = _build_detection_view(session)

        occ = session.occupancy_map.render_plotly(zones=session.zones) if show_occupancy else gr.update()
        conf = session.occupancy_map.render_confidence_plotly(zones=session.zones) if show_occupancy else gr.update()
        lp_update = gr.update()
        vp_update = gr.update()
        # See _run_simulated_stream's identical check — skip rebuilding
        # Live Points/Voxelization on a chunk that added zero new points
        # (tracked on the replayer itself since it must survive across
        # separate per-click callback invocations, not one generator's
        # closure).
        if (show_live_points or show_voxelization) and session._raw_point_count != replayer.last_render_point_count:
            if show_live_points:
                cloud = session.ensure_cloud_built()
                lp_update = _cloud_to_glb(cloud, zones=session.zones, ground_y=session.occupancy_map._ground_y)
            if show_voxelization:
                # session.last_voxel_centers is already the single,
                # incrementally-accumulated voxelization the Occupancy Map
                # feed itself computed this step (see
                # ScanSession._merge_voxels) — no separate voxelize_cloud()
                # call needed here, just render what's there.
                vp_update = _voxel_centers_to_glb(
                    session.last_voxel_centers, session.last_voxel_colors, session.last_voxel_size,
                    zones=session.zones, ground_y=session.occupancy_map._ground_y,
                )
            replayer.last_render_point_count = session._raw_point_count

        status = (
            f"[Manual] {result['point_count']:,} pts | {event['progress']*100:.0f}% replayed | "
            f"batch {result['infer_ms']:.0f} ms ({result['infer_ms']/result['n_frames']:.0f} ms/f)"
        )
        log = (f"{zone_log} " if zone_log else "") + (
            f"{src_tag} {pos_str} | {result['point_count']:,} pts | "
            f"{event['progress']*100:.0f}% replayed"
        )
        return (
            replayer, preview, gr.update(interactive=has_more),
            lp_update, vp_update, occ, conf,
            det_image, det_text, status, f"{src_tag}  {pos_str}", log,
        )

    def _export_map(location_id: str):
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is None:
            return "No active session. Run Scan first.", None, None, go.Figure(), go.Figure()
        out_dir = session.export()
        n_pts = len(session._cloud.points)
        # export() -> finalize_landmarks() only just populated zone.landmarks
        # (clustering + zone assignment happens once, at export time, not
        # live during scanning) — re-render all three views now so the
        # landmark markers actually show up instead of staying invisible
        # until a manual Reload.
        zones = session.zones
        n_zones = len(zones)
        gy = session.occupancy_map._ground_y
        return (
            f"Exported → `{out_dir}`  ({n_pts:,} pts, {n_zones} zones)",
            _cloud_to_glb(session._cloud, zones=zones, ground_y=gy),
            _voxel_centers_to_glb(
                session.last_voxel_centers, session.last_voxel_colors,
                session.last_voxel_size, zones=zones, ground_y=gy,
            ),
            session.occupancy_map.render_plotly(zones=zones),
            session.occupancy_map.render_confidence_plotly(zones=zones),
        )

    def _clear_cloud(location_id: str):
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is not None:
            session.reset_cloud()
        return None, None, go.Figure(), go.Figure()

    def _reload_occupancy(location_id: str):
        """Manually re-render the Occupancy Map + Confidence Map — a Scan run
        occasionally ends without the plots updating (Gradio drops a
        mid-generator yield), so this gives a reliable way to force a
        redraw from current session state. Returns (occupancy_fig,
        confidence_fig)."""
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is None:
            return go.Figure(), go.Figure()
        session.preview_landmarks()  # no-op post-export (raw_landmarks already cleared by finalize_landmarks)
        return (
            session.occupancy_map.render_plotly(zones=session.zones),
            session.occupancy_map.render_confidence_plotly(zones=session.zones),
        )

    def _reload_detections(location_id: str):
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        return _build_detection_view(session)

    def _voxelize(location_id: str, voxel_size: float):
        """Manual "Voxelize" button — the one place besides the live
        Occupancy Map feed allowed to (re)compute voxelize_cloud() directly,
        since this is an explicit, infrequent, user-picked-voxel-size
        recompute over the WHOLE cloud (e.g. after changing the slider), not
        part of the automatic per-chunk pipeline. Routes its result through
        session._merge_voxels() — same accumulator the live feed uses — so
        session.last_voxel_centers stays the one canonical answer afterward,
        rather than a second, disconnected computation."""
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is None:
            return None
        session.ensure_cloud_built()  # self._cloud is lazy — build it before reading
        if len(session._cloud.points) == 0:
            return None
        centers, colors, vsize = voxelize_cloud(session._cloud, voxel_size=voxel_size)
        session._merge_voxels(centers, colors, vsize)
        return _voxel_centers_to_glb(
            session.last_voxel_centers, session.last_voxel_colors, session.last_voxel_size,
            zones=session.zones, ground_y=session.occupancy_map._ground_y,
        )

    def _reload_live_points(location_id: str):
        """Manually (re)render the Live Points 3D view. Also called
        automatically once per chunk during Scan/Simulated Live Stream when
        the "Live Points" checkbox is on (see _run_local_scan/
        _run_simulated_stream) — this is the on-demand "show me current
        state right now" counterpart, e.g. after Export or when nothing is
        actively streaming."""
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is None:
            return None
        session.ensure_cloud_built()  # incremental — only merges batches added since the last call
        return _cloud_to_glb(session._cloud, zones=session.zones, ground_y=session.occupancy_map._ground_y)

    def _reload_all_views(location_id: str, voxel_size: float):
        """Single "Reload" button in the Live Reconstruction tab — refreshes
        all views at once (Live Points, Voxelization, Occupancy Map,
        Confidence Map), regardless of checkbox state (checkboxes only gate
        the automatic per-chunk recompute + visibility, not this manual
        action)."""
        occ, conf = _reload_occupancy(location_id)
        return (
            _reload_live_points(location_id),
            _voxelize(location_id, voxel_size),
            occ,
            conf,
        )

    def _toggle_live_points(show: bool, location_id: str):
        """Checkbox toggle — hides the view immediately when unchecked
        (independent of any running Scan/Stream generator); when re-checked,
        also refreshes it from current session state right away rather than
        waiting for the next chunk (which might be a while, or never, if
        nothing is currently running)."""
        if not show:
            return gr.update(visible=False)
        return gr.update(visible=True, value=_reload_live_points(location_id))

    def _toggle_voxelization(show: bool, location_id: str, voxel_size: float):
        if not show:
            return gr.update(visible=False)
        return gr.update(visible=True, value=_voxelize(location_id, voxel_size))

    def _toggle_occupancy(show: bool, location_id: str):
        if not show:
            return gr.update(visible=False), gr.update(visible=False)
        occ, conf = _reload_occupancy(location_id)
        return gr.update(visible=True, value=occ), gr.update(visible=True, value=conf)

    # ── Live navigation preview ──────────────────────────────────────────────
    # Destination is set via X/Z number inputs or a landmark dropdown (not by
    # clicking the map — gr.Plot/Plotly doesn't fire select events in this
    # Gradio version, confirmed by reading gradio/components/plot.py).
    # LiveGridPathPlanner runs against the IN-PROGRESS occupancy grid
    # (extract_full_grid(), rebuilt fresh each time — no exported map file
    # needed) — same cost model as the deployed GridPathPlanner
    # (server/tools/grid_path_planner.py), so a route through confirmed-free
    # cells is naturally preferred, but the search still flows through
    # unexplored (CLASS_UNKNOWN) territory when that's the only way to reach
    # the destination, flagged as "speculative" rather than failing outright.

    def _nav_route_full_and_status(session, target_xz, min_clearance: float = 0.0):
        """Returns (route_including_start_point_or_None, confirmed_or_None,
        reached_exactly_or_None, status_markdown). route includes the
        current camera position as its first element —
        LiveGridPathPlanner.find_path() itself omits the start point, so
        it's prepended here for the map overlay."""
        full_grid = session.occupancy_map.extract_full_grid()
        if full_grid is None:
            return None, None, None, "No occupancy data yet — scan a bit first."
        if not session.occupancy_map._trajectory:
            return None, None, None, "No camera position yet — scan a bit first."
        cam_x, _, cam_z = session.occupancy_map._trajectory[-1]
        planner = LiveGridPathPlanner(full_grid, min_path_clearance_m=min_clearance)
        result = planner.find_path((cam_x, cam_z), target_xz)
        if result is None:
            return None, None, None, (
                f"No route found toward ({target_xz[0]:.2f}, {target_xz[1]:.2f}) — "
                f"not reachable even through unexplored territory."
            )
        route, confirmed, reached_exactly = result
        full_route = [(cam_x, cam_z)] + route
        dist = sum(
            math.hypot(full_route[i][0] - full_route[i - 1][0], full_route[i][1] - full_route[i - 1][1])
            for i in range(1, len(full_route))
        )
        reach_clause = (
            "reached the destination exactly" if reached_exactly
            else (
                f"could NOT reach the exact destination — this is the "
                f"**closest approach** found, "
                f"{math.hypot(full_route[-1][0] - target_xz[0], full_route[-1][1] - target_xz[1]):.1f}m short"
            )
        )
        confirmed_clause = (
            "fully confirmed (all scanned ground)" if confirmed
            else "crosses **unexplored territory** (dashed orange on the maps)"
        )
        status = f"Route found — **{dist:.1f}m**, {reach_clause}, {confirmed_clause}."
        return full_route, confirmed, reached_exactly, status

    def _nav_find(location_id: str, target_x, target_z, min_clearance: float, nav_state: dict):
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is None:
            return nav_state, "No active session. Run Scan first.", gr.update(), gr.update()
        if target_x is None or target_z is None:
            return nav_state, "Enter a target X/Z (or pick a landmark) first.", gr.update(), gr.update()
        target_xz = (float(target_x), float(target_z))
        min_clearance = float(min_clearance or 0.0)
        route, confirmed, reached_exactly, status = _nav_route_full_and_status(
            session, target_xz, min_clearance)
        new_state = dict(
            target=target_xz, route=route, confirmed=confirmed, reached_exactly=reached_exactly,
            min_clearance=min_clearance, computed_at=session.occupancy_map._update_count,
        )
        occ_fig = session.occupancy_map.render_plotly(
            zones=session.zones, route=route, route_confirmed=confirmed)
        conf_fig = session.occupancy_map.render_confidence_plotly(
            zones=session.zones, route=route, route_confirmed=confirmed)
        return new_state, status, occ_fig, conf_fig

    def _nav_landmark_selected(landmark_name: str, location_id: str):
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is None or not landmark_name:
            return gr.update(), gr.update()
        for zone in session.labeler.zones:
            for lm in zone.landmarks:
                if lm.name == landmark_name:
                    return lm.x, lm.z
        return gr.update(), gr.update()

    def _nav_landmark_choices(location_id: str) -> list:
        location_id = (location_id or "").strip() or "default"
        session = scan_manager.get(location_id)
        if session is None:
            return []
        return sorted({lm.name for zone in session.labeler.zones for lm in zone.landmarks})

    def _nav_refresh_landmarks(location_id: str):
        """Manual 'Refresh Landmarks' button — session.labeler.zones[*]
        .landmarks is only kept current by preview_landmarks(), which runs
        once per processed chunk during an active stream; this lets the
        dropdown be refreshed on demand too (e.g. after switching Location
        ID to an existing session)."""
        return gr.update(choices=_nav_landmark_choices(location_id))

    def _apply_vlm_model(model_id: str):
        if not scan_manager.semantic_mapper_available:
            return "Semantic mapping is disabled on this server — nothing to apply."
        model_id = (model_id or "").strip()
        if not model_id:
            return "Model ID can't be empty."
        scan_manager.set_semantic_mapper_model(model_id)
        print(f"[Scan GUI] Semantic mapper VLM model switched to '{model_id}'")
        return f"Now using `{model_id}` for the next VLM call (existing SamplingParams/config unchanged)."

    # ── layout ─────────────────────────────────────────────────────────────────

    with gr.Blocks(theme=get_scan_theme(), css=SCAN_CSS, title="Scan Server") as app:

        gr.HTML(SCAN_HEADER_HTML)
        gr.HTML(SCAN_DESCRIPTION_HTML)

        depth_data_state    = gr.State(value=None)
        measure_points_state = gr.State(value=[])
        all_frames_state    = gr.State(value=[])   # list of (rgb, DepthFrame, pose 4×4)
        video_rotation_state = gr.State(value=0)   # extra rotation in degrees (0/90/180/270)
        manual_replay_state = gr.State(value=None)  # ManualDatasetReplayer, see "Manual Live Stream"
        nav_state = gr.State(value=dict(
            target=None, route=None, confirmed=None, reached_exactly=None,
            min_clearance=0.0, computed_at=-1,
        ))

        with gr.Row():

            # ── Left: inputs ─────────────────────────────────────────────────
            with gr.Column(scale=2):
                dataset_path_input = gr.Textbox(
                    label="Dataset folder (images/ + imu.csv + camera.csv)",
                    placeholder="e.g. uploads/<scan_id>/dataset — or an absolute path",
                    interactive=True,
                )
                with gr.Row():
                    preview_dataset_btn = gr.Button("Preview Dataset", size="sm", scale=2)
                    rotate_video_btn = gr.Button("Rotate Images 90°", size="sm", scale=1)
                    video_rotation_display = gr.Textbox(
                        value="Rotation: 0°", label="", interactive=False, scale=2,
                        container=False,
                    )
                pose_src_radio = gr.Radio(
                    choices=["IMU + VO", "RTAB-Map"],
                    value="IMU + VO",
                    label="Pose source",
                    info=(
                        "IMU + VO = FeatureTracker VO translation + IMU gyro rotation "
                        "(falls back to VO only if the dataset has no imu.csv).  "
                        "RTAB-Map requires a connected rtabmap_docker container "
                        "(see scan_server/rtabmap_docker/README.md) — RGB-D visual "
                        "odometry + loop closure using DA3-estimated depth; does NOT "
                        "use imu.csv at all, no per-device calibration needed."
                    ),
                    interactive=True,
                )
                imu_orientation_dd = gr.Dropdown(
                    choices=list(_IMU_ORIENTATION_LABELS.keys()),
                    value="Portrait (native)",
                    label="IMU Orientation (how the phone was physically held while recording)",
                    info=(
                        "Android's accelerometer/gyroscope are always reported in the "
                        "phone's fixed native (portrait) frame — set this to match how "
                        "you actually held the phone for this dataset so the raw imu.csv "
                        "samples get rotated into the same frame as the camera images "
                        "before any pose is computed. Leave at Portrait if you held it "
                        "in portrait, or the dataset was already normalized.  "
                        "⚠️ This is a DIFFERENT correction from the Axis Mapping accordion "
                        "below (that one remaps the already-computed pose, this one remaps "
                        "raw IMU samples before integration). If Axis Mapping alone already "
                        "gives a correct cloud, leave this at Portrait — stacking both will "
                        "double-rotate and make it worse, not better."
                    ),
                    interactive=True,
                )
                with gr.Row():
                    s_fps = gr.Slider(
                        minimum=0.1, maximum=10, value=1, step=0.1,
                        label="Sampling FPS", interactive=True,
                    )
                    batch_size_input = gr.Slider(
                        minimum=1, maximum=32, value=1, step=1,
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

                with gr.Accordion("Axis Mapping", open=False):
                    gr.Markdown(
                        "Applied to every pose **before** Scan processing — affects the "
                        "point cloud, Voxelization, and Occupancy Map together (not just "
                        "the Live Points preview). Default: Roll←+Y, Pitch←+X (X/Y swapped), "
                        "Yaw←+Z. Set this before clicking Scan; changing it mid-session and "
                        "re-scanning will mix differently-mapped poses into the same "
                        "accumulated cloud — Clear Cloud first if you do."
                    )
                    with gr.Row():
                        axis_roll_dd = gr.Dropdown(
                            choices=_AXIS_CHOICES, value="+Y",
                            label="Roll ← axis", scale=1, interactive=True,
                        )
                        axis_pitch_dd = gr.Dropdown(
                            choices=_AXIS_CHOICES, value="+X",
                            label="Pitch ← axis", scale=1, interactive=True,
                        )
                        axis_yaw_dd = gr.Dropdown(
                            choices=_AXIS_CHOICES, value="+Z",
                            label="Yaw ← axis", scale=1, interactive=True,
                        )

                with gr.Accordion("Outlier Removal (SOR)", open=False):
                    gr.Markdown(
                        "Statistical Outlier Removal, applied after every batch's voxel "
                        "downsample (point cloud, Voxelization, Occupancy Map, and the "
                        "exported map all reflect it). For each point, compares its "
                        "average distance to its **Neighbors** nearest neighbors against "
                        "the cloud-wide mean; points beyond **Std Ratio** standard "
                        "deviations are dropped. Higher Std Ratio / higher Neighbors = "
                        "gentler (keeps more points)."
                    )
                    with gr.Row():
                        sor_neighbors_input = gr.Slider(
                            minimum=4, maximum=50, step=1, value=20,
                            label="Neighbors (nb_neighbors)", scale=1, interactive=True,
                        )
                        sor_std_ratio_input = gr.Slider(
                            minimum=0.5, maximum=5.0, step=0.1, value=2.25,
                            label="Std Ratio", scale=1, interactive=True,
                        )

                with gr.Accordion("Occupancy Map Settings", open=False):
                    gr.Markdown(
                        "Read fresh at the start of every Scan/Simulated Live Stream run "
                        "(see `occupancy_map.py`'s Bayesian log-odds + height-gated ray "
                        "casting). Everything already accumulated keeps its belief — these "
                        "only change how FUTURE observations are weighed, so if a map is "
                        "already wrong, Clear Cloud and re-run to see the new settings take "
                        "full effect from scratch."
                    )
                    gr.Markdown(
                        "**Height tiers** (metres above the estimated floor):"
                    )
                    with gr.Row():
                        occ_obstacle_min_h = gr.Slider(
                            minimum=0.0, maximum=0.5, step=0.01, value=0.10,
                            label="Ground max height", scale=1, interactive=True,
                        )
                        occ_step_over_max_h = gr.Slider(
                            minimum=0.1, maximum=1.0, step=0.01, value=0.40,
                            label="Step-over max height", scale=1, interactive=True,
                        )
                        occ_obstacle_max_h = gr.Slider(
                            minimum=1.0, maximum=3.0, step=0.05, value=2.20,
                            label="Ceiling height (ignore above)", scale=1, interactive=True,
                        )
                    gr.Markdown(
                        "**Bayesian log-odds** — how strongly each single hit/miss moves a "
                        "cell's belief, and how many agreeing observations are needed to "
                        "confirm occupied/free. Raise **Miss weight** or lower the "
                        "**Occupied** threshold if too many real obstacles are being eroded "
                        "away; lower **Miss weight** or raise **Occupied** if noise/clutter "
                        "is showing up as false obstacles."
                    )
                    with gr.Row():
                        occ_logodds_hit = gr.Slider(
                            minimum=0.1, maximum=2.0, step=0.05, value=0.85,
                            label="Hit weight", scale=1, interactive=True,
                        )
                        occ_logodds_miss = gr.Slider(
                            minimum=0.05, maximum=1.5, step=0.05, value=0.40,
                            label="Miss weight", scale=1, interactive=True,
                        )
                    with gr.Row():
                        occ_logodds_occ_thresh = gr.Slider(
                            minimum=0.2, maximum=3.0, step=0.05, value=1.0,
                            label="Occupied confirm threshold", scale=1, interactive=True,
                        )
                        occ_logodds_free_thresh = gr.Slider(
                            minimum=-3.0, maximum=-0.2, step=0.05, value=-1.0,
                            label="Free confirm threshold", scale=1, interactive=True,
                        )
                        occ_height_ewma_alpha = gr.Slider(
                            minimum=0.05, maximum=1.0, step=0.05, value=0.30,
                            label="Height recency weight (EWMA α)", scale=1, interactive=True,
                        )
                    gr.Markdown(
                        "**Algorithm toggles** — disable either mechanism for comparison or a "
                        "simpler/cheaper pass. With Bayesian off, a single hit permanently "
                        "classifies a cell (no revision), so ray casting's misses become "
                        "no-ops even if left on."
                    )
                    with gr.Row():
                        occ_enable_ray_casting = gr.Checkbox(
                            value=True, label="Free-space ray casting", scale=1,
                        )
                        occ_enable_bayesian = gr.Checkbox(
                            value=True, label="Bayesian log-odds belief", scale=1,
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

                with gr.Row():
                    vlm_model_input = gr.Textbox(
                        label="Semantic Mapper VLM Model ID",
                        value=(
                            scan_manager.semantic_mapper_model_id
                            if scan_manager.semantic_mapper_available
                            else "(semantic mapping disabled on this server)"
                        ),
                        interactive=scan_manager.semantic_mapper_available,
                        scale=3,
                    )
                    vlm_model_apply_btn = gr.Button(
                        "Apply", size="sm", scale=1,
                        interactive=scan_manager.semantic_mapper_available,
                    )
                vlm_model_status = gr.Markdown("")

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
                    "Point at a dataset folder, fill the segment table, then click **Scan**."
                )

                with gr.Tabs():

                    with gr.Tab("Live Reconstruction"):
                        gr.Markdown(
                            "Continuously rebuilt as data arrives during Scan/Simulated "
                            "Live Stream: **Live Points → Voxelization → Occupancy Map** "
                            "(each chunk's new points feed the Occupancy Map after being "
                            "voxel-downsampled at the **Voxel size** below, so occupancy "
                            "cells correspond to what Voxelization shows). Uncheck a view "
                            "to skip recomputing it — Live Points/Voxelization cost grows "
                            "with the scan, unlike the Occupancy Map's incremental update."
                        )
                        with gr.Row():
                            show_live_points_cb = gr.Checkbox(
                                value=True, label="Live Points", scale=1,
                            )
                            show_voxelization_cb = gr.Checkbox(
                                value=True, label="Voxelization", scale=1,
                            )
                            show_occupancy_cb = gr.Checkbox(
                                value=True, label="Occupancy Map", scale=1,
                            )
                            reload_all_btn = gr.Button(
                                "Reload", size="sm", scale=0,
                            )
                            clear_cloud_btn = gr.Button(
                                "Clear Cloud", variant="stop", size="sm", scale=0,
                            )
                        with gr.Row():
                            voxel_size_input = gr.Slider(
                                minimum=0.02, maximum=0.5, step=0.01, value=DEFAULT_VOXEL_SIZE,
                                label="Voxel size (m) — also feeds the Occupancy Map each chunk",
                                scale=3,
                            )
                            voxelize_btn = gr.Button(
                                "Voxelize", variant="secondary", size="sm", scale=1,
                            )
                        with gr.Row():
                            with gr.Column():
                                live_cloud_plot = gr.Model3D(
                                    height=420,
                                    zoom_speed=0.5,
                                    pan_speed=0.5,
                                    clear_color=[0.05, 0.05, 0.05, 1.0],
                                    label="Live Points — 3D point cloud, builds up as data arrives",
                                )
                            with gr.Column():
                                voxel_plot = gr.Model3D(
                                    height=420,
                                    zoom_speed=0.5,
                                    pan_speed=0.5,
                                    clear_color=[0.05, 0.05, 0.05, 1.0],
                                    label="Voxelization — voxelized point cloud",
                                )
                        with gr.Row():
                            with gr.Column():
                                occupancy_plot = gr.Plot(
                                    label="Occupancy Map (top-down X-Z) — fed from each chunk's voxelization"
                                )
                            with gr.Column():
                                confidence_plot = gr.Plot(
                                    label="Confidence Map (top-down X-Z) — how much agreeing "
                                          "evidence each cell has, independent of free/obstacle"
                                )

                        with gr.Accordion("Live Navigation Preview", open=False):
                            gr.Markdown(
                                "Pick a destination (typed X/Z, or a scanned landmark) and see "
                                "a route computed against the **in-progress** map — prefers "
                                "confirmed-free cells, but still routes through unexplored "
                                "territory (flagged **speculative**, dashed orange) rather than "
                                "failing when no confirmed route exists yet. During **Simulated "
                                "Live Stream**, the route recomputes automatically as new data "
                                "arrives; in Manual mode, click **Find Route** again after each "
                                "step to refresh it."
                            )
                            with gr.Row():
                                nav_target_x = gr.Number(label="Target X (m)", value=None)
                                nav_target_z = gr.Number(label="Target Z (m)", value=None)
                                nav_landmark_dropdown = gr.Dropdown(
                                    label="Or pick a landmark", choices=[], value=None,
                                )
                                nav_refresh_landmarks_btn = gr.Button(
                                    "↻ Landmarks", size="sm", scale=0,
                                )
                            with gr.Row():
                                nav_min_clearance_input = gr.Slider(
                                    minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                                    label="Minimum path width (m) — 0 disables; routes avoid "
                                          "squeezing narrower than this when a wider option exists",
                                )
                            with gr.Row():
                                nav_find_btn = gr.Button("Find Route", variant="primary", size="sm")
                            nav_status = gr.Markdown("No destination set.")

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
                                    "**Axis remap (preview only)** — further fine-tune on top of "
                                    "the pre-scan Axis Mapping setting above, without re-scanning. "
                                    "If horizontal pan shows as wrong angle, swap axes here. "
                                    "Default: Roll←+X, Pitch←+Y, Yaw←+Z (no additional change).",
                                    scale=3,
                                )
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

                    with gr.Tab("Detections"):
                        with gr.Row():
                            reload_detections_btn = gr.Button(
                                "Reload", size="sm", scale=0,
                            )
                        detection_image = gr.Image(
                            type="numpy",
                            label="Most recent frame — GroundingDINO boxes",
                            format="png", interactive=False, sources=[],
                            height=400,
                        )
                        detection_text = gr.Markdown(
                            "Run Scan with a labelled area to see detections here."
                        )

                with gr.Row():
                    export_btn = gr.Button("Export Map", variant="secondary", scale=1)

                with gr.Row():
                    simulated_stream_btn = gr.Button(
                        "Simulated Live Stream", variant="primary", scale=3,
                    )
                    realtime_pacing_cb = gr.Checkbox(
                        label="Real-time pacing", value=False, scale=1,
                        info="Off = replay as fast as possible. On = pace to the "
                             "dataset's own recorded timing, as a real live stream would.",
                    )
                gr.Markdown(
                    "**Simulated Live Stream** replays the dataset above frame-by-frame "
                    "and IMU-sample-by-sample through the same streaming interface a real "
                    "Android live source will use later (see stream_session.py / "
                    "stream_simulator.py) — the point cloud updates as each chunk is "
                    "processed, not only at the end. Voxelization / Occupancy Map still "
                    "finalize once, when the replay finishes."
                )

                with gr.Row():
                    manual_start_btn = gr.Button(
                        "Start / Reset Manual Stream", variant="secondary", scale=2,
                    )
                    manual_feed_btn = gr.Button(
                        "Feed Next Frame ▶", variant="primary", scale=2, interactive=False,
                    )
                    manual_preview_image = gr.Image(
                        label="Next frame to feed", type="numpy", interactive=False,
                        sources=[], height=100, scale=1, show_label=True,
                    )
                gr.Markdown(
                    "**Manual Live Stream** feeds the dataset above through the same "
                    "streaming interface as Simulated Live Stream, one frame per click "
                    "instead of an automatic loop. Click **Start / Reset Manual Stream** "
                    "to load the dataset and preview its first frame, then **Feed Next "
                    "Frame** repeatedly to advance — any IMU samples/zone boundaries "
                    "between frames are applied automatically along with each click. "
                    "The button disables and the map is finalized + exported once no "
                    "frames remain."
                )

        with gr.Row():
            scan_status = gr.Textbox(
                label="Status", value="Waiting for a dataset folder…", interactive=False,
            )
            scan_position = gr.Textbox(
                label="Last Camera Position (m)", value="x=0.00  y=0.00  z=0.00",
                interactive=False,
            )

        export_log = gr.Markdown("")

        # ── event wiring ───────────────────────────────────────────────────────

        def _rotate_video(rotation, dataset_path, fps_val):
            new_rot = (rotation + 90) % 360
            gallery, msg, segs = _handle_dataset_change(dataset_path, fps_val, new_rot)
            return new_rot, f"Rotation: {new_rot}°", gallery, msg, segs

        rotate_video_btn.click(
            fn=_rotate_video,
            inputs=[video_rotation_state, dataset_path_input, s_fps],
            outputs=[video_rotation_state, video_rotation_display, frame_gallery, log_output, segment_table],
        )

        preview_dataset_btn.click(
            fn=_handle_dataset_change,
            inputs=[dataset_path_input, s_fps, video_rotation_state],
            outputs=[frame_gallery, log_output, segment_table],
        )

        dataset_path_input.change(
            fn=_handle_dataset_change,
            inputs=[dataset_path_input, s_fps, video_rotation_state],
            outputs=[frame_gallery, log_output, segment_table],
        )

        export_btn.click(
            fn=_export_map,
            inputs=[location_id_input],
            outputs=[export_log, live_cloud_plot, voxel_plot, occupancy_plot, confidence_plot],
        )

        simulated_stream_btn.click(
            fn=_run_simulated_stream,
            inputs=[dataset_path_input, s_fps, batch_size_input,
                    location_id_input, segment_table, resolution_input,
                    pose_src_radio, video_rotation_state, imu_orientation_dd, zone_type_input,
                    axis_roll_dd, axis_pitch_dd, axis_yaw_dd,
                    sor_neighbors_input, sor_std_ratio_input, voxel_size_input,
                    realtime_pacing_cb,
                    occ_obstacle_min_h, occ_step_over_max_h, occ_obstacle_max_h,
                    occ_logodds_hit, occ_logodds_miss,
                    occ_logodds_occ_thresh, occ_logodds_free_thresh, occ_height_ewma_alpha,
                    occ_enable_ray_casting, occ_enable_bayesian,
                    show_live_points_cb, show_voxelization_cb, show_occupancy_cb,
                    nav_state],
            outputs=[
                live_cloud_plot, voxel_plot, occupancy_plot, confidence_plot,
                detection_image, detection_text,
                scan_status, scan_position, log_output,
                nav_state, nav_status,
            ],
        )

        manual_start_btn.click(
            fn=_manual_stream_start,
            inputs=[dataset_path_input, s_fps, batch_size_input,
                    location_id_input, segment_table, resolution_input,
                    pose_src_radio, video_rotation_state, imu_orientation_dd, zone_type_input,
                    axis_roll_dd, axis_pitch_dd, axis_yaw_dd,
                    sor_neighbors_input, sor_std_ratio_input, voxel_size_input,
                    occ_obstacle_min_h, occ_step_over_max_h, occ_obstacle_max_h,
                    occ_logodds_hit, occ_logodds_miss,
                    occ_logodds_occ_thresh, occ_logodds_free_thresh, occ_height_ewma_alpha,
                    occ_enable_ray_casting, occ_enable_bayesian],
            outputs=[manual_replay_state, manual_preview_image, manual_feed_btn, log_output],
        )

        manual_feed_btn.click(
            fn=_manual_stream_feed,
            inputs=[manual_replay_state,
                    show_live_points_cb, show_voxelization_cb, show_occupancy_cb],
            outputs=[
                manual_replay_state, manual_preview_image, manual_feed_btn,
                live_cloud_plot, voxel_plot, occupancy_plot, confidence_plot,
                detection_image, detection_text,
                scan_status, scan_position, log_output,
            ],
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

        reload_all_btn.click(
            fn=_reload_all_views,
            inputs=[location_id_input, voxel_size_input],
            outputs=[live_cloud_plot, voxel_plot, occupancy_plot, confidence_plot],
        )

        show_live_points_cb.change(
            fn=_toggle_live_points,
            inputs=[show_live_points_cb, location_id_input],
            outputs=[live_cloud_plot],
        )
        show_voxelization_cb.change(
            fn=_toggle_voxelization,
            inputs=[show_voxelization_cb, location_id_input, voxel_size_input],
            outputs=[voxel_plot],
        )
        show_occupancy_cb.change(
            fn=_toggle_occupancy,
            inputs=[show_occupancy_cb, location_id_input],
            outputs=[occupancy_plot, confidence_plot],
        )

        clear_cloud_btn.click(
            fn=_clear_cloud,
            inputs=[location_id_input],
            outputs=[live_cloud_plot, voxel_plot, occupancy_plot, confidence_plot],
        )

        nav_find_btn.click(
            fn=_nav_find,
            inputs=[location_id_input, nav_target_x, nav_target_z, nav_min_clearance_input, nav_state],
            outputs=[nav_state, nav_status, occupancy_plot, confidence_plot],
        )
        nav_landmark_dropdown.change(
            fn=_nav_landmark_selected,
            inputs=[nav_landmark_dropdown, location_id_input],
            outputs=[nav_target_x, nav_target_z],
        )
        nav_refresh_landmarks_btn.click(
            fn=_nav_refresh_landmarks,
            inputs=[location_id_input],
            outputs=[nav_landmark_dropdown],
        )

        reload_detections_btn.click(
            fn=_reload_detections,
            inputs=[location_id_input],
            outputs=[detection_image, detection_text],
        )

        voxelize_btn.click(
            fn=_voxelize,
            inputs=[location_id_input, voxel_size_input],
            outputs=[voxel_plot],
        )

        vlm_model_apply_btn.click(
            fn=_apply_vlm_model,
            inputs=[vlm_model_input],
            outputs=[vlm_model_status],
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

        load_upload_btn.click(
            fn=_load_upload,
            inputs=[upload_dropdown],
            outputs=[dataset_path_input],
        )

    return app
