"""
beacon_preview.py — server-side, VISUALIZATION-ONLY reconstruction of
"which direction is the HRTF beacon currently pointing", purely for
server_gui.py's dashboard. The real beacon direction/audio is computed
entirely on Android (ToolDispatcher.updateHrtfBeacon(), see CLAUDE.md's
"Walking mode redesign" note) for latency reasons — nothing here feeds
back into the actual audio; this exists only so the dashboard can draw a
circle showing roughly where the beacon points, both on the live camera
frame and on the occupancy map.

find_most_open_direction_world_point() duplicates (does not import)
LocalPathPlanner.kt's findMostOpenDirection() ray-cast logic — same
"separately deployed processes" reasoning already used elsewhere in this
codebase (e.g. live_path_planner.py vs. grid_path_planner.py).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_CLASS_OBSTACLE = 3  # matches occupancy_map.py's CLASS_OBSTACLE / GridClass.OBSTACLE (LocalPathPlanner.kt)


def find_most_open_direction_world_point(
    pose: np.ndarray, grid_dict: dict, max_range_m: float = 5.0,
    cone_deg: float = 90.0, step_deg: float = 15.0,
) -> Optional[Tuple[float, float]]:
    """Mirrors LocalPathPlanner.kt's findMostOpenDirection(): yaw-only
    heading from the pose's rotation (camera-local +Z axis rotated into
    world, X-Z plane — floor-constrained navigation, pitch/roll ignored),
    ray-cast per candidate azimuth through grid_dict's class array,
    returns the world (x, z) point at the best-open azimuth (at
    min(best_dist, max_range_m) along that ray), or None if even straight
    ahead is immediately blocked/out of bounds."""
    if grid_dict is None:
        return None
    cls = grid_dict["class"]
    resolution = grid_dict["resolution"]
    origin_x = grid_dict["origin_x"]
    origin_z = grid_dict["origin_z"]
    height = grid_dict["height"]
    width = grid_dict["width"]
    if height == 0 or width == 0:
        return None

    forward = pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
    yaw = float(np.arctan2(forward[0], forward[2]))
    px, pz = float(pose[0, 3]), float(pose[2, 3])

    def passable(x: float, z: float) -> bool:
        col = int((x - origin_x) / resolution)
        row = int((z - origin_z) / resolution)
        if not (0 <= row < height and 0 <= col < width):
            return False
        return cls[row][col] != _CLASS_OBSTACLE

    steps = max(int(max_range_m / resolution), 1)
    best_az: Optional[float] = None
    best_dist = 0.0
    az = -cone_deg
    while az <= cone_deg:
        rad = yaw + np.radians(az)
        dx, dz = float(np.sin(rad)), float(np.cos(rad))
        traveled = 0.0
        for i in range(1, steps + 1):
            if not passable(px + dx * resolution * i, pz + dz * resolution * i):
                break
            traveled = resolution * i
        if traveled > best_dist or (traveled == best_dist and best_az is not None and abs(az) < abs(best_az)):
            best_dist = traveled
            best_az = az
        az += step_deg

    if best_az is None:
        return None
    rad = yaw + np.radians(best_az)
    dist = min(best_dist, max_range_m)
    return px + float(np.sin(rad)) * dist, pz + float(np.cos(rad)) * dist


def project_world_point_to_pixel(
    pose: np.ndarray, world_xz: Tuple[float, float], ground_y: float,
    frame_w: int, frame_h: int,
) -> Optional[Tuple[int, int]]:
    """Projects a world (x, z) point (at ground_y height) into this frame's
    pixel coordinates via the same pinhole-K guess used elsewhere in this
    codebase (server/tools/depth.py's _estimate_K: fx=fy=0.8*max(w,h)) —
    simplest consistent choice given MappingChunk doesn't reliably carry
    real per-device intrinsics. Returns None if the point is behind the
    camera or projects outside the frame — a real directional indicator
    would likewise only show up when actually in view."""
    x, z = world_xz
    world_pt = np.array([x, ground_y, z, 1.0])
    cam_pt = np.linalg.inv(pose) @ world_pt
    cx_cam, cy_cam, cz_cam = cam_pt[0], cam_pt[1], cam_pt[2]
    if cz_cam <= 0.05:
        return None
    f = 0.8 * max(frame_w, frame_h)
    u = f * cx_cam / cz_cam + frame_w / 2.0
    v = f * cy_cam / cz_cam + frame_h / 2.0
    if not (0 <= u < frame_w and 0 <= v < frame_h):
        return None
    return int(u), int(v)
