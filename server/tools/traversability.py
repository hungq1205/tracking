"""
traversability.py — per-frame, no-world-map polar obstacle-clearance fan
(classic Vector-Field-Histogram style), the local reactive layer walking/
guiding's HRTF beacon is now driven from instead of the old occupancy-grid
ray-cast steering (see CLAUDE.md's "Local reactive HRTF obstacle-dodge"
note). Stateless — every call classifies ground vs. obstacle from THIS
frame's own depth alone via a single-frame RANSAC ground-plane fit; no IMU,
no persisted ground_y (unlike occupancy_map.py's mapping-mode Bayesian
grid, which this deliberately does NOT reuse — a world map/RTAB-Map update
cycle is too slow/laggy for reactive per-frame dodging).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Pinhole intrinsics fallback — same fx=fy=0.8*max(w,h) approximation used
# throughout this codebase wherever real per-device intrinsics aren't
# available (server/tools/depth.py's old _estimate_K, beacon_preview.py,
# HrtfBeacon.kt's directionFromBox()).
_FOCAL_SCALE = 0.8

_GROUND_BAND_FRACTION = 0.4      # fit the ground plane from the bottom 40% of the frame
_PIXEL_STRIDE = 4                # subsample every 4th pixel per axis before back-projecting
_RANSAC_ITERS = 60
_RANSAC_INLIER_THRESH_M = 0.05   # 5cm
_MIN_GROUND_INLIERS = 30
_OBSTACLE_MIN_HEIGHT_M = 0.12    # ignore floor texture/noise
_OBSTACLE_MAX_HEIGHT_M = 2.0     # ignore ceiling
_MIN_VALID_DEPTH_M = 0.1
# A flat obstacle filling most/all of the frame (e.g. a wall dead ahead,
# nothing but a near obstacle in view) is itself a perfectly good RANSAC
# plane fit — just not a horizontal one. Reject any candidate whose normal
# isn't substantially aligned with the vertical (Y) axis so that case can't
# get silently accepted as "the floor" and read back as fully open; see the
# fully-blocked-frame regression this constant was added to fix.
_MIN_GROUND_NORMAL_VERTICALITY = 0.5


@dataclass
class TraversabilityResult:
    clearance_m: List[float]
    min_angle_deg: float
    max_angle_deg: float
    angle_step_deg: float
    max_range_m: float


def _estimate_k(w: int, h: int) -> Tuple[float, float, float]:
    f = _FOCAL_SCALE * max(w, h)
    return f, w / 2.0, h / 2.0


def _fit_ground_plane(points: np.ndarray, rng: np.random.Generator) -> Optional[np.ndarray]:
    """RANSAC-fits a plane [a,b,c,d] (a*x+b*y+c*z+d=0, [a,b,c] normalized)
    to `points` (Nx3), restricted to near-horizontal candidates (see
    _MIN_GROUND_NORMAL_VERTICALITY). Returns None if too few inliers were
    ever found for a plausible floor orientation — the caller treats that
    as "no visible floor" and degrades conservatively rather than
    guessing."""
    n = points.shape[0]
    if n < _MIN_GROUND_INLIERS:
        return None
    best_inliers = 0
    best_plane = None
    for _ in range(_RANSAC_ITERS):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        if abs(normal[1]) < _MIN_GROUND_NORMAL_VERTICALITY:
            continue  # not floor-like — e.g. a frontal wall/obstacle plane
        d = -np.dot(normal, p0)
        dist = np.abs(points @ normal + d)
        inliers = int(np.count_nonzero(dist < _RANSAC_INLIER_THRESH_M))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], d])
    if best_plane is None or best_inliers < _MIN_GROUND_INLIERS:
        return None
    return best_plane


def estimate_traversability(
    depth_map: np.ndarray,
    num_bins: int = 25,
    max_range_m: float = 5.0,
) -> TraversabilityResult:
    """`depth_map`: HxW float32 metric depth (same DA3-METRIC output
    DA3DepthDetector.check_obstacle already consumes). Returns a
    TraversabilityResult spanning this frame's own estimated horizontal
    FOV (via the pinhole fallback above) — deliberately no side/rear
    coverage, since this only ever reacts to what's actually in view this
    frame (see the module docstring)."""
    h, w = depth_map.shape
    f, cx, cy = _estimate_k(w, h)
    half_fov_deg = float(np.degrees(np.arctan((w / 2.0) / f)))
    angle_step = (2 * half_fov_deg) / num_bins

    result = TraversabilityResult(
        clearance_m=[max_range_m] * num_bins,
        min_angle_deg=-half_fov_deg, max_angle_deg=half_fov_deg,
        angle_step_deg=angle_step, max_range_m=max_range_m,
    )

    us = np.arange(0, w, _PIXEL_STRIDE)
    vs = np.arange(0, h, _PIXEL_STRIDE)
    grid_u, grid_v = np.meshgrid(us, vs)
    depths = depth_map[grid_v, grid_u]
    valid = depths > _MIN_VALID_DEPTH_M
    grid_u, grid_v, depths = grid_u[valid], grid_v[valid], depths[valid]
    if depths.size == 0:
        return result

    xs = (grid_u - cx) * depths / f
    ys = (grid_v - cy) * depths / f  # fx == fy under this pinhole approximation
    zs = depths
    points = np.stack([xs, ys, zs], axis=-1)

    def _bin_index(azimuth_deg: np.ndarray) -> np.ndarray:
        return np.clip(((azimuth_deg - result.min_angle_deg) / angle_step).astype(int), 0, num_bins - 1)

    ground_row_mask = grid_v >= h * (1 - _GROUND_BAND_FRACTION)
    plane = _fit_ground_plane(points[ground_row_mask], np.random.default_rng())

    if plane is None:
        # No confident floor found (e.g. a near obstacle filling the whole
        # frame) — degrade conservatively: treat every visible point as an
        # obstacle rather than guessing a ground plane that isn't there.
        obstacle_points = points
    else:
        normal, d = plane[:3], plane[3]
        height_above = points @ normal + d
        # Sign convention: ground points read ~0 under EITHER orientation
        # of `normal` (that's what makes them plane inliers in the first
        # place), so their own height_above can't tell us which way is
        # "up". The camera origin can: it sits on the same side of the
        # floor plane as any real obstacle (both are between the floor and
        # the camera), and its signed distance to the plane is exactly d
        # (normal . [0,0,0] + d). Flip so that reads positive.
        if d < 0:
            height_above = -height_above
        is_obstacle = (height_above > _OBSTACLE_MIN_HEIGHT_M) & (height_above < _OBSTACLE_MAX_HEIGHT_M)
        obstacle_points = points[is_obstacle]

    if obstacle_points.shape[0] == 0:
        return result

    az = np.degrees(np.arctan2(obstacle_points[:, 0], obstacle_points[:, 2]))
    bins = _bin_index(az)
    clearances = result.clearance_m
    for b, z in zip(bins, obstacle_points[:, 2]):
        z = float(min(z, max_range_m))
        if z < clearances[b]:
            clearances[b] = z
    return result
