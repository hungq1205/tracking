"""
traversability.py — per-frame, no-world-map polar obstacle-clearance fan
(classic Vector-Field-Histogram style). GUIDING's local reactive HRTF
dodge layer is still driven from this directly (see CLAUDE.md's "Local
reactive HRTF obstacle-dodge" note) — WALKING's own steering moved back to
a client-side LocalPathPlanner route through a live occupancy grid (see
"Local SLAM-backed walking corridor-lock", superseded by the walking-mode
local-map note), but still uses this module's dropoff_m for its
proximity-based step-down/stairs warning (see "Hazard warnings" note) —
that check is intentionally stateless/single-frame, decoupled from the
grid, since it only needs to answer "is something dangerous close right
now," not "where can I walk." Stateless — every call classifies ground vs.
obstacle (and, via dropoff_m, drop-off) from THIS frame's own depth alone
via a single-frame RANSAC ground-plane fit; no IMU, no persisted ground_y
(unlike occupancy_map.py's mapping-mode Bayesian grid).
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
# A point significantly BELOW the fitted ground plane (negative
# height_above beyond this) is a drop-off — a step down, ledge, or
# staircase, not floor texture noise. Distinct from _OBSTACLE_MIN_HEIGHT_M,
# which only ever looks ABOVE the plane; without a below-plane check, a
# downward step registered as neither obstacle nor ground and read back as
# fully open. Same order of magnitude as _OBSTACLE_MIN_HEIGHT_M (a real
# step/curb is typically >= 15cm) — see CLAUDE.md's hazard-warning note.
_DROP_MIN_DEPTH_M = 0.10
# The ground plane is fit from the bottom-band points, but restricted to
# NEAR-range candidates first (points within this distance) rather than
# every bottom-band point regardless of depth. Without this, a scene
# containing both the user's real local floor AND a large descending
# surface further away (stairs, a ramp, a sloped exit) in the same bottom
# 40% of frame lets RANSAC latch onto whichever is bigger/more planar —
# often the distant stairs themselves, not the floor the user is actually
# standing on — which then reads as "the ground plane" and makes the
# stairs measure as ~0 height above themselves (no drop-off detected at
# all). The true local floor is always the nearest ground-level surface,
# so restricting the fit to near-range points first strongly prefers it.
# Falls back to the full bottom-band (old behavior) if too few near-range
# points exist, so open rooms whose nearest floor patch is farther than
# this aren't regressed.
_GROUND_FIT_MAX_RANGE_M = 2.5
# A flat obstacle filling most/all of the frame (e.g. a wall dead ahead,
# nothing but a near obstacle in view) is itself a perfectly good RANSAC
# plane fit — just not a horizontal one. Reject any candidate whose normal
# isn't substantially aligned with the vertical (Y) axis so that case can't
# get silently accepted as "the floor" and read back as fully open; see the
# fully-blocked-frame regression this constant was added to fix.
_MIN_GROUND_NORMAL_VERTICALITY = 0.5
# Monocular (DA3) depth is noisy per-pixel — a handful of stray pixels
# reading a bit below the fitted plane is common floor noise, not a real
# step. Unlike obstacle clearance (where a bin's reported distance is
# naturally dominated by whichever real cluster is nearest, noise or not),
# a single noisy pixel below the plane used to be enough to flag a bin as
# a drop-off outright. Require several agreeing pixels in the SAME bin
# before trusting it as a real ledge/step rather than noise.
_MIN_DROPOFF_POINTS_PER_BIN = 3


@dataclass
class TraversabilityResult:
    clearance_m: List[float]
    min_angle_deg: float
    max_angle_deg: float
    angle_step_deg: float
    max_range_m: float
    dropoff_m: List[float]


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
    return _clearance_fan_from_depth(depth_map, num_bins, max_range_m)


def _clearance_fan_from_depth(
    depth_map: np.ndarray,
    num_bins: int,
    max_range_m: float,
) -> TraversabilityResult:
    """Body of estimate_traversability() — kept as a separate function so
    both the clearance (obstacle) and dropoff (drop-off/step-down) fans are
    computed from one shared ground-plane fit."""
    h, w = depth_map.shape
    f, cx, cy = _estimate_k(w, h)
    half_fov_deg = float(np.degrees(np.arctan((w / 2.0) / f)))
    angle_step = (2 * half_fov_deg) / num_bins

    result = TraversabilityResult(
        clearance_m=[max_range_m] * num_bins,
        min_angle_deg=-half_fov_deg, max_angle_deg=half_fov_deg,
        angle_step_deg=angle_step, max_range_m=max_range_m,
        dropoff_m=[max_range_m] * num_bins,
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
    # Prefer fitting the plane from NEAR-range bottom-band points (see
    # _GROUND_FIT_MAX_RANGE_M's own comment) — falls back to the full
    # bottom-band if that leaves too few candidates for a confident fit.
    near_ground_mask = ground_row_mask & (zs <= _GROUND_FIT_MAX_RANGE_M)
    ground_fit_mask = near_ground_mask if np.count_nonzero(near_ground_mask) >= _MIN_GROUND_INLIERS else ground_row_mask
    plane = _fit_ground_plane(points[ground_fit_mask], np.random.default_rng())

    dropoff_points = None
    if plane is None:
        # No confident floor found (e.g. a near obstacle filling the whole
        # frame) — degrade conservatively: treat every visible point as an
        # obstacle rather than guessing a ground plane that isn't there.
        # No plane also means no reference to measure "below" against, so
        # drop-off detection is skipped entirely here (dropoff_m stays at
        # the max_range_m sentinel) — the obstacle-everywhere fallback
        # already covers this frame conservatively either way.
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
        # Points significantly BELOW the plane (negative height_above) are
        # a drop-off — a step down, ledge, or staircase. See
        # _DROP_MIN_DEPTH_M's own comment for why this is a distinct check
        # from is_obstacle, not just its negation.
        is_dropoff = height_above < -_DROP_MIN_DEPTH_M
        dropoff_points = points[is_dropoff]

    if obstacle_points.shape[0] > 0:
        az = np.degrees(np.arctan2(obstacle_points[:, 0], obstacle_points[:, 2]))
        bins = _bin_index(az)
        clearances = result.clearance_m
        for b, z in zip(bins, obstacle_points[:, 2]):
            z = float(min(z, max_range_m))
            if z < clearances[b]:
                clearances[b] = z

    if dropoff_points is not None and dropoff_points.shape[0] > 0:
        az = np.degrees(np.arctan2(dropoff_points[:, 0], dropoff_points[:, 2]))
        bins = _bin_index(az)
        # Only trust a bin once several points agree it's a drop-off —
        # see _MIN_DROPOFF_POINTS_PER_BIN's comment above.
        bin_counts = np.bincount(bins, minlength=num_bins)
        confident_bins = bin_counts >= _MIN_DROPOFF_POINTS_PER_BIN
        dropoffs = result.dropoff_m
        for b, z in zip(bins, dropoff_points[:, 2]):
            if not confident_bins[b]:
                continue
            z = float(min(z, max_range_m))
            if z < dropoffs[b]:
                dropoffs[b] = z

    return result
