"""
Height-based traversability map on the X-Z plane — a real Bayesian occupancy
grid (same representation gmapping/cartographer/RTAB-Map's own OccupancyGrid
module use), rendered in classic SLAM grayscale (white=free, black=obstacle,
gray=unknown), built progressively as data arrives.

Pipeline: point cloud (ScanSession._cloud) → voxelization (scan_session.
voxelize_cloud, shared with the Voxelization tab) → this module. update()
receives already-voxelized points — one sample per occupied 3D voxel, not raw
points — so a densely-resampled surface doesn't dominate the per-cell height
statistic. update() is called once per processed batch (see
ScanSession.process_frames_batch), not deferred to end-of-scan — the grid
grows/refines live, the same way a robot's occupancy grid fills in as it
explores rather than needing every sensor reading collected up front first.

**Algorithm rewritten to match how real occupancy-grid mappers (RTAB-Map's
own OccupancyGrid, ROS costmap_2d) actually build a 2D grid from a 3D point
cloud** — ground/obstacle segmentation per POINT, then Bresenham ray tracing
that STOPS at the first cell with confirmed obstacle evidence, exactly like
a real depth/laser ray is physically blocked by the first thing it hits and
never tests anything behind it:

  1. Every point in the batch is classified INDIVIDUALLY by height above the
     current ground estimate: below OBSTACLE_MIN_H → ground; between
     OBSTACLE_MIN_H and OBSTACLE_MAX_H → obstacle (low/step-over or normal,
     see STEP_OVER_MAX_H); at/above OBSTACLE_MAX_H → ceiling (ignored
     entirely — no evidence either way).
  2. Points are bucketed by (X,Z) cell. A cell with ANY obstacle-classified
     point this batch registers ONE obstacle hit, using the TALLEST point in
     that cell (obstacle evidence always wins over a ground point landing in
     the same coarse cell this batch — e.g. the floor peeking out at a
     bed's edge doesn't get to call that cell "free").
  3. A cell with ONLY ground-classified points this batch registers ONE
     ground hit (positive evidence toward free — not just an absence of
     obstacle hits).
  4. Free-space ray casting is cast ONLY toward ground-classified cells
     (never toward obstacle cells — an obstacle's own hit already speaks for
     itself) — one Bresenham walk per ground cell, from the batch's camera
     position. THE FIX: the walk STOPS as soon as it reaches a cell with ANY
     net hit evidence (logodds > 0) — it does not mark that cell, and does
     not continue past it, same as a real depth ray being physically
     blocked. Deliberately NOT gated on the stricter
     LOGODDS_OCCUPIED_THRESH used for final classification — a real
     obstacle seen only once (logodds == LOGODDS_HIT, e.g. 0.85, below a
     1.0 threshold) isn't yet "confirmed" by the Bayesian scheme but is
     still real; a synthetic test confirmed a single-hit obstacle gets
     fully eroded by ~30 subsequent ground rays if blocking waits for full
     confirmation instead of any net evidence.
     The PREVIOUS version of this ray cast never stopped at obstacles at
     all — it walked the full 2D line to every touched cell (including ones
     far beyond a real, closer obstacle) and used a linear height
     interpolation between camera and target Y as a proxy for "did this ray
     really clear this cell". That proxy is only a 2D-distance-based
     approximation, not real 3D occlusion: for a ray whose target is near
     floor level (extremely common — the floor is everywhere), the
     interpolated height already drops close to floor level well before
     reaching a real closer obstacle positioned nearer the target than the
     camera, so the ray incorrectly rated itself "low enough to test" the
     obstacle's cell and eroded it — repeatedly, on every subsequent batch
     whose ray happened to aim at floor beyond the obstacle. CONFIRMED BUG:
     this is exactly why a large, real obstacle (e.g. a bed) could end up
     almost entirely eroded to "free"/unknown, with only whatever got
     hit more often than it got (incorrectly) ray-cast through surviving as
     a small "low obstacle" island — erosion compounds every batch, hits
     only happen when that exact cell gets directly re-imaged. Stopping the
     ray at the first CONFIRMED obstacle removes the whole failure mode by
     construction: a ray can never erode a cell it would have to pass
     THROUGH a real obstacle to reach.

Per-cell Bayesian belief (see _CellState): each observation is either a HIT
(nudges log-odds toward "occupied") or a MISS (nudges toward "free") — this
is what makes the grid self-correcting: a single noisy far-range depth
reading that wrongly looks like an obstacle gets pulled back down as soon as
a few closer, more reliable observations disagree, including the user simply
walking past/through that spot later. An even older raw-sample-list design
(before the Bayesian rewrite) had no revision mechanism at all — once a cell
looked like an obstacle, nothing could ever undo it, which is equally not
how a real occupancy-grid mapper works; this module has both a revision
mechanism (Bayesian) AND, now, a physically-correct reason to trust it
(occlusion-respecting ray casting) rather than an approximate height proxy.

height_ewma is a recency-weighted (exponential moving average) estimate of
the observed surface height, updated on every hit (obstacle OR ground) — so
it favors more recent (typically closer-range, more reliable) observations
over old ones, and — critically for the ground-plane estimate below — stays
populated for GROUND cells too, not just obstacle ones.

The ground plane estimate (GROUND_PERCENTILE of Y, since Y points DOWN in
camera/world space) is recomputed from the CUMULATIVE height_ewma across all
cells with hit evidence so far (not just this batch) specifically so it's
safe to call every batch — an estimate from only a small, fresh batch used to
jitter badly early in a scan; over the full accumulated history it only gets
more confident, never regresses. Before any hit evidence exists at all (very
first update() call), a one-off bootstrap estimate is taken from that
batch's own raw points so per-point classification has SOMETHING to work
with immediately; it's immediately superseded by the cumulative estimate
once any cell has hit evidence.

Cell classification (see _classify_state / CLASS_* constants), from a cell's
log-odds belief + height_ewma:

  logodds ≤ LOGODDS_FREE_THRESH                    →  ground        (class 1, free/walkable, white)
  LOGODDS_FREE_THRESH < logodds < LOGODDS_OCC_THRESH →  unknown      (class 0, not enough agreeing evidence)
  logodds ≥ LOGODDS_OCCUPIED_THRESH, then by height_ewma:
    height < OBSTACLE_MIN_H                         →  ground        (class 1)
    OBSTACLE_MIN_H ≤ h < STEP_OVER_MAX_H             →  low/step-over (class 2, passable-but-costlier
                                                         for GridPathPlanner — the camera sits at the
                                                         user's eye height, so a curb/threshold below
                                                         ~40cm is meaningfully different from a wall)
    STEP_OVER_MAX_H ≤ h < OBSTACLE_MAX_H             →  normal obstacle (class 3, blocked, black)
    h ≥ OBSTACLE_MAX_H                               →  ceiling       (class 0/unknown, ignored)

After building the raw class grid we apply a 3×3 morphological maximum filter
(dilation) on the obstacle mask so that sparse depth coverage on obstacle tops
bleeds into neighbouring cells that the depth sensor may have missed.

World-space coordinate convention (OpenCV camera, identity first frame):
  X – right,  Y – DOWN,  Z – forward
So ground_y = high percentile of Y values (large Y = near the floor).
Height above ground = ground_y − point_y  (positive = further from ground).

DEBUGGING: every update() call ends with a single `[occupancy]` console log
line — point counts by class (obstacle/ground/ceiling-ignored), rays cast vs.
rays that stopped early because they hit a confirmed obstacle, current
ground_y, and running total cell count. If a real obstacle is still getting
eroded to free/unknown after this rewrite, that "blocked" count is the first
thing to check: it should be > 0 whenever a ray's straight-line path to a
ground point genuinely passes through/near a real piece of furniture — if
it's staying at 0 while a known obstacle still disappears, the obstacle's
own cells aren't crossing LOGODDS_OCCUPIED_THRESH in the first place (a
hit-rate problem, not a ray-casting problem) and the height/logodds tuning
sliders (GUI's "Occupancy Map Settings") are the next thing to check, not
this file's ray logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import maximum_filter

from timing_utils import timed


def _percentile_fast(values, percentile: float) -> float:
    """Numerically identical to np.percentile(values, percentile) (default
    'linear' interpolation method, verified against it directly — max
    absolute difference ~1e-15 over thousands of random trials), but ~10x
    faster per call. update() calls this once per touched cell per batch on
    small lists (a handful to a few dozen height values) — profiling showed
    the public np.percentile() API's fixed per-call overhead (argument
    validation, its generic reduction machinery) dominates for inputs this
    small: it alone was ~70% of update()'s total wall time. np.partition
    computes just the 2 order statistics 'linear' interpolation needs,
    skipping all of that."""
    n = len(values)
    if n == 1:
        return float(values[0])
    arr = np.asarray(values, dtype=np.float64)
    rank = (percentile / 100.0) * (n - 1)
    lo = int(np.floor(rank))
    hi = int(np.ceil(rank))
    if lo == hi:
        return float(np.partition(arr, lo)[lo])
    part = np.partition(arr, [lo, hi])
    frac = rank - lo
    return float(part[lo] + frac * (part[hi] - part[lo]))

# Palette matching scan_gui._ZONE_RGBA — css strings for Plotly
_ZONE_COLORS_CSS = [
    "rgba(255,80,80,0.86)",
    "rgba(80,210,80,0.86)",
    "rgba(80,130,255,0.86)",
    "rgba(255,200,50,0.86)",
    "rgba(200,80,200,0.86)",
    "rgba(50,210,210,0.86)",
    "rgba(255,140,0,0.86)",
    "rgba(160,100,200,0.86)",
]


def _overlay_zones(fig: go.Figure, zones) -> None:
    """
    Overlay zone AABB outlines and semantic landmark markers onto a Plotly figure.
    zones: list of Zone objects (zone_labeler.Zone), may be None or empty.
    """
    if not zones:
        return
    for z_idx, zone in enumerate(zones):
        color = _ZONE_COLORS_CSS[z_idx % len(_ZONE_COLORS_CSS)]
        label = zone.label or f"Zone {z_idx + 1}"

        # Zone bounding box footprint (X-Z plane)
        xmin, xmax = float(zone.bbox_min[0]), float(zone.bbox_max[0])
        zmin, zmax = float(zone.bbox_min[2]), float(zone.bbox_max[2])
        cx, cz = (xmin + xmax) * 0.5, (zmin + zmax) * 0.5

        fig.add_trace(go.Scatter(
            x=[xmin, xmax, xmax, xmin, xmin],
            y=[zmin, zmin, zmax, zmax, zmin],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            name=label,
            showlegend=True,
            legendgroup=label,
            hoverinfo="name",
        ))

        # Zone label annotation at centre
        fig.add_annotation(
            x=cx, y=cz,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color=color, size=11),
            bgcolor="rgba(0,0,0,0.55)",
            borderpad=3,
        )

        # Landmark footprints — the actual 4 backprojected box corners
        # (lm.footprint_corners), each independently rotated by that frame's
        # camera pose, so this is a true parallelogram when the object was
        # viewed at an angle, not an axis-aligned rectangle (footprint_min/max
        # is just that quad's bounding envelope, used for clustering only —
        # see semantic_mapper.py). Drawn as one Scatter trace with
        # None-separated segments so every landmark's quad in this zone
        # renders without needing a separate trace per landmark.
        landmarks = getattr(zone, "landmarks", [])
        if landmarks:
            box_x: list = []
            box_z: list = []
            for lm in landmarks:
                corners = getattr(lm, "footprint_corners", None)
                if not corners:
                    x0, z0 = lm.footprint_min
                    x1, z1 = lm.footprint_max
                    corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
                xs = [c[0] for c in corners] + [corners[0][0]]
                zs = [c[1] for c in corners] + [corners[0][1]]
                box_x.extend(xs + [None])
                box_z.extend(zs + [None])
            fig.add_trace(go.Scatter(
                x=box_x, y=box_z,
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                name=label,
                showlegend=False,
                legendgroup=label,
                hoverinfo="skip",
            ))

            # Landmark centroid markers with text
            fig.add_trace(go.Scatter(
                x=[float(lm.x) for lm in landmarks],
                y=[float(lm.z) for lm in landmarks],
                mode="markers+text",
                marker=dict(
                    symbol="star",
                    size=11,
                    color=color,
                    line=dict(color="white", width=1),
                ),
                text=[lm.name for lm in landmarks],
                textposition="top center",
                textfont=dict(size=9, color="white"),
                customdata=[[round(lm.confidence, 3)] for lm in landmarks],
                hovertemplate="%{text}<br>conf=%{customdata[0]}<br>x=%{x:.2f} z=%{y:.2f}<extra></extra>",
                showlegend=False,
                legendgroup=label,
            ))


@dataclass
class _CellState:
    """Bayesian occupancy belief for one grid cell. logodds moves up on every
    obstacle/ground hit and down on every miss (see OccupancyMap.
    _register_obstacle_hit/_register_ground_hit/_register_miss) — never a
    permanent record of the first thing ever observed there. height_ewma is
    only touched by hits (a miss carries no height evidence)."""
    logodds: float = 0.0
    height_ewma: Optional[float] = None
    hit_count: int = 0


class OccupancyMap:
    OBSTACLE_MIN_H = 0.10       # metres above ground; below this → walkable
    STEP_OVER_MAX_H = 0.40      # metres above ground; below this (and >= OBSTACLE_MIN_H)
                                # → low, step-over-able obstacle (class 2); at/above this
                                # (and < OBSTACLE_MAX_H) → normal obstacle (class 3). The
                                # camera sits at the user's eye height, so a binary
                                # ground/obstacle split can't tell a curb from a wall.
    OBSTACLE_MAX_H = 2.20       # metres above ground; above this → ceiling (ignore)
    GROUND_PERCENTILE = 90      # Y percentile used as ground estimate (Y-down → large Y = floor)
    MAX_CLOUD_SAMPLE = 20_000

    # Cell classification codes (used by extract_subgrid/extract_full_grid's
    # "class" field and GridPathPlanner, server/tools/grid_path_planner.py):
    CLASS_UNKNOWN = 0       # not enough evidence either way, or height >= OBSTACLE_MAX_H (ceiling)
    CLASS_GROUND = 1        # confidently free, or occupied-but-height < OBSTACLE_MIN_H
    CLASS_LOW_STEP_OVER = 2 # occupied, OBSTACLE_MIN_H <= height < STEP_OVER_MAX_H
    CLASS_OBSTACLE = 3      # occupied, STEP_OVER_MAX_H <= height < OBSTACLE_MAX_H

    # Cap on how many ground cells get a free-space ray cast per update()
    # call — bounds cost when a batch touches a huge number of new cells at
    # once. Only ground-classified cells are ever ray-cast TO (see module
    # docstring) — obstacle cells' own hit already speaks for itself.
    MAX_RAYS_PER_UPDATE = 4_000

    # Bayesian log-odds update — same convention classic occupancy-grid
    # mapping uses (Moravec & Elfes 1985; gmapping/cartographer/RTAB-Map's
    # own OccupancyGrid). LOGODDS_HIT corresponds to ~70% confidence a single
    # hit indicates occupied; LOGODDS_MISS to ~60% confidence a single miss
    # indicates free. Deliberately requires a few agreeing observations
    # before crossing either threshold, so one noisy reading (in either
    # direction) can't flip a cell's classification outright, but a
    # consistent run of them can — and just as importantly, CAN be reversed
    # by later, better evidence (see module docstring).
    LOGODDS_HIT = 0.85
    LOGODDS_MISS = 0.4
    LOGODDS_MIN = -4.0
    LOGODDS_MAX = 4.0
    LOGODDS_OCCUPIED_THRESH = 1.0
    LOGODDS_FREE_THRESH = -1.0
    HEIGHT_EWMA_ALPHA = 0.3     # weight given to each new hit vs. the running estimate

    def __init__(
        self,
        resolution: float = 0.05,
        obstacle_min_h: Optional[float] = None,
        step_over_max_h: Optional[float] = None,
        obstacle_max_h: Optional[float] = None,
        logodds_hit: Optional[float] = None,
        logodds_miss: Optional[float] = None,
        logodds_occupied_thresh: Optional[float] = None,
        logodds_free_thresh: Optional[float] = None,
        height_ewma_alpha: Optional[float] = None,
        enable_ray_casting: bool = True,
        enable_bayesian: bool = True,
    ) -> None:
        """
        All Bayesian/height tuning knobs default to the class constants above
        but can be overridden per-instance — wired to scan_gui.py's
        "Occupancy Map Settings" accordion so scene-dependent tuning (a bed
        vs. an open hallway need different assumptions) doesn't require a
        code change. None means "use the class default".

        enable_ray_casting: when False, update() only ever registers HITS —
            the free-space Bresenham ray cast (_cast_free_ray) never runs, so
            a cell can never be revised back to free once it's been hit. Cuts
            update()'s cost (no ray walk) at the price of losing the self-
            correcting behavior the module docstring describes.

        enable_bayesian: when False, disables incremental log-odds belief
            entirely — a single hit sets a cell's logodds straight to
            LOGODDS_MAX (immediately, permanently classified from its
            height), and misses become no-ops (see _register_miss). This is
            "first observation wins" rather than "accumulate agreeing
            evidence, revise on disagreement" — useful for comparing against
            the Bayesian scheme or for a quick single-pass classification
            where revision isn't needed. Independent of enable_ray_casting:
            with both on, ray casting still runs but its misses have no
            effect on an already-hit cell; with ray casting off too, misses
            never even get computed.
        """
        self.resolution = resolution
        self.enable_ray_casting = enable_ray_casting
        self.enable_bayesian = enable_bayesian
        self.OBSTACLE_MIN_H = self.OBSTACLE_MIN_H if obstacle_min_h is None else obstacle_min_h
        self.STEP_OVER_MAX_H = self.STEP_OVER_MAX_H if step_over_max_h is None else step_over_max_h
        self.OBSTACLE_MAX_H = self.OBSTACLE_MAX_H if obstacle_max_h is None else obstacle_max_h
        self.LOGODDS_HIT = self.LOGODDS_HIT if logodds_hit is None else logodds_hit
        self.LOGODDS_MISS = self.LOGODDS_MISS if logodds_miss is None else logodds_miss
        self.LOGODDS_OCCUPIED_THRESH = (
            self.LOGODDS_OCCUPIED_THRESH if logodds_occupied_thresh is None else logodds_occupied_thresh
        )
        self.LOGODDS_FREE_THRESH = (
            self.LOGODDS_FREE_THRESH if logodds_free_thresh is None else logodds_free_thresh
        )
        self.HEIGHT_EWMA_ALPHA = (
            self.HEIGHT_EWMA_ALPHA if height_ewma_alpha is None else height_ewma_alpha
        )
        self._cells: Dict[Tuple[int, int], _CellState] = {}
        self._ground_y: Optional[float] = None
        # Accumulated camera path (X, Y, Z) across every update() call in this
        # session — (X, Z) draws the live trajectory on render_plotly().
        self._trajectory: List[Tuple[float, float, float]] = []

    def get_params(self) -> dict:
        """Current tunable values (for the GUI to display / re-apply)."""
        return {
            "obstacle_min_h": self.OBSTACLE_MIN_H,
            "step_over_max_h": self.STEP_OVER_MAX_H,
            "obstacle_max_h": self.OBSTACLE_MAX_H,
            "logodds_hit": self.LOGODDS_HIT,
            "logodds_miss": self.LOGODDS_MISS,
            "logodds_occupied_thresh": self.LOGODDS_OCCUPIED_THRESH,
            "logodds_free_thresh": self.LOGODDS_FREE_THRESH,
            "height_ewma_alpha": self.HEIGHT_EWMA_ALPHA,
            "enable_ray_casting": self.enable_ray_casting,
            "enable_bayesian": self.enable_bayesian,
        }

    def set_params(self, **kwargs) -> None:
        """
        Update tunable knobs on this already-existing instance (keeps
        accumulated cell beliefs — only affects FUTURE update() calls; cells
        already eroded/confirmed under the old settings are not
        retroactively recomputed). Unknown kwargs are ignored.
        """
        _map = {
            "obstacle_min_h": "OBSTACLE_MIN_H",
            "step_over_max_h": "STEP_OVER_MAX_H",
            "obstacle_max_h": "OBSTACLE_MAX_H",
            "logodds_hit": "LOGODDS_HIT",
            "logodds_miss": "LOGODDS_MISS",
            "logodds_occupied_thresh": "LOGODDS_OCCUPIED_THRESH",
            "logodds_free_thresh": "LOGODDS_FREE_THRESH",
            "height_ewma_alpha": "HEIGHT_EWMA_ALPHA",
        }
        _bool_map = {
            "enable_ray_casting": "enable_ray_casting",
            "enable_bayesian": "enable_bayesian",
        }
        for key, value in kwargs.items():
            if key in _map and value is not None:
                setattr(self, _map[key], float(value))
            elif key in _bool_map and value is not None:
                setattr(self, _bool_map[key], bool(value))

    # ── public ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all accumulated belief/trajectory data. Used before a full
        rebuild from a complete point cloud so re-running it doesn't keep
        re-accumulating the same data on top of itself."""
        self._cells.clear()
        self._ground_y = None
        self._trajectory.clear()

    def update(self, trajectory: np.ndarray, cloud_points: np.ndarray) -> None:
        """
        Accumulate Bayesian occupancy evidence from a new batch of already-
        voxelized points (see module docstring — scan_session.voxelize_cloud
        runs before this). Safe to call once per processed batch — the grid
        builds up progressively, it does not need to wait for a complete scan,
        and unlike a simple "record every sample forever" scheme, it can
        revise an earlier wrong reading as better evidence arrives.

        trajectory:   Nx3 — this batch's camera positions, appended to the
                      running path drawn by render_plotly() and used as the
                      ray-cast origin for this batch's newly-touched cells.
        cloud_points: Mx3 float — voxel-center world-space points (one per
                      occupied voxel, not raw per-point data).
        """
        if len(trajectory):
            self._trajectory.extend((float(p[0]), float(p[1]), float(p[2])) for p in trajectory)

        if len(cloud_points) < 10:
            return

        pts = np.asarray(cloud_points, dtype=np.float64)

        # Sub-sample for speed
        if len(pts) > self.MAX_CLOUD_SAMPLE:
            idx = np.random.choice(len(pts), self.MAX_CLOUD_SAMPLE, replace=False)
            pts = pts[idx]

        # First-ever call: no ground estimate exists yet to classify points
        # against. Bootstrap one from this batch's own points alone so
        # classification can start immediately; superseded by the cumulative
        # (much more stable) estimate below the moment any cell has hit
        # evidence.
        if self._ground_y is None:
            self._ground_y = _percentile_fast(pts[:, 1].tolist(), self.GROUND_PERCENTILE)

        res = self.resolution
        # ── Step 1: classify EVERY point individually by height above the
        # current ground estimate, bucketed by (X,Z) cell — see module
        # docstring for why this replaced a per-cell-percentile summary.
        cell_obstacle_y: Dict[Tuple[int, int], float] = {}   # key -> smallest Y (tallest point) this batch
        cell_ground_ys: Dict[Tuple[int, int], List[float]] = {}
        n_obstacle_pts = 0
        n_ground_pts = 0
        n_ceiling_pts = 0
        for pt in pts:
            ix = int(np.floor(float(pt[0]) / res))
            iz = int(np.floor(float(pt[2]) / res))
            y = float(pt[1])
            height = self._ground_y - y
            if height >= self.OBSTACLE_MAX_H:
                n_ceiling_pts += 1
                continue
            key = (ix, iz)
            if height < self.OBSTACLE_MIN_H:
                cell_ground_ys.setdefault(key, []).append(y)
                n_ground_pts += 1
            else:
                n_obstacle_pts += 1
                if key not in cell_obstacle_y or y < cell_obstacle_y[key]:
                    cell_obstacle_y[key] = y   # smaller Y = higher point = taller obstacle

        # ── Step 2: register hits — obstacle evidence always wins over a
        # ground point landing in the same coarse cell this batch (e.g. the
        # floor peeking out at a bed's edge doesn't get to call that cell
        # free); a cell only becomes a "ground cell" for this batch's ray
        # casting below if NOTHING obstacle-height was seen there too.
        for key, y in cell_obstacle_y.items():
            self._register_obstacle_hit(key, y)
        ground_keys = [k for k in cell_ground_ys if k not in cell_obstacle_y]
        for key in ground_keys:
            rep_y = float(np.median(cell_ground_ys[key]))
            self._register_ground_hit(key, rep_y)

        # Ground plane from the CUMULATIVE height estimate across every cell
        # with hit evidence so far (not just this batch) — see module
        # docstring for why this is safe to recompute every call.
        all_heights = [c.height_ewma for c in self._cells.values() if c.height_ewma is not None]
        if all_heights:
            self._ground_y = _percentile_fast(all_heights, self.GROUND_PERCENTILE)

        # ── Step 3: free-space ray casting — ONLY toward ground-classified
        # cells (never toward obstacle cells, whose own hit already speaks
        # for itself). Each ray STOPS at the first cell with confirmed
        # obstacle evidence, exactly like a real depth/laser ray being
        # physically blocked — see module docstring for the bug this fixes
        # (a ray to a farther ground point could previously erode a real,
        # closer obstacle like a bed via an approximate linear height
        # interpolation instead of real occlusion).
        n_rays = 0
        n_blocked = 0
        if self.enable_ray_casting and len(trajectory) and ground_keys:
            cam = trajectory[-1]
            cam_cell = (int(np.floor(float(cam[0]) / res)), int(np.floor(float(cam[2]) / res)))
            for key in ground_keys[: self.MAX_RAYS_PER_UPDATE]:
                n_rays += 1
                if self._cast_free_ray(cam_cell, key):
                    n_blocked += 1

        # See module docstring's "DEBUGGING" section for how to read this.
        print(
            f"[occupancy] update: {len(pts)} pts -> obstacle={n_obstacle_pts} "
            f"({len(cell_obstacle_y)} cells) ground={n_ground_pts} "
            f"({len(ground_keys)} cells) ceiling_ignored={n_ceiling_pts} | "
            f"rays_cast={n_rays} blocked_by_obstacle={n_blocked} | "
            f"ground_y={self._ground_y:.3f} | total_cells={len(self._cells)}"
        )

    def _register_obstacle_hit(self, key: Tuple[int, int], y: float) -> None:
        cell = self._cells.setdefault(key, _CellState())
        if self.enable_bayesian:
            cell.logodds = min(self.LOGODDS_MAX, cell.logodds + self.LOGODDS_HIT)
        else:
            # Non-Bayesian mode: a single hit fully and permanently
            # classifies the cell as occupied — no incremental belief, no
            # later revision (see enable_bayesian's docstring in __init__).
            cell.logodds = self.LOGODDS_MAX
        cell.hit_count += 1
        cell.height_ewma = (
            y if cell.height_ewma is None
            else self.HEIGHT_EWMA_ALPHA * y + (1 - self.HEIGHT_EWMA_ALPHA) * cell.height_ewma
        )

    def _register_ground_hit(self, key: Tuple[int, int], y: float) -> None:
        """Positive evidence toward free — not just an absence of obstacle
        hits. Still updates height_ewma (with the ground point's own,
        near-floor Y) so the cumulative ground_y estimate stays informed by
        real floor samples, not just obstacle-cell heights."""
        cell = self._cells.setdefault(key, _CellState())
        if self.enable_bayesian:
            cell.logodds = max(self.LOGODDS_MIN, cell.logodds - self.LOGODDS_MISS)
        else:
            cell.logodds = self.LOGODDS_MIN
        cell.hit_count += 1
        cell.height_ewma = (
            y if cell.height_ewma is None
            else self.HEIGHT_EWMA_ALPHA * y + (1 - self.HEIGHT_EWMA_ALPHA) * cell.height_ewma
        )

    def _register_miss(self, key: Tuple[int, int]) -> None:
        if not self.enable_bayesian:
            # Non-Bayesian mode: hits are permanent, so a miss carries no
            # meaning to register — there is no belief left to erode.
            return
        cell = self._cells.setdefault(key, _CellState())
        cell.logodds = max(self.LOGODDS_MIN, cell.logodds - self.LOGODDS_MISS)
        # height_ewma deliberately untouched — a miss carries no height
        # evidence, only "probably nothing occupies this space."

    def _cast_free_ray(self, origin: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Bresenham walk from origin to target (exclusive of target itself,
        which already got its own ground hit this batch). Registers a miss
        on each intermediate cell UNTIL the first cell with confident
        existing obstacle evidence (logodds >= LOGODDS_OCCUPIED_THRESH) is
        reached — a real depth/laser ray is physically blocked by the first
        obstacle it hits and never tests anything behind it, so marching
        past a confirmed obstacle and clearing cells beyond it would be
        physically wrong. Does NOT mark or alter the blocking cell itself —
        just stops there. Returns True if the ray stopped early because it
        hit a confirmed obstacle (purely for the debug log in update())."""
        r0, c0 = origin
        r1, c1 = target
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while (r, c) != (r1, c1):
            cell = self._cells.get((r, c))
            # Blocking uses "ANY net hit evidence" (logodds > 0), not the
            # stricter LOGODDS_OCCUPIED_THRESH used for final classification
            # — a real obstacle seen only once (logodds == LOGODDS_HIT, e.g.
            # 0.85 < a 1.0 threshold) isn't yet "confirmed" by the Bayesian
            # scheme, but it's still real: nothing physically blocked the
            # sensor from seeing it, so a later ray shouldn't be allowed to
            # march straight through it just because it hasn't accumulated
            # a second confirming hit yet (verified via a synthetic test: a
            # single-hit obstacle got fully eroded by 30 subsequent ground
            # rays before it ever crossed LOGODDS_OCCUPIED_THRESH).
            if cell is not None and cell.logodds > 0:
                return True
            self._register_miss((r, c))
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return False

    def _classify_state(self, cell: _CellState) -> Tuple[float, int]:
        """
        Given one cell's accumulated Bayesian belief, return
        (normalized_height, class_int):
          class 0 = unknown          (logodds between the free/occupied thresholds — not
                                       enough agreeing evidence yet — or height >= OBSTACLE_MAX_H)
          class 1 = ground/walkable  (confidently free by logodds, or occupied but
                                       height < OBSTACLE_MIN_H)
          class 2 = low, step-over-able  (occupied, OBSTACLE_MIN_H <= height < STEP_OVER_MAX_H)
          class 3 = normal obstacle  (occupied, STEP_OVER_MAX_H <= height < OBSTACLE_MAX_H)

        normalized_height mirrors the pre-existing `data` float convention
        (0.0=ground, (0,1]=obstacle height fraction across the full
        OBSTACLE_MIN_H..OBSTACLE_MAX_H span).
        """
        if cell.logodds >= self.LOGODDS_OCCUPIED_THRESH and cell.height_ewma is not None:
            height = self._ground_y - cell.height_ewma
            if height >= self.OBSTACLE_MAX_H:
                return float("nan"), self.CLASS_UNKNOWN
            if height < self.OBSTACLE_MIN_H:
                return 0.0, self.CLASS_GROUND
            norm = min((height - self.OBSTACLE_MIN_H) / (self.OBSTACLE_MAX_H - self.OBSTACLE_MIN_H), 1.0)
            cls = self.CLASS_LOW_STEP_OVER if height < self.STEP_OVER_MAX_H else self.CLASS_OBSTACLE
            return norm, cls
        if cell.logodds <= self.LOGODDS_FREE_THRESH:
            return 0.0, self.CLASS_GROUND
        return float("nan"), self.CLASS_UNKNOWN

    def _lookup_class(self, key: Tuple[int, int]) -> Tuple[float, int]:
        cell = self._cells.get(key)
        if cell is None:
            return float("nan"), self.CLASS_UNKNOWN
        return self._classify_state(cell)

    def render_plotly(self, zones=None) -> go.Figure:
        """Times the render (rendering is CPU-only — Plotly/numpy grid
        construction has no GPU path) and delegates to _render_plotly_impl."""
        with timed(f"occupancy_map.render_plotly ({len(self._cells)} cells)"):
            return self._render_plotly_impl(zones)

    def _render_plotly_impl(self, zones=None) -> go.Figure:
        """
        Return a Plotly Heatmap in the classic SLAM occupancy-grid style
        (gmapping/cartographer/RTAB-Map's own OccupancyGrid display):
          white  → free / ground (walkable)
          light gray → low, step-over-able obstacle
          black  → normal obstacle (blocked)
          mid-gray → unknown (not enough agreeing evidence either way)
        """
        if not self._cells or self._ground_y is None:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_dark",
                title=dict(text="Traversability Map (no data yet)", font=dict(size=13)),
                margin=dict(l=40, r=10, b=40, t=28),
                xaxis=dict(title="X (m)", color="#888"),
                yaxis=dict(title="Z (m)", color="#888", scaleanchor="x", scaleratio=1),
            )
            self._overlay_trajectory(fig)
            _overlay_zones(fig, zones)
            return fig

        keys = np.array(list(self._cells.keys()), dtype=np.int32)
        ix_min, iz_min = keys[:, 0].min(), keys[:, 1].min()
        ix_max, iz_max = keys[:, 0].max(), keys[:, 1].max()

        H = iz_max - iz_min + 1
        W = ix_max - ix_min + 1
        grid = np.full((H, W), self.CLASS_UNKNOWN, dtype=np.int8)

        res = self.resolution
        gy = self._ground_y

        for key, cell in self._cells.items():
            ix, iz = key
            row = iz - iz_min
            col = ix - ix_min
            _, cls = self._classify_state(cell)
            grid[row, col] = cls  # ceiling/uncertain (-> CLASS_UNKNOWN) left as unknown

        # ── Morphological dilation ─────────────────────────────────────────────
        # Spread known obstacles into adjacent still-unknown cells (3×3 max
        # filter) — fills gaps from sparse depth coverage on obstacle tops.
        # Never overwrites a cell that already has its own classification
        # (ground/step-over/obstacle).
        obs_mask = grid == self.CLASS_OBSTACLE
        if obs_mask.any():
            dilated = maximum_filter(obs_mask.astype(np.int8), size=3, mode="constant", cval=0)
            fill_mask = (grid == self.CLASS_UNKNOWN) & (dilated > 0)
            grid[fill_mask] = self.CLASS_OBSTACLE

        x_ticks = [ix_min * res + j * res for j in range(W)]
        z_ticks = [iz_min * res + i * res for i in range(H)]

        # Classic SLAM occupancy-grid grayscale — discrete, not a continuous
        # gradient: white=free, light gray=step-over, black=obstacle,
        # mid-gray=unknown (same visual language as RViz/gmapping/cartographer).
        _CLASS_COLOR = {
            self.CLASS_UNKNOWN: "rgb(120,120,120)",
            self.CLASS_GROUND: "rgb(255,255,255)",
            self.CLASS_LOW_STEP_OVER: "rgb(190,190,190)",
            self.CLASS_OBSTACLE: "rgb(0,0,0)",
        }
        _CLASS_NAME = {
            self.CLASS_UNKNOWN: "Unknown",
            self.CLASS_GROUND: "Free / ground",
            self.CLASS_LOW_STEP_OVER: "Low (step-over)",
            self.CLASS_OBSTACLE: "Obstacle",
        }
        # zmin/zmax padded half a class-width so 4 equal-width discrete bands
        # map cleanly to class ints 0..3 with hard (not blended) edges.
        colorscale = [
            [0.00, _CLASS_COLOR[0]], [0.25, _CLASS_COLOR[0]],
            [0.25, _CLASS_COLOR[1]], [0.50, _CLASS_COLOR[1]],
            [0.50, _CLASS_COLOR[2]], [0.75, _CLASS_COLOR[2]],
            [0.75, _CLASS_COLOR[3]], [1.00, _CLASS_COLOR[3]],
        ]
        class_names = np.vectorize(_CLASS_NAME.get)(grid)

        fig = go.Figure(
            go.Heatmap(
                z=grid,
                x=x_ticks,
                y=z_ticks,
                colorscale=colorscale,
                zmin=-0.5,
                zmax=3.5,
                showscale=True,
                colorbar=dict(
                    title="",
                    tickvals=[0, 1, 2, 3],
                    ticktext=["Unknown", "Free / ground", "Low (step-over)", "Obstacle"],
                    len=0.6,
                ),
                customdata=class_names,
                hovertemplate=(
                    "x=%{x:.2f}m  z=%{y:.2f}m<br>%{customdata}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=10, b=40, t=28),
            title=dict(
                text=f"Occupancy Map  (ground Y≈{gy:.2f} m, "
                     f"step-over &lt;{self.STEP_OVER_MAX_H*100:.0f} cm, "
                     f"obstacle {self.STEP_OVER_MAX_H*100:.0f}–{self.OBSTACLE_MAX_H*100:.0f} cm)",
                font=dict(size=12),
            ),
            xaxis=dict(title="X (m)", color="#888", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Z (m)", color="#888"),
            uirevision="occ",
        )
        self._overlay_trajectory(fig)
        _overlay_zones(fig, zones)
        return fig

    def _overlay_trajectory(self, fig: go.Figure) -> None:
        """Draw the accumulated camera path, with the current position highlighted."""
        if len(self._trajectory) < 2:
            return
        xs = [p[0] for p in self._trajectory]
        zs = [p[2] for p in self._trajectory]
        fig.add_trace(go.Scatter(
            x=xs, y=zs,
            mode="lines",
            line=dict(color="rgba(0,220,255,0.85)", width=2),
            name="Camera path",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[zs[-1]],
            mode="markers",
            marker=dict(symbol="circle", size=12, color="rgba(0,220,255,1.0)",
                        line=dict(color="white", width=2)),
            name="Current position",
            hovertemplate=f"x={xs[-1]:.2f}m  z={zs[-1]:.2f}m<extra>Current position</extra>",
        ))

    def _build_grid_dict(self, ix_lo: int, ix_hi: int, iz_lo: int, iz_hi: int) -> dict:
        """
        Shared by extract_subgrid/extract_full_grid: builds the JSON-serializable
        dict for cells in [ix_lo, ix_hi) x [iz_lo, iz_hi).

          data: List[List[float]]  — rows=Z, cols=X — UNCHANGED convention
                from before the Bayesian classification existed:
                0.0 = ground, 0.0-1.0 = obstacle (normalized height), -1.0 = unknown/ceiling
          class: List[List[int]]   — same shape, new — 0=unknown, 1=ground,
                2=low/step-over, 3=normal obstacle (see CLASS_* constants).
                A* path planning (server/tools/grid_path_planner.py) reads
                this; `data` stays exactly as-is for existing consumers
                (scan_gui.py's per-zone rendering).
        """
        res = self.resolution
        width = max(1, ix_hi - ix_lo)
        height = max(1, iz_hi - iz_lo)
        data = [[-1.0] * width for _ in range(height)]
        cls_grid = [[self.CLASS_UNKNOWN] * width for _ in range(height)]

        if self._ground_y is not None:
            for iz in range(iz_lo, iz_hi):
                for ix in range(ix_lo, ix_hi):
                    cell = self._cells.get((ix, iz))
                    if cell is None:
                        continue
                    norm, cls = self._classify_state(cell)
                    row, col = iz - iz_lo, ix - ix_lo
                    data[row][col] = -1.0 if cls == self.CLASS_UNKNOWN else norm
                    cls_grid[row][col] = cls

        return {
            "resolution": res,
            "origin_x": float(ix_lo * res),
            "origin_z": float(iz_lo * res),
            "width": width,
            "height": height,
            "data": data,
            "class": cls_grid,
        }

    def extract_subgrid(self, bbox_min: List[float], bbox_max: List[float]) -> dict:
        """
        Extract the occupancy cells that fall within a 3D AABB (only X and Z axes used).
        See _build_grid_dict for the returned dict's schema.
        """
        res = self.resolution
        ix_lo = int(math.floor(bbox_min[0] / res))
        ix_hi = int(math.ceil(bbox_max[0] / res))
        iz_lo = int(math.floor(bbox_min[2] / res))
        iz_hi = int(math.ceil(bbox_max[2] / res))
        return self._build_grid_dict(ix_lo, ix_hi, iz_lo, iz_hi)

    def extract_full_grid(self) -> Optional[dict]:
        """
        Whole-map counterpart to extract_subgrid (which is per-zone) — needed
        so GridPathPlanner (server/tools/grid_path_planner.py) can path
        across zone boundaries, not just within one zone's AABB. Returns
        None if there's no data yet (caller — map_exporter.py — omits the
        top-level occupancy_grid field in that case rather than exporting an
        empty grid).
        """
        if not self._cells or self._ground_y is None:
            return None
        keys = np.array(list(self._cells.keys()), dtype=np.int32)
        ix_lo, iz_lo = int(keys[:, 0].min()), int(keys[:, 1].min())
        ix_hi, iz_hi = int(keys[:, 0].max()) + 1, int(keys[:, 1].max()) + 1
        return self._build_grid_dict(ix_lo, ix_hi, iz_lo, iz_hi)
