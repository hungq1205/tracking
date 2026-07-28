"""
Height-based traversability map on the X-Z plane — a real Bayesian occupancy
grid (same representation gmapping/cartographer/RTAB-Map's own OccupancyGrid
module use), rendered as a continuous height-above-ground heatmap (0.0m
ground → OBSTACLE_MAX_H at the colorscale's top, unknown cells left blank),
built progressively as data arrives.

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
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import distance_transform_edt, maximum_filter

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

        # Landmark centroid markers with text — footprint quad outlines
        # (lm.footprint_corners) used to also be drawn here as a dotted box
        # per landmark; removed per direct user feedback (cluttered the
        # map). footprint_min/max/corners are still computed and kept on
        # the Landmark dataclass — still used for clustering (see
        # semantic_mapper.py) — just no longer rendered.
        landmarks = getattr(zone, "landmarks", [])
        if landmarks:
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
    OBSTACLE_MIN_H = 0.20       # metres above ground; below this → walkable
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
        # Guards every mutating/reading entry point below (update(),
        # reset(), seed_from_summary(), render_plotly(),
        # render_confidence_plotly(), extract_subgrid(), extract_full_grid(),
        # bounds(), extract_dirty_delta()) — see update()'s own docstring
        # for the race this fixes (gRPC servicer thread vs. Gradio polling
        # thread on a long-running WALKING/GUIDING session).
        self._lock = threading.Lock()
        self._cells: Dict[Tuple[int, int], _CellState] = {}
        # Cells touched (any _register_* call) since the last
        # extract_dirty_delta() call — the sparse counterpart to
        # extract_full_grid(), see that method's docstring for the full
        # incremental-sync design (mapping_servicer.py decides full vs.
        # delta per update).
        self._dirty_cells: set = set()
        self._ground_y: Optional[float] = None
        # Accumulated camera path (X, Y, Z) across every update() call in this
        # session — (X, Z) draws the live trajectory on render_plotly().
        self._trajectory: List[Tuple[float, float, float]] = []
        # Bumped once per update() call, regardless of what changed —
        # a cheap, always-correct "did the grid change" signal for a
        # caller deciding whether to re-run path planning against a live
        # (not-yet-exported) map. Deliberately not tied to cell/point
        # counts: a cell's classification can change (e.g. crossing the
        # confirm threshold) without any new grid cell being added.
        self._update_count: int = 0
        # Set at the top of each update() call, read by the _register_*
        # methods below to scale log-odds deltas — see update()'s docstring.
        self._current_confidence: float = 1.0

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
        with self._lock:
            self._cells.clear()
            self._dirty_cells.clear()
            self._ground_y = None
            self._trajectory.clear()
            self._update_count = 0

    def seed_from_summary(self, grid_dict: dict, ground_y: float) -> None:
        """Thread-safe entry point — see _seed_from_summary_locked() for the
        real body (unchanged below, just renamed); see update()'s own
        docstring for why this class now has a lock at all."""
        with self._lock:
            self._seed_from_summary_locked(grid_dict, ground_y)

    def _seed_from_summary_locked(self, grid_dict: dict, ground_y: float) -> None:
        """
        Coarse re-seed from a previously-EXPORTED summary (class + normalized
        height per cell — NOT the raw per-cell logodds/height_ewma, which
        isn't persisted anywhere; see MappingService/CLAUDE.md's "occupancy
        grid persistence across sessions" note for why this is a deliberate,
        accepted simplification rather than true incremental multi-day SLAM).

        Seeds each non-unknown cell with a modest, still-revisable belief —
        roughly "confirmed twice" (LOGODDS_OCCUPIED_THRESH/FREE_THRESH plus
        one hit/miss's worth) rather than LOGODDS_MAX/MIN, so real new
        evidence from this session can still move a cell either way, same as
        any other Bayesian update. CLASS_UNKNOWN cells are left with no
        entry at all, matching how genuinely-unobserved cells already work.
        Must be called before any update() call in the new session (reset()
        first if this map already has data — this does not clear anything
        itself).
        """
        if grid_dict is None:
            return
        if abs(grid_dict.get("resolution", -1.0) - self.resolution) > 1e-6:
            print(
                f"[OccupancyMap] seed_from_summary: resolution mismatch "
                f"(saved={grid_dict.get('resolution')}, current={self.resolution}) — skipping reseed."
            )
            return
        self._ground_y = ground_y
        res = self.resolution
        ix_lo = round(grid_dict["origin_x"] / res)
        iz_lo = round(grid_dict["origin_z"] / res)
        cls_grid = grid_dict["class"]
        height_grid = grid_dict["data"]
        seeded = 0
        for row in range(grid_dict["height"]):
            for col in range(grid_dict["width"]):
                cls = cls_grid[row][col]
                if cls == self.CLASS_UNKNOWN:
                    continue
                key = (ix_lo + col, iz_lo + row)
                if cls == self.CLASS_GROUND:
                    self._cells[key] = _CellState(
                        logodds=self.LOGODDS_FREE_THRESH - self.LOGODDS_MISS,
                        height_ewma=ground_y,
                        hit_count=2,
                    )
                else:
                    norm = height_grid[row][col]
                    height_above_ground = self.OBSTACLE_MIN_H + norm * (self.OBSTACLE_MAX_H - self.OBSTACLE_MIN_H)
                    self._cells[key] = _CellState(
                        logodds=self.LOGODDS_OCCUPIED_THRESH + self.LOGODDS_HIT,
                        height_ewma=ground_y - height_above_ground,
                        hit_count=2,
                    )
                seeded += 1
        print(f"[OccupancyMap] seed_from_summary: reseeded {seeded} cells from saved snapshot (ground_y={ground_y:.3f}).")

    def update(
        self, trajectory: np.ndarray, cloud_points: np.ndarray,
        confidence: float = 1.0,
        point_is_ground: Optional[np.ndarray] = None,
    ) -> None:
        """Thread-safe entry point — see _update_locked() for the real body
        (unchanged below, just renamed). Added after a real bug: server_gui.py's
        Gradio polling thread reads self._cells/self._trajectory (render_plotly/
        render_confidence_plotly/extract_full_grid/extract_subgrid/bounds/
        extract_dirty_delta) completely unsynchronized against this method's
        mutations, which run on the gRPC servicer thread on every processed
        WALKING/GUIDING batch — a bbox-expanding cell inserted mid-render
        could throw IndexError/RuntimeError inside the render, and only ONE
        of the two dashboard render call sites (_render_occupancy, not
        _annotate_mapping) had a try/except broad enough to survive that —
        the other could leave the Gradio polling loop stuck on a WALKING
        session that had run long enough to hit the race. self._lock (new)
        serializes every mutating/reading entry point on this class."""
        with self._lock:
            self._update_locked(trajectory, cloud_points, confidence, point_is_ground)

    def _update_locked(
        self, trajectory: np.ndarray, cloud_points: np.ndarray,
        confidence: float = 1.0,
        point_is_ground: Optional[np.ndarray] = None,
    ) -> None:
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
        confidence:   [0,1] — how much to trust THIS batch's depth (see
                      scan_session.py's depth-consistency check, e.g.
                      1 - frac_bad) — scales every log-odds delta registered
                      this call (see _register_obstacle_hit/_register_
                      ground_hit/_register_miss), so a batch built from
                      borderline depth moves belief more slowly than one
                      built from solid measurements, instead of every hit
                      counting the same regardless of source reliability.
                      Standard Bayesian evidence weighting, not a hard cap:
                      enough agreeing low-confidence observations can still
                      accumulate to high confidence over time (correct if
                      the underlying errors are independent noise) — it does
                      NOT protect against DA3 systematically misjudging the
                      exact same real surface the same way every time, which
                      is a correlated bias, not independent noise. Only
                      applies in Bayesian mode (enable_bayesian) — non-
                      Bayesian mode's "single hit = permanent" design has no
                      incremental belief to scale.
        point_is_ground: optional length-M bool, aligned with cloud_points —
                      when given, RTAB-Map's OWN ground/obstacle segmentation
                      (util3d::segmentObstaclesFromGround, see
                      rtabmap_server.cc's segment_ground_flags()) decides
                      ground-vs-obstacle for each point INSTEAD of this
                      module's own height-vs-OBSTACLE_MIN_H heuristic. Height
                      is still computed and still used for the OBSTACLE_MAX_H
                      ceiling filter (RTAB-Map's segmentation has no "too
                      high to matter" concept) and for the LOW_STEP_OVER vs.
                      normal-OBSTACLE tiering downstream in _classify_state()
                      — this only replaces the ground/not-ground BOUNDARY
                      decision, not height tracking itself. None (default)
                      keeps the original pure-height behavior — IMU+VO/VO
                      pose sources have no such per-point flag to offer.
                      Misaligned length falls back to height-only rather
                      than raising or silently misapplying flags to the
                      wrong points.
        """
        if len(trajectory):
            self._trajectory.extend((float(p[0]), float(p[1]), float(p[2])) for p in trajectory)

        if len(cloud_points) < 10:
            return
        self._current_confidence = max(0.0, min(1.0, confidence))
        self._update_count += 1

        pts = np.asarray(cloud_points, dtype=np.float64)

        ground_flags = None
        if point_is_ground is not None:
            ground_flags = np.asarray(point_is_ground, dtype=bool)
            if len(ground_flags) != len(pts):
                ground_flags = None

        # Sub-sample for speed
        if len(pts) > self.MAX_CLOUD_SAMPLE:
            idx = np.random.choice(len(pts), self.MAX_CLOUD_SAMPLE, replace=False)
            pts = pts[idx]
            if ground_flags is not None:
                ground_flags = ground_flags[idx]

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
        for i, pt in enumerate(pts):
            ix = int(np.floor(float(pt[0]) / res))
            iz = int(np.floor(float(pt[2]) / res))
            y = float(pt[1])
            height = self._ground_y - y
            if height >= self.OBSTACLE_MAX_H:
                n_ceiling_pts += 1
                continue
            key = (ix, iz)
            rtabmap_confirmed_ground = ground_flags is not None and bool(ground_flags[i])
            is_ground_pt = rtabmap_confirmed_ground if ground_flags is not None else (height < self.OBSTACLE_MIN_H)
            if is_ground_pt:
                # RTAB-Map's own 3D ground segmentation (util3d::
                # segmentObstaclesFromGround) is a genuinely independent
                # signal from this point's own (depth-noise-affected)
                # measured Y — when it confirms a point IS ground, force
                # its recorded height to exactly ground level
                # (self._ground_y, i.e. height=0) instead of letting that
                # point's own measurement noise into height_ewma/the
                # cumulative ground_y re-estimate below. Requested directly
                # by the user. Height-HEURISTIC-classified ground
                # (ground_flags is None — IMU+VO/VO pose mode, no
                # independent segmentation available) keeps its own real
                # measured Y unchanged — there's no independent
                # confirmation to force toward there, only the height
                # heuristic itself (which the ground_y estimate is already
                # derived from), so forcing would just be circular.
                ground_y_for_cell = self._ground_y if rtabmap_confirmed_ground else y
                cell_ground_ys.setdefault(key, []).append(ground_y_for_cell)
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
        self._dirty_cells.add(key)
        cell = self._cells.setdefault(key, _CellState())
        if self.enable_bayesian:
            cell.logodds = min(
                self.LOGODDS_MAX, cell.logodds + self.LOGODDS_HIT * self._current_confidence
            )
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
        self._dirty_cells.add(key)
        cell = self._cells.setdefault(key, _CellState())
        if self.enable_bayesian:
            cell.logodds = max(
                self.LOGODDS_MIN, cell.logodds - self.LOGODDS_MISS * self._current_confidence
            )
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
        self._dirty_cells.add(key)
        cell = self._cells.setdefault(key, _CellState())
        cell.logodds = max(
            self.LOGODDS_MIN, cell.logodds - self.LOGODDS_MISS * self._current_confidence
        )
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

    def render_plotly(
        self, zones=None,
        route: Optional[List[Tuple[float, float]]] = None,
        route_confirmed: Optional[bool] = None,
    ) -> go.Figure:
        """Times the render (rendering is CPU-only — Plotly/numpy grid
        construction has no GPU path) and delegates to _render_plotly_impl.
        `route`/`route_confirmed` — see _overlay_route(). Holds self._lock
        for the whole render (see update()'s docstring for why) — a
        WALKING/GUIDING session's own update() calls only ever briefly wait
        behind a render, never the reverse deadlock risk, since update()
        never itself calls back into a render method."""
        with self._lock, timed(f"occupancy_map.render_plotly ({len(self._cells)} cells)"):
            return self._render_plotly_impl(zones, route, route_confirmed)

    def _empty_figure(
        self, title: str, uirevision: str,
        route: Optional[List[Tuple[float, float]]] = None,
        route_confirmed: Optional[bool] = None,
    ) -> go.Figure:
        """Shared 'no data yet' placeholder for both render_plotly() and
        render_confidence_plotly()."""
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title=dict(text=title, font=dict(size=13)),
            margin=dict(l=40, r=10, b=40, t=28),
            xaxis=dict(title="X (m)", color="#888"),
            yaxis=dict(title="Z (m)", color="#888", scaleanchor="x", scaleratio=1),
            uirevision=uirevision,
        )
        self._overlay_trajectory(fig)
        self._overlay_route(fig, route, route_confirmed)
        return fig

    def _grid_bbox_and_ticks(self) -> Tuple[int, int, int, int, int, int, list, list]:
        """Shared by render_plotly()/render_confidence_plotly(): the bounding
        box of every cell ever touched, its (H, W) shape, and world-space
        tick arrays for the X/Z axes — both renders iterate the exact same
        self._cells keys into the exact same grid shape, just filling it
        with a different per-cell value."""
        keys = np.array(list(self._cells.keys()), dtype=np.int32)
        ix_min, iz_min = int(keys[:, 0].min()), int(keys[:, 1].min())
        ix_max, iz_max = int(keys[:, 0].max()), int(keys[:, 1].max())
        H = iz_max - iz_min + 1
        W = ix_max - ix_min + 1
        res = self.resolution
        x_ticks = [ix_min * res + j * res for j in range(W)]
        z_ticks = [iz_min * res + i * res for i in range(H)]
        return ix_min, iz_min, ix_max, iz_max, H, W, x_ticks, z_ticks

    _CLASS_NAME = {
        CLASS_UNKNOWN: "Unknown",
        CLASS_GROUND: "Free / ground",
        CLASS_LOW_STEP_OVER: "Low (step-over)",
        CLASS_OBSTACLE: "Obstacle",
    }

    def _render_plotly_impl(
        self, zones=None,
        route: Optional[List[Tuple[float, float]]] = None,
        route_confirmed: Optional[bool] = None,
    ) -> go.Figure:
        """
        Return a Plotly Heatmap as a continuous height-above-ground gradient
        (an elevation/height map, not the classic SLAM discrete
        free/step-over/obstacle grayscale this used to render) — 0.0m
        (ground) at the colorscale's low end, OBSTACLE_MAX_H at the high
        end, so a glance shows how TALL an obstacle is (a low curb vs. a
        chest-height shelf vs. a full wall), not just that a cell is
        occupied. Unknown cells (not enough agreeing evidence, or the
        module's own CLASS_UNKNOWN — see _classify_state) render as NaN,
        which Plotly leaves blank/transparent — visually distinct from any
        real height value without needing a dedicated color.

        Reuses _classify_state() as the single source of truth for
        ground/step-over/obstacle/unknown (same Bayesian belief the
        Occupancy Map's own path-planning export uses) and de-normalizes
        its `norm` output back to a real metres value for non-ground cells,
        rather than duplicating the classification thresholds here.

        `route`/`route_confirmed` — see _overlay_route().
        """
        if not self._cells or self._ground_y is None:
            return self._empty_figure(
                "Traversability Map (no data yet)", uirevision="occ",
                route=route, route_confirmed=route_confirmed,
            )

        ix_min, iz_min, ix_max, iz_max, H, W, x_ticks, z_ticks = self._grid_bbox_and_ticks()
        height_grid = np.full((H, W), np.nan, dtype=np.float64)
        class_grid = np.full((H, W), self.CLASS_UNKNOWN, dtype=np.int8)

        gy = self._ground_y

        for key, cell in self._cells.items():
            ix, iz = key
            row = iz - iz_min
            col = ix - ix_min
            norm, cls = self._classify_state(cell)
            class_grid[row, col] = cls
            if cls == self.CLASS_UNKNOWN:
                continue  # not enough evidence, or ceiling — left as NaN
            if cls == self.CLASS_GROUND:
                # Real height (0..OBSTACLE_MIN_H), not flattened to a flat
                # 0.0 for every ground cell regardless of its actual height
                # — _classify_state() itself still reports a fixed 0.0 for
                # GROUND (its own [0,1] normalized-obstacle-span contract is
                # relied on elsewhere — path planning, the gRPC height_norm
                # field — so left untouched), this display-only computation
                # recovers the real per-cell height directly from the same
                # height_ewma/ground_y the classification itself used. A
                # cell classified GROUND via the free-logodds path (never
                # actually height-measured) has no height_ewma — 0.0 for
                # that case is the honest answer, not a missing one.
                ground_height = (
                    (gy - cell.height_ewma) if cell.height_ewma is not None else 0.0
                )
                height_grid[row, col] = max(0.0, min(ground_height, self.OBSTACLE_MIN_H))
            else:
                # De-normalize _classify_state's (height-OBSTACLE_MIN_H)/
                # (OBSTACLE_MAX_H-OBSTACLE_MIN_H) fraction back to real
                # metres — reuses that method as the single source of truth
                # for the classification thresholds instead of duplicating
                # them here.
                height_grid[row, col] = (
                    norm * (self.OBSTACLE_MAX_H - self.OBSTACLE_MIN_H) + self.OBSTACLE_MIN_H
                )

        # ── Gap fill ─────────────────────────────────────────────────────────
        # Spread known height values into adjacent still-unknown (NaN) cells
        # (3×3 max filter) — fills gaps from sparse depth coverage on
        # obstacle tops, generalizing the same idea the old discrete
        # rendering used (dilating obstacle presence) to continuous height:
        # an unknown cell next to a tall neighbor is more likely to be part
        # of that same obstacle's top than to be ground. Never overwrites a
        # cell that already has its own classification.
        valid_mask = ~np.isnan(height_grid)
        if valid_mask.any():
            filled_for_filter = np.where(valid_mask, height_grid, -np.inf)
            dilated = maximum_filter(filled_for_filter, size=3, mode="constant", cval=-np.inf)
            fill_mask = (~valid_mask) & np.isfinite(dilated)
            height_grid[fill_mask] = dilated[fill_mask]

        class_names = np.vectorize(self._CLASS_NAME.get)(class_grid)

        fig = go.Figure(
            go.Heatmap(
                z=height_grid,
                x=x_ticks,
                y=z_ticks,
                colorscale="Turbo",
                zmin=0.0,
                zmax=self.OBSTACLE_MAX_H,
                showscale=True,
                colorbar=dict(title="Height (m)", len=0.6),
                customdata=class_names,
                hovertemplate=(
                    "x=%{x:.2f}m  z=%{y:.2f}m<br>height=%{z:.2f}m (%{customdata})<extra></extra>"
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
        self._overlay_route(fig, route, route_confirmed)
        _overlay_zones(fig, zones)
        return fig

    def render_confidence_plotly(
        self, zones=None,
        route: Optional[List[Tuple[float, float]]] = None,
        route_confirmed: Optional[bool] = None,
    ) -> go.Figure:
        """Times the render, same convention as render_plotly() — including
        holding self._lock for the whole call, see that method's own
        comment."""
        with self._lock, timed(f"occupancy_map.render_confidence_plotly ({len(self._cells)} cells)"):
            return self._render_confidence_plotly_impl(zones, route, route_confirmed)

    def _render_confidence_plotly_impl(
        self, zones=None,
        route: Optional[List[Tuple[float, float]]] = None,
        route_confirmed: Optional[bool] = None,
    ) -> go.Figure:
        """
        Return a Plotly Heatmap of per-cell CONFIDENCE — how much agreeing
        evidence a cell has accumulated, independent of whether that
        evidence says free or occupied — as opposed to render_plotly()'s
        height/classification view. A cell with logodds near 0 (barely
        observed, or genuinely contradictory observations cancelling out)
        reads low confidence even if its current best-guess classification
        happens to be ground; a cell hit/confirmed many times over reads
        high confidence regardless of which way it was classified.

        confidence = min(|logodds| / max(LOGODDS_MAX, |LOGODDS_MIN|), 1.0)
        — 0.0 at logodds==0 (the exact center of _classify_state's
        "unknown" band), ramping to 1.0 at full log-odds saturation. Cells
        with NO entry in self._cells at all (never observed, still inside
        the overall bbox) get an explicit 0.0 — unlike render_plotly()'s
        height map, "no data" and "confidence zero" are the same concept
        here, so there's no NaN/blank case to represent separately.

        `route`/`route_confirmed` — see _overlay_route().
        """
        if not self._cells or self._ground_y is None:
            return self._empty_figure(
                "Confidence Map (no data yet)", uirevision="conf",
                route=route, route_confirmed=route_confirmed,
            )

        ix_min, iz_min, ix_max, iz_max, H, W, x_ticks, z_ticks = self._grid_bbox_and_ticks()
        confidence_grid = np.zeros((H, W), dtype=np.float64)
        class_grid = np.full((H, W), self.CLASS_UNKNOWN, dtype=np.int8)

        logodds_scale = max(self.LOGODDS_MAX, abs(self.LOGODDS_MIN))
        for key, cell in self._cells.items():
            ix, iz = key
            row = iz - iz_min
            col = ix - ix_min
            _, cls = self._classify_state(cell)
            class_grid[row, col] = cls
            confidence_grid[row, col] = min(abs(cell.logodds) / logodds_scale, 1.0)

        class_names = np.vectorize(self._CLASS_NAME.get)(class_grid)

        fig = go.Figure(
            go.Heatmap(
                z=confidence_grid,
                x=x_ticks,
                y=z_ticks,
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                showscale=True,
                colorbar=dict(title="Confidence", len=0.6),
                customdata=class_names,
                hovertemplate=(
                    "x=%{x:.2f}m  z=%{y:.2f}m<br>confidence=%{z:.2f} (%{customdata})<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=10, b=40, t=28),
            title=dict(text="Confidence Map", font=dict(size=12)),
            xaxis=dict(title="X (m)", color="#888", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Z (m)", color="#888"),
            uirevision="conf",
        )
        self._overlay_trajectory(fig)
        self._overlay_route(fig, route, route_confirmed)
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

    def _overlay_route(
        self, fig: go.Figure,
        route: Optional[List[Tuple[float, float]]],
        confirmed: Optional[bool],
    ) -> None:
        """Draw a computed navigation route (see live_path_planner.py) —
        green if `confirmed` (every cell along it is genuinely observed
        ground/step-over), dashed orange if not (it had to cross at least
        one still-unexplored cell to reach the destination — "speculative",
        see live_path_planner.py's module docstring). `route` must include
        the start point as its first element: LiveGridPathPlanner.find_path()
        itself omits it (matching GridPathPlanner's existing convention),
        so the caller prepends the current camera position before passing
        a route here."""
        if not route or len(route) < 2:
            return
        xs = [p[0] for p in route]
        zs = [p[1] for p in route]
        color = "rgba(50,220,50,0.9)" if confirmed else "rgba(255,150,0,0.9)"
        fig.add_trace(go.Scatter(
            x=xs, y=zs,
            mode="lines+markers",
            line=dict(color=color, width=3, dash=None if confirmed else "dash"),
            marker=dict(size=6, color=color),
            name="Route" if confirmed else "Route (speculative — unexplored)",
            hoverinfo="skip",
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
          clearance: List[List[float]] — same shape, new — metres to the
                nearest CLASS_OBSTACLE cell (0.0 AT an obstacle cell itself),
                via a 2D Euclidean distance transform over this window's own
                class grid. Used by server/tools/grid_path_planner.py to
                prefer open space over hugging walls, not just avoid hard
                blocks. NOTE (known caveat, not fixed): this is windowed to
                [ix_lo,ix_hi)x[iz_lo,iz_hi) — an obstacle just outside the
                window is invisible to this EDT, so extract_subgrid()'s
                per-zone clearance can overstate real safety near a zone's
                AABB edge. Harmless today because grid_path_planner.py only
                ever reads the top-level grid from extract_full_grid(),
                which spans every occupied cell in one shot (no windowing
                artifact) — don't trust zones[].occupancy_grid.clearance for
                planning without revisiting this windowing.
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

        obstacle_mask = np.array(cls_grid) == self.CLASS_OBSTACLE
        if obstacle_mask.any():
            clearance = (distance_transform_edt(~obstacle_mask) * res).tolist()
        else:
            # distance_transform_edt on an all-False mask does NOT raise —
            # it returns nonsense values anchored to the array's (0,0)
            # corner (as if that corner cell were an implicit obstacle),
            # empirically verified during planning. 9.0m is a safe inert
            # sentinel: grid_path_planner.py's clearance-cost formula
            # saturates to ~1.0x (no effect) well before 9.0m anyway.
            clearance = [[9.0] * width for _ in range(height)]

        return {
            "resolution": res,
            "origin_x": float(ix_lo * res),
            "origin_z": float(iz_lo * res),
            "width": width,
            "height": height,
            "data": data,
            "class": cls_grid,
            "clearance": clearance,
        }

    def extract_subgrid(self, bbox_min: List[float], bbox_max: List[float]) -> dict:
        """
        Extract the occupancy cells that fall within a 3D AABB (only X and Z axes used).
        See _build_grid_dict for the returned dict's schema. Holds self._lock
        for the whole call — see update()'s docstring for why this class has
        a lock at all.
        """
        with self._lock:
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
        empty grid). Holds self._lock for the whole call.
        """
        with self._lock:
            if not self._cells or self._ground_y is None:
                return None
            keys = np.array(list(self._cells.keys()), dtype=np.int32)
            ix_lo, iz_lo = int(keys[:, 0].min()), int(keys[:, 1].min())
            ix_hi, iz_hi = int(keys[:, 0].max()) + 1, int(keys[:, 1].max()) + 1
            return self._build_grid_dict(ix_lo, ix_hi, iz_lo, iz_hi)

    def clear_dirty(self) -> None:
        """Discards pending dirty cells without extracting them — used after
        a FULL grid export (mapping_servicer.py), since those cells' current
        values are already covered by the full send and would otherwise be
        redundantly included in the next delta too."""
        with self._lock:
            self._dirty_cells.clear()

    def bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """(ix_lo, iz_lo, width, height) of the current full-grid bounding
        box, without paying for a full _build_grid_dict() classification
        pass — mapping_servicer.py calls this cheaply on every update to
        decide whether the box grew since the last full resync (in which
        case a delta's fixed-origin cell indices would no longer line up
        with what the client has, and a fresh extract_full_grid() is
        needed instead of extract_dirty_delta()). Thread-safe entry point —
        see _bounds_locked() for the real body (also called internally by
        extract_dirty_delta(), which already holds self._lock itself and so
        must call _bounds_locked() directly rather than re-entering this
        method — self._lock is a plain, non-reentrant Lock)."""
        with self._lock:
            return self._bounds_locked()

    def _bounds_locked(self) -> Optional[Tuple[int, int, int, int]]:
        if not self._cells:
            return None
        keys = np.array(list(self._cells.keys()), dtype=np.int32)
        ix_lo, iz_lo = int(keys[:, 0].min()), int(keys[:, 1].min())
        ix_hi, iz_hi = int(keys[:, 0].max()) + 1, int(keys[:, 1].max()) + 1
        return ix_lo, iz_lo, ix_hi - ix_lo, iz_hi - iz_lo

    def extract_dirty_delta(self) -> Optional[dict]:
        """Thread-safe entry point — see _extract_dirty_delta_locked() for
        the real body (unchanged below, just renamed)."""
        with self._lock:
            return self._extract_dirty_delta_locked()

    def _extract_dirty_delta_locked(self) -> Optional[dict]:
        """
        Sparse counterpart to extract_full_grid() — only cells touched
        (self._dirty_cells) since the last call, for incremental sync
        instead of re-shipping the whole map every update. Clears
        self._dirty_cells before returning. Cell indices are GLOBAL grid
        coordinates (ix, iz), not relative to any particular window, so
        they stay valid across calls as long as the map's overall bounding
        box hasn't changed since the last full resync (see bounds() above)
        — the caller is responsible for that decision, this method doesn't
        make it.

        Reuses _build_grid_dict's classification + clearance (EDT) pass
        over the CURRENT full bounding box to get authoritative values for
        the dirty subset — same server-side cost as extract_full_grid()
        today (this does not reduce compute, only what's put on the wire,
        which is what was actually asked for: repeatedly shipping the
        whole grid over gRPC as a session/map grows).

        Known, accepted imprecision: a cell whose CLEARANCE changed because
        a NEARBY cell (not itself) just became/stopped being an obstacle,
        without itself being touched this batch, can go briefly stale until
        it's next touched itself. Not fixed, because it's rare and
        low-impact: the clearance cost curve saturates to ~1.0x (no
        practical path-cost effect) by ~1m from any obstacle (see
        grid_path_planner.py's CLEARANCE_DECAY_RATE), and cells near a
        just-touched obstacle are overwhelmingly likely to be touched in
        the very same batch anyway (same source depth frame), so this
        would rarely bite in practice.

        Returns None if nothing is dirty (caller sends a pose-only update).
        """
        if not self._dirty_cells or self._ground_y is None:
            return None
        b = self._bounds_locked()
        if b is None:
            return None
        ix_lo, iz_lo, width, height = b
        full = self._build_grid_dict(ix_lo, ix_lo + width, iz_lo, iz_lo + height)
        cells = []
        for (ix, iz) in self._dirty_cells:
            row, col = iz - iz_lo, ix - ix_lo
            if not (0 <= row < height and 0 <= col < width):
                continue  # defensive — shouldn't happen, bounds() was just computed fresh
            cells.append({
                "ix": ix, "iz": iz,
                "class": full["class"][row][col],
                "height_norm": full["data"][row][col],
                "clearance": full["clearance"][row][col],
            })
        self._dirty_cells.clear()
        if not cells:
            return None
        return {"resolution": self.resolution, "cells": cells}
