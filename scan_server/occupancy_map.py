"""
Height-based traversability map on the X-Z plane.

Per-cell we accumulate a sample list of Y values (capped at MAX_SAMPLES_PER_CELL).
After estimating the ground plane (90th-percentile Y across all points, since Y points
DOWN in camera/world space), every cell is classified by the height of its highest
observed point above the ground:

  height < OBSTACLE_MIN_H   →  ground   (walkable, green)
  OBSTACLE_MIN_H ≤ h < MAX  →  obstacle (blocked, yellow → red by height)
  h ≥ OBSTACLE_MAX_H        →  ceiling  (ignored, not drawn)
  no data                   →  unknown  (white / transparent)

Height is estimated as  ground_y − p05(cell_Y_values)  where p05 is the 5th
percentile.  Using the strict minimum would be overly sensitive to a single noisy
point sitting above the true obstacle top; the 5th percentile is robust to ~5 %
outliers while still finding the highest real surface in a cell.

After building the raw height grid we apply a 3×3 morphological maximum filter
(dilation) so that sparse depth coverage on obstacle tops bleeds into neighbouring
cells that the depth sensor may have missed.

World-space coordinate convention (OpenCV camera, identity first frame):
  X – right,  Y – DOWN,  Z – forward
So ground_y = high percentile of Y values (large Y = near the floor).
Height above ground = ground_y − point_y  (positive = further from ground).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import maximum_filter


class OccupancyMap:
    OBSTACLE_MIN_H = 0.10       # metres above ground; below this → walkable
    OBSTACLE_MAX_H = 2.20       # metres above ground; above this → ceiling (ignore)
    GROUND_PERCENTILE = 90      # Y percentile used as ground estimate (Y-down → large Y = floor)
    HEIGHT_PERCENTILE = 5       # robust "highest point" estimator per cell (5th pct of Y)
    MAX_CLOUD_SAMPLE = 20_000
    MAX_SAMPLES_PER_CELL = 40   # cap per-cell sample list to bound memory

    def __init__(self, resolution: float = 0.05) -> None:
        self.resolution = resolution
        # Sparse dict: (ix, iz) → list of sampled Y values (capped at MAX_SAMPLES_PER_CELL)
        self._cells: Dict[Tuple[int, int], List[float]] = {}
        self._ground_y: Optional[float] = None

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, trajectory: np.ndarray, cloud_points: np.ndarray) -> None:
        """
        Accumulate height evidence from a new batch of point-cloud points.

        trajectory:   Nx3 (not used for height map, kept for API compatibility)
        cloud_points: Mx3 float — world-space points from the point cloud
        """
        if len(cloud_points) < 10:
            return

        # Sub-sample for speed
        if len(cloud_points) > self.MAX_CLOUD_SAMPLE:
            idx = np.random.choice(len(cloud_points), self.MAX_CLOUD_SAMPLE, replace=False)
            pts = cloud_points[idx]
        else:
            pts = cloud_points.copy()

        # Ground plane: 90th-percentile Y (Y is down → floor = largest Y values)
        self._ground_y = float(np.percentile(pts[:, 1], self.GROUND_PERCENTILE))

        res = self.resolution
        cap = self.MAX_SAMPLES_PER_CELL
        for pt in pts:
            ix = int(np.floor(float(pt[0]) / res))
            iz = int(np.floor(float(pt[2]) / res))
            y = float(pt[1])
            key = (ix, iz)
            if key not in self._cells:
                self._cells[key] = [y]
            elif len(self._cells[key]) < cap:
                self._cells[key].append(y)
            else:
                # Reservoir replacement: randomly replace an existing sample
                slot = int(np.random.randint(0, cap))
                self._cells[key][slot] = y

    def render_plotly(self) -> go.Figure:
        """
        Return a Plotly Heatmap where each cell is coloured by the height
        of its tallest point above the ground plane.

        Colorscale:
          NaN          → transparent / white  (unknown)
          0.0          → green                (ground, walkable)
          0.0 → 1.0    → yellow → red         (obstacle, normalised height)
        """
        if not self._cells or self._ground_y is None:
            fig = go.Figure()
            fig.update_layout(
                template="plotly_dark",
                title=dict(text="Traversability Map (no data yet)", font=dict(size=13)),
                margin=dict(l=40, r=10, b=40, t=28),
            )
            return fig

        keys = np.array(list(self._cells.keys()), dtype=np.int32)
        ix_min, iz_min = keys[:, 0].min(), keys[:, 1].min()
        ix_max, iz_max = keys[:, 0].max(), keys[:, 1].max()

        H = iz_max - iz_min + 1
        W = ix_max - ix_min + 1
        grid = np.full((H, W), np.nan, dtype=np.float32)

        res = self.resolution
        gy = self._ground_y

        for (ix, iz), y_samples in self._cells.items():
            row = iz - iz_min
            col = ix - ix_min
            # 5th percentile of Y → closest to the highest point (Y-down: smaller Y = higher)
            top_y = float(np.percentile(y_samples, self.HEIGHT_PERCENTILE))
            height = gy - top_y   # height above ground; positive = obstacle
            if height >= self.OBSTACLE_MAX_H:
                # Ceiling / sky — treat as unknown (don't block path)
                continue
            elif height < self.OBSTACLE_MIN_H:
                grid[row, col] = 0.0   # ground → walkable
            else:
                grid[row, col] = min(
                    (height - self.OBSTACLE_MIN_H) / (self.OBSTACLE_MAX_H - self.OBSTACLE_MIN_H),
                    1.0,
                )

        # ── Morphological dilation ─────────────────────────────────────────────
        # Spread known obstacle heights into adjacent unknown cells (3×3 max filter).
        # This fills gaps caused by sparse depth coverage on the tops of obstacles.
        obs_mask = np.isfinite(grid) & (grid > 0.0)
        if obs_mask.any():
            filled = maximum_filter(
                np.where(obs_mask, grid, 0.0), size=3, mode="constant", cval=0.0
            )
            # Only fill NaN neighbours; do not overwrite existing readings.
            fill_mask = np.isnan(grid) & (filled > 0.0)
            grid[fill_mask] = filled[fill_mask]

        x_ticks = [ix_min * res + j * res for j in range(W)]
        z_ticks = [iz_min * res + i * res for i in range(H)]

        colorscale = [
            [0.00, "rgb(46,204,113)"],   # 0.0 → green  (ground / walkable)
            [0.05, "rgb(255,235,59)"],   # 0.05 → yellow (very low obstacle)
            [0.40, "rgb(255,152,0)"],    # 0.4  → orange
            [0.70, "rgb(244,67,54)"],    # 0.7  → red
            [1.00, "rgb(136,14,79)"],    # 1.0  → dark magenta (tall wall)
        ]

        fig = go.Figure(
            go.Heatmap(
                z=grid,
                x=x_ticks,
                y=z_ticks,
                colorscale=colorscale,
                zmin=0.0,
                zmax=1.0,
                showscale=True,
                colorbar=dict(
                    title="Height (m)",
                    tickvals=[0.0, 0.5, 1.0],
                    ticktext=[
                        f"Ground (≤{self.OBSTACLE_MIN_H*100:.0f} cm)",
                        f"{(self.OBSTACLE_MIN_H + self.OBSTACLE_MAX_H) / 2:.1f} m",
                        f"≥{self.OBSTACLE_MAX_H:.1f} m obstacle",
                    ],
                    len=0.6,
                ),
                hovertemplate=(
                    "x=%{x:.2f}m  z=%{y:.2f}m<br>"
                    "height=%{z:.2f} (norm)<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=10, b=40, t=28),
            title=dict(
                text=f"Traversability Map  (ground Y≈{gy:.2f} m, "
                     f"obstacle {self.OBSTACLE_MIN_H*100:.0f}–{self.OBSTACLE_MAX_H*100:.0f} cm)",
                font=dict(size=12),
            ),
            xaxis=dict(title="X (m)", color="#888", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Z (m)", color="#888"),
            uirevision="occ",
        )
        return fig
