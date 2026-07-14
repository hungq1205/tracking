"""
GridPathPlanner — A* path planning over the height-tiered occupancy grid
exported by scan_server/occupancy_map.py (top-level "occupancy_grid" field in
map_labels.json, written by scan_server/map_exporter.py's export_map()).

This is deliberately NOT RTAB-Map's own Rtabmap::computePath()/OccupancyGrid
(corelib features that exist but have no "low enough to step over" semantic —
this project's occupancy grid distinguishes a low obstacle a user can step
over from a normal one, which RTAB-Map's own grid module has no concept of),
and NOT an extension of RoutePlanner's existing zone-centroid greedy walk
(server/tools/route_planner.py, unchanged — still used for zone-label
destinations). This is the path planner for "go to <landmark>" destinations,
where there's no zone AABB to walk between, just a rough (x, z) point.

Class codes below MUST match scan_server/occupancy_map.py's CLASS_* constants
exactly — duplicated here rather than imported, since server/ and scan_server/
are separately deployed processes/environments (see CLAUDE.md's Development
Environments table) and this project's map_labels.json is the actual
integration boundary between them, not a shared Python import.
"""

from __future__ import annotations

import heapq
import json
import math
from typing import Dict, List, Optional, Tuple

CLASS_UNKNOWN = 0
CLASS_GROUND = 1
CLASS_LOW_STEP_OVER = 2
CLASS_OBSTACLE = 3

_COST_BY_CLASS: Dict[int, float] = {
    CLASS_GROUND: 1.0,
    # Passable at a premium, NOT blocked — the whole-map grid will have large
    # legitimately-unscanned patches between/around zone AABBs; treating
    # unknown as blocked would fragment the map into disconnected islands
    # with no path between them at all.
    CLASS_UNKNOWN: 1.5,
    # Passable, discouraged — a curb/low obstacle a user can step over, but
    # routing around it is preferred when a ground detour isn't much longer.
    CLASS_LOW_STEP_OVER: 3.0,
}
_BLOCKED = float("inf")  # CLASS_OBSTACLE — never passable


class GridPathPlanner:
    def __init__(self, grid: dict) -> None:
        self.resolution: float = float(grid["resolution"])
        self.origin_x: float = float(grid["origin_x"])
        self.origin_z: float = float(grid["origin_z"])
        self.width: int = int(grid["width"])
        self.height: int = int(grid["height"])
        self._class: List[List[int]] = grid["class"]

    @classmethod
    def from_map_file(cls, map_labels_path: str) -> Optional["GridPathPlanner"]:
        """None if map_labels.json has no top-level occupancy_grid field
        (older maps exported before this feature existed) — caller falls
        back to RoutePlanner.compute_route's zone-centroid walk."""
        with open(map_labels_path) as f:
            data = json.load(f)
        grid = data.get("occupancy_grid")
        if not grid or not grid.get("class"):
            return None
        return cls(grid)

    # ── coordinate conversion ────────────────────────────────────────────────

    def world_to_cell(self, x: float, z: float) -> Tuple[int, int]:
        col = int((x - self.origin_x) / self.resolution)
        row = int((z - self.origin_z) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = self.origin_x + (col + 0.5) * self.resolution
        z = self.origin_z + (row + 0.5) * self.resolution
        return x, z

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def _cost(self, row: int, col: int) -> float:
        if not self._in_bounds(row, col):
            return _BLOCKED
        cls = self._class[row][col]
        if cls == CLASS_OBSTACLE:
            return _BLOCKED
        return _COST_BY_CLASS.get(cls, 1.0)

    def _passable(self, row: int, col: int) -> bool:
        return self._in_bounds(row, col) and self._class[row][col] != CLASS_OBSTACLE

    # ── planning ─────────────────────────────────────────────────────────────

    def find_path(
        self, start_xz: Tuple[float, float], goal_xz: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """
        8-connected A* with an octile heuristic, from start_xz to goal_xz
        (world (x, z) metres). Returns a simplified waypoint list ending at
        goal_xz (start_xz is NOT included — the caller already knows current
        position), or None if start/goal is blocked/out-of-grid or no path
        exists.
        """
        start = self.world_to_cell(*start_xz)
        goal = self.world_to_cell(*goal_xz)
        if not self._passable(*start) or not self._passable(*goal):
            return None
        if start == goal:
            return [goal_xz]

        raw = self._astar(start, goal)
        if raw is None:
            return None
        simplified = self._simplify(raw)
        # Replace the final waypoint's cell-center with the exact requested
        # goal (avoids a small "off by half a cell" drift in the destination).
        waypoints = [self.cell_to_world(r, c) for r, c in simplified[1:-1]]
        waypoints.append(goal_xz)
        return waypoints

    def _astar(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        def octile(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

        # (row, col) neighbor offsets + per-step base distance (1.0 cardinal,
        # sqrt(2) diagonal) — actual edge cost multiplies this by the
        # destination cell's traversal cost.
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        ]

        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        visited = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                path = [current]
                while path[-1] in came_from:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path

            r, c = current
            for dr, dc, step_dist in neighbors:
                nr, nc = r + dr, c + dc
                if not self._passable(nr, nc):
                    continue
                if dr != 0 and dc != 0:
                    # No corner-cutting through a blocked cell diagonally.
                    if not self._passable(r + dr, c) or not self._passable(r, c + dc):
                        continue
                step_cost = step_dist * self._cost(nr, nc)
                tentative_g = g_score[current] + step_cost
                neighbor = (nr, nc)
                if tentative_g < g_score.get(neighbor, _BLOCKED):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + octile(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, neighbor))

        return None

    def _line_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> Optional[float]:
        """Bresenham walk from a to b — returns the line's total traversal
        cost, or None if any swept cell is blocked. Cost-aware (not just a
        passability check): a straight line through step-over/unknown cells
        is 'clear' but not necessarily cheap, and simplification must never
        silently replace a cheaper zig-zag detour with a costlier-but-shorter
        straight line (see _simplify)."""
        r0, c0 = a
        r1, c1 = b
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        total = 0.0
        prev = (r, c)
        while True:
            if not self._passable(r, c):
                return None
            if (r, c) != prev:
                step_dist = math.sqrt(2) if (r != prev[0] and c != prev[1]) else 1.0
                total += step_dist * self._cost(r, c)
                prev = (r, c)
            if (r, c) == (r1, c1):
                return total
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def _simplify(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Greedy line-of-sight 'string pulling' — NOT Douglas-Peucker (DP only
        measures deviation distance from a straight line, it doesn't check
        the straight line is actually clear of obstacles, and even a
        "passable" straight line can be a false shortcut through costlier
        step-over/unknown cells that the original A* path deliberately
        avoided). From the current retained waypoint, find the farthest path
        cell reachable via a straight line that is BOTH clear AND no more
        expensive than following the original path to that same cell —
        preserving A*'s cost-optimality, not just its connectivity. Typically
        collapses a few-hundred-cell A* path into a handful of turn points.
        """
        if len(path) <= 2:
            return path

        # Cumulative cost of the raw path up to each index, for the
        # no-more-expensive-than-the-original-route comparison below.
        cumulative = [0.0] * len(path)
        for i in range(1, len(path)):
            r0, c0 = path[i - 1]
            r1, c1 = path[i]
            step_dist = math.sqrt(2) if (r0 != r1 and c0 != c1) else 1.0
            cumulative[i] = cumulative[i - 1] + step_dist * self._cost(r1, c1)

        simplified = [path[0]]
        anchor_idx = 0
        while anchor_idx < len(path) - 1:
            farthest = anchor_idx + 1
            for j in range(anchor_idx + 1, len(path)):
                line_cost = self._line_cost(path[anchor_idx], path[j])
                original_cost = cumulative[j] - cumulative[anchor_idx]
                if line_cost is not None and line_cost <= original_cost + 1e-6:
                    farthest = j
                else:
                    break
            simplified.append(path[farthest])
            anchor_idx = farthest
        return simplified
