"""
LiveGridPathPlanner — A* path planning over the height-tiered occupancy
grid, run LIVE against an in-progress scan instead of a finished, exported
map.

This is an adapted copy of `server/tools/grid_path_planner.py`'s
`GridPathPlanner` — same CLASS_* constants, same cost model (ground cheap,
low/step-over costlier-but-passable, obstacle blocked, unknown passable-at-
a-premium so a route can still flow through unexplored territory when
that's the only way to reach the destination), same clearance-aware A* and
cost-aware string-pulling. Duplicated rather than imported for the same
reason that file's own docstring already gives for its CLASS_* constants:
`server/` and `scan_server/` are separately deployed processes/
environments (see CLAUDE.md's Development Environments table); the only
integration point between them is `map_labels.json`, which doesn't exist
yet for a scan still in progress.

Two behavioral differences from the deployed planner:

1. `find_path()` also reports whether the returned route is fully
   CONFIRMED (every cell along it is genuinely observed ground/step-over)
   or SPECULATIVE (it had to cross at least one still-unexplored cell to
   reach the destination) — useful live, during scanning, to visually flag
   "this route might not actually be walkable, that part of the venue
   hasn't been scanned yet" — not meaningful for the deployed end-user
   planner, which only ever runs against a finished map.

2. Closest-approach fallback: if the exact destination isn't reachable
   (blocked, or literally inside an obstacle cell), `find_path()` doesn't
   fail outright — it returns a route to the closest reachable cell
   instead, flagged via `reached_exactly=False`, computed during the SAME
   A* expansion (no second search) by tracking whichever visited cell had
   the smallest heuristic distance to the goal. Only returns `None` when
   truly nothing useful can be offered (start itself unreachable, or the
   search couldn't move anywhere at all).

3. `min_path_clearance_m` (constructor param, 0 = disabled): an additional,
   steeper cost penalty for cells narrower than this desired minimum width
   — still never a hard block (a genuinely narrow gap stays usable as a
   last resort, matching this project's "never fragment the map" cost
   philosophy), just more strongly discouraged than the baseline clearance
   shaping alone.

Constructed fresh each time from `OccupancyMap.extract_full_grid()`'s
in-memory dict (the exact same shape `GridPathPlanner` reads from
`map_labels.json`) — no file I/O, cheap enough to rebuild on every
occupancy-grid change during live scanning.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Tuple

CLASS_UNKNOWN = 0
CLASS_GROUND = 1
CLASS_LOW_STEP_OVER = 2
CLASS_OBSTACLE = 3

_COST_BY_CLASS: Dict[int, float] = {
    CLASS_GROUND: 1.0,
    # Passable at a premium, NOT blocked — this is what lets a route flow
    # through unexplored territory when no fully-confirmed route exists yet
    # (see module docstring's `confirmed` flag).
    CLASS_UNKNOWN: 1.5,
    CLASS_LOW_STEP_OVER: 3.0,
}
_BLOCKED = float("inf")  # CLASS_OBSTACLE — never passable


class LiveGridPathPlanner:
    CLEARANCE_PENALTY_SCALE = 4.0
    CLEARANCE_DECAY_RATE = 8.0
    # Additional penalty magnitude for cells narrower than
    # min_path_clearance_m, on top of the baseline exponential term above —
    # see _clearance_multiplier and the module docstring's point 3.
    NARROW_PENALTY_SCALE = 8.0

    def __init__(self, grid: dict, min_path_clearance_m: float = 0.0) -> None:
        self.resolution: float = float(grid["resolution"])
        self.origin_x: float = float(grid["origin_x"])
        self.origin_z: float = float(grid["origin_z"])
        self.width: int = int(grid["width"])
        self.height: int = int(grid["height"])
        self._class: List[List[int]] = grid["class"]
        self._clearance: Optional[List[List[float]]] = grid.get("clearance")
        self.min_path_clearance_m: float = min_path_clearance_m

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

    def _clearance_multiplier(self, row: int, col: int) -> float:
        """Baseline exponential-decay shaping (see grid_path_planner.py's
        docstring for the reasoning) plus, when min_path_clearance_m > 0, an
        ADDITIONAL bounded penalty for cells narrower than that desired
        minimum width — never a hard block, just more strongly discouraged.
        At clearance=0 vs. a configured min width this can multiply the
        baseline term (already up to ~5x) by up to another ~9x, but always
        stays finite so a narrow gap remains usable as a genuine last
        resort."""
        if self._clearance is None:
            return 1.0
        clearance_m = self._clearance[row][col]
        base = 1.0 + self.CLEARANCE_PENALTY_SCALE * math.exp(
            -self.CLEARANCE_DECAY_RATE * clearance_m
        )
        if self.min_path_clearance_m > 0.0 and clearance_m < self.min_path_clearance_m:
            deficit_frac = (self.min_path_clearance_m - clearance_m) / self.min_path_clearance_m
            base *= 1.0 + self.NARROW_PENALTY_SCALE * deficit_frac
        return base

    def _cost(self, row: int, col: int) -> float:
        if not self._in_bounds(row, col):
            return _BLOCKED
        cls = self._class[row][col]
        if cls == CLASS_OBSTACLE:
            return _BLOCKED
        return _COST_BY_CLASS.get(cls, 1.0) * self._clearance_multiplier(row, col)

    def _passable(self, row: int, col: int) -> bool:
        return self._in_bounds(row, col) and self._class[row][col] != CLASS_OBSTACLE

    # ── planning ─────────────────────────────────────────────────────────────

    def find_path(
        self, start_xz: Tuple[float, float], goal_xz: Tuple[float, float]
    ) -> Optional[Tuple[List[Tuple[float, float]], bool, bool]]:
        """
        8-connected A* with an octile heuristic, from start_xz to goal_xz
        (world (x, z) metres). Returns (waypoints, confirmed, reached_
        exactly):
          waypoints        end at the final cell actually reached (start_xz
                           is NOT included).
          confirmed        False if the route had to cross any still-
                           unexplored (CLASS_UNKNOWN) cell.
          reached_exactly  False if the exact destination wasn't reachable
                           (blocked, inside an obstacle cell, or otherwise
                           unreachable) and this is instead a route to the
                           CLOSEST reachable cell — see _astar's docstring.
        Returns None only when nothing useful can be offered at all: start
        itself isn't passable, or the search couldn't move anywhere.
        """
        start = self.world_to_cell(*start_xz)
        goal = self.world_to_cell(*goal_xz)
        if not self._passable(*start):
            return None
        if start == goal and self._passable(*goal):
            confirmed = self._class[start[0]][start[1]] != CLASS_UNKNOWN
            return [goal_xz], confirmed, True

        raw, reached_exactly = self._astar(start, goal)
        if raw is None:
            return None
        confirmed = not any(self._class[r][c] == CLASS_UNKNOWN for r, c in raw)
        simplified = self._simplify(raw)
        end_xz = goal_xz if reached_exactly else self.cell_to_world(*raw[-1])
        waypoints = [self.cell_to_world(r, c) for r, c in simplified[1:-1]]
        waypoints.append(end_xz)
        return waypoints, confirmed, reached_exactly

    def _astar(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Tuple[Optional[List[Tuple[int, int]]], bool]:
        """
        Returns (path, reached_exactly). If `goal` is reached, path ends
        there and reached_exactly=True. If the search exhausts without
        reaching `goal` (unreachable, or literally inside an obstacle so
        never passable in the first place), instead returns a path to
        whichever VISITED cell had the smallest heuristic distance to
        `goal` — the closest-approach fallback — with reached_exactly=
        False. Tracked during this same expansion, no second search
        needed. Returns (None, False) only if not even `start` could be
        expanded to anything (fully sealed-off start).
        """
        def octile(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        ]

        def reconstruct(node: Tuple[int, int], came_from) -> List[Tuple[int, int]]:
            path = [node]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        visited = set()
        best_node = start
        best_dist = octile(start, goal)

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            d = octile(current, goal)
            if d < best_dist:
                best_dist = d
                best_node = current
            if current == goal:
                return reconstruct(current, came_from), True

            r, c = current
            for dr, dc, step_dist in neighbors:
                nr, nc = r + dr, c + dc
                if not self._passable(nr, nc):
                    continue
                if dr != 0 and dc != 0:
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

        if best_node == start:
            return None, False
        return reconstruct(best_node, came_from), False

    def _line_cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> Optional[float]:
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
        if len(path) <= 2:
            return path

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
