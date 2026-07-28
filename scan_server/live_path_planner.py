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

`find_natural_path()` (below `find_path()`) is a separate, independent
search — WALKING's own heading-biased, turn-averse planner (no fixed
destination), replacing the old `find_farthest_open_path()` heuristic
entirely. See its own docstring and CLAUDE.md's "Natural path planner"
note for the full design; GUIDING still uses `find_path()` above,
unchanged by any of this.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

CLASS_UNKNOWN = 0
CLASS_GROUND = 1
CLASS_LOW_STEP_OVER = 2
CLASS_OBSTACLE = 3

_COST_BY_CLASS: Dict[int, float] = {
    CLASS_GROUND: 1.0,
}
# CLASS_OBSTACLE, CLASS_LOW_STEP_OVER, and CLASS_UNKNOWN — never passable.
#
# CLASS_LOW_STEP_OVER used to be "costlier but passable" (3.0x) — a route
# could still cross a low step/curb as a last resort. Changed to a hard
# block per direct user request: this is an assistive system for a blind
# user, and a "step-over" cell is exactly the kind of low obstacle a real
# person could trip on without seeing it coming — never worth routing over,
# regardless of how much the alternative costs. _COST_BY_CLASS keeps no
# entry for it any more (moot — a blocked class is never passed to
# _COST_BY_CLASS.get() from _cost()/_passable(), only from
# _natural_step_cost()'s length_cost term, which also only ever runs for
# already-passable cells).
#
# CLASS_UNKNOWN used to be "passable at a premium" (cost 1.5, not blocked)
# specifically so a route could flow through unexplored territory when no
# fully-confirmed route existed yet — reverted per direct user feedback
# after seeing routes drawn well outside the actually-scanned area on the
# live dashboard: a blind user has no way to tell "confirmed floor" from
# "the planner's best guess about floor it's never actually seen," so a
# route MUST stay within genuinely observed cells, full stop. See
# find_path()'s own note on how a start position that itself lands on an
# unknown cell is now handled.
_BLOCKED = float("inf")


def _wrap_angle(angle_rad: float) -> float:
    """Normalizes an angle to (-pi, pi] — used throughout find_natural_path()
    for heading/turn-angle differences, which are otherwise sensitive to
    which side of the +-180 deg wrap the two angles happen to fall on."""
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


def _build_directions() -> List[Tuple[int, int, float, float]]:
    """The 8-connected (dr, dc, step_dist_in_cells, world_azimuth_rad) table
    find_natural_path()'s direction-augmented search moves through. Azimuth
    uses the SAME 0=world+Z/positive-toward-world+X convention as
    _pose_heading_rad()/HrtfBeacon.kt (row increases with world Z, col
    increases with world X — see cell_to_world()) — a cell delta (dr, dc)
    maps to a world delta (dz=dr*resolution, dx=dc*resolution), so its
    azimuth is just atan2(dc, dr) (resolution, a positive scalar, cancels
    out of atan2 entirely)."""
    raw = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    return [
        (dr, dc, math.sqrt(dr * dr + dc * dc), math.atan2(dc, dr))
        for dr, dc in raw
    ]


_DIRECTIONS: List[Tuple[int, int, float, float]] = _build_directions()


class LiveGridPathPlanner:
    CLEARANCE_PENALTY_SCALE = 4.0
    CLEARANCE_DECAY_RATE = 8.0
    # Additional penalty magnitude for cells narrower than
    # min_path_clearance_m, on top of the baseline exponential term above —
    # see _clearance_multiplier and the module docstring's point 3.
    NARROW_PENALTY_SCALE = 8.0
    # _simplify()'s cost-aware string-pulling normally only skips a
    # waypoint when the straight-line shortcut costs no more than the
    # original zig-zag route through that same stretch — a small amount of
    # slack lets it also drop a joint that would cost only marginally more,
    # trading a negligible amount of path efficiency for noticeably fewer
    # turns to announce/follow — a blind user following a beacon benefits
    # far more from "fewer, clearer turns" than from an A*-optimal polyline.
    SIMPLIFY_COST_SLACK_FRAC = 0.08

    # Previous-path attraction bias (both find_path()/GUIDING and
    # find_natural_path()/WALKING, via _prev_path_bias() below) — see that
    # method's own docstring and CLAUDE.md's path-stability note. Narrowed
    # per direct user follow-up feedback: ONLY the approach to the OLD
    # route's own FIRST joint matters (nothing beyond it — the rest of the
    # old multi-waypoint route is never referenced), and even then only the
    # FINAL PREV_PATH_BIAS_RANGE_M (1.5m) stretch leading into that joint —
    # "need to only keep up to 1.5m on the way to the joint, the remaining
    # path to the joint if has, can be freely adjust." A cell farther than
    # this from the old first joint gets ZERO bias (hard cutoff, not a
    # decay) — completely free to differ from the old route.
    PREV_PATH_PENALTY_SCALE = 2.0
    PREV_PATH_BIAS_RANGE_M = 1.5

    # path_blocked_ahead()'s "nothing left to reuse" trigger — a route
    # whose remaining length has dropped below this forces a fresh plan
    # even though nothing along it is actually blocked. Found via a real
    # bug: without this, an unobstructed straight corridor longer than the
    # route's own planning horizon would have the server reuse the SAME
    # route indefinitely, and once the client walked/joint-advanced past
    # its last waypoint it had nothing left to steer by — see
    # path_blocked_ahead()'s own docstring.
    REPLAN_NEAR_END_M = 1.0

    # find_natural_path()'s cost-function weights — directly the weights
    # the user specified for the "natural walking path" planner (see
    # CLAUDE.md's "Natural path planner" note): heading and safety terms
    # dominate, path length barely matters. Each per-step term below is
    # normalized to roughly [0, 1] before being multiplied by its weight,
    # so these constants directly express relative priority order rather
    # than needing separate unit-scaling reasoning per term.
    W_HEADING = 10.0
    W_OBSTACLE = 8.0
    W_TURN_COUNT = 7.0
    W_TURN_ANGLE = 5.0
    W_LENGTH = 2.0
    # Decay rate for the CONTINUOUS (no-cutoff) half of W_OBSTACLE's cost —
    # see _natural_step_cost()'s own comment. Deliberately much gentler
    # than CLEARANCE_PENALTY_SCALE/CLEARANCE_DECAY_RATE above (tuned for
    # find_path()'s "avoid imminent collision," negligible past ~1m) —
    # this needs to keep meaningfully discriminating over several metres
    # of open room so the search actually prefers the wider side of a
    # space, not just "not currently touching a wall." Same value already
    # used elsewhere in this codebase for an analogous "reward openness
    # over a longer range" purpose (the deleted find_farthest_open_path()
    # candidate-selection era's own clearance_decay_rate).
    SOFT_CLEARANCE_DECAY_RATE = 1.5

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
        # floor(), not int() — int() truncates toward zero (e.g. -0.3 -> 0,
        # not -1), which would silently alias a point just outside the
        # grid's negative edge onto a real in-bounds cell instead of
        # correctly reading it as out-of-bounds. Mirrors the same fix in
        # LocalPathPlanner.kt's worldToCell().
        col = math.floor((x - self.origin_x) / self.resolution)
        row = math.floor((z - self.origin_z) / self.resolution)
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
        if cls == CLASS_OBSTACLE or cls == CLASS_LOW_STEP_OVER or cls == CLASS_UNKNOWN:
            return _BLOCKED
        return _COST_BY_CLASS.get(cls, 1.0) * self._clearance_multiplier(row, col)

    def _passable(self, row: int, col: int) -> bool:
        if not self._in_bounds(row, col):
            return False
        cls = self._class[row][col]
        return cls != CLASS_OBSTACLE and cls != CLASS_LOW_STEP_OVER and cls != CLASS_UNKNOWN

    def _nearest_passable_cell(self, start: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """BFS outward from `start` (itself blocked/unknown) to the nearest
        passable cell — used when the current pose lands on a cell the grid
        doesn't yet consider walkable (usually CLASS_UNKNOWN: the user is
        standing somewhere the map hasn't confirmed as floor yet, e.g. right
        at the edge of what's been scanned), so routing has a genuine
        confirmed cell to start from instead of failing outright or
        planning from inside unexplored/blocked territory. Returns None
        only if literally nothing passable exists anywhere in the grid."""
        if self._passable(*start):
            return start
        visited = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc) or (nr, nc) in visited:
                    continue
                visited.add((nr, nc))
                if self._passable(nr, nc):
                    return nr, nc
                queue.append((nr, nc))
        return None

    # ── planning ─────────────────────────────────────────────────────────────

    def find_path(
        self, start_xz: Tuple[float, float], goal_xz: Tuple[float, float],
        previous_path_xz: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[Tuple[List[Tuple[float, float]], bool, bool]]:
        """
        8-connected A* with an octile heuristic, from start_xz to goal_xz
        (world (x, z) metres) — GUIDING's route to a real destination.

        [previous_path_xz] (optional — the PREVIOUS call's own returned
        waypoints, or None on the first plan of a session): when given,
        every candidate cell's cost gets an ADDITIONAL bounded penalty
        proportional to its distance from this old route, itself decayed
        by how far that cell is from `start_xz` (PREV_PATH_PROXIMITY_
        DECAY_RATE) — the search still reaches `goal_xz` (this never
        blocks anything, same "discourage, never forbid" philosophy as the
        clearance penalty above), it's just biased to stay close to
        wherever it was already routing, MOST STRONGLY near the user's
        current position, so a caller that only replans when genuinely
        forced to (see `path_blocked_ahead()` below) gets a new route that
        doesn't swing wildly away from the old one for no reason. Requested
        directly by the user: "incentivize the path that was previously
        planned, especially the first joint closest to the user."
        CLASS_UNKNOWN cells are BLOCKED, same as CLASS_OBSTACLE (reverted
        from an earlier "passable at a premium" design — see _BLOCKED's own
        comment for why: a blind user can't tell confirmed floor from the
        planner's best guess about floor it's never actually seen, so a
        route must never leave genuinely observed territory). If `start_xz`
        itself lands on a cell that isn't passable (most commonly: standing
        right at the edge of what's been scanned, still CLASS_UNKNOWN), the
        search instead begins from the nearest passable cell
        (`_nearest_passable_cell`) rather than failing outright — the
        returned waypoints still describe a route a real device can follow
        starting from wherever the client itself anchors the beacon (see
        ToolDispatcher.kt's own fixed-start-anchor prepend).

        Returns (waypoints, confirmed, reached_exactly):
          waypoints        end at the final cell actually reached (start_xz
                           is NOT included).
          confirmed        Always True now that unknown cells are blocked —
                           kept in the return shape for compatibility with
                           existing callers/rendering (dashed "speculative"
                           styling is simply never triggered any more).
          reached_exactly  False if the exact destination wasn't reachable
                           (blocked, inside an obstacle cell, or otherwise
                           unreachable) and this is instead a route to the
                           CLOSEST reachable cell — see _astar's docstring.
        Returns None only when nothing useful can be offered at all: no
        passable cell exists anywhere near start, or the search couldn't
        move anywhere.

        (WALKING no longer calls this at all — see find_natural_path(),
        its own independent heading-biased search, which already never
        crosses into CLASS_UNKNOWN either.)
        """
        start = self.world_to_cell(*start_xz)
        goal = self.world_to_cell(*goal_xz)
        if not self._passable(*start):
            snapped = self._nearest_passable_cell(start)
            if snapped is None:
                return None
            start = snapped
        if start == goal and self._passable(*goal):
            return [goal_xz], True, True

        raw, reached_exactly = self._astar(start, goal, start_xz, previous_path_xz)
        if raw is None:
            return None
        simplified = self._simplify(raw)
        end_xz = goal_xz if reached_exactly else self.cell_to_world(*raw[-1])
        waypoints = [self.cell_to_world(r, c) for r, c in simplified[1:-1]]
        waypoints.append(end_xz)
        return waypoints, True, reached_exactly

    def _astar(
        self, start: Tuple[int, int], goal: Tuple[int, int],
        start_xz: Optional[Tuple[float, float]] = None,
        previous_path_xz: Optional[List[Tuple[float, float]]] = None,
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

        `start_xz`/`previous_path_xz` are only used to compute the optional
        previous-path attraction bias (see find_path()'s own docstring) —
        `start_xz` is required whenever `previous_path_xz` is given.
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
                if previous_path_xz is not None and start_xz is not None:
                    step_cost += self._prev_path_bias(nr, nc, start_xz, previous_path_xz)
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

    def _prev_path_bias(
        self, row: int, col: int,
        start_xz: Tuple[float, float], previous_path_xz: List[Tuple[float, float]],
    ) -> float:
        """Additive edge-cost penalty shared by `find_path()`'s A* (GUIDING)
        and `find_natural_path()`'s direction-augmented search (WALKING) —
        see PREV_PATH_BIAS_RANGE_M's own comment for the exact scope, and
        find_path()/find_natural_path()'s own docstrings for how each
        threads `previous_path_xz` in.

        Deliberately narrow, per direct user feedback: only the OLD
        route's own FIRST joint (`previous_path_xz[0]`) is ever referenced
        — nothing beyond it — and even then only the FINAL
        PREV_PATH_BIAS_RANGE_M (1.5m) stretch of the straight-line approach
        from `start_xz` into that joint is biased. A cell farther than
        that from the old first joint gets ZERO bias (a hard cutoff, not a
        decay) — completely free to differ from the old route. Within that
        final stretch, the penalty is proportional to the cell's distance
        from the (possibly-trimmed) approach segment, so the search still
        prefers converging toward the joint along roughly the same
        corridor as before, not just "be near the joint from any angle."
        """
        if not previous_path_xz:
            return 0.0
        first_joint = previous_path_xz[0]
        cell_xz = self.cell_to_world(row, col)
        dist_to_joint_m = math.hypot(cell_xz[0] - first_joint[0], cell_xz[1] - first_joint[1])
        if dist_to_joint_m > self.PREV_PATH_BIAS_RANGE_M:
            return 0.0
        total_dist_m = math.hypot(first_joint[0] - start_xz[0], first_joint[1] - start_xz[1])
        if total_dist_m < 1e-6:
            return 0.0
        trim_m = min(self.PREV_PATH_BIAS_RANGE_M, total_dist_m)
        frac = (total_dist_m - trim_m) / total_dist_m
        bias_start = (
            start_xz[0] + frac * (first_joint[0] - start_xz[0]),
            start_xz[1] + frac * (first_joint[1] - start_xz[1]),
        )
        dist_to_segment_m = _point_segment_distance_m(cell_xz, bias_start, first_joint)
        return self.PREV_PATH_PENALTY_SCALE * dist_to_segment_m

    def segment_blocked(self, a_xz: Tuple[float, float], b_xz: Tuple[float, float]) -> bool:
        """True if the straight-line cell-walk between two world points
        passes through any cell that's impassable in the CURRENT grid —
        used by `path_blocked_ahead()` below to decide whether a
        previously-served route can still be trusted without a fresh
        search."""
        return self._line_cost(self.world_to_cell(*a_xz), self.world_to_cell(*b_xz)) is None

    def path_blocked_ahead(
        self, path_xz: List[Tuple[float, float]], pose_xz: Tuple[float, float],
    ) -> bool:
        """True if the ONE segment of a previously-served route (`path_xz`
        — waypoints only, same shape find_path()/find_natural_path() both
        return) the user is about to traverse RIGHT NOW is blocked in the
        CURRENT grid — the gate that decides whether that old route can be
        reused unchanged this update instead of replanned from scratch.

        Deliberately narrow, per direct user request ("only when the path
        is now blocked early... if only a distant end is blocked a bit then
        no worry"): a blockage further along the route doesn't matter yet —
        it'll be re-checked (and, if still blocked once actually close to
        it, trigger a real replan then) on a later update as the user gets
        nearer. Projects `pose_xz` onto the path (prepending it as the
        path's own implicit start — same fix `beacon_target_point()`
        already needed, since the served waypoints never include the
        start) to find which segment the user's CURRENT progress falls
        within, since local joint-arrival advancement happens client-side
        only and the server has no other way to know how far along an
        already-served route the user has actually gotten.

        ALSO forces a replan (returns True) once the user's progress along
        the route is within REPLAN_NEAR_END_M of its own final point —
        reusing a route with nothing left ahead of it would otherwise
        silently strand the caller once its last joint is reached. Real
        bug this fixes: WALKING's route has a bounded ~5m planning
        horizon, and "not blocked" alone would let the server keep reusing
        the SAME route indefinitely down an unobstructed corridor — the
        client would eventually joint-advance past its last waypoint with
        nothing new ever arriving to replace it (mainPathIdx running past
        the available points mutes the beacon client-side, see
        ToolDispatcher.kt's steerAlongMainPath())."""
        if not path_xz:
            return True
        pursuit_path = [pose_xz] + list(path_xz)
        total_len_m = sum(
            math.hypot(pursuit_path[i + 1][0] - pursuit_path[i][0], pursuit_path[i + 1][1] - pursuit_path[i][1])
            for i in range(len(pursuit_path) - 1)
        )
        projected = nearest_point_on_path(pursuit_path, pose_xz)
        if projected is None:
            return True
        _point, arc_len = projected
        if total_len_m - arc_len <= self.REPLAN_NEAR_END_M:
            return True
        cumulative = 0.0
        for i in range(len(pursuit_path) - 1):
            ax, az = pursuit_path[i]
            bx, bz = pursuit_path[i + 1]
            seg_len = math.hypot(bx - ax, bz - az)
            if arc_len <= cumulative + seg_len + 1e-6:
                return self.segment_blocked((ax, az), (bx, bz))
            cumulative += seg_len
        return False  # progress is at/past the route's own end — nothing left ahead to block

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
                if line_cost is not None and \
                        line_cost <= original_cost * (1.0 + self.SIMPLIFY_COST_SLACK_FRAC) + 1e-6:
                    farthest = j
                else:
                    break
            simplified.append(path[farthest])
            anchor_idx = farthest
        return simplified

    # ── natural (heading-biased, turn-averse) planning — WALKING ──────────────

    def find_natural_path(
        self,
        start_xz: Tuple[float, float],
        heading_rad: float,
        max_distance_m: float = 5.0,
        safe_clearance_m: float = 0.5,
        cone_stages_deg: Tuple[float, ...] = (45.0, 90.0, 180.0),
        min_progress_frac: float = 0.3,
        previous_path_xz: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[Tuple[List[Tuple[float, float]], bool, bool]]:
        """
        WALKING's target/route strategy (no destination) — replaces the old
        `find_farthest_open_path()`'s discrete cone-probing heuristic with a
        single, principled search: the objective is NOT the farthest or
        shortest reachable point, it's the path an experienced orientation
        & mobility instructor would choose — conservative, predictable,
        minimal-turn, and centered in open space, prioritizing the user's
        own current heading above all else. See CLAUDE.md's "Natural path
        planner" note for the full design rationale; summarized here:

        **Search**: an 8-connected Dijkstra whose STATE is (cell, incoming
        direction), not just cell — this is what lets per-step cost depend
        on whether a move continues straight or turns, not just on which
        cell it lands in (a plain cell-only A* has no way to penalize
        turning at all). Every edge's cost is the additive weighted sum:

            10 * heading_error   (quadratic in the angle between this
                                   move's direction and `heading_rad` —
                                   see _natural_step_cost())
          +  8 * obstacle_proximity  (0 once clearance >= safe_clearance_m,
                                   ramping up as it gets closer — this is
                                   ALSO what pulls the route toward the
                                   CENTER of open space: a corridor's
                                   centerline is where clearance, and thus
                                   this term, is locally lowest)
          +  7 * turn_count     (a flat penalty the instant direction
                                   changes at all)
          +  5 * turn_angle     (proportional to HOW sharp that turn is)
          +  2 * path_length    (real distance, class-tier-weighted —
                                   deliberately the smallest weight: this
                                   never overrides the terms above it)

        "Delay turning until forced" and "prefer few, gentle turns over a
        shorter zig-zag" are NOT special-cased — they're an emergent
        property of a turn-penalized shortest-path search: turning earlier
        than necessary (or more often than necessary) always costs strictly
        more for no benefit, so the minimum-cost path to any given point
        naturally goes as straight as it can for as long as it can.

        **Search region**: expansion is pruned to only cells whose BEARING
        FROM `start_xz` (a spatial region, not a per-step turn limit) falls
        within the current cone half-angle of `heading_rad` — this is what
        stops the search from ever wandering sideways through open space
        looking for a marginally cheaper route. Escalates through
        `cone_stages_deg` (default 45 deg -> 90 deg -> 180 deg) not only
        when a narrower stage finds NOTHING, but also when what it finds
        makes less than `min_progress_frac` (0.3) of `max_distance_m`
        worth of heading-axis progress — e.g. a corridor with a real,
        confirmed-open floor for a metre or two before a full-width wall
        would otherwise "succeed" at 45 deg (SOME path exists) and never
        even look for an opening off to the side, which isn't what "only
        turn when genuinely blocked" is supposed to mean. Every wider cone
        stage's reachable set is a strict superset of the narrower one's
        (same edges allowed, plus more), so escalating never loses a
        better result already found — the widest attempted stage's own
        result (even if still short of `min_progress_frac`) is always
        returned rather than nothing, matching this codebase's established
        "closest-approach beats no answer at all" philosophy. Expansion
        also never crosses into a still-unexplored (CLASS_UNKNOWN) cell —
        same "stay within confirmed territory" principle the old
        `find_path(confirmed_only=True)` used to enforce for WALKING (see
        the module docstring — that flag existed only for the now-removed
        `find_farthest_open_path()` and was deleted alongside it), here
        just baked directly into expansion instead of a post-hoc truncate.

        **Target selection**: among every cell the search actually reaches
        within `max_distance_m`, pick whichever has the greatest progress
        along the heading axis (`_directional_distance()`) — NOT the
        farthest by raw distance and NOT the cheapest by cost alone. Since
        the search cost already strongly penalizes heading deviation and
        turning, the cell that goes farthest ALONG THE HEADING naturally
        tends to be reached via the straightest, safest available route —
        this single criterion is what replaces the old two-stage straight-
        then-cone-fallback heuristic with one continuous decision.

        [previous_path_xz] (optional — the PREVIOUS call's own returned
        waypoints): threaded straight through to `_natural_step_cost()` via
        `_prev_path_bias()` — see that method's own docstring for the exact
        scope (only the final 1.5m approach into the OLD route's first
        joint is biased, nothing beyond it, nothing farther out). Always
        uses the REAL `heading_rad` for the cone/cost/target-selection
        heading terms regardless — this bias is a separate, additive,
        distance-bounded term, not a substitute reference direction.

        Returns None only if literally nothing beyond `start_xz` is
        reachable in ANY direction even at the widest (180 deg) cone stage
        — WALKING's dead-end trigger. `confirmed` is always True (expansion
        never leaves confirmed territory); `reached_exactly` is always True
        (there's no fixed destination for it to meaningfully qualify).
        """
        start = self.world_to_cell(*start_xz)
        if not self._passable(*start) or self._class[start[0]][start[1]] == CLASS_UNKNOWN:
            return None
        last_result: Optional[Tuple[List[Tuple[float, float]], bool, bool]] = None
        for i, cone_deg in enumerate(cone_stages_deg):
            result = self._search_natural(
                start, start_xz, heading_rad, max_distance_m, safe_clearance_m,
                math.radians(cone_deg), previous_path_xz,
            )
            if result is None:
                continue
            last_result = result
            waypoints, _confirmed, _reached_exactly = result
            directional = _directional_distance(start_xz, waypoints[-1], heading_rad)
            is_last_stage = i == len(cone_stages_deg) - 1
            if is_last_stage or directional >= max_distance_m * min_progress_frac:
                return result
        return last_result

    def _search_natural(
        self,
        start: Tuple[int, int], start_xz: Tuple[float, float], heading_rad: float,
        max_distance_m: float, safe_clearance_m: float, cone_half_rad: float,
        previous_path_xz: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[Tuple[List[Tuple[float, float]], bool, bool]]:
        """One cone-bounded search attempt for find_natural_path() — see
        that method's own docstring for the overall design. State =
        (row, col, incoming_direction_index), incoming_direction=-1 at the
        start (nothing to turn FROM yet, so the first move of any path
        incurs no turn penalty regardless of which direction it picks)."""
        StateT = Tuple[int, int, int]
        start_state: StateT = (start[0], start[1], -1)
        g_cost: Dict[StateT, float] = {start_state: 0.0}
        g_dist: Dict[StateT, float] = {start_state: 0.0}
        came_from: Dict[StateT, StateT] = {}
        open_heap: List[Tuple[float, StateT]] = [(0.0, start_state)]
        visited: set = set()
        reached: List[StateT] = []

        while open_heap:
            cost, state = heapq.heappop(open_heap)
            if state in visited:
                continue
            visited.add(state)
            reached.append(state)
            row, col, dir_idx = state
            dist_so_far = g_dist[state]

            for ndir_idx, (dr, dc, step_dist_cells, azimuth) in enumerate(_DIRECTIONS):
                nr, nc = row + dr, col + dc
                if not self._passable(nr, nc):
                    continue
                if self._class[nr][nc] == CLASS_UNKNOWN:
                    continue  # confirmed-territory-only, see docstring
                if dr != 0 and dc != 0:
                    if not self._passable(row + dr, col) or not self._passable(row, col + dc):
                        continue

                # Search-region prune: bearing from START (not from the
                # current cell) must stay within this attempt's cone — a
                # spatial-region constraint, deliberately distinct from the
                # per-step turn-angle cost below.
                nx, nz = self.cell_to_world(nr, nc)
                bearing = math.atan2(nx - start_xz[0], nz - start_xz[1])
                if abs(_wrap_angle(bearing - heading_rad)) > cone_half_rad:
                    continue

                step_dist_m = step_dist_cells * self.resolution
                new_dist = dist_so_far + step_dist_m
                if new_dist > max_distance_m:
                    continue  # don't expand past the planning horizon

                step_cost = self._natural_step_cost(
                    nr, nc, azimuth, heading_rad, dir_idx, step_dist_m, safe_clearance_m,
                    start_xz, previous_path_xz,
                )
                tentative_g = cost + step_cost
                nstate = (nr, nc, ndir_idx)
                if tentative_g < g_cost.get(nstate, float("inf")):
                    g_cost[nstate] = tentative_g
                    g_dist[nstate] = new_dist
                    came_from[nstate] = state
                    heapq.heappush(open_heap, (tentative_g, nstate))

        if len(reached) <= 1:
            return None  # nothing beyond start reachable in this cone

        # Target selection — see find_natural_path()'s own docstring.
        # Iterating `reached` in heap-pop (cost-ascending) order means a
        # tie in directional distance keeps whichever state was SETTLED
        # FIRST, i.e. reached more cheaply — an implicit, no-extra-code
        # cost tiebreak on top of the primary directional-distance choice.
        best_state: Optional[StateT] = None
        best_directional = float("-inf")
        for state in reached:
            if state == start_state:
                continue
            wx, wz = self.cell_to_world(state[0], state[1])
            directional = _directional_distance(start_xz, (wx, wz), heading_rad)
            if directional > best_directional:
                best_directional = directional
                best_state = state
        if best_state is None or best_directional <= 1e-6:
            return None  # nothing that actually counts as forward progress

        chain: List[StateT] = []
        cur = best_state
        while cur in came_from:
            chain.append(cur)
            cur = came_from[cur]
        chain.reverse()
        raw = [start] + [(s[0], s[1]) for s in chain]

        simplified = self._simplify(raw)
        waypoints = [self.cell_to_world(r, c) for r, c in simplified[1:]]
        return waypoints, True, True

    def _natural_step_cost(
        self, row: int, col: int, azimuth: float, heading_rad: float,
        prev_dir_idx: int, step_dist_m: float, safe_clearance_m: float,
        start_xz: Optional[Tuple[float, float]] = None,
        previous_path_xz: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        """Additive weighted step cost for find_natural_path()'s search —
        see that method's own docstring for the full formula/reasoning.
        `prev_dir_idx < 0` means this is the first step of the path (no
        prior direction to compare against), so it incurs no turn cost
        regardless of which of the 8 directions it picks. `start_xz`/
        `previous_path_xz`, when both given, add `_prev_path_bias()`'s
        bounded previous-route attraction term (see that method's own
        docstring) on top of the four terms below."""
        heading_err = abs(_wrap_angle(azimuth - heading_rad))
        heading_cost = self.W_HEADING * (heading_err / math.pi) ** 2

        # Obstacle proximity / corridor-centering — TWO terms, matching the
        # user's own two distinct priorities (#2 "center of wide space" and
        # #3 "comfortable clearance"), not one:
        #   - `steep`: 0 once this cell's own clearance already meets
        #     safe_clearance_m, ramping up (squared, for an "actually
        #     hugging something" penalty) as it falls short — this alone
        #     was the ORIGINAL implementation, and it has a real gap: two
        #     cells that both already clear safe_clearance_m (say 0.6m and
        #     3m) cost EXACTLY the same (zero) — nothing pulls the route
        #     toward the wider side of a room once the minimum is met,
        #     which is what let a route hug close to one obstacle even
        #     with much more open space just to the other side.
        #   - `soft`: a gentle exponential decay with NO hard floor/cutoff
        #     (SOFT_CLEARANCE_DECAY_RATE, deliberately much gentler than
        #     _clearance_multiplier's own CLEARANCE_DECAY_RATE=8.0, which
        #     is tuned to matter only within about a metre — this needs to
        #     keep discriminating over several metres of open room). This
        #     is what actually implements "stay near the center of wide
        #     free space" as a real, always-active preference, not just
        #     "don't get too close" — a corridor's centerline has the
        #     locally HIGHEST clearance, so minimizing this term pulls the
        #     route there with no separate centering heuristic needed.
        clearance_m = self._clearance[row][col] if self._clearance is not None else safe_clearance_m
        soft = math.exp(-self.SOFT_CLEARANCE_DECAY_RATE * clearance_m)
        steep = 0.0 if clearance_m >= safe_clearance_m else (1.0 - clearance_m / safe_clearance_m) ** 2
        obstacle_cost = self.W_OBSTACLE * (soft + steep)

        if prev_dir_idx < 0:
            turn_cost = 0.0
        else:
            prev_azimuth = _DIRECTIONS[prev_dir_idx][3]
            turn_angle = abs(_wrap_angle(azimuth - prev_azimuth))
            turn_cost = 0.0 if turn_angle < 1e-6 else (
                self.W_TURN_COUNT + self.W_TURN_ANGLE * (turn_angle / math.pi)
            )

        # Real path length, still tiered by traversability class (a
        # step-over cell costs more per metre than plain ground) — the
        # SAME _COST_BY_CLASS tiers find_path()'s A* uses, just folded in
        # at the lowest-priority weight instead of multiplying every other
        # term the way _cost()/_clearance_multiplier() do for find_path().
        cls = self._class[row][col]
        length_cost = self.W_LENGTH * step_dist_m * _COST_BY_CLASS.get(cls, 1.0)

        prev_path_cost = (
            self._prev_path_bias(row, col, start_xz, previous_path_xz)
            if start_xz is not None and previous_path_xz else 0.0
        )

        return heading_cost + obstacle_cost + turn_cost + length_cost + prev_path_cost


def _directional_distance(
    start_xz: Tuple[float, float], end_xz: Tuple[float, float], heading_rad: float,
) -> float:
    """How far `end_xz` progresses from `start_xz` along the CURRENT
    heading axis specifically — the dot product of the net start->end
    displacement with the heading's own unit vector — NOT the raw
    straight-line distance to the point, and NOT any path's actual
    (possibly curved) length to reach it. A point off to the side needs a
    lot of real displacement to make much progress along this axis (most
    of it is perpendicular), which is exactly the point: this measures
    "how far did this actually get me in the direction I'm facing" — used
    by find_natural_path()'s _search_natural() to pick the best reachable
    terminal cell."""
    heading_ux, heading_uz = math.sin(heading_rad), math.cos(heading_rad)
    dx, dz = end_xz[0] - start_xz[0], end_xz[1] - start_xz[1]
    return dx * heading_ux + dz * heading_uz


def _point_segment_distance_m(
    point_xz: Tuple[float, float], a_xz: Tuple[float, float], b_xz: Tuple[float, float],
) -> float:
    """Shortest distance from `point_xz` to the segment a_xz->b_xz (clamped
    to the segment, not the infinite line) — used by
    `LiveGridPathPlanner._prev_path_bias()` to measure how far a candidate
    cell strays from the (possibly-trimmed) approach into a previous
    route's first joint."""
    px, pz = point_xz
    ax, az = a_xz
    bx, bz = b_xz
    dx, dz = bx - ax, bz - az
    seg_len_sq = dx * dx + dz * dz
    if seg_len_sq < 1e-9:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / seg_len_sq))
    return math.hypot(px - (ax + dx * t), pz - (az + dz * t))


# Beacon look-ahead distance — matches ToolDispatcher.kt's PATH_LOOKAHEAD_M
# exactly (the middle of the user's own suggested 30-50cm range). Kept as a
# module constant here (Python port, server-side-only use — see
# nearest_point_on_path()/advance_along_path() below) rather than threaded
# as a parameter, since the two client/server copies should always agree.
PATH_LOOKAHEAD_M = 0.4


def nearest_point_on_path(
    path: List[Tuple[float, float]], current_xz: Tuple[float, float],
) -> Optional[Tuple[Tuple[float, float], float]]:
    """Python port of PathPursuit.kt's nearestPointOnPath() — projects
    current_xz onto the polyline `path` (world x/z, in order, NOT including
    the start point — same shape as PlannedPath.points/find_path()'s own
    return), finding the closest point on any segment (clamped to that
    segment, not the infinite line). Returns (point, cumulative arc-length
    from path[0]) or None for an empty path. Exists purely so
    mapping_servicer.py can replicate the CLIENT's HRTF beacon placement
    calculation server-side, for dashboard display only (see
    CLAUDE.md's "Server-planned walking path" note) — the real, latency-
    compensated placement still only ever happens on-device."""
    if not path:
        return None
    if len(path) == 1:
        return path[0], 0.0

    best_point = path[0]
    best_arc_len = 0.0
    best_dist_sq = float("inf")
    cumulative = 0.0
    for i in range(len(path) - 1):
        ax, az = path[i]
        bx, bz = path[i + 1]
        seg_dx, seg_dz = bx - ax, bz - az
        seg_len_sq = seg_dx * seg_dx + seg_dz * seg_dz
        seg_len = math.sqrt(seg_len_sq)
        if seg_len_sq > 1e-9:
            t = ((current_xz[0] - ax) * seg_dx + (current_xz[1] - az) * seg_dz) / seg_len_sq
            t = min(1.0, max(0.0, t))
        else:
            t = 0.0
        proj_x, proj_z = ax + seg_dx * t, az + seg_dz * t
        dist_sq = (current_xz[0] - proj_x) ** 2 + (current_xz[1] - proj_z) ** 2
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_point = (proj_x, proj_z)
            best_arc_len = cumulative + t * seg_len
        cumulative += seg_len
    return best_point, best_arc_len


def advance_along_path(
    path: List[Tuple[float, float]], arc_length: float, lookahead_m: float = PATH_LOOKAHEAD_M,
) -> Tuple[float, float]:
    """Python port of PathPursuit.kt's advanceAlongPath() — walks forward
    along `path` by `lookahead_m` starting from `arc_length` (as returned
    by nearest_point_on_path()), clamping at the path's final point."""
    if not path:
        return 0.0, 0.0
    if len(path) == 1:
        return path[0]

    target = arc_length + lookahead_m
    cumulative = 0.0
    for i in range(len(path) - 1):
        ax, az = path[i]
        bx, bz = path[i + 1]
        seg_len = math.hypot(bx - ax, bz - az)
        if target <= cumulative + seg_len:
            t = ((target - cumulative) / seg_len) if seg_len > 1e-6 else 0.0
            t = min(1.0, max(0.0, t))
            return ax + (bx - ax) * t, az + (bz - az) * t
        cumulative += seg_len
    return path[-1]


def beacon_target_point(
    path: List[Tuple[float, float]], pose_xz: Tuple[float, float], lookahead_m: float = PATH_LOOKAHEAD_M,
) -> Optional[Tuple[float, float]]:
    """Combines nearest_point_on_path() + advance_along_path() — the same
    two-step calculation ToolDispatcher.kt's steerBeaconAlongPath() does on
    the client (project onto the path, then move forward by a look-ahead
    distance), mirrored here purely so the dashboard can show where the
    HRTF beacon is actually pointing. Returns None (mute) if `path` is
    empty. Uses the server's own last AUTHORITATIVE pose, not a
    latency-compensated extrapolation — the real, extrapolated placement
    that actually drives the audio only ever happens on-device; this is
    display-only and doesn't need to be that precise."""
    projected = nearest_point_on_path(path, pose_xz)
    if projected is None:
        return None
    _point, arc_length = projected
    return advance_along_path(path, arc_length, lookahead_m)
