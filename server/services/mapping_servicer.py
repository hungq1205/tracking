"""
MappingServiceServicer — live SLAM-style mapping + localization for the
Android client, active whenever the client's guiding mode is on (see
CLAUDE.md's "Client-Orchestrated Live Session" section). Absorbs
scan_server's StreamingScanSession pipeline (imported in place via
sys.path, same convention tools/depth.py's DA3/mvs imports already use —
see that section for why files aren't physically relocated yet).

RTAB-Map is the ONLY pose source this service uses (no IMU+VO) — the old
scan_server GUI's IMU + VO pose source stays available there for now, but
the new live-guiding path is RTAB-Map only, so MappingChunk.imu_samples is
accepted on the wire but not consumed here.

Named zones/labels are dropped from this path entirely — navigation targets
landmarks/functional objects (Gemini -> GroundingDINO-tiny extracted via
semantic_mapper.py — see that module's docstring), not zone containers.
finalize_landmarks_flat() (scan_session.py) is the zone-free landmark
finalize used here instead of the legacy finalize_landmarks()/
map_exporter.py zone-shaped export.

Persistence: NONE — requested directly by the user ("remove the map
storing entirely, just new map every scan, guide"). A prior round of this
feature saved a coarse occupancy-grid+landmark summary to disk
(occupancy_snapshot.json) and re-seeded a revisited location_id's fresh
OccupancyMap from it; that whole mechanism (GetMapSnapshot/
ListMappedLocations, the snapshot-fallback half of FindLandmark) is gone.
Every SCAN/WALKING/GUIDING session now starts from a genuinely blank map,
every time — which was already true in-memory regardless (ScanSession.
reset_cloud() already wipes the occupancy grid AND _raw_landmarks on every
new StreamingScanSession for a given location_id, disk persistence or
not), so this change is purely "stop writing/reading the disk copy," not
a change to what a fresh session starts with.
"""

import itertools
import threading
import traceback

import cv2
import grpc
import numpy as np
from scipy.spatial.transform import Rotation

import tracking_pb2
import tracking_pb2_grpc
from live_path_planner import LiveGridPathPlanner, beacon_target_point
from stream_session import StreamingScanSession


def _latest_only_chunks(request_iterator):
    """Wraps a raw gRPC request_iterator in a single-slot mailbox: a
    background thread continuously drains request_iterator, overwriting one
    shared slot with each new chunk as it arrives — a chunk not yet
    consumed by the main loop (this generator) is silently dropped in
    favor of whatever's newest. The generator blocks on a Condition and
    yields whatever is CURRENTLY in the slot once notified.

    Re-adopted deliberately (see CLAUDE.md's "Drop-to-latest mapping-chunk
    ingestion" note) after being tried and reverted earlier in this
    project: RTAB-Map's frame-to-frame odometry needs temporal continuity,
    and dropping frames widens the motion/appearance gap between whatever
    it actually processes back-to-back, which was a confirmed contributor
    to a tracking-loss regression that time. Explicitly paired with a
    faster consecutive-tracking-loss reset (PURE_WALKING_LOST_RESET_S,
    scan_session.py) as the accepted mitigation this time, not an
    oversight — a known, accepted risk, not a correctness bug.

    NOT used for SessionMode.SCAN — see UpdateMapping's own chunk_source
    split. A scan wants completeness (every frame, in order), not
    freshness, so it bypasses this mailbox entirely and rides gRPC's own
    internal request queue instead.
    """
    condition = threading.Condition()
    state = {"chunk": None, "seq": 0, "done": False, "error": None}

    def _reader():
        try:
            for chunk in request_iterator:
                with condition:
                    state["chunk"] = chunk
                    state["seq"] += 1
                    condition.notify()
        except Exception as e:  # noqa: BLE001 - re-raised on the consumer side
            with condition:
                state["error"] = e
                condition.notify()
        finally:
            with condition:
                state["done"] = True
                condition.notify()

    threading.Thread(target=_reader, daemon=True).start()

    last_seq = 0
    while True:
        with condition:
            while state["seq"] == last_seq and not state["done"] and state["error"] is None:
                condition.wait()
            if state["error"] is not None:
                raise state["error"]
            if state["seq"] == last_seq and state["done"]:
                return
            chunk = state["chunk"]
            last_seq = state["seq"]
        yield chunk

# WALKING's planning horizon for find_natural_path() — how far ahead
# (real metres, not a fixed goal) the planner searches at all. "Typically
# 4-6m" per the user's own spec (see CLAUDE.md's "Natural path planner"
# note) — 5.0m sits in the middle of that.
_WALKING_MAX_PLANNING_DISTANCE_M = 5.0

# WALKING-only desired clearance from obstacles — find_natural_path()'s
# obstacle_proximity cost term is 0 once a cell's own clearance already
# meets this, ramping up as it falls short (never a hard block — see that
# method's own docstring). ~0.5m is a rough "one person's comfortable
# width" margin, not a hard human-body measurement. GUIDING's own planner
# instance (find_path(), unaffected by this planner) has no equivalent
# knob — see the WALKING-branch comment where this is used.
_WALKING_SAFE_CLEARANCE_M = 0.5


def _grid_dict_to_proto(grid_dict: dict) -> tracking_pb2.OccupancyGrid:
    if grid_dict is None:
        return tracking_pb2.OccupancyGrid()
    cls_flat = [c for row in grid_dict["class"] for c in row]
    height_flat = [
        (-1.0 if v != v else v)  # NaN (CLASS_UNKNOWN) -> -1.0 sentinel, matches occupancy_map.py's own convention
        for row in grid_dict["data"] for v in row
    ]
    clearance_flat = [v for row in grid_dict["clearance"] for v in row]
    return tracking_pb2.OccupancyGrid(
        width=grid_dict["width"],
        height=grid_dict["height"],
        origin_x=grid_dict["origin_x"],
        origin_z=grid_dict["origin_z"],
        cell_size=grid_dict["resolution"],
        cls=cls_flat,
        height_norm=height_flat,
        clearance_m=clearance_flat,
    )


def _delta_to_proto(delta_dict: dict) -> tracking_pb2.OccupancyGridDelta:
    return tracking_pb2.OccupancyGridDelta(
        resolution=delta_dict["resolution"],
        cells=[
            tracking_pb2.OccupancyCellUpdate(
                ix=c["ix"], iz=c["iz"], cls=c["class"],
                height_norm=(-1.0 if c["height_norm"] != c["height_norm"] else c["height_norm"]),  # NaN -> -1.0
                clearance_m=c["clearance"],
            )
            for c in delta_dict["cells"]
        ],
    )


def _landmarks_to_proto(landmarks) -> list:
    return [
        tracking_pb2.Landmark(name=lm.name, x=lm.x, z=lm.z, confidence=lm.confidence)
        for lm in landmarks
    ]


def _pose_to_proto(pose_mat: np.ndarray) -> tracking_pb2.Pose:
    t = pose_mat[:3, 3]
    quat = Rotation.from_matrix(pose_mat[:3, :3]).as_quat()  # x, y, z, w
    return tracking_pb2.Pose(
        x=float(t[0]), y=float(t[1]), z=float(t[2]),
        qx=float(quat[0]), qy=float(quat[1]), qz=float(quat[2]), qw=float(quat[3]),
    )


def _pose_heading_rad(pose_mat: np.ndarray) -> float:
    """World-frame heading (floor-plane yaw) of the camera's own forward
    axis — used as WALKING's "keep going forward" reference direction for
    find_natural_path(), since the client sends no separate heading
    field (the server's own RTAB-Map pose already carries full orientation).
    Camera-local convention throughout this project is X-right/Y-down/
    Z-forward (see HrtfBeacon.kt's own docstring) — rotate the local
    forward vector (0, 0, 1) by the pose's rotation matrix into world
    space, then take its floor-plane angle via atan2(x, z), matching
    HrtfBeacon.kt's directionTo()/worldPointFrom() azimuth convention
    exactly (0 = world +Z, positive rotates toward world +X)."""
    forward_world = pose_mat[:3, :3] @ np.array([0.0, 0.0, 1.0])
    return float(np.arctan2(forward_world[0], forward_world[2]))


def _bearing_rad(from_xz: "tuple[float, float]", to_xz: "tuple[float, float]") -> float:
    """World-frame bearing from `from_xz` to `to_xz`, same 0=world+Z/
    positive-toward-world+X convention as `_pose_heading_rad()`/
    `HrtfBeacon.kt` — used as WALKING's replan reference direction when a
    previous route exists (bearing toward its own first joint) instead of
    the user's raw live heading, see the path-stability note where this is
    called."""
    dx = to_xz[0] - from_xz[0]
    dz = to_xz[1] - from_xz[1]
    return float(np.arctan2(dx, dz))


def _planned_path_to_proto(result) -> tracking_pb2.PlannedPath:
    """`result` is whatever LiveGridPathPlanner.find_path()/
    find_natural_path() returned — None (no walkable path at all) or
    (waypoints, confirmed, reached_exactly)."""
    if result is None:
        return tracking_pb2.PlannedPath(points=[], confirmed=False, reached_exactly=False)
    waypoints, confirmed, reached_exactly = result
    return tracking_pb2.PlannedPath(
        points=[tracking_pb2.PathPoint(x=x, z=z) for x, z in waypoints],
        confirmed=confirmed,
        reached_exactly=reached_exactly,
    )


class MappingServiceServicer(tracking_pb2_grpc.MappingServiceServicer):
    def __init__(self, scan_manager, activity_monitor=None):
        self.scan_manager = scan_manager
        self.activity_monitor = activity_monitor
        # (ix_lo, iz_lo, width, height) as of the last FULL grid sent per
        # location_id — see UpdateMapping's full-vs-delta decision. Absent
        # entry means "never sent a full grid this stream", forcing one.
        self._last_full_bounds: dict = {}
        # Path-stability caches, keyed by location_id — see UpdateMapping's
        # per-mode planning block and CLAUDE.md's path-stability note.
        # _prev_path holds whatever (waypoints, confirmed, reached_exactly)
        # tuple was last SENT for this location (WALKING or GUIDING alike);
        # _prev_goal_xz additionally tracks GUIDING's own destination, so a
        # genuine destination change (not just a grid update) always forces
        # a fresh plan regardless of whether the old route still looks
        # walkable. Both popped at stream-open and on a total-tracking-loss
        # reset, same precedent _last_full_bounds already uses — a fresh
        # stream/reset means the old route is meaningless (new pose origin,
        # possibly a new local map entirely).
        self._prev_path: dict = {}
        self._prev_goal_xz: dict = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _decode_image_rgb(self, data: bytes):
        if not data:
            return None
        nparr = np.frombuffer(data, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ── RPCs ─────────────────────────────────────────────────────────────────

    def UpdateMapping(self, request_iterator, context):
        stream: "StreamingScanSession | None" = None
        location_id = None
        last_update_count = -1
        chunks_received = 0
        updates_sent = 0
        # SessionMode.WALKING specifically (not GUIDING) — see
        # process_frames_batch's pure_walking docstring and CLAUDE.md's
        # walking-mode local-map note. Initialized here (not just where
        # it's actually set, on the first chunk) so the `finally` block
        # below can safely read it even if an exception fires before any
        # chunk is processed.
        pure_walking = False
        peer = context.peer()
        # GUIDING's resolved destination — the client only learns this after
        # its own FindLandmark call succeeds, which can land several chunks
        # into the stream, not necessarily the first one (see MappingChunk.
        # has_goal's own proto comment) — so this is updated from whichever
        # chunk actually carries it, not read once at stream-open time.
        current_goal_xz: "tuple[float, float] | None" = None

        try:
            # Peel off the first chunk directly (never dropped either way)
            # to learn session_mode BEFORE deciding which ingestion
            # strategy the rest of the stream gets — see the SCAN-vs-
            # everything-else split below. An empty stream (no chunks at
            # all) just returns, same as the old loop would have on an
            # immediately-exhausted _latest_only_chunks().
            raw_iter = iter(request_iterator)
            try:
                first_chunk = next(raw_iter)
            except StopIteration:
                return

            # SCAN is the one exception to drop-to-latest ingestion (see
            # _latest_only_chunks()'s own docstring for why WALKING/GUIDING
            # want freshness over completeness instead) — requested
            # directly by the user: a scan wants EVERY frame processed, in
            # order, even if the server falls behind, buffering while busy
            # and continuing to drain after the client stops sending,
            # rather than silently discarding whatever it didn't get to.
            # gRPC's own request_iterator already queues incoming messages
            # internally (confirmed in _latest_only_chunks()'s own
            # docstring) — that queue IS the buffer; the only change needed
            # is to NOT wrap SCAN's stream in the mailbox that would
            # otherwise throw the backlog away.
            if first_chunk.session_mode == tracking_pb2.SCAN:
                chunk_source = itertools.chain([first_chunk], raw_iter)
            else:
                chunk_source = itertools.chain([first_chunk], _latest_only_chunks(raw_iter))

            for chunk in chunk_source:
                chunks_received += 1
                if chunk.has_goal:
                    current_goal_xz = (chunk.goal_x, chunk.goal_z)
                if stream is None:
                    location_id = chunk.location_id
                    if not location_id:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("location_id required on the first MappingChunk")
                        return
                    session_mode = chunk.session_mode  # only meaningful on this, the first chunk
                    walking_lite = session_mode != tracking_pb2.SCAN
                    # SessionMode.WALKING specifically — gets the same live
                    # occupancy grid GUIDING does now (client-side
                    # LocalPathPlanner routes through it toward a synthetic
                    # "straight ahead" target — see CLAUDE.md's walking-mode
                    # local-map note), just never persisted (finally block
                    # below) and reset far more aggressively on tracking
                    # loss (PURE_WALKING_LOST_RESET_S) than GUIDING would
                    # want. Still no VLM tagging/reconstruction — walking_lite
                    # already covers that for both WALKING and GUIDING.
                    pure_walking = session_mode == tracking_pb2.WALKING
                    print(f"[MappingService] UpdateMapping stream OPENED <- {peer} location_id='{location_id}' "
                          f"session_mode={tracking_pb2.SessionMode.Name(session_mode)}")
                    stream = StreamingScanSession(
                        self.scan_manager, location_id, pose_src="RTAB-Map",
                        walking_lite=walking_lite, pure_walking=pure_walking,
                    )
                    # Blur filtering disabled for the live path (both scan and
                    # walking/guiding) — confirmed with the user. Leaves
                    # self._min_sharpness at DEFAULT_MIN_SHARPNESS (0.0),
                    # which process_frames_batch's Step 0 pre-check and the
                    # per-frame novelty+blur gate both already treat as "blur
                    # gating off entirely" (see orb_novelty_gate.py's
                    # decide_accept() docstring) — no configure_novelty_gate()
                    # call needed. Walking/guiding still skip the NOVELTY
                    # check too (see process_frames_batch's walking_lite
                    # branch) — with blur also off, a walking/guiding frame
                    # is accepted whenever RTAB-Map actually produced a pose
                    # for it, full stop.
                    # A fresh stream means a fresh client-side grid too (this class'
                    # own caches are servicer-instance-scoped, so they'd otherwise
                    # survive across separate streams for the same location_id) —
                    # without this, a coincidental bounds match against a PRIOR
                    # stream's cached bounds could wrongly decide "delta suffices"
                    # on this stream's very first update, when the client actually
                    # has no local grid at all yet and needs a full resync.
                    self._last_full_bounds.pop(location_id, None)
                    self._prev_path.pop(location_id, None)
                    self._prev_goal_xz.pop(location_id, None)
                    last_update_count = stream.session.occupancy_map._update_count
                if chunks_received == 1:
                    print(f"[MappingService] First frame received <- {peer} "
                          f"image_bytes={len(chunk.image_data)}")

                frame_rgb = self._decode_image_rgb(chunk.image_data)
                if frame_rgb is None:
                    print(f"[MappingService] chunk #{chunks_received}: failed to decode image_data")
                    continue

                result = stream.push_frame(frame_rgb, chunk.frame_timestamp_ns)
                if result is None:
                    continue  # still buffering this mini-batch

                session = stream.session

                # A just-fired total-tracking-loss reset (WALKING or
                # GUIDING — see ScanSession._reset_cloud_locked()/
                # PURE_WALKING_LOST_RESET_S/GUIDING_LOST_RESET_S) clears
                # last_frame_poses, so it must be checked BEFORE the "no
                # pose yet" gate below, or the one update the client
                # actually needs (to drop its now-stale planned route)
                # would get silently swallowed by that gate instead. No
                # longer pure_walking-only — GUIDING now gets this same
                # reset too (a much longer threshold, see that constant's
                # own comment), and needs its route invalidated exactly the
                # same way WALKING's did.
                if walking_lite and session.last_reset_occurred:
                    updates_sent += 1
                    print(f"[MappingService] MappingUpdate #{updates_sent} -> {peer} "
                          f"{'pure_walking' if pure_walking else 'guiding'} RESET "
                          f"(chunks_received={chunks_received})")
                    if self.activity_monitor is not None:
                        self.activity_monitor.record_mapping(
                            f"UpdateMapping location='{location_id}' pure_walking RESET",
                            op="UpdateMapping", frame_rgb=frame_rgb, location_id=location_id,
                            rtabmap_lost=session.last_rtabmap_lost,
                            rtabmap_total=session.last_rtabmap_total,
                            # Both stale now — the reset wiped the pose/grid
                            # this path/beacon point were computed against;
                            # clear so the dashboard doesn't keep drawing a
                            # route/marker/heading arrow from before the reset.
                            planned_path=[], beacon_point=None, heading_rad=None,
                        )
                    # The client's own local grid is gone too now (a fresh
                    # RTAB-Map track means a fresh map, see
                    # ScanSession._reset_cloud_locked()) — pop the cached
                    # bounds so the next real grid update forces a full
                    # resync instead of trying to patch a grid the client
                    # no longer has (same reasoning as the pop at stream
                    # OPEN, above). The old route is equally meaningless
                    # after a reset (new pose origin, possibly a new local
                    # map) — same reasoning for _prev_path/_prev_goal_xz.
                    self._last_full_bounds.pop(location_id, None)
                    self._prev_path.pop(location_id, None)
                    self._prev_goal_xz.pop(location_id, None)
                    yield tracking_pb2.MappingUpdate(
                        reset_occurred=True, frame_timestamp_ns=chunk.frame_timestamp_ns,
                    )
                    continue

                if not session.last_frame_poses:
                    continue
                pose_mat = session.last_frame_poses[-1]
                pose_proto = _pose_to_proto(pose_mat)
                pose_xz = (float(pose_mat[0, 3]), float(pose_mat[2, 3]))

                # Server-planned route for this update (see PlannedPath's own
                # proto comment and CLAUDE.md's "Server-planned walking path"
                # note) — the client no longer runs its own A* against a
                # streamed grid, so this replaces that entirely. Needs the
                # FULL current grid regardless of grid_updated below (that
                # flag is purely a wire-bandwidth decision for `grid`/
                # `grid_delta` — extract_dirty_delta() already pays for the
                # same full classification+EDT pass internally, so this
                # isn't meaningfully more expensive than what already runs
                # every update).
                grid_for_planning = session.occupancy_map.extract_full_grid()
                # Computed unconditionally (not just for WALKING's own
                # target-selection use below) so server_gui.py can draw
                # which way the user is actually facing, for both modes —
                # see CLAUDE.md's "user-facing-direction GUI indicator" note.
                heading_rad = _pose_heading_rad(pose_mat)
                prev_entry = self._prev_path.get(location_id)
                if session_mode == tracking_pb2.WALKING:
                    # WALKING-only: find_natural_path()'s own cost function
                    # (heading bias, obstacle-proximity/clearance, turn
                    # count+angle, path length — see that method's
                    # docstring and CLAUDE.md's "Natural path planner" note)
                    # replaces the old find_farthest_open_path() heuristic
                    # entirely. No min_path_clearance_m constructor param
                    # needed here any more — find_natural_path() doesn't use
                    # _cost()/_clearance_multiplier() at all, it has its own
                    # safe_clearance_m concept (_WALKING_SAFE_CLEARANCE_M).
                    #
                    # Path-stability (see CLAUDE.md's path-stability note):
                    # a prior round of this feature replanned fresh every
                    # single update, with progress retained only via the
                    # client's own per-joint tracking (LiveSessionState.
                    # mainPathIdx) — a real, reported problem with that:
                    # re-deriving the whole route from scratch every ~1Hz
                    # update let the FIRST joint itself wander a little each
                    # time even when nothing material had changed, since
                    # find_natural_path() searches fresh off the (slightly
                    # noisy) live heading every time. Fixed two ways,
                    # requested directly by the user:
                    #   1. If a previous route exists and its very next
                    #      segment (from the CURRENT pose) isn't blocked in
                    #      the fresh grid, REUSE it unchanged — no new
                    #      search at all this update. Only the near-term
                    #      portion is checked (path_blocked_ahead()) — a
                    #      blockage further along doesn't matter yet; it'll
                    #      be re-checked (and, if still blocked once
                    #      actually close to it, trigger a real replan
                    #      then) on a later update as the user gets nearer.
                    #   2. Otherwise (no previous route, or it's now blocked
                    #      near-term), replan — but instead of the user's
                    #      raw live heading, find_natural_path()'s
                    #      "keep going this way" reference direction becomes
                    #      the bearing toward the OLD route's own first
                    #      joint (falling back to the real heading only
                    #      when there's no previous route to reference at
                    #      all) — "incentivize the path that was previously
                    #      planned, especially the first joint," per the
                    #      user's own framing. Since find_natural_path()'s
                    #      whole cost model already rewards continuing
                    #      straight in its reference direction over
                    #      turning, this alone reproduces a highly similar
                    #      route to the old one whenever nothing material
                    #      changed, without a separate distance-to-old-path
                    #      cost term (which GUIDING's own planner, below,
                    #      DOES need — find_path()'s A* has no heading
                    #      concept to redirect this way).
                    planner = LiveGridPathPlanner(grid_for_planning)
                    prev_waypoints = prev_entry[0] if prev_entry is not None else None
                    can_reuse = bool(prev_waypoints) and not planner.path_blocked_ahead(prev_waypoints, pose_xz)
                    if can_reuse:
                        path_result = prev_entry
                    else:
                        if prev_waypoints:
                            reference_heading_rad = _bearing_rad(pose_xz, prev_waypoints[0])
                        else:
                            reference_heading_rad = heading_rad
                        path_result = planner.find_natural_path(
                            pose_xz, reference_heading_rad,
                            max_distance_m=_WALKING_MAX_PLANNING_DISTANCE_M,
                            safe_clearance_m=_WALKING_SAFE_CLEARANCE_M,
                        )
                    self._prev_path[location_id] = path_result

                    # One-shot-per-batch diagnostic, not gated/removed yet —
                    # heading_rad's derivation was verified self-consistent
                    # against _project_world_to_pixel() via a synthetic
                    # non-identity-rotation test (server_gui.py), but neither
                    # has been checked against a REAL RTAB-Map pose from a
                    # live device. If the resulting path/frame-overlay still
                    # doesn't match the user's actual facing direction,
                    # compare this printed heading (and pose_mat's own
                    # position/rotation) against their real-world orientation
                    # to localize the mismatch — see CLAUDE.md's
                    # "Server-planned walking path" open-risk note.
                    print(
                        f"[MappingService] WALKING heading_rad={heading_rad:.3f} "
                        f"({np.degrees(heading_rad):.1f} deg) pose_xz={pose_xz} "
                        f"pose_y={pose_mat[1, 3]:.3f} ground_y={session.occupancy_map._ground_y:.3f} "
                        f"path_points={len(path_result[0]) if path_result else 0} "
                        f"reused_path={can_reuse}"
                    )
                elif session_mode == tracking_pb2.GUIDING and current_goal_xz is not None:
                    # Same reuse-unless-blocked-early gate as WALKING above,
                    # PLUS: a genuine destination change always forces a
                    # fresh plan regardless (the old route was toward a
                    # different point entirely, reusing it would be wrong,
                    # not just "less stable"). find_path()'s own
                    # previous_path_xz param supplies the previous-path
                    # attraction bias for GUIDING specifically — its A* has
                    # no heading concept to redirect the way WALKING's
                    # search does, so this cost term is what "incentivize
                    # the path that was previously planned" means for this
                    # mode (see find_path()'s own docstring).
                    goal_changed = self._prev_goal_xz.get(location_id) != current_goal_xz
                    planner = LiveGridPathPlanner(grid_for_planning)
                    prev_waypoints = prev_entry[0] if (prev_entry is not None and not goal_changed) else None
                    can_reuse = bool(prev_waypoints) and not planner.path_blocked_ahead(prev_waypoints, pose_xz)
                    if can_reuse:
                        path_result = prev_entry
                    else:
                        path_result = planner.find_path(pose_xz, current_goal_xz, previous_path_xz=prev_waypoints)
                    self._prev_path[location_id] = path_result
                    self._prev_goal_xz[location_id] = current_goal_xz
                else:
                    path_result = None
                    self._prev_path.pop(location_id, None)
                    self._prev_goal_xz.pop(location_id, None)
                planned_path_proto = _planned_path_to_proto(path_result)

                # Same beacon-placement calculation ToolDispatcher.kt's
                # steerBeaconAlongPath() does on-device (project the current
                # pose onto the path, then move forward by a look-ahead
                # distance) — replicated here purely so the dashboard can
                # show where the HRTF beacon is actually pointing, alongside
                # the route it's drawn from. Uses the server's own last
                # AUTHORITATIVE pose (no latency-compensated extrapolation —
                # that only ever happens client-side); display-only, not
                # part of the real navigation signal.
                # pose_xz is prepended as the path's own first point — same
                # fix as ToolDispatcher.kt's: nearest_point_on_path()/
                # advance_along_path() only ever see segments that are IN
                # the list, and the server's own waypoints never include the
                # start, so without this the very first leg (and any path
                # that collapsed to a single waypoint) had no real segment
                # to project onto/advance along, degenerating straight to
                # "point at the final target" instead of a real lookahead
                # (this is exactly what made the dashboard's beacon marker
                # sit at the far end of the route). Safe to prepend fresh
                # every call here (unlike the client's per-tick steering
                # loop) since this whole calculation is a single-instant
                # snapshot — pose_xz and path_points_xz are always paired
                # from the SAME update, never re-anchored across ticks.
                path_points_xz = [(p.x, p.z) for p in planned_path_proto.points]
                pursuit_path = [pose_xz] + path_points_xz if path_points_xz else []
                beacon_point = beacon_target_point(pursuit_path, pose_xz)

                update_count = session.occupancy_map._update_count
                grid_updated = update_count != last_update_count
                last_update_count = update_count

                # Full-vs-delta decision: a delta's cell indices are only
                # valid against whatever bounding box the client's local
                # grid was last built from — if the map's bbox grew (newly
                # explored area) since the last full send, a delta alone
                # can't communicate that growth, so a full resync is
                # required. Same "full resync when structure changes,
                # incremental otherwise" split this codebase already uses
                # for RTAB-Map loop closure (_rtabmap_full_resync() vs.
                # _rtabmap_pull_new_nodes()).
                grid_dict = None
                delta_dict = None
                full_resync = False
                grid_proto = tracking_pb2.OccupancyGrid()
                delta_proto = tracking_pb2.OccupancyGridDelta()
                if grid_updated:
                    current_bounds = session.occupancy_map.bounds()
                    full_resync = (
                        location_id not in self._last_full_bounds
                        or self._last_full_bounds[location_id] != current_bounds
                    )
                    if full_resync:
                        grid_dict = session.occupancy_map.extract_full_grid()
                        grid_proto = _grid_dict_to_proto(grid_dict)
                        session.occupancy_map.clear_dirty()  # already covered by the full send
                        self._last_full_bounds[location_id] = current_bounds
                    else:
                        delta_dict = session.occupancy_map.extract_dirty_delta()
                        if delta_dict is not None:
                            delta_proto = _delta_to_proto(delta_dict)

                updates_sent += 1
                print(f"[MappingService] MappingUpdate #{updates_sent} -> {peer} "
                      f"pose=({pose_proto.x:.2f},{pose_proto.z:.2f}) grid_updated={grid_updated} "
                      f"full_resync={full_resync} "
                      f"delta_cells={len(delta_dict['cells']) if delta_dict else 0} "
                      f"confidence={session.last_batch_confidence:.2f} "
                      f"path_points={len(planned_path_proto.points)} "
                      f"path_confirmed={planned_path_proto.confirmed} "
                      f"(chunks_received={chunks_received})")

                if self.activity_monitor is not None:
                    self.activity_monitor.record_mapping(
                        f"UpdateMapping location='{location_id}' pose=({pose_proto.x:.2f},{pose_proto.z:.2f}) "
                        f"grid_updated={grid_updated}",
                        op="UpdateMapping", frame_rgb=frame_rgb, location_id=location_id,
                        pose_x=pose_proto.x, pose_z=pose_proto.z, grid_updated=grid_updated,
                        confidence=session.last_batch_confidence,
                        landmark_count=len(session._raw_landmarks),
                        # Actual positions/labels, not just the count above —
                        # so server_gui.py can plot them on the occupancy map
                        # instead of only showing a number in the Detail text.
                        landmarks=[
                            (lm.x, lm.z, lm.name) for lm in session._raw_landmarks
                        ],
                        # Live reference, not a snapshot — server_gui.py renders it on
                        # demand at poll time (occupancy_map.py's own render_plotly()/
                        # render_confidence_plotly(), no point-cloud/voxel rendering
                        # here, that's scan_gui.py's separate, heavier debug tool).
                        occupancy_map=session.occupancy_map,
                        # RTAB-Map's own per-batch tracking-lost count (see
                        # process_frames_batch's Step 2) — server_gui.py
                        # surfaces this directly so pose freezing is visible
                        # at a glance instead of only in console spam.
                        rtabmap_lost=session.last_rtabmap_lost,
                        rtabmap_total=session.last_rtabmap_total,
                        # Full pose (not just pose_x/pose_z) + the server-planned
                        # route, so server_gui.py can draw it on both the
                        # occupancy map and the raw frame — see "Server-planned
                        # walking path" note. ground_y lets the frame overlay
                        # project the (x, z)-only path points onto the floor
                        # plane rather than guessing a height.
                        pose_proto=pose_proto,
                        planned_path=[(p.x, p.z) for p in planned_path_proto.points],
                        path_confirmed=planned_path_proto.confirmed,
                        ground_y=session.occupancy_map._ground_y,
                        # Which way the user is actually facing (world-frame
                        # yaw, same convention as HrtfBeacon.kt's azimuth) —
                        # server_gui.py draws this as an arrow on the
                        # occupancy map and a degrees readout on the frame,
                        # so a mismatch against the route/frame content is
                        # visible directly instead of only in console logs.
                        heading_rad=heading_rad,
                        # Same beacon-placement calc ToolDispatcher.kt does
                        # on-device (see beacon_target_point() above) — None
                        # when there's no path to place it on (mirrors the
                        # client's own mute-when-no-path behavior).
                        beacon_point=beacon_point,
                    )

                yield tracking_pb2.MappingUpdate(
                    pose=pose_proto,
                    grid=grid_proto,
                    grid_updated=grid_updated,
                    landmarks=_landmarks_to_proto(session._raw_landmarks),
                    confidence=session.last_batch_confidence,
                    grid_delta=delta_proto,
                    full_resync=full_resync,
                    frame_timestamp_ns=chunk.frame_timestamp_ns,
                    planned_path=planned_path_proto,
                )
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
        finally:
            print(f"[MappingService] UpdateMapping stream CLOSED <- {peer} "
                  f"location_id='{location_id}' chunks_received={chunks_received} updates_sent={updates_sent}")
            if stream is not None:
                try:
                    stream.flush()
                    # No disk persistence any more (see this module's own
                    # docstring — "remove the map storing entirely, just new
                    # map every scan, guide") — finalize_landmarks_flat() is
                    # still worth calling for a genuine SCAN (flushes any
                    # last partial VLM-tag batch + runs the final overlap-
                    # merge dedup pass into session._raw_landmarks in
                    # memory), since a FindLandmark call against this SAME
                    # in-memory ScanSession can still land between this
                    # stream closing and a later one for the same
                    # location_id resetting it. pure_walking (SessionMode.
                    # WALKING) never accumulates anything meaningful here —
                    # walking_lite already skips VLM tagging entirely.
                    if not pure_walking:
                        stream.session.finalize_landmarks_flat()
                except Exception:
                    traceback.print_exc()

    # GetMapSnapshot/ListMappedLocations (both disk-persistence RPCs) are
    # gone outright, not deprecated — no disk map to snapshot/list any
    # more (see this module's own docstring). Neither had a real client
    # caller (confirmed by search — client/android never called either).
    # Left undefined on this subclass rather than stubbed: grpc's generated
    # base servicer already returns UNIMPLEMENTED for any method a subclass
    # doesn't override, which is the honest answer for an RPC that no
    # longer does anything.

    def FindLandmark(self, request, context):
        """Landmark lookup by name — GroundingDINO-tiny now runs immediately,
        per accepted frame, during scanning (see scan_session.py's
        resolve_landmark docstring); this just searches whatever's already
        been resolved into the given location's session._raw_landmarks. Uses
        scan_manager.get() (read-only — does NOT create a session), matching
        the convention scan_gui.py already uses for read-only lookups.
        No persisted-snapshot fallback any more (see this module's own
        docstring) — a walking/guiding session (which never runs semantic
        tagging itself) can only resolve a destination that a live SCAN of
        this SAME location_id, within the SAME server process lifetime,
        already found — an accepted consequence of removing disk
        persistence entirely, not an oversight."""
        print(f"[MappingService] FindLandmark <- {context.peer()} "
              f"location_id='{request.location_id}' query='{request.query}'")
        session = self.scan_manager.get(request.location_id)
        lm = session.resolve_landmark(request.query) if session is not None else None
        if lm is None:
            print(f"[MappingService] FindLandmark: '{request.query}' not resolved")
            if self.activity_monitor is not None:
                self.activity_monitor.record_mapping(
                    f"FindLandmark query='{request.query}' -> not found",
                    op="FindLandmark", location_id=request.location_id, query=request.query, found=False,
                )
            return tracking_pb2.FindLandmarkResponse(found=False)
        if self.activity_monitor is not None:
            self.activity_monitor.record_mapping(
                f"FindLandmark query='{request.query}' -> '{lm.name}'",
                op="FindLandmark", location_id=request.location_id, query=request.query, found=True,
                matched_label=lm.name, x=lm.x, z=lm.z, confidence=lm.confidence,
            )
        print(f"[MappingService] FindLandmark -> '{lm.name}' at ({lm.x:.2f},{lm.z:.2f}) "
              f"confidence={lm.confidence:.2f}")
        return tracking_pb2.FindLandmarkResponse(
            found=True, x=lm.x, z=lm.z, confidence=lm.confidence, matched_label=lm.name,
        )
