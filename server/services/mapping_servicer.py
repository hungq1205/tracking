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
landmarks/functional objects (VLM-extracted via semantic_mapper.py), not
zone containers. finalize_landmarks_flat() (scan_session.py) is the
zone-free landmark finalize used here instead of the legacy
finalize_landmarks()/map_exporter.py zone-shaped export.

Persistence: only a coarse summary (class + normalized height per cell, not
raw per-cell logodds/height_ewma) is saved to disk. Revisiting a location
re-seeds the fresh session's OccupancyMap from that summary
(OccupancyMap.seed_from_summary) rather than truly resuming the prior
session's exact Bayesian belief — an explicit, accepted tradeoff (see that
method's docstring) over building full raw-state persistence.
"""

import json
import os
import traceback

import cv2
import grpc
import numpy as np
from scipy.spatial.transform import Rotation

import tracking_pb2
import tracking_pb2_grpc
from stream_session import StreamingScanSession
from services.beacon_preview import find_most_open_direction_world_point, project_world_point_to_pixel

SNAPSHOT_FILENAME = "occupancy_snapshot.json"


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


class MappingServiceServicer(tracking_pb2_grpc.MappingServiceServicer):
    def __init__(self, scan_manager, maps_root_dir: str, activity_monitor=None):
        self.scan_manager = scan_manager
        self.maps_root_dir = maps_root_dir
        self.activity_monitor = activity_monitor
        # Visualization-only caches for beacon_preview.py — see UpdateMapping.
        # Guiding's destination world point is cached (not recomputed) since
        # resolving it runs GroundingDINO (session.resolve_landmark) — never
        # want a debug circle to trigger that on every grid update alongside
        # the client's own real FindLandmark calls. Keyed by
        # (location_id, destination query).
        self._resolved_destinations: dict = {}
        # Walking's open-direction point, recomputed only when the grid
        # actually changes (same cost-avoidance reasoning, though cheaper);
        # keyed by location_id, reused on non-grid-updated iterations so the
        # preview doesn't just disappear between grid updates.
        self._last_open_direction: dict = {}
        # (ix_lo, iz_lo, width, height) as of the last FULL grid sent per
        # location_id — see UpdateMapping's full-vs-delta decision. Absent
        # entry means "never sent a full grid this stream", forcing one.
        self._last_full_bounds: dict = {}

    # ── persistence ──────────────────────────────────────────────────────────

    def _snapshot_path(self, location_id: str) -> str:
        return os.path.join(self.maps_root_dir, location_id, SNAPSHOT_FILENAME)

    def _load_snapshot(self, location_id: str) -> "dict | None":
        path = self._snapshot_path(location_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[MappingService] Failed to load snapshot for '{location_id}': {e}")
            return None

    def _save_snapshot(self, location_id: str, grid_dict: dict, ground_y: float, landmarks) -> None:
        if grid_dict is None:
            return
        path = self._snapshot_path(location_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "ground_y": ground_y,
            "grid": grid_dict,
            "landmarks": [
                {"name": lm.name, "x": lm.x, "z": lm.z, "confidence": lm.confidence}
                for lm in landmarks
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"[MappingService] Saved snapshot for '{location_id}': "
              f"{grid_dict['width']}x{grid_dict['height']} cells, {len(landmarks)} landmarks.")

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
        peer = context.peer()

        try:
            for chunk in request_iterator:
                chunks_received += 1
                if stream is None:
                    location_id = chunk.location_id
                    if not location_id:
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("location_id required on the first MappingChunk")
                        return
                    session_mode = chunk.session_mode  # only meaningful on this, the first chunk
                    walking_lite = session_mode != tracking_pb2.SCAN
                    print(f"[MappingService] UpdateMapping stream OPENED <- {peer} location_id='{location_id}' "
                          f"session_mode={tracking_pb2.SessionMode.Name(session_mode)}")
                    stream = StreamingScanSession(
                        self.scan_manager, location_id, pose_src="RTAB-Map", walking_lite=walking_lite,
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
                    self._last_open_direction.pop(location_id, None)
                    saved = self._load_snapshot(location_id)
                    if saved is not None:
                        stream.session.occupancy_map.seed_from_summary(saved["grid"], saved["ground_y"])
                        print(f"[MappingService] Re-seeded from existing snapshot for '{location_id}'")
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
                if not session.last_frame_poses:
                    continue
                pose_proto = _pose_to_proto(session.last_frame_poses[-1])

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
                      f"(chunks_received={chunks_received})")

                # Visualization-only "where does the HRTF beacon point" reconstruction —
                # see beacon_preview.py's module docstring. Never affects the real
                # audio (computed entirely on Android); purely feeds server_gui.py's
                # dashboard circles. Recomputed only on a full_resync (grid_dict is
                # only built then now — see the full-vs-delta split above), reusing
                # the cached point on delta/pose-only ticks in between: this is a
                # debug preview, not the real navigation signal, so refreshing it
                # less often than every tick is an acceptable tradeoff rather than
                # paying for a second extract_full_grid() call purely to keep it
                # maximally fresh.
                beacon_world_xz = None
                mode_snap = self.activity_monitor.snapshot() if self.activity_monitor is not None else {}
                client_mode = mode_snap.get("client_mode", "")
                if client_mode == "walking":
                    if full_resync and grid_dict is not None:
                        beacon_world_xz = find_most_open_direction_world_point(
                            session.last_frame_poses[-1], grid_dict
                        )
                        self._last_open_direction[location_id] = beacon_world_xz
                    else:
                        beacon_world_xz = self._last_open_direction.get(location_id)
                elif client_mode == "guiding":
                    destination = mode_snap.get("client_mode_target", "")
                    beacon_world_xz = self._resolved_destinations.get((location_id, destination))

                beacon_pixel = None
                if beacon_world_xz is not None:
                    beacon_pixel = project_world_point_to_pixel(
                        session.last_frame_poses[-1], beacon_world_xz,
                        session.occupancy_map._ground_y or 0.0,
                        frame_rgb.shape[1], frame_rgb.shape[0],
                    )

                if self.activity_monitor is not None:
                    self.activity_monitor.record_mapping(
                        f"UpdateMapping location='{location_id}' pose=({pose_proto.x:.2f},{pose_proto.z:.2f}) "
                        f"grid_updated={grid_updated}",
                        op="UpdateMapping", frame_rgb=frame_rgb, location_id=location_id,
                        pose_x=pose_proto.x, pose_z=pose_proto.z, grid_updated=grid_updated,
                        confidence=session.last_batch_confidence,
                        landmark_count=len(session._raw_landmarks),
                        # Live reference, not a snapshot — server_gui.py renders it on
                        # demand at poll time (occupancy_map.py's own render_plotly()/
                        # render_confidence_plotly(), no point-cloud/voxel rendering
                        # here, that's scan_gui.py's separate, heavier debug tool).
                        occupancy_map=session.occupancy_map,
                        beacon_world_xz=beacon_world_xz,
                        beacon_pixel=beacon_pixel,
                        # RTAB-Map's own per-batch tracking-lost count (see
                        # process_frames_batch's Step 2) — server_gui.py
                        # surfaces this directly so pose freezing is visible
                        # at a glance instead of only in console spam.
                        rtabmap_lost=session.last_rtabmap_lost,
                        rtabmap_total=session.last_rtabmap_total,
                    )

                yield tracking_pb2.MappingUpdate(
                    pose=pose_proto,
                    grid=grid_proto,
                    grid_updated=grid_updated,
                    landmarks=_landmarks_to_proto(session._raw_landmarks),
                    confidence=session.last_batch_confidence,
                    grid_delta=delta_proto,
                    full_resync=full_resync,
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
                    landmarks = stream.session.finalize_landmarks_flat()
                    grid_dict = stream.session.occupancy_map.extract_full_grid()
                    self._save_snapshot(
                        location_id, grid_dict, stream.session.occupancy_map._ground_y, landmarks
                    )
                except Exception:
                    traceback.print_exc()

    def GetMapSnapshot(self, request, context):
        print(f"[MappingService] GetMapSnapshot <- {context.peer()} location_id='{request.location_id}'")
        saved = self._load_snapshot(request.location_id)
        if saved is None:
            return tracking_pb2.MapSnapshot(found=False)
        landmarks = [
            tracking_pb2.Landmark(name=lm["name"], x=lm["x"], z=lm["z"], confidence=lm["confidence"])
            for lm in saved.get("landmarks", [])
        ]
        return tracking_pb2.MapSnapshot(
            found=True,
            grid=_grid_dict_to_proto(saved["grid"]),
            landmarks=landmarks,
        )

    def _find_in_snapshot(self, location_id: str, query: str):
        """Fallback for FindLandmark when the live session has nothing (or
        no live session exists at all) — walking/guiding sessions never
        populate their own frame store (walking_lite skips VLM tagging
        entirely, see scan_session.py), so a destination is only findable
        this way unless an earlier SCAN of this location already resolved
        it into the persisted occupancy_snapshot.json. Same case-
        insensitive substring, either-direction match resolve_landmark()
        itself uses. Returns None if no snapshot exists or nothing matches."""
        saved = self._load_snapshot(location_id)
        if saved is None:
            return None
        query_norm = query.strip().lower()
        for lm in saved.get("landmarks", []):
            name_norm = lm["name"].strip().lower()
            if query_norm in name_norm or name_norm in query_norm:
                return lm
        return None

    def FindLandmark(self, request, context):
        """Deferred landmark lookup — GroundingDINO never runs proactively
        during scanning any more (see scan_session.py's StoredFrame/
        resolve_landmark docstrings); this is the only place it runs, on
        demand, against the given location's in-memory frame store. Uses
        scan_manager.get() (read-only — does NOT create a session), matching
        the convention scan_gui.py already uses for read-only lookups.
        Falls back to the persisted snapshot's landmark list (_find_in_
        snapshot) when the live session has no answer — the only source of
        landmarks at all for a walking/guiding session, which never builds
        its own frame store (see CLAUDE.md's walking-mode redesign note)."""
        print(f"[MappingService] FindLandmark <- {context.peer()} "
              f"location_id='{request.location_id}' query='{request.query}'")
        session = self.scan_manager.get(request.location_id)
        lm = session.resolve_landmark(request.query) if session is not None else None
        if lm is None:
            saved_lm = self._find_in_snapshot(request.location_id, request.query)
            if saved_lm is not None:
                print(f"[MappingService] FindLandmark -> '{saved_lm['name']}' "
                      f"(from persisted snapshot, not live session)")
                self._resolved_destinations[(request.location_id, request.query)] = (
                    saved_lm["x"], saved_lm["z"],
                )
                if self.activity_monitor is not None:
                    self.activity_monitor.record_mapping(
                        f"FindLandmark query='{request.query}' -> '{saved_lm['name']}' (snapshot)",
                        op="FindLandmark", location_id=request.location_id, query=request.query, found=True,
                        matched_label=saved_lm["name"], x=saved_lm["x"], z=saved_lm["z"],
                        confidence=saved_lm["confidence"],
                    )
                return tracking_pb2.FindLandmarkResponse(
                    found=True, x=saved_lm["x"], z=saved_lm["z"],
                    confidence=saved_lm["confidence"], matched_label=saved_lm["name"],
                )
            print(f"[MappingService] FindLandmark: '{request.query}' not resolved")
            if self.activity_monitor is not None:
                self.activity_monitor.record_mapping(
                    f"FindLandmark query='{request.query}' -> not found",
                    op="FindLandmark", location_id=request.location_id, query=request.query, found=False,
                )
            return tracking_pb2.FindLandmarkResponse(found=False)
        # Cache for beacon_preview's guiding-mode dashboard circle (UpdateMapping) —
        # never re-resolved from there, only reused: resolve_landmark() runs
        # GroundingDINO, and the dashboard must not trigger that on every grid
        # update alongside the client's own real FindLandmark calls.
        self._resolved_destinations[(request.location_id, request.query)] = (lm.x, lm.z)
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

    def ListMappedLocations(self, request, context):
        print(f"[MappingService] ListMappedLocations <- {context.peer()}")
        if not os.path.isdir(self.maps_root_dir):
            return tracking_pb2.ListMapsResponse(location_ids=[])
        location_ids = [
            name for name in sorted(os.listdir(self.maps_root_dir))
            if os.path.exists(os.path.join(self.maps_root_dir, name, SNAPSHOT_FILENAME))
        ]
        return tracking_pb2.ListMapsResponse(location_ids=location_ids)
