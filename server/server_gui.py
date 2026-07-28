"""
Server Monitor — live view of whatever the Android client's RPCs actually
send, built directly around ActivityMonitor (services/activity_monitor.py).

There's no server-side "mode" any more (no LiveSessionState — orchestration
moved on-device, see CLAUDE.md's "Client-Orchestrated Live Session"
section), so this dashboard doesn't reconstruct state by guessing. Two
signals feed it: (1) StatusService.ReportMode — the client explicitly tells
the server which mode it's in, once per transition (ActivityMonitor.
client_mode) — this is authoritative and preferred for tab selection when
present; (2) each servicer also records a snapshot into ActivityMonitor as
real RPCs land regardless (TrackingService -> tracking bucket,
PerceptionService -> perception bucket, MappingService -> mapping bucket),
which is what actually supplies the frames/results shown in each tab, and
is the tab-selection fallback for older clients that predate ReportMode.
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation


_TAB_BY_CATEGORY = {
    "tracking": "tab_tracking",
    "perception": "tab_perception",
    "mapping": "tab_mapping",
}

_TAB_BY_CLIENT_MODE = {
    "tracking": "tab_tracking",
    "guiding": "tab_mapping",
    # Walking rejoined MappingService (RTAB-Map pose + a live occupancy grid,
    # same as guiding — see CLAUDE.md's "Grid-planned walking route" note),
    # so its activity lands in the mapping bucket again, same as guiding/
    # scanning. This used to point at tab_perception, back when walking's
    # only server traffic was AnalyzeFrame(TRAVERSABILITY) — a real,
    # previously-documented gap (the dashboard tab never matched what the
    # client was actually doing), now closed.
    "walking": "tab_mapping",
    "scanning": "tab_mapping",
}


_frame_idx = 0
_frames_dir = "frames"


def _bgr_to_rgb(frame_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr is not None else None


def _ago(at: float) -> str:
    if not at:
        return "never"
    dt = time.time() - at
    return f"{dt:.1f}s ago" if dt < 60 else time.strftime("%H:%M:%S", time.localtime(at))


def _annotate_tracking(snap: dict) -> Optional[np.ndarray]:
    frame_bgr = snap.get("frame_bgr")
    if frame_bgr is None:
        return None
    vis = frame_bgr.copy()
    box = snap.get("box_xyxy")
    if box and len(box) == 4:
        x1, y1, x2, y2 = map(int, box)
        label = snap.get("prompt", "") or "target"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis, label, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)
    return _bgr_to_rgb(vis)


def _tracking_status(snap: dict) -> str:
    if not snap:
        return "No TrackingService activity yet."
    lines = [f"op: {snap.get('op', '?')}", f"at: {_ago(snap.get('at', 0))}"]
    if "prompt" in snap:
        lines.append(f"prompt: '{snap['prompt']}'")
    if "score" in snap:
        lines.append(f"score: {snap['score']:.3f}")
    if "box_xyxy" in snap:
        lines.append(f"box: {[round(v) for v in snap['box_xyxy']]}")
    if "embedding_dim" in snap:
        lines.append(f"embedding dim: {snap['embedding_dim']}")
    return "\n".join(lines)


def _annotate_perception(snap: dict) -> Optional[np.ndarray]:
    frame_bgr = snap.get("frame_bgr")
    if frame_bgr is None:
        return None
    vis = frame_bgr.copy()
    for d in snap.get("detections") or []:
        x1, y1, x2, y2 = map(int, d["box_xyxy"])
        label = f"{d.get('label', '?')} {d.get('score', 0):.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 170, 0), 2)
        cv2.putText(vis, label, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 170, 0), 2, cv2.LINE_AA)
    obstacle = snap.get("obstacle")
    if obstacle and obstacle.get("detected"):
        h, w = vis.shape[:2]
        text = f"OBSTACLE ~{obstacle['distance_m']:.2f}m"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return _bgr_to_rgb(vis)


def _perception_status(snap: dict) -> str:
    if not snap:
        return "No PerceptionService activity yet."
    lines = [f"op: {snap.get('op', '?')}", f"at: {_ago(snap.get('at', 0))}"]
    if "ops" in snap:
        lines.append(f"requested ops: {snap['ops']}")
    if snap.get("prompt"):
        lines.append(f"prompt: '{snap['prompt']}'")
    if "detections" in snap:
        lines.append(f"detections: {len(snap['detections'])}")
        for d in snap["detections"][:5]:
            lines.append(f"  · {d.get('label', '?')} score={d.get('score', 0):.2f} box={[round(v) for v in d['box_xyxy']]}")
    if snap.get("obstacle"):
        o = snap["obstacle"]
        lines.append(f"obstacle: detected={o.get('detected')} distance={o.get('distance_m', 0):.2f}m")
    if snap.get("op") in ("Synthesize", "Embed") and "text" in snap:
        lines.append(f"text: '{snap['text']}'")
    return "\n".join(lines)


def _project_world_to_pixel(pose_proto, ground_y: float, wx: float, wz: float, frame_w: int, frame_h: int):
    """Projects a world (x, ground_y, z) point into this pose's camera
    pixel space — the inverse of scan_session.py's own back-projection.
    Camera-local convention (X-right, Y-down, Z-forward) matches every pose
    this project produces, same as HrtfBeacon.kt's own docstring. Assumed
    intrinsics (fx=fy=0.8*max(w,h), cx=w/2, cy=h/2) are the same "no real
    calibration, guess a pinhole K" convention duplicated as `_estimate_K`
    across scan_session.py/feature_tracker.py/orb_novelty_gate.py/
    rtabmap_client.py — reproduced directly here rather than importing
    across the server/scan_server boundary, matching that established
    precedent. Returns None if intrinsics can't be computed (frame_w/h <= 0)
    or the point is behind the camera.

    The "behind camera" check is deliberately HORIZONTAL-ONLY (dot product
    against the camera's own floor-plane forward direction — same
    computation _pose_heading_rad() does server-side), not the full 3D
    camera-local Z. Using the full 3D Z would let an inaccurate `ground_y`
    (a single scalar estimate, not a per-point measurement) leak into
    whether a point is considered "visible" at all — a modest camera pitch
    combined with an off `ground_y` could flip the sign and silently blank
    the WHOLE overlay even though every point is genuinely in front of the
    user horizontally. `ground_y` is still used for vertical (v) pixel
    placement, where an inaccuracy just draws the overlay a bit high/low
    rather than making it vanish entirely."""
    if frame_w <= 0 or frame_h <= 0:
        return None
    quat = np.array([pose_proto.qx, pose_proto.qy, pose_proto.qz, pose_proto.qw])
    quat_norm = float(np.linalg.norm(quat))
    # A protobuf Pose that was never actually populated (e.g. a stale/
    # default-constructed message slipping through) reads qx=qy=qz=qw=0 —
    # a zero-norm "quaternion" that isn't a rotation at all. scipy's
    # from_quat() normalizes internally, so a near-zero input silently
    # divides by ~0 and returns NaN rather than raising — every point would
    # then look "behind the camera" from a `nan <= 1e-3` comparison (always
    # False in Python) actually falling through to `int(round(nan))`,
    # which DOES raise, mid-list-comprehension, potentially breaking the
    # whole dashboard tuple update for that tick. Guard explicitly instead
    # of relying on that to surface loudly.
    if quat_norm < 1e-6:
        print(f"[server_gui] _project_world_to_pixel: degenerate pose quaternion {quat.tolist()} "
              f"(norm={quat_norm:.4f}) — pose_proto likely unset/stale, skipping projection")
        return None
    cam_r = Rotation.from_quat(quat).as_matrix()
    forward_world = cam_r @ np.array([0.0, 0.0, 1.0])
    dx = wx - pose_proto.x
    dz = wz - pose_proto.z
    horizontal_forward_dot = dx * forward_world[0] + dz * forward_world[2]
    if horizontal_forward_dot <= 1e-3:
        return None
    d_world = np.array([dx, ground_y - pose_proto.y, dz])
    d_cam = cam_r.T @ d_world  # world -> camera-local
    # Visibility was already decided above (horizontal-only) — clamp rather
    # than reject here, so a `ground_y` inaccurate enough to push the full
    # 3D Z near/below zero degrades to "drawn near the edge" instead of
    # "silently dropped" for a point that IS genuinely in front of the user.
    z_for_projection = max(d_cam[2], 0.05)
    f = 0.8 * max(frame_w, frame_h)
    u = f * d_cam[0] / z_for_projection + frame_w / 2.0
    v = f * d_cam[1] / z_for_projection + frame_h / 2.0
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(u)), int(round(v))


def _annotate_mapping(snap: dict) -> Optional[np.ndarray]:
    frame_rgb = snap.get("frame_rgb")
    if frame_rgb is None:
        return None
    if snap.get("op") != "UpdateMapping":
        return frame_rgb
    vis = frame_rgb.copy()
    heading_rad = snap.get("heading_rad")
    heading_txt = f" heading={np.degrees(heading_rad):.0f}deg" if heading_rad is not None else ""
    text = f"pose=({snap.get('pose_x', 0):.2f},{snap.get('pose_z', 0):.2f}) conf={snap.get('confidence', 0):.2f}{heading_txt}"
    cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA)
    rtabmap_lost = snap.get("rtabmap_lost", 0)
    if rtabmap_lost:
        # frame_rgb is RGB order (see _decode_image_rgb) — (255,0,0) is red
        # here, matching the green text above which is also RGB.
        cv2.putText(
            vis, f"RTAB-Map TRACKING LOST ({rtabmap_lost}/{snap.get('rtabmap_total', 0)})",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA,
        )
    pose_proto = snap.get("pose_proto")
    planned_path = snap.get("planned_path") or []
    if pose_proto is not None and planned_path:
        h, w = vis.shape[:2]
        # dict.get(key, default) only falls back when the KEY is missing,
        # not when it's present but None — record_mapping() always passes
        # the `ground_y` key, so if occupancy_map._ground_y hadn't been
        # estimated yet at record time (no ground evidence seen so far
        # this session), this used to silently pass ground_y=None through
        # into arithmetic below instead of falling back to pose_proto.y.
        ground_y = snap.get("ground_y")
        if ground_y is None:
            ground_y = pose_proto.y
        # Only the user's own pose (bottom center, by construction of
        # _project_world_to_pixel) -> the first waypoint is drawn here —
        # the rest of the route is already visible on the occupancy map,
        # and projecting the full path onto a forward-facing camera frame
        # degenerates badly past the first joint (points behind/beside the
        # camera, or far down a corridor, don't read as a useful line).
        pixels = [
            _project_world_to_pixel(pose_proto, ground_y, wx, wz, w, h)
            for wx, wz in planned_path[:1]
        ]
        if planned_path and all(p is None for p in pixels):
            print(f"[server_gui] _annotate_mapping: ALL {len(pixels)} planned_path points "
                  f"projected as not-visible — pose=({pose_proto.x:.2f},{pose_proto.y:.2f},"
                  f"{pose_proto.z:.2f}) quat=({pose_proto.qx:.3f},{pose_proto.qy:.3f},"
                  f"{pose_proto.qz:.3f},{pose_proto.qw:.3f}) ground_y={ground_y:.2f} "
                  f"first_path_point={planned_path[0]}")
        # cyan — distinct from the green pose text / red tracking-lost
        # warning already drawn on this same overlay.
        color = (0, 255, 255)
        origin = (w // 2, h - 1)
        first = pixels[0] if pixels else None
        if first is not None:
            cv2.line(vis, origin, first, color, 2, cv2.LINE_AA)
            cv2.circle(vis, first, 4, color, -1, cv2.LINE_AA)

        # HRTF beacon target — same beacon_target_point() calculation
        # ToolDispatcher.kt's steerBeaconAlongPath() does on-device (project
        # onto the path, then move forward by a look-ahead distance),
        # replicated server-side purely for this display — see
        # CLAUDE.md's "Server-planned walking path" note. Magenta, a larger
        # ring, so it reads as "the sound source" distinct from the plain
        # cyan route dots.
        beacon_point = snap.get("beacon_point")
        if beacon_point is not None:
            bp = _project_world_to_pixel(pose_proto, ground_y, beacon_point[0], beacon_point[1], w, h)
            if bp is not None:
                cv2.circle(vis, bp, 10, (255, 0, 255), 2, cv2.LINE_AA)
    return vis


def _render_occupancy(snap: dict, record_is_checked: bool):
    """Height-heatmap occupancy grid, reusing occupancy_map.py's own
    render_plotly() directly — no reimplementation of its classification/
    colorscale logic. Deliberately no point-cloud/voxel view here (that
    stays scan_gui.py's separate, heavier offline debug tool — see
    CLAUDE.md's server_gui.py note); this is occupancy-grid-only, matching
    what the client actually navigates against.
    Overlays the current server-planned route via render_plotly()'s own
    route/route_confirmed params (same _overlay_route() scan_gui.py's Live
    Navigation Preview already uses) — route must include the start point
    (current pose) as its first element, per that function's own contract.
    Wrapped in try/except: occupancy_map is a live object reference,
    mutated concurrently by the gRPC streaming thread while this renders on
    the Gradio polling thread — a render racing a mutation should degrade
    (skip this tick, try again next poll) rather than crash the dashboard."""
    global _frame_idx
    occ_map = snap.get("occupancy_map")
    if occ_map is None:
        return None
    planned_path = snap.get("planned_path") or []
    route = [(snap.get("pose_x", 0.0), snap.get("pose_z", 0.0))] + list(planned_path) if planned_path else None
    try:
        fig = occ_map.render_plotly(route=route, route_confirmed=snap.get("path_confirmed"))
        # HRTF beacon target, in world space — same beacon_target_point()
        # calculation ToolDispatcher.kt's steerBeaconAlongPath() does
        # on-device, replicated server-side purely for this display (see
        # CLAUDE.md's "Server-planned walking path" note). Magenta,
        # matching the frame overlay's own beacon marker color.
        beacon_point = snap.get("beacon_point")
        if beacon_point is not None:
            fig.add_trace(go.Scatter(
                x=[beacon_point[0]], y=[beacon_point[1]],
                mode="markers", marker=dict(size=14, color="magenta", symbol="circle-open", line=dict(width=3)),
                name="HRTF beacon", hoverinfo="skip",
            ))
        # Which way the user is actually facing (world-frame yaw from
        # _pose_heading_rad(), same 0=+Z/positive-toward-+X convention
        # HrtfBeacon.kt's azimuth uses) — drawn as a short arrow from the
        # current position, so a mismatch between "where the route goes"
        # and "which way the user is actually pointed" is visible directly
        # on this plot instead of only inferable from console logs.
        heading_rad = snap.get("heading_rad")
        if heading_rad is not None:
            hx, hz = snap.get("pose_x", 0.0), snap.get("pose_z", 0.0)
            arrow_len = 0.6
            ax = hx + arrow_len * np.sin(heading_rad)
            az = hz + arrow_len * np.cos(heading_rad)
            fig.add_annotation(
                x=ax, y=az, ax=hx, ay=hz, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3,
                arrowcolor="yellow", text="",
            )
            # add_annotation() draws the arrow but doesn't add a legend
            # entry — an invisible marker at the arrow tip stands in for one
            # (same trick used wherever this codebase wants a legend label
            # for something that isn't itself a Scatter trace).
            fig.add_trace(go.Scatter(
                x=[ax], y=[az], mode="markers",
                marker=dict(size=0.1, color="yellow"),
                name="Facing direction", hoverinfo="skip",
            ))
        # Discovered landmarks (session._raw_landmarks, live during a SCAN —
        # see "Semantic mapper adapted to..." in CLAUDE.md) — white diamonds
        # with their name as hover text, distinct from the beacon/facing
        # markers above.
        landmarks = snap.get("landmarks") or []
        if landmarks:
            fig.add_trace(go.Scatter(
                x=[lm[0] for lm in landmarks], y=[lm[1] for lm in landmarks],
                mode="markers+text", marker=dict(size=10, color="white", symbol="diamond", line=dict(width=1, color="black")),
                text=[lm[2] for lm in landmarks], textposition="top center",
                name="Landmarks", hoverinfo="text",
            ))

        return fig
    except Exception:
        return None


def _render_beacon_polar(snap: dict):
    """Polar view of the HRTF beacon's current direction relative to the
    user's forward view — replaces the old beacon_preview.py world-point
    reconstruction (which modeled the beacon as a position; it's a pure
    steering angle now, see CLAUDE.md's "Local reactive HRTF obstacle-
    dodge" note). Forward = 12 o'clock, azimuth-right reads clockwise
    (matches HrtfBeacon.kt's convention). Lives on the Mapping tab now
    (moved off the old standalone Perception tab — see CLAUDE.md's
    "Grid-planned walking route" note): shown alongside the occupancy map
    since that's the grid GUIDING's and now WALKING's beacon both actually
    steer through.

    The marker is the ACTUAL final azimuth the client is playing —
    client-computed, reported via StatusService.ReportBeaconDirection since
    the server has no other way to know it (GUIDING: goal-biased,
    EMA-smoothed TraversabilityScorer output; WALKING: bearing to the
    client's current LocalPathPlanner waypoint along its grid-planned
    route). The bars are the last AnalyzeFrame(TRAVERSABILITY) fan
    (perception bucket) — for GUIDING this is literally what the marker was
    scored from; for WALKING the steering itself comes from the occupancy
    grid instead, so these bars are only the separate step-down/drop-off
    hazard-check fan (see CLAUDE.md's "Hazard warnings" note) — background
    context, not what picked the marker's angle."""
    trav = snap.get("perception", {}).get("traversability")
    if trav is None:
        return None
    clearances = trav["clearance_m"]
    n = len(clearances)
    angles = [trav["min_angle_deg"] + i * trav["angle_step_deg"] for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=clearances, theta=angles, width=[trav["angle_step_deg"]] * n,
        marker=dict(color=clearances, colorscale="Viridis", cmin=0, cmax=trav["max_range_m"]),
        name="clearance",
    ))
    muted = snap.get("beacon_muted", True)
    az = snap.get("beacon_azimuth_deg", 0.0)
    if snap.get("beacon_at", 0.0) > 0:
        fig.add_trace(go.Scatterpolar(
            r=[trav["max_range_m"] * 0.95], theta=[az], mode="markers+text",
            marker=dict(size=18, color=("#888" if muted else "magenta"), symbol="circle-open", line=dict(width=3)),
            text=["HRTF (muted)" if muted else "HRTF"], textposition="top center",
            name="beacon",
        ))
    fig.update_layout(
        polar=dict(
            angularaxis=dict(rotation=90, direction="clockwise"),
            radialaxis=dict(range=[0, trav["max_range_m"]]),
        ),
        showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=360,
    )
    return fig


def _beacon_status(snap: dict) -> str:
    if snap.get("beacon_at", 0.0) <= 0:
        return "No beacon direction reported yet."
    muted = snap.get("beacon_muted", True)
    if muted:
        return f"HRTF: muted (nothing safe to point at)  ·  reported {_ago(snap.get('beacon_at', 0))}"
    return f"HRTF azimuth: {snap.get('beacon_azimuth_deg', 0.0):.1f}°  ·  reported {_ago(snap.get('beacon_at', 0))}"


def _mapping_status(snap: dict) -> str:
    if not snap:
        return "No MappingService activity yet."
    lines = [f"op: {snap.get('op', '?')}", f"at: {_ago(snap.get('at', 0))}"]
    if "location_id" in snap:
        lines.append(f"location_id: '{snap['location_id']}'")
    if snap.get("op") == "UpdateMapping":
        lines.append(f"pose: ({snap.get('pose_x', 0):.2f}, {snap.get('pose_z', 0):.2f})")
        rtabmap_lost = snap.get("rtabmap_lost", 0)
        rtabmap_total = snap.get("rtabmap_total", 0)
        if rtabmap_lost:
            lines.append(f"⚠ RTAB-Map tracking: LOST {rtabmap_lost}/{rtabmap_total} frames this batch")
        elif rtabmap_total:
            lines.append(f"RTAB-Map tracking: OK ({rtabmap_total}/{rtabmap_total})")
        lines.append(f"grid_updated: {snap.get('grid_updated')}")
        lines.append(f"confidence: {snap.get('confidence', 0):.2f}")
        lines.append(f"landmarks so far: {snap.get('landmark_count', 0)}")
    elif snap.get("op") == "FindLandmark":
        lines.append(f"query: '{snap.get('query', '')}'")
        if snap.get("found"):
            lines.append(f"-> '{snap.get('matched_label')}' at ({snap.get('x', 0):.2f}, {snap.get('z', 0):.2f}) "
                          f"confidence={snap.get('confidence', 0):.2f}")
        else:
            lines.append("-> not found")
    return "\n".join(lines)


def _log_html(entries: list) -> str:
    if not entries:
        return "<div style='color:#555;padding:6px;font-family:monospace'>No activity yet.</div>"
    color_by_category = {"tracking": "#4af", "perception": "#fa4", "mapping": "#8f6"}
    rows = []
    for e in entries[:60]:
        ts = time.strftime("%H:%M:%S", time.localtime(e["at"]))
        text = str(e["text"]).replace("<", "&lt;").replace(">", "&gt;")
        color = color_by_category.get(e.get("category", ""), "#aaa")
        rows.append(
            f"<tr>"
            f"<td style='color:#555;padding:2px 8px;font-size:0.75em;white-space:nowrap'>{ts}</td>"
            f"<td style='color:{color};padding:2px 6px;font-size:0.78em;white-space:nowrap'>[{e.get('category', '?')}]</td>"
            f"<td style='color:#ccc;padding:2px 6px;font-size:0.82em;word-break:break-word'>{text}</td>"
            f"</tr>"
        )
    return (
        "<div style='background:#0d0d0d;border-radius:4px;padding:4px;"
        "max-height:420px;overflow-y:auto;font-family:monospace'>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _frames_to_video(frames_dir: str, output_path: str, fps: int = 10):
    """Crude but effective frames-to-video using cv2."""
    if not os.path.isdir(frames_dir):
        print(f"Frames directory '{frames_dir}' not found, nothing to do.")
        return "Frames directory not found."
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if not frame_files:
        print("No frames to generate video.")
        return "No frames found to generate video."

    first_frame_path = os.path.join(frames_dir, frame_files[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        video.write(cv2.imread(frame_path))

    video.release()
    print(f"Video saved to {output_path}")

    global _frame_idx
    _frame_idx = 0
    shutil.rmtree(frames_dir)
    print(f"Cleaned up {frames_dir}.")
    return f"Video saved to {output_path}. Frames directory cleared."


def _client_mode_text(snap: dict) -> str:
    mode = snap.get("client_mode", "")
    if not mode:
        return "Client mode: unknown (client hasn't called ReportMode yet — older build?)"
    target = snap.get("client_mode_target", "")
    text = f"Client mode: {mode}"
    if target:
        text += f"  ·  target/destination: '{target}'"
    text += f"  ·  reported {_ago(snap.get('client_mode_at', 0))}"
    return text


def create_ui(activity_monitor) -> gr.Blocks:
    def _poll():
        snap = activity_monitor.snapshot() if activity_monitor is not None else {
            "tracking": {}, "perception": {}, "mapping": {}, "last_category": "",
            "client_mode": "", "client_mode_target": "", "client_mode_at": 0.0, "log": [],
        }
        tab_id = (
            _TAB_BY_CLIENT_MODE.get(snap["client_mode"])
            or _TAB_BY_CATEGORY.get(snap["last_category"], "tab_log")
        )

        # Broad, deliberate try/except around every render call — found
        # investigating a real "dashboard freezes during a long WALKING
        # session" report: _render_occupancy already guarded itself this
        # way, but an uncaught exception from any OTHER render helper here
        # (e.g. _annotate_mapping) would abort this whole _poll() call.
        # Skipping a tick on a transient failure (stale-but-valid previous
        # values via gr.update()) is far better than one bad tick wedging
        # the dashboard until a reload — see trigger_mode="multiple" above
        # for the other half of this fix.
        try:
            mapping_image = _annotate_mapping(snap["mapping"])
        except Exception as e:
            print(f"[server_gui] _poll: _annotate_mapping failed: {e}")
            mapping_image = gr.update()
        try:
            occupancy_fig = _render_occupancy(snap["mapping"], False)  # Recording is handled by a separate event
        except Exception as e:
            print(f"[server_gui] _poll: _render_occupancy failed: {e}")
            occupancy_fig = gr.update()

        return (
            gr.update(selected=tab_id),
            _client_mode_text(snap),
            _annotate_tracking(snap["tracking"]),
            _tracking_status(snap["tracking"]),
            _annotate_perception(snap["perception"]),
            _perception_status(snap["perception"]),
            _render_beacon_polar(snap),
            _beacon_status(snap),
            mapping_image,
            _mapping_status(snap["mapping"]),
            occupancy_fig,
            _log_html(snap["log"]),
            snap,  # Pass the full snapshot to the state component
        )

    with gr.Blocks(title="Vision Assistant — Server Monitor") as app:
        gr.Markdown("## Vision Assistant — Server Monitor")
        ui_client_mode = gr.Textbox(label="", show_label=False, interactive=False)
        ui_snapshot_state = gr.State()

        ui_tabs = gr.Tabs(selected="tab_log")
        with ui_tabs:
            with gr.Tab("Tracking (TrackingService)", id="tab_tracking"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ui_track_image = gr.Image(label="Last frame (annotated)", type="numpy")
                    with gr.Column(scale=1):
                        ui_track_status = gr.Textbox(label="Detail", lines=8, interactive=False)

            with gr.Tab("Perception (on-demand queries)", id="tab_perception"):
                gr.Markdown(
                    "Ad-hoc AnalyzeFrame calls only — `run_detection`/`check_obstacle` tool "
                    "invocations. Walking/guiding's own HRTF beacon display moved to the "
                    "Mapping tab (below), since that's the grid both modes actually steer "
                    "through now — see CLAUDE.md's \"Grid-planned walking route\" note."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        ui_perc_image = gr.Image(label="Last frame (annotated)", type="numpy")
                    with gr.Column(scale=1):
                        ui_perc_status = gr.Textbox(label="Detail", lines=10, interactive=False)

            with gr.Tab("Mapping + Beacon (guiding / walking / scanning)", id="tab_mapping"):
                with gr.Row():
                    ui_map_image = gr.Image(label="Last mapping frame", type="numpy", height=420)
                    ui_occupancy_plot = gr.Plot(label="Occupancy Map (height)")
                with gr.Row():
                    record_checkbox = gr.Checkbox(label="Record occupancy map to frames/")

                    def handle_record_change(is_checked, snap):
                        if is_checked and snap.get("mapping"):
                            fig = _render_occupancy(snap["mapping"], is_checked)
                            if fig:
                                if not os.path.exists(_frames_dir):
                                    os.makedirs(_frames_dir)
                                fig.write_image(f"{_frames_dir}/{_frame_idx:05d}.png")
                                globals()["_frame_idx"] += 1

                    generate_button = gr.Button("Generate occupancy.mp4 and clear frames")
                video_status = gr.Textbox(label="Video Status", interactive=False, show_label=False)

                def generate_video_action():
                    return _frames_to_video(_frames_dir, "occupancy.mp4")

                generate_button.click(fn=generate_video_action, inputs=[], outputs=[video_status])
                # Beacon graph side by side with BOTH detail boxes (mapping +
                # beacon), not a separate full-width status row floating
                # between the two plots — requested directly by the user.
                with gr.Row():
                    with gr.Column(scale=2):
                        ui_beacon_plot = gr.Plot(label="Beacon direction (forward = up)")
                    with gr.Column(scale=1):
                        ui_map_status = gr.Textbox(label="Mapping detail", lines=6, interactive=False)
                        ui_beacon_status = gr.Textbox(label="Beacon detail", lines=4, interactive=False)

            with gr.Tab("Activity Log", id="tab_log"):
                ui_log = gr.HTML()

        timer = gr.Timer(value=0.5)
        timer.tick(
            fn=_poll,
            inputs=[],
            # trigger_mode="multiple" (default is "once", which per Gradio's
            # own docs "would not allow any submissions while an event is
            # pending") — found investigating a real report that the
            # dashboard would freeze during a long WALKING session and only
            # a browser reload (a fresh Timer) temporarily fixed it: a
            # single pathologically slow _poll() call (heavy render cost on
            # a large occupancy grid, or contention against the gRPC
            # servicer thread) would otherwise silently swallow every tick
            # queued behind it, wedging the dashboard on its last successful
            # frame instead of just skipping a beat.
            trigger_mode="multiple",
            outputs=[
                ui_tabs,
                ui_client_mode,
                ui_track_image,
                ui_track_status,
                ui_perc_image,
                ui_perc_status,
                ui_beacon_plot,
                ui_beacon_status,
                ui_map_image,
                ui_map_status,
                ui_occupancy_plot,
                ui_log,
                ui_snapshot_state,
            ],
        )
        
        ui_occupancy_plot.change(
            fn=handle_record_change,
            inputs=[record_checkbox, ui_snapshot_state],
            outputs=[]
        )

    return app
