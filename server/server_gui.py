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

import time
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go


_TAB_BY_CATEGORY = {
    "tracking": "tab_tracking",
    "perception": "tab_perception",
    "mapping": "tab_mapping",
}

_TAB_BY_CLIENT_MODE = {
    "tracking": "tab_tracking",
    "guiding": "tab_mapping",
    "walking": "tab_mapping",
    "scanning": "tab_mapping",
}


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


def _annotate_mapping(snap: dict) -> Optional[np.ndarray]:
    frame_rgb = snap.get("frame_rgb")
    if frame_rgb is None:
        return None
    if snap.get("op") != "UpdateMapping":
        return frame_rgb
    vis = frame_rgb.copy()
    text = f"pose=({snap.get('pose_x', 0):.2f},{snap.get('pose_z', 0):.2f}) conf={snap.get('confidence', 0):.2f}"
    cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2, cv2.LINE_AA)
    rtabmap_lost = snap.get("rtabmap_lost", 0)
    if rtabmap_lost:
        # frame_rgb is RGB order (see _decode_image_rgb) — (255,0,0) is red
        # here, matching the green/magenta text above which are also RGB.
        cv2.putText(
            vis, f"RTAB-Map TRACKING LOST ({rtabmap_lost}/{snap.get('rtabmap_total', 0)})",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA,
        )
    beacon_pixel = snap.get("beacon_pixel")
    if beacon_pixel is not None:
        cv2.circle(vis, beacon_pixel, 14, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, "HRTF", (beacon_pixel[0] + 16, beacon_pixel[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    return vis


def _render_occupancy(snap: dict):
    """Height-heatmap occupancy grid, reusing occupancy_map.py's own
    render_plotly() directly — no reimplementation of its classification/
    colorscale logic. Deliberately no point-cloud/voxel view here (that
    stays scan_gui.py's separate, heavier offline debug tool — see
    CLAUDE.md's server_gui.py note); this is occupancy-grid-only, matching
    what the client actually navigates/plays HRTF audio against.
    Wrapped in try/except: occupancy_map is a live object reference,
    mutated concurrently by the gRPC streaming thread while this renders on
    the Gradio polling thread — a render racing a mutation should degrade
    (skip this tick, try again next poll) rather than crash the dashboard.
    Also overlays beacon_world_xz (beacon_preview.py's server-side,
    visualization-only reconstruction of where the HRTF beacon points —
    the real audio direction is computed entirely on Android) as a magenta
    circle marker, same color/meaning as the frame-view circle above."""
    occ_map = snap.get("occupancy_map")
    if occ_map is None:
        return None
    try:
        fig = occ_map.render_plotly()
        beacon_xz = snap.get("beacon_world_xz")
        if beacon_xz is not None:
            fig.add_trace(go.Scatter(
                x=[beacon_xz[0]], y=[beacon_xz[1]], mode="markers", name="HRTF",
                marker=dict(size=16, color="magenta", symbol="circle-open", line=dict(width=3)),
            ))
        return fig
    except Exception:
        return None


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

        return (
            gr.update(selected=tab_id),
            _client_mode_text(snap),
            _annotate_tracking(snap["tracking"]),
            _tracking_status(snap["tracking"]),
            _annotate_perception(snap["perception"]),
            _perception_status(snap["perception"]),
            _annotate_mapping(snap["mapping"]),
            _mapping_status(snap["mapping"]),
            _render_occupancy(snap["mapping"]),
            _log_html(snap["log"]),
        )

    with gr.Blocks(title="Vision Assistant — Server Monitor") as app:
        gr.Markdown("## Vision Assistant — Server Monitor")
        ui_client_mode = gr.Textbox(label="", show_label=False, interactive=False)

        ui_tabs = gr.Tabs(selected="tab_log")
        with ui_tabs:
            with gr.Tab("Tracking (TrackingService)", id="tab_tracking"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ui_track_image = gr.Image(label="Last frame (annotated)", type="numpy")
                    with gr.Column(scale=1):
                        ui_track_status = gr.Textbox(label="Detail", lines=8, interactive=False)

            with gr.Tab("Perception (walking / OCR-adjacent)", id="tab_perception"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ui_perc_image = gr.Image(label="Last frame (annotated)", type="numpy")
                    with gr.Column(scale=1):
                        ui_perc_status = gr.Textbox(label="Detail", lines=10, interactive=False)

            with gr.Tab("Mapping (guiding / walking / scanning)", id="tab_mapping"):
                gr.Markdown(
                    "Occupancy grid only (no point cloud / voxel / confidence view — "
                    "that stays scan_gui.py's separate, heavier offline debug tool)."
                )
                with gr.Row():
                    ui_map_image = gr.Image(label="Last mapping frame", type="numpy")
                    ui_occupancy_plot = gr.Plot(label="Occupancy Map (height)")
                ui_map_status = gr.Textbox(label="Detail", lines=6, interactive=False)

            with gr.Tab("Activity Log", id="tab_log"):
                ui_log = gr.HTML()

        timer = gr.Timer(value=0.5)
        timer.tick(
            fn=_poll,
            inputs=[],
            outputs=[
                ui_tabs,
                ui_client_mode,
                ui_track_image,
                ui_track_status,
                ui_perc_image,
                ui_perc_status,
                ui_map_image,
                ui_map_status,
                ui_occupancy_plot,
                ui_log,
            ],
        )

    return app
