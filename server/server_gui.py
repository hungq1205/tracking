from __future__ import annotations

import time
from typing import Optional

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_tab_id(mode: str) -> str:
    return {
        "reading": "tab_reading",
        "guiding": "tab_nav",
        "tracking": "tab_tracking",
    }.get(mode, "tab_info")


def _annotate_tracking(frame_bgr: np.ndarray, detection: Optional[dict]) -> np.ndarray:
    vis = frame_bgr.copy()
    if detection and detection.get("score", 0) > 0.3:
        x1, y1, x2, y2 = map(int, detection["box_xyxy"])
        label = f"{detection.get('target', '?')}: {detection['score']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            vis, label, (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA,
        )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _reading_html(state) -> str:
    buf = state.reading_buffer
    words = len(buf.split()) if buf else 0
    summaries = state.page_summaries or []
    direction = state.reading_direction or "ltr"
    label = state.reading_label or ""

    summaries_html = ""
    if summaries:
        items = "".join(f"<li style='color:#ccc;margin:3px 0'>{s}</li>" for s in summaries)
        summaries_html = f"<ul style='margin:8px 0;padding-left:18px'>{items}</ul>"

    return (
        f"<div style='font-family:monospace;padding:10px;background:#111;border-radius:6px'>"
        f"<p style='color:#4af;margin:0 0 6px'><b>READING MODE</b>"
        f"{'  ·  <b>' + label + '</b>' if label else ''}</p>"
        f"<p style='color:#aaa;margin:0 0 4px'>"
        f"Direction: {direction}  ·  Buffer: {len(buf)} chars / {words} words  ·  "
        f"Pages scanned: {len(summaries)}</p>"
        f"{summaries_html}"
        f"</div>"
    )


def _injections_html(session) -> str:
    entries = list(session.context_injections)[-20:]
    if not entries:
        return "<div style='color:#555;padding:6px;font-family:monospace'>No context injections yet.</div>"
    rows = []
    for e in reversed(entries):
        ts = time.strftime("%H:%M:%S", time.localtime(e["at"]))
        text = e["text"].replace("<", "&lt;").replace(">", "&gt;")
        is_tool = text.startswith("[TOOL]")
        is_sys = text.startswith("[SYSTEM]")
        color = "#4af" if is_tool else ("#fa4" if is_sys else "#aaa")
        rows.append(
            f"<tr>"
            f"<td style='color:#555;padding:2px 8px;font-size:0.75em;white-space:nowrap'>{ts}</td>"
            f"<td style='color:{color};padding:2px 6px;font-size:0.82em;word-break:break-word'>{text}</td>"
            f"</tr>"
        )
    return (
        "<div style='background:#0d0d0d;border-radius:4px;padding:4px;"
        "max-height:220px;overflow-y:auto;font-family:monospace'>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _build_nav_map(session) -> Optional[plt.Figure]:
    if session is None:
        return None
    state = session.state
    rp = getattr(session, "_route_planner", None)
    if rp is None or not getattr(rp, "zones", None):
        return None

    dest_label = (state.nav_destination or "").lower()
    route_set = {z.lower() for z in (state.nav_route or [])}
    current_wp = ""
    if state.nav_route and state.nav_route_idx < len(state.nav_route):
        current_wp = state.nav_route[state.nav_route_idx].lower()

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    for zone in rp.zones:
        x0, z0 = float(zone.bbox_min[0]), float(zone.bbox_min[2])
        x1, z1 = float(zone.bbox_max[0]), float(zone.bbox_max[2])
        lbl_lower = zone.label.lower()

        if lbl_lower == dest_label:
            fc, ec, lw = "#aa2222", "#ff6666", 2.0
        elif lbl_lower == current_wp:
            fc, ec, lw = "#224488", "#44aaff", 2.0
        elif lbl_lower in route_set:
            fc, ec, lw = "#1a3355", "#336699", 1.5
        else:
            fc, ec, lw = "#1e1e3a", "#444466", 1.0

        rect = mpatches.FancyBboxPatch(
            (x0, z0), x1 - x0, z1 - z0,
            boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.85,
        )
        ax.add_patch(rect)
        cx, cz = float(zone.centroid[0]), float(zone.centroid[2])
        ax.text(cx, cz, zone.label, ha="center", va="center",
                color="#ddd", fontsize=7, fontweight="bold", zorder=5)

    if state.nav_last_position is not None:
        px, _, pz = state.nav_last_position
        ax.plot(float(px), float(pz), "o", color="#ff4444", markersize=9, zorder=10)
        ax.annotate("You", (float(px), float(pz)),
                    xytext=(6, 6), textcoords="offset points",
                    color="#ff8888", fontsize=8)

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.18)
    title = f"→ {state.nav_destination}" if state.nav_destination else "Navigation Map"
    ax.set_title(title, color="#bbb", fontsize=9, pad=6)
    ax.set_xlabel("X (m)", color="#666", fontsize=8)
    ax.set_ylabel("Z (m)", color="#666", fontsize=8)
    plt.tight_layout(pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_ui(gui_frame_queue, _vlm_unused, orchestrator=None, servicer=None) -> gr.Blocks:
    _prev_nav_fig: list = [None]

    def _nav_map_fig(session):
        if _prev_nav_fig[0] is not None:
            plt.close(_prev_nav_fig[0])
            _prev_nav_fig[0] = None
        fig = _build_nav_map(session)
        _prev_nav_fig[0] = fig
        return fig

    def _poll():
        session = servicer.current_session if servicer is not None else None
        state = session.state if session is not None else None
        frame_bgr = servicer.latest_frame if servicer is not None else None
        frame_rgb = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_bgr is not None else None
        )

        mode = state.mode if state else "idle"
        tab_id = _current_tab_id(mode)

        # --- Conversation log for chatbot ---
        chat_pairs = []
        if session:
            for entry in session.conversation_log:
                if entry["role"] in ("user", "assistant"):
                    chat_pairs.append({"role": entry["role"], "content": entry["text"]})

        # --- Context injections ---
        injections_html = (
            _injections_html(session) if session
            else "<div style='color:#555;padding:6px'>No active session.</div>"
        )

        # --- Tracking tab ---
        det = state.last_detection if state else None
        track_frame = (
            _annotate_tracking(frame_bgr, det) if (frame_bgr is not None and det)
            else frame_rgb
        )
        if state and state.mode == "tracking":
            track_status = f"Target: {state.tracking_target}"
            if det:
                track_status += f"  Score: {det.get('score', 0):.2f}"
        else:
            track_status = "Tracking idle"

        # --- Reading tab ---
        reading_html = (
            _reading_html(state) if (state and state.mode == "reading")
            else "<div style='color:#666;padding:10px'>Reading mode not active.</div>"
        )

        # --- Guiding tab ---
        guide_dino = (
            cv2.cvtColor(session.last_guide_dino_frame, cv2.COLOR_BGR2RGB)
            if session and session.last_guide_dino_frame is not None
            else frame_rgb
        )
        guide_depth = (
            session.last_guide_depth_image
            if session and session.last_guide_depth_image is not None
            else None
        )
        guide_map = _nav_map_fig(session)

        guide_status = "Guiding idle"
        if state and state.mode == "guiding":
            if state.nav_route:
                idx = state.nav_route_idx
                wp = state.nav_route[idx] if idx < len(state.nav_route) else "—"
                guide_status = (
                    f"→ {state.nav_destination}  |  Next: {wp}  "
                    f"|  Step {idx + 1}/{len(state.nav_route)}"
                )
            else:
                cache = state.walking_obstacle_cache or []
                now_t = time.time()
                active = [
                    f"{e['label']} ({e['expires_at'] - now_t:.1f}s)"
                    for e in cache if e["expires_at"] > now_t
                ]
                guide_status = "Free-walk" + (f"  |  Known: {', '.join(active)}" if active else "")

        guide_obstacle = ""
        if session:
            for inj in reversed(session.context_injections):
                if "[SYSTEM] Obstacle" in inj["text"] and time.time() - inj["at"] < 20.0:
                    guide_obstacle = inj["text"].replace("[SYSTEM] ", "")
                    break

        return (
            gr.update(selected=tab_id),  # ui_tabs
            track_frame,                  # ui_track_image
            track_status,                 # ui_track_status
            reading_html,                 # ui_read_html
            guide_dino,                   # ui_guide_dino
            guide_depth,                  # ui_guide_depth
            guide_map,                    # ui_guide_map
            guide_status,                 # ui_guide_status
            guide_obstacle,               # ui_guide_obstacle
            frame_rgb,                    # ui_info_frame
            chat_pairs,                   # ui_info_chatbot
            injections_html,              # ui_injections
        )

    # ---------------------------------------------------------------- layout
    with gr.Blocks(title="Vision Assistant Monitor") as app:
        gr.Markdown("## Vision Assistant — Server Monitor")
        btn_reset = gr.Button("Clear Logs", variant="stop", size="sm")

        ui_tabs = gr.Tabs(selected="tab_info")
        with ui_tabs:

            with gr.Tab("Chat", id="tab_info"):
                ui_info_frame = gr.Image(label="Latest Frame", type="numpy")
                ui_info_chatbot = gr.Chatbot(
                    label="Conversation (Gemini Live transcriptions)", height=350,
                )

            with gr.Tab("Tracking", id="tab_tracking"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ui_track_image = gr.Image(
                            label="Live Frame (annotated)", type="numpy",
                        )
                    with gr.Column(scale=1):
                        ui_track_status = gr.Textbox(
                            label="Detection Info", lines=5, interactive=False,
                        )

            with gr.Tab("Reading", id="tab_reading"):
                ui_read_html = gr.HTML()

            with gr.Tab("Guiding", id="tab_nav"):
                with gr.Row():
                    ui_guide_dino = gr.Image(label="Detection (DINO boxes)", type="numpy")
                    ui_guide_depth = gr.Image(label="Depth Map (DA3)", type="numpy")
                with gr.Row():
                    ui_guide_det_interval = gr.Slider(
                        0.5, 10.0, value=2.0, step=0.5, label="Detection Interval (s)",
                    )
                    ui_guide_rect_frac = gr.Slider(
                        0.1, 1.0, value=0.5, step=0.05, label="Path Width (fraction of frame)",
                    )
                    ui_guide_depth_thr = gr.Slider(
                        0.5, 10.0, value=3.0, step=0.5, label="Depth Threshold (m)",
                    )
                ui_guide_map = gr.Plot(label="Route Map")
                ui_guide_status = gr.Textbox(
                    label="Guiding Status", lines=2, interactive=False,
                )
                ui_guide_obstacle = gr.Textbox(
                    label="Obstacle Alert", lines=1, interactive=False,
                )

        with gr.Accordion("Context Injections ([SYSTEM] events + tool calls)", open=True):
            ui_injections = gr.HTML()

        def _set_walk_cfg(attr, value):
            cfg = getattr(getattr(servicer, "tools_bundle", None), "walking_config", None) if servicer else None
            if cfg is not None:
                setattr(cfg, attr, value)

        ui_guide_det_interval.change(
            fn=lambda v: _set_walk_cfg("detection_interval", v),
            inputs=[ui_guide_det_interval],
        )
        ui_guide_rect_frac.change(
            fn=lambda v: _set_walk_cfg("inner_rect_fraction", v),
            inputs=[ui_guide_rect_frac],
        )
        ui_guide_depth_thr.change(
            fn=lambda v: _set_walk_cfg("depth_threshold_m", v),
            inputs=[ui_guide_depth_thr],
        )

        timer = gr.Timer(value=0.5)
        timer.tick(
            fn=_poll,
            inputs=[],
            outputs=[
                ui_tabs,
                ui_track_image,
                ui_track_status,
                ui_read_html,
                ui_guide_dino,
                ui_guide_depth,
                ui_guide_map,
                ui_guide_status,
                ui_guide_obstacle,
                ui_info_frame,
                ui_info_chatbot,
                ui_injections,
            ],
        )

        def _reset():
            if servicer and servicer.current_session:
                servicer.current_session.conversation_log.clear()
                servicer.current_session.context_injections.clear()

        btn_reset.click(fn=_reset, inputs=[], outputs=[])

    return app
