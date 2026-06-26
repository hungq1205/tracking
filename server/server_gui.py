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
# Module-level helpers (pure functions, no GUI state)
# ---------------------------------------------------------------------------

def _current_tab_id(ctx) -> str:
    if ctx.reading_state != "idle":
        return "tab_reading"
    if ctx.nav_state != "idle":
        return "tab_nav"
    if ctx.active_agent == "tracking":
        return "tab_tracking"
    if ctx.active_agent == "info":
        return "tab_info"
    return "tab_info"


def _annotate_ocr_frame(frame_bgr: np.ndarray, blocks: list) -> np.ndarray:
    """Return an RGB numpy array with red OCR block boxes drawn."""
    vis = frame_bgr.copy()
    for block in blocks:
        x1, y1, x2, y2 = map(int, block["box"])
        score = block.get("score", 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 30, 220), 2)
        cv2.putText(
            vis, f"{score:.2f}", (x1, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 220), 1, cv2.LINE_AA,
        )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _ocr_blocks_html(blocks: list) -> str:
    """Format OCR block list as a compact HTML table."""
    if not blocks:
        return "<div style='color:#666;padding:6px;font-family:monospace'>No OCR blocks detected.</div>"
    rows = []
    for i, b in enumerate(blocks):
        score = b.get("score", 0)
        text = b.get("text", "").replace("<", "&lt;").replace(">", "&gt;")
        rows.append(
            f"<tr style='border-bottom:1px solid #2a2a2a'>"
            f"<td style='color:#888;padding:3px 8px;font-size:0.78em;white-space:nowrap'>{i+1}</td>"
            f"<td style='color:#4af;padding:3px 6px;font-size:0.78em;white-space:nowrap'>{score:.2f}</td>"
            f"<td style='color:#ddd;padding:3px 6px;font-size:0.85em'>{text}</td>"
            f"</tr>"
        )
    return (
        "<div style='background:#0d0d0d;border-radius:4px;padding:4px;"
        "max-height:260px;overflow-y:auto;font-family:monospace'>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr>"
        "<th style='color:#666;font-size:0.75em;padding:2px 8px;text-align:left'>#</th>"
        "<th style='color:#666;font-size:0.75em;padding:2px 6px;text-align:left'>conf</th>"
        "<th style='color:#666;font-size:0.75em;padding:2px 6px;text-align:left'>text</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _annotate_tracking(frame_bgr: np.ndarray, detection: Optional[dict]) -> np.ndarray:
    """Return an RGB numpy array with the target bounding box drawn."""
    vis = frame_bgr.copy()
    if detection and detection.get("score", 0) > 0.35:
        x1, y1, x2, y2 = map(int, detection["box_xyxy"])
        label = f"{detection.get('target', '?')}: {detection['score']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            vis, label, (x1, max(y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA,
        )
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def _reading_html(ctx) -> str:
    """Build styled HTML for the reading buffer."""
    sentences = ctx.read_sentences
    pos = ctx.read_position
    state = ctx.reading_state
    label = ctx.active_label or ""

    if not sentences:
        if state != "idle" and ctx.scan_buffer:
            count = len(ctx.scan_buffer)
            preview = ctx.scan_buffer[:300].replace("<", "&lt;").replace(">", "&gt;")
            ellipsis = "…" if count > 300 else ""
            return (
                f"<div style='font-family:monospace;padding:10px;background:#111;border-radius:6px'>"
                f"<p style='color:#aaa;margin:0 0 6px'>"
                f"<b style='color:#4af'>{state.upper()}</b>"
                f"{'  ·  label: <b>' + label + '</b>' if label else ''}"
                f"  ·  {count} chars</p>"
                f"<pre style='color:#ccc;white-space:pre-wrap;margin:0'>{preview}{ellipsis}</pre>"
                f"</div>"
            )
        return "<div style='color:#666;padding:10px'>No text scanned yet.</div>"

    total = len(sentences)
    # read_position advances AFTER speaking; last spoken = pos - 1
    current_idx = pos - 1 if state == "reading_aloud" and pos > 0 else -1

    parts = [
        "<div style='font-family:Georgia,serif;padding:10px;background:#111;"
        "border-radius:6px;line-height:1.75'>",
        f"<p style='color:#888;font-size:0.82em;margin:0 0 10px'>"
        f"<b style='color:#4af'>{state.upper()}</b>"
        f"{'  ·  <b>' + label + '</b>' if label else ''}"
        f"  ·  sentence {max(pos, 1)} / {total}</p>",
    ]
    for i, sentence in enumerate(sentences):
        escaped = sentence.replace("<", "&lt;").replace(">", "&gt;")
        if i < current_idx:
            style = "color:#555;margin:2px 0"
        elif i == current_idx:
            style = (
                "color:#fff;background:#1a4a7a;padding:2px 6px;"
                "border-radius:3px;font-weight:bold;margin:2px 0;display:inline-block"
            )
        else:
            style = "color:#ccc;margin:2px 0"
        parts.append(f"<p style='{style}'>{escaped}</p>")
    parts.append("</div>")
    return "".join(parts)


def _build_nav_map(orchestrator) -> Optional[plt.Figure]:
    """Render a 2-D top-down (X-Z plane) map of zones with current position."""
    ctx = orchestrator.context
    nav_agent = orchestrator.agents_by_name.get("navigation")
    rp = getattr(nav_agent, "_route_planner", None)
    if rp is None or not rp.zones:
        return None

    dest_label = (ctx.nav_destination or "").lower()
    route_set = {z.lower() for z in ctx.nav_route}
    current_wp = ""
    if ctx.nav_route and ctx.nav_route_idx < len(ctx.nav_route):
        current_wp = ctx.nav_route[ctx.nav_route_idx].lower()

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

    if ctx.nav_last_position is not None:
        px, _, pz = ctx.nav_last_position
        ax.plot(float(px), float(pz), "o", color="#ff4444",
                markersize=9, zorder=10)
        ax.annotate("You", (float(px), float(pz)),
                    xytext=(6, 6), textcoords="offset points",
                    color="#ff8888", fontsize=8)

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.18)
    title = f"→ {ctx.nav_destination}" if ctx.nav_destination else "Navigation Map"
    ax.set_title(title, color="#bbb", fontsize=9, pad=6)
    ax.set_xlabel("X (m)", color="#666", fontsize=8)
    ax.set_ylabel("Z (m)", color="#666", fontsize=8)
    plt.tight_layout(pad=0.6)
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_ui(gui_frame_queue, _vlm_unused, orchestrator=None, servicer=None) -> gr.Blocks:
    # gui_frame_queue is kept for API compat; GUI reads servicer.latest_frame directly.

    # Closure cell for nav map figure lifecycle (prevents matplotlib figure leaks)
    _prev_nav_fig: list = [None]

    def _nav_map_fig():
        if _prev_nav_fig[0] is not None:
            plt.close(_prev_nav_fig[0])
            _prev_nav_fig[0] = None
        if orchestrator is None:
            return None
        fig = _build_nav_map(orchestrator)
        _prev_nav_fig[0] = fig
        return fig

    # ---------------------------------------------------------------- poll
    def _poll():
        ctx = orchestrator.context if orchestrator else None
        frame_bgr = servicer.latest_frame if servicer is not None else None
        frame_rgb = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_bgr is not None else None
        )

        tab_id = _current_tab_id(ctx) if ctx else "tab_idle"

        # --- Tracking tab ---
        ta = orchestrator.agents_by_name.get("tracking") if orchestrator else None
        det = getattr(ta, "last_detection", None) if ta else None
        track_frame = (
            _annotate_tracking(frame_bgr, det) if frame_bgr is not None else None
        )
        vio_line = ""
        if ctx and ctx.current_pose is not None:
            t = ctx.current_pose[:3, 3]
            vio_line = f"\nVIO pos: ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})"
        if det:
            track_status = f"Target: {det['target']}  Score: {det['score']:.2f}{vio_line}"
        elif ctx and ctx.active_agent == "tracking":
            track_status = f"Searching…{vio_line}"
        else:
            track_status = "Tracking idle"

        # --- Reading tab ---
        ra = orchestrator.agents_by_name.get("reading") if orchestrator else None
        ocr_blocks = getattr(ra, "last_ocr_blocks", []) if ra else []
        ocr_frame_bgr = getattr(ra, "last_ocr_frame", None) if ra else None
        ocr_frame = (
            _annotate_ocr_frame(ocr_frame_bgr, ocr_blocks)
            if ocr_frame_bgr is not None else None
        )
        blocks_html = _ocr_blocks_html(ocr_blocks)
        reading_html = (
            _reading_html(ctx) if ctx
            else "<div style='color:#666;padding:10px'>No session.</div>"
        )

        # --- Navigation tab ---
        nav_fig = _nav_map_fig()
        nav_status = "Navigation idle"
        if ctx:
            if ctx.nav_state == "navigating":
                idx = ctx.nav_route_idx
                wp = (
                    ctx.nav_route[idx]
                    if ctx.nav_route and idx < len(ctx.nav_route) else "—"
                )
                nav_status = f"→ {ctx.nav_destination}  |  Next waypoint: {wp}"
            elif ctx.nav_state == "destination_reached":
                nav_status = f"Arrived at {ctx.nav_destination}"

        obstacle_text = ""
        if servicer is not None and servicer.last_nav_result:
            r = servicer.last_nav_result
            if time.time() - r["at"] < 20.0:
                obstacle_text = f"⚠ {r['description']}  —  {r['depth_m']:.1f} m"

        # --- Shared ---
        raw_log = list(servicer.chat_log) if servicer is not None else []
        chat_pairs = [
            msg
            for user_text, bot_text in raw_log
            for msg in ({"role": "user", "content": user_text}, {"role": "assistant", "content": bot_text})
        ]

        return (
            gr.update(selected=tab_id),  # ui_tabs          [auto-switch]
            track_frame,                  # ui_track_image
            track_status,                 # ui_track_status
            ocr_frame,                    # ui_read_ocr_frame
            blocks_html,                  # ui_read_blocks_html
            reading_html,                 # ui_read_html
            nav_fig,                      # ui_nav_map
            frame_rgb,                    # ui_nav_frame
            nav_status,                   # ui_nav_status
            obstacle_text,                # ui_nav_obstacle
            frame_rgb,                    # ui_info_frame
            chat_pairs,                   # ui_info_chatbot
            chat_pairs,                   # ui_shared_chatbot
        )

    # ---------------------------------------------------------------- layout
    with gr.Blocks(title="Vision Assistant Monitor") as app:
        gr.Markdown("## Vision Assistant — Server Monitor")
        btn_reset = gr.Button("Stop / Reset Session", variant="stop", size="sm")

        ui_tabs = gr.Tabs(selected="tab_info")
        with ui_tabs:

            with gr.Tab("Info / Chat", id="tab_info"):
                ui_info_frame = gr.Image(
                    label="Latest Frame",
                    type="numpy",
                )
                ui_info_chatbot = gr.Chatbot(
                    label="Conversation", height=300
                )

            with gr.Tab("Tracking", id="tab_tracking"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ui_track_image = gr.Image(
                            label="Live Frame (annotated)",
                            type="numpy",

                        )
                    with gr.Column(scale=1):
                        ui_track_status = gr.Textbox(
                            label="Detection Info",
                            lines=5,
                            interactive=False,
                        )

            with gr.Tab("Reading", id="tab_reading"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ui_read_ocr_frame = gr.Image(
                            label="OCR View (latest frame)",
                            type="numpy",

                        )
                    with gr.Column(scale=1):
                        ui_read_blocks_html = gr.HTML(label="Detected Text Blocks")
                ui_read_html = gr.HTML()

            with gr.Tab("Navigation", id="tab_nav"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ui_nav_map = gr.Plot(label="Top-Down Map")
                    with gr.Column(scale=1):
                        ui_nav_frame = gr.Image(
                            label="Camera View",
                            type="numpy",

                        )
                ui_nav_status = gr.Textbox(
                    label="Navigation Status", lines=2, interactive=False,
                )
                ui_nav_obstacle = gr.Textbox(
                    label="Obstacle Alert", lines=1, interactive=False,
                )

        with gr.Accordion("Session Chat (all modes)", open=False):
            ui_shared_chatbot = gr.Chatbot(
                label="All Exchanges", height=250
            )

        timer = gr.Timer(value=0.5)
        timer.tick(
            fn=_poll,
            inputs=[],
            outputs=[
                ui_tabs,
                ui_track_image,
                ui_track_status,
                ui_read_ocr_frame,
                ui_read_blocks_html,
                ui_read_html,
                ui_nav_map,
                ui_nav_frame,
                ui_nav_status,
                ui_nav_obstacle,
                ui_info_frame,
                ui_info_chatbot,
                ui_shared_chatbot,
            ],
        )

        def _reset():
            if orchestrator:
                orchestrator.reset_context()

        btn_reset.click(fn=_reset, inputs=[], outputs=[])

    return app
