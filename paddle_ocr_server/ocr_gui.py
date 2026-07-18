"""
OCR pipeline monitor — live view of the most recent /ocr request, one panel
per pipeline stage: received -> preprocessed -> raw text blocks (boxed) ->
merged paragraphs (boxed) -> final text only. Mounted onto the same FastAPI
app server.py already serves (see mount_gradio_app call there), so
`uvicorn server:app` stays the single entrypoint — reachable at /gui.
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
import gradio as gr
import numpy as np


def _ago(at: float) -> str:
    if not at:
        return "never"
    dt = time.time() - at
    return f"{dt:.1f}s ago" if dt < 60 else time.strftime("%H:%M:%S", time.localtime(at))


def _draw_blocks(image_rgb: Optional[np.ndarray], blocks: list, color: tuple) -> Optional[np.ndarray]:
    if image_rgb is None:
        return None
    vis = image_rgb.copy()
    for b in blocks:
        x1, y1, x2, y2 = map(int, b["box"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = b["text"] if len(b["text"]) <= 24 else b["text"][:21] + "..."
        cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def _blocks_text(blocks: list) -> str:
    if not blocks:
        return "(none)"
    return "\n".join(f"[{b['score']:.2f}] {b['text']}" for b in blocks)


def create_ui(monitor) -> gr.Blocks:
    def _poll():
        snap = monitor.snapshot()
        if not snap:
            empty_status = "No /ocr requests received yet."
            return None, None, None, "(none)", None, "(none)", "", empty_status

        raw_blocks = snap.get("raw_blocks") or []
        merged_blocks = snap.get("merged_blocks") or []
        processed_rgb = snap.get("processed_rgb")

        status = (
            f"file: {snap.get('filename', '?')}  ·  {snap.get('image_bytes', 0)} bytes  ·  "
            f"{_ago(snap.get('at', 0))}"
        )
        if snap.get("error"):
            status += f"\nERROR: {snap['error']}"

        return (
            snap.get("original_rgb"),
            snap.get("preprocessed_rgb"),
            _draw_blocks(processed_rgb, raw_blocks, (255, 140, 0)),
            _blocks_text(raw_blocks),
            _draw_blocks(processed_rgb, merged_blocks, (0, 200, 0)),
            _blocks_text(merged_blocks),
            "\n".join(b["text"] for b in merged_blocks),
            status,
        )

    with gr.Blocks(title="OCR Pipeline Monitor") as app:
        gr.Markdown(
            "## OCR Pipeline Monitor\n"
            "Live view of the most recent `/ocr` request, stage by stage."
        )
        ui_status = gr.Textbox(label="Last request", lines=2, interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("**1. Received**")
                ui_received = gr.Image(label="Decoded input", type="numpy")
            with gr.Column():
                gr.Markdown("**2. Preprocessed**")
                ui_preprocessed = gr.Image(label="Denoised / thresholded", type="numpy")

        with gr.Row():
            with gr.Column():
                gr.Markdown("**3. Raw text blocks** (before merge)")
                ui_raw_image = gr.Image(label="Boxes on orientation-corrected image", type="numpy")
                ui_raw_text = gr.Textbox(label="Raw blocks", lines=8, interactive=False)
            with gr.Column():
                gr.Markdown("**4. Merged paragraphs**")
                ui_merged_image = gr.Image(label="Boxes on orientation-corrected image", type="numpy")
                ui_merged_text = gr.Textbox(label="Merged blocks", lines=8, interactive=False)

        gr.Markdown("**5. Final text**")
        ui_final_text = gr.Textbox(label="Final text only", lines=6, interactive=False)

        timer = gr.Timer(value=0.5)
        timer.tick(
            fn=_poll,
            inputs=[],
            outputs=[
                ui_received,
                ui_preprocessed,
                ui_raw_image,
                ui_raw_text,
                ui_merged_image,
                ui_merged_text,
                ui_final_text,
                ui_status,
            ],
        )

    return app
