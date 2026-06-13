import gradio as gr
import cv2
import queue
from typing import Dict, Any, Generator

def create_ui(gui_frame_queue):
    def run_experiment() -> Generator[Dict[str, Any], None, None]:
        print("[SERVER GUI] Listening for live Edge signal...")
        while True:
            try:
                frame = gui_frame_queue.get(timeout=2.0)
                yield {
                    ui_image: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ui_status: "Status: Streaming Live from Edge Client"
                }
            except queue.Empty:
                yield {ui_status: "Status: Waiting for Edge connection..."}

    with gr.Blocks(title="Object Tracker") as app:
        gr.Markdown("## Object Tracker - Remote Server Monitor")
        
        with gr.Row():
            with gr.Column(scale=1):
                btn_start = gr.Button("Run", variant="primary")
                btn_stop = gr.Button("Stop", variant="stop")
            with gr.Column(scale=2):
                ui_image = gr.Image(label="Processed GPU Output Pipeline View")
                ui_status = gr.Textbox(label="Framework Metrics & Connection Status", lines=2, interactive=False)

        run_event = btn_start.click(
            fn=run_experiment,
            inputs=[],
            outputs=[ui_image, ui_status]
        )
        btn_stop.click(fn=None, cancels=[run_event])

    return app