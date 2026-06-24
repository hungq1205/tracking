import gradio as gr
import cv2
import queue
from typing import Dict, Any, Generator

def create_ui(gui_frame_queue, vlm_instance, orchestrator=None, conversation_queue=None):
    def reset_session():
        if vlm_instance:
            vlm_instance.reset()
        if orchestrator:
            orchestrator.reset_context()

    def run_experiment(instruction) -> Generator[Dict[str, Any], None, None]:
        if vlm_instance:
            vlm_instance.update_params({"base_instruction": instruction})

        print("[SERVER GUI] Listening for live Edge signal...")
        chat_history = []
        while True:
            # drain all pending conversation turns first
            if conversation_queue is not None:
                while True:
                    try:
                        entry = conversation_queue.get_nowait()
                        user_text = entry.get("user") or ""
                        asst_text = entry.get("assistant") or ""
                        if user_text:
                            chat_history.append({"role": "user", "content": user_text})
                        if asst_text:
                            chat_history.append({"role": "assistant", "content": asst_text})
                    except queue.Empty:
                        break
                    except Exception:
                        pass  # skip malformed entries

            try:
                frame = gui_frame_queue.get(timeout=2.0)
                yield {
                    ui_image: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ui_status: "Status: Streaming Live from Edge Client",
                    ui_chat: chat_history,
                }
            except queue.Empty:
                yield {
                    ui_status: "Status: Waiting for Edge connection...",
                    ui_chat: chat_history,
                }

    with gr.Blocks(title="Object Tracker") as app:
        gr.Markdown("## Object Tracker - Remote Server Monitor")

        with gr.Column():
            with gr.Row():
                btn_start = gr.Button("Run", variant="primary", scale=1)
                btn_stop = gr.Button("Stop", variant="stop", scale=1)

            with gr.Accordion("VLM Streaming Settings", open=False):
                ui_instruction = gr.Textbox(
                    value=(
                        "You assist a vision-impaired user. What you see is the user's POV. "
                        "Short noun phrases, comma-separated. Example: bottle on table, person walking."
                    ),
                    label="Base Instruction",
                    lines=2,
                )
            ui_image = gr.Image(label="Processed GPU Output Pipeline View")
            ui_status = gr.Textbox(label="Framework Metrics & Connection Status", lines=2, interactive=False)
            ui_chat = gr.Chatbot(label="Conversation Log", height=300)

        run_event = btn_start.click(
            fn=run_experiment,
            inputs=[ui_instruction],
            outputs=[ui_image, ui_status, ui_chat],
        )
        btn_stop.click(fn=reset_session, inputs=[], outputs=[], cancels=[run_event])

    return app
