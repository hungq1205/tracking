import gradio as gr
import cv2
import queue
from typing import Dict, Any, Generator

def create_ui(gui_frame_queue, vlm_instance):
    def run_experiment() -> Generator[Dict[str, Any], None, None]:
        print("[SERVER GUI] Listening for live Edge signal...")
        while True:
            try:
                frame = gui_frame_queue.get(timeout=2.0)
                
                # Format the internal VLM history for the Gradio Chatbot component
                formatted_history = []
                if vlm_instance:
                    for msg in vlm_instance.full_conversation_history:
                        role = msg.get("role")
                        content = msg.get("content")
                        if role == "user":
                            # Filter out internal timestamp prompts (e.g., Time=1.0-2.0s) from the GUI log
                            user_text = " ".join([item["text"] for item in content if item["type"] == "text" and not item["text"].startswith("Time=")]) if isinstance(content, list) else content
                            user_text = user_text.strip()
                            # Only log the user turn if there is actual query text beyond timestamps
                            if user_text:
                                formatted_history.append({"role": "user", "content": user_text})
                        elif role == "assistant":
                            formatted_history.append({"role": "assistant", "content": content})

                yield {
                    ui_image: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    ui_status: "Status: Streaming Live from Edge Client",
                    ui_chatbot: formatted_history
                }
            except queue.Empty:
                yield {ui_status: "Status: Waiting for Edge connection..."}

    with gr.Blocks(title="Object Tracker") as app:
        gr.Markdown("## Object Tracker - Remote Server Monitor")
        
        with gr.Column():
            with gr.Row():
                btn_start = gr.Button("Run", variant="primary", scale=1)
                btn_stop = gr.Button("Stop", variant="stop", scale=1)
            ui_image = gr.Image(label="Processed GPU Output Pipeline View")
            ui_status = gr.Textbox(label="Framework Metrics & Connection Status", lines=2, interactive=False)
            ui_chatbot = gr.Chatbot(label="VLM Conversation History")

        run_event = btn_start.click(
            fn=run_experiment,
            inputs=[],
            outputs=[ui_image, ui_status, ui_chatbot]
        )
        btn_stop.click(fn=None, cancels=[run_event])

    return app