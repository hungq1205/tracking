import gradio as gr
import cv2
import queue
from typing import Dict, Any, Generator

def create_ui(gui_frame_queue, vlm_instance, orchestrator=None):
    default_prompt = (
        "You are a vision assistant for a blind user. This is their POV.\n"
        "OUTPUT FORMAT RULES:\n"
        "IF HAZARD DETECTED → start with \"ALERT: \" + short warning. "
        "Examples: ALERT: step down ahead. ALERT: a cable on floor ahead. ALERT: a dog approaching from the front.\n"
        "IF NO HAZARD → lowercase noun phrases or user actions, comma-separated. "
        "Examples: brown chair left, 2 food cans on table, user took 1 apple.\n"
        "NEVER use all caps. NEVER write long sentences. MAX 20 TOKENS."
    )

    def reset_session():
        if vlm_instance:
            vlm_instance.reset()
        if orchestrator:
            orchestrator.reset_context()

    def run_experiment(max_tokens, instruction, t_round, v_round) -> Generator[Dict[str, Any], None, None]:
        if vlm_instance:
            # Package parameters into a dict for the VLM wrapper
            vlm_instance.update_params({
                "max_new_tokens": int(max_tokens),
                "base_instruction": instruction,
                "text_round": int(t_round),
                "visual_round": int(v_round)
            })

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
            
            with gr.Accordion("VLM Streaming Settings", open=False):
                ui_max_tokens = gr.Slider(minimum=10, maximum=256, value=25, step=5, label="Max New Tokens")
                ui_instruction = gr.Textbox(value=default_prompt, label="Base Instruction", lines=2)
                ui_text_round = gr.Number(value=32, label="Text KV Rounding (tokens)")
                ui_visual_round = gr.Number(value=24, label="Visual KV Rounding (frames)")
            ui_image = gr.Image(label="Processed GPU Output Pipeline View")
            ui_status = gr.Textbox(label="Framework Metrics & Connection Status", lines=2, interactive=False)
            ui_chatbot = gr.Chatbot(label="VLM Conversation History")

        run_event = btn_start.click(
            fn=run_experiment,
            inputs=[ui_max_tokens, ui_instruction, ui_text_round, ui_visual_round],
            outputs=[ui_image, ui_status, ui_chatbot]
        )
        btn_stop.click(fn=reset_session, inputs=[], outputs=[], cancels=[run_event])

    return app