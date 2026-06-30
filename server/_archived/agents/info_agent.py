from typing import Optional

import numpy as np

from agents.base import AgentRequest, AgentResult, BaseAgent
from tools.cloud_vlm import CloudVLMClient


class InfoAgent(BaseAgent):
    name = "info"

    def __init__(self, cloud_vlm: CloudVLMClient):
        self.cloud_vlm = cloud_vlm

    def handle(self, request: AgentRequest) -> AgentResult:
        user_text = request.user_text
        ctx = request.context

        # Inject reading context if mid-session
        if ctx.reading_state != "idle" and ctx.scan_buffer:
            user_text = (
                f"<<<READING_CONTEXT_START>>>\n{ctx.scan_buffer}\n<<<READING_CONTEXT_END>>>\n{user_text}"
            )
        if request.rag_context:
            user_text = f"Relevant saved memory:\n{request.rag_context}\n\nUser: {user_text}"

        frame = request.frame
        reply = self.cloud_vlm.query(user_text, frame)
        return AgentResult(
            agent_name=self.name,
            state="INFO",
            reply_text=reply or "I could not generate a response.",
        )
