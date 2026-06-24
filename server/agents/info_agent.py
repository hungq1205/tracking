import threading
from typing import Optional

from agents.base import AgentRequest, AgentResult, BaseAgent


class InfoAgent(BaseAgent):
    name = "info"

    def __init__(self, vlm, vlm_lock: Optional[threading.Lock] = None):
        self.vlm = vlm
        self.vlm_lock = vlm_lock or threading.Lock()

    def handle(self, request: AgentRequest) -> AgentResult:
        if self.vlm is None:
            return AgentResult(
                agent_name=self.name,
                state="ERROR",
                reply_text="StreamingVLM is not initialized.",
            )

        user_text = request.user_text
        ctx = request.context

        if ctx.reading_state != "idle" and ctx.scan_buffer:
            user_text = f"Memory:\n{ctx.scan_buffer}\n\nUser: {user_text}"
        if request.rag_context:
            user_text = f"Memory:\n{request.rag_context}\n\nUser: {user_text}"

        with self.vlm_lock:
            reply = self.vlm.chat(user_text)
        if reply and reply.endswith(" ..."):
            reply = reply[:-4]
        return AgentResult(
            agent_name=self.name,
            state="INFO",
            reply_text=reply or "I could not generate a response.",
        )
