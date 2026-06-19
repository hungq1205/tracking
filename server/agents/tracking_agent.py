from domain.intents import Intent
from agents.base import AgentRequest, AgentResult, BaseAgent


class TrackingAgent(BaseAgent):
    name = "tracking"

    def __init__(self, detector):
        self.detector = detector

    def handle(self, request: AgentRequest) -> AgentResult:
        intent = request.intent
        target = intent.target
        state = "IDLE"
        payload = {}

        if intent.intent == Intent.START_TRACKING and target:
            if request.frame is None:
                return AgentResult(
                    agent_name=self.name,
                    state="TARGET_NOT_FOUND",
                    payload={"target": target},
                    reply_text=f"I cannot see a frame yet to find '{target}'.",
                )
            det = self.detector.detect(request.frame, target)
            if det.score > 0.35:
                state = "INITIALIZING"
                payload = {"target": target, "memory_hint": request.rag_context}
                if request.rag_context:
                    reply = f"Starting tracking for '{target}'. From memory: {request.rag_context[:120]}."
                else:
                    reply = f"Starting tracking for '{target}'."
            else:
                state = "TARGET_NOT_FOUND"
                payload = {"target": target}
                if request.rag_context:
                    reply = f"I could not find '{target}' in view. From memory: {request.rag_context[:120]}."
                else:
                    reply = f"I could not find '{target}' in view."
        elif intent.intent == Intent.STOP_TRACKING:
            state = "STOPPED"
            reply = "Stopping tracking."
        else:
            reply = ""

        return AgentResult(
            agent_name=self.name,
            state=state,
            payload=payload,
            reply_text=reply,
        )
