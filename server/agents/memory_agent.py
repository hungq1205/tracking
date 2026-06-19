import threading

from agents.base import AgentRequest, AgentResult, BaseAgent
from domain.intents import Intent


class MemoryAgent(BaseAgent):
    name = "memory"

    def __init__(self, store, rag_store, vlm=None, vlm_lock=None):
        self.store = store
        self.rag_store = rag_store
        self.vlm = vlm
        self.vlm_lock = vlm_lock or threading.Lock()

    def append(self, label: str, text: str, source: str = "ocr") -> tuple[str, str]:
        appended, full_text = self.store.append(label, text, source=source)
        if appended:
            self.rag_store.add_text(label, appended, source=source)
        return appended, full_text

    def handle(self, request: AgentRequest) -> AgentResult:
        intent = request.intent
        label = intent.label or request.context.active_label or "default"

        if intent.intent == Intent.SAVE_MEMORY:
            return AgentResult(
                agent_name=self.name,
                state="SAVED",
                payload={"label": label, "action": "save_requested"},
                reply_text=f"Ready to save to memory label '{label}'. Scan the screen to capture text.",
                speak=True,
            )

        if intent.intent == Intent.READ_MEMORY:
            full_text = self.rag_store.get_full_text(label)
            if not full_text:
                return AgentResult(
                    agent_name=self.name,
                    state="EMPTY",
                    payload={"label": label},
                    reply_text=f"I have no saved memory for '{label}'.",
                )
            # Signal orchestrator to delegate to reading agent for chunked playback
            return AgentResult(
                agent_name=self.name,
                state="READ_MEMORY_REQUESTED",
                payload={"label": label, "text": full_text},
                reply_text="",
                speak=False,
            )

        if intent.intent == Intent.REMEMBER_OBJECT:
            if request.frame is None:
                return AgentResult(
                    agent_name=self.name,
                    state="ERROR",
                    payload={"label": label},
                    reply_text="No frame available to remember.",
                    speak=True,
                )
            description = self._describe_object(label)
            self.rag_store.add_object(label, request.frame, description)
            return AgentResult(
                agent_name=self.name,
                state="OBJECT_SAVED",
                payload={"label": label, "description": description[:80]},
                reply_text=f"'{label}' saved to memory: {description[:60]}...",
                speak=True,
            )

        return AgentResult(agent_name=self.name, state="IDLE", reply_text="", speak=False)

    def _describe_object(self, label: str) -> str:
        if self.vlm is None:
            return f"{label} (no VLM available for description)"
        try:
            with self.vlm_lock:
                description = self.vlm.chat(
                    f"Describe this object briefly for memory storage. What is it? Label hint: {label}"
                )
            return description or f"{label} (no description generated)"
        except Exception as e:
            return f"{label} (description failed: {e})"
