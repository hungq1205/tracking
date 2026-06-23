import time
from typing import Dict, List, Optional

import numpy as np

from agents.base import AgentRequest, AgentResult, BaseAgent
from domain.intents import Intent, ParsedIntent
from orchestrator.router import RuleRouter
from orchestrator.session import SessionContext


class Orchestrator:
    def __init__(
        self,
        agents: List[BaseAgent],
        general_parser,
        reading_parser,
        tracking_parser=None,
        router: Optional[RuleRouter] = None,
        rag_store=None,
    ):
        self.agents_by_name: Dict[str, BaseAgent] = {a.name: a for a in agents}
        self.general_parser = general_parser
        self.reading_parser = reading_parser
        self.tracking_parser = tracking_parser
        self.router = router or RuleRouter(self.agents_by_name)
        self.context = SessionContext()
        self.rag_store = rag_store

    def orchestrate(
        self,
        user_text: str,
        frame: Optional[np.ndarray] = None,
        frame_tick: bool = False,
    ) -> AgentResult:
        # Pick parser based on current mode
        if user_text:
            if self.context.reading_state != "idle":
                parser = self.reading_parser
            elif self.tracking_parser and self.context.active_agent == "tracking":
                parser = self.tracking_parser
            else:
                parser = self.general_parser
            intent = parser.parse(user_text)
        else:
            intent = ParsedIntent()

        if frame_tick and not user_text:
            intent = ParsedIntent(intent=Intent.INFO)

        # Always retrieve RAG context for questions and tracking so agents have memory hints
        rag_context = ""
        if self.rag_store is not None and user_text:
            rag_query = None
            if intent.intent == Intent.INFO:
                rag_query = user_text
            elif intent.intent == Intent.START_TRACKING and intent.target:
                rag_query = intent.target
            if rag_query:
                try:
                    hits = self.rag_store.query_global(rag_query, top_k=3)
                    if hits:
                        rag_context = "\n".join(f"[{lbl}] {text}" for text, lbl, _ in hits)
                except Exception as e:
                    print(f"[Orchestrator] RAG context lookup failed: {e}")

        agent_name = self.router.select(intent, self.context, frame_tick=frame_tick)
        if not agent_name:
            return AgentResult(agent_name="none", state="IDLE", reply_text="", speak=False)

        agent = self.agents_by_name[agent_name]
        request = AgentRequest(
            user_text=user_text or "",
            frame=frame,
            context=self.context,
            intent=intent,
            frame_tick=frame_tick,
            rag_context=rag_context,
        )
        result = agent.handle(request)

        # SAVE_MEMORY: transition to scanning mode, wait for frame ticks to accumulate
        if (
            result.agent_name == "memory"
            and result.payload.get("action") == "save_requested"
        ):
            label = result.payload.get("label", "default")
            self.context.reading_state = "scanning"
            self.context.scan_buffer = ""
            self.context.scan_buffer_char_count = 0
            self.context.active_label = label
            self.context.active_agent = "reading"
            return AgentResult(
                agent_name="memory",
                state="SAVE_STARTED",
                payload={"label": label},
                reply_text=f"Ready to scan. Point camera at content to save to '{label}'.",
                speak=True,
            )

        # READ_MEMORY: pre-load the stored text into scan_buffer and delegate to reading agent
        if result.agent_name == "memory" and result.state == "READ_MEMORY_REQUESTED":
            text = result.payload.get("text", "")
            label = result.payload.get("label", "")
            self.context.scan_buffer = text
            self.context.scan_buffer_char_count = len(text)
            self.context.reading_state = "scanning"
            self.context.active_label = label
            self.context.active_agent = "reading"
            reading = self.agents_by_name["reading"]
            read_request = AgentRequest(
                user_text=user_text or "",
                frame=frame,
                context=self.context,
                intent=ParsedIntent(intent=Intent.READ_ALOUD, label=label),
                frame_tick=False,
            )
            result = reading.handle(read_request)

        self._update_context(result, frame_tick=frame_tick)
        return result

    def reset_context(self):
        self.context = SessionContext()
        print("[Orchestrator] Session context reset.")

    def on_frame_tick(self, frame: np.ndarray) -> Optional[AgentResult]:
        if self.context.reading_state != "scanning":
            return None
        return self.orchestrate(user_text="", frame=frame, frame_tick=True)

    def _update_context(self, result: AgentResult, frame_tick: bool = False) -> None:
        now = time.time()
        self.context.last_intent_at = now

        if result.agent_name == "reading":
            state = result.state
            if state in ("STOPPED", "DONE_READING"):
                info_agent = self.agents_by_name.get("info")
                if info_agent and getattr(info_agent, "vlm", None):
                    with info_agent.vlm_lock:
                        info_agent.vlm.strip_reading_context()
                self.context.active_agent = None
                self.context.reading_state = "idle"
                self.context.scan_buffer = ""
                self.context.read_sentences = []
                self.context.read_position = 0
            elif state == "STARTED":
                self.context.active_agent = "reading"
                self.context.reading_state = "scanning"
                label = result.payload.get("label")
                if label:
                    self.context.active_label = label
            elif state == "SCANNING":
                self.context.active_agent = "reading"
                self.context.last_frame_ocr_at = now
            elif state == "READING_ALOUD":
                self.context.active_agent = "reading"
                self.context.reading_state = "reading_aloud"
            elif state == "PAUSED":
                self.context.reading_state = "paused"
            # SCREEN_READ leaves reading_state as-is (idle)

        elif result.agent_name == "memory":
            label = result.payload.get("label")
            if label:
                self.context.active_label = label

        elif result.agent_name == "tracking":
            if result.state in ("INITIALIZING", "STOPPED", "TARGET_NOT_FOUND"):
                self.context.active_agent = "tracking"

        elif result.agent_name == "info":
            if self.context.active_agent is None:
                self.context.active_agent = "info"
