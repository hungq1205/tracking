import threading
from typing import Optional

import numpy as np

from agents.base import AgentRequest, AgentResult, BaseAgent
from domain.intents import Intent
from interfaces import IObjectDetector

_DINO_THRESHOLD = 0.35


class MemoryAgent(BaseAgent):
    name = "memory"

    def __init__(self, store, rag_store, detector: Optional[IObjectDetector] = None, vlm=None, vlm_lock=None):
        self.store = store
        self.rag_store = rag_store
        self.detector = detector
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
            image_to_save, located = self._locate_object(request.frame, label)
            description = self._describe_object(label)
            self.rag_store.add_object(label, image_to_save, description)
            location_note = "" if located else " (object not precisely located in frame)"
            return AgentResult(
                agent_name=self.name,
                state="OBJECT_SAVED",
                payload={"label": label, "description": description[:80], "located": located},
                reply_text=f"'{label}' saved to memory{location_note}: {description[:60]}...",
                speak=True,
            )

        return AgentResult(agent_name=self.name, state="IDLE", reply_text="", speak=False)

    def _locate_object(self, frame: np.ndarray, label: str) -> tuple[np.ndarray, bool]:
        """Return (cropped_frame, located). Falls back to full frame if DINO can't find the object."""
        if self.detector is None:
            return frame, False
        try:
            det = self.detector.detect(frame, label)
            if det.score > _DINO_THRESHOLD:
                return self._crop_frame(frame, det.box_xyxy), True
        except Exception:
            pass
        return frame, False

    def _crop_frame(self, frame: np.ndarray, box_xyxy: tuple) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(max(0, v)) for v in box_xyxy)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]

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
