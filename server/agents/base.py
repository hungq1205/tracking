from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from domain.intents import ParsedIntent

if TYPE_CHECKING:
    from orchestrator.session import SessionContext


@dataclass
class AgentRequest:
    user_text: str
    frame: Optional[np.ndarray]
    context: SessionContext
    intent: ParsedIntent
    frame_tick: bool = False
    rag_context: str = ""


@dataclass
class AgentResult:
    agent_name: str
    state: str
    payload: Dict[str, Any] = field(default_factory=dict)
    reply_text: str = ""
    speak: bool = True


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def handle(self, request: AgentRequest) -> AgentResult:
        pass
