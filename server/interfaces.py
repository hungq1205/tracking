from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Tuple, Dict, Any, Optional
import numpy as np

if TYPE_CHECKING:
    from domain.intents import ParsedIntent

@dataclass
class ObjectTrack:
    box_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    confidence: float
    visible: bool
    status: str
    debug: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HandTrack:
    box_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    confidence: float
    visible: bool

@dataclass
class GuidanceState:
    object_track: ObjectTrack
    hand_track: HandTrack
    delta_x: float
    delta_y: float
    distance_px: float
    instruction: str

@dataclass
class Detection:
    box_xyxy: Tuple[float, float, float, float]
    score: float

@dataclass
class LabeledDetection:
    label: str
    box_xyxy: Tuple[float, float, float, float]
    score: float

class IObjectTracker(ABC):
    @abstractmethod
    def initialize(self, frame: np.ndarray, prompt: str) -> ObjectTrack:
        pass

    @abstractmethod
    def update(self, frame: np.ndarray) -> ObjectTrack:
        pass

class IHandDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> HandTrack:
        pass


class IObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, prompt: str) -> Detection:
        pass


class IIntentParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> "ParsedIntent":
        pass