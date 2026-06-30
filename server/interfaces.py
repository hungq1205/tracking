from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Detection:
    box_xyxy: Tuple[float, float, float, float]
    score: float


@dataclass
class LabeledDetection:
    label: str
    box_xyxy: Tuple[float, float, float, float]
    score: float


class IObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, prompt: str) -> Detection:
        pass