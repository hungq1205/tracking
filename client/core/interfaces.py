from abc import ABC, abstractmethod
from dataclasses import dataclass, field
# Assuming numpy is installed, otherwise add to requirements.txt
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

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
    landmarks: List[Tuple[float, float]] = field(default_factory=list)

@dataclass
class GuidanceState:
    object_track: ObjectTrack
    hand_track: HandTrack
    delta_x: float
    delta_y: float
    distance_px: float
    instruction: str
    fps: float = 0.0

@dataclass
class Detection:
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

class IClientApp(ABC):
    @abstractmethod
    def run(self, video_source: Any, prompt: str, server_ip: str, stream_to_server: bool):
        """
        Runs the client application.
        :param video_source: Path to video file, camera index, or a camera object.
        :param prompt: The object detection prompt.
        :param server_ip: IP address of the gRPC server.
        :param stream_to_server: Whether to stream rendered frames to the server's GUI.
        """