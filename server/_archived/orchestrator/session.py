from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

ReadingState = Literal["idle", "scanning", "reading_aloud", "paused"]
NavState = Literal["idle", "navigating", "destination_reached"]


@dataclass
class SessionContext:
    active_agent: Optional[str] = None
    active_label: Optional[str] = None
    last_frame_ocr_at: float = 0.0
    last_intent_at: float = 0.0

    reading_state: ReadingState = "idle"
    reading_direction: str = "ltr"  # "ltr" | "rtl"

    scan_buffer: str = ""
    scan_buffer_char_count: int = 0
    memory_text_cache: str = ""  # snapshot of label's persisted text at scan start

    read_sentences: List[str] = field(default_factory=list)
    read_position: int = 0

    # VIO state — initialized lazily on first IMU frame received
    # Using Any to avoid a hard import of gtsam at module load time
    vio_estimator: Any = None        # VIOEstimator instance or None
    imu_preintegrator: Any = None    # IMUPreintegrator instance or None
    current_pose: Optional[Any] = None  # latest 4×4 np.ndarray from VIO (or None)

    # Navigation state
    nav_state: NavState = "idle"
    nav_destination: Optional[str] = None
    nav_location_id: Optional[str] = None
    nav_route: List[str] = field(default_factory=list)   # ordered zone labels
    nav_route_idx: int = 0
    nav_last_obstacle_at: float = 0.0
    nav_last_depth_check_at: float = 0.0
    nav_last_localize_at: float = 0.0
    nav_last_position: Optional[List[float]] = None       # [x, y, z] in map space
