import sys
import os
import numpy as np
from typing import Optional

from tools.traversability import TraversabilityResult, estimate_traversability

CORRIDOR_FRACTION = 0.5  # "the middle area" — see check_obstacle()


class DA3DepthDetector:
    """
    Metric obstacle detection via Depth Anything 3 (ONNX, DA3-METRIC).

    DA3OnnxEstimator.estimate() already returns real metric depth straight
    from the model's own "metric_depth" output head (da3_wrapper.py) for the
    DA3-METRIC checkpoints this project uses (default DA3METRIC-LARGE.onnx)
    — no separate scale-alignment pass is needed or run here. (An earlier
    version of this detector fit DA3's depth against sparse ORB-triangulated
    anchors via a since-removed scan_server/mvs.py helper; that was solving
    for a *relative*-depth model, which this one isn't.)

    Per frame: DA3 inference → metric depth map → either a corridor
    10th-percentile obstacle decision (check_obstacle) or a full polar
    traversability fan (estimate_traversability, see traversability.py) —
    both share one _depth_map() call so a caller requesting both DEPTH and
    TRAVERSABILITY in one AnalyzeFrame round trip doesn't pay for DA3
    inference twice.
    """

    OBSTACLE_THRESHOLD_M = 1.0

    def __init__(
        self,
        onnx_path: str = "DA3METRIC-LARGE.onnx",
        device: Optional[str] = None,
    ):
        """
        ONNX-only — DA3OnnxEstimator via build_estimator(onnx_path=...), the same
        backend/default path (DA3METRIC-LARGE.onnx) frame_extractor/scan_gui.py
        already use for this exact model.
        """
        scan_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "scan_server")
        )
        if scan_root not in sys.path:
            sys.path.insert(0, scan_root)

        from da3_wrapper import build_estimator

        self._da3 = build_estimator(onnx_path=onnx_path, device=device or "cpu")

    def _depth_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """HxW float32 metric depth, metres — shared DA3 call for both
        check_obstacle and estimate_traversability."""
        rgb = frame_bgr[:, :, ::-1]
        return self._da3.estimate(rgb).depth_map

    def check_obstacle(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """Returns (obstacle_present, min_depth_metres). Deliberately no
        RANSAC ground-plane fit (unlike estimate_traversability) — one DA3
        call + a percentile over the middle-width corridor, so this stays
        fast enough for a client to poll on its own fixed cadence
        (ToolDispatcher.kt's obstacle-ahead beep, decoupled from the much
        slower MappingService stream) as well as for the on-demand
        Gemini-invoked check_obstacle tool."""
        w_bgr = frame_bgr.shape[1]
        depth_metric = self._depth_map(frame_bgr)

        cx_start = int(w_bgr * (0.5 - CORRIDOR_FRACTION / 2))
        cx_end = int(w_bgr * (0.5 + CORRIDOR_FRACTION / 2))
        corridor = depth_metric[:, cx_start:cx_end]
        valid_c = corridor > 0.1
        if not valid_c.any():
            return False, 1.0

        min_depth = float(np.percentile(corridor[valid_c], 5))
        return min_depth < self.OBSTACLE_THRESHOLD_M, min_depth

    def estimate_traversability(
        self, frame_bgr: np.ndarray, num_bins: int = 25, max_range_m: float = 5.0,
    ) -> TraversabilityResult:
        """Per-angle obstacle-clearance fan for THIS frame alone — see
        traversability.py's module docstring for the full design (local
        reactive HRTF obstacle-dodge, replaces the old occupancy-grid
        ray-cast steering)."""
        depth_metric = self._depth_map(frame_bgr)
        return estimate_traversability(depth_metric, num_bins=num_bins, max_range_m=max_range_m)
