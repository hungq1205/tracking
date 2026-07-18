import sys
import os
import numpy as np
from typing import Optional

CORRIDOR_FRACTION = 1 / 3


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

    Per frame: DA3 inference → metric depth map → corridor 10th-percentile
    depth → obstacle decision.
    """

    OBSTACLE_THRESHOLD_M = 1.5

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

    def check_obstacle(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """Returns (obstacle_present, min_depth_metres)."""
        w_bgr = frame_bgr.shape[1]
        rgb = frame_bgr[:, :, ::-1]

        depth_metric = self._da3.estimate(rgb).depth_map  # HxW float32, metres

        cx_start = int(w_bgr * (0.5 - CORRIDOR_FRACTION / 2))
        cx_end = int(w_bgr * (0.5 + CORRIDOR_FRACTION / 2))
        corridor = depth_metric[:, cx_start:cx_end]
        valid_c = corridor > 0.1
        if not valid_c.any():
            return False, 1.0

        min_depth = float(np.percentile(corridor[valid_c], 10))
        return min_depth < self.OBSTACLE_THRESHOLD_M, min_depth
