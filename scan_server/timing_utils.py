"""
Shared console-timing helper for the Live Reconstruction pipeline (point
cloud -> voxelization -> occupancy map -> rendering). A separate module (not
folded into scan_session.py) specifically so occupancy_map.py can also use it
without creating a circular import (scan_session.py already imports FROM
occupancy_map.py).

Usage:
    with timed("depth estimation (4 frames)"):
        ...

Prints one line per block: "[timing] <label>: <elapsed> ms".
"""

import time
from contextlib import contextmanager


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[timing] {label}: {elapsed_ms:.1f} ms")
