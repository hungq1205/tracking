from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import open3d as o3d


@dataclass
class Zone:
    label: str
    bbox_min: List[float]
    bbox_max: List[float]
    landmarks: List[Any] = field(default_factory=list)  # List[Landmark] from semantic_mapper


class ZoneLabeler:
    """
    Interactive zone labeler on top of an Open3D point cloud.

    Usage:
        labeler = ZoneLabeler()
        zone = labeler.label_interactive(cloud, "Kitchen")
        # Opens an Open3D window where the operator shift+clicks boundary points.
        # Returns a Zone once the window is closed (Q or X), or None if < 2 points picked.
    """

    def __init__(self):
        self.zones: List[Zone] = []

    def label_interactive(self, cloud: o3d.geometry.PointCloud, label: str) -> Optional[Zone]:
        print(f"\n[ZoneLabeler] Defining zone '{label}'")
        print("  Shift+click to pick boundary points.")
        print("  Press Q or close the window when done.")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"Label zone: {label}", width=1024, height=768)
        vis.add_geometry(cloud)
        vis.run()
        vis.destroy_window()

        picked = vis.get_picked_points()
        if len(picked) < 2:
            print(f"[ZoneLabeler] Only {len(picked)} point(s) picked — zone not created.")
            return None

        pts = np.asarray(cloud.points)[list(picked)]
        bbox_min = pts.min(axis=0).tolist()
        bbox_max = pts.max(axis=0).tolist()

        zone = Zone(label=label, bbox_min=bbox_min, bbox_max=bbox_max)
        self.zones.append(zone)
        print(f"[ZoneLabeler] Zone '{label}': min={bbox_min}  max={bbox_max}")
        return zone

    def clear(self):
        self.zones.clear()
