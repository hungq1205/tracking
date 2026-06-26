import json
import os
from datetime import datetime, timezone
from typing import List

import open3d as o3d

from zone_labeler import Zone

# scan_server/data/maps — sibling directory to this file
_MAPS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "data", "maps")
)


def export_map(
    cloud: o3d.geometry.PointCloud,
    zones: List[Zone],
    location_id: str,
    maps_root: str = _MAPS_ROOT,
) -> str:
    """
    Writes map_geometry.ply and map_labels.json under maps_root/location_id/.
    Returns the output directory path.
    """
    out_dir = os.path.join(maps_root, location_id)
    os.makedirs(out_dir, exist_ok=True)

    ply_path = os.path.join(out_dir, "map_geometry.ply")
    o3d.io.write_point_cloud(ply_path, cloud)

    metadata = {
        "location_id": location_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "point_count": len(cloud.points),
        "zones": [
            {
                "label": z.label,
                "bbox_min": z.bbox_min,
                "bbox_max": z.bbox_max,
            }
            for z in zones
        ],
    }
    json_path = os.path.join(out_dir, "map_labels.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[MapExporter] Saved '{location_id}' → {out_dir}  "
        f"({len(cloud.points):,} pts, {len(zones)} zones)"
    )
    return out_dir
