import json
import os
from datetime import datetime, timezone
from typing import List, Optional

import open3d as o3d

from zone_labeler import Zone

# scan_server/data/maps — sibling directory to this file
_MAPS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "data", "maps")
)


def _zone_to_dict(z: Zone, occupancy_map=None) -> dict:
    """Serialise a Zone to a JSON-compatible dict, optionally including occupancy grid."""
    d: dict = {
        "label": z.label,
        "bbox_min": z.bbox_min,
        "bbox_max": z.bbox_max,
        "landmarks": [
            {
                "name": lm.name,
                "x": round(lm.x, 4),
                "z": round(lm.z, 4),
                "confidence": round(lm.confidence, 4),
            }
            for lm in getattr(z, "landmarks", [])
        ],
    }
    if occupancy_map is not None:
        d["occupancy_grid"] = occupancy_map.extract_subgrid(z.bbox_min, z.bbox_max)
    return d


def export_map(
    cloud: o3d.geometry.PointCloud,
    zones: List[Zone],
    location_id: str,
    maps_root: str = _MAPS_ROOT,
    zone_type: str = "",
    occupancy_map=None,
) -> str:
    """
    Writes map_geometry.ply and map_labels.json under maps_root/location_id/.

    New fields vs. prior schema (backward-compatible — NavigationAgent ignores them):
      zone_type          : high-level venue descriptor e.g. "hospital"
      zones[].landmarks  : list of {name, x, z, confidence} semantic landmarks
      zones[].occupancy_grid : 2D traversability subgrid for this area

    Returns the output directory path.
    """
    out_dir = os.path.join(maps_root, location_id)
    os.makedirs(out_dir, exist_ok=True)

    ply_path = os.path.join(out_dir, "map_geometry.ply")
    o3d.io.write_point_cloud(ply_path, cloud)

    metadata = {
        "location_id": location_id,
        "zone_type": zone_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "point_count": len(cloud.points),
        "zones": [_zone_to_dict(z, occupancy_map) for z in zones],
    }
    json_path = os.path.join(out_dir, "map_labels.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    n_landmarks = sum(len(getattr(z, "landmarks", [])) for z in zones)
    print(
        f"[MapExporter] Saved '{location_id}' → {out_dir}  "
        f"({len(cloud.points):,} pts, {len(zones)} zones, {n_landmarks} landmarks)"
    )
    return out_dir
