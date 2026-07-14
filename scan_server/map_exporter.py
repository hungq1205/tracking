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
                "footprint_min": [round(v, 4) for v in lm.footprint_min],
                "footprint_max": [round(v, 4) for v in lm.footprint_max],
                # Actual 4 backprojected box corners (parallelogram when
                # viewed at an angle) — footprint_min/max above is just this
                # quad's AABB envelope. Falls back to the AABB's 4 corners if
                # an older in-memory Landmark predates this field.
                "footprint_corners": [
                    [round(px, 4), round(pz, 4)]
                    for px, pz in getattr(lm, "footprint_corners", None) or (
                        (lm.footprint_min[0], lm.footprint_min[1]),
                        (lm.footprint_max[0], lm.footprint_min[1]),
                        (lm.footprint_max[0], lm.footprint_max[1]),
                        (lm.footprint_min[0], lm.footprint_max[1]),
                    )
                ],
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
      zones[].occupancy_grid : 2D traversability subgrid for this area, with a
        "class" sub-grid (0=unknown,1=ground,2=low/step-over,3=normal
        obstacle — see OccupancyMap._classify_state) alongside the original
        "data" float grid, unchanged.
      occupancy_grid (top-level) : whole-map counterpart to the per-zone ones
        above (OccupancyMap.extract_full_grid()) — needed so
        server/tools/grid_path_planner.py's A* can path across zone
        boundaries, not just within one zone's AABB. Omitted if there's no
        accumulated cloud data yet.

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
    if occupancy_map is not None:
        full_grid = occupancy_map.extract_full_grid()
        if full_grid is not None:
            metadata["occupancy_grid"] = full_grid

    json_path = os.path.join(out_dir, "map_labels.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    n_landmarks = sum(len(getattr(z, "landmarks", [])) for z in zones)
    print(
        f"[MapExporter] Saved '{location_id}' → {out_dir}  "
        f"({len(cloud.points):,} pts, {len(zones)} zones, {n_landmarks} landmarks)"
    )
    return out_dir
