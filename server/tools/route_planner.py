from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Zone:
    label: str
    bbox_min: np.ndarray   # shape (3,)
    bbox_max: np.ndarray   # shape (3,)

    @property
    def centroid(self) -> np.ndarray:
        return (self.bbox_min + self.bbox_max) / 2.0

    def contains(self, point: np.ndarray) -> bool:
        return bool(np.all(point >= self.bbox_min) and np.all(point <= self.bbox_max))

    def arrival_radius(self) -> float:
        return float(np.linalg.norm(self.bbox_max - self.bbox_min) / 2.0)


class RoutePlanner:
    def __init__(self, zones: List[dict]):
        self.zones: List[Zone] = [
            Zone(
                label=z["label"],
                bbox_min=np.array(z["bbox_min"], dtype=np.float32),
                bbox_max=np.array(z["bbox_max"], dtype=np.float32),
            )
            for z in zones
        ]
        self._by_label: dict[str, Zone] = {z.label.lower(): z for z in self.zones}

        # Landmark name -> (owning Zone, x, z) — parsed from zones[].landmarks[]
        # (map_labels.json, written by scan_server/semantic_mapper.py's
        # extraction + scan_server/map_exporter.py's export), previously
        # silently dropped here. Landmarks have no y/height, just a rough
        # ground-plane centroid — enough as an A* path-planner target
        # (server/tools/grid_path_planner.py), not a precise object pose.
        self._landmark_index: dict[str, Tuple[Zone, float, float]] = {}
        for zone_obj, zone_dict in zip(self.zones, zones):
            for lm in zone_dict.get("landmarks", []):
                name = str(lm.get("name", "")).strip()
                if not name:
                    continue
                self._landmark_index[name.lower()] = (
                    zone_obj, float(lm["x"]), float(lm["z"]),
                )
        self._landmark_names: List[str] = list(self._landmark_index.keys())

    @classmethod
    def from_map_file(cls, map_labels_path: str) -> "RoutePlanner":
        with open(map_labels_path) as f:
            data = json.load(f)
        return cls(data.get("zones", []))

    def find_zone(self, label: str) -> Optional[Zone]:
        return self._by_label.get(label.lower())

    def find_landmark(self, name: str) -> Optional[Tuple[Zone, float, float]]:
        """Case-insensitive exact match first, then substring fallback (no
        embedding/semantic search — landmark names are short structured
        strings, e.g. "computer", not prose). Returns (owning Zone, x, z) or
        None if nothing matches."""
        key = name.strip().lower()
        if not key:
            return None
        hit = self._landmark_index.get(key)
        if hit is not None:
            return hit
        for lm_key in self._landmark_names:
            if key in lm_key or lm_key in key:
                return self._landmark_index[lm_key]
        return None

    def find_zone_containing(self, position: np.ndarray) -> Optional[Zone]:
        for z in self.zones:
            if z.contains(position):
                return z
        if not self.zones:
            return None
        dists = [np.linalg.norm(z.centroid - position) for z in self.zones]
        return self.zones[int(np.argmin(dists))]

    def compute_route(self, start_pos: np.ndarray, dest_label: str) -> List[Zone]:
        """
        Greedy nearest-neighbour route from start_pos to dest_label.
        At each step pick the unvisited zone that is both nearest to the
        current position AND strictly closer to the destination than the
        current position.  Destination is always appended last.
        """
        dest = self.find_zone(dest_label)
        if dest is None:
            return []

        remaining = [z for z in self.zones if z.label.lower() != dest_label.lower()]
        route: List[Zone] = []
        current_pos = start_pos.copy()

        while remaining:
            dist_to_dest = float(np.linalg.norm(current_pos - dest.centroid))
            candidates = [
                z for z in remaining
                if np.linalg.norm(z.centroid - dest.centroid) < dist_to_dest
            ]
            if not candidates:
                break
            nearest = min(candidates, key=lambda z: np.linalg.norm(z.centroid - current_pos))
            route.append(nearest)
            current_pos = nearest.centroid
            remaining.remove(nearest)

        route.append(dest)
        return route

    def route_announcement(self, route: List[Zone], dest_label: str) -> str:
        if not route:
            return f"I could not find a route to {dest_label}."
        if len(route) == 1:
            return f"Navigating to {route[0].label}. I will alert you of obstacles along the way."
        stops = ", then ".join(z.label for z in route[:-1])
        return (
            f"Navigating to {route[-1].label}. "
            f"You will pass through: {stops}. "
            "I will alert you of obstacles along the way."
        )
