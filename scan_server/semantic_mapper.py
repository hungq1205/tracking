"""
SemanticMapper — VLM + GroundingDINO landmark extraction for offline scanning.

Usage:
    from semantic_mapper import SemanticMapper, Landmark

    mapper = SemanticMapper(vlm_client, grounding_dino_detector)
    raw = mapper.extract_landmarks(frame_bgr, depth_map, world_pose, K,
                                   zone_type="hospital", area_name="lobby",
                                   frame_idx=42)
    clustered = mapper.cluster_landmarks(raw)
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ── sys.path bootstrap: add server/ so "from tools.detector import ..." resolves ──
_SERVER_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)


@dataclass
class Landmark:
    name: str
    x: float        # world X coordinate (metres)
    z: float        # world Z coordinate (metres)
    confidence: float
    frame_idx: int  # source frame index within the session


_USER_PROMPT_TEMPLATE = """\
Current zone: {zone_type}
Current area: {area_name}

Analyze the image.

The zone describes the type of environment. Select navigation-relevant landmarks \
that are appropriate for this {area_clause} and visible in the image.

Examples:
- Theater: stage, seat rows, aisle, exit door, vending machine, ticket counter, restroom sign
- House: sofa, television, dining table, wardrobe, refrigerator, sink, door
- Supermarket: checkout counter, shopping cart, produce aisle, beverage aisle, refrigerator, entrance, exit
- Office: desk, meeting table, reception desk, elevator, staircase, printer
- Hospital: reception desk, waiting chairs, elevator, nurse station, restroom, exit

Output:
{{ "zone": "{zone_type_val}", "area": "{area_name}", "grounding_dino_prompt": "<object1> . <object2> . <object3> . ..." }}
"""


class SemanticMapper:
    """
    Extracts semantic landmarks from single RGB-D frames using a VLM and GroundingDINO.

    Typical call sequence per video segment:
        1. For each sampled keyframe: extract_landmarks(...)  → List[Landmark]
        2. After segment ends:        cluster_landmarks(all_raw) → List[Landmark]
    """

    SAMPLE_EVERY_N = 5      # call VLM+detector on 1 out of every N frames
    CLUSTER_RADIUS_M = 0.5  # spatial threshold for merging duplicate detections

    def __init__(self, vlm, detector) -> None:
        """
        vlm      : any CloudVLMClient (typically OpenRouterVLMClient)
        detector : GroundingDINODetector instance
        """
        self._vlm = vlm
        self._detector = detector

    # ── public ──────────────────────────────────────────────────────────────────

    def extract_landmarks(
        self,
        frame_bgr: np.ndarray,
        depth_map: np.ndarray,
        world_pose: np.ndarray,
        K: np.ndarray,
        zone_type: str,
        area_name: str,
        frame_idx: int = 0,
    ) -> List[Landmark]:
        """
        Full pipeline for one frame:
          1. Build VLM prompt with zone/area context.
          2. Call VLM → parse JSON → extract grounding_dino_prompt.
          3. Call detector.detect_all(frame, prompt) → bounding boxes with labels.
          4. For each box centre: depth lookup + unproject + world transform.
          5. Return List[Landmark].

        frame_bgr  : BGR numpy array (H×W×3 uint8)
        depth_map  : float32 metric depth array (H×W), metres
        world_pose : 4×4 float64 camera-to-world matrix
        K          : 3×3 float64 camera intrinsics
        """
        area_clause = (
            f"{area_name} in the {zone_type}" if zone_type.strip() else area_name
        )
        prompt = _USER_PROMPT_TEMPLATE.format(
            zone_type=zone_type or "",
            area_name=area_name,
            area_clause=area_clause,
            zone_type_val=zone_type or "",
        )

        try:
            vlm_response = self._vlm.query(prompt, image=frame_bgr)
            dino_prompt = self._parse_vlm_response(vlm_response)
        except Exception as e:
            print(f"[SemanticMapper] VLM call failed (frame {frame_idx}): {e}")
            return []

        if not dino_prompt:
            return []

        try:
            raw_detections = self._detector.detect_all(
                frame_bgr, dino_prompt, box_threshold=0.35, text_threshold=0.25
            )
        except Exception as e:
            print(f"[SemanticMapper] Detector failed (frame {frame_idx}): {e}")
            return []

        h, w = depth_map.shape[:2]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx_k = float(K[0, 2])
        cy_k = float(K[1, 2])

        landmarks: List[Landmark] = []
        for det in raw_detections:
            x0, y0, x1, y1 = det.box_xyxy
            u = (x0 + x1) / 2.0
            v = (y0 + y1) / 2.0
            u_int = int(np.clip(round(u), 0, w - 1))
            v_int = int(np.clip(round(v), 0, h - 1))

            depth = float(depth_map[v_int, u_int])
            if not (0.1 < depth < 15.0):
                continue

            X_cam = (u - cx_k) * depth / fx
            Y_cam = (v - cy_k) * depth / fy
            Z_cam = depth

            pt_cam = np.array([X_cam, Y_cam, Z_cam, 1.0], dtype=np.float64)
            pt_world = world_pose @ pt_cam

            landmarks.append(Landmark(
                name=det.label,
                x=float(pt_world[0]),
                z=float(pt_world[2]),
                confidence=det.score,
                frame_idx=frame_idx,
            ))

        return landmarks

    def cluster_landmarks(self, landmarks: List[Landmark]) -> List[Landmark]:
        """
        Merge raw per-frame landmarks:
          - Group by name (case-insensitive).
          - Within each name group, single-linkage cluster by spatial proximity.
          - Per cluster: average (x, z), keep highest confidence, preserve name casing
            from the highest-confidence detection.

        Returns one representative Landmark per cluster.
        """
        if not landmarks:
            return []

        by_name: dict = defaultdict(list)
        for lm in landmarks:
            by_name[lm.name.lower()].append(lm)

        clustered: List[Landmark] = []
        for _, group in by_name.items():
            clusters: List[List[Landmark]] = []
            for lm in group:
                placed = False
                for cl in clusters:
                    rep = cl[0]
                    dist = math.sqrt((lm.x - rep.x) ** 2 + (lm.z - rep.z) ** 2)
                    if dist < self.CLUSTER_RADIUS_M:
                        cl.append(lm)
                        placed = True
                        break
                if not placed:
                    clusters.append([lm])

            for cl in clusters:
                avg_x = sum(lm.x for lm in cl) / len(cl)
                avg_z = sum(lm.z for lm in cl) / len(cl)
                best = max(cl, key=lambda lm: lm.confidence)
                clustered.append(Landmark(
                    name=best.name,
                    x=avg_x,
                    z=avg_z,
                    confidence=best.confidence,
                    frame_idx=best.frame_idx,
                ))

        return clustered

    # ── private ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_vlm_response(response: str) -> str:
        """
        Extract grounding_dino_prompt from a VLM JSON response string.
        Handles markdown code fences and extra whitespace. Returns "" on failure.
        """
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return ""
        try:
            data = json.loads(text[start:end])
            return str(data.get("grounding_dino_prompt", "")).strip()
        except json.JSONDecodeError:
            return ""
