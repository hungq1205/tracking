"""
SemanticMapper — VLM landmark tagging + deferred GroundingDINO backprojection.

GroundingDINO detection (and therefore world (x,z) localization) no longer
runs proactively per accepted frame during scanning — it's expensive and
pointless for objects the user never asks about. Instead:
  1. Accepted frames are batched (ScanSession owns the buffering now, keyed
     off novelty+blur gating — see scan_session.py) and sent to the VLM via
     tag_landmarks_batch(), which returns per-image landmark/object NAME
     tags only (no boxes, no world coordinates).
  2. GroundingDINO only runs later, on demand, via
     _detect_and_backproject() — called from ScanSession.resolve_landmark()
     when the user actually asks to navigate to something (tag-match first,
     else a first-hit scan across stored frames — see scan_session.py) and
     from ScanSession's finalize-time export path (once per unique tag seen
     across a session's stored frames).

Usage:
    from semantic_mapper import SemanticMapper, Landmark

    mapper = SemanticMapper(vlm_client, grounding_dino_detector)
    tag_lists = mapper.tag_landmarks_batch([frame_bgr_1, frame_bgr_2, ...])
    # ... later, on demand ...
    landmarks = mapper._detect_and_backproject(frame_bgr, depth_map, world_pose,
                                               K, "water bottle", frame_idx=7)
    clustered = mapper.cluster_landmarks(all_resolved_landmarks)
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ── sys.path bootstrap: add server/ so "from tools.detector import ..." resolves ──
_SERVER_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)


@dataclass
class Landmark:
    name: str
    x: float                          # world X centroid of footprint (metres)
    z: float                          # world Z centroid of footprint (metres)
    confidence: float
    frame_idx: int                    # source frame index within the session
    footprint_min: Tuple[float, float]  # (x_min, z_min) world-space AABB corner — clustering only
    footprint_max: Tuple[float, float]  # (x_max, z_max) world-space AABB corner — clustering only
    # The actual 4 backprojected box corners (image TL,TR,BR,BL order), each
    # independently rotated by the camera's world_pose — a parallelogram when
    # the camera views the object at an angle, not an axis-aligned rectangle.
    # footprint_min/max above are just this quad's bounding envelope, kept
    # only because _aabb_overlap_ratio's clustering math wants a cheap AABB;
    # rendering should use footprint_corners for the true shape.
    footprint_corners: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ]


def _aabb_overlap_ratio(a: Landmark, b: Landmark) -> float:
    """Intersection area / area of the smaller of the two X-Z footprints."""
    ax0, az0 = a.footprint_min
    ax1, az1 = a.footprint_max
    bx0, bz0 = b.footprint_min
    bx1, bz1 = b.footprint_max
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(az1, bz1) - max(az0, bz0)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(1e-9, (ax1 - ax0) * (az1 - az0))
    area_b = max(1e-9, (bx1 - bx0) * (bz1 - bz0))
    return inter / min(area_a, area_b)


def _merge_landmark_group(members: List[Landmark]) -> Landmark:
    """Union the AABB footprints of a group of same-label, overlapping
    landmarks; the merged shape keeps the highest-confidence member's actual
    quadrilateral (footprint_corners) rather than trying to geometrically
    union several parallelograms into one shape."""
    best = max(members, key=lambda lm: lm.confidence)
    x0 = min(lm.footprint_min[0] for lm in members)
    z0 = min(lm.footprint_min[1] for lm in members)
    x1 = max(lm.footprint_max[0] for lm in members)
    z1 = max(lm.footprint_max[1] for lm in members)
    return Landmark(
        name=best.name,
        x=(x0 + x1) / 2.0,
        z=(z0 + z1) / 2.0,
        confidence=best.confidence,
        frame_idx=best.frame_idx,
        footprint_min=(x0, z0),
        footprint_max=(x1, z1),
        footprint_corners=best.footprint_corners,
    )


_TAG_PROMPT_TEMPLATE = """\
You are analyzing {n} images from a room/environment scan, numbered 1 to {n} in order.

For EACH image, list the distinct landmark/object names visible in it that would be \
useful for indoor navigation or later finding a specific object (furniture, fixtures, \
appliances, signage, containers, and other notable objects).

Respond with EXACTLY {n} lines, one per image, in the same order as the images. Each \
line must be a comma-separated list of short lowercase object names for that image \
only — no numbering, no extra commentary, no markdown. If an image has no notable \
objects, output an empty line for it.

Example (for 3 images):
chair, table, lamp
door, shelf

"""


class SemanticMapper:
    """
    Tags accepted frames with VLM-generated landmark/object names (batched,
    no boxes/coordinates), and — only on demand — runs GroundingDINO +
    backprojection to localize a specific query. See module docstring.
    """

    OVERLAP_MERGE_RATIO = 0.5    # merge same-label footprints overlapping >= 50%
    IMAGES_PER_PROMPT = 5        # frames buffered per VLM tagging call (multi-view
                                 # context) — buffering itself lives in ScanSession now.

    def __init__(self, vlm, detector) -> None:
        """
        vlm      : multi-image VLM client (GemmaVLMClient) — query(prompt, images=[...]) -> str
        detector : GroundingDINODetector instance
        """
        self._vlm = vlm
        self._detector = detector

        # Debug snapshot of the most recently processed frame — surfaced by the
        # scan GUI's "Detections" tab so raw GroundingDINO boxes + the VLM
        # response can be inspected live, independent of the final Landmarks.
        # Note: last_detections only updates now when _detect_and_backproject()
        # actually runs (on-demand resolve_landmark()/finalize-time export),
        # not on every accepted frame during scanning — see module docstring.
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.last_detections: list = []
        self.last_vlm_response: str = ""
        self.last_error: Optional[str] = None

    # ── public ──────────────────────────────────────────────────────────────────

    def tag_landmarks_batch(self, frames: List[np.ndarray]) -> List[List[str]]:
        """
        One VLM call across up to IMAGES_PER_PROMPT frames, asking for
        per-image landmark/object NAME tags only — no GroundingDINO, no
        boxes, no world coordinates (see module docstring for why that's
        deferred). Returns one tag list per input frame, same order.
        Stateless — the caller (ScanSession) owns buffering frames up to a
        batch and mapping each result back onto its own stored frame.
        """
        if not frames:
            return []
        n = len(frames)
        prompt = _TAG_PROMPT_TEMPLATE.format(n=n)

        self.last_frame_bgr = frames[-1]
        self.last_error = None

        try:
            response = self._vlm.query(prompt, images=list(frames))
            self.last_vlm_response = response
        except Exception as e:
            print(f"[SemanticMapper] Tag batch VLM call failed ({n} images): {e}")
            self.last_error = f"VLM call failed: {e}"
            return [[] for _ in range(n)]

        tag_lists = self._parse_tag_response(response, n)
        print(f"[SemanticMapper] Tag batch ({n} images): {tag_lists}")
        return tag_lists

    @staticmethod
    def _parse_tag_response(response: str, n: int) -> List[List[str]]:
        """Defensive parse: split on newlines, pad/truncate to exactly `n`
        lines if the VLM didn't follow the requested format, then split each
        line on commas. Never raises — worst case returns n empty lists."""
        text = response.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines() if not line.startswith("```")
            ).strip()

        lines = text.split("\n") if text else []
        if len(lines) < n:
            lines = lines + [""] * (n - len(lines))
        elif len(lines) > n:
            print(f"[SemanticMapper] Tag response had {len(lines)} lines, expected {n} — truncating.")
            lines = lines[:n]

        return [
            [t.strip().lower() for t in line.split(",") if t.strip()]
            for line in lines
        ]

    def _detect_and_backproject(
        self,
        frame_bgr: np.ndarray,
        depth_map: np.ndarray,
        world_pose: np.ndarray,
        K: np.ndarray,
        dino_prompt: str,
        frame_idx: int,
    ) -> List[Landmark]:
        try:
            raw_detections = self._detector.detect_all(
                frame_bgr, dino_prompt, box_threshold=0.35, text_threshold=0.25
            )
            self.last_detections = raw_detections  # overwritten per frame; ends on the batch's last frame
        except Exception as e:
            print(f"[SemanticMapper] Detector failed (frame {frame_idx}): {e}")
            self.last_error = f"Detector failed: {e}"
            return []

        print(f"[SemanticMapper] frame {frame_idx}: {len(raw_detections)} raw detection(s) "
              f"for prompt {dino_prompt!r}")
        for det in raw_detections:
            print(f"    box={tuple(round(v, 1) for v in det.box_xyxy)} "
                  f"label={det.label!r} score={det.score:.3f}")

        h, w = depth_map.shape[:2]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx_k = float(K[0, 2])
        cy_k = float(K[1, 2])

        landmarks: List[Landmark] = []
        for det in raw_detections:
            x0, y0, x1, y1 = det.box_xyxy
            x0c = int(np.clip(round(x0), 0, w - 1))
            x1c = int(np.clip(round(x1), 0, w - 1))
            y0c = int(np.clip(round(y0), 0, h - 1))
            y1c = int(np.clip(round(y1), 0, h - 1))
            if x1c <= x0c or y1c <= y0c:
                print(f"    SKIP '{det.label}': degenerate box after clipping to frame bounds "
                      f"({w}x{h}) — box was {det.box_xyxy}")
                continue

            # Robust depth: median over an eroded central ~60% sub-region of the
            # box, avoiding edge/background pixel bleed that a single center
            # pixel (or the box's own corners, which often fall on background)
            # is prone to.
            bw, bh = x1c - x0c, y1c - y0c
            ex, ey = int(bw * 0.2), int(bh * 0.2)
            sx0, sx1 = x0c + ex, x1c - ex
            sy0, sy1 = y0c + ey, y1c - ey
            if sx1 <= sx0 or sy1 <= sy0:
                sx0, sx1, sy0, sy1 = x0c, x1c, y0c, y1c  # box too small to erode

            region = depth_map[sy0:sy1, sx0:sx1]
            valid = region[(region > 0.1) & (region < 15.0)]
            if valid.size == 0:
                print(f"    SKIP '{det.label}': no valid depth (0.1-15.0m) in box region "
                      f"depth_map[{sy0}:{sy1}, {sx0}:{sx1}] "
                      f"(raw range there: {region.min():.2f}-{region.max():.2f}m)")
                continue
            depth = float(np.median(valid))

            # Unproject the 4 image-space box corners at that one representative
            # depth, each through the FULL world_pose (rotation included) — so
            # these are 4 independently-rotated 3D points, not a shared-plane
            # rectangle. When the camera views the object at an angle this
            # forms a real parallelogram in world X-Z, not an axis-aligned
            # box; footprint_corners keeps that shape, footprint_min/max below
            # is just its bounding envelope (kept only for the clustering
            # overlap math in _aabb_overlap_ratio, not for rendering).
            corners_uv = np.array(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
            )
            X_cam = (corners_uv[:, 0] - cx_k) * depth / fx
            Y_cam = (corners_uv[:, 1] - cy_k) * depth / fy
            pts_cam = np.stack(
                [X_cam, Y_cam, np.full(4, depth), np.ones(4)], axis=1
            )
            pts_world = (world_pose @ pts_cam.T).T[:, :3]

            xs, zs = pts_world[:, 0], pts_world[:, 2]
            corners_xz = tuple((float(px), float(pz)) for px, pz in zip(xs, zs))
            landmarks.append(Landmark(
                name=det.label,
                x=float(xs.mean()),
                z=float(zs.mean()),
                confidence=det.score,
                frame_idx=frame_idx,
                footprint_min=(float(xs.min()), float(zs.min())),
                footprint_max=(float(xs.max()), float(zs.max())),
                footprint_corners=corners_xz,
            ))
            print(f"    OK '{det.label}': depth={depth:.2f}m -> world (x={xs.mean():.2f}, z={zs.mean():.2f})")

        return landmarks

    def cluster_landmarks(self, landmarks: List[Landmark]) -> List[Landmark]:
        """
        Merge raw landmarks that are almost certainly duplicate detections of
        the same physical object:
          - Group by name (case-insensitive).
          - Within each name group, connect any pair whose X-Z footprints
            overlap by >= OVERLAP_MERGE_RATIO (intersection / smaller area),
            then merge each connected component (scipy connected_components
            over the pairwise overlap graph — handles chains of 3+ overlapping
            detections correctly, unlike greedy first-element clustering).
          - Per merged group: union footprint, highest-confidence detection's
            name casing/frame_idx.

        Returns one representative Landmark per merged group.
        """
        if not landmarks:
            return []

        by_name: dict = defaultdict(list)
        for lm in landmarks:
            by_name[lm.name.lower()].append(lm)

        merged: List[Landmark] = []
        for _, group in by_name.items():
            n = len(group)
            if n == 1:
                merged.append(group[0])
                continue

            rows, cols = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    if _aabb_overlap_ratio(group[i], group[j]) >= self.OVERLAP_MERGE_RATIO:
                        rows += [i, j]
                        cols += [j, i]

            adj = coo_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
            )
            n_comp, comp_labels = connected_components(adj, directed=False)
            for comp_id in range(n_comp):
                members = [group[i] for i in range(n) if comp_labels[i] == comp_id]
                merged.append(_merge_landmark_group(members))

        return merged
