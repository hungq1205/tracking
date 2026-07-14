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
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
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

    SAMPLE_INTERVAL_S = 1.0      # pick the sharpest frame out of every ~1s of footage
                                 # (by actual capture timestamp) to feed the VLM pipeline
    SAMPLE_EVERY_N_FALLBACK = 5  # frame-count cadence used only when no capture
                                 # timestamp is available (see consider_frame())
    SHARPNESS_SCORING_MAX_DIM = 480  # downscale-before-scoring, same as CameraManager.kt
    OVERLAP_MERGE_RATIO = 0.5    # merge same-label footprints overlapping >= 50%
    IMAGES_PER_PROMPT = 5        # sampled frames buffered per VLM call (multi-view context,
                                 # shared grounding_dino_prompt) — net effect: one VLM call
                                 # per SAMPLE_INTERVAL_S * IMAGES_PER_PROMPT ≈ 5s of footage.

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
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.last_detections: list = []
        self.last_vlm_response: str = ""
        self.last_error: Optional[str] = None

        # Sampled frames awaiting a batched VLM call — see extract_landmarks().
        self._pending: List[dict] = []

        # Sharpest-frame-per-window accumulator — see consider_frame().
        self._window_start_ns: Optional[int] = None
        self._window_best: Optional[dict] = None
        self._fallback_frame_count = 0

    # ── public ──────────────────────────────────────────────────────────────────

    def consider_frame(
        self,
        frame_bgr: np.ndarray,
        depth_map: np.ndarray,
        world_pose: np.ndarray,
        K: np.ndarray,
        zone_type: str,
        area_name: str,
        frame_idx: int = 0,
        timestamp_ns: Optional[int] = None,
    ) -> List[Landmark]:
        """
        Call on every raw frame (not pre-sampled) — internally keeps only the
        sharpest frame (Variance-of-Laplacian, same metric as the Android
        client's CameraManager.kt) seen within the current ~SAMPLE_INTERVAL_S
        window of actual capture time, and forwards just that one frame into
        the VLM-batching pipeline (extract_landmarks) once the window
        elapses. This avoids baking a motion-blurred/rolling-shutter frame
        into the semantic map just because it happened to land on a sampling
        boundary.

        Falls back to a fixed every-Nth-frame cadence (no sharpness scoring)
        when timestamp_ns is None — i.e. a pose source that doesn't thread
        frame_timestamps_ns through (see scan_session.py).
        """
        if timestamp_ns is None:
            self._fallback_frame_count += 1
            if self._fallback_frame_count % self.SAMPLE_EVERY_N_FALLBACK != 0:
                return []
            return self.extract_landmarks(
                frame_bgr, depth_map, world_pose, K, zone_type, area_name, frame_idx
            )

        candidate = {
            "frame_bgr": frame_bgr, "depth_map": depth_map, "world_pose": world_pose,
            "K": K, "zone_type": zone_type, "area_name": area_name, "frame_idx": frame_idx,
            "score": self._sharpness(frame_bgr),
        }

        if self._window_start_ns is None:
            self._window_start_ns = timestamp_ns

        landmarks: List[Landmark] = []
        if (timestamp_ns - self._window_start_ns) / 1e9 >= self.SAMPLE_INTERVAL_S:
            landmarks = self._forward_window_best()
            self._window_start_ns = timestamp_ns

        if self._window_best is None or candidate["score"] > self._window_best["score"]:
            self._window_best = candidate

        return landmarks

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
        Buffers this frame; once IMAGES_PER_PROMPT frames are buffered, fires
        ONE multi-image VLM call across all of them — one shared
        grounding_dino_prompt from multi-view context instead of judging
        landmarks off a single frame — then runs GroundingDINO + backprojection
        per-frame as before (bounding boxes are inherently frame-specific).

        Returns [] on every call except the one that fills the buffer, which
        returns landmarks for the whole batch at once. Call flush() at
        segment/session end so a partial leftover buffer isn't silently
        dropped.

        frame_bgr  : BGR numpy array (H×W×3 uint8)
        depth_map  : float32 metric depth array (H×W), metres
        world_pose : 4×4 float64 camera-to-world matrix
        K          : 3×3 float64 camera intrinsics
        """
        self._pending.append({
            "frame_bgr": frame_bgr,
            "depth_map": depth_map,
            "world_pose": world_pose,
            "K": K,
            "zone_type": zone_type,
            "area_name": area_name,
            "frame_idx": frame_idx,
        })
        if len(self._pending) < self.IMAGES_PER_PROMPT:
            return []
        return self._process_pending()

    def flush(self) -> List[Landmark]:
        """Process a partial leftover buffer (fewer than IMAGES_PER_PROMPT
        frames buffered), plus any not-yet-forwarded sharpest-in-window frame
        — call at segment/session end."""
        landmarks = self._forward_window_best()
        if self._pending:
            landmarks.extend(self._process_pending())
        return landmarks

    # ── private (windowing) ─────────────────────────────────────────────────────

    def _forward_window_best(self) -> List[Landmark]:
        best = self._window_best
        self._window_best = None
        if best is None:
            return []
        return self.extract_landmarks(
            best["frame_bgr"], best["depth_map"], best["world_pose"], best["K"],
            best["zone_type"], best["area_name"], best["frame_idx"],
        )

    @classmethod
    def _sharpness(cls, frame_bgr: np.ndarray) -> float:
        """Variance of the Laplacian — the standard fast blur metric (higher
        = more in-focus detail). Scored on a downscaled copy since this runs
        on every raw frame, not just the ones that end up kept."""
        h, w = frame_bgr.shape[:2]
        long_edge = max(h, w)
        if long_edge > cls.SHARPNESS_SCORING_MAX_DIM:
            scale = cls.SHARPNESS_SCORING_MAX_DIM / long_edge
            small = cv2.resize(
                frame_bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── private ─────────────────────────────────────────────────────────────────

    def _process_pending(self) -> List[Landmark]:
        batch = self._pending
        self._pending = []

        # Use the most recent frame's zone/area context — a full batch
        # normally sits entirely within one segment, so this only matters for
        # a partial batch spanning a zone transition (rare, minor imprecision).
        zone_type = batch[-1]["zone_type"]
        area_name = batch[-1]["area_name"]
        area_clause = (
            f"{area_name} in the {zone_type}" if zone_type.strip() else area_name
        )
        prompt = _USER_PROMPT_TEMPLATE.format(
            zone_type=zone_type or "",
            area_name=area_name,
            area_clause=area_clause,
            zone_type_val=zone_type or "",
        )

        self.last_frame_bgr = batch[-1]["frame_bgr"]
        self.last_detections = []
        self.last_error = None

        frame_idxs = [item["frame_idx"] for item in batch]
        print(f"[SemanticMapper] VLM call: {len(batch)} images, frames {frame_idxs}, "
              f"zone='{zone_type}' area='{area_name}'")
        try:
            vlm_response = self._vlm.query(
                prompt, images=[item["frame_bgr"] for item in batch]
            )
            self.last_vlm_response = vlm_response
            print(f"[SemanticMapper] VLM raw response: {vlm_response!r}")
            dino_prompt = self._parse_vlm_response(vlm_response)
            print(f"[SemanticMapper] Parsed grounding_dino_prompt: {dino_prompt!r}")
        except Exception as e:
            print(f"[SemanticMapper] VLM call failed (frames {frame_idxs}): {e}")
            self.last_error = f"VLM call failed: {e}"
            return []

        if not dino_prompt:
            print(f"[SemanticMapper] Empty/unparseable grounding_dino_prompt for frames {frame_idxs} — "
                  f"skipping detection for this batch. Raw response was: {vlm_response!r}")
            self.last_error = "VLM response had no usable grounding_dino_prompt"
            return []

        landmarks: List[Landmark] = []
        for item in batch:
            landmarks.extend(self._detect_and_backproject(
                item["frame_bgr"], item["depth_map"], item["world_pose"], item["K"],
                dino_prompt, item["frame_idx"],
            ))
        print(f"[SemanticMapper] Batch frames {frame_idxs}: {len(landmarks)} landmarks total.")
        return landmarks

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
