"""
SemanticMapper — landmark tagging + backprojection, adapted wholesale from
frame_extractor/tagging.py's Gemini -> GroundingDINO-tiny pipeline
(frame_extractor/app.py is the reference implementation this mirrors).

Supersedes the earlier Gemma-VLM-tag-then-defer-GroundingDINO design (see
CLAUDE.md's "Novelty+blur frame gating and deferred landmark resolution"
section for that history) — every accepted frame is now tagged AND detected
immediately, in one batched call, exactly like frame_extractor/app.py does
for its own "new" frames:
  1. Gemini proposes open-set tags for the frame (no fixed vocabulary).
  2. Those tags become GroundingDINO-tiny's own detection prompt for that
     SAME frame (frame_extractor/tagging.py's `_build_prompt`), so what gets
     boxed tracks what Gemini actually saw.
  3. Every box is immediately backprojected into a world (x, z) Landmark
     using the frame's own depth map + pose + intrinsics (already available
     at frame-acceptance time — no reason to defer this any more, unlike
     the old design's expensive full GroundingDINO + separate VLM call).

Real, accepted trade-off from adopting this pipeline as-is: a landmark
Gemini never tags is never detected at all (GroundingDINO-tiny here is only
ever prompted with Gemini's own tags, never an arbitrary open query) — unlike
the old deferred design's Tier-2 "scan every frame with the literal query"
fallback. ScanSession.resolve_landmark() now just searches already-resolved
landmarks by name instead of running any on-demand detection — see that
method's own docstring.

Usage:
    from semantic_mapper import SemanticMapper, Landmark

    mapper = SemanticMapper(frame_tagger)
    landmark_lists = mapper.tag_and_backproject_batch(
        frames_bgr, depth_maps, world_poses, Ks, frame_idxs,
    )
    clustered = mapper.cluster_landmarks([lm for lms in landmark_lists for lm in lms])
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ── sys.path bootstrap: add frame_extractor/ so "from tagging import ..." resolves ──
_FRAME_EXTRACTOR_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frame_extractor")
)
if _FRAME_EXTRACTOR_ROOT not in sys.path:
    sys.path.insert(0, _FRAME_EXTRACTOR_ROOT)

from tagging import FrameTagger, TagDetection, TagResult  # noqa: E402


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


def _center_distance(a: Landmark, b: Landmark) -> float:
    """Straight-line distance between two landmarks' (x, z) centers —
    catches the common case of the SAME physical object detected in two
    different (overlapping-FOV) batched frames, where perspective can shift
    the backprojected footprint enough that the two boxes barely overlap
    even though the centers land close together."""
    return ((a.x - b.x) ** 2 + (a.z - b.z) ** 2) ** 0.5


def _names_related(a: str, b: str) -> bool:
    """True if two landmark names plausibly describe the same object, not
    just when they're identical — e.g. "lamp" and "desk lamp" (a more
    generic vs. more specific tag for the same physical thing, common when
    Gemini tags the same object slightly differently across frames in a
    batch) should still be eligible to merge. Whole-WORD containment, not a
    raw substring check — "lamp" is contained in {"desk", "lamp"} but NOT in
    "clamp" (single word, doesn't match "lamp" exactly), so unrelated words
    that merely share letters don't get merged."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    shorter, longer = (a_words, b_words) if len(a_words) <= len(b_words) else (b_words, a_words)
    return bool(shorter) and shorter.issubset(longer)


def _merge_landmark_group(members: List[Landmark]) -> Landmark:
    """Union the AABB footprints of a group of same-label, nearby/overlapping
    landmarks, but take the POSITION (and confidence/name/frame_idx/quad
    shape) straight from whichever member GroundingDINO was most confident
    about — no averaging. A merged position should be a real detection's
    own reported center, not a blended point no individual detection
    actually reported."""
    best = max(members, key=lambda lm: lm.confidence)
    x0 = min(lm.footprint_min[0] for lm in members)
    z0 = min(lm.footprint_min[1] for lm in members)
    x1 = max(lm.footprint_max[0] for lm in members)
    z1 = max(lm.footprint_max[1] for lm in members)

    return Landmark(
        name=best.name,
        x=best.x,
        z=best.z,
        confidence=best.confidence,
        frame_idx=best.frame_idx,
        footprint_min=(x0, z0),
        footprint_max=(x1, z1),
        footprint_corners=best.footprint_corners,
    )


class SemanticMapper:
    """
    Tags + detects every accepted frame immediately via a FrameTagger
    (Gemini -> GroundingDINO-tiny, frame_extractor/tagging.py) and
    backprojects every resulting box straight into a world-space Landmark.
    See module docstring.
    """

    OVERLAP_MERGE_RATIO = 0.5    # merge same-label footprints overlapping >= 50%
    MERGE_DISTANCE_M = 0.75      # OR merge same-label detections whose centers are this close (metres)
    IMAGES_PER_PROMPT = 5        # frames buffered per batched tag+detect call (batching
                                 # itself lives in ScanSession now, mirroring frame_extractor's
                                 # per-flush batching — buffering amortizes CUDA-launch overhead).

    def __init__(self, frame_tagger: FrameTagger) -> None:
        """
        frame_tagger : FrameTagger instance (frame_extractor/tagging.py) —
                        loaded once, reused across every batch.
        """
        self._tagger = frame_tagger

        # Debug snapshot of the most recently processed frame — surfaced by the
        # scan GUI's "Detections" tab so Gemini tags + GroundingDINO-tiny boxes
        # can be inspected live.
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.last_detections: List[TagDetection] = []
        self.last_tags: List[str] = []
        self.last_error: Optional[str] = None

    # ── public ──────────────────────────────────────────────────────────────────

    def tag_and_backproject_batch(
        self,
        frames_bgr: List[np.ndarray],
        depth_maps: List[np.ndarray],
        world_poses: List[np.ndarray],
        Ks: List[np.ndarray],
        frame_idxs: List[int],
    ) -> List[List[Landmark]]:
        """
        One batched Gemini tag + GroundingDINO-tiny detect call (FrameTagger.
        tag_and_detect_batch, same as frame_extractor/app.py's "Extract new
        frames" button) across up to IMAGES_PER_PROMPT frames, immediately
        followed by per-frame backprojection of every detected box into a
        world-space Landmark. Returns one Landmark list per input frame, same
        order. Stateless — the caller (ScanSession) owns buffering frames up
        to a batch.
        """
        n = len(frames_bgr)
        if n == 0:
            return []

        self.last_frame_bgr = frames_bgr[-1]
        self.last_error = None

        rgbs = [f[:, :, ::-1] for f in frames_bgr]  # BGR -> RGB, matches FrameTagger's expected input
        try:
            results: List[TagResult] = self._tagger.tag_and_detect_batch(rgbs)
        except Exception as e:
            print(f"[SemanticMapper] tag_and_detect_batch failed ({n} images): {e}")
            self.last_error = f"tag_and_detect_batch failed: {e}"
            return [[] for _ in range(n)]

        out: List[List[Landmark]] = []
        for depth_map, world_pose, K, frame_idx, result in zip(
            depth_maps, world_poses, Ks, frame_idxs, results
        ):
            self.last_detections = result.detections
            self.last_tags = result.tags
            landmarks = self._backproject(depth_map, world_pose, K, result.detections, frame_idx)
            out.append(landmarks)

        total = sum(len(lms) for lms in out)
        print(f"[SemanticMapper] Tag+detect batch ({n} images): "
              f"tags={[r.tags for r in results]} -> {total} landmark(s)")
        return out

    def _backproject(
        self,
        depth_map: np.ndarray,
        world_pose: np.ndarray,
        K: np.ndarray,
        detections: List[TagDetection],
        frame_idx: int,
    ) -> List[Landmark]:
        h, w = depth_map.shape[:2]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx_k = float(K[0, 2])
        cy_k = float(K[1, 2])

        landmarks: List[Landmark] = []
        for det in detections:
            x0, y0, x1, y1 = det.box_xyxy
            x0c = int(np.clip(round(x0), 0, w - 1))
            x1c = int(np.clip(round(x1), 0, w - 1))
            y0c = int(np.clip(round(y0), 0, h - 1))
            y1c = int(np.clip(round(y1), 0, h - 1))
            if x1c <= x0c or y1c <= y0c:
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
                continue
            depth = float(np.median(valid))

            # Unproject the 4 image-space box corners at that one representative
            # depth, each through the FULL world_pose (rotation included) — see
            # Landmark.footprint_corners' own docstring for why this is a real
            # parallelogram, not an axis-aligned box.
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

        return landmarks

    def cluster_landmarks(self, landmarks: List[Landmark]) -> List[Landmark]:
        """
        Merge raw landmarks that are almost certainly duplicate detections of
        the same physical object — a real, expected side effect of batching
        several frames per Gemini/GroundingDINO call: overlapping-FOV frames
        in the same batch routinely re-detect the same object, so the raw
        landmark list needs this cleanup pass before it's usable.

        Connects any pair of landmarks whose names are RELATED (`_names_related`
        — identical, or one is a more generic/specific version of the other,
        e.g. "lamp"/"desk lamp"; not just an exact case-insensitive match)
        AND that's EITHER close together (`_center_distance <=
        MERGE_DISTANCE_M`) OR whose X-Z footprints substantially overlap
        (`_aabb_overlap_ratio >= OVERLAP_MERGE_RATIO`) — distance alone
        catches the common case (same object, two viewing angles shift the
        backprojected box enough that it barely overlaps), overlap alone
        still catches a large object whose two detected centers land
        further apart than MERGE_DISTANCE_M. Connected components (scipy
        connected_components over the pairwise merge graph, across ALL
        landmarks at once — not pre-bucketed by exact name — since a name
        match is now one of the merge conditions, not a precondition for
        even considering a pair) handles chains of 3+ nearby/overlapping/
        related detections correctly, e.g. "lamp" -> "desk lamp" -> "small
        desk lamp" all merging transitively even though "lamp" and "small
        desk lamp" alone might not be considered related.

        Per merged group: highest-confidence member's own position (x, z),
        name, confidence, frame_idx, and quad shape — no averaging (see
        `_merge_landmark_group`) — plus the union footprint AABB.

        Returns one representative Landmark per merged group.
        """
        if not landmarks:
            return []
        n = len(landmarks)
        if n == 1:
            return list(landmarks)

        rows, cols = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if not _names_related(landmarks[i].name, landmarks[j].name):
                    continue
                if (_center_distance(landmarks[i], landmarks[j]) <= self.MERGE_DISTANCE_M
                        or _aabb_overlap_ratio(landmarks[i], landmarks[j]) >= self.OVERLAP_MERGE_RATIO):
                    rows += [i, j]
                    cols += [j, i]

        adj = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
        )
        n_comp, comp_labels = connected_components(adj, directed=False)
        merged: List[Landmark] = []
        for comp_id in range(n_comp):
            members = [landmarks[i] for i in range(n) if comp_labels[i] == comp_id]
            merged.append(members[0] if len(members) == 1 else _merge_landmark_group(members))
        return merged
