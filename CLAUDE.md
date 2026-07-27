# Tracking — Vision-Assistance System

> **For AI agents:** This file is the authoritative living document for this project.
> **Every time you add, remove, or significantly change a component, update the relevant section here.**
> Keep it accurate and scannable — future agents rely on it to understand the system without re-reading the whole codebase.

---

## Purpose

This is an AI-powered assistive system for **vision-impaired people**.
The Android app (the only client this project ships — see
"Client-Orchestrated Live Session" below) runs on a phone the user carries
or wears. The system:

1. **Tracks objects** in the scene and gives real-time spatial guidance ("move left", "closer")  
2. **Reads text aloud** — screens, documents, labels — using OCR and TTS sentence-by-sentence  
3. **Builds 3D maps** of environments live, automatically, while guiding — no offline scanning step  
4. **Answers questions** about the scene via voice or text chat  
5. **Stores and recalls memories** so the user can say "where did I put my keys?"  

All modalities (voice input, voice output, vision, memory) combine into one
portable, real-time pipeline, orchestrated entirely on-device via Gemini
Live — the server is a pure heavy-compute + live-mapping backend with no
conversational state of its own.

---

## System Architecture
Note that python env is at: server/.venv/ — scan_server (imported in-process
by the main server for live mapping) is actually run/tested via conda env
`hrtf`; see Development Environments below.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Android app  ·  client/android/                                     │
│  • Gemini Live session opened directly from the device (API key      │
│    embedded in-app) — conversational orchestration + tool-calling    │
│    loop run entirely on-device (live/GeminiLiveClient.kt +           │
│    live/ToolDispatcher.kt)                                           │
│  • Local: ORB tracking continuation, MediaPipe hands, on-device      │
│    memory store, A* path planning, HRTF beacon math                  │
│  • 3rd-party direct: OCR (OCR.space), Gemini Live itself             │
│  • Device-native: phone/alarm/calendar (in-process, no wire hop)     │
└───────────┬─────────────────────────────────────────────────────────┘
            │ gRPC — heavy compute only (JPEG frames + IMU for mapping)
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Main Server  ·  server/grpc_server.py  ·  port 50051               │
│  ┌────────────────────┐ ┌────────────────────┐ ┌──────────────────┐ │
│  │  TrackingService   │ │ PerceptionService  │ │  MappingService  │ │
│  │  (DetectObject/    │ │  (AnalyzeFrame/    │ │  (live SLAM +    │ │
│  │   GetEmbedding)    │ │   Synthesize/Embed)│ │   localization)  │ │
│  └────────────────────┘ └────────────────────┘ └──────────────────┘ │
│  AI models loaded once on startup:                                   │
│  • Depth detector (tools/depth.py) — DA3DepthDetector only (DA3-     │
│    METRIC ONNX, native metric depth output, no scale alignment);     │
│    Sparse/Stereo obstacle detectors and the DA3 torch backend were   │
│    removed from this path                                            │
│  • GroundingDINO   — open-vocab object detection                     │
│  • DINOv2 ViT-S/14 — re-ID embeddings (cosine ≥ 0.75 = same target) │
│  • MiniLM (RagStore.embed_text) — text embeddings for on-device RAG │
│  • KokoroTTS — Synthesize RPC                                        │
│  • DA3 (torch) + RTAB-Map — live mapping pose/depth (MappingService, │
│    scan_server/ modules imported in-process, RTAB-Map pose only)     │
│  Gradio monitor dashboard (port 7860) — live view of actual RPC      │
│  traffic (services/activity_monitor.py), not client-reported mode —  │
│  see server_gui.py note                                              │
└─────────────────────────────────────────────────────────────────────┘

OCR.space (https://ocr.space/) — called directly by Android for reading
mode, no self-hosted OCR server (see "Reading-mode OCR — OCR.space +
block-level dedup + line-level filters" below; replaces the earlier
self-hosted paddle_ocr_server, deleted outright).
RTAB-Map pose service (scan_server/rtabmap_docker/, port 5556) — required
by MappingService; no per-device camera-IMU calibration needed.
```

---

## gRPC Services & Protobuf

**Single source of truth:** `client/android/app/src/main/proto/tracking.proto`
(Android's build generates its own Kotlin stubs from this at build time).
Python stubs are generated from the same file into `server/` and
`test_module/` — there's no more `client/proto/` copy since every other
Python client that needed it was deleted.

> **After editing `tracking.proto` regenerate Python stubs:**
> ```bash
> PROTO_DIR=client/android/app/src/main/proto
> python -m grpc_tools.protoc -I"$PROTO_DIR" \
>   --python_out=<dir> --grpc_python_out=<dir> "$PROTO_DIR/tracking.proto"
> ```
> Run once for `server/`, once for `test_module/`. Android regenerates its
> own Kotlin stubs automatically at build time via the protobuf Gradle
> plugin — no manual step needed there.

### TrackingService (port 50051)

| RPC | Input | Output | What it does |
|-----|-------|--------|--------------|
| `DetectObject` | prompt string + JPEG bytes | box_xyxy + score | GroundingDINO detection — request carries its own frame (no server-side "latest frame" cache any more, see below) |
| `GetEmbedding` | box_xyxy + JPEG bytes | float vector | DINOv2 ViT-S/14 embedding for re-ID |

Both RPCs' request messages carry `image_data` directly (added when the
Chat/VoiceChat/VoiceChatStream/StreamFrame RPCs were removed — those used to
keep a server-side `latest_frame` warm via continuous streaming; without
them, each call must bring its own frame). The only remaining caller is
`TrackingBackend.kt`'s local-ORB-tracking init/renewal loop.

`PerceptionService`/`MappingService` (below) are the RPCs Gemini's on-device
tool calls actually use — see "Client-Orchestrated Live Session".

### StatusService (port 50051)

| RPC | Input | Output | What it does |
|-----|-------|--------|--------------|
| `ReportMode` | mode string + target string | (empty) | Tells the server which `LiveSessionState.mode` the client just entered — carries no data any other service needs, exists purely so `server_gui.py`'s dashboard can select the right tab directly. See "Client-reported mode" below. |
| `ReportBeaconDirection` | azimuth_deg + muted bool | (empty) | Tells the server the HRTF beacon's actual final steering angle (post goal-bias, post EMA smoothing — all computed client-side). Same "dashboard-only, no other consumer" precedent as `ReportMode` — the server has no other way to know it. Called once per local-avoidance tick while walking/guiding is active. See "Local reactive HRTF obstacle-dodge" below. |

---

## Client-Orchestrated Live Session

**Status: complete, and the only client.** Android is now the sole client
this project ships — the Pi thin client, Mediator, Desktop operator GUI,
and every RPC/message that existed only for them (`MediatorService`, the
old zone-based `MapService`, `TrackingService`'s `Chat`/`VoiceChat`/
`VoiceChatStream`/`StreamFrame`, `server/live_session.py`'s `LiveAPISession`,
all of `server/live_tools/`) have been deleted, not just deprecated. The
server is now purely a heavy-compute + live-mapping backend — no
conversational state, no other client to keep compatible with.

**The plan**: move Gemini Live orchestration (the conversational session +
tool-calling loop) from the server onto the Android client itself — Android
holds the Gemini API key and talks to Gemini Live directly. The gRPC server
stops being a conversational "mediator" and becomes a pure heavy-compute /
mapping backend, called directly by an on-device tool-dispatch loop. Work
splits by weight: real models/heavy CV stay server-side (called remotely);
plain bookkeeping (memory storage, path-finding over a received grid,
proximity checks) becomes local Android logic; things that were previously
"calling an outside API via our server" (OCR, Gemini) become direct
Android→3rd-party calls, bypassing the server as a proxy. Named zones/labels
are dropped entirely — navigation targets landmarks/functional objects found
by the VLM, not zone AABBs. Mapping is no longer a separate offline
operator-led scan — it runs live, automatically, whenever the client's
guiding mode is on, fed by real streamed frames instead of a replayed
recorded dataset.

### RPCs (`tracking.proto`)

`TrackingService` keeps only `DetectObject`/`GetEmbedding` (still called
directly by `TrackingBackend.kt`); everything else Gemini's on-device tools
call goes through the two services below.

- **`PerceptionService`** — stateless heavy-compute primitives, implemented
  in `server/services/perception_servicer.py`, thin wrappers around the same
  `tools/*.py` model wrappers `TrackingServiceServicer` already uses (no new
  model code):
  - `AnalyzeFrame(image, ops: {DETECT, EMBED, DEPTH, TRAVERSABILITY, CORRIDOR},
    prompt?, box?) → detections[], embedding?, obstacle?, traversability?,
    corridor?` — one round trip for whatever combo a caller needs
    (`run_detection`/`check_obstacle` on-demand tools; walking mode's own
    periodic DEPTH-op polling was removed long ago, see "Local reactive HRTF
    obstacle-dodge" below), via `detector.detect_all()` (sorted by score,
    replaces the old single-best `detect()` for this path)/
    `embedder.get_embedding()`/`depth_detector.check_obstacle()`/
    `depth_detector.estimate_traversability()` (per-angle obstacle-clearance
    fan — GUIDING's local dodge signal, see "Local reactive HRTF
    obstacle-dodge" below; BOTH modes' step-down hazard check, see "Hazard
    warnings" below, reads its `dropoff_m` field). `CORRIDOR` op /
    `find_corridor()` — added for WALKING's corridor-lock beacon design,
    deleted outright once that design was superseded by "Grid-planned
    walking route" (walking's beacon comes from an occupancy-grid path now,
    not a per-frame corridor pick); the enum value and `corridor` response
    field are left in the proto (unused) rather than renumbered.
  - `Synthesize(text) → stream(PcmChunk)` — KokoroTTS, unchanged voice.
  - `Embed(text) → vector` — `RagStore.embed_text()` (new method, raw
    sentence-transformer encode with no storage/search attached — Android
    does its own on-device vector storage + cosine search).
- **`MappingService`** — live SLAM-style mapping + localization, implemented
  in `server/services/mapping_servicer.py`, reusing `scan_server`'s
  `StreamingScanSession`/`ScanSessionManager` pipeline via the same
  sys.path-import convention `grpc_server.py` already used for
  `live_session.py` and `tools/depth.py`'s DA3/mvs imports (files aren't
  physically relocated into `server/` yet — see the approved plan's Phase 2
  for that follow-up cleanup pass).
  - `UpdateMapping(stream MappingChunk) → stream MappingUpdate{pose, grid,
    grid_updated, landmarks, confidence, grid_delta, full_resync,
    reset_occurred}` — bidi stream. `reset_occurred` (new) is only ever set
    for `SessionMode.WALKING`'s pose-only sessions — see "Local SLAM-backed
    walking corridor-lock" below; `SCAN`/`GUIDING` leave it false.
    **RTAB-Map is the ONLY pose source used here** (not IMU+VO) —
    `MappingChunk.imu_samples` is accepted on the wire but not consumed; the
    old scan_server GUI's IMU+VO pose source is untouched and still used by
    that separate, still-intact offline tool. Buffers frames the same way
    `StreamingScanSession.push_frame` always has (mini-batch, default 4) —
    a `MappingUpdate` is only yielded once a batch is actually processed,
    not per raw frame. `grid_updated` mirrors `OccupancyMap._update_count`
    gating (skips sending anything grid-related when nothing changed since
    the last update). When it IS true, only ONE of `grid`/`grid_delta`
    actually carries data — see "Occupancy grid delta sync" below for which
    one and why (this replaced always re-sending the full grid on every
    change, which didn't scale as a session/map grew).
    **`landmarks` now stays EMPTY for the duration of an active stream** —
    GroundingDINO/backprojection no longer runs proactively per accepted
    frame during scanning (see "Novelty+blur frame gating and deferred
    landmark resolution" below); real positions only exist after
    finalize-time export at stream end, or via a `FindLandmark` query
    mid-session. This replaced the old behavior of streaming
    `session._raw_landmarks`'s raw, unclustered accumulation live.
  - `FindLandmark(location_id, query) → found, x, z, confidence,
    matched_label` — the ONLY place GroundingDINO runs now: on demand,
    against `location_id`'s session frame store, via
    `ScanSession.resolve_landmark()` (tag-match first, else a first-hit
    scan — see below). Uses `scan_manager.get()` (read-only, does not
    create a session). `ToolDispatcher.kt`'s `recomputeRoute()` calls this
    for EVERY guiding destination now (no live-streamed landmarks list to
    check first). **No disk-snapshot fallback any more** — see "Map
    persistence removed entirely" further below; a walking/guiding
    session can only resolve a destination a live SCAN already found in
    this same server-process lifetime.
  - On stream end (client closes/disconnects — `finally` block, so this
    fires on a clean stop or an abrupt drop): `stream.flush()` (last partial
    batch), `session.finalize_landmarks_flat()` (see below) — **no longer
    persists anything to disk** (originally wrote an `occupancy_snapshot.json`
    per location here; removed, see "Map persistence removed entirely").
  - `GetMapSnapshot`/`ListMappedLocations` — **REMOVED outright** (not
    just disabled) — see "Map persistence removed entirely" below.
  - Registered only if `RTABMAP_ADDR` is set and a dedicated DA3 estimator
    loads — `grpc_server.py` loads a SEPARATE `DA3Estimator` for this
    (doesn't reuse walking mode's `da3_onnx`), since concurrent-inference
    thread-safety of one shared instance across two independent live gRPC
    streams hasn't been verified; costs extra GPU memory for that
    correctness guarantee instead of assuming it's fine.

### Zone-free landmarks (`scan_server/scan_session.py`)

`finalize_landmarks_flat()` — additive alongside the existing zone-based
`finalize_landmarks()`. Does the same global overlap-merge clustering
(`semantic_mapper.cluster_landmarks()`) but returns the flat merged list
directly instead of bucketing each landmark into whichever `Zone` AABB
contains/is-nearest-to it — there's no zone to assign into any more. Both
now source their landmarks via `_resolve_all_frame_store_landmarks()` (see
below) instead of an immediate per-frame accumulation.

### Novelty+blur frame gating and deferred landmark resolution

**The "defer ALL GroundingDINO/backprojection to on-demand query time" half
of this section is superseded — see "Semantic mapper adapted to
frame_extractor's Gemini -> GroundingDINO-tiny pipeline" below.** The
novelty+blur gate itself (`OrbNoveltyGate`/`scan_session.py`'s Step 0/Step 3
gating, `StoredFrame`→`_PendingTagFrame` batching) is still exactly as
described here — only what happens to a frame ONCE it's accepted changed:
detection now runs immediately, batched, per accepted frame (frame_extractor's
own design), not deferred to a later on-demand query. Kept here for the
novelty-gating history, which is unchanged.

A real quality/cost problem, surfaced and resolved with the user during this
work: the live pipeline used to (1) select frames for the VLM landmark
extractor via a weak heuristic — "sharpest frame in each ~1s window"
(relative selection only, never a hard reject), falling back to a fixed
every-5th-frame cadence with no capture timestamp — with **zero real novelty
gating**, so a near-duplicate view and a genuinely new one were treated
identically; and (2) ran GroundingDINO detection + world-coordinate
backprojection **immediately, on every accepted frame, during scanning** —
expensive, and pointless for objects the user never actually asks about.

**Chosen strategy**: port `frame_extractor/`'s already-proven `OrbNoveltyGate`
(ORB descriptor match + Essential-Matrix-RANSAC against EVERY previously-
accepted frame, not just the last one — see that module's own docstring) and
its hard `min_sharpness` blur-reject threshold into the live pipeline, and
defer ALL GroundingDINO/backprojection to on-demand query time.

- **`scan_server/orb_novelty_gate.py`** (new) — duplicates
  `frame_extractor/extractor.py`'s `OrbNoveltyGate`/`_sharpness_score`/
  `_estimate_K`/`_rotation_deg` rather than importing them (same
  "separately deployed processes" reason `live_path_planner.py` duplicates
  `server/tools/grid_path_planner.py` — see that file's own docstring). Two
  structural differences from the original: `evaluate()`'s reference match/
  RANSAC loop is split into `_match_against_references()`, with a new
  `evaluate_with_keypoints()` entry point that skips ORB detection entirely
  — `scan_session.py`'s `FeatureTracker.track()` already detects ORB
  keypoints/descriptors once per frame for pose estimation (confirmed: for
  ALL THREE pose sources, including RTAB-Map, which runs it as a
  depth-consistency side channel) and this reuses `self.tracker._prev.
  keypoints/descriptors` immediately after that call rather than paying for
  a second detection pass. `decide_accept(...)` mirrors
  `frame_extractor/extractor.py`'s `extract_new_frames()` loop body's
  tested accept/reject boolean algebra exactly (novelty fraction/count +
  rotation guard + blur reject), copied rather than re-derived.
- **RTAB-Map pose mode gets a cheaper, native novelty signal** instead of a
  redundant Python ORB pass: `rtabmap_server.cc`'s TRACK reply now carries a
  4th field, `inlier_fraction` (`float`, `odomInfo.reg.inliers /
  max(odomInfo.reg.matches, 1)`) — `rtabmap::OdometryInfo`'s nested
  `RegistrationInfo reg` member is already populated as a side effect of the
  existing `odom->process()` call, so reading it costs no new SLAM work.
  Confirmed via the vendored header in the built image
  (`/usr/local/include/rtabmap-0.23/rtabmap/core/RegistrationInfo.h`) that
  `reg.inliers`/`reg.matches` are the right fields — NOT `new_node_id`,
  which reflects keyframe-spacing/displacement policy (how far the camera
  moved), not visual-overlap with the existing map, and would conflate two
  different questions if reused for this. `scan_session.py` uses
  `1.0 - inlier_fraction` as `new_fraction`, with an approximate rotation
  guard computed client-side (diffing the current pose against
  `self._rtabmap_last_accepted_pose`, since the wire has no per-frame match
  COUNT, only a ratio — `min_new_count` is a deliberate no-op for this
  branch, passed as its own threshold so the check always passes). Required
  a docker image rebuild (`docker build --load -t tracking-rtabmap -f
  scan_server/rtabmap_docker/Dockerfile scan_server/rtabmap_docker`) —
  verified via a real TRACK round trip: a first-ever frame reads
  `inlier_fraction=0.0` (no prior map), an exact repeat reads `1.0`
  (well-explained by the existing map), matching expected direction.
  `rtabmap_client.py`'s `_TRACK_TAIL_FMT` extended `"<Bi"` → `"<Bif"`, with
  the same layered backward-compat fallback `node_id` itself used when it
  was added (`TrackedFrame.inlier_fraction` defaults to `1.0` — "not novel"
  — against an older server build, so an un-rebuilt server degrades safely
  rather than crashing).
- **One shared per-frame decision, three consumers**: `process_frames_batch()`
  computes `(accepted, sharpness)` once per frame (in Step 2's per-pose-
  source loops, right where `FeatureTracker.track()` already runs) and
  reuses it for: (1) cloud/TSDF fusion — `accepted` is now an ADDITIONAL
  condition alongside the existing depth-consistency `trustworthy` check
  (non-RTAB-Map branch only, same as that check), logged with a
  `[novelty-blur]`-prefixed print matching `[depth-consistency]`'s style;
  (2) frame-store admission; (3) VLM tagging batching. `ScanSession`'s
  `configure_novelty_gate()`/`ScanSessionManager.configure_novelty_gate_
  defaults()` follow the exact same live-tunable pattern as `OccupancyMap`'s
  `configure_occupancy_map()`/`configure_occupancy_defaults()` — **one real
  deviation, called out rather than hidden**: OpenCV gives no way to
  reconfigure `cv2.ORB_create`'s `nfeatures` post-construction, so changing
  an ORB-detector-affecting knob must recreate `self.novelty_gate`, which
  also discards its accumulated reference-frame set (unlike
  `OccupancyMap.set_params()`, which mutates in place with zero side
  effects).
- **`StoredFrame`** (new dataclass, `scan_session.py`) — one novelty+blur-
  gated accepted frame: JPEG-compressed bytes (`cv2.imencode`, not a raw
  ndarray, to bound memory), `depth_map`, `world_pose`, `K`, `frame_idx`,
  `timestamp_ns`, and `landmark_tags: List[str]` (filled in once its VLM
  batch completes). Kept in `ScanSession._frame_store` — **session-scoped,
  in-memory only, never persisted to disk** (an explicit, accepted
  limitation, confirmed with the user — lost on server restart; cleared by
  `reset_cloud()` on every fresh `StreamingScanSession`, same as
  `_raw_landmarks`). No cap/eviction policy needed: naturally bounded by the
  novelty gate itself (sparse, non-redundant frames only).
- **`SemanticMapper.tag_landmarks_batch(frames) -> List[List[str]]`**
  (replaces `consider_frame()`/`extract_landmarks()`/the old windowed
  selection entirely) — ONE VLM call across up to `IMAGES_PER_PROMPT` (5)
  frames, asking for **per-image landmark/object NAME tags only** — N lines
  of comma-separated names, one line per image, no shared
  `grounding_dino_prompt`, no boxes, no world coordinates. Stateless —
  `ScanSession` now owns the buffering (`self._tag_pending`, a subset of
  `self._frame_store`, flushed once it reaches `IMAGES_PER_PROMPT` or at
  finalize time) instead of `SemanticMapper` itself. Parsing is defensive
  (pads/truncates + logs a warning if the VLM doesn't return exactly N
  lines, mirroring the old `_parse_vlm_response`'s defensive style — never
  raises). `SemanticMapper._detect_and_backproject()` (GroundingDINO-detect
  + depth-median-sample + 4-corner backprojection) is UNCHANGED, just no
  longer called proactively — only from the deferred resolution below.
- **`ScanSession.resolve_landmark(query) -> Optional[Landmark]`** — the
  ONLY place GroundingDINO runs now, on demand: **Tier 1**, the first
  stored frame whose VLM tag list already mentions `query`
  (case-insensitive substring, either direction) — GroundingDINO runs on
  just that ONE frame to get its box; if it doesn't confirm a box there,
  falls through to **Tier 2**, a first-hit scan across every remaining
  stored frame, in order, until one hits (covers objects the VLM never
  proactively tagged, e.g. "water bottle" — GroundingDINO is open-
  vocabulary). Deliberately "first find wins," not "best confidence across
  all frames" — cheap, and Tier 1 (the common case) costs exactly one
  detection call. Snapshots `_frame_store` under `self._lock` then releases
  it before the (possibly slow) detector calls, so a concurrent
  `process_frames_batch()` isn't blocked.
- **`ScanSession._resolve_all_frame_store_landmarks()`** — the finalize-time
  (`finalize_landmarks()`/`finalize_landmarks_flat()`) replacement for the
  old immediate accumulation: flushes any leftover partial tag-pending
  batch, collects every UNIQUE tag seen across the session's frame store
  (first-seen order), resolves each one via `resolve_landmark()` (one
  GroundingDINO call per unique tag, not per frame), then runs the existing
  `cluster_landmarks()` as a final dedup pass over the resolved list (catches
  near-duplicate resolutions from synonym tags, e.g. "chair" vs. "office
  chair" landing at nearly the same spot). Must be called WITHOUT
  `self._lock` already held — `resolve_landmark()` acquires it internally
  and `threading.Lock()` is not reentrant; `finalize_landmarks()`/
  `finalize_landmarks_flat()` were restructured so their own locking no
  longer wraps this call.
- **`FindLandmark` RPC** (`tracking.proto`, `mapping_servicer.py`) — thin
  wrapper: `scan_manager.get(location_id)` (read-only) →
  `session.resolve_landmark(query)`. See the MappingService bullet above.
- **`start_scan()`/`stop_scan()`** (new voice tools, `ToolDeclarations.kt`/
  `ToolDispatcher.kt`) — lets the user proactively trigger the mapping
  pipeline (frame gate → point cloud → occupancy map → landmark tagging)
  without a navigation destination; mirrors `start_walking()`/
  `stop_walking()` exactly (`state.mode = "scanning"`, same
  `startMappingStream()`/`stopMappingStream()`, no waypoint tracking, no
  obstacle polling). System prompt instructs Gemini to verbally ask the
  user to pan slowly around the space after calling it. `MainViewModel.kt`'s
  frame-feed gate (`sessionState.mode == "guiding" || "walking"`) extended
  to include `"scanning"` so `feedMappingFrame()` actually keeps flowing.
  No new server-side work — reuses the exact same `UpdateMapping` pipeline.
- **`recomputeRoute()` fixed** (`ToolDispatcher.kt`) — was a literal no-op
  placeholder (confirmed by reading the file: guiding to even an
  already-known landmark never routed anywhere). Now `suspend`, and always
  resolves `state.guidingDestinationLabel` via a fresh `FindLandmark` call
  (no live-streamed landmarks list to check first — see above), then feeds
  a hit's `(x, z)` into the already-existing but previously-unwired
  `LocalPathPlanner`/`HrtfBeacon` to compute `state.navWaypoints` — this is
  the piece that makes guiding actually route anywhere for the first time.
  Landmark-name matching deliberately stays simple (server-side
  case-insensitive substring, either direction) — matches whatever
  `resolve_landmark()` (`scan_session.py`) can find among already-resolved
  landmarks (see "Semantic mapper adapted to frame_extractor's RAM++ ->
  GroundingDINO-tiny pipeline" below for why there's no more on-demand
  detection fallback to lean on for a near-miss). **Known follow-on, not yet
  built**: nothing memoizes "already tried and failed this session," so an
  unresolved destination retries the RPC on every grid update — worth a
  debounce if it turns out to spam the RPC in practice.

### Semantic mapper adapted to frame_extractor's Gemini -> GroundingDINO-tiny pipeline

Requested directly by the user: `scan_server/semantic_mapper.py` now adapts
`frame_extractor/tagging.py` + `app.py`'s pipeline wholesale, instead of the
Gemma-VLM-tag-then-defer-GroundingDINO design the previous section
describes. Every novelty+blur-gated accepted frame is tagged AND detected
**immediately**, batched, exactly like `frame_extractor/app.py`'s "Extract
new frames" button does for its own output:
1. Gemini (`gemini-3.1-flash-lite` via the `google-genai` SDK, no local
   checkpoint) proposes open-set tags for the frame — same multi-image-
   per-call convention `gemma_vlm.py`'s `GemmaVLMClient` already established
   in this codebase, just aimed at short detection-prompt tags instead of a
   navigation-landmark description.
2. Those tags become GroundingDINO-**tiny**'s own detection prompt for that
   SAME frame (`tagging.py`'s `_build_prompt` — lowercase, article-prefixed,
   `" . "`-separated), so what gets boxed tracks what Gemini actually saw.
3. Every box is immediately backprojected into a world `(x, z)` `Landmark`
   using the frame's own depth map + pose + intrinsics (already available
   at frame-acceptance time) — the same backprojection math the old design
   used, just no longer deferred.

**Bug found from a real scan's console output: landmark names carried a
leading article ("a nightstand", "an outlet")** — `post_process_grounded_
object_detection`'s returned `label` is the matched TEXT SPAN decoded
straight out of the GroundingDINO prompt's own tokens, and since
`_build_prompt()`'s convention prefixes every tag with "a "/"an " (step 2
above), that article rode along into the detection label verbatim. Fixed
via `tagging.py`'s new `_strip_leading_article()`, applied to every
`TagDetection.label` in `_detect_batch()` — so `Landmark.name` (and
anything downstream: `resolve_landmark()`'s substring match, spoken
navigation destinations) sees the plain noun, not the prompt artifact.

**`frame_extractor/tagging.py` itself was also changed in the same pass**:
its `FrameTagger` originally tagged via RAM++ (a local open-set image
tagging model, ~3GB checkpoint) — swapped for Gemini specifically because a
RAM++ checkpoint isn't available in every deployment of this pipeline
(notably `scan_server/`'s own environment). `FrameTagger.__init__` dropped
`ram_checkpoint`/`ram_image_size`/`ram_tag_threshold` in favor of
`gemini_api_key`/`gemini_model_id` (`DEFAULT_GEMINI_MODEL =
"gemini-3.1-flash-lite"`); `_tag_batch()` now calls
`self._gemini_client.models.generate_content(model=gemini_model_id,
contents=[*pil_images, prompt], config=types.GenerateContentConfig(
system_instruction=..., temperature=0.1, response_mime_type="text/plain"))`
instead of a local `ram_model.generate_tag()` forward pass, with the same
defensive N-line parse (`_parse_tag_response`, pad/truncate + per-line
comma-split, capped at `max_tags_per_frame`) `semantic_mapper.py`'s old
VLM-tag parser used. `_detect_batch` (GroundingDINO-tiny) is completely
UNCHANGED — still a local transformers model, still prompted by whatever
tags the tagging step returns, regardless of which model produced them.
`frame_extractor/app.py`'s own standalone GUI was updated in lockstep — the
"RAM++ checkpoint path" textbox became "Gemini API Key"/"Gemini tagging
model id" textboxes (reusing `GEMINI_API_KEY`/new `GEMINI_TAGGING_MODEL_ID`
env vars as their defaults), `_get_tagger()`'s cache key and `run()`'s
signature updated to match. `frame_extractor/requirements.txt`'s
`recognize-anything` (RAM++ package) install line was dropped; the
`transformers==4.46.3` pin stays (still needed for GroundingDINO-tiny's
`post_process_grounded_object_detection` signature, unrelated to RAM++) —
`google-genai` needs no separate entry, already pinned in
`scan_server/requirements.txt`. `patch_ram_package.py`/RAM++ itself are
left in the repo, unreferenced, not deleted, in case RAM++ tagging is ever
revisited.

**`SemanticMapper` (`scan_server/semantic_mapper.py`)** now wraps a single
`FrameTagger` (imported directly from `frame_extractor/tagging.py` via a
sys.path bootstrap, not duplicated — unlike `orb_novelty_gate.py`'s
deliberate duplication of `frame_extractor/extractor.py`, there's no
"separately deployed process" reason to duplicate this one, since
`scan_server/` already imports across that boundary for other things) in
place of the old `vlm`/`detector` pair. `tag_and_backproject_batch(frames_bgr,
depth_maps, world_poses, Ks, frame_idxs)` replaces `tag_landmarks_batch()` +
`_detect_and_backproject()` — one call does tag, detect, AND backproject,
returning one `Landmark` list per input frame. `cluster_landmarks()` is
unchanged (same overlap-merge logic) — see "Confidence-weighted, distance-
based landmark merging" below for a later refinement to that logic.

**`scan_session.py`** — `StoredFrame`/`_frame_store` are gone; replaced by
`_PendingTagFrame`/`_tag_pending`, a short-lived buffer (still batched at
`SemanticMapper.IMAGES_PER_PROMPT`, same batching motive as before —
amortizing per-call overhead, now a Gemini API round trip + GroundingDINO
CUDA launch, across a batch) holding raw frames until a full batch is ready
for `_flush_tag_pending()`, which now appends every resolved `Landmark`
straight into `self._raw_landmarks` instead of recording per-frame tag
strings. This is a real, deliberate reversion of a previous optimization:
**`_raw_landmarks` is populated LIVE again during an active scan stream**
(it had been left permanently empty for the stream's duration by the
deferred design — see the previous section's own note) —
`MappingService.UpdateMapping` already read `session._raw_landmarks` for
its per-update `landmark_count`/`landmarks` proto fields, so this switch
makes those fields meaningful again during scanning, not just after
finalize. `walking_lite` sessions are unaffected either way — they still
skip semantic tagging entirely (see "Session-mode pipeline split" below).

- **`resolve_landmark(query)`** no longer runs any detection — GroundingDINO-
  tiny already ran on every accepted frame, so this is now a plain
  case-insensitive substring search (either direction) over
  `self._raw_landmarks`, returning the highest-confidence match. **Real,
  accepted trade-off**: a landmark Gemini never tagged in ANY accepted frame
  is now never findable at all — the old design's Tier-2 "scan every stored
  frame with the literal open-vocabulary query" fallback (e.g. finding "water
  bottle" even though the VLM never said "water bottle") no longer exists,
  since there's no stored-frame archive left to re-scan on demand. This is
  an inherent consequence of adopting frame_extractor's pipeline as-is
  (GroundingDINO-tiny there is only ever prompted with the tagging step's
  own tags).
- **`_resolve_all_frame_store_landmarks()`** → **`_finalize_raw_landmarks()`**
  — no longer resolves anything (everything's already resolved live); just
  flushes any leftover partial batch and runs `cluster_landmarks()` once
  over the accumulated `_raw_landmarks`, same as before.
- `finalize_landmarks()`/`finalize_landmarks_flat()` call sites updated
  accordingly; behavior (zone assignment / flat list) unchanged.

**Model construction (`scan_server/scan_server.py`, `server/grpc_server.py`)**
— both now build a `FrameTagger(gemini_api_key, gemini_model_id,
gdino_model_id, device)` instead of a `GemmaVLMClient` + `GroundingDINODetector`
pair. New env vars: `GEMINI_TAGGING_MODEL_ID` (default `FrameTagger.
DEFAULT_GEMINI_MODEL` = `gemini-3.1-flash-lite`; reuses the existing
`GEMINI_API_KEY` for auth, no new key needed), `GDINO_TAGGING_MODEL_ID`
(default `IDEA-Research/grounding-dino-tiny`) — deliberately separate from
`DA3_ONNX_PATH`/other model env vars, and distinct from the full-size
`GroundingDINODetector` (`server/tools/detector.py`) TrackingService still
uses, which stays untouched. Both entry points add `frame_extractor/` to
`sys.path` (same convention `scan_server/` already uses for
`server/tools/*`) so `from tagging import FrameTagger` resolves.
`SCAN_GEMMA_MODEL_ID`/`gemma_vlm.py` are no longer used by any live path
(the file itself is left in place, unreferenced, not deleted).

**`ScanSessionManager.semantic_mapper_model_id`/`set_semantic_mapper_model`**
— the old live VLM-model-hot-swap (`scan_gui.py`'s "Apply" button) has no
equivalent here: Gemini + GroundingDINO-tiny are configured once at server
startup, not a single swappable model id. `semantic_mapper_model_id` now
returns a fixed descriptive string (reading `FrameTagger._gemini_model_id`);
`set_semantic_mapper_model` was removed, and `scan_gui.py`'s textbox is now
read-only with the Apply button repurposed as an inert "Info" button
explaining why.

**Known, accepted limitations** (direct consequences of adopting this
pipeline wholesale, not bugs):
- Open-vocabulary queries outside whatever the tagging step actually tags
  in a session are unreachable (see `resolve_landmark()`'s note above) — a
  real narrowing of what's findable vs. the old design's Tier-2 fallback.
- GroundingDINO-**tiny** (not the full-size model TrackingService still
  uses) trades some detection accuracy/recall for speed, matching
  `frame_extractor/app.py`'s own choice.
- Tagging is now a network round trip (Gemini API) per flushed batch,
  instead of a local forward pass — adds latency + an external dependency
  vs. RAM++, in exchange for not needing a local checkpoint anywhere this
  pipeline runs.
- Not verified end-to-end against a live scan/device from this environment
  — compile-verified only (`python3 -m py_compile`), same standing caveat
  as other rounds of this project's work.

### Distance-based landmark merging (confidence picks the winner) (`scan_server/semantic_mapper.py`)

Requested directly by the user: batching several frames per Gemini/
GroundingDINO-tiny call (`IMAGES_PER_PROMPT`, see above) means overlapping-
FOV frames in the same batch routinely re-detect the same physical object —
`cluster_landmarks()`'s job is cleaning that up before landmarks reach
`_raw_landmarks`/export, and its merge criterion + merged-position formula
were both too narrow for that in practice.

- **Merge criterion — distance OR overlap, not overlap alone.** A same-label
  pair now merges if EITHER their centers are within `MERGE_DISTANCE_M`
  (0.75m — same order of magnitude as typical indoor furniture) of each
  other, OR their footprints still overlap by `OVERLAP_MERGE_RATIO` (0.5,
  unchanged). Distance alone catches the common case this was built for:
  the same object seen from two different angles across a batch can
  backproject to two boxes that barely overlap at all, even though their
  centers land close together; overlap alone is kept as a second trigger
  for a large object whose two detected centers happen to land farther
  apart than `MERGE_DISTANCE_M` despite the boxes clearly describing the
  same thing. `_center_distance()` (new) — plain X-Z Euclidean distance
  between two `Landmark`s' `(x, z)`. Connected-components clustering
  (unchanged) still handles chains of 3+ nearby/overlapping detections
  correctly regardless of which trigger connected each pair.
- **Merged position — highest-confidence member's own position, not an
  average.** `_merge_landmark_group()` previously placed a merged landmark
  at the midpoint of the union of every member's AABB — a value no
  individual detection actually reported. A confidence-weighted average
  was tried next, then reverted per direct user feedback in the same
  round: the merged position should be a REAL detection's own reported
  center, not any blended point — so `_merge_landmark_group()` now simply
  takes `x`/`z` straight from `max(members, key=confidence)`, same as it
  already did for `confidence`/`name`/`frame_idx`/`footprint_corners`.
  `footprint_min`/`footprint_max` (union AABB across all members) is the
  only field that still reflects the whole group rather than one member.
- Verified via two synthetic tests (no device needed — pure `Landmark`
  dataclass math): (1) two same-label landmarks 0.3m apart with confidences
  0.9/0.4 merge into one landmark sitting exactly at the 0.9-confidence
  member's own (x, z) — not a blended point — while a third same-label
  landmark 5m away stays separate; (2) a 3-member chain (0.3m spacing
  between consecutive members, but the 1st and 3rd are 0.6m apart — outside
  `MERGE_DISTANCE_M` on their own) still merges into a single landmark via
  transitive connectivity.

### Client-side frame selection (blur filtering moved off the server, then removed for mapping modes)

Confirmed with the user: blur/clarity filtering is now Android's job, not
the server's — and, in a later round, confirmed OFF entirely for mapping
modes specifically (both client- and server-side, see "Blur filtering
removed for scan/walking" below). `CameraManager.kt` runs **two
independent selection policies**, chosen per-frame based on
`sessionState.mode` (`MainViewModel.kt`'s frame collector forwards it
verbatim into `cameraManager.mappingMode` every processed frame — `""` for
non-mapping modes) — replacing the old fixed `targetFps` and an earlier,
now-superseded window+clearest-frame design:

- **Mapping modes (guiding/scanning — walking REMOVED from this set, see
  below)** — NO blur/clarity filtering: whichever frame arrives once the
  interval has elapsed since the last send is forwarded directly, no
  window/candidate comparison at all. **Guiding uses `frameIntervalMs`**;
  **scanning uses its own `scanIntervalMs`** — confirmed with the user: a
  scan pass wants denser frame coverage for reconstruction/landmark
  tagging than ambient guiding steering needs, so the two are
  independently tunable rather than sharing one control. Both stay
  ms-based internally (CameraManager.kt/persistence unchanged) but
  SettingsScreen exposes them as plain FPS number-input fields (not
  sliders) — "Mapping FPS" (0.2–10, default 1.0 = 1000ms) and "Scan FPS"
  (2–20, default 10.0 = 100ms) — converted fps↔ms only at the UI boundary
  (`doConnect()`'s `(1000f / fps).roundToInt()`), confirmed with the user
  as the preferred input style over a slider. Switching mode away from
  mapping, OR between guiding/scanning (different interval), resets the
  send-gate so a stale timestamp from a different mode doesn't suppress
  the new mode's first send — tracked via `activeMappingSubmode`, reset
  whenever `mappingMode` changes.
- **Everything else (tracking/reading/Q&A/idle/walking)** — still
  blur-aware: `recentBufferMs` (SettingsScreen slider, 0–1000ms, 50ms
  steps, default 100) drives a small rolling buffer of the last
  `recentBufferMs` of frames, no window boundaries or send-gap logic at
  all. `clearestRecentFrame()` is a pull-based accessor — the sharpest
  frame currently in the buffer, used exactly where `ToolDispatcher.kt`'s
  `latestFrame()` closure is called (OCR, `run_detection`, tracking-init
  retries, AND now walking/guiding's own local-avoidance tick — see "Local
  reactive HRTF obstacle-dodge"): each of those wants "the current frame"
  on demand, not a subscribed stream. Continuous per-frame consumers (hand
  tracking, local ORB tracking, the UI overlay) still get a steady trickle
  via the same `frameFlow`, emitted by `handleRecentEmit()` at most once
  per `recentBufferMs` from whatever's currently sharpest in the buffer —
  a real-time counterpart to the same pull, not a second independent
  mechanism. **Walking moved here from the mapping-modes bucket above**
  once it dropped `MappingService`/RTAB-Map entirely — see "Local reactive
  HRTF obstacle-dodge."

See the Key Files Map entry above for the exact algorithm.

**Investigated whether any server-side blur-reject needed removing to
match — found there wasn't one active to begin with**: `orb_novelty_gate.py`'s
`decide_accept()` already treats `min_sharpness <= 0` as "blur gating
disabled entirely" (its own docstring says so), and `scan_session.py`'s
`DEFAULT_MIN_SHARPNESS = 0.0` is the value every live `ScanSession`
actually uses, since `mapping_servicer.py` doesn't call
`configure_novelty_gate()`/`ScanSessionManager.configure_novelty_gate_
defaults()` to override it (a brief exception during "DA3 model default +
per-frame processing" below's live debugging turned this ON with real
`SCAN_MIN_SHARPNESS`/`WALKING_MIN_SHARPNESS` values — see "Blur filtering
removed for scan/walking" further below for why that was reverted). Only
`frame_extractor/`'s standalone offline tool and `scan_gui.py`'s
manually-configurable GUI slider ever actually exercise blur-rejection in
practice now.

### Blur filtering removed for scan/walking (client + server)

Confirmed with the user: no blur/clarity filtering at all for mapping
modes (scan or walking/guiding), on either side of the wire — a reversal
of the brief `SCAN_MIN_SHARPNESS`/`WALKING_MIN_SHARPNESS` experiment from
"DA3 model default + per-frame processing" below, which turned out to add
complexity without being what was wanted here.

- **Android**: `CameraManager.kt`'s mapping-mode branch no longer buffers a
  window or compares candidate sharpness at all (see "Client-side frame
  selection" above) — it just forwards whichever frame arrives once
  `frameIntervalMs`/`scanIntervalMs` has elapsed since the last send.
  `computeSharpness()` still runs per frame (needed for the unrelated
  `recentBufferMs` path — tracking/reading/Q&A, unaffected by this change),
  just isn't consulted for the mapping-mode send decision any more.
- **Server**: `mapping_servicer.py` no longer calls `configure_novelty_gate()`
  at all when constructing a live `StreamingScanSession` — `self._min_sharpness`
  stays at `DEFAULT_MIN_SHARPNESS` (0.0), which both `process_frames_batch()`'s
  Step 0 pre-DA3 pre-check and the per-frame novelty+blur gate already treat
  as "blur gating off entirely" (existing convention, no new code needed).
  Walking/guiding still skip the NOVELTY check too (unchanged from "Walking/
  guiding skip the novelty gate entirely" below) — with blur now also off,
  a walking/guiding frame is accepted whenever RTAB-Map actually produced a
  pose for it, full stop; nothing else gates fusion into the occupancy grid.
- **Debug logging trimmed to walking/guiding only**: the per-batch
  `[timing] depth estimation`/`[timing] pose computation`/`[timing]
  back-projection + frame-store tagging` prints in `process_frames_batch()`
  now only fire when `walking_lite` is true — scan mode's console output
  was too noisy to read through during live debugging. The per-node
  `[depth-consistency] RTAB-Map node N flagged untrustworthy` print was
  removed outright (not gated) — it's scan-only-meaningful bookkeeping
  (only `_rtabmap_process_nodes`/`get_cloud()`, which walking_lite skips
  entirely, ever reads `_rtabmap_untrusted_node_ids`), so gating it to
  walking would have made it print something misleading (a node's geometry
  "will be skipped when pulled via get_cloud()" during a mode that never
  calls `get_cloud()` at all). `[novelty-blur]`/`[depth-consistency] frame
  REJECTED` (the per-frame ones inside Step 3's back-projection loop) were
  ALREADY implicitly walking/guiding-only on the live RTAB-Map path — their
  guard is `if not use_rtabmap_pose or walking_lite:`, and `mapping_servicer.py`
  always passes `pose_src="RTAB-Map"`, so that condition reduces to just
  `walking_lite` there — no change needed. The `[ScanSession:...] RTAB-Map
  tracking LOST on N/M frames` line and the `[MappingService] MappingUpdate
  #N -> ...` summary line are unconditional regardless of mode (both are
  single concise lines, not per-frame diagnostic spam, and pose-lost status
  now also has a GUI home — see below).
- **RTAB-Map pose-lost now visible in the dashboard, not just console**:
  `session.last_rtabmap_lost`/`last_rtabmap_total` (already tracked per
  batch — see Step 2) are now passed into `ActivityMonitor.record_mapping()`
  as `rtabmap_lost`/`rtabmap_total`. `server_gui.py`'s Mapping tab shows
  this two ways: a red `RTAB-Map TRACKING LOST (N/M)` overlay burned into
  the annotated frame image when `rtabmap_lost > 0` (frame_rgb is RGB
  order, so red is `(255,0,0)` — matches the existing green pose text/
  magenta HRTF marker convention on the same overlay), and a line in the
  Detail textbox (`⚠ RTAB-Map tracking: LOST N/M frames this batch` or
  `RTAB-Map tracking: OK (M/M)`).

### RTAB-Map-native ground segmentation (replaces the height-only heuristic for RTAB-Map pose mode)

Requested directly by the user, after confirming with the RTAB-Map docker
daemon actually running (`docker info` reachable) that a full rebuild +
live smoke test was possible from this environment. Previously, EVERY pose
source (IMU+VO/VO and RTAB-Map alike) classified ground-vs-obstacle purely
by height above a fitted/estimated `ground_y` — `OccupancyMap.update()`'s
own `height < OBSTACLE_MIN_H` check. RTAB-Map mode now instead uses RTAB-Map's
OWN ground/obstacle segmentation (`util3d::segmentObstaclesFromGround` — the
exact function RTAB-Map's own `OccupancyGrid`/`LocalGridMaker` classes use
internally, previously never called by this project's server at all) as the
ground/not-ground BOUNDARY decision; height is still computed and still used
for the `OBSTACLE_MAX_H` ceiling filter and the LOW_STEP_OVER vs. normal-
OBSTACLE tiering downstream. IMU+VO/VO sessions are unaffected — they have no
per-point RTAB-Map segmentation to offer, so `OccupancyMap.update()`'s new
`point_is_ground` param stays `None` for them and the original height-only
path runs unchanged.

**Coordinate mismatch, found by reading RTAB-Map's own source (not assumed):**
`segmentObstaclesFromGround()` hardcodes its reference "ground normal" as
world `Eigen::Vector4f(0,0,1,0)` — confirmed in `util3d_mapping.hpp`, which
always calls `normalFiltering(..., Eigen::Vector4f(0,0,1,0), ...)`
regardless of any parameter passed in, so there is no public way to tell it
"up" is a different axis. This project's own convention throughout
(`HrtfBeacon.kt`, `RotationTracker.kt`, `occupancy_map.py`'s `ground_y`) is
camera-optical X-right/Y-down/Z-forward, where "up" (away from the floor)
is `-Y`, not `+Z`. Fixed by building a temporary axis-remapped COPY of the
cloud purely for this call — `(x, y, z) -> (x, z, -y)`, maps our `-Y`-up
onto `+Z`-up — and applying the returned ground/obstacle point INDICES
(remap-invariant) back onto the real, un-remapped cloud; the remapped copy
is discarded immediately, never sent anywhere. `viewPoint` is passed as the
camera's own local origin `(0,0,0,1)` (remapped the same way, matching how
RTAB-Map's own `LocalGridMaker` passes the SENSOR's own local position —
not this function's default `(0,0,100,0)`, which models a viewpoint far
above pointing down, wrong for a camera at roughly head/room height looking
sideways).

- **`rtabmap_server.cc`** — new `segment_ground_flags()` (the axis-remap +
  `segmentObstaclesFromGround` call, using RTAB-Map's own `Grid/*` defaults
  confirmed against `Parameters.h`: `NormalK=20`, `MaxGroundAngle=45°`,
  `ClusterRadius=0.1`, `MinClusterSize=10`, `FlatObstacleDetected=true`,
  `MaxGroundHeight=0` disabled). Runs per-node, in the node's own
  CAMERA-LOCAL frame, BEFORE the world-pose transform — matches RTAB-Map's
  own default pipeline order (`Grid/PreVoxelFiltering=true`: voxel-filter
  first, segment second) since `reconstruct_node_cloud()` already voxelizes
  before this runs. `GET_CLOUD`'s wire reply gained a trailing
  `uint8[point_count] is_ground` array per node, after the existing `rgb`
  array — documented in the header comment.
- **`rtabmap_client.py`** — `ReconstructedNode` gained `is_ground: np.ndarray`
  (bool, aligned with `points`/`colors`). Same backward-compat convention
  `node_id`/`inlier_fraction` already established: an un-rebuilt server that
  didn't send the trailing bytes degrades to all-`False` rather than
  crashing.
- **`scan_session.py`** — new `_voxel_majority_flags(points, flags, centers,
  vsize)`: aggregates the per-POINT `is_ground` into a per-VOXEL majority
  vote aligned with `voxelize_cloud()`'s own returned `centers`. Open3D's
  `VoxelGrid` exposes no per-voxel source-point membership directly, so this
  independently recomputes each point's grid index using the exact same
  fixed anchor (`_VOXEL_GRID_MIN_BOUND`) and voxel size `voxelize_cloud()`
  itself uses — `create_from_point_cloud_within_bounds`'s own `origin` is
  guaranteed to equal that anchor exactly (see `voxelize_cloud`'s own
  docstring), so the same point always buckets into the same voxel index
  either way. A tie resolves to obstacle (the safer default). Wired into
  `_rtabmap_process_nodes()`: `is_ground` is threaded through the SAME SOR
  keep-mask that already filters `points`/`colors` (both the cached-mask and
  fresh-SOR paths), so all three arrays stay index-aligned through to
  `voxelize_cloud()`, then majority-voted per voxel and passed to
  `occupancy_map.update(..., point_is_ground=...)`.
- **`occupancy_map.py`** — `OccupancyMap.update()` gained `point_is_ground:
  Optional[np.ndarray] = None`. When given (and correctly aligned — a
  misaligned length falls back to height-only rather than raising or
  misapplying flags to the wrong points), it decides ground-vs-obstacle
  per point directly instead of the `height < OBSTACLE_MIN_H` comparison;
  height is still computed for the ceiling filter and downstream tiering.

**Verified two ways, both against the REAL rebuilt `tracking-rtabmap` docker
image (not assumed/simulated)**:
1. A synthetic scene (uniform-noise-textured RGB for real ORB/GFTT features,
   a depth map constructed so the lower half of the frame hits a flat floor
   at camera-local `Y=+1.0` and the upper half hits a "wall") sent through a
   real `TRACK`/`GET_CLOUD` round trip: ground points came back at EXACTLY
   `Y=1.00` (min=max=mean), every obstacle point on the correct other side —
   direct confirmation the axis-remap orientation is correct, not just
   plausible.
2. The same node's cloud run through the real `voxelize_cloud()` ->
   `_voxel_majority_flags()` -> `OccupancyMap.update(point_is_ground=...)`
   pipeline end-to-end: `ground_y` converged to `1.025` (true value `1.0`),
   ground/obstacle cells classified with no height threshold involved in the
   boundary decision at all.

**Known, accepted limitations**: `groundNormalsUp` left at RTAB-Map's own
default (`0.0`) — its exact effect wasn't independently verified beyond
"the two synthetic tests above passed," since it disambiguates normal
direction in cases the tests didn't specifically construct for. Not
verified against a REAL indoor recording (textured real furniture/floor
materials, real lighting/shadows, real depth noise) — only the synthetic
uniform-noise scene above; a real scan may need `Grid/MaxGroundAngle`/
`ClusterRadius` retuned if the defaults (inherited from RTAB-Map's own
LIDAR/robot-oriented tuning) don't generalize well to this project's
monocular-depth-estimated, handheld-camera use case.

### Occupancy-grid persistence across sessions (coarse re-seed)

**REMOVED — see "Map persistence removed entirely" further below.** This
whole re-seed mechanism (`occupancy_snapshot.json`, `OccupancyMap.
seed_from_summary()`, `MappingServiceServicer._save_snapshot()`/
`_load_snapshot()`, the `GetMapSnapshot`/`ListMappedLocations` RPCs) was
deleted per a later, direct user request: "remove the map storing
entirely, just new map every scan, guide." `seed_from_summary()` itself is
left in `occupancy_map.py`, unreferenced (same "kept in case revisited"
precedent this codebase uses elsewhere) — everything that called it is
gone. Kept here for history/rationale only.

A real design gap, surfaced and resolved with the user during this work:
the Bayesian belief behind the occupancy grid (`_CellState.logodds`/
`height_ewma` per cell) only ever existed in memory for one continuous scan
session — nothing serialized/reloaded it, so a location revisited days
later would silently start from a blank grid despite maps being meant to
persist. **Chosen approach: coarse re-seed from the last exported summary**
(not full raw-state persistence, and not "always start blank either") —
`occupancy_snapshot.json` stores only the lossy summary (`class` +
normalized `height` per cell, same shape as `extract_full_grid()`, plus
`ground_y` and the flat landmark list), and `OccupancyMap.seed_from_summary()`
(new method) reconstructs an approximate starting belief from it when a
session for that `location_id` starts:
- `CLASS_UNKNOWN` cells get no entry at all (matches how genuinely-
  unobserved cells already work).
- Ground/obstacle cells are seeded with a modest, still-revisable belief —
  logodds at roughly "confirmed twice" (`LOGODDS_OCCUPIED_THRESH`/
  `FREE_THRESH` plus one hit/miss's worth), not `LOGODDS_MAX`/`MIN` — real
  new evidence from the new session can still move a cell either way, same
  as any other Bayesian update. `height_ewma` is reconstructed from the
  normalized height via the exact inverse of `_classify_state()`'s own
  formula.
- Skips the reseed entirely (logs and no-ops) if the saved grid's
  `resolution` doesn't match the current instance's — a resolution change
  isn't meaningfully reseedable, same "skip rather than corrupt" pattern
  already used elsewhere in this module (`_merge_voxels`' vsize-mismatch
  reset, the RTAB-Map SOR cache's length-mismatch fallback).
- Verified via direct round-trip: seeding from a summary reproduces the
  exact same `class`/normalized-`height` grid on immediate re-extraction,
  while confirming the seeded `logodds` is NOT a literal copy of the
  original raw belief and is NOT pinned to `LOGODDS_MAX`/`MIN` (i.e.
  genuinely still revisable, not a frozen snapshot).
- **Known, accepted limitation**: this is NOT true incremental multi-day
  SLAM — a cell confirmed by 50 observations yesterday and one confirmed by
  1 observation both reseed to the same modest belief today. Full raw-state
  persistence was considered and explicitly deferred in favor of this
  simpler approach.

### Mode exclusivity + server-reported client mode

Two related fixes, found and shipped together.

**Real bug, found via a live device session**: `LiveSessionState.mode` is
documented as a single exclusive value, but nothing enforced that —
`toolStartGuiding()`/`toolStartWalking()`/`toolStartScan()`/
`toolEnterReadingMode()` never stopped a still-active tracking session
before switching `state.mode`. Confirmed via real server logs: a user who
said "track my water bottle" then later started guiding/scanning kept
seeing continuous `DetectObject prompt='water bottle'` calls indefinitely
alongside the new mode's own traffic — `MainViewModel.startLocalTracking()`'s
init-retry loop (retries every 1s until the target is confirmed) has no
tie-in to `state.mode` at all, so nothing ever told it to stop. Beyond the
wasted calls, this contends for GPU/model resources with mapping's own
DA3/RTAB-Map calls, plausibly worsening real symptoms seen in that same
session (RTAB-Map `Resource temporarily unavailable`, "tracking LOST on 4/4
frames", a 44s depth-estimation stall). Fixed with `ToolDispatcher.
stopActiveModes()` — called first by every mode-entry tool
(`toolStartTracking`/`toolStartGuiding`/`toolStartWalking`/`toolStartScan`/
`toolEnterReadingMode`), tearing down whatever was previously active
(tracking's local loop + HRTF beacon, or a still-open mapping stream +
walking ticks) regardless of which mode is starting next.

**`StatusService.ReportMode`** (new RPC, `tracking.proto`) — the server has
no other way to know what mode the client is in (no server-side session any
more). `ToolDispatcher.reportMode(mode, target)` is called once per
`state.mode` transition (fire-and-forget on `Dispatchers.IO` — a dropped
report only degrades `server_gui.py`'s tab selection for a moment, never
something the tool-call flow itself should fail over). `ActivityMonitor.
client_mode` stores it; `server_gui.py`'s tab auto-selection now prefers
this explicit signal over inferring from whichever RPC category last fired
(the inference fallback still exists, for older clients that predate this
RPC). Also fixed in the same pass: `ActivityMonitor._record()` used to
`bucket.clear()` before applying new fields — meaning a sticky field like
the Mapping tab's `occupancy_map` reference (only set by `UpdateMapping`)
would flicker away every time an interleaved `FindLandmark` call recorded
into the same bucket without it. Changed to merge instead of clear; every
`*_status()`/render helper already branches on the current op and only
reads fields relevant to it, so stale keys from a differing op are
harmless.

### Walking mode redesign — ambient occupancy-grid steering, no spoken alerts

**Superseded — see "Local reactive HRTF obstacle-dodge" below.** The
occupancy-grid ray-cast steering this section describes
(`LocalPathPlanner.findMostOpenDirection()`/`castOpenRay()`,
`HrtfBeacon.worldYawRad()`, `ToolDispatcher.updateHrtfBeacon()`) has been
removed outright, not deprecated — walking dropped `MappingService`/
RTAB-Map entirely in favor of a per-frame local reactive signal with no
world map at all. Kept here for history: the "no spoken alerts, one
continuous ambient signal" framing established in this section is still
current, just now fed by a different (and much lower-latency) mechanism.

Confirmed with the user during this work: walking mode used to run TWO
overlapping mechanisms simultaneously — it already opened the same
`MappingService.UpdateMapping` stream guiding does (RTAB-Map occupancy
grid), but since it has no destination, `recomputeRoute()` immediately
no-oped and the HRTF beacon stayed permanently muted; ALL of walking's
actual obstacle feedback instead came from a separate, independent
`startWalkingTicks()` loop — a fixed-700ms-interval
`PerceptionService.AnalyzeFrame`(DEPTH op) poll that spoke a
"[SYSTEM] Obstacle ~Xm ahead" alert through Gemini when triggered, gated by
a `walkingObstacleCache`/`quick_label_obstacle` dedup mechanism to avoid
repeat alerts for the same obstacle.

**Chosen replacement**: give the HRTF beacon a real, continuous signal to
compute from the SAME occupancy grid guiding already streams, instead of
running a second independent obstacle-check pipeline. `LocalPathPlanner.
findMostOpenDirection(pose, maxRangeM=5f, coneDeg=90f, stepDeg=15f)` casts
a short ray (resolution-sized steps) per candidate egocentric azimuth
within ±90° of current heading, and returns whichever direction travels
farthest before hitting an obstacle or leaving passable cells — ties favor
the smallest |azimuth| (prefer continuing straight over an equally-open
sharp turn); returns `null` only when even the first step in every
candidate direction is already blocked. Current heading comes from
`HrtfBeacon.worldYawRad(pose)` (new) — yaw-only, pitch/roll deliberately
ignored since navigation here is floor-constrained pedestrian movement
(same assumption CLAUDE.md's "Continuous obstacle clearance" note makes for
server-side path planning). `ToolDispatcher.updateHrtfBeacon()` branches on
`state.mode == "walking"`: no waypoint list to consult, just this direction
every grid update, muting when `null`.

This is purely ambient/continuous — no spoken interruptions, no [SYSTEM]
messages, matching the always-on PixieGuide-style beacon behavior guiding/
walking already use for waypoints. `startWalkingTicks()`,
`toolQuickLabelObstacle()`, the `quick_label_obstacle` tool declaration, and
`LiveSessionState.walkingObstacleCache`/`WalkingObstacleEntry`/
`pruneExpiredObstacles()` were all removed outright (not deprecated) — the
`check_obstacle` tool (on-demand, Gemini-invoked "is there anything in
front of me?" query, still `PerceptionService.AnalyzeFrame` DEPTH op) is
unrelated and untouched. Also updated: `ToolDeclarations.kt`'s WALKING
system-prompt section, since Gemini no longer receives or reacts to
obstacle `[SYSTEM]` messages for this mode at all.

**Explicitly unaffected by this change**: RTAB-Map's own per-frame depth
estimation inside `scan_session.py`'s live mapping/reconstruction pipeline
— that's a completely separate depth usage (feeds the occupancy grid
itself) from `PerceptionService.AnalyzeFrame`'s `DEPTH` op, which is what
got removed from walking mode's polling loop specifically.

### Local reactive HRTF obstacle-dodge (replaces walking's occupancy-grid steering)

**Superseded for WALKING — see "Local SLAM-backed walking corridor-lock"
below.** GUIDING's beacon (goal-biased scoring off this same traversability
fan) is completely unaffected and still works exactly as documented in this
section. Kept here for history/rationale — walking's per-tick,
no-world-state design below is a further iteration on the same "steering
command, not a position" philosophy this section established, not a
reversal of it.

Worked out with the user across several rounds of design discussion, then
implemented. Two problems with the occupancy-grid-based steering documented
in "Walking mode redesign" above drove this:

1. **Latency.** The occupancy grid only updates once RTAB-Map has processed
   a mini-batch and the map has been rebuilt — too slow to react to
   something that just stepped into view. Walking has no destination
   either, so a *world map* was never actually needed for it in the first
   place.
2. **What the sound should mean.** The beacon must be a pure **steering
   command**, not a position: a fixed-radius circle around the user's head,
   azimuth-only (elevation pinned to 0°) — "turn until the sound is ahead,"
   not "the thing is over there at this distance." Guiding's old
   `directionTo()`-driven beacon (real elevation + real distance to a
   waypoint) doesn't match this either, going forward.

**Chosen design** (standard two-layer navigation split — global planner
decides *where to ultimately go*, local planner decides *what's safe to
step toward right now*, HRTF only ever expresses the local layer as a
steering angle):

- **Walking** drops `MappingService`/RTAB-Map entirely — no pose, no grid,
  no world state of any kind. Every tick: current frame → server computes a
  per-frame, ground-segmented obstacle-clearance fan across a polar range
  of egocentric angles (no accumulation, no memory of earlier frames) →
  client picks the most open direction, EMA-smooths it, points the beacon
  there.
- **Guiding** keeps `MappingService`/RTAB-Map for the *global* layer exactly
  as before (route to a landmark via `LocalPathPlanner`'s A*, arrival
  detection), but the beacon itself is now driven by the same local fan,
  **goal-biased**: score = clearance − distance-from-goal-bearing −
  steering-effort, so the beacon nudges around an obstacle while still
  pulling generally toward the route, instead of blindly maximizing open
  space (ignoring the destination) or blindly pointing at the waypoint
  through a wall (the old `directionTo()`-only behavior).
- Both modes' beacon output is now azimuth-only / fixed-radius —
  `directionTo()` is still used for guiding, but only to supply the *goal
  azimuth* fed into the local scorer, never to place the beacon directly.

**Server — the traversability fan (`server/tools/traversability.py`,
new)**: `estimate_traversability(depth_map, num_bins, max_range_m)` is
stateless and single-frame — no IMU, no persisted `ground_y` (unlike
`occupancy_map.py`'s mapping-mode Bayesian grid, which this deliberately
does NOT reuse). Per call: back-projects a subsampled pixel grid to
camera-space 3D points via the same pinhole-K fallback used everywhere else
in this codebase (`fx=fy=0.8*max(w,h)`), RANSAC-fits a ground plane from
the bottom ~40% of the frame, classifies obstacle vs. ground by signed
height above that plane, buckets obstacle points by azimuth
(`atan2(X,Z)`), and returns the nearest-obstacle clearance per bin (a bin
with nothing in range reads `max_range_m` — fully open, not a missing
value). The angular range is derived from the frame's own estimated FOV,
not a fixed cone — this only ever reacts to what's actually in view this
frame, no side/rear awareness (an accepted limitation, discussed with the
user).

**Two real bugs found via synthetic ground-truth testing (RANSAC plane fit
against a hand-built floor+box scene, back-projected through the exact same
pinhole model the function itself uses) before this shipped**:

1. **Sign-flip bug.** The obvious way to decide which side of the fitted
   plane counts as "up" — check whether the ground points themselves read a
   positive or negative signed distance — doesn't work: ground points read
   ≈0 under *either* orientation of the plane normal (that's what makes them
   inliers in the first place), so their own sign carries no information.
   Fixed by using the CAMERA ORIGIN's signed distance instead (the `d`
   coefficient, since `normal·(0,0,0)+d = d`) — the camera and any real
   obstacle are always on the same side of the floor plane (both are
   between the floor and the camera), so flipping when `d < 0` reliably
   orients "obstacle-positive" regardless of which way RANSAC happened to
   point the normal. Verified: 0/20 failures detecting a synthetic box
   after the fix, vs. ~40% before it.
2. **Frontal-obstacle-as-floor bug.** A flat obstacle filling most/all of
   the frame (e.g. the user standing right up against a wall) is itself a
   perfectly good RANSAC plane fit — just not a horizontal one — and would
   get silently accepted as "the floor," reading back as fully open exactly
   when it's most dangerously wrong. Fixed by rejecting any candidate plane
   whose normal isn't substantially vertical (`_MIN_GROUND_NORMAL_
   VERTICALITY`) during RANSAC scoring itself, not just after the fact —
   a plane that fails this never wins, so the "no confident floor" fallback
   (treat every visible point as an obstacle) correctly engages instead.
   Verified: a synthetic fully-blocked frame now reads as fully blocked in
   20/20 trials, not fully open.

`server/tools/depth.py`'s `DA3DepthDetector` was refactored to share one
`_depth_map()` call between `check_obstacle()` (existing corridor check)
and the new `estimate_traversability()` — a caller requesting both `DEPTH`
and `TRAVERSABILITY` in one `AnalyzeFrame` round trip doesn't pay for DA3
inference twice.

**Proto (`tracking.proto`)**: `AnalysisOp.TRAVERSABILITY` (new), new
self-describing `TraversabilityInfo` message (mirrors `OccupancyGrid`'s own
self-describing width/height/cell_size convention — `min_angle_deg`/
`max_angle_deg`/`angle_step_deg`/`max_range_m` alongside the clearance
array, so the client never hardcodes bin count or FOV), and
`StatusService.ReportBeaconDirection` (azimuth_deg + muted, dashboard-only,
same fire-and-forget precedent as `ReportMode` — see that RPC's own entry
above).

**Android (`client/android/app/src/main/java/com/tracking/client/live/`)**:

- **`TraversabilityScorer.kt`** (new) — `pickSteeringAngle()`: classic
  Vector-Field-Histogram-style scoring. Each candidate bin's score is its
  own corridor-windowed clearance (minimum over a small window of
  neighboring bins — approximates the user's body needing to actually fit
  through a gap, not just one ray missing an obstacle), minus a
  goal-bearing penalty (guiding only — `null` for walking, which has no
  destination) and a steering-effort penalty against the beacon's current
  smoothed azimuth. Returns `null` (mute) when even the best bin's own
  clearance is below a floor threshold. `smoothAzimuth()` — EMA toward the
  picked angle, shortest-path around the ±180° wrap.
- **`ToolDispatcher.kt`** — `updateHrtfBeacon()` (old, grid-driven) is gone;
  replaced by `startLocalAvoidanceTicks()`/`runAvoidanceTick()`, a
  `Dispatchers.IO` loop on its own cadence (`avoidanceIntervalMs`,
  independent of guiding's slower `MappingService` frame cadence — the two
  are different layers with different latency needs). Each tick: pull the
  current frame via `latestFrame()` (`clearestRecentFrame()` — already
  blur-aware and pull-based, the right fit for a reactive per-tick pull) →
  `AnalyzeFrame(ops=[TRAVERSABILITY])` → compute the goal azimuth for
  guiding (`HrtfBeacon.directionTo(pose, wx, wz).azimuthDeg` against the
  current waypoint — azimuth only, its elevation/distance are unused now)
  → `TraversabilityScorer.pickSteeringAngle()` → `smoothAzimuth()` →
  `hrtfBeacon.updateDirection(smoothed, 0f, OPEN_DIRECTION_DISTANCE_M)` (a
  fixed nominal radius, same precedent `HrtfBeacon.directionFromBox()`
  already set for tracking mode's own no-real-distance case) or `.mute()` →
  fire-and-forget `reportBeaconDirection()`. `toolStartWalking()` no longer
  calls `startMappingStream()` at all; `toolStartGuiding()` calls BOTH
  `startMappingStream()` (global route) and `startLocalAvoidanceTicks()`
  (local dodge). `stopActiveModes()`/`shutdown()` stop the tick job
  unconditionally alongside the mapping stream and the beacon.
- **`LocalPathPlanner.kt`** — `findMostOpenDirection()`/`castOpenRay()`
  removed outright (dead once walking dropped the grid). `findPath()`/A*
  unchanged — guiding's global layer still needs it.
- **`HrtfBeacon.kt`** — `worldYawRad()` removed outright (its only caller
  was `findMostOpenDirection()`). `directionTo()`/`directionFromBox()`
  unchanged.
- **`LiveSessionState.kt`** — new `smoothedBeaconAzimuthDeg: Float?`,
  carried across ticks so smoothing has something to smooth FROM; reset to
  `null` at the start of a fresh walking/guiding session, but deliberately
  NOT reset on a single muted tick (a brief mute — e.g. one bad frame —
  shouldn't discard smoothing continuity for whenever the beacon un-mutes).
- **`CameraManager.kt`** — the mapping-submode special case (no blur
  filtering, own send-interval) now covers only `guiding`/`scanning`, not
  `walking`. Walking's frames flow through the ordinary blur-aware
  `recentBufferMs` path instead (the same one tracking/reading/idle already
  use) — which turns out to be exactly the right fit, since
  `runAvoidanceTick()` wants "the current sharp frame on demand," not a
  subscribed interval stream. `MainViewModel.kt`'s mirrored
  `mappingModeActive` condition narrowed the same way.
- **New "Avoidance FPS" setting** (`SettingsScreen.kt`/`SettingsViewModel`)
  — same plain-FPS-number-input convention as "Mapping FPS"/"Scan FPS"
  (fps↔ms conversion only at `doConnect()`), default 350ms/≈2.86fps,
  threaded through `MainViewModel.connect()` into
  `ToolDispatcher(avoidanceIntervalMs=...)`.

**GUI**: `server/services/beacon_preview.py` (the old server-side,
visualization-only world-point reconstruction of "where does the beacon
point") is deleted outright, along with its usage in `mapping_servicer.py`
(`_resolved_destinations`/`_last_open_direction` caches, the
`beacon_world_xz`/`beacon_pixel` computation block) — it modeled the OLD
grid-based beacon and has no correct equivalent for a client-computed
azimuth. Replaced by `ReportBeaconDirection` (above) feeding a NEW
`server_gui.py` panel — originally placed on a standalone **Perception
tab** (walking touched no `MappingService` traffic at the time this was
written, only `PerceptionService.AnalyzeFrame(TRAVERSABILITY)` +
`StatusService`), later merged into the **Mapping tab** instead once
walking rejoined `MappingService` — see "Grid-planned walking route"'s
dashboard note below for the current layout: `_render_beacon_polar()`
draws the last traversability fan as a Plotly polar bar chart (forward =
12 o'clock, azimuth-right reads clockwise, matching `HrtfBeacon.kt`'s sign
convention) with a marker at the client-reported final azimuth, greyed out
when muted. The old magenta circle overlays on the Mapping tab's frame/
occupancy-map views are removed (nothing to draw any more — the real
beacon direction was never grid-space to begin with now).

**Known, accepted limitations** (discussed with the user, not solved by
this design):

- No IMU-assisted ground-plane fit — purely single-frame RANSAC. A frame
  with literally no visible floor (very close obstacle filling the view)
  degrades to "treat everything in range as an obstacle" rather than
  guessing, per the frontal-obstacle-as-floor fix above.
- The fan has no memory — an obstacle just outside the current frame gets
  no warning until it's back in view. This is the direct tradeoff for
  dropping the (laggy but persistent) occupancy-grid world model.
- Guiding now makes two independent per-cycle round trips (the slow
  `MappingService` stream frame for the route, the faster
  `PerceptionService.AnalyzeFrame` call for the local dodge) — accepted
  since they serve genuinely different layers and the latter is a
  lightweight unary call, not a stream.

### Local SLAM-backed walking corridor-lock (supersedes walking's Local reactive HRTF obstacle-dodge)

**Superseded — see "Grid-planned walking route" below.** In real on-device
testing this design did not work: a single-frame "widest open arc" corridor
selection turned out too easy to false-trigger the dead-end alert on
(anything vaguely close in ANY direction could read as "no corridor," even
with open space elsewhere), and the world-anchored lock's dwell/hysteresis
behavior didn't feel right in practice. Replaced with a design that goes
back to a real, continuously-updated local occupancy grid (the RTAB-Map
integration/reset-on-loss machinery this section built is still current —
only the beacon-steering mechanism on top of it changed). Kept here for
the RTAB-Map F2M research/rationale, which is still accurate.

Worked out across a long design discussion, starting from a real weakness
in the per-tick design just above: a single stateless frame has no memory
across a bad reading (motion blur, a momentary lost tracker), and re-scoring
"most open direction" every tick doesn't behave like a real sound source —
turning your head shouldn't make a good target jump to a new direction
computed from scratch. Two things came out of that discussion:

1. **RTAB-Map's own odometry already solves the memory problem, for
   pose specifically.** Confirmed by reading `rtabmap_server.cc`:
   `Odom/Strategy` is set to `"0"` — F2M (Frame-to-Map), RTAB-Map's default
   — which matches new frames against a local multi-frame feature map, not
   naively against just the previous frame; a single bad frame doesn't
   permanently break tracking the way naive frame-to-frame VO would. This
   is exactly the ORB-SLAM3-style "local map, not one frame" robustness
   that was being asked for, already running in this project's stack.
2. **The beacon should behave like a real landmark in the room.** Instead
   of re-deriving a direction every tick, the beacon locks onto a specific
   corridor — a real point in 3D space — and steers toward that fixed
   **world** target continuously as the user turns/moves, only
   reconsidering the lock after a sustained (not incidental) gaze change.

**Chosen design** — a state machine, not a per-tick reactive score:

- **Locked**: the beacon plays toward a fixed world `(x, z)` target.
  `HrtfBeacon.directionTo()` (already existed, unchanged) recomputes the
  egocentric azimuth to that fixed point every tick from the current pose —
  turn right and the sound swings toward the left, exactly like a real
  external sound source, because the *target* doesn't move, only the
  listener's pose does.
- **Dwell hysteresis**: while locked, if the current bearing to the lock
  diverges by more than `REEVALUATE_ANGLE_DEG` (35°), a timer starts; only
  after holding that divergence for `DWELL_MS` (3s) does the client
  re-evaluate the *current* view for a new corridor and switch the lock (or
  drop it if the current view has none). A glance back toward the lock
  before 3s cancels the timer outright — no re-evaluation happens. This is
  what stops the target from jittering every time the head moves
  incidentally while walking toward it.
- **Arrival**: within `ARRIVAL_RADIUS_M` (1.0m, matching
  `checkWaypointProgress()`'s existing guiding radius) of the locked
  target, the lock clears and the next tick searches fresh — otherwise the
  beacon would eventually point behind the user forever.
- **Dead end**: whenever unlocked (never acquired a lock yet, or one was
  just dropped) and the current view also has no viable corridor, a
  distinct alert plays instead of silence or the corridor tone — repeated,
  not one-shot, so the user keeps getting a cue while turning to search.
- **Recovery**: the instant a corridor is found while unlocked, the lock
  acquires immediately — no dwell wait. The dwell timer only protects an
  *existing* lock from being abandoned too eagerly; there's nothing to
  protect when nothing is locked.

**Server — corridor detection (`server/tools/traversability.py`)**:
`estimate_traversability()`'s ground-plane-fit + azimuth-binned-clearance
body was extracted into `_clearance_fan_from_depth()` (no logic change) so
a new `find_corridor(depth_map, num_bins, max_range_m,
min_corridor_width_m=0.7, min_corridor_range_m=1.2) -> CorridorResult`
could reuse it instead of duplicating the RANSAC ground fit. `find_corridor`
scans the same clearance-per-bin array for contiguous runs where every bin
clears `min_corridor_range_m` (deep enough to actually step into, not just
a doorway visible but not reachable), computes each run's real-world width
(`angle_span_rad * representative_range`, representative_range = the run's
own minimum clearance — conservative), keeps runs `>= min_corridor_width_m`,
and picks the widest (ties: smallest `|center_azimuth|`).
`found=False` (all other fields 0) means nothing qualified — the dead-end
case. `DA3DepthDetector.find_corridor()` (`server/tools/depth.py`) mirrors
the existing `estimate_traversability()` wrapper, sharing the same
`_depth_map()` DA3 call — requesting `CORRIDOR` costs no extra inference.
New `AnalysisOp.CORRIDOR` + `CorridorInfo{found, azimuth_deg, distance_m,
width_m}` in `tracking.proto`/`AnalyzeFrameResponse`, handled in
`perception_servicer.py` alongside `DEPTH`/`TRAVERSABILITY`.

**Server — reviving `MappingService` for walking, pose-only
(`mapping_servicer.py`, `scan_session.py`, `stream_session.py`)**: walking
needed to drop `MappingService` entirely per "Local reactive HRTF
obstacle-dodge" above (no world map, no destination) — but world-anchoring
a lock target requires CONTINUOUS pose, which a stateless per-frame
`AnalyzeFrame` call can't provide. Walking reopens its own
`UpdateMapping` stream (`SessionMode.WALKING`), but strictly for RTAB-Map's
F2M pose tracking — no grid, no landmarks, no VLM tagging, no destination.
`process_frames_batch()` gained `pure_walking: bool` (parallel to the
existing `walking_lite`, which stays `True` for both `WALKING` and
`GUIDING`): `pure_walking` is `WALKING` specifically and skips Step 4's
Bayesian `occupancy_map.update()`/`_merge_voxels()` entirely — nothing ever
reads the grid for pure walking any more (corridor detection is a separate
stateless call, not fed from this stream at all), so computing it would be
pure waste. `mapping_servicer.py`'s `UpdateMapping` branches early for
`pure_walking`: `MappingUpdate` carries only `pose` (+ `reset_occurred`,
below) — no grid/delta/landmarks work at all. The stream's `finally` block
also skips `finalize_landmarks_flat()`/snapshot-save for `pure_walking` —
nothing meaningful ever accumulates (walking_lite already skips VLM
tagging, and pure_walking additionally skips the grid), so saving a
snapshot here would silently overwrite a real prior SCAN's snapshot with
an empty one for that `location_id`.

**Total-tracking-loss reset**: F2M relocalization is robust against a
single bad frame, but not infinite — if RTAB-Map reports zero pose for
`PURE_WALKING_LOST_RESET_S` (2.0s) of UNBROKEN consecutive loss (any single
tracked frame clears the streak — this means genuinely no sign of
recovery, not merely frequent loss), `pure_walking` sessions reset:
`ScanSession.reset_cloud()` was split into a lock-free `_reset_cloud_locked()`
body plus a thin `reset_cloud()` wrapper, since `process_frames_batch()`
already holds `self._lock` for its whole body and `threading.Lock()` isn't
reentrant — the streak check calls the lock-free body directly, right after
Step 2 and before Step 3 would otherwise back-project the known-garbage
pose into anything. New `MappingUpdate.reset_occurred` (proto) signals this
to the client on the one update where it fires; `ToolDispatcher.kt`'s
walking pose-stream collector responds by clearing `corridorLockTarget`/
`corridorDwellStartMs` — silently (confirmed with the user: matches
walking's existing "no spoken alerts, ambient only" design; the beacon just
mutes until a fresh pose locks in, same as any other momentary gap).

**Client-side warm-up (cold-start latency)**: the pipeline's first-ever
inference pass is typically much slower than steady state (model/CUDA
warm-up) — real enough that by the time a throwaway first frame is
actually processed, the phone has likely moved from where it was captured,
making that frame a poor session origin. `ToolDispatcher.toolStartWalking()`
sends exactly one frame on a disposable stream
(`warmUpWalkingPoseStream()`), awaits the round trip (or a timeout,
best-effort — proceeds into real walking either way), then discards it by
letting that stream close and opening a fresh one
(`startWalkingPoseStream()`) — no server-side code needed for the discard
itself, since a new `StreamingScanSession` already calls `reset_cloud()` on
construction. `sendSystemNote()` cues bookend this ("loading" then "ready")
so Gemini can tell the user walking mode is starting up, then that it's
live — the one place walking breaks its own "no spoken alerts" convention,
since this is a one-time session-start event, not an ongoing obstacle
signal.

**Client (`client/android/app/src/main/java/com/tracking/client/live/`)**:
- **`HrtfBeacon.kt`** — new `worldPointFrom(pose, azimuthDeg, distanceM):
  Pair<Float, Float>`, the exact inverse of the existing `directionTo()`:
  rotates a floor-plane forward vector by the pose's quaternion (direct
  rotation, not `directionTo()`'s conjugate) and offsets by the pose
  position. Converts a corridor detection's one-shot egocentric reading
  into the fixed world target the lock holds onto.
- **`LiveSessionState.kt`** — new `corridorLockTarget: Pair<Float, Float>?`
  / `corridorDwellStartMs: Long?`, cleared in `reset()` and on a
  `reset_occurred` signal. `smoothedBeaconAzimuthDeg` is GUIDING-only now.
- **`ToolDispatcher.kt`** — `startLocalAvoidanceTicks()`'s tick body now
  branches on mode: GUIDING still calls the unchanged `runAvoidanceTick()`
  (TraversabilityScorer); WALKING calls the new
  `runWalkingAvoidanceTick()`, which pulls one frame via `latestFrame()`
  (the same blur-aware `recentBufferMs` pull guiding/tracking already use —
  see "Client-side frame selection" above, unaffected by any of this) and
  feeds it to BOTH `sendWalkingPoseFrame()` (walking's own pose stream,
  NOT `feedMappingFrame()`/`MainViewModel`'s continuous camera collector —
  `CameraManager.kt`/`MainViewModel.kt` needed no changes at all) and a new
  `fetchCorridor()` (`AnalyzeFrame(CORRIDOR)`), then runs the state machine
  above. `startWalkingPoseStream()`/`stopWalkingPoseStream()`/
  `sendWalkingPoseFrame()` are fully separate from guiding/scanning's
  `startMappingStream()`/`feedMappingFrame()` (untouched) — walking owns
  its own job/channel (`walkingPoseJob`/`walkingPoseChannel`).
- **Dead-end alert** — `playDeadEndAlert()` uses `android.media.ToneGenerator`
  (`TONE_SUP_ERROR`, rate-limited via `ALERT_PERIOD_MS`), not a bundled
  audio asset: confirmed with the user that a genuinely distinct sound was
  wanted (not just a pulsed version of the existing fluttering-loop
  corridor tone), and a synthesized tone works out-of-the-box with no audio
  file to source — can be swapped for a bundled asset later behind the same
  call site if preferred.

**Known, accepted limitations** (discussed with the user):

- No long-horizon drift correction — this is RTAB-Map's raw odometry
  (F2M), not a loop-closed graph (walking never pulls `get_cloud()`/runs
  graph optimization, see `pure_walking` above). Accepted because a lock's
  lifetime is inherently short (cleared on arrival or dwell-triggered
  re-evaluation within a handful of metres/seconds) — ordinary VO drift at
  that horizon is a non-issue.
- `server_gui.py`'s dashboard was NOT updated in this pass —
  `_TAB_BY_CLIENT_MODE["walking"]` still pointed at `tab_perception` (a
  leftover from the fully-stateless design this supersedes), even though
  walking's `UpdateMapping` traffic already landed in `ActivityMonitor`'s
  mapping bucket. A real, known gap at the time — fixed in "Grid-planned
  walking route" below's dashboard note, once walking's `MappingService`
  usage was no longer a lock-based special case and a straight tab-mapping
  fix made sense.

### Grid-planned walking route (supersedes the corridor-lock design above)

**Superseded — see "Server-planned walking path + client-side latency
bridging" below.** Path PLANNING itself (the A* this section describes)
moved server-side; the client-side `LocalPathPlanner.kt` this section
introduced is deleted outright, not deprecated. Kept here for history —
the underlying planning goals (curve around obstacles, prefer open space,
alert only on a true dead end) are still exactly what the server-side
planner does now, just relocated.

The corridor-lock design above didn't work in real testing (see its own
"Superseded" note). Feedback from that testing, addressed by this design:

1. **Bring back a real local occupancy grid**, fed continuously by
   RTAB-Map every frame — same grid GUIDING already builds — instead of a
   single-frame "widest open arc" read. Never persisted, and reset far
   more aggressively on tracking loss than a scan/guiding session would
   want (0.5s of unbroken loss, was 2.0s — confirmed with the user: the
   grid is cheap to rebuild, so a fast reset-and-restart beats limping
   along on stale state while F2M tries to recover).
2. **Plan an actual path, not a single direction.** The route doesn't need
   to be straight — it can curve around an obstacle. Walking has no
   destination, so it always targets a synthetic point straight ahead of
   the CURRENT pose, replanned from scratch on every grid update (no lock,
   no dwell timer — a turn is reflected on the very next update).
3. **Prefer the most open path, not the shortest one** — nudge around an
   obstacle while keeping distance from it, rather than hugging the
   nearest edge of a gap.
4. **Only alert when there is truly no walkable path anywhere in view** —
   not just because something is detected roughly ahead. A chair a metre
   to one side with open space elsewhere should never trigger anything.

**Chosen implementation — reuse `LocalPathPlanner` (GUIDING's existing A*)
wholesale, pointed at a synthetic target instead of a real destination**:
its clearance-aware cost model (see "Continuous obstacle clearance" above)
already does exactly #3 — an exponential-decay penalty for low-clearance
cells means the search naturally prefers a wider gap over a tighter
shortest path — and its existing closest-approach fallback (see
"Closest-approach navigation" above) already does exactly #2 — when the
literal forward target is blocked, A* falls back to the reachable cell
closest to it, which is exactly "bend the route toward whatever's open" —
and `findPath()` already returns `null` only when NOTHING is reachable at
all (start cell blocked, or the search can't expand anywhere), which is
exactly #4's trigger condition. No new path-search algorithm was needed —
this whole redesign is almost entirely reusing already-built, already-
tested machinery differently, not new pathfinding code.

- **Server**: `scan_session.py`'s `pure_walking` flag (SessionMode.WALKING)
  no longer skips Step 4 (`occupancy_map.update()`/`_merge_voxels()`) — it
  gets the exact same live grid GUIDING does. `PURE_WALKING_LOST_RESET_S`
  dropped from 2.0 to 0.5. `mapping_servicer.py`'s `UpdateMapping` no
  longer special-cases `pure_walking` into a pose-only branch — it flows
  through the same grid/delta/full_resync logic as GUIDING now, with only
  two `pure_walking`-specific behaviors left: (a) a `session.last_reset_occurred`
  check before the normal "no pose yet" gate, which also pops
  `self._last_full_bounds[location_id]` so the client's next real update is
  forced to a full resync (the client's own grid is gone too after a
  reset); (b) skipping `finalize_landmarks_flat()`/snapshot-save at stream
  close (still never persisted).
- **Client (`ToolDispatcher.kt`)**: walking now calls the SAME
  `startMappingStream()`/`feedMappingFrame()` GUIDING/SCANNING use — no
  more separate pose-only stream, no more warm-up handshake (dropped as
  unnecessary complexity now that everything is continuously replanned
  relative to current pose anyway — a stale first position self-corrects
  within a tick or two, there's no persistent lock to be wrong about).
  `recomputeRoute()` (already existed, GUIDING-only before) now branches:
  GUIDING keeps its exact existing FindLandmark-based logic; WALKING
  computes `HrtfBeacon.worldPointFrom(pose, 0f, WALKING_LOOKAHEAD_M)` (6m)
  as the target, calls the same `LocalPathPlanner.findPath()`, and either
  steers toward the result (`steerWalkingBeacon()`, new — points the
  beacon at `state.navWaypoints[state.navWaypointIdx]`, called after every
  replan AND after every `checkWaypointProgress()` waypoint advance so it
  tracks smoothly off fresh pose between full replans) or, on `null`,
  plays `playDeadEndAlert()` (unchanged tone mechanism, new trigger
  condition). `checkWaypointProgress()` now also branches — advances
  `navWaypointIdx`/speaks arrival notes for GUIDING only (matches
  walking's existing "ambient only, no spoken alerts" design), but
  re-steers WALKING's beacon on every call regardless. On the mapping
  stream's `resetOccurred` signal (either mode, though only WALKING ever
  actually sets it server-side), the collector now clears
  `mutableGrid`/`lastMappingGrid`/`navWaypoints` and mutes the beacon —
  the local grid the client had is gone too.
- **`CameraManager.kt`/`MainViewModel.kt`**: walking rejoined the
  no-blur-filter mapping-mode frame bucket (reverting the carve-out from
  "Local reactive HRTF obstacle-dodge") — new `walkingIntervalMs` (default
  350ms, set from `avoidanceIntervalMs` in `MainViewModel.connect()`)
  keeps its send rate fast/responsive rather than inheriting guiding's
  much slower default. `MainViewModel`'s `mappingModeActive` gate and
  `CameraManager.processFrame()`'s interval selection both now cover
  `"walking"` alongside `"guiding"`/`"scanning"`. `clearestRecentFrame()`
  (backing `latestFrame()`) still works for WALKING's own avoidance tick
  even while in mapping mode — `CameraManager`'s `recentBuffer` is kept
  fresh unconditionally regardless of `mappingMode`, only the *continuous*
  `frameFlow` emission is skipped during mapping mode, not the pull-based
  buffer itself.
- **Dead code removed**: `AnalysisOp.CORRIDOR`/`CorridorInfo` are left in
  the proto (marked unused in their own comments — no wire-compat benefit
  to reclaiming the values) but `find_corridor()` (`traversability.py`,
  `depth.py`, `perception_servicer.py`) and the whole corridor-lock
  Kotlin state machine (`LiveSessionState.corridorLockTarget`/
  `corridorDwellStartMs`, `ToolDispatcher`'s `startWalkingPoseStream()`/
  `stopWalkingPoseStream()`/`sendWalkingPoseFrame()`/
  `warmUpWalkingPoseStream()`/`fetchWalkingAnalysis()`) are deleted
  outright. `traversability.py`'s `_clearance_fan_from_depth()`/
  `estimate_traversability()`/`dropoff_m` are all still current — see
  "Hazard warnings" below, which is unaffected by this section's changes
  except for one trigger simplification it documents itself.
- **Dashboard fixed** (`server/server_gui.py`) — the tab-gap noted in the
  superseded corridor-lock section above is now closed:
  `_TAB_BY_CLIENT_MODE["walking"]` points at `tab_mapping` (was
  `tab_perception`, stale since before this whole redesign). The old
  standalone Perception-tab beacon panel (`_render_beacon_polar()`/
  `_beacon_status()`) was also merged directly into the Mapping tab,
  alongside the occupancy map — both GUIDING and WALKING steer through
  that same grid now, so showing the beacon on a separate tab no longer
  made sense. The Perception tab itself still exists, narrowed to just
  ad-hoc `run_detection`/`check_obstacle` debug traffic.
- **Bug fix — `worldToCell()`/`world_to_cell()` truncation** (`LocalPathPlanner.kt`,
  `live_path_planner.py`): both used `Float/float -> Int` truncation
  (`.toInt()` / `int(...)`), which rounds TOWARD ZERO, not down — a point
  just outside the grid's negative edge (e.g. `x - originX == -0.3`) read
  as cell `0` instead of cell `-1`, silently aliasing an out-of-bounds
  query onto a real in-bounds cell rather than correctly reading it as
  unmapped. Since `findPath()`'s very first check is
  `passable(start)`, a mis-resolved start cell could read the WRONG
  cell's class and bail out immediately — `LocalPathPlanner.findPath()`
  returning `null` (WALKING's dead-end alert, silence otherwise) even
  though the user's actual current position was fine. Both fixed to use
  `floor()`/`math.floor()` instead.

**Known, accepted limitations** (same horizon as the corridor-lock design
this supersedes): no long-horizon drift correction (RTAB-Map's raw F2M
odometry, no loop-closed graph — walking never pulls `get_cloud()`); the
grid itself has no persistence across a reset (by design — see point 1
above).

### Server-planned walking path + client-side latency bridging (supersedes grid-planned walking route)

Requested directly by the user, via a detailed written spec, after
observing that "grid-planned walking route"'s beacon was fully gated on
the server round trip (RTAB-Map + occupancy update) with no bridging of
in-between latency, and that walking's fixed "6m straight ahead" target
made for a stop-and-turn experience rather than a smooth one. The user's
own framing: the goal is **not** SLAM accuracy — it's a continuously
stable HRTF steering cue with minimal perceived latency. Two changes
follow directly from that:

1. **Path PLANNING moves fully server-side.** The occupancy grid was
   already computed server-side; now the actual route search
   (`LiveGridPathPlanner`, previously only used by `scan_gui.py`'s debug
   UI) runs there too and the client receives a ready-to-follow path
   instead of a grid to search itself. GUIDING: plans toward the
   `FindLandmark`-resolved destination (client still resolves the name —
   only the coordinate now also travels to the server). WALKING: plans
   toward "the farthest open direction within a turn budget" — a NEW
   server-side heuristic, `find_farthest_open_path()`, since there's no
   real destination to route toward at all.
2. **The client bridges the ~1Hz gap between server updates with its own
   cheap local motion estimate** — frame-to-frame rotation (no scale
   ambiguity, unlike translation) via monocular Essential-matrix
   decomposition, and walked distance via Android's built-in step
   detector + a fixed stride length — so the beacon keeps responding
   smoothly to a head turn or a few steps without waiting on the network.

**Server — path planning (`scan_server/live_path_planner.py`,
`server/services/mapping_servicer.py`)**: `find_farthest_open_path(planner,
start_xz, heading_rad, max_turn_rad=π/2, num_candidates=9,
probe_range_m=8.0)` (new) reuses `LiveGridPathPlanner.find_path()` as a
black box rather than a new constrained search — probes `num_candidates`
goal points spread evenly across `[heading - max_turn, heading +
max_turn]` at `probe_range_m`, and keeps whichever result has the greatest
ACTUAL path length (`_path_length()`, cumulative segment length, not
straight-line distance to the probe — a route that has to detour around
an obstacle should score on how far it really lets the user walk).
`find_path()`'s own closest-approach fallback already makes a blocked/
short candidate score low, and its existing clearance-cost model already
biases each candidate toward wider corridors, so no new search machinery
was needed — this is the server-side analogue of the old (reverted)
per-tick client-side `findMostOpenDirection()`/`castOpenRay()` design (see
"Local reactive HRTF obstacle-dodge" above for why that got reverted —
re-deciding a target once per ~1Hz server update, instead of every ~300ms
client tick, doesn't have that jitter failure mode). Returns `None` only
if every candidate direction returns `None` (nothing walkable anywhere) —
WALKING's sole dead-end trigger, matching the user's explicit "only alert
when there's truly no path" requirement. Verified via synthetic ground-
truth tests (a long corridor at a turn-budget-respecting angle vs. a
short dead-end straight ahead; a fully sealed-off start) — same technique
used elsewhere in this codebase for `traversability.py`'s ground-plane fit.
`heading_rad` is derived server-side from the session's own RTAB-Map
pose (new `_pose_heading_rad()` in `mapping_servicer.py` — rotates the
camera-local forward vector `(0,0,1)` by the pose's rotation matrix into
world space, `atan2(x, z)`, matching `HrtfBeacon.kt`'s own azimuth
convention) — no new client-sent heading field needed.
`MappingServiceServicer.UpdateMapping` builds a `LiveGridPathPlanner` from
`session.occupancy_map.extract_full_grid()` on every update (not gated on
`grid_updated` — pose changes far more often than grid structure does, and
the route needs to originate from the current pose) and branches on
`session_mode`: `WALKING` → `find_farthest_open_path()`; `GUIDING` with a
resolved goal → `planner.find_path(pose_xz, goal_xz)`; otherwise → an
empty path. **The `grid`/`grid_delta` wire fields are still computed and
sent every update, unchanged** — the client no longer consumes them for
anything (path search moved server-side), but this was deliberately left
as-is in this pass rather than also ripping out the delta-sync machinery;
a real, small, known bandwidth waste worth revisiting in a follow-up, not
a correctness issue.

**Bug found via the dashboard (once it could actually show the route —
see the Mapping-tab overlay note below) and fixed: WALKING's route left
confirmed territory, hugging the edge of the explored area through
unknown cells.** `probe_range_m` (8m default) is very often farther than
the currently mapped area actually extends — every candidate goal then
lands outside the grid's own bounds, `find_path()`'s closest-approach
fallback (correctly) finds whatever reachable cell gets geometrically
closest to that unreachable off-map point, and since unknown cells are
passable-at-a-premium (not blocked), the resulting route can wind along
the boundary of what's mapped chasing a point that was never actually
reachable — visibly confirmed by the dashboard's own dashed-orange
"speculative" route styling. Fixed by adding `confirmed_only: bool` to
`find_path()` (`live_path_planner.py`): when True, the raw A* path is
truncated at the first `CLASS_UNKNOWN` cell (new `_truncate_to_confirmed()`
helper) BEFORE simplification, so the returned route can never leave
genuinely observed territory (`confirmed` is then always `True`;
`reached_exactly` reflects whether that confirmed prefix happens to reach
the goal). `find_farthest_open_path()` now calls `find_path(...,
confirmed_only=True)` for every candidate — WALKING-only; GUIDING's own
`find_path()` call is deliberately left unqualified, since flooding into
unexplored territory is exactly how it's supposed to reach a destination
that isn't fully mapped yet. Verified via a synthetic test reproducing the
observed bug shape (a small confirmed blob, a probe point far outside its
bounds) — confirms every returned waypoint lands on a non-`CLASS_UNKNOWN`
cell; re-ran the earlier corridor/dead-end synthetic tests too, unaffected.

**Follow-up, requested directly by the user (superseded — see "Directional-
distance candidate selection" below): candidate selection wasn't
weighing openness at all, only raw length — a route that hugs a wall for
longer could beat a shorter route straight through open space.**
`find_path()`'s own per-cell clearance-cost model only shapes WHICH route
reaches a GIVEN candidate goal — it never affected WHICH candidate goal
`find_farthest_open_path()` picked among the `num_candidates` directions
it probes, since that selection was pure `max(length)`. Fixed two ways,
confirmed with the user as "prioritize open space over a bit of extra
length":
- New `_path_min_clearance(planner, start_xz, waypoints)`
  (`live_path_planner.py`) — a candidate's BOTTLENECK width: the smallest
  `clearance_m` value among cells actually along its route, sampled every
  `planner.resolution` metres along each simplified straight-line segment
  (not an average — a route that's mostly wide but squeezes through one
  tight gap is only as safe as that squeeze).
- `find_farthest_open_path()` now scores each candidate `length * quality`,
  not `length` alone, where `quality = 1 / (1 + clearance_penalty_scale *
  exp(-clearance_decay_rate * min_clearance))` — same exponential-decay
  shape `LiveGridPathPlanner._clearance_multiplier()` already uses for
  per-cell costing, but with a much GENTLER decay rate (1.5 vs. the
  per-cell model's 8.0) — the per-cell model only cares about avoiding
  imminent collision and saturates by ~1m, but candidate SELECTION should
  keep rewarding a genuinely spacious room over a merely-adequate corridor
  well beyond that range, per the user's explicit ask.
- `mapping_servicer.py` also now constructs WALKING's `LiveGridPathPlanner`
  with `min_path_clearance_m=0.6` (was the default-disabled 0.0) — an
  additional, steeper A* cost penalty for narrow cells, reinforcing the
  same preference WITHIN whichever candidate direction is chosen, not just
  when picking between candidates. Left at the default for GUIDING's own
  planner instance (now constructed separately per-mode, not shared) — an
  aggressive narrow-gap penalty there risks failing to route through a
  real, legitimately narrow doorway on the way to an actual destination.

Verified via a synthetic test: two candidate directions, one a long
(5.5m) narrow (0.5m-wide) corridor, one a shorter (5.0m) but much wider
(3m) room — confirmed the wide room wins now (it lost under the old
pure-length scoring). **Found and fixed a real test-construction mistake
first**: an initial version of this test bounded the synthetic world with
`CLASS_OBSTACLE` everywhere outside the carved corridors, which made
`confirmed_only`'s truncation frontier read an artificially low clearance
for BOTH candidates equally (right at the edge of any carved region is
adjacent to "obstacle" by that construction, regardless of the corridor's
real width) — masking the very difference the test was trying to measure.
Rebuilt with `CLASS_UNKNOWN` as the default backdrop and explicit
real walls (`CLASS_OBSTACLE`) only at each corridor's own edges, matching
how a real partially-explored map actually looks; the fix then measured
correctly. Re-ran the earlier corridor-preference/dead-end/no-clearance-
data-fallback tests too — all still pass.

**Directional-distance candidate selection (supersedes the openness-
weighted `length * quality` scoring above; itself later superseded —
`find_farthest_open_path()` was deleted outright when WALKING moved to
`find_natural_path()`'s cost-function search, see "Natural path planner"
further below)** — requested directly by the user as a deliberate
simplification: "longest" for
`find_farthest_open_path()`'s candidate selection no longer means the
candidate's own raw path length (weighted by clearance or otherwise). It's
now a two-tier rule:
1. **Primary — directional distance.** New `_directional_distance(start_xz,
   end_xz, heading_rad)` (`live_path_planner.py`) — the candidate's NET
   start→end displacement dotted with the CURRENT heading's own unit
   vector, i.e. "how far did this candidate actually get me in the
   direction I'm facing," not the candidate's own (possibly very different)
   probe angle and not its raw path length. A candidate angled far from
   center needs a lot of real travel to make much progress along this
   axis, which naturally penalizes wide-angle candidates relative to ones
   that go more directly forward — even if the wide-angle one has a longer
   raw path.
2. **Tiebreak — shortest actual path.** Among candidates whose directional
   distance is within `directional_tie_tolerance_frac` (0.01, i.e. 1%,
   with a small absolute floor so a near-zero/negative best directional
   distance doesn't collapse the window to nothing) of the best one, pick
   whichever has the smallest `_path_length()` — pure cumulative length,
   "accounting for every direction," not just the heading-aligned
   component. Prefers the more direct/efficient route among near-equally-
   forward options over one that wanders further off-axis for a marginal
   edge.

`_path_min_clearance()` and the `clearance_penalty_scale`/
`clearance_decay_rate` params are deleted outright — no remaining caller.
The underlying PER-CELL clearance-cost model is unaffected and still
active (`LiveGridPathPlanner`'s `min_path_clearance_m=0.6` for WALKING,
set in `mapping_servicer.py`, still shapes WHICH route reaches a GIVEN
candidate goal) — only the top-level candidate-selection formula changed.
Verified via two synthetic tests: (1) a straight-ahead 5m corridor beats
an 85°-angled 8m corridor (directional distance 5.0 vs. 0.8, despite the
8m one being the longer RAW path — confirms directional distance, not raw
length, is now primary); (2) a straight-ahead 5.0m corridor beats a
15°-angled 5.3m corridor whose directional distance (5.10) is within
tolerance of the straight one's (5.00) — confirms the shorter path wins
the tiebreak once directional distances are close. Re-ran the existing
dead-end (fully sealed-off start → `None`) test too, unaffected.

### RTAB-Map confidence weighting skipped for WALKING (`scan_session.py`)

Requested directly by the user: WALKING needs an immediately-usable
occupancy grid for real-time navigation, not scan-grade caution about a
momentarily-uncertain frame. `process_frames_batch()`'s Step 4 still
COMPUTES the same per-batch depth-consistency confidence as before
(`batch_confidence = mean(1 - frac_bad)` across the batch's frames) and
still surfaces it via `self.last_batch_confidence` (e.g. `server_gui.py`'s
dashboard "confidence=" text is unaffected) — but the value actually fed
into `occupancy_map.update(..., confidence=...)` (the WEIGHT that scales
how much a hit/miss moves a cell's log-odds belief, see occupancy_map.py's
"Confidence-weighted occupancy updates" note) is now forced to `1.0` for
`pure_walking` specifically (`update_confidence = 1.0 if pure_walking else
batch_confidence`). A real obstacle now registers at full log-odds
strength on a single walking-mode hit, rather than needing several
confirming hits to reach the same belief a full-confidence hit already
would — matching the user's explicit "we need immediate navigation, take
all into account" framing. GUIDING and SCAN are unaffected — they still
use the real computed `batch_confidence` (unchanged), since neither has
WALKING's same "must react to a single frame, right now" requirement.

**Reported by the user from a live run: the route (both on the occupancy
map AND — more tellingly — the frame overlay, which showed NOTHING at
all despite a wide-open, obviously-walkable floor) didn't align with
their actual facing direction.** Investigated as far as possible without
device access:
- Re-verified `_pose_heading_rad()` (`mapping_servicer.py`) and
  `_project_world_to_pixel()` (`server_gui.py`) against EACH OTHER with a
  proper non-identity-rotation synthetic test (several yaw angles, not
  just the identity-rotation case the original frame-overlay test used,
  which is blind to any rotation-direction/order bug) — a "straight
  ahead" probe point constructed from the computed heading always
  projects to dead-center in the frame, for every tested yaw. The two
  functions are mutually self-consistent; if there's a bug, it's not in
  how they interact.
- That leaves two real possibilities, either of which the frame showing
  NOTHING at all (not even a partial/off-center line) is consistent with:
  (a) RTAB-Map's actual pose convention doesn't match what's assumed
  (camera-to-world, X-right/Y-down/Z-forward) despite that being
  established elsewhere in this codebase, or (b) `ground_y` (a single
  scalar estimate, not a per-point measurement) is inaccurate enough that,
  combined with camera pitch, it flips the FULL 3D camera-local Z
  negative for points that are genuinely in front of the user
  horizontally — a failure mode the original identity-rotation-only test
  couldn't have caught, since it always passed `ground_y == pose.y`
  (zero Y delta).
- **Fixed (b) regardless of which is the true cause**: `_project_world_to_pixel()`'s
  "is this point visible" decision is now HORIZONTAL-ONLY (dot product of
  the world delta against the camera's own floor-plane forward direction —
  the same computation `_pose_heading_rad()` does), decoupled from the
  full 3D projection. `ground_y` is still used for vertical (v) pixel
  placement, but an inaccurate value now degrades to "drawn at the wrong
  height" instead of "silently dropped entirely." Verified via a
  synthetic test: a deliberately absurd `ground_y` (50m off) combined
  with a modest 15° camera pitch no longer drops a genuinely-in-front
  point (it draws off-screen vertically instead, which is the honest
  degrade), while a point truly behind the user is still correctly
  rejected.
- **Not resolved**: whether (a) is also true is still open — added a
  one-shot-per-batch console diagnostic in `mapping_servicer.py`
  (`heading_rad` in both radians and degrees, `pose_y`, `ground_y`, and
  the resulting path-point count) specifically so the next live run's
  console output can be compared against the user's own real-world sense
  of which way they were facing, to localize the mismatch precisely
  rather than guessing further from this environment.

**Proto (`tracking.proto`)** — `MappingChunk` gained `has_goal`/`goal_x`/
`goal_z` (GUIDING only, set once `FindLandmark` resolves a destination;
left unset for WALKING). New `PathPoint`/`PlannedPath` messages
(`points`, `confirmed`, `reached_exactly` — mirrors `find_path()`'s own
return shape). `MappingUpdate` gained `frame_timestamp_ns` (echoes back
whichever `MappingChunk` this update was computed from — see latency
compensation below) and `planned_path`.

**Client — rotation tracking (`RotationTracker.kt`, new)**: mirrors
`TrackingBackend.kt`'s existing on-device ORB detect/match pattern (same
`ORB.create`/`BFMatcher` calls, same JPEG→gray decode) but computes
`Calib3d.findEssentialMat` + `Calib3d.recoverPose` between consecutive
frames instead of a Homography for a 2D box — confirmed both are present
in the `org.opencv:opencv:4.11.0` AAR this project already ships (checked
via `javap` against the actual jar before writing this). Deliberately
rotation-only: monocular vision has no scale for translation (`recoverPose`'s
`t` output is discarded), but rotation decomposition has no such ambiguity
— exactly as metrically correct as RTAB-Map's own rotation. Uses the same
"no real calibration, guess a pinhole K" convention
(`fx=fy=0.8*max(w,h)`) `HrtfBeacon.kt`'s `directionFromBox()` and
`server/tools/depth.py`'s `_estimate_K` already established. Maintains an
internal accumulated-rotation-since-reset quaternion
(`accumulatedRotation()`/`resetAccumulator()`), composed via `newAccum =
oldAccum ⊗ conjugate(recoverPose's R, as a quaternion)` — **not verified
against a live device**: `recoverPose`'s `R` maps a point in the
PREVIOUS frame's camera coordinates into the CURRENT frame's, the
opposite direction needed to compose onto a running world-frame
orientation, hence the conjugate; if a head turn ever sounds mirrored on
a real device, this composition is the first thing to check.

**Client — translation estimate (`PdrStepEstimator.kt`, new)**: registers
Android's built-in `Sensor.TYPE_STEP_DETECTOR` directly (confirmed via
codebase search that `ImuSensor.kt`/`ImuRecorder.kt` are orphaned dead
code — never instantiated anywhere in the live app, shaped for offline
accel/gyro CSV export, not step counting — so this is a small
purpose-built listener instead of resurrecting them). `distanceSinceReset()
= stepsSinceReset * STRIDE_LENGTH_M` (fixed at `0.7f` — no per-user stride
model, matching the user's own "centimetre accuracy is unnecessary"
framing; the error only ever has to survive the ~1s gap before the next
authoritative server fix corrects it, per the design below, so it never
gets the chance to accumulate the way a full PDR session's would).

**Client — combining the two (`HrtfBeacon.extrapolate()`, new)**: given an
authoritative server `Pose`, a rotation-delta quaternion, and a walked
distance, returns a best-current-estimate `Pose` — composes the rotation
(`authoritative.quat ⊗ rotationDelta`, derived to be consistent with
`RotationTracker`'s own accumulation convention above) and walks the
distance along the NEW (rotation-updated) heading, not the stale
authoritative one (negligible difference over the short bridge window,
and the more correct choice of the two). `worldPointFrom()` (walking's old
synthetic-target helper) is deleted outright — dead once server-side
planning took over target selection.

**Client — latency compensation (`ToolDispatcher.kt`,
`LiveSessionState.PoseSendSnapshot`)**: the server's `update.pose`
describes whichever frame carried `update.frameTimestampNs` — already
slightly stale by network + RTAB-Map/DA3 processing time by the time the
response arrives. `buildMappingChunk()` snapshots
`rotationTracker.accumulatedRotation()`/`pdrStepEstimator.
distanceSinceReset()` into `state.poseSendHistory` (a bounded
timestamp-keyed buffer) at the moment each chunk is actually sent. When
the matching `MappingUpdate` arrives, the mapping-stream collector looks
up that snapshot, computes the ADDITIONAL rotation/distance accumulated
since THAT send (`deltaSinceSend = conjugate(accumAtSend) ⊗ accumNow`,
`distanceSinceSend = distanceNow - distanceAtSend`), and fast-forwards
`update.pose` by that amount via `HrtfBeacon.extrapolate()` rather than
accepting the server's pose as "now" outright. Both estimators'
accumulators reset to zero at this point — a fresh bridging window starts
for the next gap. Buffer entries at/before the consumed timestamp are
pruned.

**Client — path-following (`PathPursuit.kt`, new)**: replaces
`LocalPathPlanner.kt`'s role entirely (deleted outright, along with
`MutableOccupancyGrid.kt` — the client no longer holds a local grid at
all now that it doesn't search one). Pure stateless polyline geometry, no
search: `nearestPointOnPath()` projects the current (extrapolated)
position onto the path (clamped per-segment, not the infinite line) and
returns the projection's cumulative arc-length; `advanceAlongPath()` walks
forward from that arc-length by a configurable look-ahead
(`PATH_LOOKAHEAD_M = 0.4f`, the middle of the user's suggested 30-50cm
range), clamping at the path's end. Matches the user's own spec exactly:
"compute the closest point on the path, project onto it, move forward by
a small look-ahead distance" — the beacon target is always a point ON the
path, sliding forward continuously as the user progresses, never a fixed
waypoint index and never off the path.

**Client — unified steering (`ToolDispatcher.kt`)**: `runAvoidanceTick()`
(GUIDING's old goal-biased `TraversabilityScorer` dodge) and
`runWalkingAvoidanceTick()`/`recomputeRoute()`/`steerWalkingBeacon()`/
`checkWaypointProgress()` (WALKING's old grid-planned-route driving code)
are ALL deleted, replaced by one `runUnifiedAvoidanceTick()` +
`steerBeaconAlongPath()` pair used by BOTH modes: the server-planned path
is already obstacle-aware (searched against the occupancy grid's own
clearance costs), so there's no separate local re-scoring needed any
more — a local dodge on top of an already-obstacle-aware path could also
push the beacon off the path, which contradicts the user's explicit "must
never leave the path" requirement. `TraversabilityScorer.kt` is deleted
outright as a result (no remaining caller). The step-down/drop-off hazard
check (`checkAndWarnHazard()`) is unrelated to steering and unaffected —
still runs every tick for both modes via the same
`AnalyzeFrame(TRAVERSABILITY)` call, now serving only that purpose (not a
steering input) for GUIDING too, matching WALKING's already-established
design. GUIDING's arrival announcement moves to a new
`checkGuidingArrival()` (distance from the extrapolated position to the
path's final point, one-shot per destination via `guidingArrivalAnnounced`)
since there's no more discrete waypoint index to advance past.

**Known, accepted limitations / open risks** (discussed above inline,
collected here):
- Rotation composition order (`RotationTracker`'s conjugate convention) is
  derived, not device-verified — flagged as the first thing to check if a
  head turn ever sounds mirrored.
- Fixed stride length, no per-user gait model — acceptable per the user's
  own framing, since the bridging window is short enough that the error
  never compounds across a whole session the way general PDR positioning
  would.
- `find_farthest_open_path()`'s candidate-cone approach is a v1 heuristic,
  not an exact "maximize walkable distance under a turn budget" search.
- `grid`/`grid_delta` are still computed and sent server-side even though
  the client no longer reads them — a real, small, deliberately-deferred
  cleanup, not a correctness bug.
- Not verified end-to-end on a real device/RTAB-Map rig from this
  environment — compile-verified only (`./gradlew :app:compileDebugKotlin`,
  `python3 -m py_compile`), same caveat as every round of this feature's
  development.

### WALKING route-selection redesign — straight-ahead-first, morph-around, minimal joints

**Superseded — see "Natural path planner (heading-biased, turn-averse A*)"
below.** The two-stage straight-ahead/cone-fallback heuristic this section
describes (`find_farthest_open_path()`) has been deleted outright, not
deprecated — the user requested a full replacement with a proper cost-
function-driven search after finding this heuristic still wasn't producing
a natural-feeling route (see that section for the full spec and design).
Kept here for history — the underlying goals (prioritize heading, morph
around obstacles, minimal joints) are unchanged, just achieved differently
now (an actual weighted cost function baked into a direction-aware search,
not a discrete probe-and-pick heuristic).

Reported directly by the user from the dashboard: the WALKING route was
visibly favoring a diagonal detour along the edge of the mapped area
instead of continuing straight in the user's actual facing direction, even
when the floor straight ahead was open. Root cause: the old
`find_farthest_open_path()` always ran its full multi-candidate cone
search and picked purely by directional-distance-then-length across ALL
candidates — nothing in that scoring gave the exact-heading candidate any
priority over a diagonal one that happened to score marginally higher, so
a modest score edge for an angled candidate could win even when straight
ahead was perfectly walkable.

**Redesigned as a two-stage strategy in `find_farthest_open_path()`
(`scan_server/live_path_planner.py`), confirmed directly with the user**:
1. **Stage 1 — straight ahead, always tried first.** A single
   `find_path(start_xz, straight_target, confirmed_only=True)` toward a
   target `probe_range_m` directly along `heading_rad`. A*'s own
   clearance-cost model already bends this route around a SINGLE obstacle
   in the way while still reaching as far forward as it can — this alone
   is "morph the path around a blockage but keep heading the same general
   direction," no separate logic needed. If the result makes at least
   `min_forward_progress_frac` (0.35) of `probe_range_m` worth of real
   forward progress (`_directional_distance()`), it's returned directly —
   no cone search at all, so the user's own heading stays centered in the
   route whenever there's any reasonable way to honor it.
2. **Stage 2 — cone fallback, only when straight-ahead is missing or too
   shallow** (a real wall-like blockage dead ahead): the previous
   multi-candidate probe + directional-distance-primary/length-tiebreak
   selection, unchanged in mechanism — this is the "turn to another
   direction if needed" case. Straight-ahead's own (too-shallow) result is
   also compared against the winning fallback candidate on the same terms
   before deciding, so a straight sliver that's still objectively better
   than every turn option isn't discarded just for having triggered the
   fallback path.

Both stages still call `find_path(..., confirmed_only=True)` — unchanged
from before, still needed so a route never leaves genuinely observed
territory (see the existing "Bug found via the dashboard" note above this
section). Verified via synthetic grids: a fully open field returns a
single straight waypoint (1 joint); a small obstacle centered dead ahead
with open space on both sides returns a 3-waypoint route that goes around
it while still reaching the full forward distance; a full-width wall
straight ahead correctly falls back and, when one side has a real opening
(an L-shaped corridor), turns into it while still maximizing forward
progress.

**Minimum-joints polyline (`LiveGridPathPlanner._simplify()`)**: added
`SIMPLIFY_COST_SLACK_FRAC` (0.08) — the existing cost-aware string-pulling
only used to drop a waypoint when the straight-line shortcut cost NO MORE
than the original zig-zag route it replaces; now it also drops a waypoint
when the shortcut costs up to 8% more. Requested directly by the user
("choose a polyline path with as minimum joints as possible") — a blind
user following a beacon benefits far more from fewer, clearer turns to
react to than from an A*-optimal (but jointier) polyline, and the existing
clearance-cost model already keeps a shortcut route from actually hugging
an obstacle (a straight line through a tight, low-clearance gap costs
enough to fail the slack check on its own). Still current — `_simplify()`
is shared by both `find_path()` (GUIDING) and `find_natural_path()`
(WALKING, below).

### Natural path planner (heading-biased, turn-averse A*) — supersedes WALKING route-selection redesign

Requested directly by the user via a fully-specified planner design (a
priority-ordered objective list, an explicit weighted cost formula, and an
explicitly recommended architecture: distance-transform clearance -> A*
with heading/clearance/turn costs -> polyline simplification) after the
straight-ahead/cone-fallback heuristic (previous section) still wasn't
producing the route an orientation & mobility instructor would choose.
**`find_farthest_open_path()` is deleted outright** (not deprecated) —
`LiveGridPathPlanner.find_natural_path()` (`scan_server/live_path_planner.py`)
replaces it entirely as WALKING's target/route strategy. GUIDING's own
`find_path()` (toward a real destination) is completely unaffected — this
whole redesign is WALKING-only.

**The user's stated priority order** (highest to lowest): (1) continue in
the current heading whenever safely possible; (2) stay near the center of
wide free space; (3) maintain comfortable obstacle clearance; (4) delay
turning until forward progress is genuinely blocked; (5) minimize turn
count; (6) minimize total turn angle; (7) minimize path length. The
planner must never sacrifice a higher priority to improve a lower one.

**Architecture — a direction-augmented Dijkstra, not a plain cell-only
A***: a normal grid A* (like `find_path()`'s own `_astar()`) has no way to
penalize TURNING, only which cell a step lands in — there's no notion of
"the direction I was already moving in." `find_natural_path()`'s search
state is `(row, col, incoming_direction)` instead of just `(row, col)`,
letting each edge's cost depend on whether that step continues straight or
changes direction. This is the "custom A* cost: heading bias + clearance
reward + turn penalty + corridor-center preference" the user's own
recommended architecture called for — implemented as one additive
weighted cost per edge, mirroring their formula almost verbatim
(`_natural_step_cost()`):

```
step_cost = 10 * heading_error      (quadratic, see below)
          +  8 * obstacle_proximity  (0 once clearance >= safe_clearance_m)
          +  7 * turn_count          (flat, only when direction changes)
          +  5 * turn_angle          (proportional to how sharp the turn is)
          +  2 * path_length         (real metres, class-tier-weighted)
```

- **Heading bias**: `heading_error = |wrap(step_azimuth - heading_rad)|`,
  penalized as `W_HEADING * (heading_error / pi) ** 2` — quadratic, so 0
  deg is free, 15 deg is tiny (~0.07), 30 deg moderate (~0.28), 60 deg
  large (~1.11), 90 deg very large (~2.5), matching the user's own example
  bands. Backward motion (180 deg) isn't specially prohibited — at ~10
  per step it's simply expensive enough, accumulated over any real
  backward-biased route, that it's "almost never selected unless no other
  route exists" purely as an emergent consequence of the search always
  finding the minimum-TOTAL-cost path.
- **Obstacle proximity / corridor-centering, same term**: reuses the
  EXISTING `clearance` field (`scipy.ndimage.distance_transform_edt`,
  already computed by `occupancy_map.py` for `find_path()`'s own clearance
  shaping) rather than a second distance-transform pass. `proximity = 0`
  once a cell's own clearance already meets `safe_clearance_m`, else
  `(1 - clearance/safe_clearance_m) ** 2` (steep, not linear — "maintain
  COMFORTABLE clearance" reads as more than a soft nudge once you're
  actually close). No separate "stay centered" logic exists or is needed:
  a corridor's centerline is exactly where clearance — and therefore this
  term's cost — is locally lowest, so the search is pulled there for free.
- **Turn count + turn angle, two separate terms**: a flat `W_TURN_COUNT`
  the instant a step's direction differs at all from the previous step's,
  PLUS `W_TURN_ANGLE * (turn_angle / pi)` scaling with how sharp that turn
  actually is — a 45 deg jog costs less than a 135 deg near-reversal, but
  even the gentlest turn still pays the flat per-turn cost, matching
  "prefer `──────────────┐│` over `──────╱────╲────`" even when the
  zigzag is shorter.
- **Path length, deliberately the smallest weight**: real distance in
  metres, tiered by the SAME `_COST_BY_CLASS` traversability multiplier
  `find_path()`'s A* uses (a step-over cell costs more per metre than
  plain ground) — but only ever a tie-breaker in practice, never a
  deciding factor on its own, per the user's explicit ordering.
- **"Delay turning until forced" and "prefer few gentle turns over a
  shorter zigzag" are NOT special-cased anywhere** — they're an emergent
  property of a turn-penalized shortest-path search: turning earlier or
  more often than strictly necessary always costs strictly more for zero
  benefit, so the minimum-cost path to any given point naturally goes as
  straight as possible for as long as possible. No extra logic was needed
  to implement these two priorities specifically — confirmed by the
  synthetic tests below never observing an unnecessary early/extra turn.

**Obstacle "inflation" — implemented as cost, not a hard block, a
deliberate departure from the user's literal suggested pipeline step**:
the recommended architecture said "inflate obstacles" as a preprocessing
step. A LITERAL hard-block inflation (marking every cell within
`safe_clearance_m` of an obstacle as impassable) would contradict this
codebase's established, repeatedly-reaffirmed "never hard-block a
genuinely narrow gap, only discourage it" philosophy (see
`min_path_clearance_m`'s own docstring, `grid_path_planner.py`'s
clearance-cost note) — a real doorway or narrow hallway narrower than
`safe_clearance_m` would become completely unreachable rather than merely
discouraged. The `obstacle_proximity` cost term above achieves the same
practical effect (strongly avoid getting close to obstacles) while
preserving the "still usable as a genuine last resort" property this
project consistently chooses elsewhere.

**Search region — a literal spatial cone, escalating**: expansion is
pruned to cells whose BEARING FROM THE START (not from the current cell —
a region constraint, distinct from the per-step turn-angle cost above)
falls within the current stage's cone half-angle of `heading_rad`. Starts
at 45 deg per the user's spec; if that yields nothing OR only a sliver of
real forward progress (less than `min_progress_frac`, 0.3, of
`max_distance_m`), escalates to 90 deg, then 180 deg. **The
"only escalate on a sliver of progress, not just on zero" refinement was
a fix found during testing**: an initial version escalated only when a
narrower cone found NOTHING AT ALL — but a corridor with a metre or two of
real open floor before a full-width wall would "succeed" at 45 deg (SOME
path exists) and never even look for a real opening off to the side,
which isn't what "only turn when genuinely blocked" means. Since every
wider cone's reachable set is a strict superset of a narrower one's (same
edges, plus more), escalating never discards a better result already
found. The widest attempted stage's own result — even if still short of
`min_progress_frac` — is always returned rather than `None`, matching this
codebase's established "closest-approach beats no answer" philosophy;
`None` (WALKING's dead-end trigger) is reserved for truly nothing
reachable at all, even at 180 deg.

Expansion never crosses into a still-unexplored (`CLASS_UNKNOWN`) cell —
pruned directly during expansion now, rather than via the old
`find_path(confirmed_only=True)` post-hoc truncation, which was removed
outright alongside `find_farthest_open_path()` since nothing else called
it (`_truncate_to_confirmed()` deleted too). This preserves the same
"never leave confirmed territory" bug fix that parameter existed for (see
the earlier "Bug found via the dashboard" note).

**Target selection**: among every cell the search actually reaches within
`max_distance_m`, pick whichever has the greatest `_directional_distance()`
progress along the heading axis — not the farthest by raw distance, not
the cheapest by cost alone. Since edge cost already strongly penalizes
heading deviation and turning, the cell that progresses farthest ALONG
THE HEADING axis naturally tends to be the one reached via the straightest,
safest available route — this single criterion is what replaces the old
two-stage straight-then-cone-fallback heuristic with one continuous
decision. Ties (same directional distance) are broken implicitly by
whichever state the search SETTLED first (heap-pop/cost order), which
tends to favor the cheaper of two equally-far options with no extra code
needed for an explicit tiebreak.

**Verified via synthetic grids** (same technique used throughout this
project — hand-built occupancy grids with a real `scipy.ndimage.
distance_transform_edt`-computed `clearance` field, not just class labels):
a fully open field returns a single straight waypoint (0 turns); a small
obstacle centered dead ahead with open space on both sides morphs around
it (2 waypoints, 1 turn) while still reaching ~full forward distance; a
full-width wall with an opening far to one side correctly escalates
through the cone stages and turns into the opening (confirming the
sliver-of-progress escalation fix actually works, not just the
zero-progress case); a wall positioned closer than `safe_clearance_m` to
the starting position causes the route to drift toward the open side for
comfortable clearance (confirming corridor-centering emerges from the
clearance-cost term with no separate centering logic); a start cell fully
boxed in by obstacles on all 8 sides correctly returns `None`.

**mapping_servicer.py** — the WALKING branch now constructs a plain
`LiveGridPathPlanner(grid_for_planning)` (no `min_path_clearance_m` —
`find_natural_path()` doesn't use `_cost()`/`_clearance_multiplier()` at
all, it has its own independent `safe_clearance_m` concept) and calls
`planner.find_natural_path(pose_xz, heading_rad, max_distance_m=
_WALKING_MAX_PLANNING_DISTANCE_M, safe_clearance_m=_WALKING_SAFE_CLEARANCE_M)`
— `_WALKING_MAX_PLANNING_DISTANCE_M` (5.0m, the middle of the user's
"typically 4-6m" spec) and `_WALKING_SAFE_CLEARANCE_M` (0.5m, repurposed
from the old `_WALKING_MIN_PATH_CLEARANCE_M` constant) replace the old
`_WALKING_MAX_TURN_RAD`/`_WALKING_MIN_PATH_CLEARANCE_M` pair. `heading_rad`
(`_pose_heading_rad()`) is unchanged — still computed unconditionally per
update (see "Facing-direction GUI indicator" below), still what feeds this
planner. GUIDING's own `LiveGridPathPlanner(grid_for_planning)` +
`find_path(pose_xz, current_goal_xz)` call is completely untouched.

**Known, accepted limitations** (not yet verified live):
- Not verified end-to-end on a real device/RTAB-Map rig — synthetic-grid-
  verified only (`py_compile` + the hand-built-grid tests above), same
  standing caveat as every round of this feature's development. Whether
  the route "feels natural" in practice, and whether `min_progress_frac`/
  `safe_clearance_m`/the weight constants need real-world tuning, is
  unverified from this environment.
- The direction-augmented search visits up to (cells within
  `max_distance_m` of start, bounded further by the cone) × 8 direction-
  states — bounded and Python-`heapq`-based like `find_path()`'s own A*,
  but a genuinely large explored area combined with a wide (180 deg)
  escalation could cost more per update than the old heuristic's fixed
  9-candidate probe did. Not benchmarked against a live RTAB-Map grid from
  this environment.
- `cone_stages_deg`/`min_progress_frac` are call-site defaults on
  `find_natural_path()`, not yet independently GUI-tunable the way
  `min_path_clearance_m` is exposed in `scan_gui.py`'s own settings — a
  reasonable follow-up if real-world tuning turns out to be needed.

**Follow-up round, from a real live-device screenshot** (heading arrow now
confirmed pointing the right way — see "Facing-direction GUI indicator"
below — but the route itself was hugging one obstacle despite clearly more
open space on the other side; the joint count already looked reasonable
and needed no change; and a serious HRTF-placement bug was found while
checking the beacon marker's position against the route):

**1. Corridor-centering gap, fixed** — `_natural_step_cost()`'s
`obstacle_proximity` term originally had a hard cutoff: ANY cell whose
clearance already met `safe_clearance_m` cost exactly 0, so a cell 0.6m
from a wall and one 3m from a wall (much more open) scored IDENTICALLY —
nothing ever pulled the route toward the wider side of a room once the
bare minimum was satisfied, which is exactly what let it hug close to one
obstacle with far more open space just to the other side. Fixed by
splitting the term into two: the original hard-cutoff `steep` component
(priority #3, "maintain comfortable clearance") stays, PLUS a new `soft`
component with no cutoff at all — a gentle, always-active exponential
decay (`SOFT_CLEARANCE_DECAY_RATE=1.5`, deliberately much gentler than
`_clearance_multiplier`'s own `CLEARANCE_DECAY_RATE=8.0`, which is tuned
for "avoid imminent collision" and is negligible past ~1m — this needs to
keep discriminating over several metres of open room) that genuinely
implements priority #2 ("stay near the center of wide free space") as an
always-active preference, not just a threshold. Verified via a synthetic
asymmetric-room test: a wall close on one side, wide open space on the
other — the route now visibly drifts toward the open side instead of
running parallel to the wall at a constant offset.

**2. Polyline joint count — confirmed already correct, no change made.**
Re-checked against the user's own "1-3 joints is fine, more starts to feel
taxing" guidance: the existing synthetic tests already produce 1 joint
(open field), 2 (single obstacle morph), and 3 (wall with a real opening)
— already inside the stated comfortable range. `SIMPLIFY_COST_SLACK_FRAC`
left unchanged.

**3. HRTF beacon placement bug — a real navigation bug, not just a
dashboard display quirk.** Investigating "why does the beacon marker sit
so far up the route instead of close to the user" found the actual root
cause: `nearest_point_on_path()`/`advance_along_path()`
(`live_path_planner.py`'s Python mirror AND `PathPursuit.kt`, the real
on-device logic they mirror) only ever operate on whichever segments are
IN the `path` list passed to them — and the server's own waypoints NEVER
include the start position (documented convention throughout this
codebase). This means the very first leg (start -> first real waypoint)
never existed as a segment to project onto/advance along at all; worse, a
path that collapsed to a SINGLE waypoint (which `find_natural_path()`'s
new minimal-joint design produces often — see the "open field" test
above) hit both functions' `len(path) == 1` special case, which returns
`path[0]` directly, ignoring the current position and the look-ahead
distance ENTIRELY — placing the beacon straight at the far target instead
of a short lookahead ahead of the user. This is exactly the symptom in
the screenshot (magenta marker sitting at the route's far end).

Fixed on BOTH sides by prepending the pursuit's actual starting position
to the path BEFORE calling `nearestPointOnPath()`/`advanceAlongPath()` —
not inside those functions themselves (kept as generic, unchanged
utilities, still exact mirrors of each other):
- **Client (`ToolDispatcher.kt`)**: the mapping-stream collector now sets
  `state.plannedPath = listOf(update.pose.x to update.pose.z) +
  rawWaypoints` (only when `rawWaypoints` is non-empty — an empty path
  stays empty, preserving the existing dead-end-mute detection via
  `plannedPath.isEmpty()`). Critically, this anchor is a FIXED point,
  captured ONCE per `MappingUpdate` (the authoritative `update.pose` the
  server actually planned from), not re-added every `steerBeaconAlongPath()`
  tick — re-adding a FRESH (moving) pose every tick was considered and
  rejected: since the newly-prepended point would always be at distance
  0 from itself, it would trivially "win" the nearest-point search every
  single time regardless of real progress, which would break correct
  progression along a genuine multi-waypoint route (the beacon would
  never advance past treating "toward waypoint 1" as the target, even
  after the user had already walked past it toward waypoint 2). A FIXED
  anchor avoids this: real distance-based nearest-point search against the
  REST of the path still works exactly as before once the user has
  genuinely progressed past the first leg.
- **Server (`mapping_servicer.py`)**: the dashboard-only `beacon_point`
  mirror prepends `pose_xz` fresh on every call instead — safe here
  (unlike the client) because this is a single-instant snapshot
  calculation, not a repeated per-tick one; `pose_xz` and the path are
  always from the exact same `MappingUpdate`, never re-anchored across
  ticks.

**4. HRTF distance/loudness was never real — also fixed.** Checking
whether the beacon's perceived distance ("should sound different than
close one") was correct surfaced a second, independent gap:
`HrtfBeaconPlayer.updateDirection(azimuth, elevation, distanceM)` already
computes a real distance-based gain falloff (`overallGain = (1 -
distanceM/MAX_RANGE_M).coerceIn(...)`) — but `steerBeaconAlongPath()` was
passing the FIXED `beaconRadiusM` constant (6f) into it instead of any
REAL distance, so the beacon's loudness NEVER actually varied with true
proximity to anything, regardless of how close or far the target really
was. Fixed: `steerBeaconAlongPath()` now computes a real distance via
`HrtfBeacon.directionTo(pose, gx, gz).distanceM` where `(gx, gz)` is
`state.plannedPath.last()` — the path's actual FINAL point (the real
destination for GUIDING, or how far the currently-planned route reaches
for WALKING), NOT the short look-ahead steering point used for azimuth.
This distinction is deliberate: the look-ahead point is always ~
`PATH_LOOKAHEAD_M` (0.4m) away by construction, so ITS real distance would
barely vary tick to tick and wouldn't communicate anything meaningful — the
path's actual endpoint is a genuine, continuously-varying "how far is
there left to go" signal, while azimuth still comes from the look-ahead
point for smooth, responsive steering. `beaconElevationDeg`/`beaconRadiusM`
(ToolDispatcher constructor params) are left declared but unused by this
path now — not removed, since they're still user-settable via
`SettingsScreen` and could matter again if tracking mode's own beacon path
(a separate call site, `directionFromBox()`-driven, unaffected by any of
this) is ever revisited. Not verified end-to-end on a real device from
this environment — the gain curve itself (`MAX_RANGE_M=15f`,
`MIN_GAIN`/`MAX_GAIN`) was not retuned, only fed real data for the first
time.

**5. Dashboard layout — beacon graph now side by side with BOTH detail
boxes** (`server_gui.py`'s Mapping tab): requested directly by the user.
The mapping Detail textbox used to be its own full-width row between the
frame/occupancy-map row and the beacon row; now the beacon polar plot
shares one `gr.Row()` with a single column holding BOTH the mapping detail
and the beacon detail textboxes stacked together, instead of the mapping
detail floating alone above the beacon section.

### Facing-direction GUI indicator (dashboard display only)

Requested directly by the user, in the middle of debugging whether the
WALKING route/heading was actually wrong (see the two entries above and
below) — a direct visual answer to "which way does the server think the
user is currently facing," so it can be compared at a glance against the
route and the actual camera frame instead of only inferred from console
`heading_rad` prints.

`_pose_heading_rad(pose_mat)` (`mapping_servicer.py`) is now computed
UNCONDITIONALLY per update (previously only inside the WALKING branch,
for `find_farthest_open_path()`'s own use) and passed into
`record_mapping()` as `heading_rad` — `None` on a `pure_walking` RESET
update (stale, same reasoning as clearing `planned_path`/`beacon_point`
there). `server_gui.py` renders it two ways, both reusing the field
directly with no new computation:
- **Occupancy map** (`_render_occupancy`): a short yellow arrow via
  `fig.add_annotation()` from the current position, `arrow_len` (0.6m)
  along `(sin(heading_rad), cos(heading_rad))` — same world X/Z
  convention every other angle in this codebase uses. `add_annotation()`
  draws the arrow but doesn't add a legend entry on its own, so a
  near-invisible marker (`size=0.1`) at the arrow tip stands in for one.
- **Frame overlay** (`_annotate_mapping`): appended to the existing green
  pose/confidence text line as `heading=NNdeg`.

Purely diagnostic/display — doesn't change any planning or steering
behavior. If the arrow's direction visibly doesn't match where the camera
frame is actually pointed, that's the clearest possible signal that
`_pose_heading_rad()`'s assumed camera-local convention (or RTAB-Map's
actual reported convention) is the real bug, vs. the route-selection
algorithm itself.

### WALKING occupancy-map "no accumulation / doubled voxel size" — tried, reverted

Briefly tried, per two follow-up questions from the user (was there any
accumulation before a WALKING frame's points land on the map — point-cloud
accumulation was already confirmed absent (`walking_lite` skips Step 3b's
`get_cloud()`/SOR pull entirely, WALKING's points come only from Step 3's
per-frame local back-projection); Bayesian LOG-ODDS BELIEF accumulation
was real, though, so `OccupancyMap.update()` gained a per-call
`bayesian_override` param and `scan_session.py` passed
`bayesian_override=False` + a doubled `occupancy_voxel_size` for
`pure_walking`, meaning a single hit classified its cell immediately at a
coarser resolution instead of needing several agreeing observations.

**Reported by the user as not working in practice — reverted outright.**
`OccupancyMap.update()` is back to its original signature (no
`bayesian_override` param); `scan_session.py`'s Step 4 is back to the
plain `voxelize_cloud(new_cloud_clean, voxel_size=occupancy_voxel_size,
...)` / `self.occupancy_map.update(traj, occ_centers,
confidence=update_confidence)` calls, unconditional on `pure_walking`
again. What specifically broke wasn't diagnosed from this environment (no
device/live run available here) — if revisited, start by checking whether
single-hit immediate classification let through more depth-noise false
positives than the accumulate-and-revise design was filtering out, since
that's the main safety property `enable_bayesian=True` provides.

### Server-side HRTF beacon placement mirror (dashboard display only)

Requested directly by the user: show where the HRTF beacon is actually
pointing on the dashboard, not just the route it's derived from. The real
beacon placement — pose extrapolation via `RotationTracker`/
`PdrStepEstimator`, then `PathPursuit`'s project-onto-path-then-advance —
only ever happens on-device (see "Server-planned walking path" above);
this adds a **server-side mirror of the placement math only** (not the
extrapolation, which is meaningless server-side — the server only ever has
its own last authoritative pose, never a latency-bridged estimate), purely
so `server_gui.py` can show a directly comparable "this is where the sound
source is" marker alongside the route it's already drawing.

`scan_server/live_path_planner.py` gained a direct Python port of
`PathPursuit.kt`: `nearest_point_on_path()`, `advance_along_path()`, and
`beacon_target_point()` (combines both — project the current pose onto
`planned_path`, then move forward by `PATH_LOOKAHEAD_M`, a module constant
kept at `0.4` to match `ToolDispatcher.kt`'s own constant exactly). Verified
against hand-computed expected points (pose at the path start, pose offset
to the side of a segment, pose near the path's end clamping at the final
point) — all three match the Kotlin implementation's documented behavior
exactly.

`mapping_servicer.py`'s `UpdateMapping` calls `beacon_target_point()` right
after computing `planned_path_proto` (using the SAME `pose_xz` and path
points already being sent to the client) and records the result into
`ActivityMonitor` as `beacon_point` (`None` when there's no path — mirrors
the client's own mute-when-no-path behavior; also explicitly cleared on a
`pure_walking` reset, alongside `planned_path`, so the dashboard doesn't
keep showing a stale marker from before the reset). `server_gui.py` draws
it as a magenta marker in both places the route already appears: a
`go.Scatter` point on the occupancy map (`_render_occupancy`, added via
`fig.add_trace()` after `render_plotly()` returns), and a magenta ring on
the frame overlay (`_annotate_mapping`, via the same
`_project_world_to_pixel()` the route polyline already uses) — distinct
color from the cyan route so it reads as "the sound source," not just
another route point.

### Hazard warnings — step-down frames fed to Gemini Live (walking + guiding)

Requested directly by the user: when there's a step down/stairs, the
current frame should go to Gemini Live with context so it can give the
user a specific spoken warning — not just an ambient tone/beacon nudge.
Applies to BOTH walking and guiding — both modes' avoidance ticks already
fetch a `TraversabilityInfo` fan every cycle (`fetchTraversability()`), so
this slots into both with no new per-tick round trip.

**Originally also covered plain nearby obstacles — removed.** The first
version of this warned on ANY close obstacle in the fan (`clearance_m`
below a threshold, anywhere), which turned out to false-trigger constantly
in real testing: something a metre off to one side with plenty of open
space elsewhere would still fire a warning, which is exactly the "too
eager" complaint that also killed the corridor-lock design above. Per the
user's explicit call: keep the step-down/drop-off check proximity-based
(falling is dangerous enough to warrant its own signal regardless of
whether a path exists around it) but drop the plain-obstacle trigger
entirely — a nearby obstacle the grid/path-planner already routes around
silently is not itself hazard-warning-worthy; only "no path at all"
(WALKING's `playDeadEndAlert()`, see "Grid-planned walking route" above)
is.

**A real gap surfaced while designing the step-down check**: the existing
clearance fan (`_clearance_fan_from_depth`) only ever classified points
ABOVE the fitted ground plane as obstacles. A downward step, ledge, or
staircase registers as neither obstacle nor ground under that scheme —
simply absent from `clearance_m`.

**Fix — `dropoff_m`, a second per-bin array (`server/tools/traversability.py`,
`tracking.proto`'s `TraversabilityInfo`)**: alongside the existing
`is_obstacle = height_above > _OBSTACLE_MIN_HEIGHT_M` check,
`_clearance_fan_from_depth()` now also computes `is_dropoff = height_above
< -_DROP_MIN_DEPTH_M` (0.15m — same order of magnitude as
`_OBSTACLE_MIN_HEIGHT_M`, a real step/curb is typically at least that
deep) whenever a ground plane was found, bins those points by azimuth the
same way, and returns the nearest-drop-off-per-bin distance as `dropoff_m`
(same shape/sentinel-at-`max_range_m` semantics as `clearance_m`). Only
computed when `plane is not None` — no plane means no reference to measure
"below" against, and the existing "no confident floor -> treat everything
as obstacle" fallback already covers that frame conservatively. Still
current and unaffected by the corridor-lock design's removal —
`estimate_traversability()`/`dropoff_m` are what both `runAvoidanceTick()`
(GUIDING) and `runWalkingAvoidanceTick()` (WALKING) call every tick now.

**Client (`ToolDispatcher.kt`) — `checkAndWarnHazard(trav, frame)`**,
shared by both ticks: scans `dropoff_m` for the nearest drop-off; within
`STEP_DOWN_WARN_RANGE_M` (3.0m), sends the current frame via
`sendVideoFrame()` (walking/guiding don't otherwise stream any video into
the Gemini Live session — frames only reach Gemini via explicit tool
calls like `get_latest_frame`/`start_vision_stream` — so this is
necessary, not redundant with anything already flowing) followed by a
`sendSystemNote()` whose text deliberately asks Gemini to describe what it
SEES rather than just read back the distance number. Additive, not a
replacement, for whichever ambient signal the mode already gives (the
beacon's own steering, or `playDeadEndAlert()`'s tone for walking's
dead-end case) — the tone/beacon stays the zero-latency instant cue, this
is the more-informative one a beat later once Gemini responds.
Rate-limited (`hazardActive`/`lastHazardWarnedAtMs`): fires once when the
hazard first appears, then at most once per `HAZARD_REWARN_COOLDOWN_MS`
(8s) while it persists — a standing hazard doesn't get re-announced every
single tick, but also isn't silently forgotten if the user just stands
there.

**`dropoff_m` false-positive fix (`server/tools/traversability.py`)**:
`is_dropoff` originally flagged a bin the instant a SINGLE pixel read more
than `_DROP_MIN_DEPTH_M` below the fitted ground plane — with monocular
(DA3) depth being noisy per-pixel, a couple of stray floor pixels
routinely dipped below that threshold, so nearly every frame read a
drop-off SOMEWHERE, defeating the whole point of a proximity check. Fixed
via `_MIN_DROPOFF_POINTS_PER_BIN` (4) — a bin's drop-off distance is only
trusted once that many points in the SAME bin agree, mirroring
`_MIN_GROUND_INLIERS`'s own "require several agreeing observations"
principle for the plane fit itself. Obstacle detection (`clearance_m`)
never had this problem — a bin's reported distance is naturally dominated
by whichever real cluster is nearest, noise or not.

**Grouped into one hedged prompt, obstacle warning brought back
alongside drop-off (`ToolDispatcher.kt`'s `checkAndWarnHazard()`)**:
requested directly by the user, given that neither heuristic (obstacle
clearance OR drop-off) is reliable enough to assert as fact even after the
fix above — single-frame monocular depth + RANSAC ground-plane fitting is
inherently rough. `checkAndWarnHazard()` now checks BOTH `clearance_m`
(new `OBSTACLE_WARN_RANGE_M`, 2.0m) and `dropoff_m`
(`STEP_DOWN_WARN_RANGE_M`, 3.0m still) and, if either trips, sends ONE
system note per tick (never two) with a single fixed, deliberately hedged
prompt: check the frame for a step-down/drop-off/obstacle right in front
of the user, only warn if one is CLEARLY visible right in front, otherwise
say nothing — handing the actual judgment call to Gemini's own vision on
the real frame rather than asserting a distance number as fact. This
replaced an earlier, more detailed version of the prompt that spelled out
which heuristic(s) fired and their distances — simplified per direct user
feedback to just the fixed instruction above.

### Session-mode pipeline split — SCAN vs. WALKING/GUIDING (`walking_lite`)

Real bug, found via a live device session and fixed: walking mode was
running the exact same FULL pipeline scanning does — RTAB-Map's own
`get_cloud()` reconstruction pull + SOR + server-side voxelize (measured
15s+ and 12s+ respectively on a real batch) AND VLM/semantic tagging — on
every single mini-batch, making walking unusably slow (30-53s stalls
between occupancy updates) for a mode that only ever needed a live
occupancy grid, never a persisted point cloud or landmark discovery.

**Update (post "Grid-planned walking route"): `SessionMode.WALKING` is live
again, through the SAME `feedMappingFrame()`/`startMappingStream()` path
GUIDING/SCANNING use — not a separate stream.** Walking dropped
`MappingService` entirely per "Local reactive HRTF obstacle-dodge," making
`SessionMode.WALKING` dead for a while; a later redesign ("Local SLAM-backed
walking corridor-lock," now itself superseded) revived it on a dedicated
pose-only stream; the current design ("Grid-planned walking route") merged
it back into the shared stream, now carrying the FULL grid — walking gets
the same live occupancy grid GUIDING does (`pure_walking` no longer skips
Step 4), just never persisted and reset far more aggressively on tracking
loss (`PURE_WALKING_LOST_RESET_S`, 0.5s). `walking_lite`'s existing
`pure_walking`-gated skip of Step 3b (get_cloud/SOR) and VLM tagging still
applies regardless, so the original "unusably slow" bug this section fixed
stays fixed — walking's per-batch cost is still Step 3's local
back-projection + Step 4's grid update, same as it's always been for this
mode, just no longer skipping Step 4 specifically.

**Chosen fix**: an explicit `SessionMode` (`tracking.proto`: `SCAN`,
`WALKING`, `GUIDING`) on `MappingChunk`, set once by the client
(`ToolDispatcher.feedMappingFrame()`, from `state.mode`) and read by the
server only on a stream's first chunk. `MappingServiceServicer.
UpdateMapping` translates non-`SCAN` modes into `StreamingScanSession(...,
walking_lite=True)`, which threads through to `ScanSession.
process_frames_batch(walking_lite=True)`:

- **Step 3's local back-projection** (normally IMU+VO/VO only) also runs
  for RTAB-Map-posed frames when `walking_lite` — using RTAB-Map's pose
  (still authoritative) + this frame's own already-computed DA3 depth,
  gated by the SAME depth-consistency check IMU+VO/VO already use (now
  actually populated for the RTAB-Map branch too — previously left at a
  hardcoded "trustworthy" default there, since only the node-veto side
  channel needed it before). No TSDF fusion (PLY/Live-Points quality isn't
  needed for a live occupancy grid, which is already self-correcting).
- **Step 3b (RTAB-Map's `get_cloud()`/SOR/server-voxelize pull) is skipped
  entirely** — Step 3's local back-projection already fed `new_cloud`,
  which Step 4's occupancy update (already fully generic across pose
  sources) picks up unchanged. This is the single biggest cost removed.
- **Semantic tagging (`_PendingTagFrame`/`_tag_pending`) is
  skipped entirely** — semantic mapping is scan-only now; walking/guiding
  never discover landmarks live.
- **`FindLandmark` gained a persisted-snapshot fallback**
  (`MappingServiceServicer._find_in_snapshot()`) for exactly this reason —
  a walking/guiding session's `_raw_landmarks` is always empty (semantic
  tagging never ran), so `session.resolve_landmark()` alone would never find
  anything for it; `_find_in_snapshot()` does the same case-insensitive
  substring match against whatever a PRIOR scan already persisted to
  `occupancy_snapshot.json`, giving "if there's already a map with
  semantics scanned during scan mode, use it" a real implementation.
- **`toolStopScan()` (Android) now auto-transitions straight into walking**
  once scanning stops — `stopMappingStreamAndAwait()` closes the chunk
  channel (clean half-close, not a job cancel) and `join()`s the collect
  loop so the server's `UpdateMapping` `finally` block (flush + finalize +
  snapshot save) has actually run before `toolStartWalking()` fires,
  rather than racing a stream that's still finalizing server-side.

**Verified** (mocked estimator/RTAB-Map client, no GPU/docker dependency):
`process_frames_batch(use_rtabmap_pose=True, walking_lite=True)` populates
the occupancy map via local back-projection, leaves `_raw_landmarks` empty,
and never calls the mock `get_cloud()` (which raises if invoked) — matching
the intended behavior exactly.

**Known, deliberately out of scope for this pass**: scanning itself still
runs its full live pipeline (get_cloud/SOR/VLM tagging per mini-batch, same
as before) rather than the deferred "accumulate RGB-D during scan, batch-
process + ORB-novelty-gate for VLM tagging only after scan stops" design
also discussed — that's a separate, larger follow-up, not yet built.

### Occupancy grid delta sync

Real bug, found and fixed during this work: `UpdateMapping` used to
re-transmit the ENTIRE accumulated occupancy grid (`extract_full_grid()`)
every single time `grid_updated` was true — fine for a small map, but a
session/map that grows over a longer guiding/walking/scanning session would
keep re-shipping a larger and larger payload on every batch, indefinitely.

**Chosen fix**: incremental cell-level sync, falling back to a full resync
only when needed — same "full resync when structure changes, incremental
otherwise" split this codebase already uses for RTAB-Map loop closure
(`_rtabmap_full_resync()` vs. `_rtabmap_pull_new_nodes()`).

- **`OccupancyMap._dirty_cells`** (new, `occupancy_map.py`) — every
  `_register_obstacle_hit`/`_register_ground_hit`/`_register_miss` call
  (i.e. every cell actually touched by an `update()`, not just ones whose
  classification flipped) adds that cell's key to this set. `extract_dirty_
  delta()` (new) returns only those cells — `{ix, iz, class, height_norm,
  clearance}` per cell, addressed by GLOBAL grid index, not row/col
  relative to any particular window — and clears the set. `bounds()` (new)
  cheaply returns just the current bounding box `(ix_lo, iz_lo, width,
  height)` without paying for a full classification pass, so the caller can
  decide full-vs-delta before committing to either. `clear_dirty()` (new)
  discards pending dirty cells after a full send makes them redundant.
  **Known, accepted imprecision**: a cell whose CLEARANCE changed because a
  NEARBY cell (not itself) just became/stopped being an obstacle, without
  itself being touched this batch, can go briefly stale until it's next
  touched itself — not fixed, because `grid_path_planner.py`'s
  `CLEARANCE_DECAY_RATE` already saturates the clearance-cost curve to
  ~1.0x (no practical path-cost effect) by ~1m from any obstacle, and cells
  near a just-touched obstacle are overwhelmingly likely to be touched in
  the very same batch anyway (same source depth frame) — a full EDT is
  still recomputed on every delta export (cheap, vectorized), just not
  propagated to untouched neighbor cells' delta payload.
- **`mapping_servicer.py`'s full-vs-delta decision** — `MappingServiceServicer.
  _last_full_bounds` (keyed by `location_id`) caches the bounds as of
  the last FULL grid sent. A `full_resync` is forced when there's no cached
  entry (first update for this stream — reset via `_last_full_bounds.pop()`
  at stream-open, since this cache is servicer-instance-scoped and would
  otherwise wrongly survive across separate streams for the same
  `location_id` and cause a coincidental bounds match to skip a resync the
  new stream actually needs) or when `bounds()` no longer matches the
  cached value (the explored area grew). Otherwise `extract_dirty_delta()`
  is sent instead. (The `_last_open_direction` cache this bullet used to
  also mention was part of `beacon_preview.py`'s dashboard-only beacon
  reconstruction, deleted outright — see "Local reactive HRTF
  obstacle-dodge".)
- **`MutableOccupancyGrid.kt`** (new, Android) — the client no longer
  replaces `LiveSessionState.lastMappingGrid` wholesale on every update.
  `state.mutableGrid` is a persistent, patchable backing store:
  `fromFull()` replaces it wholesale on `full_resync`; `applyDelta()`
  patches specific cells in place otherwise, converting each delta cell's
  global `(ix, iz)` back to a local `(row, col)` via the SAME origin the
  grid was last fully built from (`(originX/cellSize).roundToInt()`, etc.
  — this only stays valid because a delta is never applied except on top of
  a grid whose bounds provably haven't changed, per the server's own
  `full_resync` decision). `toProto()` cheaply repackages the current
  mutable arrays back into an immutable `Tracking.OccupancyGrid` for
  existing consumers (`LocalPathPlanner`'s A*, guiding-only now — see
  "Local reactive HRTF obstacle-dodge" for why walking no longer reads this
  grid at all) — no reimplementation needed there at all, `ToolDispatcher`'s
  collect loop just assigns `state.lastMappingGrid =
  state.mutableGrid?.toProto()` after applying whichever kind of update
  arrived, same as before.
- **Server CPU cost is unchanged** — `extract_dirty_delta()` still runs the
  same `_build_grid_dict()` classification + EDT pass over the full current
  bounding box as `extract_full_grid()` always did (needed for
  authoritative clearance values); this fix reduces what goes out over
  gRPC, not server-side compute. A real reduction in classification cost
  itself (e.g. incrementally maintaining classification/EDT instead of
  recomputing from scratch each export) was considered out of scope for
  this pass.

### Drop-to-latest mapping-chunk ingestion — tried, reverted, then re-adopted (`mapping_servicer.py`)

**Current state: RE-ADOPTED, deliberately, as of the "Server-planned
walking path" work.** `UpdateMapping` iterates `_latest_only_chunks(
request_iterator)`, not the raw `request_iterator` directly — chunks
arriving while the server is still busy on an earlier one ARE silently
dropped again. This is a **conscious re-reversal**, not a regression: the
user explicitly asked for it (to keep the server from working through a
growing backlog) after being shown the exact history below, and paired it
with a shorter consecutive-tracking-loss reset
(`PURE_WALKING_LOST_RESET_S`, `scan_session.py`, bumped from 0.5s to 1.0s
in the same pass) as the accepted mitigation for the known risk this
reintroduces. The mailbox itself had to be reconstructed from this
section's own prose (the original revert happened within uncommitted
session work, so there was no git history to recover the literal code
from) — same single-slot-mailbox shape as before: a background daemon
thread drains the real `request_iterator`, overwriting one shared slot +
a monotonic sequence counter under a `threading.Condition`; the generator
blocks on the condition and yields whatever's currently in the slot once
notified. Verified via a standalone synthetic test (slow consumer sees a
strictly-increasing but gapped sequence, always including the first and
last item; a fast consumer sees ~every item; an upstream exception/an
empty source both terminate cleanly) rather than trusted on prose alone.

**What this trades away (unchanged from the original attempt — read this
before assuming the risk is gone)**: gRPC's own request iterator queues
incoming messages internally — if the server-side loop falls behind the
client's send rate even briefly (a slow `push_frame()` mini-batch, GPU
contention from another stream, a depth-estimation stall), the DEFAULT
behavior (full queue) works through that backlog in arrival order, so
RTAB-Map always sees every consecutive frame even if its reported pose
lags real time. The mailbox instead keeps the server current in wall-clock
time by dropping whatever piled up while it was busy — but RTAB-Map's own
frame-to-frame odometry needs CONTINUITY, not freshness: dropping frames
widens the visual/motion gap between two frames it actually processes back
to back, which was the confirmed contributor to a real tracking-loss
regression the first time this was tried (see "DA3 model default +
per-frame processing" below, Round 1's reasoning — that investigation's
own conclusions about `mini_batch` are unaffected either way). This mirrors
`CameraManager.kt`'s own `frameFlow` (`extraBufferCapacity` + `DROP_OLDEST`)
— the same "prefer fresh over complete" policy already used client-side,
and now also reflected in `ToolDispatcher.kt`'s outbound
`mappingChunkChannel` (`Channel.CONFLATED`, replacing `Channel.UNLIMITED`)
and `LiveSessionState.lastSentSnapshot` (a single latest-sent pose/PDR
snapshot, replacing a multi-entry history buffer — see "Server-planned
walking path" for why a full history stopped making sense once the server
no longer guarantees processing every sent chunk in order). Applies only
to `UpdateMapping` — the only streaming RPC left in this codebase
(`TrackingService`/`PerceptionService` are unary, one frame per call, so
they have no equivalent queueing exposure).

**Not verified end-to-end against a live RTAB-Map rig** — the mailbox's
own drop/keep-latest logic is unit-tested (above), but whether it actually
degrades real tracking-loss frequency in practice, and whether the 1.0s
reset is enough headroom to absorb that, is unverified from this
environment, same standing caveat as this session's other changes.

### DA3 model default + per-frame processing + pre-DA3 blur gate (`scan_session.py`, `stream_session.py`, `da3_wrapper.py`, `grpc_server.py`)

Real incident, worked through across several rounds of live debugging in
one session — documented in full because the first two rounds chased the
wrong cause before landing on the real one.

**The symptom**: after the drop-to-latest mailbox change above landed, a
live scanning session showed RTAB-Map reporting `tracking LOST` on every
single batch, pose frozen for the entire session (`grid_updated=False` on
every `MappingUpdate`, occupancy grid never filled in) — confirmed
visually too (walking around produced zero pose change).

**Round 1 (partial, insufficient on its own): reduce `mini_batch`.**
Reasoning: with the original `mini_batch=4`, one processing cycle (DA3 for
4 frames + RTAB-Map pose for 4 frames) cost ~1-2.5s end to end; while the
server was busy inside that cycle, the drop-to-latest mailbox kept
overwriting its slot with newer arrivals, so the first chunk pulled for
the NEXT mini-batch could be 1-2.5s newer in camera motion than the batch
just finished — a gap RTAB-Map's frame-to-frame odometry has no tolerance
for (same underlying sensitivity as the fps-subsampling issue documented
under "3D Scanning Pipeline"'s "Found via real recordings, fixed"). Trying
`mini_batch=1` alone made the symptom WORSE, not better — pose stuck at the
exact literal `(0.00, 0.00)` from the very first batch, never tracking
even once, across 45 consecutive updates. This ruled out "just a timing
gap" as the sole explanation and pointed at something structurally broken
about single-frame processing specifically.

**Round 2 (wrong root cause, corrected in round 3): suspected DA3's model
needed multiple views for metric scale.** `grpc_server.py`'s
`SCAN_DA3_TORCH_MODEL_ID` env var **defaulted to `depth-anything/da3-large`**
— DA3's multi-view flagship model, NOT a monocular-metric one — and its own
docs note reference-view-selection "only applied when number of views ≥
3" (`[INFO] Selecting reference view...` had indeed stopped appearing in
the logs once `mini_batch` dropped to 1). The working theory was that a
multi-view model resolves absolute scale from cross-view geometric
consistency the same way classical multi-view SfM does, and with only one
view there's nothing to anchor scale against — RTAB-Map's RGB-D odometry
needs metrically-correct depth to register anything, so unscaled
single-view output would explain total tracking failure. **This diagnosis
was corrected by the user**: the deployment was intended to run
`DA3METRIC-LARGE` (a model the DA3 README explicitly documents as "a
specialized model fine-tuned for METRIC depth estimation in MONOCULAR
settings" — i.e., single-view IS its designed, intended regime), not
`depth-anything/da3-large`. The mismatch was real, though — nothing in
`grpc_server.py`/`da3_wrapper.py` ever set `SCAN_DA3_TORCH_MODEL_ID` to a
metric model, so **the code's actual default was silently loading the
wrong (non-metric, multi-view-only) model regardless of intent** — the
most likely real explanation for why even `mini_batch=4` sessions only
ever tracked intermittently, and why `mini_batch=1` broke it completely
(no cross-view signal left to compensate for the wrong model at all).

**Round 3 (current, the actual fix): removed `depth-anything/da3-large` as
a default entirely.** `da3_wrapper.py`'s `DA3Estimator.__init__` and
`grpc_server.py`'s `SCAN_DA3_TORCH_MODEL_ID` fallback both now default to
`depth-anything/DA3METRIC-LARGE` — matching this repo's own established
casing convention for the same checkpoint family (`DA3_ONNX_PATH`'s
`DA3METRIC-LARGE.onnx` default) and the casing shown working in DA3's own
docs/benchmark examples (`depth-anything/DA3-LARGE`,
`depth-anything/DA3NESTED-GIANT-LARGE`). With the correct monocular-metric
model now the default, `mini_batch` was reverted back to **1** — the
original request (DA3 should run on the latest incoming frame only, not a
joint multi-frame batch) — since a monocular-metric model has no
multi-view dependency to lose by processing one frame at a time, and 1 is
the smallest possible per-cycle latency (shrinking the drop-to-latest
mailbox's busy-window to a minimum, which round 1 was already reaching
for). **Not verified end-to-end against a live GPU rig from this
environment** — if RTAB-Map still fails to track at `mini_batch=1` with
the corrected model, the wrong-model theory was incomplete and something
else needs investigating.

**Round 4 (still recurring — added diagnostics, not yet a fix): confirmed
DA3Estimator never validates its own metric-scale output.** Tracing
through the vendored `depth_anything_3` package (`utils/io/
output_processor.py`) found two real issues: (1) `Prediction.depth` is
read straight from `model_output["depth"]` — `Prediction.scale_factor` is
carried through into the dataclass but NEVER multiplied into `depth`
anywhere in the package's own pipeline; (2) `Prediction.is_metric` is set
via `getattr(model_output, "is_metric", 0)` where `model_output` is a
plain `dict` — `getattr` never finds a dict key, so this field reads `0`
unconditionally regardless of the model's real output, a bug in the
vendored package itself, not real signal. Net effect: there was no actual
confirmation anywhere in this codebase that `DA3METRIC-LARGE`'s depth
output is correctly metric-scaled — `da3_wrapper.py` was trusting it
blindly. Added a one-shot diagnostic in `DA3Estimator.estimate_batch()`
(`self._logged_depth_stats`, logs once per estimator instance) printing
`min`/`median`/`max` of the raw depth tensor plus `is_metric`/
`scale_factor` — `[DA3] depth stats (one-shot, ...)`. A plausible indoor
room should read roughly 0.3-8.0 in whatever unit this turns out to be;
if the logged numbers are wildly off that range (or `scale_factor` is a
real non-`None`/non-1.0 value), that's the confirmation needed to apply an
explicit correction — not yet done, this is diagnostics-only pending that
log output. RTAB-Map's own `track_batch()` already
sends one TRACK request per frame internally regardless of Python-level
batch size (confirmed by reading `rtabmap_client.py`), so nothing about
the pose RPC path itself needed to change across any round. `scan_gui.py`'s
two `StreamingScanSession(...)` call sites both pass
`mini_batch=int(batch_size)` explicitly from their own GUI slider, so the
offline tool's behavior/default was unaffected by any of this — only the
live path (which relies on the constructor default) picked up the changes.

### Walking/guiding skip the novelty gate entirely; per-mode blur threshold (`scan_session.py`, `mapping_servicer.py`)

Confirmed with the user: "is this view novel compared to earlier ones" is
a scan-only question — walking/guiding (`walking_lite`) never discover
landmarks or persist geometry the way a scan pass does (VLM tagging and
RTAB-Map's `get_cloud()`/SOR pull are already skipped entirely for
`walking_lite`, see "Session-mode pipeline split" above), so there's no
frame store/reconstruction for a novelty check to protect in the first
place. `process_frames_batch()`'s RTAB-Map branch now branches three ways
per frame instead of two: `t.pose is None` (tracking lost, always
rejected, regardless of mode) → `walking_lite` (blur-only: `accepted =
sharpness >= self._min_sharpness`, no novelty/rotation check at all) →
else/SCAN (unchanged full `decide_accept()` novelty+blur gate). Blur is
still worth gating in walking/guiding — a badly-blurred frame still
corrupts the occupancy grid it feeds — just at a more lenient threshold
than scan mode needs for clean reconstruction/tagging: `mapping_servicer.py`
at the time had `SCAN_MIN_SHARPNESS = 80.0` / `WALKING_MIN_SHARPNESS =
50.0` (was one shared `MIN_SHARPNESS`), and `configure_novelty_gate(
min_sharpness=...)` picked between them from the already-known
`walking_lite` flag at stream open. **Superseded**: both constants and the
`configure_novelty_gate()` call were removed in "Blur filtering removed
for scan/walking" above — blur gating is fully off for the live path
again, walking/guiding's condition reduces to just "did RTAB-Map produce a
pose." The `reject_reason` splitting described below is unaffected and
still current. Also fixed in the same pass: the `[novelty-blur] frame REJECTED`
log line used to say "not novel enough or too blurry" unconditionally,
even though `decide_accept()`'s short-circuiting boolean algebra means at
most one of those is ever actually why a given frame failed — genuinely
confusing when a frame with `sharpness=95 > min_sharpness=80` (clearly not
blurry) still got logged as possibly-blurry. Both `_evaluate_novelty()`
(IMU+VO/VO) and the RTAB-Map branch now compute and return a specific
`reject_reason` string (novel-enough / rotation / blur / tracking-lost, as
appropriate) alongside `(accepted, sharpness)`, threaded through
`novelty_flags` and used directly in the log line instead of the old
generic phrasing.

**New Step 0 in `ScanSession.process_frames_batch()` — a blur pre-check
that runs BEFORE Step 1 (DA3), not after.** The existing novelty+blur gate
(Step 3, `orb_novelty_gate.py`'s `decide_accept()`) only ever vetoed
FUSION — by the time it rejects a frame, DA3 depth estimation and RTAB-Map/
VO pose computation for that frame have already run and been paid for in
full, regardless of the outcome. Real logs from a live session showed this
happening on nearly every batch: 600-1200ms of DA3 + 300-1200ms of RTAB-Map
pose work, immediately followed by every frame in the batch being rejected
by the novelty/blur gate — pure waste. Step 0 computes each frame's
`_sharpness_score()` up front (cheap, CPU-only, no GPU/model involved) and
drops any frame below `self._min_sharpness` BEFORE `frames_rgb` (and the
correspondingly-trimmed `imu_poses`/`frame_timestamps_ns`) ever reaches
Step 1 — if NOTHING in the batch survives, Step 1 onward (DA3, RTAB-Map,
back-projection, occupancy update) is skipped entirely for that cycle, and
`process_frames_batch()` returns immediately with the unchanged point
count and the last known camera position, at `infer_ms=0.0`. Step 3's
RTAB-Map branch reuses Step 0's precomputed sharpness (via the new
`frame_sharpness` list, index-aligned with the trimmed `frames_rgb`)
instead of recomputing it a second time. Same "0 disables entirely"
convention `DEFAULT_MIN_SHARPNESS = 0.0` already established — a session
that never configures a real `min_sharpness` sees byte-identical behavior
to before this pre-check existed, so `scan_gui.py`'s own default is
unaffected.

At the time, `mapping_servicer.py` briefly turned Step 0's gate ON for the
live path with a single `MIN_SHARPNESS = 80.0` constant, chosen from real
observed sharpness scores in that same live session (clearly-blurred
in-motion frames measured 11-100, clean frames 500-990 — 80 sat just above
the blurred cluster without touching the clean one). This was later split
into `SCAN_MIN_SHARPNESS`/`WALKING_MIN_SHARPNESS` per mode (see "Walking/
guiding skip the novelty gate entirely" below), then removed entirely per
"Blur filtering removed for scan/walking" above, which is current — blur
gating is off for the live path again.

### Android implementation (`client/android/app/src/main/java/com/tracking/client/live/`)

New package, one Gemini Live session per connection, replacing the old
`VoiceChatStream`-relayed flow entirely for this client:

- `GeminiLiveClient.kt` — raw WebSocket client for the Gemini Live
  `BidiGenerateContent` protocol (OkHttp, added as an explicit dependency —
  `grpc-okhttp` shades its own OkHttp internally and doesn't expose
  `okhttp3.*` on the compile classpath). Deliberately not Firebase AI
  Logic's `LiveModel`/`LiveSession` (the official wrapper for this same
  protocol) — that requires a Firebase project + `google-services.json`,
  heavier than "embed an API key and call Gemini directly" (the chosen
  tradeoff). **Built from the documented wire schema
  (https://ai.google.dev/api/live), not verified against a live backend
  from this environment** — the first thing to check if a real device fails
  to connect. `sendSystemNote()` sends a `clientContent` turn with
  `turnComplete=true`, not `realtimeInput.text` — found and fixed during
  this work: `realtimeInput` has no explicit turn-completion field (turn
  end is inferred from audio VAD / video activity), and this client is
  push-to-talk with no audio or video flowing into the Live session between
  PTT presses, so an injected note had nothing to ever trigger a
  turn-completion and just sat buffered — present in context, but never
  spoken — until the next real spoken turn happened to close it. The old
  server-side session (`server/live_session.py`, deleted) used the same
  `realtime_input(text=...)` call but got away with it because it also kept
  a continuous frame-tick video stream flowing into the same session
  (walking/reading ticks), which kept turns cycling closed on their own.
  `clientContent.turnComplete=true` forces immediate generation regardless
  of VAD/activity state. Known, unguarded edge case: if a note fires while
  PTT is actively held, this could force-complete the user's in-progress
  spoken turn early rather than queuing behind it — accepted since these
  are safety-relevant async ticks (obstacle/waypoint alerts), not everyday
  chat turns.
- `ToolDeclarations.kt` — Kotlin port of `tool_declarations.py`'s
  `SYSTEM_PROMPT` + `TOOL_DECLARATIONS`, sent directly in the Live session's
  setup message now. `start_guiding`'s destination is documented as a
  landmark name, not a zone label (zones are gone). `start_scan()`/
  `stop_scan()` (new) trigger the same mapping pipeline without a
  destination — see "Novelty+blur frame gating and deferred landmark
  resolution" in the 3D Scanning Pipeline section.
- `LiveSessionState.kt` — Kotlin port of `LiveSessionState` (mode, reading
  blocks, nav waypoints, walking obstacle cache), held in `MainViewModel` now
  instead of server-side. `readingBuffer: String` is now `readingBlocks:
  MutableList<String>` (+ `readingBufferText()` to join them) — see
  "Reading-mode OCR" below for why a flat string's dedup broke down.
- `ToolDispatcher.kt` — Kotlin port of `_dispatch_tool` + the `live_tools/*.py`
  implementations. One implementation per tool, routed by weight:
  - **Remote** (`grpc.perceptionStub`/`mappingStub`, new fields on
    `GrpcClientManager`): `run_detection`/`check_obstacle` →
    `AnalyzeFrame`; `read_aloud` → `Synthesize`; `query_memory`/
    `save_memory`/etc.'s vector step → `Embed`; `start_guiding`/
    `start_walking`/`start_scan` → `UpdateMapping` bidi stream (fed by
    `feedMappingFrame()`, called from `MainViewModel`'s existing camera-frame
    collector whenever `mode` is `guiding`/`walking`/`scanning`).
    `recomputeRoute()` (fixed — was a no-op placeholder) resolved
    `guidingDestinationLabel` via `FindLandmark` on every grid update while
    unresolved, then routed with `LocalPathPlanner` below — **this
    paragraph describes the original migration; path planning has since
    moved server-side and `LocalPathPlanner.kt` is deleted, see
    "Server-planned walking path + client-side latency bridging"**.
  - **3rd-party direct**: `OcrClient.kt` — multipart POST straight to
    OCR.space, bypassing the gRPC server as a proxy (and the earlier
    self-hosted `paddle_ocr_server`, deleted outright — see "Reading-mode
    OCR — OCR.space + block-level dedup + line-level filters" below).
  - **Local**: `LocalMemoryStore.kt` (on-device JSON files under
    `filesDir/memory/` — labels/notes + a small embedding index, cosine
    search in Kotlin; the `Embed` RPC is the only remote step),
    `LocalPathPlanner.kt` (deleted — see the note above; was a full Kotlin
    port of `live_path_planner.py`'s A* run against the streamed grid),
    `HrtfBeacon.kt`
    (`directionTo()` — egocentric azimuth/elevation of the next waypoint
    from the current `Pose`, quaternion-rotation math, camera==head pose
    since the camera is glasses-mounted — see the earlier HRTF
    beacon-placement design discussion; `directionFromBox()` — a second,
    much simpler entry point for tracking mode, which has no 3D pose/depth
    at all, only a 2D ORB box: approximates azimuth/elevation from the box
    center's pixel offset via the same pinhole-FOV guess
    `server/tools/depth.py`'s `_estimate_K` uses server-side
    (fx=fy=0.8*max(w,h)), with a fixed nominal `distanceM` since 2D-only
    tracking has nothing to compute a real distance from), `audio/
    HrtfBeaconPlayer.kt` (the actual sound `HrtfBeacon.kt`'s numbers drive:
    loops `assets/fluttering.mp3` continuously during guiding/walking
    (`directionTo`, muted when no waypoint exists yet) AND during tracking
    mode (`directionFromBox`, muted when the target isn't currently
    visible) — `ToolDispatcher.updateTrackingBeacon()` is the tracking-mode
    call site, invoked from `MainViewModel`'s local-ORB-tracking update loop
    on every frame the target is visible in)
    driven by `audio/HrtfConvolver.kt` (REAL HRTF: direct time-domain
    FIR convolution of the mono loop against the pair of ear filters
    nearest the current azimuth/elevation, picked via nearest-neighbor over
    precomputed unit vectors — not a parametric ITD/pan approximation).
    Filters come from `assets/hrtf_kemar.bin`, a compact binary conversion
    (`710 positions × 2 ears × 512 taps @ 44.1kHz`, int16 fixed-point,
    ~1.4MB) of the public MIT KEMAR HRIR set (SOFA `SimpleFreeFieldHRIR`
    convention) — converted once from `/usr/share/libmysofa/
    MIT_KEMAR_normal_pinna.sofa` (shipped by the `libmysofa1` apt package,
    used by PipeWire's spatial-audio tooling) via a one-off `h5py` script,
    not part of the app's runtime pipeline. Azimuth sign convention was
    verified against the raw SOFA data before wiring it in: our `+90°`
    (HrtfBeacon's "positive = right") must select a SOFA position where the
    RIGHT ear's impulse response has the larger energy — confirmed
    numerically (right-ear energy ~15x left at that position) before
    trusting the `sofaAzimuth = -ourAzimuth` conversion in
    `HrtfConvolver.nearestIndex()`. Deliberately NOT Android's `Spatializer`
    API (API 32+, and it renders straight to THIS device's own output —
    no way to capture the result and forward it elsewhere): audio output is
    moving to a separate low-power edge device (`edge/EdgeDevice.kt`) that
    should do as little DSP as possible, so ALL rendering (convolution, or
    the plain equal-power-pan fallback used only if `hrtf_kemar.bin` fails
    to load) happens phone-side; `onChunk` exposes each already-rendered
    PCM16 chunk for that future device to just stream to earbuds, no
    compute of its own. Filter changes are crossfaded across one ~46ms
    chunk to avoid an audible click when the nearest HRIR direction
    changes. `MemoryTextUtils` in `LocalMemoryStore.kt` ports
    `memory_store.py`'s sentence-overlap filtering exactly — still used for
    `save_memory`'s overlap-trim, but no longer for the reading buffer
    itself; see "Reading-mode OCR — OCR.space + block-level dedup +
    line-level filters" below for why that moved to a different algorithm
    (`OcrBlockFilters.integrateBlock()`).
  - **Device** (unchanged): routes straight to the existing
    `DeviceToolHandler`/`AndroidDeviceToolHandler` — in-process now, no
    `DeviceToolCall`/`DeviceToolResult` wire round-trip needed since Gemini
    Live runs on the same device that executes them.
- **Known gap**: `search_youtube`/`get_video_info` are declared (so Gemini
  knows they exist) but `ToolDispatcher` returns a clear "not yet available"
  error — the old server-side `music_tools.py` (yt-dlp-based) was never
  ported to a callable RPC in this pass. `play_video`/`stop_music` still
  work (routed to the device handler, which already expects a
  pre-resolved stream URL).
- `MainViewModel.kt`: `connect()` gained `geminiApiKey`/`ocrApiKey`/
  `locationId`/`blurSharpnessThreshold`/`saveDebugOcrFrames` params (new
  `SettingsScreen.kt` fields, persisted via `SettingsViewModel` — see
  "Reading-mode OCR" below for the latter two); `doLiveSession()` rewritten around
  `GeminiLiveClient.events()` + `ToolDispatcher.dispatch()` in place of the
  old `stub.voiceChatStream(...)` call. The old IMU-frame relay to the
  server was dropped entirely (Gemini Live never used it — "IMU frames
  ignored in Live path" — and the new RTAB-Map-only mapping path doesn't
  need client-sent IMU either).
- `ScanViewModel.kt`/`ScanScreen.kt`/`ScanUiState.kt` and the Scan nav
  route/button are **deleted** — the offline record-and-upload workflow no
  longer exists on Android (mapping is live now, see above).
- Verified via `./gradlew :app:compileDebugKotlin` (full success, including
  the new proto-generated `PerceptionServiceGrpcKt`/`MappingServiceGrpcKt`
  stubs) — a full `:app:assembleDebug` was NOT verified in this environment
  (fails on an unrelated, pre-existing toolchain issue: `jlink` missing from
  the installed JDK, needed for `compileDebugJavaWithJavac`'s JDK-image
  transform — not caused by these changes). Kotlin compilation is real
  type-checking across the whole new + modified codebase, but is not a
  substitute for an on-device run.

### Reading-mode OCR — OCR.space + block-level dedup + line-level filters

Reading mode's OCR backend was switched from the self-hosted
`paddle_ocr_server` microservice (deleted outright, not deprecated — its
whole directory is gone) to OCR.space, a free hosted OCR API, per a direct
user decision (accepting the tradeoff: an external network dependency and
rate-limited free tier, vs. no local GPU/PaddlePaddle service to run).
Developed and tuned first in a standalone Gradio test harness
(`~/Desktop/gt.py`, outside this repo) against real, noisy phone-camera OCR
output of a physical book, before being ported here — that harness's own
comments carry the full "why" for each threshold below; this section is the
condensed version.

**The core problem this redesign solves**: `MemoryTextUtils.filterNewSentences()`
(still used for `save_memory`'s overlap-trim) assumes near-identical OCR
text on a repeat read — a sentence is only recognized as "already read" via
EXACT substring containment. Real OCR noise varies pass to pass (typos,
misread watermark text, a badly-truncated reread of the same page), so that
check rarely fires and the reading buffer fills with near-duplicate
paragraphs. Confirmed directly against real output: the same paragraph
reread 3-5 times with 1-2 word OCR differences each time ("shed"/"she'd",
"Morn"/"Mom") all got appended in full.

**Fix — block-level fuzzy dedup (`OcrBlockFilters.integrateBlock()`,
`TextBlockFilters.kt`) — superseded, see "Reading-mode OCR correction +
fuzzy stitching" below for the current `integrateRawBlock()`/`ReadingBlock`
design; kept here for the original dedup rationale, which is still
current**: `LiveSessionState.readingBuffer: String` became
`readingBlocks: MutableList<String>` (+ `readingBufferText()` to join them
for callers that just want the accumulated text). Each OCR pass is folded
into `readingBlocks` as one whole BLOCK, matched against every block
already stored via `blockSimilarity()` — an ASYMMETRIC, WORD-LEVEL
containment score: "how much of the SHORTER text's content shows up as
runs of >=2 consecutive matching words somewhere in the longer one",
computed via a Kotlin port of Python difflib's Ratcliff/Obershelp matching-
blocks algorithm (`findLongestMatch()`/order-preserving, not a plain
bag-of-words/bigram overlap — that would let common short phrases match
across unrelated positions). Two cheaper approaches were tried and
discarded first: a plain symmetric character-level ratio scores a short,
garbled reread of an already-stored page far too low purely from the
length mismatch, letting it slip past as a spurious "new" block;
character-level containment (no word tokenization) fixed that but then
scored UNRELATED paragraphs too high, since arbitrary short substrings
coincidentally recur throughout any English text. Requiring >=2-consecutive-
WORD matching runs needs a genuine shared phrase, cleanly separating true
rereads (~0.75-0.9 in practice) from unrelated paragraphs (~0.05-0.1) — a
`DUP_CONTAINMENT = 0.45` threshold sits comfortably in that gap. A
duplicate doesn't get re-appended; if the new capture is longer (cleaner/
more complete), it replaces the stored block outright. A middle "partial
merge" band (stitching in just the sentences that don't already appear)
was also tried and reverted: at real-world OCR noise levels almost no
sentence matches exactly, so it re-appended nearly the whole paragraph
anyway — duplicates are a single yes/no call now.

**Line-level filters, chained in `OcrClient.analyze()` before dedup ever
runs** — OCR.space's overlay (`isOverlayRequired=true`) gives per-word
axis-aligned boxes (`OcrLine`/`OcrWord`, `TextBlockFilters.kt`), letting
each detected line be filtered on its own merits:

1. **Rotation-consistency** (`filterLinesByRotation()`) — OCR.space's own
   `detectOrientation` only corrects whole-PAGE rotation (0/90/180/270);
   it has no concept of one box being crooked relative to the rest of an
   already-correctly-oriented page. Each line's orientation is instead
   ESTIMATED from the slope between its first and last word centers (needs
   >=2 words; single-word lines have no evidence and are kept by default).
   A line whose angle deviates more than `ROTATION_MAX_DEVIATION_DEG` (15°)
   from the page's dominant (median) angle is dropped — catches a stray
   diagonal watermark stamp the body text doesn't share.
2. **Short/small/isolated noise** (`filterLinesByNoise()`) — catches page
   numbers, watermark stamps, and garbled debris that survive rotation
   filtering untouched (a single-word line has no slope to measure at
   all). A line is dropped only when it's BOTH ISOLATED (edge-to-edge gap
   to the nearest other line is large relative to the page's own median
   line height, `ISOLATION_GAP_FACTOR = 1.8`) AND either short
   (`MIN_TEXT_CHARS = 8`) or undersized (`SMALL_HEIGHT_FRAC = 0.72`).
   Isolation is required, not optional — it's what tells a genuine short
   in-paragraph word/exclamation (sitting right next to its paragraph,
   normal font size) apart from a page number floating alone in the
   margin; length or font size alone would risk dropping real text.
3. **Per-line blur** (`filterLinesByBlur()`) — a real gap found from a
   live screenshot: a page mid-turn can be motion-blurred on one side and
   still sharp on the other, but `CameraManager`'s existing blur check
   (see below) measures sharpness for the WHOLE frame as one aggregate
   number, which the sharp side's pixels pull well above any reasonable
   threshold — the frame passes the whole-frame check easily while still
   containing an individually unreadable region. This crops each line's
   OWN bounding box out of the frame and measures variance-of-Laplacian
   sharpness (same metric/formula as `CameraManager.computeSharpness()` —
   `meanStdDev`, variance = stddev²) on just that patch, independent of
   the rest of the frame. `MIN_LINE_SHARPNESS = 60.0`; 0 disables.

Each dropped line survives into the optional debug frame overlay (below)
with a distinct color, so a real run's filtering decisions are visible,
not just inferred from the final text.

**Whole-frame blur skip/retry (`ToolDispatcher.acquireSharpFrame()`,
distinct from the per-line filter above — this one decides whether to spend
an OCR CALL at all, before any line-level data exists to filter)**: before
`toolScanCurrentView()` calls OCR.space, it checks the current frame's
overall sharpness via `CameraManager.clearestRecentFrameWithSharpness()`
(new — same pull as the existing `clearestRecentFrame()`, also exposing the
score). Below `blurSharpnessThreshold` (SettingsScreen-configurable,
default 40, 0 disables), it re-samples up to `BLUR_MAX_RETRIES` (2) times,
waiting `BLUR_RETRY_WAIT_MS` (500ms) between each — standing in for "let
the live camera produce a fresher frame" — before giving up on the cycle
entirely (no OCR call made) rather than spending one on known-bad input.

**Debug frame storage (`DebugFrameStore.kt`, off by default)**: writing
every scanned camera frame to device storage has real storage-growth and
privacy cost in a production assistive app, so this is gated behind
`SettingsScreen`'s "Save debug OCR frames" toggle (`saveDebugOcrFrames`,
persisted via `SettingsViewModel`) — off, it's a no-op (`ToolDispatcher`'s
`saveDebugFrame` callback stays `null`, wired from `MainViewModel` only
when the toggle is on). When on, `annotateOcrFrame()` draws each line's box
on a copy of the frame (green = kept, red = dropped for rotation, orange =
dropped as noise, magenta = dropped as locally blurry) and
`MainViewModel` writes it to `filesDir/debug_ocr_frames/` on `Dispatchers.IO`
(fire-and-forget, same precedent as `reportMode()`'s dropped-report
tolerance — a failed debug save is never something the read-aloud flow
itself should fail over).

### Reading-mode OCR correction + fuzzy stitching (raw/corrected split)

Ported from `~/Desktop/gt.py`'s own iteration on the reading-mode dedup
above (same standalone Gradio harness cited at the top of the previous
section) — developed and tuned there against real noisy phone-camera OCR
output before being carried into the production Android client. Two real
gaps in the original block-level dedup drove this:

1. **No OCR-error correction at all.** Raw OCR text (typos, garbled words,
   broken punctuation) was stored and spoken verbatim — the harness added
   an LLM correction pass; this ports that pass into the app.
2. **Overlapping-but-different reads either got wrongly merged or silently
   lost content.** A sliding scan (camera panning across a page) produces
   captures like "A B C" then "B C D E" — the old `integrateBlock()` only
   ever replaced-if-longer or left alone; it never appended the genuinely
   new tail. Worse, when the old whole-block "same page, cleaner capture"
   replace path fired but the new capture's own OCR pass happened to start
   partway through the existing text (not from its beginning), the
   original opening content was silently discarded forever — a real,
   observed, data-losing bug (see `gt.py`'s own commit history for the
   exact reproduction) that this design fixes on both sides at once.

**`LiveSessionState.readingBlocks` is now `MutableList<OcrBlockFilters.
ReadingBlock>`** (`TextBlockFilters.kt`) — `ReadingBlock(id, raw, corrected)`
replaces the plain `MutableList<String>` `integrateBlock()` used. `raw` is
the OCR output, stored immediately; `corrected` starts as a mirror of `raw`
and is only later, asynchronously, patched in place once a Gemini
correction response lands for that specific block — `readingBufferText()`
joins `.corrected` (== raw for any block whose correction hasn't landed
yet), so reading/dedup/`get_reading_section` never wait on the LLM call.

**`OcrBlockFilters.integrateRawBlock(blocks, newRawText)`** (replaces
`integrateBlock()`, deleted outright — no remaining caller) returns
`(kind, block)`:
- **`"stitched"`** — `newRawText` fuzzily CONTINUES the MOST RECENT block
  (`findFuzzyOverlapMerge()`) — that block's raw is extended in place with
  only the genuinely-new tail. Only ever checked against `blocks.last()`:
  stitching against an arbitrary earlier block would conflate "this
  continues from where I just was" with "this is a reread of something
  from a while ago" (a different, position-agnostic question the existing
  `blockSimilarity()` duplicate/updated check below already answers).
  Two-pass: (1) a strict exact-word `matchingBlocks()` anchor requiring the
  match to reach near the END of the existing block's words AND the START
  of the new capture's words (`boundarySlack`, 2 words); falling back to
  (2) `findWindowedFuzzyOverlap()` — OCR quality is typically WORST right
  at a frame's edge (motion blur, a partially-cut-off word), exactly where
  the strict anchor needs to be clean, so pass 2 slides a window comparing
  trailing/leading word pairs via character-ratio fuzzy equality
  (`wordFuzzyEqual`, Ratcliff/Obershelp `charMatchRatio >= 0.6`, reusing
  the same `matchingBlocks()` machinery at the character level) instead of
  requiring exact string equality — tolerates typos IN the anchor words
  themselves, not just gaps around them. Verified against both a
  clean-boundary case and a typo'd-boundary case (`"foxx jurnps"` still
  continuing `"fox jumps"`), plus stress-tested against 4 genuinely
  unrelated text pairs sharing only common filler words at the seam with
  zero false-positive stitches.
- **`"updated"`** — matched an existing block via `blockSimilarity()`
  (whole-text containment, position-agnostic, unchanged from the original
  dedup) and this capture is longer/cleaner. Previously a blind overwrite;
  now goes through **`findMidtextRealignMerge()`** first — finds where in
  the EXISTING block's own text the best word-level match against the new
  capture begins (unrestricted position, unlike the boundary-anchored
  stitch search above), and returns the existing block's own PREFIX up to
  that point + the new capture in full, instead of discarding whatever
  existing content precedes the overlap. Falls back to a plain overwrite
  only if no anchor is found at all. This is the fix for the data-loss bug
  described above — verified via the exact real-world capture pair that
  originally exposed it (a two-sentence opening that the old code
  silently dropped is now preserved, confirmed word-for-word present in
  the merged result).
- **`"duplicate"`**/**`"new"`**/**`"empty"`** — unchanged in spirit from
  the original `integrateBlock()`.
- Every outcome is logged (`Log.d("OcrBlockFilters", ...)`, full text, not
  truncated) — ported directly from a real debugging need in `gt.py`: an
  earlier version there silently swallowed correction failures and only
  showed a truncated preview, which made "did it actually append or
  replace" impossible to answer from the log alone.

**`GeminiCorrectionClient.kt`** (new) — plain REST call to Gemini's
`generateContent` endpoint (`gemini-3.1-flash-lite`), deliberately NOT the
`BidiGenerateContent` WebSocket protocol `GeminiLiveClient.kt` uses (this
is one-shot request/response, no session/turn state) and deliberately NOT
a Google SDK (same reasoning `GeminiLiveClient.kt`'s own comment gives for
avoiding Firebase AI Logic — a hand-rolled OkHttp + `org.json` call needs
no extra project setup, consistent with this module's existing style).
Reuses the same `geminiApiKey` already entered for Gemini Live — no
separate Settings field; a blank key means `MainViewModel` passes `null`
for `ToolDispatcher`'s `geminiCorrectionClient`, which disables correction
entirely (blocks behave exactly as before: `corrected` permanently mirrors
`raw`). Two system prompts, selected by detected language (fix-only for
English, fix+translate-to-English otherwise) — ported verbatim from
`gt.py`'s own prompts. Language detection is **ML Kit's on-device Language
Identification** (`com.google.mlkit:language-id`, new Gradle dependency) —
chosen over trying to port/bundle `gt.py`'s fastText model, since ML Kit is
a standard Android library needing no model-management code here (lazily
downloads its own small model on first use). Deliberately does NOT swallow
failures internally — throws on any error, logging the real exception
first (same "never silently degrade into looking like nothing needed
correcting" lesson `gt.py` learned the hard way once already).

**`ToolDispatcher`'s correction queue** — Kotlin port of `gt.py`'s
`ReadingPipeline` correction-queue/worker split: OCR results are stored
(and, for `"new"`/`"stitched"`, spoken) IMMEDIATELY in
`toolScanCurrentView()`; correction is pushed onto an unlimited
`Channel<CorrectionJob>` and runs on ONE background coroutine
(`correctionWorkerJob`, started in `init {}` only when
`geminiCorrectionClient` is non-null) that drains it strictly
SEQUENTIALLY — one Gemini call finishes before the next starts, same
contract `gt.py`'s worker thread had — entirely decoupled from the
tool-call path, so a slow or failed correction never blocks scanning or
reading. `"new"`, `"stitched"`, AND `"updated"` all get queued for
correction (matching `gt.py`'s push condition exactly) even though only
`"new"`/`"stitched"` are spoken as new text this cycle — an `"updated"`
block (a longer/cleaner reread) still needs its own fresh correction. Each
job carries `(blockId, rawText, langCode)`; `OcrBlockFilters.
applyCorrection()` on the worker side only patches `corrected` if that
block's `raw` still matches `rawTextAtRequest` — a block whose raw was
superseded by an even newer capture while its correction was in flight
gets its stale result silently discarded (the newer capture already
queued its own fresh correction). `shutdown()` cancels the worker job and
closes the channel alongside every other mode's cleanup.

**`start_live_reading()`/`stop_live_reading()` (new tools) — continuous
capture, superseding the note below about `gt.py`'s capture-pacing model
NOT being ported (it now is)**: mirrors `gt.py`'s Live Reading tab's
3-stage pipeline exactly, adapted to Kotlin coroutines —
`ToolDispatcher.toolStartLiveReading()` starts two background jobs instead
of `enter_reading_mode()`'s one-shot state reset:
- **Capture** (`liveReadingCaptureJob`) — every `LIVE_READING_INTERVAL_MS`
  (5s, not yet Settings-exposed — a reasonable follow-up), calls the SAME
  `acquireSharpFrame()` blur skip/retry `scan_current_view()` uses, and
  sends the result to `liveReadingOcrChannel` (`Channel<ByteArray>`,
  unlimited). A still-blurry frame after `acquireSharpFrame()`'s own
  retries waits `LIVE_READING_BLUR_GIVEUP_MS` (4s) before the next attempt
  instead of the normal interval — same `BLUR_GIVEUP_WAIT_S` precedent
  `gt.py` established. Runs on its own fixed real-time cadence, never
  blocked by how long OCR/correction take — same "play it out like
  realtime reading" simplification `gt.py` settled on (one interval
  control, not a separate capture-interval/video-advance split).
- **OCR worker** (`liveReadingOcrJob`) — drains the channel ONE AT A TIME
  (never starts a new OCR call until the previous one's response is back),
  calls `OcrClient.analyze()` then `OcrBlockFilters.integrateRawBlock()`.
  `"new"`/`"stitched"` are spoken IMMEDIATELY via a new shared
  `speakReadingText()` helper (factored out of `toolReadAloud()`, reused
  by both) — RAW text, since correction hasn't run yet, same as the
  one-shot `scan_current_view()` path. `"new"`/`"stitched"`/`"updated"`
  all get queued onto the SAME `correctionChannel`/`correctionWorkerJob`
  the one-shot path already uses — one shared correction queue regardless
  of which reading tool produced the block.

`toolStopLiveReading()` cancels both jobs and closes the channel but
leaves `state.readingBlocks` intact — `read_aloud(scope="all")`/
`get_reading_section()`/`save_reading_buffer()` still work afterward,
mirroring `gt.py`'s separate Start/Stop/Clear-Buffer semantics
(`stopActiveModes()`/`toolExitReadingMode()`/`shutdown()` all now also
call `stopLiveReadingPipeline()`, so switching to any other mode, exiting
reading mode entirely, or tearing down the whole session correctly halts
it too).

**`save_reading_buffer(label)` (new tool)** — pulls `state.
readingBufferText()` directly rather than relying on `save_memory(label,
note)` with Gemini retyping the scanned text into `note` itself (the
original save-memory path is a general-purpose tool with no idea reading
mode's buffer even exists) — avoids Gemini having to faithfully reproduce
a whole scanned document into a function-call argument, which risks
truncation/paraphrasing for anything beyond a short note. Works
identically regardless of whether the buffer was filled by one-shot
`scan_current_view()` calls or a live-reading session, since both write
into the same `state.readingBlocks`. Saved through the existing
`LocalMemoryStore.append()` + `embedAndStore()` path (same as
`save_memory()`), so it's queryable later via the existing
`query_memory()` semantic search — no new retrieval path needed.

**Known, accepted limitations** (same standing caveat as the rest of this
feature's development): not verified end-to-end on a real device — compile-
verified only (`./gradlew :app:compileDebugKotlin`, clean build, zero new
warnings). `LIVE_READING_INTERVAL_MS` is a hardcoded constant, not yet a
Settings field (unlike `frameIntervalMs`/`scanIntervalMs`/
`avoidanceIntervalMs`, which are) — a reasonable follow-up if 5s needs
per-user tuning. `gt.py`'s block-by-block debug viewer (raw vs. corrected
side by side, Prev/Next buttons) was NOT ported — that's specific to that
standalone test harness's Gradio UI; the production app speaks results
instead of displaying them, so there's no equivalent need.

### Server-side scan_server.py — deleted, restored, then reduced to a plain Gradio launcher

`scan_server/scan_server.py` (originally: a FastAPI `/api/upload`
entrypoint + `mount_gradio_app` launcher) was deleted once
`ScanViewModel.kt` (its only HTTP client) was removed, then restored on
user request as `scan_gui.py`'s launcher — `scan_gui.py`'s rich Live
Reconstruction/Occupancy Map/Voxelization/Live Navigation Preview panels
were left in place through the deletion (meant to eventually be merged
into `server_gui.py`'s Mapping tab, which still only shows a much thinner
last-frame/pose/grid view) but had no way to run without this file.

**Current state: plain Gradio launch, no HTTP API at all.** Video
ingestion (originally added to this file as a `POST /api/upload` endpoint
accepting a video file, decoded server-side via OpenCV) has been moved
**into the GUI itself** — `scan_gui.py`'s "Load from Video" accordion
(was "Load from Android Upload") now has its own `gr.Video` upload widget
+ "Extract Frames" button, calling `_extract_video()` directly in-process.
With that gone, `scan_server.py` had nothing left to serve over HTTP, so
the FastAPI app / `mount_gradio_app()` / `uvicorn.run()` wrapper this file
used to also carry (both when originally restored and during the
video-upload experiment) was removed outright — `_build_ui()` returns the
`gr.Blocks` app directly and `__main__` just calls `demo.launch(server_name=
"0.0.0.0", server_port=GRADIO_PORT)`. `uploads/` on disk is unchanged in
meaning (still where extracted `dataset/` folders land, still what
`create_scan_ui(..., upload_dir=...)` browses for the "Past Uploads"
dropdown) — only the thing that populates it changed, from an HTTP POST to
a GUI button.

One other deliberate deviation from the original file, carried through
every version since restoration: the `--da3-model torch` path's
`SCAN_DA3_TORCH_MODEL_ID` fallback is `depth-anything/DA3METRIC-LARGE`,
not the original `depth-anything/da3-large` — the latter is a multi-view,
non-metric checkpoint that caused a real RTAB-Map total-tracking-failure
incident (see "DA3 model default + per-frame processing + pre-DA3 blur
gate" below); `grpc_server.py` already carries this same fix, and leaving
this file's own copy un-fixed would silently reintroduce that bug for
anyone launching offline scans via `--da3-model torch` with no
`SCAN_DA3_TORCH_MODEL_ID` set. Run: `cd scan_server && python
scan_server.py` (port 7861, `SCAN_GRADIO_PORT` to override; `--da3-model
onnx` for the lighter DA3-METRIC ONNX path instead of torch).

**`scan_gui.py`'s `_extract_video()`** (the ported video-ingestion logic,
now GUI-side): decodes an uploaded video via `cv2.VideoCapture` into
`uploads/<scan_id>/dataset/`: `images/000000000.jpg, ...` +
`camera.csv` (`timestamp_ns,filename`) — the same layout
`stream_simulator.py` already expects, so nothing downstream needed to
change. Every decodable frame is extracted (no subsampling at extraction
time); `timestamp_ns` is derived from the video's own reported fps
(`frame_idx / fps`, falling back to 30fps if the container reports
0/NaN), so the *existing* replay-time fps slider
(`stream_simulator.build_event_timeline`) subsamples an ingested video
exactly like it already subsamples an Android-recorded dataset — no
separate code path needed. **No `imu.csv` is produced** — a video
container carries no IMU stream, and this isn't a special case downstream:
`ScanSession.set_imu_file()` is simply never called for it, and IMU + VO
pose mode already falls back to VO-only rotation when no `imu.csv` is
present (the same fallback an Android recording with a missing/corrupt
`imu.csv` would already hit); RTAB-Map pose mode needs no IMU either way.
On success, the extracted `dataset/` path is written straight into
`dataset_path_input` (same output slot `_load_upload()`/"Load Selected"
already targets), so the existing `dataset_path_input.change()` wiring
picks it up and loads the gallery/segment table automatically — no new
propagation path needed. `_upload_dir` also now defaults to a local
`scan_server/uploads/` folder when `create_scan_ui()` is called without an
explicit `upload_dir` (previously this made the "Past Uploads"
dropdown/video-extraction target `None`-guarded and silently inert without
one), so the GUI is self-sufficient even outside `scan_server.py`'s own
wiring.

### Foreground-service migration + calls/SMS/YouTube (`LiveAssistantService`)

The whole point of this round of work: the Gemini Live session (gRPC, camera
frames, mic, tool dispatch) used to live entirely inside `MainViewModel`'s
`viewModelScope` — Activity/ViewModelStore-scoped, so it died on screen-off/
task-swipe. That made call-answering/SMS-reacting while the phone is in a
pocket impossible. Also added: voice-callable YouTube search/playback.

**`live/LiveAssistantService.kt`** (new) — a `LifecycleService`, foreground +
bound, now owns everything `MainViewModel` used to own directly:
`GrpcClientManager`, `CameraManager`, `PushToTalkRecorder`,
`StreamingAudioPlayer`, `HrtfBeaconPlayer`, `TrackingBackend`, `HandTracker`,
`RotationTracker`, `PdrStepEstimator`, `AndroidDeviceToolHandler`,
`LocalMemoryStore`, `LiveSessionState`, `GeminiLiveClient`, `ToolDispatcher`,
`uiState`. Started via `startForegroundService()` (survives independently of
any binding) AND bound via a plain `LocalBinder` (no DI/Hilt in this
codebase). Holds a `PARTIAL_WAKE_LOCK` while a session is connected (new to
this codebase — no prior wakelock usage). **Deliberately does NOT call
`stopSelf()` in `onTaskRemoved`** — the one intentional divergence from
`device/PlaybackService.kt`'s otherwise-mirrored foreground-notification
pattern (`NotificationChannel` + `NotificationCompat`, `IMPORTANCE_LOW`),
since surviving task removal is the entire point. `onStartCommand` also
handles `ACTION_INCOMING_CALL`/`ACTION_SMS_RECEIVED` intents from the two
new receivers below, relaying to `liveClient?.sendSystemNote(...)` — a
no-op if no session is currently connected.

**`ui/MainViewModel.kt`** — reduced to a thin bound-client facade: binds to
`LiveAssistantService` from `init {}` (via Application context — keeps
`MainActivity.kt` completely unchanged, still a plain `viewModel()` call, no
new plumbing there), re-exposes the bound service's `uiState`/
`pendingYoutubeVideoId` via `flatMapLatest`, and delegates
`connect()`/`disconnect()`/`startPtt()`/`stopPtt()`/`clearError()` to the
bound instance. **`onCleared()` deliberately does NOT call `disconnect()`**
— only unbinds; the whole point is the session outlives this ViewModel.

**`camera/CameraManager.kt`** — `bind(lifecycleOwner)` no longer takes a
`PreviewView` and is now called exactly ONCE for the Service's entire
lifetime (from `LiveAssistantService.onCreate()`), binding Preview +
ImageAnalysis together as before (the existing "never call
`bindToLifecycle` twice" comment is now actually never violated, instead of
merely being a warning). The on-screen preview plugs in/out via new
`attachPreviewSurface(previewView)`/`detachPreviewSurface()` — just calling
`Preview.setSurfaceProvider(...)`/`(null)` on the already-bound `Preview`
use case, independent of any lifecycle binding. `MainScreen.kt`'s
`AndroidView` factory calls `viewModel.attachCameraPreview(previewView)`
(new `MainViewModel` method) instead of the old direct `cameraManager.bind`
call, with a `DisposableEffect` calling `detachCameraPreview()` on dispose.
Both `CameraManager` and `MainViewModel` remember a `pendingPreviewView` and
re-apply it once the async `ProcessCameraProvider`/Service-binding
completes, guarding the real race between a `PreviewView` attaching before
either finishes.

**Telephony (`receivers/CallBackgroundReceiver.kt`, new)** — a transient
`BroadcastReceiver` for `PHONE_STATE`; on ringing, resolves the caller via
`ContactsContract.PhoneLookup` and starts (not binds)
`LiveAssistantService` with `ACTION_INCOMING_CALL`. New `answer_phone_call`
tool (`ToolDeclarations.kt`/`ToolDispatcher.kt`/`AndroidDeviceToolHandler
.answerPhoneCall()`) — `ANSWER_PHONE_CALLS` + `TelecomManager
.acceptRingingCall()`. **Known, accepted risk, not yet verified on a real
device**: proceeding on the documented API for a non-default-dialer app;
search results were inconclusive on whether this actually works without
claiming the dialer role on current Android versions — if it throws
`SecurityException` in practice, that's the first thing to check (the
handler already catches and surfaces that specific exception distinctly).

**SMS (`receivers/SmsBackgroundReceiver.kt`, new)** — same transient-receiver
pattern for `SMS_RECEIVED`, starting `LiveAssistantService` with
`ACTION_SMS_RECEIVED`. New `send_sms`/`check_unread_sms` tools
(`AndroidDeviceToolHandler`) — `send_sms` reuses the existing
`lookupContactNumber()` helper (same name→number resolution
`make_phone_call` already does) before falling back to treating the input
as a raw number; `check_unread_sms` queries `Telephony.Sms.Inbox` with
`read = 0`. **Personal/sideloaded distribution only** (confirmed with the
user) — Google Play's policy restricting `READ_SMS`/`RECEIVE_SMS` to
default SMS/Phone/Assistant handler apps is a Store-listing policy, not an
OS-level technical block, so it doesn't apply here; would need
re-addressing before any Play Store submission.

**YouTube (`live/YouTubeSearchClient.kt`, new)** — direct 3rd-party OkHttp
calls to YouTube Data API v3 (`/search`, `/videos`), same pattern as
`OcrClient.kt`; new "YouTube API Key" Settings field
(`SettingsViewModel.kt`/`SettingsScreen.kt`, mirrors the existing
`ocrApiKey` field exactly). `search_youtube`/`get_video_info` were already
declared (previously routed to a hardcoded "not available" stub in
`ToolDispatcher.kt` — that stub is now replaced with real calls). New
`play_youtube_video(video_id)` tool — deliberately NOT wired through
`play_video`/`PlaybackService` (which expects a resolved stream URL):
playback uses the official `com.pierfrancescosoffritti.androidyoutubeplayer`
(IFrame) library instead, chosen specifically for ToS compliance over a
yt-dlp-style stream resolver. **Real, accepted limitation**: the IFrame
player is a WebView-based `YouTubePlayerView` that can only run with a
visible on-screen surface — YouTube's own ToS requires the player be
visibly rendered during playback — so `play_youtube_video` only works while
`MainActivity`'s UI is actually in the foreground with the small embedded
player visible (`MainScreen.kt`'s `YouTubePlayerOverlay`), unlike every
other tool in this app, which works with the screen off. `stop_music` now
stops BOTH playback surfaces (the existing `PlaybackService`/ExoPlayer path
AND the YouTube player, via `ToolDispatcher`'s new `onStopYoutubeVideo`
callback) — matches the user's "stop the music" intent regardless of which
tool started it, rather than adding a second, easily-confused stop tool.

**Manifest additions**: `ANSWER_PHONE_CALLS`/`READ_PHONE_STATE`/
`READ_CALL_LOG` (telephony), `RECEIVE_SMS`/`READ_SMS`/`SEND_SMS`,
`FOREGROUND_SERVICE_MICROPHONE`/`FOREGROUND_SERVICE_CAMERA`/`WAKE_LOCK`/
`POST_NOTIFICATIONS`/`REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`, plus the new
`<service>` (`foregroundServiceType="microphone|camera"`) and the app's
first two `<receiver>` entries. `MainActivity.kt`'s existing single bulk
`permissionLauncher.launch(...)` call was extended with all the new
dangerous permissions (same mechanism, no new pattern). Battery-optimization
exemption (`REQUEST_IGNORE_BATTERY_OPTIMIZATIONS`) is a Settings-screen
button (`PowerManager.isIgnoringBatteryOptimizations` check + the special
intent), not an automatic launch-time prompt.

**Not verified end-to-end on a real device from this environment** —
compile-verified only (`./gradlew :app:compileDebugKotlin`), same standing
caveat as every other round of Android work in this repo. Specifically
unverified: the Service actually surviving task-swipe/process death in
practice; `acceptRingingCall()` without the dialer role; `EXTRA_INCOMING_
NUMBER` availability given `READ_CALL_LOG` on the target Android version;
SMS broadcast delivery/priority timing; the YouTube player's actual
play/pause behavior across foreground/background transitions.

### Ducking TTS/music while the user is speaking (PTT) — superseded

**Superseded — see "Continuous VAD-gated listening + interruptible
sentence-level reading" below.** Push-to-talk itself was removed; the
`StreamingAudioPlayer.duck()`/`unduck()` mechanism this section describes
was deleted outright (replaced by `stopAndFlush()`), and the
`PlaybackService.ACTION_PAUSE`/`ACTION_RESUME` actions are now triggered by
VAD utterance-boundary events instead of button press/release. Kept here
for history — the underlying insight (nothing coordinated recording state
with playback state) is still what motivated the next redesign.

Real gap found and fixed: nothing coordinated "the user is actively
recording" with "audio is currently playing" anywhere in this codebase —
`ChatPanel.kt`'s PTT button was never blocked (fine, no fix needed there),
but `StreamingAudioPlayer` (Gemini's own TTS voice) and `PlaybackService`'s
ExoPlayer (music) both play completely independently of recording state,
and `PushToTalkRecorder` requests no audio focus at all — so TTS/music
audio could bleed into the mic as echo/feedback while the user tries to
speak, and `LiveAssistantService.startPtt()`/`stopPtt()` had zero
interaction with either playback path.

**Chosen fix: duck (pause) both playback surfaces for the duration of a PTT
hold, rather than requesting OS audio focus** — this app already knows
exactly when the user starts/stops speaking (`startPtt()`/`stopPtt()`,
explicit push-to-talk, not VAD-triggered), so explicit pause/resume calls at
those two points are simpler and more reliable than negotiating Android's
audio-focus stack (which `StreamingAudioPlayer`'s raw `AudioTrack` and
`PushToTalkRecorder`'s raw `AudioRecord` don't participate in at all today).

- **`StreamingAudioPlayer.duck()`/`unduck()`** (new) — `AudioTrack.pause()`/
  `.play()`, not `.stop()`/release — incoming chunks from `writeChunk()`
  (still arriving from Gemini's audio stream regardless of PTT state) keep
  queuing in the track's own buffer while paused, so playback resumes
  smoothly rather than dropping audio.
- **`PlaybackService`** gained `ACTION_PAUSE`/`ACTION_RESUME` (new,
  `onStartCommand` branches on these BEFORE the existing `stream_url`-based
  play flow) — `player?.pause()`/`.play()`; a pause/resume signal with no
  player yet (nothing playing) calls `stopSelf()` rather than lingering as
  an empty non-foreground service. Reused as a plain `startService()` call
  (this service has always been started-only, never bound — see `onBind()`
  stub) from `LiveAssistantService`.
- **`LiveAssistantService.startPtt()`/`stopPtt()`** now call
  `streamingPlayer.duck()`/`unduck()` and send `PlaybackService`'s new
  actions, symmetric with each other.
- **YouTube IFrame player** — its `YouTubePlayer` instance lives in
  Compose (`MainScreen.kt`'s `YouTubePlayerOverlay`), not the Service, so
  it can't be reached from `startPtt()`/`stopPtt()` directly; instead
  `YouTubePlayerOverlay` takes `isRecording` (already surfaced via
  `AppUiState`/`uiState.isRecording` — no new StateFlow needed) and a
  `LaunchedEffect(isRecording, player)` calls `player.pause()`/`.play()`
  directly. **Known, accepted quirk**: if the user manually paused the
  YouTube video themselves via its own on-screen controls, ending a PTT
  hold will resume it anyway (this effect can't distinguish "paused because
  we ducked it" from "user paused it") — a minor rough edge, not fixed,
  consistent with this feature's overall scope.
- **Not changed**: `PushToTalkRecorder`'s `AudioSource.MIC` (no platform
  echo cancellation) — considered, but not needed once playback is fully
  ducked during recording rather than left running concurrently; there's no
  overlapping audio left for echo cancellation to clean up.

### Continuous VAD-gated listening + interruptible sentence-level reading

Push-to-talk (tap-and-hold) is removed entirely, per the user's explicit
request. The mic now streams continuously, gated by a client-side amplitude
VAD — finally consuming `vadThreshold`/`startThreshold` ("Noise Gate"/
"Start Volume" in Settings), which had been threaded all the way from
`SettingsScreen.kt` through `MainViewModel.connect()` into
`LiveAssistantService.connect()` since the original migration but never
actually read inside the function body (a real, confirmed-dead pair of
parameters — `Parameter 'vadThreshold' is never used` was a standing
compiler warning until this landed).

**`audio/ContinuousVadRecorder.kt`** (new, replaces `PushToTalkRecorder.kt`
— deleted outright) — same `AudioRecord` capture shape (16kHz mono PCM16,
512-sample chunks, daemon thread, `computeRms()` reused verbatim) but runs
for the Service's whole connected lifetime (`start()` in `connect()`,
`stop()` in `disconnect()`/`onDestroy()`), not per-gesture. Per-chunk state
machine: **IDLE** — RMS `>= startThreshold` transitions to SPEAKING, fires
`onSpeechStart()`, forwards the chunk; otherwise the chunk is discarded
(never sent to Gemini), only `onVolumeChange()` fires (UI meter). **SPEAKING**
— every chunk forwarded via `onChunkReady`; once RMS has stayed below
`noiseGate` (`vadThreshold`) for `hangoverMs` (800ms), transitions back to
IDLE and fires `onSpeechEnd()` — "the user's line has been registered."

**Explicit design choice, confirmed with the user: NO ducking during
capture.** Music/reading keep playing normally through the entire window
from `onSpeechStart` to `onSpeechEnd` — this is a deliberate reversal of the
previous PTT-era ducking (see the now-superseded section above). Ducking
only happens once the utterance is fully registered, since that's when
Gemini's response is about to arrive and would otherwise overlap whatever
was already playing.

**`live/LiveAssistantService.kt`'s `onSpeechEnd` handler** — the actual
orchestration point: `liveClient.sendAudioStreamEnd()`, sets
`isRecording=false, isAwaitingResponse=true` on `AppUiState`, calls
`toolDispatcher.interruptReadingForUserTurn()` (see below), and sends
`PlaybackService.ACTION_PAUSE`.

**Response-drain timing (replaces `doLiveSession()`'s previously-no-op
`TurnComplete` handling)** — `resumeAfterResponse()` (music/`isAwaitingResponse`
only, deliberately never touches reading) fires after an estimated delay
computed from `responseBytesWritten * 1000 / 48000` (24kHz 16-bit mono
bytes/sec) minus elapsed time since the first response chunk, plus a small
`DRAIN_MARGIN_MS` (200ms) safety margin. **Accepted approximation**:
estimates drain time from bytes-written/sample-rate rather than polling the
AudioTrack's real playback-head position — good enough given the small
buffer sizes involved, not gold-plated on purpose.

**Reading interruption is a hard stop, not a duck** — `StreamingAudioPlayer`
is a single shared `AudioTrack` used by BOTH Gemini's own conversational
voice and reading-mode TTS, so they can never truly play concurrently on it.
`StreamingAudioPlayer.duck()`/`unduck()` were removed (see the superseded
section above); replaced by **`stopAndFlush()`** — `pause(); flush(); play()`
— which discards whatever reading audio is still queued (as opposed to a
plain pause, which would preserve it and play it back before Gemini's
response, stale and out of order) so the response starts clean.

**`live/ToolDispatcher.kt` — sentence-level reading pipeline (replaces
`speakReadingText()`, deleted outright)**:
- `splitIntoSentences(text)` — same `(?<=[.!?])\s+` boundary regex
  `toolGetReadingSection()`'s `splitChunks()` already used, just without
  repacking into ~500-char groups.
- `speakSentences(sentences)` — fire-and-forget, launches a cancellable
  `readingJob`. **Lookahead pre-synthesis**: sentence N+1's `Synthesize`
  call starts as an `async` child the moment sentence N *begins* playing
  (not when it finishes) — maximizes overlap so there's no gap between
  sentences as long as synthesis keeps pace with playback. On natural
  completion, clears `pendingReadingContinuation`. On cancellation, captures
  `sentences.subList(currentIndex, size)` into `pendingReadingContinuation`
  — storing the actual remaining sentence STRINGS (not a block-id+index
  pair) sidesteps any indexing drift across appended reading blocks.
- **`interruptReadingForUserTurn()`** (new, public) — called by
  `LiveAssistantService`'s `onSpeechEnd`: cancels `readingJob`, then calls
  the new `flushPlayback` constructor lambda (wired to
  `streamingPlayer.stopAndFlush()`). No-ops if nothing was reading.
- **New tool `continue_reading()`** → `toolContinueReading()` — resumes
  `pendingReadingContinuation` via `speakSentences()`, or returns
  `{"status":"nothing_to_continue"}`. **Deliberately never auto-triggered**
  — only this explicit voice command resumes interrupted reading, per the
  user's explicit requirement.
- `toolReadAloud()` and the live-reading OCR worker's "new"/"stitched"
  branch both switched from awaiting `speakReadingText()` to firing
  `speakSentences()` and returning immediately. **This is a deliberate,
  necessary behavior change, not an accidental regression**: Gemini Live's
  function calling is synchronous-only ("the model will not start
  responding until you've sent the tool response") — if `toolReadAloud()`
  awaited an entire page's TTS before returning, Gemini couldn't react to
  the user's interrupting speech at all until the whole page finished,
  defeating the point of VAD-based interruption. `toolReadAloud()`'s
  response shape changed accordingly: `{"status":"reading_started",...}`
  instead of the old `{"status":"read_aloud",...}` (which implied
  completion).

**`live/ToolDeclarations.kt`** — new `continue_reading` decl; SYSTEM_PROMPT
note that reading stops automatically the instant the user talks (nothing
to call for that) and `continue_reading()` resumes it on request.

**UI simplified** (`ui/ChatPanel.kt` deleted outright — no remaining
caller; `ui/MainScreen.kt` stripped down to exactly three things, per direct
user request: camera preview, the Settings button, and the YouTube IFrame
player). The chat history/transcript, connection chip, tracking/hand
bounding-box overlay, and guiding/walking banner are all gone from the
screen — `AppUiState` still carries the underlying data (nothing upstream
was removed), only this screen stopped rendering it. The YouTube player is
now **always mounted** (not conditionally composed on a non-null pending
video id) so it's immediately ready rather than being torn down/recreated
between playbacks; it shows/hides its own close button and loads/pauses
based on a nullable `videoId` param instead. `MainViewModel.startPtt()`/
`stopPtt()` and `LiveAssistantService.startPtt()`/`stopPtt()` are deleted
outright — nothing calls them any more.

**Known, accepted limitations**:
- No barge-in interruption of Gemini's OWN in-progress spoken reply if the
  user starts a new utterance while it's still talking — only reading-mode
  TTS gets interrupted by a new turn. `LiveServerEvent.Interrupted` (still
  a no-op) is the existing hook if this is revisited later.
- `continue_reading()` resumes the exact sentence list captured at
  interruption time — a stale-but-harmless snapshot, not re-derived from
  `readingBlocks` if reading-mode activity changed in between.
- VAD thresholds/hangover and the response-drain estimate are unverified on
  a real device (no device available from this environment, same standing
  caveat as every other round of Android work here) — compile-verified only.

### Self-echo / output-aware VAD gating, service restart recovery, reading-interrupt race (found via user report, fixed)

Real incident, reported directly by the user right after continuous
listening shipped: with the mic always live, the device's own speaker
output (Gemini's spoken reply, reading-mode TTS, the HRTF beacon's
continuous guidance tone during walking/guiding, music/YouTube) could be
picked up by the mic and misread as the user talking — feeding the
assistant's own voice back to itself and looping. Investigated and fixed
five distinct issues from that report:

1. **Self-echo/self-interruption loop — the core issue, fixed with real
   AEC, not just muting.** `ContinuousVadRecorder` used to capture on plain
   `AudioSource.MIC` with zero awareness of playback state. Fixed two ways,
   deliberately NOT by pausing playback during capture (that was already
   ruled out earlier — see the "Continuous VAD-gated listening" section
   above; the HRTF beacon alone makes blanket muting-while-anything-plays
   unworkable, since it's audible through nearly all of walking/guiding —
   gating the VAD off for that whole duration would make the assistant deaf
   during navigation):
   - **`AudioSource.VOICE_COMMUNICATION`** (not `MIC`) — enables platform
     acoustic echo cancellation/noise suppression on most devices for this
     capture session, the standard Android idiom for full-duplex voice
     apps. `AcousticEchoCanceler`/`NoiseSuppressor` are also explicitly
     attached to the `AudioRecord`'s session (`isAvailable()`-guarded, both
     released in the loop's `finally`) as a belt-and-suspenders measure —
     some devices need the explicit attach even with this source. This is
     the primary defense and preserves genuine barge-in.
   - **Output-aware threshold raise, defense-in-depth** — `ContinuousVadRecorder
     .start()` gained an `isOutputActive: () -> Boolean` param, polled once
     per chunk; while true, the effective start threshold is multiplied by
     `OUTPUT_ACTIVE_THRESHOLD_MULTIPLIER` (3.0x), so residual echo that
     leaks past AEC needs to be well above ambient to mis-register, while a
     genuinely louder direct interruption still gets through.
   - **Every output source wired in, per the user's explicit "check every
     output audio" instruction** (`LiveAssistantService.connect()`'s
     `isOutputActive` lambda): `streamingPlayer.isPlaying` (Gemini's own
     reply + reading TTS), `hrtfBeacon.isEmitting` (new — `overallGain >
     0.02f`, the beacon's own continuous guidance tone), `PlaybackService
     .isPlaying` (new — a companion `MutableStateFlow`, since
     `PlaybackService` is started-only/unbound and there was previously no
     way to query its state at all), and `_pendingYoutubeVideoId != null`
     (a proxy for the YouTube overlay — its real player instance lives in
     Compose/a WebView, not reachable from the Service, so presence of a
     pending video id stands in for "probably playing").
2. **(Same root cause as #1, not a separate fix.)**
3. **Service killed by the OS while running never recovered — fixed with a
   persisted-flag-gated auto-reconnect.** `LiveAssistantService.onStartCommand`
   now branches on `intent == null` specifically (not `intent?.action ==
   null`) — a null Intent object is unambiguous for "the system restarted
   this process after killing it" (`START_STICKY`); `MainViewModel`'s own
   initial `startForegroundService()` call always passes a real Intent
   object (its `action` happens to be null, but the object itself isn't),
   so this doesn't false-trigger on ordinary app launch. `restoreSessionFromPrefsIfAvailable()`
   reconnects using the exact same `SharedPreferences` keys
   `SettingsViewModel` already persists — but ONLY if a new
   `session_was_active` boolean (set `true` in `connect()`, `false` in
   `disconnect()`) says a session was genuinely live when the process died;
   without this flag, a user who explicitly disconnected would get silently
   reconnected the next time the OS happened to kill+restart the otherwise-
   idle Service, which is not what they asked for.
4. **Response-drain race (the tail-end echo of Gemini's own reply)** — partly
   the same root cause as #1 (now covered by AEC + the output-aware
   threshold raise while `isAwaitingResponse`), partly a timing margin issue:
   `DRAIN_MARGIN_MS` bumped 200ms → 350ms as extra cushion. Explicitly
   documented as no longer load-bearing on its own now that real AEC is the
   primary defense.
5. **Interrupted sentence-level reading could desync from a real race** —
   `ToolDispatcher.interruptReadingForUserTurn()` used to `readingJob
   ?.cancel()` then immediately call `flushPlayback()`. Coroutine
   cancellation is cooperative (`speakSentences()`'s loop only checks
   `ensureActive()` once per chunk, not mid-chunk) — cancelling and flushing
   without waiting let an already-in-flight `playPcm()` call for the
   about-to-be-abandoned sentence land in the AudioTrack buffer AFTER the
   flush, which could both audibly overlap the start of Gemini's response
   and leave `pendingReadingContinuation`'s resume cursor pointing at the
   wrong sentence. Fixed by blocking on `runBlocking { job.join() }` between
   `cancel()` and `flushPlayback()` — safe here specifically because this
   function is always invoked from `ContinuousVadRecorder`'s own dedicated
   background thread (never the main thread), so the brief block (at most
   one chunk's worth of synthesis/playback, typically well under 100ms)
   doesn't stall the UI, only that thread's next mic-chunk read.

**Not verified end-to-end on a real device from this environment** — same
standing caveat as the rest of this feature. Specifically unverified:
whether platform AEC is actually effective against the HRTF beacon's
synthesized tone and TTS speech on a real device (AEC implementations vary
by OEM and are tuned for voice-call content, not arbitrary media), and
whether `OUTPUT_ACTIVE_THRESHOLD_MULTIPLIER`'s 3.0x is the right value in
practice (a real-device tuning knob if false triggers or missed barge-in
turn out to be common).

### Voice-message-sent confirmation cue

Requested directly by the user: a gentle audible confirmation every time
the client successfully hands a recorded utterance off to Gemini Live —
so the user gets non-visual feedback that their voice message actually
went out, not just that the mic stopped recording.

`GeminiLiveClient.send()` (private) now returns `Boolean` — whichever
`OkHttp WebSocket.send()` itself reports (`false` if there's no open
socket) — instead of discarding it. `sendAudioStreamEnd()` (the call that
marks the end of one recorded utterance, i.e. one "voice message") is the
only caller whose result is used: it now returns that `Boolean` up to
`LiveAssistantService`. The other `send()` callers
(`sendAudioChunk`/`sendVideoFrame`/`sendSystemNote`/`sendToolResponse`)
are unaffected — still `Unit`-returning, discarding the result as before.

**Fires once per utterance, not per audio chunk.** The natural place to
hook this is `LiveAssistantService`'s existing
`vadRecorder.onSpeechEnd` handler (see "Continuous VAD-gated listening"
above) — chunks stream continuously at ~512-sample granularity while
SPEAKING (`onChunkReady`), so a cue per chunk would be near-constant
noise; "a voice message" is the whole registered utterance, which is
exactly what `onSpeechEnd`/`sendAudioStreamEnd()` represents. `onSpeechEnd`
now checks `sendAudioStreamEnd()`'s return value and only plays the cue
when it's `true` — a dropped send (e.g. socket closed mid-utterance)
correctly stays silent rather than confirming something that didn't
actually happen.

`LiveAssistantService.playVoiceSentCue()` — a synthesized
`ToneGenerator` tone (`TONE_PROP_BEEP2`, ~50ms, 40% of `MAX_VOLUME`), same
"no bundled audio asset needed" precedent `ToolDispatcher.
playDeadEndAlert()` already established for the dead-end tone. Deliberately
**not** added to `ContinuousVadRecorder`'s `isOutputActive()` output-aware
VAD gating (unlike `streamingPlayer`/`hrtfBeacon`/`PlaybackService`/the
YouTube overlay, see "Self-echo / output-aware VAD gating" above) — it's a
single ~50ms blip immediately after the mic has already transitioned back
to IDLE, not a sustained output plausibly mistaken for new speech; adding
it to that gate was considered unnecessary complexity for a real risk this
small. `voiceSentToneGenerator` is released in `onDestroy()` alongside the
Service's other cleanup.

### News / Radio tools (Vietnam only, hardcoded)

Requested directly by the user: `get_top_news`/`search_news`/`play_radio`/
`stop_radio` tools, scoped specifically to Vietnam/Vietnamese — no
`country`/`language` parameters are exposed to Gemini at all (hardcoded
internally), so there's nothing for the model to get wrong there.

- **`NewsClient.kt`** (new) — direct 3rd-party call to Google News' public
  RSS feed (`news.google.com/rss` / `.../rss/search`, `hl=vi&gl=VN&
  ceid=VN:vi` hardcoded), same "no server proxy" pattern as
  `OcrClient.kt`/`YouTubeSearchClient.kt` — no API key, no quota to run
  out of, no Settings field needed at all. Parses the RSS `<item>` entries
  (title/source/pubDate/link) via Android's built-in `android.util.Xml`
  `XmlPullParser` rather than adding a new XML dependency for a handful of
  fields.
- **`RadioClient.kt`** (new) — direct 3rd-party call to the free, keyless
  Radio-Browser API (`radio-browser.info`), searching `country=Vietnam`
  stations by name, sorted by votes. Resolves a station to its
  `url_resolved` stream URL only — it never plays anything itself.
  `play_radio`'s tool handler (`ToolDispatcher.toolPlayRadio()`) hands that
  URL to the SAME `play_video`/`PlaybackService` (ExoPlayer) path YouTube's
  resolved-stream fallback already uses — a live radio stream is just
  another stream URL, so no separate radio player class was written.
- **`stop_radio`** is declared as its own tool (matching what Gemini is
  told it can call) but routes straight to the existing `toolStopMusic()`
  — same handler `stop_music` already uses, which already stops
  whichever playback surface is actually active. No separate stop logic
  was written; the two tool names are aliases from the dispatcher's point
  of view.
- Both clients are constructed unconditionally as private fields inside
  `ToolDispatcher` (`newsClient`/`radioClient`) — unlike
  `youtubeSearchClient` (which needs a Settings-provided API key and is
  therefore nullable/constructor-injected), neither news nor radio needs
  any configuration, so there's no equivalent "not configured" degrade
  path for either.
- **Known, accepted limitations**: Google News RSS and Radio-Browser are
  both public, unauthenticated, best-effort services with no SLA — a
  future outage or shape change in either feed degrades to the existing
  "empty results" convention (`found: false` / "station not found"), not a
  crash. Not verified end-to-end on a real device from this environment —
  compile-verified only, same standing caveat as the rest of this file's
  Android work.

### Radio/YouTube real-device bugs (crash, silent tool failures, VAD over-triggering)

Three distinct bugs found from real-device logcat the user provided, after
the News/Radio tools above landed — all now fixed.

**1. `play_radio` crashed the whole app.** Real stack trace:
`IllegalStateException: No suitable media source factory found for content
type: 2` (content type 2 = HLS/`.m3u8`) thrown synchronously from
`PlaybackService.onStartCommand()`'s `setMediaItem()`/`prepare()` — many
internet radio stations serve HLS, but the app only depended on
`media3-exoplayer` core, with no HLS extension module registered, and an
uncaught exception inside a Service's `onStartCommand` kills the whole
process, not just that Service. Fixed two ways: added
`androidx.media3:media3-exoplayer-hls` (`build.gradle.kts`), and wrapped
`setMediaItem`/`prepare`/`play` in try/catch in `PlaybackService.kt` — an
unsupported stream format now logs and stops cleanly (`stopSelf()`)
instead of crashing, since `Player.Listener.onPlayerError` (already
present) only ever catches ASYNC playback errors, not this synchronous
factory-lookup exception.

**2. `search_youtube` worked but `play_youtube_video` silently failed —
root-caused via decompiling the `android-youtube-player` library itself**
(no source available locally, so `javap -c` against the AAR's `classes.jar`
was used to read `YouTubePlayerBridge.parsePlayerError`): it only maps the
5 officially-documented YouTube IFrame API error codes (`2`/`5`/`100`/
`101`/`150`) to a named `PlayerConstants.PlayerError` — any other raw code
falls through to `UNKNOWN`. A real logcat capture showed `onError: UNKNOWN`
for a normal, playable video, meaning the WebView received a genuinely
non-standard error — consistent with a missing IFrame `origin` parameter
(a known class of issue for WebView-embedded YouTube players). The library
supports `IFramePlayerOptions.Builder().origin(...)`, but this app's
`YouTubePlayerOverlay` (`MainScreen.kt`) never set one, relying on
automatic initialization with none configured.

**Round 1 (wrong origin value): `origin("https://www.youtube.com")`.**
Real-device testing showed this made things WORSE, not better — the error
(now confirmed as "152-4") started firing IMMEDIATELY on load, before any
playback attempt, instead of during buffering as before. Root cause of
the regression: `origin` must identify the EMBEDDING app/page, not claim
to BE youtube.com itself — using YouTube's own domain as the origin is
semantically backwards and plausibly reads as spoofing to Google's
anti-bot/referrer verification (which actively rejects a WebView with no
valid Referer/origin OR a suspicious one — confirmed against user-supplied
documentation of this exact error code's cause).

**Round 2 (current): `origin("https://www.youtube-nocookie.com")`** — the
value YouTube's own documented workaround for WebView/native-app embeds
uses. Automatic initialization (origin left fully unset, the library's
default) is the WORST case for this specific error, not a safe default, so
`enableAutomaticInitialization = false` + explicit `initialize()` stays.
**Not confirmed fixed on-device from this environment** — same standing
caveat as round 1; if 152-4 still recurs, the next thing to check is
whether this library's `loadDataWithBaseURL()`-based HTML injection (no
custom HTTP headers possible through its public API) is fundamentally
incompatible with YouTube's verification, which would require bypassing
the library for a raw WebView + `loadUrl(url, extraHeaders)` with an
explicit `Referer` header — a much larger change, not attempted here.

Also fixed in the same pass (found while adding the `onError`/
`onStateChange` debug logging that surfaced bug #2): `YouTubePlayerOverlay`'s
listener object is created once inside `AndroidView`'s `factory` lambda
(which only runs once per view instance), so its log lines were closing
over the FIRST composition's `videoId` parameter value, not whatever video
was actually loaded later — every log line displayed a stale, wrong
video id. Fixed via `rememberUpdatedState(videoId)` (a `State` object
whose identity is stable across recompositions but whose `.value` always
reflects the latest one), read inside the listener instead of the raw
parameter.

**3. VAD required shouting to trigger — real bug, not just a threshold
tuning issue.** `ContinuousVadRecorder`'s output-aware gating (see
"Self-echo / output-aware VAD gating" above) multiplies the start
threshold by `OUTPUT_ACTIVE_THRESHOLD_MULTIPLIER` (3x) whenever it
believes audio is currently playing, to avoid the mic mistaking echo for
speech. The YouTube signal feeding that check was `_pendingYoutubeVideoId
!= null` — "a video is loaded" — which stays true for as long as the
on-screen player exists, INCLUDING while it's paused (e.g. ducked during
`isAwaitingResponse`, or manually paused via the on-screen controls) —
permanently engaging the 3x multiplier with nothing actually audible.
Given this session's heavy YouTube testing right before the report, this
was almost certainly the cause. Fixed by threading the IFrame player's
REAL `PlayerConstants.PlayerState` through to the Service instead of
proxying on "is a video loaded":
`YouTubePlayerOverlay`'s `onStateChange` now calls a new
`onPlaybackStateChanged: (Boolean) -> Unit` param (true only for
`PlayerState.PLAYING`) → `MainScreen.kt` wires it to
`MainViewModel.reportYoutubePlaybackState()` (new) →
`LiveAssistantService.reportYoutubePlaybackState()` (new) sets a new
`_isYoutubePlaying` flag, which replaces `_pendingYoutubeVideoId != null`
in the `isOutputActive()` lambda. `_isYoutubePlaying` is also reset to
`false` alongside every existing `_pendingYoutubeVideoId = null` site
(`onStopYoutubeVideo`, `disconnect()`, `dismissYoutubeVideo()`) for
consistency, though `onStateChange` reporting a non-PLAYING state on
stop/dismiss already covers it in practice.

**Debug logging added in the same investigation** (still in place,
useful for confirming any of the above on a real run):
`GeminiLiveClient.handleServerMessage()` previously had no handling at all
for a top-level `"error"` key from Gemini's WebSocket — a rejected/
malformed session setup was silently dropped with zero log output,
indistinguishable from "the model just didn't call any tool." Now logs
(`GeminiLiveClient` tag) any `error` object in full, any `goAway` message,
and — as a catch-all — the top-level keys of any server message that
doesn't match a known shape.

### Reading-mode crash on read_aloud (uncaught exception in the lookahead prefetch coroutine)

Real, reproducible crash reported by the user: scan/OCR/Q&A all worked,
but `read_aloud` crashed every time — the one pipeline those don't share
is `ToolDispatcher.speakSentences()`'s sentence-level lookahead reading
(see "Reading-mode OCR correction + fuzzy stitching" above). Root cause:
`speakSentences()`'s launched coroutine used `async` for prefetching the
NEXT sentence's TTS synthesis, with only a `catch (CancellationException)`
around the whole body — no catch-all for real exceptions. Kotlin
structured concurrency propagates a failing `async` child's exception to
its parent Job EAGERLY, even before the child is ever `.await()`ed — so a
real failure (a `Synthesize` gRPC hiccup, an `AudioTrack` error) could
escape the per-sentence `try { prefetch.await() } catch` entirely, and
since this coroutine ran on a scope with no `CoroutineExceptionHandler`
installed, an uncaught exception on what is effectively a root coroutine
crashes the whole process — not just this feature.

Fixed with three layers, in `speakSentences()`/`StreamingAudioPlayer.kt`:
- A `CoroutineExceptionHandler` added directly to `speakSentences()`'s
  `scope.launch(...)` call — the actual crash-preventing fix, logging
  `[reading] speakSentences failed: ...` under the `ToolDispatcher` tag
  instead of taking down the app.
- Each `playPcm(c)` call inside the loop is now individually try/caught —
  one bad audio chunk no longer aborts the rest of the page.
- `StreamingAudioPlayer.writeChunk()` (raw `AudioTrack.write()`, no
  try/catch previously — unlike its sibling `stopAndFlush()`, which
  already guards its own `AudioTrack` calls) now catches and logs instead
  of throwing, covering a concurrent stop/release racing an in-flight
  chunk write.

Not verified end-to-end against the user's specific original crash from
this environment (no device/stack trace was available to confirm the
exact exception) — this closes the one structural gap in the pipeline
capable of crashing the whole app regardless of the underlying cause, and
adds the logging needed to identify that cause precisely if it recurs.

### Visual re-ID for object memory (get_object_from_memory / is_this_object)

`get_object_from_memory`/`remember_object` were previously TEXT-ONLY — no
image or visual embedding was ever stored, only a text description
embedded via MiniLM (`PerceptionService.Embed`) for semantic search. This
is now extended with a second, separate visual index, so "where's my
keys"-style recall can also visually confirm a candidate object currently
in view rather than trusting the text match alone.

- **`LocalMemoryStore.kt`** — new per-label visual-embedding store,
  `<label>.objemb.json` (parallel to, and deliberately separate from, the
  existing `<label>.vec.json` TEXT embedding index — different embedding
  space entirely: DINOv2 ViT-S/14 384-dim visual re-ID vectors vs. MiniLM
  text vectors). New methods: `addObjectEmbedding(label, vector)` appends
  one visual embedding (a label can accumulate several across multiple
  `remember_object` calls — different angles/lighting of the same physical
  item); `getObjectEmbeddings(label)`; `hasObjectEmbeddings(label)`;
  `bestObjectSimilarity(label, vector)` — max cosine similarity against
  every stored embedding for that label (`-1f` if none stored).
- **`ToolDispatcher.kt`** — two new shared helpers: `detectAll(prompt,
  frame)` runs `PerceptionService.AnalyzeFrame(DETECT)` (GroundingDINO
  open-vocab, same RPC `toolRunDetection` already uses) and returns every
  detection, not just the best one; `embedBox(box, frame)` runs
  `AnalyzeFrame(EMBED)` for a DINOv2 visual embedding of one box (reuses
  the same underlying model `TrackingService.GetEmbedding`/
  `server/tools/embedder.py`'s `DINOv2Embedder` already provide, just via
  the `AnalyzeFrame` path). No new gRPC RPCs or proto changes were needed
  — both helpers route through the existing `AnalyzeFrame` RPC's `DETECT`/
  `EMBED` ops, just combined into a detect-then-embed-each-candidate
  pipeline instead of being exercised individually.
- **`remember_object(label, description)`** now also attempts a visual
  capture: if the current camera frame is available, it runs
  `detectAll(description, frame)`, takes the highest-scoring detection,
  embeds it via `embedBox`, and stores the vector via
  `addObjectEmbedding`. Best-effort — a failed/no detection still saves
  the text description alone (`visual_captured: false` in that case).
- **`get_object_from_memory(query)`** rewritten: still resolves the query
  via the existing text semantic search first (MiniLM cosine, unchanged)
  to get a candidate label+description. If a frame is available AND that
  label has stored visual embeddings, it additionally visually verifies:
  runs `detectAll(description, frame)` to find every matching box
  currently in view, embeds each via `embedBox`, and scores each against
  the label's stored embeddings via `bestObjectSimilarity`. Requires
  similarity `>= OBJECT_MATCH_MIN_SIM` (0.55) to count as a visual match.
  Deliberately does NOT guess between near-tied candidates (e.g. two
  similar bottles in view) — if the top two candidates' scores are within
  `AMBIGUITY_MARGIN` (0.05) of each other, returns `ambiguous: true` with
  a message instructing Gemini to ask the user to clarify (point more
  directly, describe left/right/color) instead of silently picking one
  and mis-tracking. Falls back to the old text-only result when there's no
  frame or no stored visual reference for that label (`visual_confirmed`
  omitted in that case; `visual_confirmed: false` when a frame+embeddings
  existed but no confident visual match was found).
- **New tool `is_this_object(label)`** (`ToolDeclarations.kt`/
  `ToolDispatcher.kt`) — answers "is this my X?" / "is this the X I
  remembered?": requires the label to already have a stored visual
  reference (errors otherwise, telling the caller it needs to have been
  visible during a prior `remember_object` call). Runs
  `detectAll(description, frame)` against the label's stored TEXT
  description as the GroundingDINO prompt, then picks the LARGEST
  bounding box by area among the detections (not the highest-scoring one
  — deliberately "whatever the user is most likely pointing the camera
  at"), embeds it, and compares against the label's stored embeddings.
  Uses a stricter threshold, `IS_THIS_OBJECT_MIN_SIM` (0.65), than
  `get_object_from_memory`'s disambiguation threshold, since this is a
  direct binary confirmation rather than a best-of-several pick. Returns
  `{is_match, similarity, label, box_xyxy}` (or a `reason` field:
  `no_matching_object_in_view` / `embedding_failed` when `is_match` is
  false due to a hard stop rather than a low score).
- Thresholds live in `ToolDispatcher.kt`'s companion object:
  `OBJECT_MATCH_MIN_SIM=0.55f`, `AMBIGUITY_MARGIN=0.05f`,
  `IS_THIS_OBJECT_MIN_SIM=0.65f` — chosen distinct from
  `server/tools/embedder.py`'s `DINOv2Embedder` docstring's own general
  "cosine >= 0.75 = same target" guidance for continuous ORB-tracking
  re-ID, since these are different tasks (best-of-several-candidates pick
  vs. a stricter yes/no confirm) with different appropriate bars.

Verified via `./gradlew :app:compileDebugKotlin` — BUILD SUCCESSFUL, no
new warnings introduced by this change. Not verified end-to-end on a real
device — same standing caveat as the rest of this file's Android work.

### Voice-message-sent cue volume + Gemini Live persona/voice ("Pixie")

Two direct user requests, addressed together since both touch
`GeminiLiveClient`'s session setup:

- **Cue volume** — "Voice-message-sent confirmation cue" (above) was too
  quiet to reliably notice; `LiveAssistantService.playVoiceSentCue()`'s
  `ToneGenerator` volume bumped from 40% to 100% of `ToneGenerator.
  MAX_VOLUME`.
- **Persona + voice** — `GeminiLiveClient` gained a `voiceName: String =
  "Leda"` constructor param, sent as `generationConfig.speechConfig.
  voiceConfig.prebuiltVoiceConfig.voiceName` in the setup message (Gemini
  Live API's documented schema for selecting a prebuilt TTS voice — "Puck"/
  "Zephyr" are alternates). A voice alone only changes the TTS timbre, not
  how the model behaves, so `ToolDeclarations.SYSTEM_PROMPT` also gained a
  `PERSONA — Pixie` section (a tiny, energetic, high-pitched, whimsical
  fairy) ahead of `CORE RULES` — explicitly scoped as a STYLE layer only:
  brevity, hazard warnings, and `[SYSTEM]` events still always take
  priority over character flavor, since parts of this app are safety-
  critical (step-down/obstacle warnings, incoming-call alerts) and must
  stay fast and clear regardless of persona.

### Reading-mode TTS moved on-device (`ReadingTtsPlayer.kt`) — no longer needs the gRPC server

Requested directly by the user, after a real logcat capture (during this
same session) traced every `read_aloud`/live-reading TTS failure back to
`ECONNREFUSED` connecting to the gRPC server's `PerceptionService.
Synthesize` RPC (KokoroTTS) — scan/OCR/Q&A all kept working throughout,
since OCR.space and Gemini Live's own voice never touch that server at
all; only reading-mode TTS specifically depended on it being reachable.
Reading mode now uses Android's own on-device `android.speech.tts.
TextToSpeech` instead — no server round trip, no network dependency, for
this path at all.

- **`ReadingTtsPlayer.kt`** (new, `com.tracking.client.audio`) — wraps
  `TextToSpeech`. Android's own engine already plays queued utterances
  back-to-back on its own (`speak(text, QUEUE_ADD, ...)`), so no manual
  "wait for the previous one, then start the next" logic is needed for
  gapless playback itself. What this class adds, per direct user request:
  a bounded lookahead of at most `MAX_QUEUED_AHEAD` (3) sentences handed to
  the OS queue at any one time, refilled one at a time (via
  `UtteranceProgressListener.onDone()`) as each finishes — rather than
  dumping an entire, possibly still-growing (live reading) buffer into the
  OS queue up front. `speakFrom(globalStartIndex, sentences,
  onSentenceDone, onAllDone)` — `onSentenceDone(newCursor)` fires after
  EACH sentence finishes (new cursor = one past it), so the caller's
  persistent position stays accurate throughout, not just at the end.
  `stop()` clears the OS queue immediately (synchronous, no coroutine
  cancellation/join race to manage, unlike the old design below).
- **`ToolDispatcher.kt`** — `synthesizeSentence()`/the old async-prefetch-
  of-1 loop in `speakSentencesFrom()` are gone; it now just calls
  `readingTts.speakFrom(...)`, with `onSentenceDone` writing straight into
  `state.readingCursorSentenceIndex` (unchanged field/semantics — see
  "Reading mode redesign" below for that cursor's own design).
  `interruptReadingForUserTurn()` simplified from a
  cancel-then-`runBlocking`-join-then-flush dance to a single
  `readingTts.stop()` call — TextToSpeech.stop() is synchronous and halts
  immediately, so there's no cancellation race left to guard against.
  `playPcm`/`flushPlayback` constructor params are removed outright (no
  remaining caller — reading no longer shares `StreamingAudioPlayer`'s
  `AudioTrack`/callback with Gemini's own voice at all).
- **`StreamingAudioPlayer.stopAndFlush()`** deleted outright (its only
  caller was the removed `flushPlayback` callback) — `StreamingAudioPlayer`
  now exclusively handles Gemini's own spoken voice.
- **`LiveAssistantService.kt`** — new `readingTts by lazy { ReadingTtsPlayer
  (this) }`, Service-scoped like `hrtfBeacon`/`streamingPlayer` (reused
  across reconnects, not recreated per `connect()` the way `ToolDispatcher`
  itself is) — `ToolDispatcher.shutdown()` only `stop()`s it per
  connection; only `onDestroy()` actually `release()`s the underlying
  `TextToSpeech` engine.
- **`AndroidManifest.xml`** — new `<queries>` block declaring
  `android.intent.action.TTS_SERVICE`, required on Android 11+ (API 30+)
  for the app to see the device's installed TTS engine at all under
  package-visibility rules — without it, `TextToSpeech`'s init callback
  would report failure on those OS versions.
- **Known, accepted tradeoff**: reading-mode audio no longer reaches
  `LocalEdgeDevice`/`edgeDevice.emitAudio()` — that forwarding came from
  the old `playPcm` callback, which fed both `streamingPlayer.writeChunk()`
  AND the edge device from the same PCM chunks. Native `TextToSpeech` plays
  directly through the OS's own audio pipeline with no raw-PCM interception
  point exposed by this design, so only Gemini's own spoken voice (still
  wired directly in `LiveAssistantService`'s `LiveServerEvent.Audio`
  handler) reaches the edge device now. Revisiting this would mean either
  `synthesizeToFile()` to a temp WAV and re-decoding it for forwarding, or
  a custom `AudioTrack`-based TTS bridge — a real added-complexity/latency
  tradeoff, not attempted here since it wasn't asked for.
- Not verified end-to-end on a real device from this environment —
  compile-verified only, same standing caveat as the rest of this file's
  Android work.

### Pixie + Angle modules — discrete 4-point HRTF cue, shared heading tracking (tracking mode + guiding/walking)

Requested directly by the user: bring the design validated in
`test_module/pixie_hrtf_app/` (see "Pixie HRTF Test Harness" below) into the
production app, replacing `HrtfBeaconPlayer`'s continuous-panning beacon in
BOTH places that used it — tracking mode (guide a hand to an object) and
guiding/walking mode (guide the user's facing/steering direction) — with
two new, explicitly-named, reusable modules. Confirmed with the user along
the way: tracking's "up/down" is screen-space (the target's vertical
position in the camera frame), NOT depth/distance; volume convention is
"quiet when correctly positioned, louder when off" for both callers; the
production Pixie **teleports** between 4 fixed points (no gradual/sine-eased
flight like the test harness's autonomous `PixieMotion` — that was a
test-only validation loop, the real Pixie is 100% caller-commanded so
there's nothing to animate between).

- **`PixieController`** (`client/android/app/src/main/java/com/tracking/client/audio/PixieController.kt`,
  new) — a deliberately DUMB 4-point HRTF cue player, no angle/deviation
  logic baked in at all (different callers compute point/volume from
  completely different signals — see below — so that mapping stays with
  the caller). `enum class PixiePoint { LEFT, RIGHT, UP, DOWN, CENTER }`
  (CENTER = silent placeholder); `start()`/`stop()`/`move(point)`
  (teleport — just switches which cached HRTF filter index is used, no
  gradual flight)/`setVolume(gain: Float)` (0f..1f, smoothed internally via
  `GAIN_SMOOTHING` to avoid clicks)/`mute() = setVolume(0f)`. 4 fixed
  cardinal filter pairs — `LEFT` (azimuth -90°), `RIGHT` (+90°), `UP`
  (elevation +45°), `DOWN` (elevation -45°) — resolved via
  `HrtfConvolver.nearestIndex()` ONCE at `start()`, never re-searched per
  chunk (contrast `HrtfBeaconPlayer.updateDirection()`'s continuous
  brute-force scan over all 710 HRIR directions). `cueVolume` (new,
  mirrors `HrtfBeaconPlayer.cueVolume`) — the Settings "Cue Volume" master
  multiplier, applied on top of whatever gain a caller's own deviation
  mapping computes via `setVolume()`. `isEmitting` (new, mirrors
  `HrtfBeaconPlayer.isEmitting`) — `currentGain > 0.02f`, wired into
  `ContinuousVadRecorder`'s output-aware VAD gating the same way
  `hrtfBeacon.isEmitting` used to be. Reuses `HrtfConvolver` (unchanged,
  real HRTF) and the same `assets/fluttering.mp3` loop `HrtfBeaconPlayer`
  already decodes (`decodeAssetToMonoPcm`/`clipToShort`/`resampleLinear`
  widened from `private` to `internal` in `HrtfBeaconPlayer.kt` so
  `PixieController` can reuse them without duplicating ~90 lines of
  MediaCodec decode boilerplate). **`HrtfBeaconPlayer`/`HrtfConvolver`
  themselves stay in the codebase — `HrtfConvolver` becomes a shared
  dependency, `HrtfBeaconPlayer` itself is left unreferenced** (tracking/
  guiding/walking all now call `PixieController` instead) — same "kept in
  case revisited" precedent this codebase already uses elsewhere (e.g.
  `beacon_preview.py`, `gemma_vlm.py`).
- **`AngleTracker`** (`client/android/app/src/main/java/com/tracking/client/live/AngleTracker.kt`,
  new) — a drop-in replacement for `RotationTracker` (same
  `accumulatedRotation()`/`resetAccumulator()`/`reset()` shapes, so
  `ToolDispatcher`'s existing pose-extrapolation math — the mapping-stream
  collector's latency compensation, `buildMappingChunk()`,
  `stopActiveModes()` — barely changed, just the injected type/name), on
  top of which it adds:
  - **Luma-direct input** — `processLumaFrame(luma, width, height,
    rowStride, rotationDegrees)` takes the camera's raw Y-plane directly
    (see `CameraManager.kt`'s new `lumaFlow` below) instead of
    `RotationTracker.processFrame(frameJpeg: ByteArray)`'s
    `BitmapFactory.decodeByteArray` → `ARGB_8888` → `cvtColor` round trip —
    ported from the same lag investigation that motivated
    `pixie_hrtf_app`'s own luma-direct camera source.
  - **Configurable resolution/feature count**, defaulting to **1000
    features / 480px** for this app — heavier than the test harness's own
    300/320 defaults, a deliberate separate choice for production. This
    tracker's own working image is intentionally independent of (much
    smaller than) whatever `CameraManager` sends to `MappingService`/
    RTAB-Map over the network (`frameIntervalMs`/`walkingIntervalMs`, 640px
    cap) — that JPEG mapping pipeline is completely unaffected.
  - `currentHeadingDeg()`/`driftedAngleDeg()` (new) — clean accessors
    other modules can read, exposing what used to only be computed ad hoc
    inline in `ToolDispatcher`'s mapping-stream collector.
  - `setAuthoritativeHeadingDeg(headingDeg)` (new) — the "update direction
    when direction info sent from RTAB-Map through the server" half:
    called once per accepted `MappingService` pose fix (via the new
    `HrtfBeacon.worldHeadingDeg(pose)` helper), remembers the new baseline
    AND resets the accumulator in one call — replaces the bare
    `resetAccumulator()` call that used to sit at that point in the
    mapping-stream collector.
  - Still deliberately rotation-only (Essential-matrix decomposition, no
    scale ambiguity) — same "NOT verified against a live device, first
    thing to check if a head turn sounds mirrored" caveat `RotationTracker`
    always carried, since the composition convention is unchanged.
  - **`RotationTracker.kt` itself stays in the codebase, unreferenced**
    (same precedent as `HrtfBeaconPlayer.kt` above).
- **`CameraManager.kt`** — additive-only `lumaFlow: SharedFlow<LumaFrame>`
  (capacity 4, `DROP_OLDEST`) alongside the existing JPEG `frameFlow`:
  `processFrame(imageProxy)` pulls `imageProxy.planes[0]` (Y-plane) and
  emits it whenever `mappingMode` is `"guiding"` or `"walking"` — no
  changes to any existing emission/gating logic. **Known, accepted
  limitation, same as the existing JPEG path**: this reuses whatever
  resolution the shared `ImageAnalysis` use case is bound at
  (`HIGHEST_AVAILABLE_STRATEGY`, often full sensor resolution) — not a
  newly-introduced cost, and not fixed in this pass (changing the shared
  resolution strategy would affect every other consumer, out of scope).
- **`HrtfBeacon.worldHeadingDeg(pose): Float`** (new, `HrtfBeacon.kt`) —
  rotates the camera-local forward vector `(0,0,1)` by `pose`'s quaternion
  and reads `atan2(x, z)`, same convention as `AngleTracker`'s own
  `headingDegOf()` and `mapping_servicer.py`'s `_pose_heading_rad()`. Feeds
  `angleTracker.setAuthoritativeHeadingDeg()` on every accepted server fix.
- **Tracking mode → Pixie (`ToolDispatcher.updateTrackingPixie()`,
  replaces `updateTrackingBeacon()`)** — reuses
  `HrtfBeacon.directionFromBox()` **unchanged** (already gives screen-space
  azimuth/elevation from the tracked box's center — exactly the "y axis of
  the frame" signal confirmed with the user, no depth/size heuristic
  needed). A **2-phase state machine** per the user's explicit spec,
  `private enum class TrackingAxis { HORIZONTAL, VERTICAL }`: phase
  HORIZONTAL (left/right) runs first; once the target is centered
  horizontally (within `TRACKING_H_DEADZONE_DEG`), phase VERTICAL (up/down)
  takes over; if the target then drifts back out past that SAME horizontal
  deadzone while in VERTICAL, phase drops back to HORIZONTAL first —
  re-centering always takes priority over the forward/back cue. Reset to
  HORIZONTAL in `toolStartTracking()`/`stopActiveModes()`.
  `TRACKING_H_DEADZONE_DEG`/`TRACKING_V_DEADZONE_DEG` = 5°;
  `TRACKING_H_RAMP_END_DEG`/`TRACKING_V_RAMP_END_DEG` = 25° — chosen
  distinctly from navigation's own 180° ramp end (below), since
  `directionFromBox()`'s screen-space azimuth/elevation only ranges to
  roughly the frame's own half-FOV (~25-32° at the frame edge, given that
  function's `fx=fy=0.8*max(w,h)` pinhole assumption) — reusing a
  180°-scale ramp here would mean volume rarely nearing full even at the
  frame edge.
- **Guiding/walking → Pixie (`ToolDispatcher.steerBeaconAlongPath()`)** —
  the existing bearing computation (`PathPursuit` + `HrtfBeacon.
  directionTo(pose, tx, tz)`) is **completely unchanged** (it already
  correctly reflects `AngleTracker`-bridged heading, since it's derived
  from the already-extrapolated `pose`) — only the output step changed:
  `pixieController.move(LEFT/RIGHT by azimuth sign)` +
  `pixieController.setVolume(gainForDeviation(bearing.azimuthDeg,
  NAV_DEADZONE_DEG=5, NAV_RAMP_END_DEG=180))`, silent within ±5°, full
  volume by ±180° (no pinning at a lower cutoff, per the user's explicit
  spec) — replacing the old fixed-radius `hrtfBeacon.updateDirection(...,
  goalDistanceM)` call (distance-based loudness is gone; Pixie's 4-point
  design has no continuous distance concept). Mute path (no path at all)
  and `reportBeaconDirection()` calls are otherwise unchanged.
- **`gainForDeviation(deg, deadZoneDeg, rampEndDeg): Float`** (new,
  `ToolDispatcher`'s companion object) — the shared deviation-to-volume
  helper both callers above use: silent within `deadZoneDeg`, linear ramp
  to 1.0 by `rampEndDeg`, pinned at 1.0 beyond — same shape validated in
  `pixie_hrtf_app`'s own `gainForDeviation`.
- **Wiring**: `ToolDispatcher`'s constructor — `hrtfBeacon: HrtfBeaconPlayer`
  → `pixieController: PixieController`; `rotationTracker: RotationTracker`
  → `angleTracker: AngleTracker`; `beaconElevationDeg`/`beaconRadiusM`
  params dropped entirely (no equivalent in Pixie's fixed-4-point design).
  `LiveAssistantService.kt` constructs `PixieController`/`AngleTracker(
  maxFeatures=1000, processResolution=480)` in place of
  `HrtfBeaconPlayer`/`RotationTracker` (same Service-scoped lifetime,
  reused across reconnects); its `connect()` still ACCEPTS
  `beaconElevationDeg`/`beaconRadiusM` params (kept only so
  `MainActivity`/`MainViewModel`/`SettingsScreen`'s existing call
  signature + persisted prefs don't need touching) but no longer forwards
  them anywhere — a real, deliberately-accepted dead parameter, same
  "leave unused but Settings-exposed rather than rip out the UI" precedent
  this codebase already uses for these exact two fields. A new
  `angleLumaJob` collects `cameraManager.lumaFlow` into
  `toolDispatcher.feedAngleLumaFrame(...)` (replacing the old
  `feedRotationFrame(jpegBytes)` call), alongside the existing
  `localProcessingJob`'s `frameFlow` collector — both cancelled together in
  `disconnect()`/`onDestroy()`.
- Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, only
  the expected `beaconElevationDeg`/`beaconRadiusM` "never used" warnings
  plus pre-existing unrelated warnings) — **not verified end-to-end on a
  real device from this environment**, same standing caveat as the rest of
  this file's Android work; specifically unverified: whether Pixie's 4
  fixed HRTF points read as clearly distinct left/right/up/down in
  practice, and whether the tracking-mode 2-phase handoff feels natural
  rather than jumpy.

### Memory/tracking fixes: description-as-detection-prompt bug, clear_memory, dedicated vision captioning

Three issues reported directly by the user from a real device session
(`save_memory`ing a plush toy, then trying to track it): descriptions
Gemini Live composed for `remember_object()` were too long/scene-y
("A small, white, fluffy, short-haired dog with pointy ears, sitting
patiently." — pose/setting detail nobody asked for); tracking a
previously-remembered object logged `DetectObject`/`GetEmbedding` with the
literal saved LABEL as the GroundingDINO prompt (e.g. `prompt='Cutie
Patootie'`) instead of the richer stored description, which is a poor
open-vocabulary detection prompt for a proper-noun/toy name; and there was
no way to delete a mislabeled/duplicate memory once saved.

- **Real bug, found by reading the code: `start_tracking`'s `description`
  parameter was declared on the tool schema but never actually read.**
  `ToolDispatcher.dispatch()`'s `"start_tracking"` case only ever forwarded
  `args.optString("target", "")` — even when Gemini correctly followed the
  TRACKING system-prompt instructions and called `start_tracking(target=
  label, description=description)` after a `get_object_from_memory()` hit,
  the description was silently dropped and `target` (the bare label) was
  used as `TrackingBackend.initialize()`'s DetectObject prompt regardless.
  Fixed: `toolStartTracking(target, description = "")` now computes
  `detectionPrompt = description.ifBlank { target }` and threads it through
  a new `onTrackingStateChanged(active, target, detectionPrompt)` callback
  (was `(active, target)`) to `LiveAssistantService.startLocalTracking(
  label, detectionPrompt = label)`, which passes `detectionPrompt` — not
  `label` — into `trackingBackend.initialize(frame, detectionPrompt)`.
  `target`/`state.trackingTarget`/`reportMode()` still use the plain label
  throughout, unaffected — only the actual detection prompt changed.
- **`clear_memory(label)`** (new tool) — `LocalMemoryStore.delete(label)`
  removes all three of a label's files (doc/text-embedding/visual-
  embedding indexes, whichever exist) and returns whether anything was
  actually deleted. The user-facing fix for exactly the kind of duplicate/
  mislabeled save that caused this investigation (a physical object saved
  under two slightly different names, e.g. "Cutie Pie" vs "Cutie
  Patootie") — SYSTEM_PROMPT's MEMORY section now also tells Gemini to
  always reuse the exact same label for the same object and to check
  `list_memory_labels()`/`query_memory()` first if unsure, rather than
  saving a fresh near-duplicate.
- **`GeminiObjectDescriptionClient`** (new, `client/android/app/src/main/
  java/com/tracking/client/live/`) — a dedicated one-shot vision call to
  `gemini-3.1-flash-lite`'s `generateContent` endpoint (same plain-OkHttp-
  REST pattern as `GeminiCorrectionClient.kt`, not the Live WebSocket
  protocol, not a Google SDK), sending the current frame as inline base64
  JPEG + a system prompt constraining the output to ONE short phrase
  (5-10 words: color, shape/size, entity type, at most one or two other
  clearly-visible distinguishing features) — explicitly excluding pose,
  background/setting, and sentence-level narration. `ToolDispatcher.
  toolRememberObject()` now calls this (when configured and a frame is
  available) to REGENERATE the description from the actual image, storing/
  embedding/using-as-detection-prompt that result instead of whatever free-
  form text Gemini Live's own conversational turn originally composed —
  the `description` argument Gemini passes in is now only a fallback,
  used as-is if the vision client is unset or the call fails (network
  error, blank API key). Constructed in `LiveAssistantService.connect()`
  alongside `geminiCorrectionClient`, same null-when-no-key convention (no
  separate Settings field — reuses `geminiApiKey`). `remember_object`'s own
  tool-schema description text was updated to say so explicitly, so Gemini
  doesn't spend effort composing a long description that's likely to be
  discarded anyway.
- Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no
  new warnings) — not verified end-to-end on a real device from this
  environment, same standing caveat as the rest of this file's Android
  work; specifically unverified: whether `gemini-3.1-flash-lite`'s actual
  output reliably stays within the requested short-phrase format across
  real objects.

**Unrelated server-side fix from the same investigation**: `server/tools/
detector.py`'s `GroundingDINODetector.detect_all()` crashed with
`KeyError: 'text_labels'` — `_post_process_grounded()` already layers a
call-signature fallback for the `transformers` `box_threshold`→`threshold`
kwarg rename (<=4.46.x vs >=5.x), but the RETURN dict's label key was
renamed in the same version bump (`"labels"` → `"text_labels"`) and only
the kwarg side was actually handled. Fixed with the same fallback pattern:
`labels = results.get("text_labels", results.get("labels"))`. `detect()`
(used by `TrackingBackend`'s local-ORB-tracking init loop, not
`detect_all()`) never read labels at all and was unaffected.

### remember_object's own crop-then-embed pipeline (confirms, not new)

Worth spelling out explicitly since it came up directly from the user:
`toolRememberObject()` already does exactly "get the description, run
GroundingDINO with it to find the object's box, then embed a CROP of that
box" — not the whole frame. `detectAll(finalDescription, frame)` runs
GroundingDINO over the full frame with the (now short, vision-generated —
see "Memory/tracking fixes" above) description as the open-vocab prompt,
picks the highest-scoring box, then `embedBox(box, frame)` sends that box
to `PerceptionService.AnalyzeFrame(EMBED)` — server-side,
`DINOv2Embedder.get_embedding()` (`server/tools/embedder.py`) does the
actual `crop = frame[y1:y2, x1:x2]` before running DINOv2, so the stored
visual embedding is already of the cropped object region, never the whole
scene. This was already true before this round of fixes — what changed is
only that the DETECTION PROMPT is now a short, dedicated-vision-generated
phrase instead of whatever long sentence Gemini Live composed, which is a
better GroundingDINO prompt and should improve box quality too.

### Reading-mode fixes: scanned frame fed to Gemini Live, short/long-text gate, scan-complete cue, session-ready greeting

Four more direct user requests from the same round:

- **Scanned frames now reach Gemini Live as visual context.** Previously,
  reading mode's OCR path (`toolScanCurrentView()`/live reading) never sent
  the camera frame to Gemini at all — only the extracted TEXT went to
  Gemini (indirectly, via `read_aloud`'s spoken TTS or `get_reading_section`
  results); Gemini itself never actually SAW what was scanned. Both
  `toolScanCurrentView()` and the live-reading OCR worker now call
  `sendVideoFrame(frame)` right after acquiring a sharp frame — plain
  `realtimeInput` video (no `turnComplete`, same contract
  `sendWalkingAmbientFrame()`/the hazard-warning frame send already use),
  so it never forces a response on its own, just gives Gemini buffered
  visual context for whenever it next speaks (e.g. the user follows up with
  "what does this say"/"what am I looking at"). **Follow-up, requested
  directly by the user: "clear [this context] on end reading session, if
  possible."** Checked against this project's own Gemini Live API
  reference — the `BidiGenerateContent` protocol has no operation to
  selectively purge already-sent `realtimeInput` video frames from a live
  session's context (only automatic context-window compression and full
  session resumption, neither a targeted "forget these specific frames").
  The "if possible" is realized as a soft, prompted clear instead:
  `toolExitReadingMode()` (`ToolDispatcher.kt`) now sends a `[SYSTEM]` note
  telling Gemini to disregard the reading session's visual context now
  that its buffer was discarded — same convention this codebase already
  uses for every other behavior signal Gemini needs to react to — gated on
  there having been anything to clear (`chars_discarded > 0`) so an
  already-empty reading session doesn't send a pointless note.
  `toolStopLiveReading()` is unaffected — it only PAUSES capture and
  explicitly keeps the buffer, so there's nothing to tell Gemini to
  forget there.
- **`read_aloud()` no longer always speaks the whole capture immediately.**
  Requested directly: a short capture (a label, a sign — under
  `READ_ALOUD_IMMEDIATE_MAX_WORDS`=40 words) is still read straight away,
  same as before; a LONGER capture (a full page) is now saved to the
  reading session instead — `toolReadAloud()` reads `toolScanCurrentView()`'s
  own `new_text` result (previously discarded/unused), word-counts it, and
  if it's at/above the threshold, returns `status: "stored_long_text"`
  WITHOUT calling `speakSentencesFrom()` — SYSTEM_PROMPT's READING MODE
  section now tells Gemini to offer to read it (`continue_reading()`) or
  pull a specific part (`get_reading_section()`) instead of narrating it
  itself. The gate only looks at THIS scan's own new material, not the
  whole historical buffer — a scan that found nothing new (a duplicate
  reread) still falls through to the pre-existing "speak whatever's unread
  so far" behavior, preserving the fix `read_aloud()` already had for
  "scope='new' silently did nothing on a reread."
- **A "pop" confirmation tone once a scan's full pipeline is actually
  done.** `playScanCompleteCue()` (new, `ToneGenerator.TONE_PROP_ACK`,
  80ms — reuses the same shared `toneGenerator` instance `playDeadEndAlert()`
  already manages, no bundled audio asset) — since these users can't see
  the screen, an audio cue that OCR+correction+storage has genuinely
  finished (not just that a frame was grabbed) is real feedback.
  `CorrectionJob` gained an `onComplete: (() -> Unit)?` field, invoked in
  the correction worker's `finally` block (fires on success OR failure —
  staying silent forever after a correction error would be worse than
  confirming a possibly-uncorrected completion) — this is the actual
  "correct, store done" signal when a correction is queued.
  `toolScanCurrentView()` passes `onComplete = { playScanCompleteCue() }`
  when a correction job gets queued, and calls `playScanCompleteCue()`
  immediately otherwise (correction disabled, or nothing new to correct —
  no later completion event will ever arrive for those cases), and pops on
  every explicit user-triggered scan regardless of outcome (new/stitched/
  updated/duplicate — the user explicitly acted and deserves feedback every
  time). The live-reading OCR worker uses the same mechanism but gates the
  pop to `block != null && kind != "duplicate"` only — that loop runs
  UNATTENDED every `LIVE_READING_INTERVAL_MS` and would otherwise pop
  constantly while the camera just holds steady on an already-seen page.
- **Gemini Live now announces itself ready.** `LiveAssistantService`'s
  `LiveServerEvent.SetupComplete` handler now calls `client.sendSystemNote(
  "[SYSTEM] The session just connected and is ready. Briefly say you're
  ready to help...")` right after logging/appending the connected message —
  same `[SYSTEM]`-event convention (CORE RULES: respond to these
  immediately) already used for incoming-call/SMS notes;
  `sendSystemNote()`'s `turnComplete=true` forces an actual spoken turn
  rather than sitting buffered, so the user gets an audible "ready" cue
  instead of silence until they happen to speak first.
- Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no
  new warnings) — not verified end-to-end on a real device from this
  environment, same standing caveat as the rest of this file's Android
  work.

### `test_module/edge_mock_app/` — added Mode 2: test a REAL edge device, acting as client/android would

First investigated (see the standing findings below, still accurate),
then corrected per direct clarification from the user once the
investigation surfaced a role confusion: `edge_mock_app` originally only
implemented `EdgeMockZmqServer.kt`, which plays the role of a MOCK EDGE
DEVICE (binds ports, generates fake mic/camera data) — useful for testing
`client/android`'s own `EdgeZmqTestClient` without real Pi hardware, but
the OPPOSITE of what the user actually needed: they already have REAL
edge hardware (mic + camera + speaker), and want a lightweight stand-in
for `client/android`'s side of the conversation to validate that hardware
BEFORE wiring it into the full, heavy tracking app.

**New Mode 2, added alongside the original (now "Mode 1") — a CLIENT that
connects OUT to a real edge device's bound ports and runs a short
concurrent load test:**

- **`EdgeDeviceClient.kt`** (new) — direct duplicate of `client/android`'s
  own `EdgeZmqTestClient.kt` (separate standalone app/process, same
  "duplicate rather than share a module" precedent this codebase already
  uses elsewhere) — connects to `mic_out`/`frame_out` (PULL) and
  `audio_in` (PUSH) on a given host at the SAME ports/wire framing
  (`[8-byte seq][8-byte timestamp]` header + raw payload, two ZMQ message
  parts) already established by `EdgeMockZmqServer.kt`/`mock_edge_server.py`
  — this is the real, already-agreed protocol a physical edge device
  speaks, not something invented for this pass.
- **`LoadTestRunner.kt`** (new) — runs all THREE channels concurrently for
  a configurable duration (default 4s, per direct spec): continuously
  PUSHes mock "audio to render" to the edge (mono 24kHz PCM16, 512-sample
  chunks) while concurrently consuming whatever `mic_out`/`frame_out` the
  real edge device sends back, then reports counts/bytes/measured rates
  plus structural type-validation (mic: even-length PCM16 bytes; frame:
  real JPEG SOI/EOI magic-byte framing). **Deliberately does NOT assert a
  fixed target fps/sample-rate** — confirmed directly with the user that
  the real edge device's own camera resolution/rate isn't known ahead of
  time, so this measures and reports whatever the hardware actually does
  rather than failing against a guessed number.
- **Audio-out format is a deliberate, documented choice, not a guess**:
  mono 24kHz PCM16 was picked because it's the ONE audio format that's
  ACTUALLY wired in `client/android` today (`LiveAssistantService.
  emitAudio()` forwards Gemini's own raw voice PCM to `EdgeDevice.
  audioFlow`) — the HRTF/Pixie steering cue (stereo, 44.1kHz) does NOT
  reach the edge device in production at all yet (see the standing
  findings below), so simulating that format here would test against a
  contract that doesn't exist. If/when HRTF output does get wired to the
  edge device, this generator (and the real `EdgeDevice.audioFlow`
  contract itself) will need revisiting — mixing a mono 24kHz stream and a
  stereo 44.1kHz stream into one raw-bytes channel isn't a well-defined
  operation on its own; that would need either two separate channels or a
  documented mixed/resampled format, a real open design question not
  decided in this pass.
- **UI** (`activity_main.xml`/`MainActivity.kt`): a second section below
  the existing Mode 1 controls — edge device IP input, duration input
  (seconds, default "4"), a Start/Stop button, and a live-updating (300ms
  poll, same cadence Mode 1's stats poller already uses) status view that
  becomes the final report once the duration elapses.
- Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL) —
  not verified against real edge hardware from this environment (none
  available here), same standing "not device-verified" caveat as the rest
  of this project's Android work.

**Standing findings from the original investigation (still accurate,
describes why Mode 2 was needed and what it does/doesn't fix)**:

- `LocalEdgeDevice.kt` (`client/android/app/src/main/java/com/tracking/
  client/edge/`) is still an in-process, no-op stub — `connect()`/
  `disconnect()` are literally empty; `client/android` has no real remote
  transport of its own yet (Mode 2 tests the FUTURE real hardware
  standalone, it doesn't wire it into `client/android` itself).
- Real data shapes, for reference: camera frames are JPEG quality 50,
  downscaled to a 640px long edge, emitted at a MODE-DEPENDENT variable
  rate (`frameIntervalMs`/`scanIntervalMs`/`walkingIntervalMs`, or once per
  `recentBufferMs` tick otherwise) — not a fixed fps. Mic input has no
  edge-device path at all in `client/android` today — capture is local
  (`ContinuousVadRecorder.kt`, 16kHz mono) straight to Gemini;
  `EdgeDevice`'s interface has no mic-input flow at all (a real gap if a
  physical edge device's mic is ever meant to replace the phone's own).
- **Still open, not addressed by Mode 2** (Mode 2 only tests the edge
  device in isolation): actually wiring a real edge device into
  `client/android` needs (1) a real `EdgeDevice` implementation with an
  actual network transport (`LocalEdgeDevice` is the only implementation
  today, and it's in-process), (2) a mic-input flow added to the
  `EdgeDevice` interface (doesn't exist), and (3) a decision on how
  Gemini's voice PCM and the HRTF/Pixie cue both reach the edge device's
  speaker — two different formats today, no combined contract defined.

### Memory label-name lookup bug + tracking-mode hand-guidance bugs (two real reported bugs, both fixed)

Two distinct real bugs, both reported directly by the user from a live
device session and root-caused by reading the code (not guessed at).

**1. Memory lookup missed the "query IS the label" case entirely.**
Reported: right after `remember_object(label="Cutie Patootie",
description=...)` saved successfully, saying "track Cutie Patootie"
immediately got "can't be found." Root cause, confirmed via the server's
own `Embed` call logs: `get_object_from_memory(query)`/`query_memory(
question)` only ever did SEMANTIC vector search — embed the query text via
MiniLM, cosine-compare against each label's stored DESCRIPTION text. A
proper-noun label like "Cutie Patootie" has near-zero semantic similarity
to its own stored description ("small yellow plush bird...") — they share
no lexical/semantic content at all, even though the label IS literally
what's being asked about, so this class of query was essentially
unfindable by design. A second, compounding bug found in the same
investigation: `LocalMemoryStore.listLabels()` derived label names from
the sanitized FILENAME (`safeName()` maps "Cutie Patootie" → file
`Cutie_Patootie.json`, spaces→underscores) instead of reading each doc's
own stored `"label"` field — so even a smarter lookup comparing against
`listLabels()` would have compared against the wrong (underscore-joined)
string.

- **`LocalMemoryStore.listLabels()`** now reads each doc's own `doc.
  optString("label", ...)` instead of stripping `.json` off the filename —
  falls back to the filename only if a doc fails to parse. **A third bug,
  found from a follow-up report, was in this same function's file
  FILTER**: it only excluded `*.vec.json` (the text-embedding sidecar
  index), not `*.objemb.json` (the visual-embedding sidecar `remember_
  object()` creates whenever it captures a DINOv2 reference) — so saving
  one object with a visual capture (e.g. label "cutie") produced TWO
  entries in `listLabels()`: the real "cutie" doc, and a bogus
  filename-derived entry from `cutie.objemb.json` slipping through the
  filter. New private `isDocFile(fileName)` excludes both sidecar suffixes
  explicitly.
- **`LocalMemoryStore.findLabelsMatching(query)`** (new) — case-insensitive
  substring match (either direction) of the query against every real
  label. `toolGetObjectFromMemory()` now tries this FIRST: a single match
  short-circuits straight to that label (skipping the Embed round trip
  entirely — also fixes the double `Embed` call seen in the reported logs,
  since Gemini's own `get_object_from_memory` call no longer needs one at
  all for this case); multiple matches return `ambiguous: true` instead of
  guessing; zero matches fall through to the original semantic search,
  unchanged. `toolQueryMemory()` got the same treatment, additively —
  label hits are prepended to (and deduped against) the semantic results,
  since a question naming a saved label deserves that memory even if its
  wording doesn't semantically resemble the stored text.

**2. Tracking mode's hand-guidance was broken two separate ways.**

- **Spoke nonsense with no hand in view.** `ToolDispatcher` had its OWN
  periodic (every `TRACKING_GUIDANCE_INTERVAL_MS`=8s) spoken-guidance
  trigger (`runPeriodicTrackingGuidance()`) that sent the current frame +
  a "guide their hand" instruction to Gemini UNCONDITIONALLY — no check
  for whether a hand was actually visible anywhere in frame. This
  duplicated a SECOND, already-correct mechanism in
  `LiveAssistantService.kt`'s own frame collector
  (`trackingGuidanceLastAtMs`/`trackingGuidanceIntervalMs`=5s), which only
  fires once BOTH the target object AND a MediaPipe-detected hand box are
  visible together, sending real box coordinates instead of a raw frame
  (cheaper, and gives Gemini exact spatial data instead of asking it to
  eyeball an image). The `ToolDispatcher` copy was deleted outright — not
  fixed in place — since the correct version already existed: removed
  `trackingGuidanceJob`/`lastTrackingGuidanceAtMs`/
  `startTrackingGuidanceTicks()`/`stopTrackingGuidanceTicks()`/
  `runPeriodicTrackingGuidance()`/`TRACKING_GUIDANCE_INTERVAL_MS`/
  `TRACKING_GUIDANCE_POLL_MS` and their call sites in `toolStartTracking()`/
  `toolStopTracking()`/`stopActiveModes()`.
- **The Pixie audio cue guided VIEW direction, not the hand, despite the
  whole point of tracking mode being hand guidance.** `updateTrackingPixie()`
  computed the tracked object's position relative to the FRAME CENTER
  (`HrtfBeacon.directionFromBox()`) — i.e. "which way to turn/look to
  center the object in view," not "which way to move your hand toward
  it." The hand's own position was never part of the computation at all.
  Fixed: **`HrtfBeacon.directionBetweenPoints(targetX, targetY, refX,
  refY, frameWidth, frameHeight)`** (new) generalizes the same pinhole-FOV
  azimuth/elevation approximation to an ARBITRARY reference point, not
  just frame center — `directionFromBox()` is now defined as the special
  case `directionBetweenPoints(centerX, centerY, frameWidth/2,
  frameHeight/2, ...)`, kept for any future caller that genuinely wants
  view-direction-relative-to-center (tracking mode no longer uses it).
  `updateTrackingPixie()`'s signature changed to `(objectVisible,
  objectCenterX, objectCenterY, handVisible, handCenterX, handCenterY,
  frameWidth, frameHeight)` and now calls `directionBetweenPoints(object,
  HAND)` — muting whenever EITHER the object or a hand isn't currently
  visible (there's nothing meaningful to steer without both). `LiveAssistantService.kt`'s
  frame collector was reordered so hand detection (MediaPipe `HandTracker`)
  runs BEFORE the tracking-object update block instead of after — pure
  computation only hoisted (the `_uiState` merge that publishes hand data
  to the UI stays in its original position, unchanged, to avoid a
  `guidanceData.copy()` ordering hazard where the object-track branch's
  own `it.copy(guidanceData = guidance)` would otherwise clobber
  hand fields set moments earlier) — so the same frame's hand center is
  available in time to feed `updateTrackingPixie()`.

Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no new
warnings) — not verified end-to-end on a real device from this
environment, same standing caveat as the rest of this project's Android
work.

### Renewal target-switch bug — the hand-overlap skip above wasn't enough

Real bug, reported directly by the user with a live dashboard screenshot:
right as their hand reached the tracked object, the tracking box jumped to
a completely different toy elsewhere in frame, while the real target sat
untouched under their hand. The skip-while-occluded fix above only guards
SCHEDULING a new `renewal()` call — it does nothing about a renewal that
was ALREADY launched (async, a full gRPC round trip) just before the hand
started overlapping, nor did `renewal()` ever check whether the detection
it got back was even spatially consistent with the object already being
tracked. `renewal()`'s only acceptance checks were a detection-score floor
(0.2) and a fairly loose (0.4) cosine-similarity check against the stored
embedding — nothing stopped a visually-similar-but-different object
elsewhere in frame from passing both, especially once the real target's
own embedding read poorly because a hand was starting to cover it.

Two guards added to `renewal()` (`TrackingBackend.kt`):

1. **Spatial consistency, the primary fix**: the new detection's raw box
   must `boxesOverlap()` (existing AABB helper) with `lastBox` (the
   CURRENT tracked position) before anything else is even considered — a
   genuine re-identify of the SAME physical object should never jump to
   an unrelated part of the frame, regardless of how the prompt/embedding
   checks scored it. Rejects and logs, doesn't touch the reference.
2. **A second, later hand-overlap check**: new `@Volatile lastHandBoxXyxy`
   field, overwritten every `update()` call (which runs every frame) —
   right before `renewal()` actually commits the reference swap, it
   re-checks the new box against this LATEST hand position (not the hand
   box from when the renewal was originally scheduled), closing the race
   where the hand starts covering the target while an already-in-flight
   renewal is still out to the server.

Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no new
warnings — only the same pre-existing warning list, including the
already-known `GlobalScope` "delicate API" one at `renewal()`'s own launch
site) — not verified end-to-end on a real device from this environment,
same standing caveat as the rest of this project's Android work.

### Tracking-init confidence gate + left/middle/right initial position, no clock positions

Three more direct user requests/reports from the same round:

- **`TrackingBackend.initialize()` accepted almost any detection.**
  Real bug, confirmed via a live log: `DetectObject` returned score=0.383
  for a wrong/weak match and it was accepted as the tracking target
  outright — the check was only `detection.score <= 0f`, i.e. rejected
  literally nothing except a hard zero. Fixed with a real threshold,
  `INIT_CONFIDENCE_MIN = 0.45f` — deliberately only for this ONE-TIME
  "is this even the right object" gate, not `update()`'s/renewal's own
  ongoing thresholds (10 matches / 0.2 score), which stay lenient on
  purpose so an ALREADY-confirmed target isn't lost over one weak frame.
- **One-time left/middle/right position, replacing clock positions for
  tracking guidance entirely.** Previously there was no initial
  announcement at all once tracking found the object but before a hand
  appeared — the first spoken guidance only ever came from the hand+object
  periodic mechanism (see "Memory label-name lookup bug + tracking-mode
  hand-guidance bugs" above), so with no hand in view yet, the user got
  silence. Now, the FIRST time the object becomes visible each tracking
  session, `LiveAssistantService`'s frame collector checks: if no hand is
  in frame yet, it sends ONE `[SYSTEM]` note with the object's rough
  position (`left`/`middle`/`right`, from which third of the frame
  `track.centerX` falls in) and instructs Gemini to say it plainly, once.
  A new one-shot flag, `trackingInitialPositionAnnounced` (reset in
  `startLocalTracking()`), guarantees this fires at most once per session
  — marked "announced" on the object's first sighting regardless of
  whether a hand was already visible at that exact moment (so it can never
  fire again later just because a hand happened to be briefly absent on
  the very first frame). The ongoing hand+object directional guidance
  system note (unchanged mechanism, still every ~5s) was reworded to
  explicitly require plain words ("move left", "a bit right", "up",
  "down", "closer") and explicitly forbid clock positions — the general
  VOCAL STYLE section of SYSTEM_PROMPT recommends clock positions for
  spatial answers elsewhere, and without an explicit override Gemini was
  applying that here too, which the user didn't want for hand-guidance
  specifically. SYSTEM_PROMPT's TRACKING section documents both behaviors
  (the one-shot position note and the "never clock positions here" rule)
  directly.

Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no new
warnings) — not verified end-to-end on a real device from this
environment, same standing caveat as the rest of this project's Android
work.

### Tracking guidance kept saying "move" with the hand already on target — root cause + fix

Real bug, reported directly by the user: the periodic hand+object spoken
guidance kept telling the user to "move" even once their hand was already
on the target. Root cause, found by rereading the actual system note text
sent to Gemini Live: it handed Gemini the RAW box coordinates for both the
target and the hand (`"Target box=[...], hand box=[...]. Give brief
directional guidance..."`) and asked GEMINI to work out arrival/direction
itself from those two number lists — LLMs are unreliable at that kind of
precise box-overlap arithmetic purely from raw coordinates in a text
prompt, especially under a "keep it brief" constraint, so it would often
guess "move" regardless of the real geometry.

**Fixed by computing the judgment client-side instead of asking Gemini to
derive it** — the same approach Pixie's own audio cue already uses
(`HrtfBeacon.directionBetweenPoints`), just for the spoken channel too.
`LiveAssistantService.kt`'s periodic guidance block now: (1) checks AABB
overlap between the hand and target boxes directly (`obj[0] < handBox[2]
&& ...`) — real overlap, not just "centers are close"; (2) if not
overlapping, computes `dx`/`dy` between box centers, classifies each axis
against a deadzone (5% of frame dimension) and a "a lot" vs "a bit"
magnitude threshold (25% of frame dimension), and builds a short direction
description (e.g. `"left (a bit) and up (a lot)"`); (3) sends Gemini a
`[SYSTEM]` note stating the ALREADY-COMPUTED verdict directly — either
`"ARRIVED — say a short confirmation"` or `"move <computed direction> —
do not recompute from coordinates"` — never raw box numbers. This removes
the geometric-reasoning burden from the LLM turn entirely; Gemini's only
job is to phrase the already-known answer briefly.

Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no new
warnings) — not verified end-to-end on a real device from this
environment, same standing caveat as the rest of this project's Android
work.

### Tracking re-identify cadence + skip-while-occluded-by-hand

Two direct user requests:

- **Re-identify interval 1s → 4s.** `TrackingBackend`'s `renewalIntervalMs`
  constructor default changed from `1000L` to `4000L` (no explicit
  override at its one construction site, `LiveAssistantService.kt`'s
  `trackingBackend by lazy { TrackingBackend(grpcManager) }`, so this is
  the effective live value).
- **Skip the renewal `DetectObject` call while the hand overlaps the
  target.** A hand reaching for/holding the object typically occludes it,
  so re-identifying right then is likely to see a wrong/blocked view —
  wasted server round trip, and a real risk of the renewal's own
  visual-similarity check (`isSimilar()`) rejecting a good reference based
  on a bad, hand-obscured frame. `TrackingBackend.update()` gained a
  `handBoxXyxy: List<Float> = emptyList()` param (from MediaPipe hand
  detection — `LiveAssistantService.kt`'s frame collector, same hoisted
  `handBox` `updateTrackingPixie()` already consumes) and a new private
  `boxesOverlap(a: FloatArray, b: List<Float>)` AABB check — any portion
  of overlap counts, not a minimum-fraction threshold. The renewal-trigger
  condition became `elapsed > renewalIntervalMs && !boxesOverlap(box,
  handBoxXyxy)`. Deliberately does NOT advance `lastRenewalMs` when
  skipped this way, so renewal fires as soon as the hand moves off the
  target and the interval has already elapsed, rather than waiting a full
  extra interval on top.

Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no new
warnings) — not verified end-to-end on a real device from this
environment, same standing caveat as the rest of this project's Android
work.

### SCAN mode: real queued ingestion (not drop-to-latest) + WALKING cold-start warm-up gate

Two related but independent fixes, both requested directly by the user
after asking how SCAN mode actually works.

**1. SCAN now queues every frame instead of dropping stale ones.** Root
cause of "the server doesn't seem to keep processing after I stop
scanning, and semantic mapping barely finds anything": `UpdateMapping`'s
drop-to-latest mailbox (`_latest_only_chunks()`, see that section above)
and `ToolDispatcher`'s `Channel.CONFLATED` outbound channel both silently
discard any frame arriving while the previous one is still being
processed — correct for WALKING/GUIDING (fresh pose beats a complete
backlog for real-time steering), but exactly backwards for SCAN, which
wants EVERY frame processed, in order, buffering while busy and continuing
to drain after the user stops — not silently losing most of what was
captured. With most frames dropped, there was rarely a real backlog left
to drain by the time the client disconnected (finalize looked instant),
and semantic tagging (`SemanticMapper`/`FrameTagger` — already live
per-frame during scan, see "Semantic mapper adapted to..." above, this was
NOT missing code, just rarely fed enough frames to do anything visible)
had far fewer frames to work with than the user's actual camera sweep.
- **Client (`ToolDispatcher.kt`)** — `startMappingStream()`'s outbound
  channel is now `Channel.UNLIMITED` specifically when `state.mode ==
  "scanning"`, `Channel.CONFLATED` otherwise (unchanged for WALKING/
  GUIDING).
- **Server (`mapping_servicer.py`)** — `UpdateMapping` peels off the
  request stream's first chunk directly (never dropped either way) to
  learn `session_mode` before choosing how to consume the rest:
  `SessionMode.SCAN` bypasses `_latest_only_chunks()` entirely, riding
  gRPC's own internal request-iterator queue instead (already documented,
  in `_latest_only_chunks()`'s own docstring, as providing exactly this
  "buffer and work through the backlog in arrival order" behavior) —
  WALKING/GUIDING keep the existing mailbox unchanged. No new queue
  primitive was needed — this is purely "stop throwing away what gRPC
  already buffered for us," for SCAN only.
- **Practical effect**: `stopMappingStreamAndAwait()` (already called by
  `toolStopScan()`, already documented as "waits through the server's
  finalize") now actually has a real backlog to wait through when one
  exists — this mechanism already existed, it just had nothing meaningful
  to wait for before, since both ends had already thrown the backlog away
  by the time the client disconnected.
- **Not addressed in this pass**: whether `GEMINI_API_KEY` is actually set
  in the user's server environment — `grpc_server.py` silently disables
  the whole `SemanticMapper`/landmark-extraction path (logs "Mapping
  SemanticMapper init failed" or never logs "ready" at all) if it isn't,
  which would ALSO explain "I don't see semantic mapping happening"
  independent of the queueing bug above. Worth checking server startup
  logs for `[SERVER] Mapping SemanticMapper ready` before assuming the
  queueing fix alone resolves it.

**2. WALKING mode's cold-start lag — a warm-up gate, reintroduced in a
different form.** A similar mechanism existed once before (see "Server-
planned walking path"'s "Client-side warm-up" note) and was deliberately
removed as unneeded complexity — the user's real-device report now
confirms it WAS needed: `toolStartWalking()` used to activate everything
(local avoidance ticks, Pixie, PDR, the "walking started" report)
IMMEDIATELY, before the server had processed a single frame — so a real
device would sit in a "walking mode active" state with a stale/absent
route and a beacon that had nothing to steer by, for however long the
server's first RTAB-Map + occupancy round trip actually took (worse
straight after a cold GPU/model start).

- **`feedMappingFrame()`** now gates on a new `walkingReady`/
  `walkingFirstFrameSent` pair: while walking and not yet ready, it sends
  EXACTLY ONE frame (the first one offered) and silently drops every
  subsequent one — no point spending camera/encode work or server GPU
  time on more frames while the server is still busy on that first
  round trip. Unaffected for WALKING once ready, and unaffected for
  GUIDING/SCANNING entirely (this gate is walking-only).
- **`toolStartWalking()`** now only opens the mapping stream and returns a
  `"walking_starting"` status — none of ticks/Pixie/PDR/the "walking
  started" report happen here any more.
- **`activateWalkingOnceReady()`** (new) — fired from the mapping-stream
  collector the instant the FIRST real `MappingUpdate` for the session
  arrives (i.e., the server has now genuinely processed at least one
  frame end-to-end and has a real pose/grid). Only NOW does it start
  `startLocalAvoidanceTicks()`/`pixieController.start()`/
  `pdrStepEstimator.start()`, call `onGuidanceUpdate("walking", ...)`/
  `reportMode("walking")`, and — the actual "notify the user" step,
  matching CORE RULES' "[SYSTEM] events respond immediately" convention
  — send a `[SYSTEM]` note telling Gemini walking is now genuinely active,
  so the spoken "walking mode started" the user hears is honest, not
  premature. `walkingReady`/`walkingFirstFrameSent` are reset in
  `toolStartWalking()`, `toolStopWalking()`, and `stopActiveModes()` (the
  last one covers switching away from walking mid-warm-up before it ever
  became ready).
- Deliberately reuses the SAME persistent `startMappingStream()`/
  `feedMappingFrame()` path WALKING already shares with GUIDING/SCANNING —
  unlike the old (removed) design, there's no separate throwaway/discard
  stream for the warm-up frame; the first frame IS the first frame of the
  real session.

Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL) and
`python3 -m py_compile server/services/mapping_servicer.py` — not verified
end-to-end on a real device/live server from this environment, same
standing caveat as the rest of this project's work. Specifically
unverified: real achieved server-side fps for a queued scan (the user's
own "8fps" figure is the CLIENT's configured Scan FPS send rate — see
"Client-side frame selection"'s `scanIntervalMs`/Settings "Scan FPS"
field — not something newly enforced server-side by this change; the
server just no longer discards whatever arrives faster than it can keep
up with).

### Mode-specific tracking-loss reset thresholds + WALKING route-change hysteresis

Two more direct user requests, server-side (`scan_session.py`/
`mapping_servicer.py`).

**1. GUIDING now gets its own (much longer) tracking-loss reset.**
Previously the consecutive-RTAB-Map-tracking-loss reset (`_reset_cloud_
locked()` — RTAB-Map + local grid wiped, `reset_occurred` sent to the
client) only ever applied to `pure_walking` (`SessionMode.WALKING`
specifically) — GUIDING had no reset check at all, gated out by the code
checking `if pure_walking:` rather than the broader `walking_lite` (true
for both WALKING and GUIDING). Now both get it, at different thresholds:
`PURE_WALKING_LOST_RESET_S` (0.5s, unchanged) for WALKING, new
`GUIDING_LOST_RESET_S` (2.0s) for GUIDING — resetting an in-progress route
over a brief tracking hiccup is far more disruptive than resetting
walking's short-lived, cheap-to-rebuild local grid, so GUIDING gets more
patience before giving up and starting over. SCAN (`walking_lite=False`)
still never resets on tracking loss at all — its reconstruction is too
valuable to nuke over a brief hiccup. `mapping_servicer.py`'s `reset_
occurred`-forwarding branch was widened from `pure_walking and session.
last_reset_occurred` to `walking_lite and session.last_reset_occurred` to
match, so GUIDING's client-side route/beacon invalidation on reset (the
same handling WALKING already had) now actually fires.

**2. WALKING's planned route no longer changes on every single update
(SUPERSEDED — see "Main-path/sub-path joint navigation" below).** Reverted
per direct user feedback in the very next round of this feature: "not that
we retain the whole path for 6s but we retain the joints on the path" —
the 6s server-side dwell/hysteresis this point describes locked the
ENTIRE route in place for a fixed duration, which wasn't the right
granularity; joint-level retention became the client's job instead (see
below). `mapping_servicer.py`'s WALKING branch is back to calling
`find_natural_path()` fresh on every update, no dwell state at all —
`_WALKING_HEADING_REEVALUATE_DEG`/`_WALKING_HEADING_DWELL_S`/
`_WALKING_REPLAN_NEAR_END_M`/`_angle_diff_rad()`/the three
`_walking_lock_heading_rad`/`_walking_diverge_since_wall`/
`_walking_last_path` dicts were all removed outright (not kept
unreferenced — this was reverted within the same short span of work, not
a design left behind for possible future reuse). Kept here for history —
the underlying jitter problem this originally addressed is now solved
differently (see below), not reintroduced.

Previously `find_natural_path()` was called fresh every update using
whatever the CURRENT heading happened to be — a brief, incidental head
turn (checking a doorway, glancing at a noise) could reroute the beacon
mid-stride, per a direct user report. Fixed with a dwell-hysteresis gate
in `mapping_servicer.py`, keyed by `location_id` (`_walking_lock_heading_
rad`/`_walking_diverge_since_wall`/`_walking_last_path`, all cleared at
stream-open and on `reset_occurred`, same precedent as `_last_full_
bounds`): the route only actually changes once the CURRENT heading has
diverged from the heading the EXISTING route was planned toward by more
than `_WALKING_HEADING_REEVALUATE_DEG` (30°), continuously, for at least
`_WALKING_HEADING_DWELL_S` (6s) — a glance back within that window cancels
the divergence timer and the existing route is kept untouched. Below the
30° deadzone, the timer is reset outright (not just paused) — matching the
same "a glance back cancels it" behavior the older, reverted corridor-lock
design used for its own dwell timer. Reusing an old-but-still-valid route
is safe because `PathPursuit` (client-side) already projects the CURRENT
position onto whatever path it's given and advances a look-ahead from
there — a route computed several seconds and a few steps ago is still
perfectly walkable, not stale in any way that matters. Two additional
forced-replan triggers, both independent of heading: no route exists yet
for this session (first update, or the last attempt found no path at all
— always worth retrying immediately rather than waiting out a dwell period
to even check for an opening), and the user is nearly at the end of the
currently-locked route (`_WALKING_REPLAN_NEAR_END_M`, 1.0m) — otherwise
someone walking straight exactly as directed (never diverging in heading)
would eventually run out of route with nothing left to trigger an
extension. New `_angle_diff_rad()` helper computes the shortest signed
angular difference, wrapped to `(-π, π]` — verified in isolation against
the wraparound case specifically (170° vs. -170° must read as a 20°
difference, not 340°), since a naive subtraction gets this wrong exactly
at the ±180° boundary.

Verified via `python3 -m py_compile scan_server/scan_session.py server/
services/mapping_servicer.py` and an isolated check of `_angle_diff_rad`'s
wraparound correctness — not verified end-to-end against a live server/
device from this environment, same standing caveat as the rest of this
project's work. Specifically unverified: whether 30°/6s/1.0m are the right
real-world tuning values, or whether the hysteresis feels natural in
practice rather than sluggish (a route stuck 6s behind a genuine, decisive
turn) or still-jittery (if 30° turns out too tight for normal walking-gait
head/camera wobble).

### Settings volume sliders — Gemini Live voice / other sound

Requested directly by the user: Gemini Live's own spoken voice played
noticeably louder than Pixie's cue, with only Pixie's own "Cue Volume"
slider to balance against. Two new independent master-volume sliders,
same 0–1f / 9-step convention as the existing Cue Volume slider:

- **`StreamingAudioPlayer.setVolume(gain)`** (new) — plain `AudioTrack.
  setVolume()`, applied both on `start()` and retroactively via
  `setVolume()` so a mid-session Settings change takes effect on the next
  `connect()` without needing a fresh `AudioTrack`. Covers Gemini's own
  spoken reply only.
- **`ReadingTtsPlayer.setVolume(gain)`** (new) — passed as a `Bundle`
  (`TextToSpeech.Engine.KEY_PARAM_VOLUME`) to every `engine.speak(...)`
  call in `fillQueue()`, instead of `null`. Covers reading-mode TTS.
- **`PlaybackService`** — reads `other_sound_volume_bits` directly from
  `getSharedPreferences("tracking_prefs", ...)` at the top of
  `onStartCommand()` (this Service has no other channel to receive a
  live value through — it's launched via plain Intent, not bound) and
  applies it via `newPlayer.volume = volume` before
  `setMediaItem`/`prepare`/`play`. Covers music/radio/resolved-YouTube-
  stream playback (ExoPlayer). Does **not** cover the embedded YouTube
  IFrame player itself — a WebView with its own independent volume, not
  reachable from here.
- **`SettingsViewModel`/`SettingsScreen`** — new `geminiVoiceVolume`/
  `otherSoundVolume` fields (`gemini_voice_volume_bits`/
  `other_sound_volume_bits` in `tracking_prefs`, default 1f each), two new
  sliders right after the existing Cue Volume slider. `MainActivity.kt`'s
  `onConnect` lambda and `MainViewModel.connect()`/
  `LiveAssistantService.connect()` all extended with the two new trailing
  `Float` params, applied via `streamingPlayer.setVolume(geminiVoiceVolume)`/
  `readingTts.setVolume(otherSoundVolume)` right after `pixieController.
  cueVolume = cueVolume.coerceIn(0f, 1f)`.
- **Bonus fix found while touching this code**:
  `LiveAssistantService.restoreSessionFromPrefsIfAvailable()` (the
  process-restart auto-reconnect path — see "Self-echo / output-aware VAD
  gating..." above) never restored `cueVolume` from prefs at all — always
  defaulted to `1f` after a process restart regardless of the user's
  actual Settings value. Fixed alongside the two new fields, all three now
  restored the same way.

Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no new
warnings) — not verified end-to-end on a real device from this
environment, same standing caveat as the rest of this file's Android work.

### Main-path/sub-path joint navigation + flapping-sound drift cue + clock-direction turns (supersedes WALKING route-change hysteresis)

Requested directly by the user, via a fully-specified redesign, after
finding the 6s whole-path dwell-hysteresis above didn't match what was
actually wanted: **"its not that we retain the whole path for 6s but we
retain the joints on the path."** `mapping_servicer.py`'s WALKING branch
reverted to calling `find_natural_path()` fresh every update (see the
"SUPERSEDED" note on that section above) — retaining progress moved
entirely to the CLIENT, at the granularity of individual path JOINTS, not
the whole route.

**Two-layer navigation, both modes now share the same mechanism**
(`ToolDispatcher.kt`):

- **Main path** (`state.plannedPath`, `state.mainPathIdx`) — the
  server-planned route's own joints, adopted FRESH from every
  `MappingUpdate` (no more pose-prepending — that was only ever needed
  for `PathPursuit`'s whole-path arc-length projection, which this design
  no longer uses; `PathPursuit.kt` is superseded, left in place
  unreferenced). `mainPathIdx` resets to 0 whenever a fresh main path
  arrives (the server always plans FROM the current pose forward, so a
  fresh path's own joint 0 is already the correct next target) — the only
  thing that actually ADVANCES `mainPathIdx` is a genuine local arrival
  event, between server updates, never the mere arrival of a new server
  path. This is what "retaining the joints" means in practice: the
  route's IDENTITY can shift slightly between ~1Hz server updates without
  the user's in-progress navigation toward the current joint being
  disrupted.
- **Sub-path** (`computeSubPath()`, `steerAlongMainPath()`) — recomputed
  fresh every avoidance tick (`avoidanceIntervalMs`, ~2x/sec at its
  default), from the CURRENT (latency-bridged, see `HrtfBeacon.
  extrapolate()`) pose toward `state.plannedPath[state.mainPathIdx]`. A
  fresh `AnalyzeFrame(TRAVERSABILITY)` fan each tick (re-added — see
  "Local reactive HRTF obstacle-dodge" for the original design this
  revives a much-narrowed version of) is consulted only to check whether
  the DIRECT line to the current main joint is blocked closer than the
  joint itself; if so, a single dodge point is placed at the nearest open
  bearing in the fan (VFH-style: smallest deviation from the direct
  bearing among bins clearing `SUBPATH_SAFE_CLEARANCE_M`), `SUBPATH_
  DODGE_MARGIN_M` short of that bearing's own measured clearance, capped
  at `SUBPATH_DODGE_MAX_M`. The beacon always steers toward the sub-path's
  own first point (`subPath.first()`) — a dodge point when one exists,
  the main joint directly otherwise.
- **Arrival ("close enough, no other joint still in between")** — since
  the sub-path is recomputed fresh every tick from the CURRENT pose, this
  reduces to one check: once a tick's freshly-computed sub-path is down
  to just the main joint itself (`subPath.size == 1`, i.e. no dodge point
  currently pending) AND the user has arrived within
  `JOINT_ARRIVAL_RADIUS_M` of it, `advanceMainPathJoint()` fires —
  `mainPathIdx++`, then announces the new joint's clock direction (see
  below). A dodge point never needs its own explicit "arrived, remove it"
  step: the very next tick's fresh recompute already reflects having
  passed it (or not) from the new pose, with no separate state to track.

**Drift-to-volume mapping, changed per the user's explicit spec**: `NAV_
DEADZONE_DEG`/`NAV_RAMP_END_DEG` changed from 5°/180° to **3°/100°** —
silent for a drift of 0–3°, ramps to full volume across 3–100°, pinned at
full beyond 100° (narrower than the old ±180° full-range mapping — the
cue now reads as "off" much sooner once roughly facing the right way).
"Drift" is the egocentric bearing (`HrtfBeacon.directionTo(pose,
steerTarget)`) between the user's current facing direction and the
sub-path's own steering target — exactly what `gainForDeviation()`
already computed, just re-parameterized and re-targeted (sub-path point,
not a `PathPursuit` look-ahead point).

**Clock-direction turn announcements** (`announceClockDirection()`,
`clockPositionFor()`) — requested directly by the user: "at start and
each time moving to the next main path joint, give instruction in clock
direction (like 3 o'clock)." Fires exactly twice per navigational event,
never on a bare server update: once, one-shot per session
(`mainJointAnnounced`, reset in `stopActiveModes()`), the first time a
real main path arrives after `start_walking()`/`start_guiding()`; and
once per `advanceMainPathJoint()` call, for the NEW joint just become
current. `clockPositionFor()` maps the egocentric azimuth to the nearest
1–12 clock number (12 = straight ahead, 3 = directly right, 6 = behind, 9
= directly left) — deliberately the OPPOSITE of tracking mode's own
"never clock positions" rule (see "Tracking-init confidence gate..."
above): that rule is specific to hand-guidance's plain-words convention,
while navigation's own turn cues are exactly the case clock positions
exist for.

**Also updated (`ToolDeclarations.kt`)**: the WALKING section now tells
Gemini the cue is a continuous "flapping sound" (quiet when correctly
facing, louder the more off) and to call it that — never "beep" — if the
user asks what the ongoing sound is, and to respond immediately to the
clock-direction `[SYSTEM]` cues from both `start_walking()`/
`start_guiding()`.

`HrtfBeacon.worldPointFrom(pose, azimuthDeg, distanceM)` was re-added
(the exact inverse of `directionTo()` — same math a now-superseded
"Server-planned walking path" design once had under this name, deleted
when that design moved on) specifically to turn a sub-path dodge bearing
into a real world waypoint.

**Known, accepted limitations** (not yet verified live):
- `computeSubPath()`'s dodge search is a single-point VFH-style pick, much
  smaller in scope than the old (removed) `TraversabilityScorer` — since
  the MAIN route is already obstacle-aware, this only ever needs to
  smooth out something that appeared since the main path was last
  planned, not do the primary obstacle avoidance itself.
- Not verified end-to-end on a real device/live server from this
  environment — compile-verified only (`./gradlew :app:compileDebugKotlin`,
  `python3 -m py_compile`), same standing caveat as every round of this
  feature's development. Specifically unverified: whether
  `JOINT_ARRIVAL_RADIUS_M`/`SUBPATH_SAFE_CLEARANCE_M`/the dodge-distance
  bounds need real-world tuning, and whether the 2x/sec sub-path
  recompute cadence (the existing, user-configurable `avoidanceIntervalMs`
  "Avoidance FPS" setting, left at its existing default rather than
  hardcoded to exactly 500ms) reads as smooth in practice.

### Novelty-gated vision-check alert (replaces flat-interval cadence)

Requested directly by the user: WALKING/GUIDING's periodic Gemini
vision-check (`runPeriodicVisionCheck()`) used to fire on a flat timer
regardless of whether the scene had actually changed. Now gated by a
client-side ORB-feature novelty signal — "if we have a 70% new orb
features (can take from yuv for fast processing and use existing orb ft
if has) compare to the previous clear frame (40 blur threshold), then
consider calling the gemini live api for alert if any (but still 3sec
cooldown)."

`AngleTracker` (already running continuously on every luma frame during
WALKING/GUIDING for rotation tracking — see "Pixie + Angle modules")
gained the gate directly, reusing the SAME just-computed ORB keypoints/
descriptors each luma frame already produces — no second detection pass:
`evaluateNovelty()` computes a Laplacian-variance sharpness score (same
formula `CameraManager.computeSharpness()` uses) and, only on a "clear"
frame (`NOVELTY_BLUR_THRESHOLD = 40.0`), matches its descriptors against
the last stored clear-frame reference. `NOVELTY_NEW_FRACTION` (0.70) of
the current frame's own keypoints having no acceptable match
(`NOVELTY_MATCH_DISTANCE_MAX`, a Hamming-distance cutoff) in the
reference triggers `noveltyTriggered`, and that frame becomes the new
reference. A deliberately simplified, SINGLE-reference version of
`orb_novelty_gate.py`'s server-side design (which matches against every
prior accepted frame, not just the latest) — good enough for "did the
view meaningfully change since we last looked," not a scan-quality gate.
`consumeNoveltyTrigger()` — pull-and-clear, called once per avoidance
tick (`runUnifiedAvoidanceTick()`): only when it returns true does
`runPeriodicVisionCheck(frame)` get called at all, which still enforces
its own `PERIODIC_ALERT_INTERVAL_MS` cooldown internally — changed from
6000L to **3000L** (the "still 3sec cooldown" the user explicitly kept as
a floor even with novelty gating in place, protecting against a rapidly-
flickering novelty signal). `reset()` clears the novelty reference/flag
alongside the rest of `AngleTracker`'s state on mode exit.

Not verified end-to-end on a real device from this environment —
specifically unverified: whether `NOVELTY_NEW_FRACTION`/
`NOVELTY_MATCH_DISTANCE_MAX` need real-world tuning against actual indoor
scenes (chosen from this codebase's existing ORB-matching conventions,
not measured against real novelty-gate data the way the server-side
`orb_novelty_gate.py` thresholds originally were).

### Interrupting [SYSTEM] sends

Requested directly by the user: "every single gemini live api call with
[SYSTEM] tag should be a interrupting call, override current call if
overlap." Previously there was no mechanism at all for interrupting
Gemini's own in-progress spoken reply client-side (`LiveServerEvent.
Interrupted` was — and still is — a no-op; see "Continuous VAD-gated
listening"'s own known-limitations note, which explicitly called this
out as unimplemented).

- **`GeminiLiveClient.onInterrupt: (() -> Unit)?`** (new) — invoked at the
  top of `sendSystemNote()`, BEFORE the note is actually sent. Every
  `[SYSTEM]`-tagged send in this codebase already funnels through this
  one function (both `ToolDispatcher`'s `sendSystemNote` constructor
  lambda and `LiveAssistantService`'s own direct calls — incoming-call/
  SMS notes, the session-ready greeting, hazard/vision-check alerts, the
  new clock-direction turn cues above), so wiring the interrupt here
  covers all of them from one place, matching the user's "every single"
  requirement with no per-call-site changes needed elsewhere.
- **`StreamingAudioPlayer.interrupt()`** (new) — `pause(); flush();
  play()` on the existing `AudioTrack`, WITHOUT tearing it down (unlike
  `stop()`) — discards whatever's still queued (the stale tail of an
  in-progress response) so the new turn's chunks play immediately, clean,
  rather than being appended after the old ones. Safe to call on an
  idle/paused track (guarded the same way `stop()` already guards its own
  `AudioTrack` calls).
- **`LiveAssistantService.doLiveSession()`** wires `client.onInterrupt =
  { streamingPlayer.interrupt() }` right after constructing each
  `GeminiLiveClient`.

Whether the Gemini Live API's own `clientContent` protocol additionally
treats an incoming turn as barge-in against a response it's still
generating (separate from this client-side audio flush) is unconfirmed
from this environment — the API's documented behavior for `clientContent`
sent mid-generation wasn't independently verified here; the client-side
flush above is the one guaranteed, controllable half of "interrupting,"
regardless.

Verified via `./gradlew :app:compileDebugKotlin` (BUILD SUCCESSFUL, no new
warnings) — not verified end-to-end on a real device from this
environment, same standing caveat as the rest of this file's Android work.

### Map persistence removed entirely — always a fresh map

Requested directly by the user: "remove the map storing entirely, just
new map every scan, guide." Superseded "Occupancy-grid persistence across
sessions (coarse re-seed)" above (`occupancy_snapshot.json`), which
re-seeded a revisited `location_id`'s fresh `OccupancyMap` from a coarse
disk summary saved at the end of a prior SCAN.

**Note this was already the behavior for a session's in-memory
state regardless of disk persistence** — `StreamingScanSession.__init__`
already unconditionally calls `self.session.reset_cloud()`, which wipes
the occupancy grid AND `_raw_landmarks` on every new stream for a given
`location_id` (see `scan_session.py`'s own comment on why: "a live camera
source has no leftover state"). So this change is purely "stop writing/
reading the disk copy" — it doesn't change what a session starts with,
which was already a blank map every time.

- **`server/services/mapping_servicer.py`** — `_snapshot_path()`/
  `_load_snapshot()`/`_save_snapshot()` deleted outright, along with the
  `SNAPSHOT_FILENAME` constant and the now-unused `json`/`os` imports.
  `MappingServiceServicer.__init__` dropped its `maps_root_dir` param
  entirely (nothing left to use it for) — `grpc_server.py`'s construction
  call and its own now-unused `maps_root_dir` module variable updated to
  match. `UpdateMapping`'s stream-open `seed_from_summary()` call and the
  `finally` block's `_save_snapshot()` call are both gone — the `finally`
  block still calls `stream.session.finalize_landmarks_flat()` for a
  genuine SCAN (still worth it: flushes any last partial VLM-tag batch +
  runs final dedup clustering into `_raw_landmarks`, in memory, since a
  `FindLandmark` call against this SAME in-memory `ScanSession` can still
  land between this stream closing and a later one resetting it), just
  doesn't persist the result anywhere.
- **`GetMapSnapshot`/`ListMappedLocations` RPCs deleted outright** from
  the servicer (not just disabled) — confirmed neither had any real
  caller (`client/android` never called either; nothing else in this repo
  did). Left undefined on the subclass rather than stubbed: gRPC's
  generated base servicer already returns `UNIMPLEMENTED` for any method
  a subclass doesn't override, the honest answer for an RPC that no
  longer does anything. The RPCs/messages themselves are left in
  `tracking.proto` (unused) rather than renumbered — no wire-compat
  benefit to reclaiming the values, same precedent this codebase already
  uses for other retired-but-not-renumbered proto entries (e.g.
  `AnalysisOp.CORRIDOR`).
- **`FindLandmark`** — the persisted-snapshot fallback (`_find_in_snapshot()`)
  is gone; only `session.resolve_landmark()` against the live, in-memory
  `ScanSession` is left. **Real, accepted consequence**: a walking/guiding
  session (which never runs semantic tagging itself — see "Session-mode
  pipeline split" above) can now only resolve a destination that a live
  SCAN of this SAME `location_id`, within the SAME server-process
  lifetime, already found. Previously, an earlier SCAN's landmarks
  survived a server restart via the disk snapshot; now they don't survive
  even past the `ScanSession` being reset (e.g. by a later stream for the
  same `location_id`), let alone a server restart.
- **`OccupancyMap.seed_from_summary()`** (`occupancy_map.py`) is left in
  place, unreferenced — same "kept in case revisited" precedent this
  codebase uses elsewhere (`HrtfBeaconPlayer.kt`, `gemma_vlm.py`,
  `beacon_preview.py`) — nothing calls it any more.
- **`server/data/maps/`** is no longer written to or read from at all by
  the live path. Any pre-existing `occupancy_snapshot.json`/legacy
  zone-based `map_geometry.ply`/`map_labels.json` files there are simply
  dead data now — not deleted by this change, just unread.

Verified via a real import + construction of `MappingServiceServicer`
(confirms `GetMapSnapshot`/`ListMappedLocations` still resolve as
inherited no-op stubs, not missing attributes) and
`python3 -m py_compile server/services/mapping_servicer.py
server/grpc_server.py` — not verified end-to-end against a live server
from this environment, same standing caveat as the rest of this project's
work.

### Path stability — reuse the previous route unless blocked near-term; low-step-over now blocked; joint lead time increased

Three related requests from the same round, all aimed at reducing route
jitter for a blind user following the beacon.

**1. Route reuse + previous-path attraction, both WALKING and GUIDING
(`scan_server/live_path_planner.py`, `server/services/mapping_servicer.py`).**
Previously `UpdateMapping` called `find_natural_path()`/`find_path()` FRESH
on every single update (~1Hz), with progress retained only client-side by
joint index (see "Main-path/sub-path joint navigation" above) — but the
route ITSELF (not just which joint the client is tracking) could still
shift a little every update even when nothing material changed, since
`find_natural_path()` searches fresh off the live (slightly noisy) heading
every time. Fixed with a per-`location_id` cache
(`MappingServiceServicer._prev_path`/`_prev_goal_xz`, popped at stream-open
and on a total-tracking-loss reset, same precedent `_last_full_bounds`
already uses):

- **Reuse gate — `LiveGridPathPlanner.path_blocked_ahead(path_xz, pose_xz)`
  (new)**: projects the current pose onto the previously-served route
  (prepending it as the path's own implicit start, same fix
  `beacon_target_point()` already needed — the served waypoints never
  include the start) to find which segment the user's progress currently
  falls within (local joint advancement is client-side only, so the
  server has no other way to know how far along an old route the user's
  actually gotten), then checks ONLY that one segment
  (`segment_blocked()`, a straight-line cell-walk via the existing
  `_line_cost()`) against the CURRENT grid. Deliberately narrow, per
  direct user request ("only when the path is now blocked early... if
  only a distant end is blocked a bit then no worry") — a blockage
  further along doesn't matter yet; it's re-checked (and, if still
  blocked once actually close to it, triggers a real replan then) on a
  later update as the user gets nearer. If the segment isn't blocked, the
  old route is resent completely unchanged — no new search at all.
- **When a replan IS needed, bias it toward the old route rather than
  starting over from scratch** — the mechanism differs by mode since
  their cost models differ:
  - **WALKING**: `find_natural_path()`'s "keep going this way" reference
    direction becomes the bearing from the current pose toward the OLD
    route's own first joint (`_bearing_rad()`, new), falling back to the
    user's real live heading only when there's no previous route to
    reference at all. Since the whole cost model already rewards
    continuing straight in its reference direction over turning, this
    alone reproduces a highly similar route whenever nothing material
    changed — no separate distance-to-old-path cost term needed for this
    mode.
  - **GUIDING**: `find_path()` gained an optional `previous_path_xz`
    param — every candidate cell's A* edge cost gets an additional
    bounded penalty (`_prev_path_bias()`, new constants
    `PREV_PATH_PENALTY_SCALE`/`PREV_PATH_PROXIMITY_DECAY_RATE`)
    proportional to its distance from the old route, itself decayed by
    how far the cell is from the CURRENT pose — so the bias matters most
    near the user and fades further out ("especially the first joint
    closest to the user," per the user's own framing). Needed here
    specifically because `find_path()`'s A* has no heading concept to
    redirect the way WALKING's search does. A genuine destination change
    (`current_goal_xz` differs from `_prev_goal_xz[location_id]`) always
    forces a fresh, unbiased plan regardless — the old route was toward a
    different point entirely, reusing/biasing toward it would be wrong,
    not just "less stable."
- **Client-side consequence, a real fix required by this change
  (`ToolDispatcher.kt`)**: the mapping-stream collector used to reset
  `state.mainPathIdx = 0` on EVERY incoming path unconditionally, correct
  only because the server used to always replan fresh from the current
  pose. Now that the server can resend the exact same route many updates
  in a row, resetting to 0 every time would have silently discarded the
  client's own local joint-arrival progress on every ~1Hz update — fixed
  via `pathsRoughlyEqual()` (new): the index only resets when the
  incoming path's content actually differs from what's already tracked.

**2. `CLASS_LOW_STEP_OVER` is now hard-blocked in path planning, not just
costlier (`scan_server/live_path_planner.py`)** — `_passable()`/`_cost()`
now treat it exactly like `CLASS_OBSTACLE` (previously a 3.0x cost
multiplier, still passable as a last resort). Requested directly by the
user: this is an assistive system for a blind user, and a low step/curb is
exactly the kind of thing that's dangerous specifically because it can't
be seen coming — never worth routing over regardless of the alternative's
cost. `_COST_BY_CLASS` no longer carries an entry for it (moot — a blocked
class never reaches that lookup from `_cost()`/`_passable()`, and
`_natural_step_cost()`'s `length_cost` term only ever runs for cells that
already passed the passability check). `occupancy_map.py`'s own
classification/rendering is untouched — the dashboard still shows
low-step-over as its own distinct tier, only the path search now refuses
to cross it. Accepted consequence: `find_path()`/`find_natural_path()`'s
closest-approach fallback and dead-end/cone-escalation paths will trigger
somewhat more often in areas with a lot of low obstacles — intentional,
not a regression.

**3. Joint arrival radius 0.6m -> 1.5m (`ToolDispatcher.kt`)** —
`JOINT_ARRIVAL_RADIUS_M` (main-path joint advancement + its clock-direction
announcement, see "Main-path/sub-path joint navigation" above) bumped per
direct user request for more lead time before a turn is actually needed.
The mechanism itself (advance before literal arrival, announce the new
joint's clock direction relative to the user's CURRENT facing — 12 o'clock
is always straight ahead, via `HrtfBeacon.directionTo()`'s existing
egocentric convention) was already correct and needed no other change.

Not verified end-to-end on a real device/live server from this
environment — compile-verified only (`./gradlew :app:compileDebugKotlin`,
`python3 -m py_compile`), same standing caveat as every other round of
this project's work. Specifically unverified: whether
`PREV_PATH_PENALTY_SCALE`/`PREV_PATH_PROXIMITY_DECAY_RATE` need real-world
tuning, and whether reusing a stable route for many consecutive updates
ever feels "stale" in practice despite the near-term blocked-check.

### search_objects — informational Q&A over saved object memory (`ToolDispatcher.kt`, `ToolDeclarations.kt`)

New tool, requested directly by the user: `search_objects(targets?)`
answers an INFORMATIONAL question about previously-saved (labeled)
objects currently in view — e.g. "which one is my AC remote and which is
my fan remote" or "what's on the table, any of my belongings" — distinct
from `start_tracking`/`get_object_from_memory` (an explicit "find me.../
get me to.../look for..." request that initiates hand-guidance tracking
mode) and from generic, unlabeled objects (water, a pen, etc.), which this
tool never handles at all.

- **With `targets` given** — each target string is resolved to saved
  memory label(s) via the existing `LocalMemoryStore.findLabelsMatching()`
  (case-insensitive substring, either direction); a target matching no
  saved label is reported separately (`not_in_memory`), not silently
  dropped. Each matched label gets its OWN dedicated
  `PerceptionService.AnalyzeFrame(DETECT)` call against its own stored
  description — simple, no ambiguity about which detection belongs to
  which label, and cheap enough for the handful of targets a real
  question names.
- **With `targets` omitted/empty** — every saved label that has a stored
  DINOv2 visual reference (`hasObjectEmbeddings()`) is checked via ONE
  combined multi-phrase GroundingDINO call (every candidate's own stored
  description joined by `" . "` — the same phrase-grounding convention
  `frame_extractor/tagging.py`'s multi-tag prompting already established)
  rather than one call per label — efficient regardless of how many
  objects have been saved. Each returned detection is attributed back to
  whichever candidate label's own description contains (or is contained
  by) its decoded label text — safe because GroundingDINO only ever
  decodes a literal span of whichever prompt phrase it matched.
- **Per-candidate verdict, thresholds specified directly by the user** —
  for whichever detection scores highest via `bestObjectSimilarity()`
  against that label's own stored embedding(s): `>= SEARCH_OBJECTS_
  CONFIRM_SIM` (0.5) -> `"found"` (box included); `>= SEARCH_OBJECTS_
  RESEMBLE_SIM` (0.3) only -> `"resembles"` (a possible, not confident,
  match — the SYSTEM_PROMPT tells Gemini to say so honestly, not claim
  certainty); below that (or no detection at all) -> omitted from the
  response entirely, rather than reported as a firm miss (a long
  "not found" list for an all-labels query would be noise, not help).
  Server-side `detect_all()`'s own default `box_threshold=0.35` (unchanged)
  is what filters candidate detections before this similarity step —
  matches the user's own ">0.35 GroundingDINO" spec exactly, no additional
  client-side re-filtering needed.
- `ToolDeclarations.kt`'s TRACKING section gained a bullet distinguishing
  this tool from `start_tracking`/`get_object_from_memory` by intent
  (informational question vs. an explicit find/track request), matching
  the distinction requested directly by the user.

Not verified end-to-end on a real device from this environment — compile-
verified only (`./gradlew :app:compileDebugKotlin`), same standing caveat
as the rest of this file's Android work.

---

## 3D Scanning Pipeline

**Read this section as algorithm/design history, not a current entrypoint.**
The dataset-record-and-replay framing below (Android ScanScreen, `/api/upload`,
Segment Table, `scan_gui.py`'s Simulated/Manual Stream buttons,
`stream_simulator.py`) is **gone** — `ScanViewModel.kt`/`ScanScreen.kt` and
`scan_server/scan_server.py` were deleted (see "Client-Orchestrated Live
Session"). The underlying algorithms it describes — `occupancy_map.py`'s
Bayesian occupancy grid, `feature_tracker.py`'s depth-consistency gate, TSDF
fusion, RTAB-Map integration — **are still live and current**, just now fed
by `MappingService.UpdateMapping`'s real Android-streamed frames instead of
a replayed dataset (`scan_session.py`/`occupancy_map.py` themselves barely
changed). `server/tools/route_planner.py`, `grid_path_planner.py`, and
`localization.py` (referenced several times below as the zone-based
navigation path) were also deleted — that whole path-planning/localization
job now happens via `MappingService` (server-side `LiveGridPathPlanner`,
see "Server-planned walking path + client-side latency bridging") +
Android's `HrtfBeacon.kt`/`PathPursuit.kt` instead, per the new architecture.

```
Android ScanScreen records a dataset/ folder simultaneously:
  images/000000000.jpg, ...  (camera.csv: timestamp_ns,filename)
  imu.csv                    (timestamp_ns,ax,ay,az,gx,gy,gz — same clock domain as camera.csv)
  → zipped client-side, POST /api/upload to scan server (extracted there)
     OR point the Gradio UI at an existing dataset folder path
      │
dataset/ folder (images/ + camera.csv + imu.csv)
      │
      ├─ imu.csv → scan_session.set_imu_file(orientation=...)
      │    → ImuIntegrator: rotates raw accel/gyro into the frame matching how
      │      the phone was physically held (IMU Orientation dropdown in
      │      scan_gui.py — portrait/landscape-left/landscape-right; Android's
      │      SensorManager always reports in the fixed native-portrait body
      │      frame regardless of hold, so this is required whenever a scan was
      │      recorded landscape); then gravity/bias init from stationary period
      │
      ▼
Pose source (selected per scan in the Gradio UI, ScanSession.process_frames_batch)
— only 2 are supported ("Auto"/"VO only"/"DA3 poses" were removed: DA3 poses'
5-frame sliding-window pose stitching added complexity without being needed
once RTAB-Map covers the no-calibration-needed case; "VO only"/"Auto"
collapsed into "IMU + VO", which already falls back to VO-only behavior on
its own when no imu.csv is present):
  • IMU + VO        — feature_tracker.py ORB+PnP → pose_graph.py scipy LM
  •                   optimization + ORB-based loop closure (local,
  •                   batch-scoped) for translation; IMU gyro-integrated
  •                   rotation when imu.csv is present, else VO rotation too
  • RTAB-Map        — rtabmap_client.py → scan_server/rtabmap_docker/
  •                   (ZeroMQ, no ROS → standalone RTAB-Map RGB-D odometry +
  •                   loop closure process); one continuous session per
  •                   location scan; needs only camera intrinsics (sent
  •                   per-frame) — no camera-IMU calibration at all, unlike
  •                   the ORB-SLAM3 mode it replaces (unreliable Kalibr
  •                   calibration). Depth from DA3-ONNX estimated server-side
  •                   (Python) for both modes, but ONLY RTAB-Map mode also
  •                   reconstructs the point cloud server-side (C++) — its
  •                   own cloudRGBFromSensorData per node, transformed by its
  •                   CURRENT graph-corrected pose, pulled via GET_CLOUD (see
  •                   rtabmap_client.get_cloud()) — replacing this project's
  •                   own DA3-depth back-projection (scan_session.py's
  •                   _back_project_frame, below) for that mode only.
  •                   IMU + VO still back-projects locally.
      │
      ▼
vio/ — VIOEstimator (GTSAM iSAM2) — not currently called by either pose path
  above (ImuIntegrator does simple gyro-integration dead-reckoning
  instead) or elsewhere in live code; only referenced from server/_archived/.
      │
      ▼
scan_session.py's dense back-projection (IMU + VO mode ONLY — RTAB-Map mode
skips this entirely, see above). No separate scale-alignment pass: DA3's own
metric depth (da3_wrapper.py, DA3Estimator.estimate_batch — multi-view joint
inference per mini-batch) is used directly, fed straight into
_back_project_frame: X=(u-cx)d/fx, Y=(v-cy)d/fy, Z=d → world via the pose
computed in Step 2 (FeatureTracker ORB+PnP, IMU-fused, or RTAB-Map — never
VIOEstimator, per the note above).
  • Depth-consistency gate (feature_tracker.py, added after a real bug: a
    warped/displaced wall showed up in Live Points and Voxelization, smooth
    and internally coherent rather than noisy — traced to DA3Estimator's
    per-mini-batch (mini_batch=4 frames) multi-view joint inference having
    zero continuity across batches or with the pose pipeline that fuses its
    output, so one difficult batch (motion blur, low texture, oblique
    angle) could produce depth that's self-consistent WITHIN that batch but
    wrong relative to the rest of the map, with no check anywhere before
    permanent fusion). FeatureTracker.track()/_estimate_relative_pose now
    also runs _triangulate_depth_agreement on every PnP-solved frame: the
    same PnP-inlier 2D correspondences are re-triangulated via two-view
    geometry (camera-independent of any dense depth map), transformed into
    the CURRENT frame's own camera coordinates, and compared per-point
    against DA3's dense depth at the CURRENT frame's own pixels — checking
    curr's own depth, not prev's, matters: an earlier version checked prev's
    depth (an artifact of PnP lifting prev-frame keypoints) and attributed
    the verdict to the wrong frame, a one-call lag caught via synthetic
    testing (case 5 in the test suite: prev corrupted / curr clean must NOT
    reject) before it shipped. The aggregate metric is the FRACTION of
    inlier points whose relative error exceeds a per-point threshold
    (0.30 default), not the median — a coherent warp often covers only part
    of a frame (e.g. one wall, not the whole view), so a median would let
    the well-behaved majority mask it; fraction-bad tracks the actually-bad
    fraction directly (verified: measured frac_bad matched injected
    corruption fraction almost exactly, e.g. 0.40 measured for 40% injected,
    0.70 for 70%, on synthetic ground-truth 3D geometry). A frame flagged
    untrustworthy (`last_depth_trustworthy=False`) has its dense point cloud
    skipped for permanent fusion in Step 3 (`[depth-consistency] frame
    REJECTED` console log, with frac_bad/n/threshold) — pose and semantic
    landmark extraction are unaffected, only the geometry fusion is gated.
    Frames the check couldn't evaluate (too few PnP inliers) default to
    trustworthy, so missing data is never penalized as if it were bad data.
      │                                            │
      ▼                                            ▼ (RTAB-Map mode instead)
                                     rtabmap_client.get_cloud() — server's own
                                     cloudRGBFromSensorData per node, already
                                     voxelized, transformed by its CURRENT
                                     corrected pose (see architecture diagram
                                     above) — ScanSession._rtabmap_pull_new_
                                     nodes()/_rtabmap_full_resync() (below)
      │                                            │
      └────────────────────┬───────────────────────┘
                            ▼
point_cloud_fusion (in scan_session.py) — same downstream stage for both pose sources
  • Each batch's (or, for RTAB-Map, each new node's) points are voxel-
    downsampled (0.02 m, RTAB-Map does this server-side instead) +
    outlier-removed ONCE (cheap), then buffered raw in self._raw_cloud_batches
    — NOT accumulated into a live, ever-growing self._cloud anymore (see
    "Progressive Occupancy Map" below for why that used to be wasted work on
    every batch)
  • ensure_cloud_built() lazily merges every buffered batch into self._cloud
    (one more voxel pass to fuse adjacent batches' voxels) — only called
    on demand: Reload (Live Points tab), Voxelize, Export, or
    finalize_voxel_and_occupancy() — never from inside process_frames_batch
      │
      ▼
zone_labeler.py  (called from scan_session after each segment)
  • Named AABB from camera path in that segment (set_label_from_positions)
      │
      ▼
map_exporter.py
  • maps/{location_id}/map_geometry.ply  (binary Open3D PLY, built via
    ensure_cloud_built() at export time)
  • maps/{location_id}/map_labels.json   (metadata + zones[])
```

**Streaming interface:** the pipeline above is driven entirely through
`stream_session.py`'s `StreamingScanSession` — a push-based incremental
session (`push_frame()`/`push_imu()`/`start_zone()`/`end_zone()`/`finish()`)
that calls `ScanSession.process_frames_batch()`/`finalize_voxel_and_occupancy()`/
`export()`, fed one frame/IMU sample at a time. There is no batch path
anymore — `scan_gui.py`'s old "Scan" button and its handler (`_run_local_scan`,
which needed a finished on-disk dataset processed as one big pre-sliced job)
were removed once the streaming interface fully replaced it; the Segment
Table input stays (both remaining modes still use it for zone `start_s`/
`end_s`/`zone_name` boundaries), but nothing reads a whole dataset in one
shot anymore. `stream_simulator.py` drives `StreamingScanSession` from an
already-uploaded `uploads/<scan_id>/dataset/` folder two ways, both exposed
as GUI buttons in the "Live Reconstruction" tab:
- `replay_dataset()` — auto-driven generator (`scan_gui.py`'s **"Simulated
  Live Stream"** button, `_run_simulated_stream`), replaying every frame/IMU
  sample as fast as possible (or real-time-paced, `realtime_pacing_cb`) with
  no per-frame pause.
- `ManualDatasetReplayer` — single-step counterpart (`scan_gui.py`'s
  **"Start / Reset Manual Stream"** + **"Feed Next Frame"** buttons,
  `_manual_stream_start`/`_manual_stream_feed`): `has_more()`/
  `peek_next_frame_preview()` let the GUI show a small preview of the next
  not-yet-fed frame (`manual_preview_image`) before each click;
  `step()` applies every IMU sample/zone boundary preceding that frame, then
  pushes exactly it, one frame per click, until `has_more()` is false — at
  which point the feed button disables itself and the session auto-finalizes
  + exports, same as `replay_dataset()`'s own end-of-replay step.

Both replay modes share `build_event_timeline()` (camera.csv/imu.csv reading
+ fps subsampling + timestamp-sorted event list) so they see the identical
sampled frame set — only the driver (an auto loop vs. one call per click)
differs. **Fixed**: the `i % interval == 0` subsampling always includes
frame 0 but had no guarantee of landing on the dataset's actual LAST frame —
at a coarse interval (low fps), the gap between the last sampled frame and
the recording's true end grew (up to `interval - 1` frames), so both replay
modes would reach `has_more() == False` well before the actual end, losing
more of the tail the lower the fps was set. `build_event_timeline()` now
always appends the dataset's true last frame if the modulo sampling didn't
already land on it.

**Found via real recordings, fixed**: even with the tail fix above, RTAB-Map
pose mode specifically still lost many more frames than the fps setting
should account for, worse at lower fps — traced to `build_event_timeline()`'s
fps subsampling being applied to RTAB-Map the same as IMU + VO. RTAB-Map's
own frame-to-frame visual odometry needs enough overlap between consecutive
TRACKED frames to match features; skipping frames increases inter-frame
motion, and a frame RTAB-Map's odometry reports "lost" on contributes NO
node/data to the reconstruction at all (verified on a real 79-frame
recording: ~10% tracking-lost at fps=10, climbing to ~70% at fps=0.5, with
node count collapsing from 71→28→10→3 as fps dropped — IMU + VO doesn't have
this failure mode and was never affected). Fixed via a new `native_fps` flag
on `build_event_timeline()`, set from `stream._use_rtabmap` (the
already-resolved flag `replay_dataset()`/`ManualDatasetReplayer` receive
from their `StreamingScanSession`) — when true, `interval` is forced to 1
(every recorded frame included) regardless of the fps slider, so RTAB-Map
always gets the continuous frame-to-frame tracking it needs; IMU + VO keeps
normal fps subsampling unchanged. Verified: frame count now stays at the
dataset's full 79 regardless of the fps setting, with only small run-to-run
noise (77-79 nodes across repeated identical runs) instead of the previous
systematic 79→40→21→10 collapse.

Both fire `start_zone()`/`end_zone()` at Segment Table boundaries
crossed during replay — standing in for a real operator's button presses,
since the dataset format has no live equivalent for that yet. A real Android
live source (bidi gRPC stream, frame+IMU multiplexed the way
`VoiceChatChunk`/`IMUFrame` already are for the voice/agent path — see gRPC
Services & Protobuf) is planned to drive `StreamingScanSession` the same way,
replacing only `stream_simulator.py`'s role, not `stream_session.py` itself.
Both pose sources are supported in the streaming path, including RTAB-Map
(needs no retained raw IMU samples, unlike the old ORB-SLAM3 mode this
replaced).

**Progressive, Bayesian, SLAM-style Occupancy Map + semantic-destination A\*
navigation:** `occupancy_map.py`'s 2D X-Z traversability grid builds up live,
once per processed batch (`OccupancyMap.update()`, called from
`ScanSession.process_frames_batch`'s Step 4) — not deferred to end-of-scan —
same as a real SLAM system's occupancy grid filling in as it explores. The
Occupancy Map is **fully inferred from the voxelization**, not a
separately-computed approximation of it: each batch/node's own new points
are run through `voxelize_cloud()` (the exact same function `scan_gui.py`'s
Voxelization view calls) at `occupancy_voxel_size` (the GUI's Voxelization
slider), and the resulting voxel CENTERS — not the raw or fine-voxel-
downsampled points — are what feeds `update()`.

**There is exactly one voxelization, not two independently-computed ones**:
`voxelize_cloud()` (scan_session.py) anchors every call to a fixed grid
(`_VOXEL_GRID_MIN_BOUND`/`_MAX_BOUND`, via Open3D's
`create_from_point_cloud_within_bounds`) instead of Open3D's own
per-call bounding-box-derived origin — verified empirically that the
default per-call origin drifts by ~1e-5 between two point sets covering the
same physical region, enough to misalign voxel bins; a fixed anchor makes
any subset's voxelization an exact subset of any other's at the same
voxel_size. `ScanSession._merge_voxels()` is the ONLY place
`last_voxel_centers`/`last_voxel_colors`/`last_voxel_size` get written —
called from `process_frames_batch`/`_rtabmap_process_nodes` right after each
batch/node's own `voxelize_cloud()` call already feeds `occupancy_map.update()`,
merging into `self._voxel_dict` (keyed by grid index, so the same physical
voxel seen again updates in place rather than duplicating) — and from
scan_gui.py's on-demand "Voxelize" button (an explicit, infrequent, possibly
different-voxel-size recompute over the whole cloud, which resets the
accumulator first since indices from two sizes aren't comparable). `_run_
simulated_stream`/`_manual_stream_feed`'s per-chunk Voxelization-view render
now just reads `session.last_voxel_centers` directly — no second
`voxelize_cloud()` call — and `finalize_voxel_and_occupancy()` no longer
re-voxelizes the whole cloud either (dropped its `voxel_size` param
entirely, along with `StreamingScanSession.finish()`'s, since neither had
anything left to do with it). Occupancy cells correspond exactly to what's
displayed in the Voxelization view: same voxels, same coordinates, computed
once.

**Found and fixed after the above landed**: `voxelize_cloud()`'s `MAX_VOXELS`
auto-coarsening (rescales `voxel_size` up, cube-root scaling, whenever a
call's own occupied-voxel count would exceed it) exists to bound a ONE-SHOT
whole-cloud call's render cost — but `_merge_voxels()` resets its entire
accumulator whenever the incoming `vsize` differs from the last call's (a
genuine voxel-size change isn't mergeable with the old grid). Since each
batch/node's own point count varies independently of total scan size, any
single batch that happened to exceed `MAX_VOXELS` at `occupancy_voxel_size`
got silently coarsened to a different `vsize` for just that call — silently
wiping every previously-accumulated voxel the instant that happened, and
corrupting the Occupancy Map's effective resolution too. Symptom: the
Voxelization view showing holes and disconnected fragments (whatever
survived after the last accidental mid-scan reset) despite Live Points
(fed from the same raw batches, unaffected) looking complete. Fixed by
passing `max_voxels=_NO_COARSEN_MAX_VOXELS` (a no-op-large sentinel) at both
`voxelize_cloud()` call sites that feed the Occupancy Map/voxel accumulator
— `OccupancyMap.update()` already subsamples internally
(`MAX_CLOUD_SAMPLE`) if a single call gets too many points, so there was no
cost reason for the cap there either. The manual "Voxelize" button (a
genuine one-shot whole-cloud recompute, at a user-picked size) keeps the
normal `MAX_VOXELS` cap — coarsening there is legitimate and expected.

**Also found and fixed in the same pass**: `StreamingScanSession.__init__`
now calls `self.session.reset_cloud()` right after
`scan_manager.get_or_create()` — `ScanSessionManager` keys sessions by
`location_id` and returns the SAME `ScanSession` across separate
`StreamingScanSession` instantiations, so starting a new stream (Simulated
or Manual) without an explicit "Clear Cloud" first used to silently resume
on top of whatever an earlier, unrelated run for that location had already
accumulated — a real live camera source has no such leftover state. Also:
`reset_cloud()` wasn't clearing the new `_voxel_dict` accumulator (only
`last_voxel_centers`/`colors`), a latent bug from the same voxelization work
above — fixed alongside it.

In the GUI (`scan_gui.py`'s "Live Reconstruction" tab), Live Points
still rebuilds via `ensure_cloud_built()` (a separate, raw-point view,
unrelated to voxelization) once per chunk across all three ways of driving
`StreamingScanSession` — Simulated Live Stream, Manual Live Stream, and each
chunk `push_frame()` returns a result for — the visible pipeline is Live
Points → Voxelization → Occupancy Map, each toggleable via its own checkbox
(default all on).

**Two independent algorithm toggles** (`OccupancyMap.enable_ray_casting`/
`enable_bayesian`, both default `True`, GUI checkboxes in "Occupancy Map
Settings"): `enable_ray_casting=False` skips the free-space Bresenham ray
cast in `update()` entirely — cells only ever get HIT, never revised back to
free. `enable_bayesian=False` disables incremental log-odds belief — a
single hit sets a cell's logodds straight to `LOGODDS_MAX` (immediate,
permanent classification from its height) and `_register_miss` becomes a
no-op (a miss has nothing meaningful to erode without incremental belief) —
"first observation wins" instead of "accumulate agreeing evidence, revise on
disagreement." The two are independent: with ray casting on but Bayesian
off, ray casting still runs but its misses are no-ops against already-hit
cells. Verified: with both on, ray casting fills in far more confident
"ground" cells (self-correcting free space, per the module's own design);
disabling either collapses toward "everything ever hit stays an obstacle
forever."

**Rewritten to match how real occupancy-grid mappers build a 2D grid from a
3D point cloud (RTAB-Map's own OccupancyGrid, ROS costmap_2d), fixing a real
bug where a large obstacle (reported: an entire bed) got eroded to
free/unknown, with only a small fraction surviving as "low obstacle":**
`update()` now classifies every POINT individually by height above ground
(not a per-cell-per-batch percentile summary), buckets by (X,Z) cell —
obstacle evidence in a cell always wins over a ground point landing in the
same coarse cell that batch — and casts free-space rays ONLY toward
ground-classified cells (never toward obstacle cells, whose own hit already
speaks for itself). **The actual fix**: `_cast_free_ray`'s Bresenham walk
now STOPS at the first cell with any net hit evidence (`logodds > 0`) —
exactly like a real depth/laser ray being physically blocked by the first
obstacle it hits and never testing anything behind it — instead of the
previous linear-height-interpolation gate (`MISS_HEIGHT_MARGIN`/
`VIRGIN_ASSUMED_HEIGHT`, both removed along with the sliders that exposed
them). That gate was only a 2D-distance-based proxy for real 3D occlusion:
for a ray whose target is near floor level (extremely common — the floor is
everywhere), the interpolated height dropped close to floor level well
before reaching a real closer obstacle positioned nearer the target than
the camera, so the ray incorrectly rated itself "low enough to have tested"
the obstacle and eroded it — repeatedly, once per batch whose ray happened
to aim at floor beyond the obstacle, with erosion compounding faster than
the obstacle's own comparatively rare direct re-hits. Blocking is
deliberately gated on ANY net hit evidence (`logodds > 0`), not the
stricter `LOGODDS_OCCUPIED_THRESH` used for final classification — verified
via a synthetic test that a real obstacle seen only once (`logodds ==
LOGODDS_HIT`, e.g. 0.85, below a 1.0 confirm threshold) still got fully
eroded by ~30 subsequent adversarial ground rays if blocking waited for
full Bayesian confirmation instead of any net evidence. Verified end to end
with synthetic tests: (1) a single-hit obstacle survives 30 adversarial
"ray to floor beyond it" batches unchanged; (2) a twice-hit (confirmed)
obstacle stays classified as obstacle through 20 more; (3) genuine one-off
sensor noise, never reconfirmed, still self-corrects to free once that same
cell is directly re-observed as floor; (4) genuinely open, never-hit floor
space still gets carved to free purely via ray casting — so the fix removes
the false-erosion failure mode without weakening real self-correction.
`update()` also now positively registers ground hits (nudges logodds toward
free directly, not just inferred from absent obstacle hits) via a new
`_register_ground_hit`, and bootstraps a one-off `ground_y` estimate from a
batch's own points on the very first call (previously classification wasn't
possible at all until enough hit evidence existed elsewhere). Every
`update()` call ends with a single `[occupancy]` console log line — point
counts by class, rays cast vs. rays that stopped early on a confirmed
obstacle, current `ground_y`, running cell count — specifically added for
debugging a future recurrence: a real obstacle disappearing again with
`blocked_by_obstacle` staying near 0 would point at a hit-rate problem (the
obstacle's own cells never crossing `logodds > 0`), not a ray-casting bug.

**GPU acceleration + console timing:** `scan_session.py` detects Open3D CUDA
support once at import (`_CUDA_AVAILABLE`, logged to console) and routes
`voxel_down_sample`/`remove_statistical_outlier` through the tensor
(`o3d.t.geometry.PointCloud`) GPU API via `_voxel_down_sample_accel()`/
`_remove_statistical_outlier_accel()` whenever a cloud is large enough
(`_GPU_ACCEL_MIN_POINTS = 50_000`) that the legacy<->tensor transfer
overhead is worth it — benchmarked: below that, plain CPU is as fast or
faster once transfer cost is counted; above it, GPU wins clearly (3x+ at
~500k points on realistic depth-camera-shaped point distributions — a
uniform-random test cloud is a misleading adversarial case for GPU KNN and
was excluded from that conclusion). `voxelize_cloud()` itself has no GPU
path (Open3D's only CUDA voxel structure, VoxelBlockGrid, is a TSDF-style
volumetric grid, not a drop-in replacement) but its grid_index → world
center conversion is vectorized (numpy, not a per-voxel Python loop).
**Known, benign discrepancy**: Open3D's GPU and CPU `voxel_down_sample`
are not bit-identical on the same input (a handful of boundary points can
land in adjacent voxels differently) — immaterial for Live Points/
Voxelization display or Occupancy Map classification, but means a session
whose cloud crosses the GPU threshold mid-scan won't produce byte-identical
output to an all-CPU run of the same data. **Also confirmed: the GPU path
isn't even deterministic against ITSELF** — two separate calls with the
identical input can yield the same point count but different exact
positions (almost certainly non-associative floating-point reduction order
in parallel voxel-centroid averaging) — same "harmless for this pipeline,
don't test point-for-point equality across GPU calls" conclusion applies.

For RTAB-Map mode specifically, `ScanSession._rtabmap_process_nodes()`
(shared by `_rtabmap_pull_new_nodes`/`_rtabmap_full_resync`) BATCHES the
SOR step across every node into ONE combined cloud + ONE
`remove_statistical_outlier` call, instead of one small call per node —
each node's own cloud individually stayed below `_GPU_ACCEL_MIN_POINTS`, so
a loop-closure resync with dozens of nodes previously did that many
separate CPU-only SOR calls whose fixed per-call overhead added up badly
(a real ~14s stall observed on a 39-node resync). The combined SOR's
keep-mask is sliced back into per-node segments afterward so each node
still gets its own cleaned cloud; `voxelize_cloud()`/`occupancy_map.update()`
stay per-node (batching those would reproduce the sunburst artifact). On
top of that batching, `self._rtabmap_sor_keep_mask` caches each node's
keep-mask by node_id so a resync's re-pull of an ALREADY-seen node skips
SOR entirely (rigid pose transforms don't change SOR's outlier decision —
see 3D Scanning Pipeline's "RTAB-Map full-resync SOR caching" note) —
`_rtabmap_process_nodes()` only batches/SORs genuinely new node ids now.

**The actual dominant cost turned out to be `occupancy_map.update()`
itself, not SOR/voxelize** — profiling (`cProfile`) found `np.percentile()`
alone was ~70% of `update()`'s total time, called on tiny per-cell height
lists where its generic per-call overhead (argument validation, reduction
machinery) swamps the real computation, AND called TWICE per touched cell
(once for hit registration, once again — redundantly — for the ray-cast
target height). Fixed in `occupancy_map.py`: `_percentile_fast()` (verified
numerically identical to `np.percentile`, ~1e-15 max diff over thousands of
trials, via `np.partition` instead of the full percentile machinery, ~10x
faster per call) replaces every `np.percentile` call in `update()`, and the
per-cell height is now computed once and reused for both the hit and the
ray-cast target. Net effect measured directly: `occupancy_map.update()`
dropped from ~170ms/node to ~25-40ms/node on the same real dataset (~4-9x).

**scan_gui.py's Live Reconstruction tab — two more found via the console
timing logs**: (1) Live Points/Voxelization were unconditionally rebuilt +
re-rendered on EVERY processed chunk, even one that added zero new points
(e.g. an RTAB-Map incremental pull that returned no new nodes) — pure wasted
work re-voxelizing/re-rendering the exact same unchanged cloud. Both
`_run_simulated_stream` and `_manual_stream_feed` track
`_last_render_point_count` (a local generator variable / closure-scoped
respectively) and skip the rebuild+render entirely when
`session._raw_point_count` hasn't changed since the last one. (2)
`_voxel_centers_to_glb` (the Voxelization view's box-per-voxel renderer)
built one separate `trimesh.creation.box()` object per voxel in a Python
loop then concatenated them — O(voxel count), profiled at 8+ seconds for
~28k voxels. Rewritten to build ONE combined mesh directly via vectorized
numpy broadcasting of a single reference box's vertex/face template (still
sourced from one real `trimesh.creation.box()` call, just reused instead of
rebuilt per voxel) — verified geometrically identical (exact vertex/face
counts, exact box centroids, exact per-voxel colors) at ~30x faster (8261ms
→ 267ms for the same ~28k voxels).

`timing_utils.py`'s `timed()` context manager prints `[timing] <label>: <ms>`
for every stage of this pipeline — depth estimation, pose computation,
back-projection, voxel/outlier cleanup, `voxelize_cloud()`,
`occupancy_map.update()`/`.render_plotly()`, `ensure_cloud_built()`, and GLB
rendering (`_cloud_to_glb`/`_voxel_centers_to_glb` in `scan_gui.py`) — a
separate tiny module (not folded into `scan_session.py`) specifically so
`occupancy_map.py` can use it too without a circular import.
RTAB-Map's own docker container (`scan_server/rtabmap_docker/`) was checked
and does NOT use GPU (no CUDA libraries in the base image, `nvidia-smi`
unavailable) — deliberately left CPU-only: a real per-frame TRACK
round-trip against the built image measured ~54 ms median (real dataset,
320×240), well within budget for this project's typical sampling rates, so
GPU-enabling it (would need a from-scratch CUDA-accelerated OpenCV rebuild
inside the container — large effort, uncertain payoff) wasn't pursued.

Each cell holds a **Bayesian log-odds belief** (`_CellState.logodds`), the
same representation gmapping/cartographer/RTAB-Map's own OccupancyGrid
module use (Moravec & Elfes 1985) — NOT a permanent record of the first
thing ever observed there. Every batch, `update()` classifies each POINT
individually by height above ground (obstacle vs. ground vs. ignored
ceiling — see the rewrite note above), then each obstacle-touched cell gets
one HIT (`_register_obstacle_hit`, log-odds nudged toward occupied, plus an
exponentially-weighted `height_ewma` recency-biased toward the most recent
observation) and each ground-only cell gets one positive ground HIT
(`_register_ground_hit`, log-odds nudged toward free — not just inferred
from an absence of obstacle hits). Free-space ray casting (`_cast_free_ray`,
Bresenham from each batch's camera position, cast ONLY toward
ground-classified cells) registers a MISS (`_register_miss`, log-odds
nudged toward free) on every intermediate cell that has NOT already crossed
`logodds > 0` — **including cells that already have hit evidence below that
bar** — but STOPS entirely, without marking anything, the instant it
reaches a cell that HAS. This is what makes the grid self-correcting: a
single noisy far-range depth reading that wrongly looks like an obstacle
gets pulled back down as soon as a few closer, disagreeing observations
arrive — including simply the user walking past/through that spot later,
registering a direct ground hit right on that cell — while a REAL obstacle
along a ray's path stops the ray before it ever reaches (or passes through
to erode) anything past it, matching how a real depth/laser ray is
physically blocked by the first thing it hits. An earlier raw-sample-list
version of this module had no revision mechanism at all (ray casting only
ever filled in cells with NO existing data) — not how any real
occupancy-grid mapper works; a later version added revision back but used a
linear-height-interpolation proxy for occlusion instead of a real stop
condition, which is what let a real obstacle (a bed, confirmed via a bug
report) get progressively eroded by rays aimed at floor beyond it. Both
are now fixed by this occlusion-respecting design — see the rewrite note
above for the full mechanism and the synthetic tests that verified it.

**All of these are tunable from the GUI**, not hardcoded — `OccupancyMap.
__init__`/`set_params()` accept overrides for every constant named above
plus the log-odds weights (`LOGODDS_HIT`/`LOGODDS_MISS`/
`LOGODDS_OCCUPIED_THRESH`/`LOGODDS_FREE_THRESH`) and `HEIGHT_EWMA_ALPHA`.
`ScanSession.configure_occupancy_map()` applies new values to the current
map in place (only affects future `update()` calls — already-accumulated
cell beliefs aren't retroactively recomputed) and remembers them for the
next `reset_cloud()`; `ScanSessionManager.configure_occupancy_defaults()`
applies to every existing session and seeds new ones. `scan_gui.py`'s
"Occupancy Map Settings" accordion exposes all of this — scene-dependent
tuning (a bedroom full of low furniture vs. an open hallway) needs different
values and no one constant fits every room.
The ground-plane estimate is recomputed from the *cumulative* `height_ewma`
across every cell with hit evidence so far (not just the latest batch), so
it only gets more confident over time rather than jittering on partial data.
Rendered as a continuous height-above-ground heatmap (`render_plotly`,
"Turbo" colorscale): 0.0m (ground) at the low end, `OBSTACLE_MAX_H` at the
high end, unknown cells (not enough agreeing evidence either way) left as
NaN (blank) — replaced an earlier classic-SLAM 4-color discrete grayscale
(white/light-gray/black/mid-gray) that only showed occupied-or-not, not how
TALL an obstacle actually is. `_render_plotly_impl` still calls
`_classify_state()` as the single source of truth for ground/step-over/
obstacle/unknown (same Bayesian belief `extract_subgrid`/`extract_full_grid`
use for path planning) and de-normalizes its 0–1 height fraction back to
real metres for display, rather than duplicating the classification
thresholds; the old discrete-band gap-fill (dilating obstacle presence into
sparse-coverage gaps) is generalized to continuous data the same way — an
unknown cell next to a tall neighbor gets that neighbor's height via a 3×3
max filter, rather than just being marked "obstacle."
**Known past bug, now fixed:** calling `occupancy_map.reset()` + one
`update()` with the *entire* session's trajectory + full voxel cloud (which
`finalize_voxel_and_occupancy()` used to do) broke this — `update()`'s ray
casting picks one representative camera position per call, which is fine
per-batch but produced a sunburst/hatch artifact when given the whole
session's trajectory at once (thousands of rays from a single final camera
position to every cell in the map). `finalize_voxel_and_occupancy()` no
longer touches the Occupancy Map at all; the progressive per-batch updates
are already correct on their own.

Cell classification has 4 tiers (`_classify_state`/`CLASS_*`): unknown
(not enough agreeing evidence, or logodds in the uncertain middle band),
ground, **low/step-over** (occupied, height below `STEP_OVER_MAX_H`, default
0.40 m — a curb or low obstacle a user can step over), and normal obstacle
(occupied, height at/above that) — since the camera sits at the user's eye
height, a binary ground/obstacle split can't distinguish those. Exported both per-zone (`extract_subgrid`, existing) and whole-map
(`extract_full_grid`, new — `map_labels.json`'s top-level `occupancy_grid`
field) so `server/tools/grid_path_planner.py`'s `GridPathPlanner` (A*, ground
cheap / low-step-over costlier-but-passable / normal-obstacle blocked /
unknown passable-at-a-premium so unscanned patches don't fragment the map)
can path across zone boundaries to an arbitrary `(x, z)` point — not just
between named zones like `route_planner.py`'s existing greedy zone-centroid
walk. `route_planner.py`'s `find_landmark()` resolves a spoken destination
("the computer") to a landmark's rough `(x, z)` (from `zones[].landmarks[]`,
previously parsed by the scan pipeline but never read back at runtime) when
it doesn't match a zone label; `navigation_tools.py`'s `tool_start_navigation`
tries `find_zone()` first, then `find_landmark()` + `GridPathPlanner`, falling
back to the legacy zone-centroid route only when no `occupancy_grid` exists
(older maps) or A* finds no path. `LiveSessionState.nav_waypoints` (new,
parallel to the legacy `nav_route`) tracks progress along an A*-planned route
in `live_session.py`'s `_check_proximity` — landmarks have no AABB, so
arrival/waypoint-reached use fixed radii instead.

**Fixed for RTAB-Map pose mode:** loop closure corrects RTAB-Map's own pose
graph retroactively — `rtabmap_server.cc`'s `TRACK` reply carries a
`loop_closure` flag (`rtabmap_client.TrackedFrame.loop_closure`) whenever
that frame's processing closed one; `ScanSession._rtabmap_full_resync()`
reacts by discarding every RTAB-Map-sourced point/occupancy-map cell
accumulated so far and re-pulling the ENTIRE reconstructed surface
(`rtabmap_client.get_cloud(since_node_id=0, ...)`, RTAB-Map's own
`cloudRGBFromSensorData` per node transformed by its now-corrected
`getLocalOptimizedPoses()`), replayed into the Occupancy Map one node at a
time (never one call for the whole history — see the sunburst-artifact note
below) to keep every already-reconstructed node's cloud at its corrected
position. This is still a known limitation (unfixed) for IMU + VO: its
clouds come from `scan_session.py`'s own DA3-depth back-projection, baked in
at each batch's pose at the time — `pose_graph.py`'s VO path retroactively
fixes keyframe *poses* but never re-projects points already fused into the
cloud/occupancy grid.

**Found via real recordings, fixed**: `rtabmap_server.cc`'s TRACK reply used
to set `loop_closure=1` on ANY accepted loop closure (confirmed correct
against RTAB-Map 0.23's actual `Rtabmap.cpp` source — `_loopClosureHypothesis`
is genuinely reset every `process()` call, not a stale/sticky flag), but
RTAB-Map's `RGBD/ProximityBySpace` (on by default) accepts a loop closure
against ANY spatially-nearby node — during a slow, close-range, single-room
scan the trajectory stays near itself almost the whole time, so this fired
on nearly every frame, each one triggering `_rtabmap_full_resync()`'s
expensive whole-history reprocess (verified: ~14-25s per occurrence on a
real 79-frame recording) — RTAB-Map mode looked like it "re-rendered
everything every frame" instead of building progressively like VO/IMU+VO.
Fixed two ways in `rtabmap_server.cc`: (1) `RGBD/ProximityAngle` tightened
from RTAB-Map's default 45° to 20° (`make_parameters()`), requiring a more
similar viewing angle before a proximity match is accepted at all; (2) even
when RTAB-Map does accept one, the TRACK reply's `loop_closure` flag is only
set if the resulting `getMapCorrection()` moved by more than
`SIGNIFICANT_CORRECTION_TRANSLATION_M`/`_ANGLE_RAD` (2 cm / ~1°) since the
last correction that was reported — compared against a baseline that's only
updated when reported (not every check), so a series of individually-tiny
corrections still accumulates and eventually crosses the threshold rather
than being masked forever. Verified on the same real recording: dropped
from a full resync on nearly every frame to exactly one for the whole
79-frame scan, with `raw_point_count` climbing smoothly batch-to-batch via
the fast incremental path in between — matching VO/IMU+VO's progressive
behavior. Both thresholds may need further tuning for larger/multi-room
scans or faster motion (see `rtabmap_docker/README.md`'s "Known limitations").

**Depth-consistency gate, extended to RTAB-Map pose mode**: the same
warped/displaced-geometry bug documented above under "Depth-consistency
gate" (a coherent, internally-consistent-but-wrong DA3 batch fusing
permanently into the map) also affects RTAB-Map mode — its own RGB-D
odometry and `cloudRGBFromSensorData` reconstruction are handed the exact
same per-mini-batch DA3 depth as IMU + VO, just consumed by RTAB-Map's own
C++ engine instead of `scan_session.py`'s back-projection, so the
IMU+VO-only fix (`feature_tracker.py`'s `_triangulate_depth_agreement`)
never touched it — confirmed by a user report that switching to RTAB-Map
still showed the same warped-wall artifact after that fix landed. RTAB-Map
is an opaque server-side service, though, with no client-side visibility
into which pixels of which frame produced which point in a reconstructed
node — `TrackedFrame`/`ReconstructedNode` had no id linking them together at
all, so the IMU+VO fix's per-frame back-projection skip has no RTAB-Map
equivalent to hook into.

Fixed by extending the wire protocol (`rtabmap_docker/src/rtabmap_server.cc`,
`rtabmap_client.py`): the TRACK reply now carries a `new_node_id` (int32,
-1 if this frame didn't become a node — not every processed frame does),
computed as `slam.getLastLocationId() == data.id() ? data.id() : -1`
(`getLastLocationId()` returns whichever node RTAB-Map most recently added,
which is only THIS frame's id if this exact call added one — confirmed via
Rtabmap.h's public API and verified end-to-end: TRACK-reported node_ids are
always a subset of what a subsequent GET_CLOUD returns). `scan_session.py`'s
RTAB-Map branch reuses `self.tracker.track()` (already reset per session by
`reset_cloud()`, so its VO state never leaks across pose-source choices) as
a side channel purely for `last_depth_trustworthy` — its returned pose is
discarded, RTAB-Map's own pose stays authoritative — and records any
untrustworthy frame's `node_id` into `self._rtabmap_untrusted_node_ids`.
`_rtabmap_process_nodes()` (shared by the incremental pull and the
loop-closure full resync) filters those node ids out before fusion, logging
`[depth-consistency] dropped N/M RTAB-Map node(s)...`; `_rtabmap_last_pulled_
node_id` still advances past a dropped node so it isn't endlessly
re-requested. The untrusted-id set persists across a full resync (a node's
depth quality doesn't change just because loop closure moved its pose) and
is reset in `reset_cloud()` like the rest of per-session RTAB-Map state.
Requires rebuilding `tracking-rtabmap`'s docker image to pick up the new
binary (`docker build -t tracking-rtabmap:latest scan_server/rtabmap_docker/`
— already done and smoke-tested once during this fix: a real TRACK→GET_CLOUD
round trip against the rebuilt server confirmed the reported node_id exactly
matches what GET_CLOUD later returns for that node) — restart the running
`rtabmap` container/service to pick it up.

**TSDF fusion for raw geometry quality (IMU + VO / VO pose sources only)**:
the depth-consistency gate above only rejects OBVIOUSLY bad frames (frac_bad
over threshold); a frame that passes still gets its dense back-projection
permanently baked into `self._raw_cloud_batches`/Live Points/the exported
PLY with zero revision against other frames observing the same surface —
the residual noise still visible after the gate landed. Fixed by adding an
`open3d.pipelines.integration.ScalableTSDFVolume` (`ScanSession.
_tsdf_volume`, `scan_session.py`) integrated alongside (not instead of) the
existing per-frame back-projection in `process_frames_batch`'s Step 3, for
every frame the depth-consistency gate already marked trustworthy (same
skip reused — untrustworthy frames aren't integrated into TSDF either).
`ensure_cloud_built()` now branches: for non-RTAB-Map sessions with any
TSDF-integrated frames, it returns `self._tsdf_volume.extract_point_cloud()`
(cached, only re-extracted when new frames were integrated since the last
call) instead of the old `_raw_cloud_batches` merge — this is the ONLY call
site that changed, so Live Points, PLY export, and
`finalize_voxel_and_occupancy()` all pick up the denoised cloud automatically
with no changes of their own. **Deliberately NOT applied to RTAB-Map pose
mode**: RTAB-Map's own server-side reconstruction is re-transformed by its
CURRENT graph-corrected pose on every loop-closure resync (`_rtabmap_full_
resync`) — a documented advantage over IMU+VO's "loop closure corrects
keyframe *poses* but never re-projects already-fused points" limitation. A
client-side TSDF volume integrated at pose-at-time-of-integration would
reintroduce that exact staleness with no correction path, regressing a
problem RTAB-Map mode's design already solves — so TSDF is additive only
for non-RTAB-Map sessions; RTAB-Map keeps its existing, more-robust geometry
source untouched (`_session_uses_rtabmap`, set once per
`process_frames_batch` call, gates the branch). Step 3's existing back-
projection → `_raw_cloud_batches` → Occupancy Map feed is completely
unchanged for both pose sources — the Occupancy Map's already-tested,
already-correct incremental Bayesian design (its "one call per
representative camera position" invariant is load-bearing, see the
sunburst-artifact note below) is never touched by this change. Verified:
an isolated synthetic test confirmed colors survive TSDF integration
un-desaturated (Open3D's `RGBDImage.create_from_color_and_depth` defaults
`convert_rgb_to_intensity=True`, which would silently gray out Live
Points/PLY colors if left at the default — must be passed `False`
explicitly) and geometry lands at the correct depth; an integration test
through the real `ScanSession` pipeline confirmed TSDF only integrates
trustworthy frames (in lockstep with the depth-consistency gate), only
applies to non-RTAB-Map sessions (RTAB-Map sessions provably keep
`_tsdf_integrated_count == 0` and use the old raw-merge path), and the
extraction cache correctly invalidates only when new frames are integrated.

**Continuous obstacle clearance (2D distance transform, not a 3D ESDF)**:
`grid_path_planner.py`'s A* previously only had 3 flat cost tiers
(ground/low-step-over/obstacle-blocked) — no notion of "how close to a
wall," so a route could hug a wall as tightly as an equally-short route
through open space. A full 3D ESDF (Voxblox/nvblox-style) was considered
and rejected: a pedestrian is floor-constrained (their body occupies a
fixed vertical column at whatever floor position they're at — they can't
exploit a 3D distance field the way a drone or arm could), and those
libraries reintroduce ROS/GPU dependencies this project has deliberately
avoided everywhere (RTAB-Map itself was chosen specifically as a
standalone-corelib, no-ROS design). A plain 2D Euclidean distance transform
over the existing occupancy grid captures the practically useful part at a
fraction of the complexity. `occupancy_map.py`'s `_build_grid_dict()`
(shared by `extract_subgrid`/`extract_full_grid`) now additionally computes
a `"clearance"` field (metres to nearest `CLASS_OBSTACLE` cell) via
`scipy.ndimage.distance_transform_edt` over that window's own class grid —
`scipy.ndimage` was already a project dependency, already imported in this
exact file, zero new dependency. Known, empirically-verified footgun:
`distance_transform_edt` on an all-obstacle-free mask does NOT raise, it
returns nonsense values anchored to the array's (0,0) corner as if that
corner were an implicit obstacle — guarded with an explicit `if
obstacle_mask.any()` check, falling back to a 9.0m sentinel (chosen because
the clearance-cost formula below is already indistinguishable from 1.0x by
that distance). Known caveat, documented not fixed: `extract_subgrid()`'s
per-zone clearance is windowed to that zone's AABB, so an obstacle just
outside a zone boundary is invisible to that zone's EDT — harmless today
because `grid_path_planner.py` only ever reads the top-level grid from
`extract_full_grid()`, which spans every occupied cell in one shot (no
windowing artifact).

`server/tools/grid_path_planner.py`'s `GridPathPlanner` reads this new
`"clearance"` field (`Optional` — older exported maps without it degrade
byte-for-byte to the pre-existing flat-tier behavior, verified) and applies
an exponential-decay cost multiplier — `1.0 + CLEARANCE_PENALTY_SCALE *
exp(-CLEARANCE_DECAY_RATE * clearance_m)` (defaults 4.0/8.0) — on top of
the existing tiered cost, for every non-blocked cell (`CLASS_OBSTACLE`
keeps its hard `_BLOCKED`, unaffected). Bounded, unlike a naive
`1/clearance` formula (max ~5x at clearance=0, never `inf`, so a
wall-hugging corridor stays finitely traversable when it's the only
option) and decays to ~1.0x (no effect) by roughly 1m. `_line_cost()`/
`_simplify()`'s string-pulling already routed through `_cost()`, so it
automatically inherited clearance-awareness with no separate change.
Verified via a direct cost-function comparison (not an open-field A*
routing comparison, which turned out to give A* no actual incentive to
deviate from a straight line when start/goal are fixed points — the real
unit under test is `_cost()`, which both `_astar`'s edge weights and
`_line_cost` already share): two hand-picked, equal-length straight paths
(one hugging a wall at clearance=0.05m, one at clearance=0.25m) cost 58.90
vs. 24.66 respectively with the clearance field present, and identically
16.00 either way with it absent — confirming both the discrimination and
the backward-compat no-op.

**RTAB-Map full-resync SOR caching (found via a real recording, fixed)**: a
loop-closure full resync (`_rtabmap_full_resync`) re-pulls and re-cleans
EVERY node's cloud from scratch, including nodes already SOR'd during an
earlier incremental pull — real profiling on an actual scan showed this
costing ~7.8s of `remove_statistical_outlier` alone on a 1.2M-point, 44-node
resync, almost entirely nodes that had already been cleaned before. Fixed
via `ScanSession._rtabmap_sor_keep_mask` (`scan_session.py`), a per-node_id
cache of SOR's boolean keep-mask: a rigid transform (rotation+translation —
exactly what a corrected pose applies) preserves every pairwise Euclidean
distance between points, so SOR's k-nearest-neighbor-based outlier decision
for a given node is IDENTICAL before and after a pose correction — only the
points' world-space position changes, never which ones are outliers.
`_rtabmap_process_nodes()` now splits incoming nodes into already-cached
(skip SOR entirely, apply the cached mask directly to the freshly re-posed
points) vs. never-seen (run SOR as before, then cache the resulting mask);
falls back to recomputing for a node if its cached mask's length doesn't
match the freshly-pulled point count (defensive — shouldn't happen, since a
node's own stored SensorData never changes once created, but avoids
misapplying a wrong-length mask if it ever does). The cache persists across
resyncs (a node's outlier decision doesn't change just because its pose
did) and is reset in `reset_cloud()` like the rest of per-session RTAB-Map
state. Verified two ways: (1) an isolated test confirmed SOR's keep-mask on
a synthetic cloud with real outliers came back 100% identical before vs.
after an arbitrary rigid transform — the caching optimization's core
assumption, checked empirically rather than assumed; (2) an integration
test through the real `ScanSession._rtabmap_process_nodes()` confirmed the
underlying SOR function is called exactly once across two pulls of the same
node_id (new node, then a simulated resync with a different pose), and that
the fused cloud correctly reflects the NEW pose-transformed points filtered
by the OLD (still-valid) cached mask. Known, accepted imprecision: exact
mask reuse is only guaranteed bit-for-bit on the CPU SOR path (verified
deterministic); this project's GPU voxel/outlier ops are already documented
elsewhere as not perfectly deterministic against themselves between calls,
so a node whose SOR ran via the GPU path could in principle see a
vanishingly small mask drift on a hypothetical re-run — same "harmless for
this pipeline" class of imprecision as that existing caveat, not a new one
introduced by this change (and moot regardless, since the whole point is
the GPU/CPU SOR call doesn't run a second time at all for a cached node).

**Confidence Map (2D top-down heatmap, side-by-side with the Occupancy
Map)**: shows how much agreeing evidence a cell has accumulated,
independent of whether that evidence says free or occupied — a cell with
`logodds` near 0 (barely observed, or genuinely contradictory observations
cancelling out) reads low confidence even if its current best-guess
classification happens to be ground; a cell confirmed many times over
reads high confidence regardless of which way it was classified.
`OccupancyMap.render_confidence_plotly()` computes `confidence = min(
|logodds| / max(LOGODDS_MAX, |LOGODDS_MIN|), 1.0)` per cell — 0.0 at
`logodds==0` (the exact center of `_classify_state`'s "unknown" band),
ramping to 1.0 at full log-odds saturation; a cell with NO entry in
`self._cells` at all reads an explicit 0.0 (unlike the height map's NaN/
blank convention — "no data" and "confidence zero" are the same concept
here). Rendered with Plotly's "Viridis" colorscale (perceptually uniform,
and visually distinct from the height heatmap's "Turbo" so the two
side-by-side panels are never confused at a glance). `_grid_bbox_and_ticks()`
factors out the bbox/tick-array computation both `render_plotly()` and
`render_confidence_plotly()` share (pure duplication otherwise); `_CLASS_NAME`
and `_empty_figure()` are likewise shared. Verified: a cell hit 3x reads
meaningfully higher confidence than one hit once, both bounded in [0,1], and
a genuinely never-touched cell inside the overall bbox reads exactly 0.0.

**Live Navigation Preview (scan_gui.py's "Live Navigation Preview"
accordion, below the Occupancy/Confidence Maps)**: lets the operator pick a
destination — typed X/Z, or a dropdown of the current session's semantic
landmarks (`session.labeler.zones[*].landmarks[*]`, already kept live by
`preview_landmarks()`, called once per processed chunk — no new landmark
plumbing needed) — and see a route computed against the map WHILE SCANNING
IS STILL IN PROGRESS, not just against a finished export. Destination is
NOT set by clicking the map: Gradio's `gr.Plot` (the Plotly wrapper both
maps use) does not fire click/select events in the installed version
(6.19) — confirmed by reading `gradio/components/plot.py`'s `EVENTS =
[Events.change]` (no `select`) vs. `native_plot.py`'s `NativePlot` (which
does support `select`, but is Gradio's own Vega-Lite bar/line/scatter chart
component, not a Plotly heatmap host) — so click-to-target was dropped in
favor of number inputs, keeping the existing maps' hover/zoom fully intact.

`scan_server/live_path_planner.py`'s `LiveGridPathPlanner` is an adapted
copy of `server/tools/grid_path_planner.py`'s `GridPathPlanner` — same
CLASS_* constants, same cost model (ground cheap, low/step-over
costlier-but-passable, obstacle blocked, unknown passable-at-a-premium),
same clearance-aware A* and cost-aware string-pulling — duplicated rather
than imported for the exact reason that file's own docstring already gives
for its CLASS_* constants: `server/` and `scan_server/` are separately
deployed processes/environments, and there's no `map_labels.json` yet for a
scan still in progress to serve as an integration point anyway. Built fresh
each time from `OccupancyMap.extract_full_grid()`'s in-memory dict — no
file I/O. The "unknown passable-at-a-premium, never hard-blocked" cost tier
already IS the "flood toward unexplored territory" behavior that was asked
for: a fully-confirmed route is naturally preferred (it's cheaper), but the
search still flows through unexplored cells when that's the only way to
reach the destination, rather than failing outright — no separate two-pass
"try confirmed, then fall back" algorithm was needed, just running the
already-proven single-search design live instead of only at static export
time. The one addition over the deployed planner: `find_path()` also
reports `confirmed: bool` — False if the returned route had to cross any
`CLASS_UNKNOWN` cell — so the caller can flag a speculative/exploratory
route distinctly from a confirmed one (`occupancy_map.py`'s `_overlay_route()`
draws it dashed orange vs. solid green on both maps).

`OccupancyMap._update_count` (bumped once per real `update()` call, i.e.
whenever `cloud_points` clears the existing `< 10` early-return — not on
every call unconditionally) is the "did the grid actually change" signal
`scan_gui.py`'s per-chunk loop uses to decide whether to recompute the
route ("constantly extend the navigation... as new data comes in", without
recomputing redundantly on a chunk that added nothing). Verified end to end
through the real `_run_simulated_stream` pipeline (retrieved from the built
Gradio app's `app.fns` registry, same technique used earlier this session,
against a real uploaded dataset): across 19 real occupancy updates, exactly
19 distinct `nav_state["computed_at"]` values were recorded and exactly 19
`LiveGridPathPlanner` instances were constructed — a precise 1:1 match,
proving recompute fires exactly once per real grid change, never
redundantly. `find_path()`'s own confirmed/speculative/unreachable
correctness was separately verified in isolation with hand-built grids
(mirroring the clearance-cost verification approach from earlier this
session): an all-ground grid returns `confirmed=True`; a destination only
reachable through one unexplored gap in a wall returns `confirmed=False`;
a destination behind a solid, gapless wall correctly returns `None`
(genuinely unreachable, even through unexplored territory).

**Scope note**: the automatic per-chunk recompute is wired into
`_run_simulated_stream` (the continuous auto-replay mode — the natural fit
for "as new data comes in") only. Manual mode (`_manual_stream_feed`, one
frame per click) already has 4 fixed-tuple-position return branches from
earlier work in this session; threading an additional `gr.State` through
all of them was judged higher-risk than the value it added, since the
operator is already clicking through frame-by-frame in that mode anyway —
they can click "Find Route" again after a step to refresh it, which works
identically to the auto-replay mode's button (both call the same
`_nav_find`/`_nav_route_full_and_status` helpers).

**Confidence-weighted occupancy updates**: every hit used to nudge a
cell's log-odds by the same fixed amount regardless of how trustworthy the
DA3 depth behind it was — so a cell built entirely from borderline (but
not outright-rejected) depth could reach the same reported confidence as
one built from solid measurements, given enough hits. DA3 itself exposes
no usable per-pixel confidence for the model this project uses (checked:
the `Prediction.conf` field exists in the package's dataclass but isn't
populated here), and RTAB-Map doesn't solve this either — it's built for
real depth sensors with roughly homogeneous noise, no concept of "this
pixel came from a monocular NN guess." The one *measured* (not heuristic)
signal already in this codebase is the depth-consistency gate's per-frame
`frac_bad` — reused here as a continuous confidence weight instead of only
a binary accept/reject. `OccupancyMap.update()` takes a new `confidence:
float = 1.0` param, stored as `self._current_confidence` for that call and
multiplied into every log-odds delta in `_register_obstacle_hit`/
`_register_ground_hit`/`_register_miss` (Bayesian mode only — non-Bayesian
mode's "single hit = permanent" design has no incremental belief to
scale). `scan_session.py` computes a BATCH-level (not per-point) confidence
from the existing `depth_checks`/RTAB-Map per-frame side-channel:
`mean(1 - frac_bad)` across the batch's frames, defaulting to 1.0 if none
had a usable check. Batch-level, not per-point, is deliberate — the
depth-consistency check is already per-FRAME, and per-point confidence
would need interpolating a sparse keypoint signal across the whole dense
cloud, a much larger and more fragile undertaking for uncertain payoff.
This is standard Bayesian evidence weighting, not a hard cap — verified:
enough agreeing low-confidence observations can still saturate to the same
ceiling given enough of them (correct if the underlying errors are
independent noise); it does NOT protect against DA3 systematically
misjudging the exact same real surface the same way every time, which is a
correlated bias, not independent noise — a known, honestly-documented
limitation, not oversold as a full fix.

**Closest-approach navigation + minimum path width**: `LiveGridPathPlanner.
find_path()` (`live_path_planner.py`) no longer fails outright when the
exact destination isn't reachable (blocked, or literally inside an
obstacle cell) — it now returns a route to the CLOSEST reachable cell
instead, computed during the SAME A* expansion (no second search): the
search already tracks, for every visited cell, its heuristic distance to
the goal; if the goal itself is never reached, the cell with the smallest
such distance becomes the fallback destination. Return type changed from
`(waypoints, confirmed)` to `(waypoints, confirmed, reached_exactly)` —
`reached_exactly=False` flags a closest-approach result distinctly from a
route that genuinely reaches the requested point. Only returns `None` when
truly nothing useful can be offered (start itself not passable, or the
search can't move anywhere at all — verified via a fully sealed-off start
cell). Also added `min_path_clearance_m` (constructor param, 0=disabled,
GUI-exposed as "Minimum path width" in the Live Navigation Preview
accordion): an additional bounded penalty in `_clearance_multiplier` for
cells narrower than this desired width, on top of the existing exponential
clearance shaping — never a hard block (same "never fragment the map"
philosophy as the rest of this cost model), just more strongly discouraged.
Verified via the same direct-cost-comparison technique used for the
original clearance feature: an equal-length wall-hugging path's cost
roughly doubled relative to an already-wide-open path once a minimum width
was configured, while the wide-open path's own cost was unaffected (its
clearance already exceeds the minimum). `scan_gui.py`'s `nav_state` carries
`min_clearance` alongside `target`/`route`, set whenever "Find Route" is
clicked and read by the per-chunk auto-recompute loop — avoids threading
yet another positional parameter through `_run_simulated_stream`'s already
long signature.

---

## Client Implementations

**Android (`client/android/`) is the only client this project ships** —
documented in full under Key Files Map's `android/` block and this file's
"Client-Orchestrated Live Session" section. It runs Gemini Live directly
on-device and calls `TrackingService`/`PerceptionService`/`MappingService`
for heavy compute; no other RPC surface exists.

Every other client that used to exist here — the Pi thin client
(`client/pi_client.py`), Mediator (`client/mediator_gui.py`), Desktop video
GUI (`client/desktop_video_stream_gui.py`), Edge main
(`client/edge_main.py`), the headless/mock test clients
(`client/headless_edge_client.py`, `client/android_mock_client.py`), and
their shared support code (`client/rpc_client/`, `client/core/`,
`client/proto/`) — has been **deleted**, along with the server-side RPCs
(`MediatorService`, the old zone-based `MapService`) and orchestration
(`server/live_session.py`, `server/live_tools/`) that existed only to serve
them. `test_module/mock_frame_server.py` (mocked the now-deleted Android
Scan screen) was deleted too. If any of this is ever needed again, it's in
git history — it wasn't archived in-tree.

---

## Key Files Map

```
tracking.proto                   gRPC + protobuf definitions (edit here, regenerate stubs)
tracking_pb2{,_grpc}.py          Generated — DO NOT EDIT (exists in server/, test_module/)

server/
  grpc_server.py                 Main server entry point; loads models; starts gRPC + Gradio.
                                   No more ToolsBundle/WalkingConfig wiring or MapServiceServicer
                                   registration — those existed only for the deleted orchestration.
                                   Builds one ActivityMonitor, passed into every servicer (including
                                   StatusServiceServicer, always registered — unlike MappingService
                                   it needs no RTABMAP_ADDR/model deps) and into server_gui.create_ui().
  server_gui.py                  Gradio dashboard (port 7860), rewritten around ActivityMonitor —
                                   4 tabs (Tracking / Perception / "Mapping + Beacon" / Activity Log),
                                   each showing the last frame + result for that RPC category.
                                   Tab auto-selection prefers ActivityMonitor.client_mode
                                   (StatusService.ReportMode, authoritative) over inferring from
                                   whichever RPC category most recently updated (fallback for older
                                   clients) — "walking" maps to tab_mapping now, same as guiding/
                                   scanning (was tab_perception for a while, back when walking didn't
                                   touch MappingService at all — see "Local reactive HRTF
                                   obstacle-dodge" — fixed once walking rejoined it, see "Grid-planned
                                   walking route"). Mapping tab shows the last frame and
                                   occupancy_map.py's own render_plotly() SIDE BY SIDE in one row,
                                   against the LIVE OccupancyMap reference ActivityMonitor's mapping
                                   bucket holds — deliberately no point-cloud/voxel/confidence view (those
                                   stay scan_gui.py's separate, heavier offline debug tool —
                                   render_confidence_plotly() was shown here too originally, dropped as
                                   unnecessary for this at-a-glance live dashboard); wrapped in try/except
                                   since that reference is mutated concurrently by the gRPC streaming
                                   thread while this renders on the Gradio polling thread — a race should
                                   skip a tick, not crash the dashboard. render_plotly() is now called
                                   with route=[pose]+planned_path/route_confirmed=path_confirmed (see
                                   "Server-planned walking path") — the same route/route_confirmed params
                                   scan_gui.py's Live Navigation Preview already exercises, reused
                                   directly. _annotate_mapping() ALSO projects that same planned_path onto
                                   the raw frame now (new _project_world_to_pixel() helper, world→camera
                                   pinhole projection using pose_proto+ground_y, same "no real
                                   calibration, guess a pinhole K" convention as scan_session.py's
                                   _estimate_K) — drawn as a cyan polyline, distinct from the green pose
                                   text / red tracking-lost warning already on that overlay. RTAB-Map
                                   pose-lost is now surfaced directly (not just console): a red "RTAB-Map
                                   TRACKING LOST (N/M)" burned into the
                                   annotated frame (frame_rgb is RGB order, red=(255,0,0)) plus a line in
                                   the Detail textbox, both driven by ActivityMonitor's rtabmap_lost/
                                   rtabmap_total fields (mapping_servicer.py) — see "Blur filtering
                                   removed for scan/walking" above. The Mapping tab also carries the
                                   beacon-direction panel — _render_beacon_polar()/_beacon_status(), a
                                   Plotly polar chart of the last AnalyzeFrame(TRAVERSABILITY) fan plus a
                                   marker at the client-reported final azimuth (ActivityMonitor.
                                   beacon_azimuth_deg/beacon_muted, fed by StatusService.
                                   ReportBeaconDirection) — moved here from a standalone Perception tab
                                   once walking's steering itself moved to the occupancy grid (see
                                   "Grid-planned walking route"): GUIDING's marker is scored straight off
                                   the plotted fan, WALKING's marker instead comes from its grid-planned
                                   waypoint bearing (the fan shown for WALKING is only its separate
                                   step-down/drop-off hazard check). The Perception tab itself still
                                   exists, narrowed to ad-hoc run_detection/check_obstacle debug traffic
                                   only. The old magenta beacon-position circles on the Mapping tab's frame/
                                   occupancy-map views are REMOVED (beacon_preview.py, below, is
                                   deleted — the beacon is a pure steering angle now, not a world
                                   position, so there's nothing left to project onto those views). No
                                   more Chat/Reading tabs or zone-based nav map (those depended on
                                   the deleted server-side LiveSessionState/session.conversation_log/
                                   session._route_planner — this file no longer reads a `servicer`
                                   or `session` object at all, only the ActivityMonitor).
  services/
    activity_monitor.py          ActivityMonitor — thread-safe "last RPC per category" snapshot
                                   (tracking/perception/mapping buckets + a rolling event log),
                                   fed by all 4 servicers below as real client RPCs land, polled by
                                   server_gui.py. Replaces the deleted LiveSessionState/session-based
                                   dashboard feed. Also holds client_mode/client_mode_target/
                                   client_mode_at (record_client_mode(), fed by StatusServiceServicer
                                   only) — the client's own explicit, authoritative mode report (see
                                   "Mode exclusivity + server-reported client mode" above); server_gui.py
                                   prefers this for tab selection, falling back to inferring from
                                   whichever RPC category most recently updated only when absent
                                   (older client builds that predate StatusService). Per-category
                                   buckets are MERGED on each record, not cleared — a sticky field
                                   like mapping's occupancy_map (only set by UpdateMapping) would
                                   otherwise flicker away whenever an interleaved FindLandmark call
                                   recorded into the same bucket without it.
    servicer.py                  TrackingServiceServicer — DetectObject/GetEmbedding only, each
                                   request carries its own image_data (no server-side "latest frame"
                                   cache any more). Records into ActivityMonitor's tracking bucket
                                   (frame, prompt/box/score or embedding dim) on every call.
    perception_servicer.py       PerceptionServiceServicer — AnalyzeFrame/Synthesize/Embed. Records
                                   into ActivityMonitor's perception bucket (frame + detections/
                                   obstacle/traversability for AnalyzeFrame, text for Synthesize/
                                   Embed). Handles the TRAVERSABILITY op via depth_detector.
                                   estimate_traversability() — see "Local reactive HRTF obstacle-dodge".
    mapping_servicer.py          MappingServiceServicer — UpdateMapping/FindLandmark. No disk
                                   persistence at all (see "Map persistence removed entirely" —
                                   GetMapSnapshot/ListMappedLocations were deleted outright, not
                                   just disabled; every session starts from a blank map). Records
                                   into ActivityMonitor's mapping bucket on every yielded MappingUpdate (frame, pose,
                                   grid_updated, confidence, and a LIVE reference to
                                   session.occupancy_map for server_gui.py to render on demand — not a
                                   snapshot, see activity_monitor.py's merge-not-clear note) and every
                                   FindLandmark call. Decides full-vs-delta occupancy grid sync per
                                   update (_last_full_bounds cache vs. OccupancyMap.bounds()) — see
                                   "Occupancy grid delta sync" above. No longer resolves any beacon-
                                   direction preview (beacon_preview.py and its _resolved_destinations/
                                   _last_open_direction caches were deleted outright — see "Local
                                   reactive HRTF obstacle-dodge"; the beacon's actual direction is now
                                   reported directly by the client via StatusService.
                                   ReportBeaconDirection instead of reconstructed server-side).
                                   UpdateMapping reads `for chunk in _latest_only_chunks(request_iterator):`
                                   — re-adopted drop-to-latest (tried, reverted, then re-adopted; see
                                   "Drop-to-latest mapping-chunk ingestion" below for the full history and
                                   the known risk this accepts). record_mapping() also now passes
                                   pose_proto (the full Pose, not just pose_x/pose_z), planned_path
                                   (world x/z points), path_confirmed, and ground_y — so server_gui.py can
                                   draw the server-planned route on both the occupancy map and the raw
                                   frame (see "Server-planned walking path").
                                   No configure_novelty_gate() call any more — blur filtering was tried
                                   (SCAN_MIN_SHARPNESS/WALKING_MIN_SHARPNESS) then removed entirely per
                                   "Blur filtering removed for scan/walking" above; self._min_sharpness
                                   stays at DEFAULT_MIN_SHARPNESS (0.0, blur gating off). record_mapping()
                                   now also passes rtabmap_lost/rtabmap_total (from session.last_rtabmap_
                                   lost/last_rtabmap_total) so server_gui.py can show RTAB-Map pose-lost
                                   status directly instead of only in console output.
                                   pure_walking (SessionMode.WALKING) flows through the SAME grid/delta/
                                   full_resync logic GUIDING does now — see "Grid-planned walking route"
                                   (an earlier, now-superseded corridor-lock design special-cased
                                   pure_walking into a pose-only branch here; that branch is gone). The
                                   only pure_walking-specific behavior left in UpdateMapping: a
                                   session.last_reset_occurred check (before the normal "no pose yet"
                                   gate, so a just-fired reset's one signal update isn't silently
                                   swallowed) that also pops self._last_full_bounds[location_id] to force
                                   the next real update to a full resync; and skipping
                                   finalize_landmarks_flat()/snapshot-save in the finally block (nothing
                                   meaningful ever accumulates for a never-persisted walking session).
                                   server_gui.py's _TAB_BY_CLIENT_MODE["walking"] now points at
                                   tab_mapping to match (was a known gap, pointing at tab_perception —
                                   see "Grid-planned walking route"'s dashboard-fix note).
                                   UpdateMapping now also builds a LiveGridPathPlanner (live_path_planner.py,
                                   below) — a SEPARATE instance per mode, not shared — from
                                   extract_full_grid() every update and computes a PlannedPath
                                   (find_natural_path() for WALKING, find_path() toward chunk.goal_x/z
                                   for GUIDING) — path PLANNING is server-side now, not just the grid —
                                   see "Server-planned walking path + client-side latency bridging" and
                                   "Natural path planner". A per-location_id _prev_path/_prev_goal_xz
                                   cache now gates whether either planner is even called this update —
                                   see "Path stability" above: the old route is reused unchanged unless
                                   planner.path_blocked_ahead() says its immediate next segment is now
                                   blocked, and a genuine replan biases toward the old route rather than
                                   starting over (WALKING: reference heading = bearing toward the old
                                   route's own first joint; GUIDING: find_path()'s new
                                   previous_path_xz param). WALKING's planner is now constructed plain
                                   (LiveGridPathPlanner(grid_for_planning), no min_path_clearance_m —
                                   find_natural_path() has its own independent safe_clearance_m concept
                                   instead, _WALKING_SAFE_CLEARANCE_M=0.5, repurposed from the old
                                   _WALKING_MIN_PATH_CLEARANCE_M constant), called with max_distance_m=
                                   _WALKING_MAX_PLANNING_DISTANCE_M (5.0, new — replaces the old
                                   _WALKING_MAX_TURN_RAD). GUIDING's planner is unaffected — still
                                   LiveGridPathPlanner(grid_for_planning) + find_path(pose_xz,
                                   current_goal_xz). heading_rad (_pose_heading_rad()) is now computed
                                   UNCONDITIONALLY every update (previously WALKING-branch-only) so
                                   server_gui.py can draw it for both modes — see "Facing-direction GUI
                                   indicator". New _planned_path_to_proto() helper (unchanged).
                                   grid/grid_delta are still computed and sent unchanged even though the
                                   client no longer reads them for planning (deliberately deferred cleanup,
                                   see that section's own limitations note).
    status_servicer.py           StatusServiceServicer — ReportMode + ReportBeaconDirection (new).
                                   Neither carries data any other service needs; both record straight
                                   into ActivityMonitor (record_client_mode()/record_beacon_direction()).
                                   ReportBeaconDirection is dashboard-only like ReportMode, but called
                                   once per local-avoidance tick (far more often) — no console print for
                                   it, unlike ReportMode, since that cadence would spam the log.
  ARCHITECTURE.md                Detailed server internals (component map, data flows, tool→function map;
                                   describes the pre-migration architecture, not fully updated)
  tools/
    detector.py                  GroundingDINO wrapper
    depth.py                     Obstacle detector: DA3DepthDetector only — Depth Anything 3 ONNX
                                   (build_estimator(onnx_path=...), default DA3METRIC-LARGE.onnx, see
                                   DA3_ONNX_PATH). Uses DA3OnnxEstimator's "metric_depth" output head
                                   directly (da3_wrapper.py) — no separate scale-alignment pass, since
                                   the DA3-METRIC checkpoint already outputs real metric depth, unlike
                                   a relative-depth model. An earlier version fit DA3's depth against
                                   sparse ORB-triangulated anchors via a scan_server/mvs.py helper that
                                   never actually existed in this repo (a latent, now-removed bug —
                                   that whole alignment step was solving a problem this model doesn't
                                   have). SparseObstacleDetector (ORB-only, relative depth) and
                                   StereoDepthDetector (plane sweep MVS) were removed — this is now
                                   the only depth path for PerceptionService.AnalyzeFrame's DEPTH op,
                                   so DEPTH_MODEL is no longer read. Refactored to share one
                                   _depth_map() DA3 call between check_obstacle() and
                                   estimate_traversability() (delegates to traversability.py, below) —
                                   see "Local reactive HRTF obstacle-dodge" (GUIDING) / "Hazard warnings"
                                   (both modes' step-down check). find_corridor() — added for the
                                   corridor-lock design, then deleted outright once that design was
                                   superseded by "Grid-planned walking route" — see that section.
    traversability.py            estimate_traversability(depth_map, num_bins, max_range_m) —
                                   stateless, single-frame polar obstacle-clearance fan: back-project via
                                   the same pinhole-K fallback used throughout this codebase, RANSAC-fit
                                   a ground plane from the frame's bottom ~40%, classify obstacle vs.
                                   ground by height above it, bucket by azimuth. No IMU, no persisted
                                   ground_y (deliberately NOT occupancy_map.py's accumulated-belief
                                   design — a world map is too slow for reactive per-frame dodging). Two
                                   real bugs found via synthetic ground-truth testing before this
                                   shipped (sign-flip using the wrong reference point; a frontal
                                   obstacle filling the frame getting accepted as "the floor") — see
                                   "Local reactive HRTF obstacle-dodge" for both (GUIDING still uses
                                   this function directly, unchanged). Its ground-fit + azimuth-binning
                                   body lives in _clearance_fan_from_depth() (no logic change from the
                                   original inline version), which also computes dropoff_m (on
                                   TraversabilityResult) — nearest BELOW-ground-plane distance per bin,
                                   the step-down/stairs counterpart to clearance_m's above-plane
                                   obstacles — see "Hazard warnings — step-down frames fed to Gemini
                                   Live", still current and unaffected by find_corridor()'s removal
                                   (below). find_corridor()/CorridorResult — added to let WALKING select
                                   a single "widest open arc" corridor target for the world-anchored-lock
                                   design, deleted outright once that design was superseded by
                                   "Grid-planned walking route" (walking's beacon comes from an actual
                                   occupancy-grid path now, not a per-frame corridor pick).
    rag_store.py                 Sentence-transformer text embeddings (embed_text(), backs PerceptionService.Embed);
                                   storage/search methods (add_text/query_global) are now unused server-side —
                                   storage lives on Android (LocalMemoryStore.kt) — kept for reference/DummyRagStore
    embedder.py                  DINOv2Embedder (ViT-S/14) — visual re-ID embeddings
    tts.py                       KokoroTTS.synthesize_pcm_chunks() — backs PerceptionService.Synthesize
  _archived/                     Old orchestrator/, agents/, cloud_vlm, intent_parser (reference only)
  data/
    maps/{location_id}/          No longer written/read by the live path at all — MappingService
                                   has no disk persistence any more (see "Map persistence removed
                                   entirely"). Any occupancy_snapshot.json/legacy map_geometry.ply/
                                   map_labels.json files here from before that change are dead data,
                                   not deleted by it

scan_server/
  scan_server.py                 Entry point; plain Gradio launch (no HTTP API — see "Server-side
                                   scan_server.py" above), port 7861. _build_ui() loads the DA3
                                   estimator/GroundingDINO/Gemma/RTAB-Map client and returns
                                   create_scan_ui(...); __main__ just demo.launch(...).
  scan_gui.py                    Gradio UI — dataset folder path + segment table → export.
                                   create_scan_ui(scan_manager, upload_dir) — "Load from Video" accordion
                                   (was "Load from Android Upload"): gr.Video upload + "Extract Frames"
                                   button, _extract_video() decodes it via cv2.VideoCapture into
                                   uploads/<scan_id>/dataset/ (images/+camera.csv, no imu.csv — see
                                   "Server-side scan_server.py" above) and writes the result straight
                                   into dataset_path_input, same as "Load Selected" already did for a
                                   past upload; _upload_dir defaults to a local scan_server/uploads/
                                   folder when upload_dir isn't passed in. dataset_path_input then
                                   pre-fills the dataset folder path for everything below;
                                   _handle_dataset_change()/
                                   _read_camera_index() read camera.csv for real per-frame timestamps
                                   (no assumed constant FPS); IMU Orientation dropdown (portrait/
                                   landscape-left/landscape-right) passed to session.set_imu_file() —
                                   rotates raw imu.csv into the frame the phone was actually held in
                                   for that recording (see 3D Scanning Pipeline / scan_session.ImuIntegrator).
                                   No batch "Scan" button anymore — every run replays the dataset
                                   frame-by-frame through StreamingScanSession (see 3D Scanning
                                   Pipeline's "Streaming interface" note); the old _run_local_scan
                                   handler and its _collect_frames_in_range()/compute_da3_windows()
                                   helpers were deleted along with it. Only 2 pose sources remain
                                   (IMU + VO, RTAB-Map) — "Auto"/"VO only"/"DA3 poses" and their
                                   DA3_WINDOW/DA3_STRIDE sliding-window pose stitching were removed
                                   entirely (see 3D Scanning Pipeline). Two ways to drive a
                                   scan now, both against the same dataset folder + Segment Table:
                                   "Simulated Live Stream" button (_run_simulated_stream) — auto,
                                   replays the whole dataset via stream_simulator.replay_dataset(),
                                   optionally real-time-paced (realtime_pacing_cb); and "Start / Reset
                                   Manual Stream" + "Feed Next Frame" buttons (_manual_stream_start/
                                   _manual_stream_feed) — single-step, via
                                   stream_simulator.ManualDatasetReplayer: Start builds a
                                   ManualDatasetReplayer (wrapping a fresh StreamingScanSession) and
                                   shows a small preview (manual_preview_image) of the first not-yet-fed
                                   frame; each "Feed Next Frame" click applies any IMU samples/zone
                                   boundaries before that frame, pushes exactly it, updates the preview
                                   to the next upcoming frame, and disables the button once
                                   replayer.has_more() is false — at which point the session
                                   auto-finalizes + exports, same as Simulated Live Stream's own
                                   end-of-replay step. manual_replay_state (gr.State) holds the
                                   ManualDatasetReplayer across separate per-click callback
                                   invocations (Gradio callbacks are otherwise stateless per request).
                                   "Live Reconstruction" tab — single tab (Live Points + Voxelization
                                   side by side, Occupancy Map below) replacing the old separate Live
                                   Points/Voxelization/Occupancy Map tabs. All 3 rebuild continuously,
                                   once per processed chunk, across all 3 ways of driving
                                   StreamingScanSession above (_run_simulated_stream/
                                   _manual_stream_feed) — Live Points -> Voxelization -> Occupancy Map
                                   as one visible pipeline (see 3D Scanning Pipeline's "Progressive,
                                   Bayesian, SLAM-style Occupancy Map" note for how the Occupancy Map
                                   feed itself changed). show_live_points_cb/show_voxelization_cb/
                                   show_occupancy_cb checkboxes (default all ON) gate BOTH whether
                                   that view's component is visible (wired via .change() ->
                                   gr.update(visible=...), independent of any running generator) AND
                                   whether the per-chunk recompute happens at all — Live
                                   Points/Voxelization recompute cost grows with total scan size
                                   (ensure_cloud_built()/voxelize_cloud() have no incremental-merge
                                   primitive in Open3D), unlike the Occupancy Map's genuinely
                                   incremental update(), so unchecking one is the actual performance
                                   control for a large scan; both _run_simulated_stream and
                                   _manual_stream_feed track a skip-if-unchanged point count
                                   (_last_render_point_count locally / replayer.last_render_point_count
                                   on the ManualDatasetReplayer, since manual mode has no generator
                                   closure to hold it across clicks) so a chunk that added zero new
                                   points doesn't trigger a wasted rebuild either way. Single "Reload"
                                   button (_reload_all_views) refreshes all 3 at once regardless of
                                   checkbox state, for after Export or when nothing is actively
                                   streaming; checking a box back on also immediately refreshes just
                                   that view (_toggle_live_points/_toggle_voxelization/_toggle_occupancy).
                                   Height Filter tab removed (was rebuild_occupancy_map()-based,
                                   obsoleted by the progressive Occupancy Map and the ray-casting bug
                                   it triggered — see that note). "Occupancy Map Settings" accordion
                                   exposes every occupancy_map.py Bayesian/height constant as a slider
                                   (height tiers, log-odds hit/miss weights + confirm thresholds, height
                                   EWMA α, plus enable_ray_casting/enable_bayesian checkboxes) — read
                                   fresh at the start of every run (Simulated or Manual) via
                                   session.configure_occupancy_map(); only affects future update()
                                   calls, so a map already built under old settings needs Clear Cloud +
                                   re-run to fully reflect new ones. voxel_size_input (moved into the
                                   Live Reconstruction tab) now does double duty — Voxelization's
                                   display voxel size AND, via occupancy_voxel_size, what each chunk's
                                   new points get coarsened to before feeding the Occupancy Map.
                                   "Live Reconstruction" tab's Occupancy Map is now a Row of TWO
                                   plots side by side — occupancy_plot (height heatmap) and
                                   confidence_plot (render_confidence_plotly()) — sharing
                                   show_occupancy_cb for visibility/recompute (one checkbox, both
                                   plots; see occupancy_map.py). Below that, a "Live Navigation
                                   Preview" accordion: nav_target_x/nav_target_z (gr.Number) or
                                   nav_landmark_dropdown (populated from session.labeler.zones[*]
                                   .landmarks[*]) set a destination; nav_min_clearance_input (gr.Slider,
                                   0=disabled) sets LiveGridPathPlanner's min_path_clearance_m; nav_find_btn
                                   computes a route via live_path_planner.LiveGridPathPlanner against the
                                   in-progress occupancy grid, overlaid on both plots (_overlay_route()).
                                   nav_state (gr.State: target/route/confirmed/reached_exactly/
                                   min_clearance/computed_at) persists the destination across per-chunk
                                   callback invocations, matching
                                   manual_replay_state's existing pattern. _run_simulated_stream
                                   recomputes the route automatically each chunk when
                                   occupancy_map._update_count advances (see 3D Scanning Pipeline's
                                   "Live Navigation Preview" note) — Manual mode does not auto-
                                   recompute (scope decision, see that note); re-click Find Route
                                   there instead. Destination is NOT click-on-map: gr.Plot (Plotly)
                                   doesn't fire select events in this Gradio version (6.19) — confirmed
                                   via gradio/components/plot.py's EVENTS list.
  scan_css.py                    CSS + theme + header/description HTML for Gradio UI
  timing_utils.py                timed() context manager — "[timing] <label>: <ms>" console log,
                                   used across scan_session.py/scan_gui.py/occupancy_map.py for every
                                   Live Reconstruction pipeline stage (depth estimation through GLB/
                                   Plotly rendering). Its own tiny module, not folded into
                                   scan_session.py, specifically so occupancy_map.py (which
                                   scan_session.py already imports FROM) can use it too without a
                                   circular import
  scan_session.py                ScanSession + ScanSessionManager (per-location state)
                                   process_frames_batch()'s pure_walking param (SessionMode.WALKING
                                   specifically, not GUIDING) enables a consecutive-tracking-loss
                                   reset check (PURE_WALKING_LOST_RESET_S=1.0s unbroken RTAB-Map pose
                                   loss — deliberately short, see "Grid-planned walking route"; bumped
                                   from 0.5s alongside re-adopting the drop-to-latest mailbox, see
                                   "Drop-to-latest mapping-chunk ingestion"). Step 4's
                                   occupancy_map.update()/_merge_voxels() is NO LONGER skipped for
                                   pure_walking (an earlier corridor-lock-era design skipped it; the
                                   current design needs the live grid for WALKING's server-planned
                                   route same as GUIDING — see "Server-planned walking path"). The
                                   confidence WEIGHT passed to occupancy_map.update() is forced to 1.0
                                   for pure_walking specifically (was the real computed
                                   batch_confidence, same as GUIDING/SCAN still use) — see "RTAB-Map
                                   confidence weighting skipped for WALKING". last_batch_confidence
                                   itself still reflects the real computed value regardless of mode
                                   (dashboard-diagnostic use only). reset_cloud() was split into a lock-free
                                   _reset_cloud_locked() body + thin wrapper so the reset check (already
                                   inside process_frames_batch's own self._lock) can call it without
                                   deadlocking (threading.Lock isn't reentrant).
                                   set_imu_file() loads imu.csv (ImuIntegrator); compute_segment_poses(frame_timestamps_ns)
                                   pre-computes per-frame IMU poses from camera.csv's actual timestamps;
                                   process_frames_batch()'s Step 0 (new) runs a blur pre-check —
                                   _sharpness_score() per frame, BEFORE Step 1 (DA3) — and drops any frame
                                   below self._min_sharpness from frames_rgb/imu_poses/frame_timestamps_ns
                                   before DA3/pose ever run on it; if nothing survives, the whole batch
                                   short-circuits with 0 infer_ms and no DA3/RTAB-Map call at all — see
                                   "DA3 model default + per-frame processing + pre-DA3 blur gate" above.
                                   Step 3's RTAB-Map branch reuses this precomputed sharpness (frame_sharpness
                                   list) instead of recomputing it, and branches THREE ways per frame
                                   (tracking-lost / walking_lite blur-only / scan's full novelty+blur gate) —
                                   see "Walking/guiding skip the novelty gate entirely" above (blur is
                                   currently always off live-path-wide, see "Blur filtering removed for
                                   scan/walking", so Step 0 is a permanent no-op there and the walking_lite
                                   branch reduces to "accept whenever RTAB-Map produced a pose"). The
                                   [timing] depth estimation/[timing] pose computation/[timing] back-
                                   projection + frame-store tagging prints only fire when walking_lite —
                                   scan's console output was trimmed for readability during live debugging
                                   (same section). Then picks one of 2 pose sources (IMU + VO or
                                   RTAB-Map) — see 3D Scanning Pipeline. Also has IncrementalImuIntegrator,
                                   the streaming counterpart to ImuIntegrator (sample-by-sample push() instead
                                   of a whole-CSV load; fixed ~1s time window for gravity init instead of
                                   "first 5% of file", since a live source doesn't know its total length).
                                   self._cloud is NOT kept live during scanning by default — each batch's
                                   own back-projected points get voxel-downsampled + outlier-removed once
                                   (cheap, small, per-batch) and buffered raw in self._raw_cloud_batches;
                                   ensure_cloud_built() merges these into self._cloud — called on demand
                                   (Reload/Voxelize/Export/finalize_voxel_and_occupancy) AND now also once
                                   per processed chunk during Scan/Simulated Live Stream when the GUI's
                                   Live Points checkbox is on (see scan_gui.py's "Live Reconstruction"
                                   tab) — incremental via self._merged_batch_count (only NEW batches
                                   since the last call get concatenated into self._cloud_raw_accum, a
                                   separate always-raw accumulator kept apart from self._cloud
                                   specifically so re-voxel-downsampling never re-averages an
                                   already-computed centroid as if it were one point); the final
                                   voxel_down_sample pass is still O(current total size) every call
                                   regardless (no incremental voxel-merge primitive in Open3D) — the
                                   GUI checkbox, not this method, is the real cost control for a large
                                   scan. self._raw_point_count is a live running total for GUI progress
                                   text without forcing a build.
                                   ensure_cloud_built() branches for non-RTAB-Map sessions with any TSDF-
                                   integrated frames (self._tsdf_integrated_count > 0): returns
                                   self._tsdf_volume.extract_point_cloud() (cached, re-extracted only when
                                   new frames were integrated since the last call) instead of the above
                                   raw-batch merge — see 3D Scanning Pipeline's "TSDF fusion for raw
                                   geometry quality" note for why (denoises Live Points/PLY export, RTAB-
                                   Map sessions deliberately excluded and keep using the raw-merge path
                                   unchanged). self._tsdf_volume (ScalableTSDFVolume) is integrated
                                   alongside — not instead of — the existing per-frame back-projection in
                                   Step 3, for every depth-consistency-gate-trusted frame, in the non-
                                   RTAB-Map branch only.
                                   process_frames_batch()'s `occupancy_voxel_size` param (defaults to
                                   DEFAULT_VOXEL_SIZE, threaded from scan_gui.py's voxel_size_input):
                                   each batch/node's own new points get an ADDITIONAL voxel-downsample
                                   at this size — on top of the existing fine VOXEL_SIZE cleanup that
                                   still feeds self._raw_cloud_batches/Live Points unchanged — specifically
                                   before feeding self.occupancy_map.update(), so occupancy cells
                                   correspond to what the GUI's Voxelization view shows (see 3D Scanning
                                   Pipeline's "Progressive, Bayesian, SLAM-style Occupancy Map" note).
                                   Still strictly per-call — never a re-feed of accumulated history.
                                   Every voxelize_cloud() call here also goes through _merge_voxels()
                                   right after feeding occupancy_map.update() — the single accumulator
                                   that IS self.last_voxel_centers/colors/voxel_size, so there is exactly
                                   one voxelization, not a second independent one for display (see 3D
                                   Scanning Pipeline's note on this).
                                   RTAB-Map mode does NOT back-project locally at all (Step 3 skips
                                   _back_project_frame entirely when use_rtabmap_pose) — Step 3b instead
                                   calls _rtabmap_pull_new_nodes() (normal case: rtabmap_client.get_cloud
                                   (since_node_id=self._rtabmap_last_pulled_node_id) — only nodes
                                   reconstructed since last pull) or _rtabmap_full_resync() (whenever a
                                   TRACK reply this batch set self.last_rtabmap_loop_closure — discards
                                   self._raw_cloud_batches + resets self.occupancy_map, re-pulls
                                   EVERYTHING via since_node_id=0 so previously-pulled nodes get their
                                   now loop-closure-corrected poses); both feed the Occupancy Map one
                                   node at a time, run through voxelize_cloud() at occupancy_voxel_size
                                   (occupancy_map.update(node's own pose, that node's voxel centers),
                                   then _merge_voxels() — fully inferred from the voxelization, same as
                                   IMU + VO) — never one call for the whole history, see
                                   occupancy_map.py's sunburst-artifact note).
                                   configure_occupancy_map(**kwargs) applies
                                   occupancy_map.py tuning overrides to the current OccupancyMap in place
                                   and remembers them in self._occupancy_params so reset_cloud()'s fresh
                                   OccupancyMap re-applies them instead of reverting to class defaults;
                                   ScanSessionManager.configure_occupancy_defaults() does the same across
                                   every existing session and seeds new ones — see scan_gui.py's
                                   "Occupancy Map Settings" accordion
                                   Also owns: self.novelty_gate (OrbNoveltyGate) + self._min_sharpness/
                                   _min_new_fraction/_min_new_count/_min_rotation_deg (configure_
                                   novelty_gate()/ScanSessionManager.configure_novelty_gate_defaults(),
                                   same live-tunable pattern as occupancy above); self._tag_pending
                                   (List[_PendingTagFrame], short-lived batching buffer, cleared by
                                   reset_cloud()); resolve_landmark()/_finalize_raw_landmarks() (name
                                   search + final cluster over self._raw_landmarks, populated live by
                                   every flushed tag+detect batch) — see "Semantic mapper adapted to
                                   frame_extractor's Gemini -> GroundingDINO-tiny pipeline" above for the
                                   full design.
  orb_novelty_gate.py             OrbNoveltyGate — ORB match + Essential-Matrix-RANSAC frame novelty
                                   gate + _sharpness_score() blur-reject, duplicated (not imported) from
                                   frame_extractor/extractor.py — see "Novelty+blur frame gating and
                                   deferred landmark resolution" above. evaluate_with_keypoints() skips
                                   ORB detection (reuses FeatureTracker's already-computed keypoints);
                                   decide_accept() mirrors extract_new_frames()'s tested accept/reject
                                   logic exactly.
  stream_session.py              StreamingScanSession — push-based incremental wrapper around ScanSession,
                                   the ONLY way scan_gui.py drives a scan now (no batch path exists anymore):
                                   push_frame()/push_imu()/start_zone()/end_zone()/finish(). Buffers frames
                                   into mini_batch-sized chunks and calls process_frames_batch() once a
                                   chunk is ready. mini_batch defaults to 1 (was 4, briefly 3 mid-debugging
                                   — see "DA3 model default + per-frame processing" above for the full
                                   history); scan_gui.py's own call sites always pass an explicit mini_batch
                                   from their GUI slider, so only mapping_servicer.py's live path (which
                                   relies on the constructor default) picked up any of these changes.
                                   pure_walking constructor param threaded straight through to every
                                   process_frames_batch() call — see scan_session.py's entry and
                                   "Grid-planned walking route".
                                   start_zone()/end_zone() are the live replacement for a
                                   Segment Table row (zone AABB computed from positions seen between the
                                   two calls). Only 2 pose sources supported — resolve_pose_flags(pose_src,
                                   rtabmap_available) returns (use_imu, use_rtabmap); "Auto"/"VO only"/
                                   "DA3 poses" were removed (see 3D Scanning Pipeline). RTAB-Map needs no
                                   retained raw IMU samples, unlike the ORB-SLAM3 mode this replaced.
                                   occupancy_voxel_size constructor param (threaded from scan_gui.py's
                                   voxel_size_input) passed straight through to every
                                   process_frames_batch() call in _process_chunk() — see scan_session.py's
                                   entry above.
  stream_simulator.py            build_event_timeline() — reads camera.csv/imu.csv, subsamples frames to
                                   ~fps_val (always appending the dataset's true last frame even if the
                                   modulo sampling missed it), returns one timestamp-sorted event list;
                                   shared by the two replay drivers below so both see the identical
                                   sampled frame set. native_fps=True (set from the caller's
                                   stream._use_rtabmap) forces interval=1 — RTAB-Map's own frame-to-frame
                                   odometry needs every recorded frame for reliable tracking, unlike
                                   IMU + VO which tolerates fps subsampling fine (see 3D
                                   Scanning Pipeline's "Streaming interface" note).
                                   replay_dataset() — auto-driven generator, replays an existing
                                   uploads/<scan_id>/dataset/ folder frame-by-frame/IMU-sample-by-sample
                                   through StreamingScanSession's public API (scan_gui.py's "Simulated Live
                                   Stream" button). ManualDatasetReplayer — single-step counterpart
                                   (scan_gui.py's "Start / Reset Manual Stream" + "Feed Next Frame"
                                   buttons): has_more()/peek_next_frame_preview()/step() let the GUI show a
                                   preview of the next not-yet-fed frame and advance exactly one frame per
                                   call, applying any IMU samples/zone boundaries preceding it first. Both
                                   drivers simulate a live source before a real one (Android gRPC) exists
                                   (see 3D Scanning Pipeline) and fire start_zone()/end_zone() at Segment
                                   Table boundaries crossed during replay — standing in for an operator's
                                   real-time button presses, since the dataset format has no live
                                   equivalent for that yet. replay_dataset() alone supports real-time
                                   pacing vs. max-speed replay (realtime_pacing_cb); ManualDatasetReplayer
                                   has no pacing concept — the GUI click IS the pace.
  feature_tracker.py             ORB + PnP pose estimator; returns (world_pose, rel_pose) tuple.
                                   Every PnP-solved frame also gets an independent depth-consistency
                                   check (_triangulate_depth_agreement) — re-triangulates the PnP-inlier
                                   2D correspondences via two-view geometry (no dense depth involved),
                                   compares against DA3's dense depth at the CURRENT frame's own pixels,
                                   and sets last_depth_trustworthy/last_depth_agree_err/
                                   last_depth_agree_n — see 3D Scanning Pipeline's "Depth-consistency
                                   gate" note for why (a real warped-wall bug) and how (fraction-bad,
                                   not median; curr's own depth, not prev's)
  da3_wrapper.py                 BaseDepthEstimator + DA3Estimator (torch) / DA3OnnxEstimator — DA3 depth,
                                   used as scan_session's default estimator (dense depth for every pose source).
                                   DA3Estimator.estimate_batch() logs a one-shot [DA3] depth stats line (min/
                                   median/max/is_metric/scale_factor) on first call — see "DA3 model default +
                                   per-frame processing" above (round 4) for why: the vendored package never
                                   applies Prediction.scale_factor to Prediction.depth, and Prediction.is_metric
                                   is unconditionally 0 due to a getattr-on-dict bug in the vendored package
                                   itself, so this codebase had no real confirmation depth is metric-scaled.
  vio/                           Visual-Inertial Odometry module (GTSAM iSAM2)
    __init__.py                  Exports IMUPreintegrator, VIOEstimator
    defaults.py                  Sensor noise model constants (accel/gyro sigmas)
    imu_preintegrator.py         GTSAM PreintegratedCombinedMeasurements; gravity/bias init
    vio_estimator.py             GTSAM ISAM2 + CombinedImuFactor + BetweenFactor<Pose3>
  zone_labeler.py                Zone AABB management (Zone dataclass has landmarks: List[Landmark])
  map_exporter.py                PLY + JSON export; map_labels.json includes zone_type, landmarks[],
                                   occupancy_grid per area (now with a "class" sub-grid alongside the
                                   original "data" float grid — 0=unknown,1=ground,2=low/step-over,3=normal
                                   obstacle, see occupancy_map.py), plus a top-level whole-map occupancy_grid
                                   (extract_full_grid()) for server/tools/grid_path_planner.py's A*
  occupancy_map.py                OccupancyMap — 2D X-Z traversability grid, builds up progressively
                                   (once per batch, not deferred) via a Bayesian log-odds belief per
                                   cell (_CellState.logodds/height_ewma) — see 3D Scanning Pipeline's
                                   "Progressive, Bayesian, SLAM-style Occupancy Map" note.
                                   _dirty_cells/extract_dirty_delta()/bounds()/clear_dirty() (new) —
                                   sparse-export counterpart to extract_full_grid(), see
                                   "Client-Orchestrated Live Session"'s "Occupancy grid delta sync" note
                                   for the full design (mapping_servicer.py decides full vs. delta per
                                   update). update()
                                   takes a confidence: float = 1.0 param (scan_session.py's per-batch
                                   depth-consistency agreement), multiplied into every log-odds delta
                                   in _register_obstacle_hit/_register_ground_hit/_register_miss —
                                   Bayesian evidence weighting, not a hard cap, see 3D Scanning
                                   Pipeline's "Confidence-weighted occupancy updates" note.
                                   update() classifies every POINT individually by height above ground (not a
                                   per-cell-per-batch percentile), buckets by (X,Z) cell — obstacle
                                   evidence always wins over a ground point in the same coarse cell
                                   that batch. _register_obstacle_hit()/_register_ground_hit()/
                                   _register_miss() nudge a cell's belief up/down per observation
                                   (self-correcting — a wrong reading can be revised by later
                                   disagreeing evidence, unlike a plain running-average or
                                   raw-sample-list scheme); _classify_state() derives (CLASS_UNKNOWN/
                                   GROUND/LOW_STEP_OVER/OBSTACLE, STEP_OVER_MAX_H=0.40m default) from
                                   a cell's current belief, used by both render_plotly() and
                                   extract_subgrid()/extract_full_grid(). _cast_free_ray()
                                   Bresenham-walks from each batch's camera position ONLY toward
                                   ground-classified cells (never toward obstacle cells) and STOPS at
                                   the first cell with any net hit evidence (logodds > 0) — real
                                   occlusion, matching how RTAB-Map's own OccupancyGrid/ROS
                                   costmap_2d build a 2D grid from a 3D cloud, replacing an earlier
                                   linear-height-interpolation gate that could erode a real, closer
                                   obstacle via a ray aimed at farther floor (see 3D Scanning
                                   Pipeline's rewrite note — this was a real, confirmed bug: an
                                   entire bed eroding to free/unknown). render_plotly() renders a
                                   continuous height-above-ground heatmap ("Turbo" colorscale,
                                   0.0m=ground to OBSTACLE_MAX_H, unknown cells left as NaN/blank) —
                                   replaced an earlier discrete 4-color grayscale that only showed
                                   occupied-or-not, not obstacle height; still built from
                                   _classify_state()'s belief, de-normalized back to real metres for
                                   display. Every
                                   constant here (height tiers, the 4 log-odds weights/thresholds,
                                   HEIGHT_EWMA_ALPHA) is a constructor/set_params() override, not
                                   hardcoded — see scan_gui.py's "Occupancy Map Settings" accordion.
                                   enable_ray_casting/enable_bayesian (both default True, same
                                   accordion) — independent toggles: ray casting off skips
                                   _cast_free_ray entirely (cells only ever get hit, never revised back
                                   to free); Bayesian off makes hits set logodds straight to
                                   LOGODDS_MAX/LOGODDS_MIN (single hit = permanent) and _register_miss
                                   a no-op. Every update() call ends with one "[occupancy]" console
                                   log line (point counts by class, rays cast vs. blocked-by-obstacle,
                                   ground_y, cell count) for debugging a future recurrence.
                                   _build_grid_dict() (shared by extract_subgrid/extract_full_grid) also
                                   computes a "clearance" field (metres to nearest CLASS_OBSTACLE cell,
                                   via scipy.ndimage.distance_transform_edt over that window's own class
                                   grid — falls back to a 9.0m sentinel when a window has zero obstacle
                                   cells, since the EDT function returns nonsense corner-anchored values
                                   on an all-False mask rather than raising) — read by server/tools/
                                   grid_path_planner.py for clearance-aware A* costs, see 3D Scanning
                                   Pipeline's "Continuous obstacle clearance" note.
                                   render_confidence_plotly() — continuous per-cell CONFIDENCE heatmap
                                   (|logodds| normalized to [0,1], independent of free/obstacle
                                   classification), side-by-side with render_plotly() in scan_gui.py.
                                   Both share _grid_bbox_and_ticks()/_empty_figure()/_CLASS_NAME and
                                   both accept optional route/route_confirmed params drawn via
                                   _overlay_route() (green=confirmed, dashed orange=speculative — see
                                   live_path_planner.py). _update_count, bumped once per real update()
                                   call, is the "did the grid change" signal scan_gui.py's Live
                                   Navigation Preview uses to gate route recomputation — see 3D
                                   Scanning Pipeline's "Live Navigation Preview" note.
  live_path_planner.py           LiveGridPathPlanner — adapted copy of server/tools/grid_path_planner.py's
                                   GridPathPlanner (same CLASS_*/cost model/clearance-aware A*), built
                                   fresh each time from OccupancyMap.extract_full_grid()'s in-memory
                                   dict (no file I/O) instead of a static exported map.json, so
                                   scan_gui.py can preview a route while a scan is still in progress.
                                   find_path() returns (waypoints, confirmed, reached_exactly):
                                   confirmed=False if the route crossed a CLASS_UNKNOWN cell;
                                   reached_exactly=False if the exact destination wasn't reachable and
                                   this is instead a closest-approach fallback to the nearest reachable
                                   cell (found during the same A* expansion, no second search) — only
                                   None when nothing useful can be offered at all (start itself
                                   blocked, or fully sealed off). min_path_clearance_m (constructor
                                   param, GUI-exposed) adds a bounded penalty for cells narrower than a
                                   desired width, on top of the existing clearance shaping — see 3D
                                   Scanning Pipeline's "Closest-approach navigation + minimum path
                                   width" note. Duplicating rather than importing GridPathPlanner is
                                   intentional (server/scan_server process boundary, see "Live
                                   Navigation Preview" note).
                                   Now also imported by server/services/mapping_servicer.py — no longer
                                   scan_gui.py-only — for the live WALKING/GUIDING path (see
                                   "Server-planned walking path" above).
                                   CLASS_LOW_STEP_OVER is now hard-blocked in _passable()/_cost(), same
                                   as CLASS_OBSTACLE (was a 3.0x cost multiplier, still passable) — see
                                   "Path stability" above. find_path() gained an optional
                                   previous_path_xz param (_prev_path_bias(), new PREV_PATH_PENALTY_SCALE/
                                   PREV_PATH_PROXIMITY_DECAY_RATE constants) — an additional A* edge-cost
                                   term biasing toward a previously-served route, decayed by distance from
                                   the CURRENT pose so it matters most near-term. New segment_blocked()/
                                   path_blocked_ahead() methods — the reuse-vs-replan gate
                                   mapping_servicer.py's UpdateMapping now checks every update before
                                   calling either planner method at all (see "Path stability" above).
                                   _simplify() gained SIMPLIFY_COST_SLACK_FRAC (0.08) — drops a waypoint
                                   whose straight-line shortcut costs up to 8% more than the original
                                   route through that stretch, not just no-more, trading negligible
                                   efficiency for fewer joints for a blind user to follow — shared by
                                   both find_path() and find_natural_path() below.
                                   find_path()'s old confirmed_only param (+ the _truncate_to_confirmed()
                                   helper backing it) was REMOVED outright — it existed only for the
                                   now-deleted find_farthest_open_path(); GUIDING's own find_path() call
                                   (toward a real destination) never used it, deliberately flooding into
                                   unexplored territory when that's the only way to reach the destination.
                                   find_natural_path() (new LiveGridPathPlanner method) — WALKING's
                                   target/route strategy, replacing find_farthest_open_path() entirely —
                                   see CLAUDE.md's "Natural path planner" note for the full design. A
                                   direction-augmented Dijkstra (state = cell + incoming direction, not
                                   just cell, via the new module-level _DIRECTIONS table/_wrap_angle()) —
                                   each edge's cost is an additive weighted sum (_natural_step_cost():
                                   W_HEADING=10 * quadratic heading_error, W_OBSTACLE=8 * clearance-based
                                   proximity (0 once safe_clearance_m is met — also what pulls the route
                                   toward corridor centers, no separate centering logic needed),
                                   W_TURN_COUNT=7 flat + W_TURN_ANGLE=5 * angle whenever direction
                                   changes, W_LENGTH=2 * class-tiered real distance). Expansion is pruned
                                   to a spatial cone (bearing FROM START, not a per-step turn limit)
                                   escalating 45deg -> 90deg -> 180deg (_search_natural(),
                                   cone_stages_deg param) whenever a narrower stage finds nothing OR only
                                   a sliver of forward progress (min_progress_frac, 0.3, of
                                   max_distance_m) — the "only escalate on zero progress" version was a
                                   real bug found via testing (a corridor with a metre of open floor
                                   before a full wall "succeeded" at 45deg and never looked for a real
                                   opening off to the side). Never expands into CLASS_UNKNOWN (same
                                   confirmed-territory principle the old confirmed_only used to enforce,
                                   now pruned directly during expansion instead of post-hoc truncated).
                                   Target selection: among every cell reached within max_distance_m, pick
                                   whichever has the greatest _directional_distance() progress along
                                   heading_rad — not farthest by raw distance, not cheapest by cost alone
                                   (still reused from the deleted find_farthest_open_path() era —
                                   the only piece of that function's own machinery kept). Verified via
                                   synthetic grids (open field -> straight; small obstacle -> morphs
                                   around, stays forward; wall with a far-off opening -> escalates cone
                                   and turns into it; a too-close wall -> drifts to the open side for
                                   comfortable clearance; boxed-in start -> None).
                                   _natural_step_cost()'s obstacle_proximity gained a SECOND, no-cutoff
                                   "soft" term (SOFT_CLEARANCE_DECAY_RATE=1.5) alongside the original
                                   hard-cutoff "steep" one — the steep-only version let two cells that
                                   both already cleared safe_clearance_m score identically regardless of
                                   how much MORE open one was, which is what let a route hug one obstacle
                                   despite far more open space elsewhere; see the follow-up round in
                                   "Natural path planner" above.
                                   nearest_point_on_path()/advance_along_path()/beacon_target_point() (new)
                                   — direct Python port of PathPursuit.kt, server-side only for
                                   server_gui.py's dashboard marker — see "Server-side HRTF beacon
                                   placement mirror". mapping_servicer.py now prepends pose_xz to the
                                   path before calling beacon_target_point() — see the "HRTF beacon
                                   placement bug" follow-up note in "Natural path planner" above (a real
                                   navigation bug, not just this dashboard mirror — PathPursuit.kt got the
                                   equivalent client-side fix).
  semantic_mapper.py             SemanticMapper — wraps a FrameTagger (frame_extractor/tagging.py's
                                   Gemini -> GroundingDINO-tiny pipeline) and runs it IMMEDIATELY,
                                   batched, per accepted frame — see "Semantic mapper adapted to
                                   frame_extractor's Gemini -> GroundingDINO-tiny pipeline" above
                                   (supersedes the old Gemma-VLM-tag-then-defer-GroundingDINO design).
                                   tag_and_backproject_batch(frames_bgr, depth_maps, world_poses, Ks,
                                   frame_idxs) -> List[List[Landmark]]: one batched FrameTagger.
                                   tag_and_detect_batch() call across up to IMAGES_PER_PROMPT=5 frames,
                                   immediately followed by per-frame backprojection (depth-median-sample
                                   + 4-corner unprojection) of every detected box — stateless,
                                   ScanSession owns the buffering (_PendingTagFrame/_tag_pending).
                                   Landmark dataclass; cluster_landmarks() merges same-label
                                   detections that are close OR overlapping (MERGE_DISTANCE_M /
                                   OVERLAP_MERGE_RATIO), position = highest-confidence member's own
                                   (x, z), no averaging — see "Distance-based landmark merging
                                   (confidence picks the winner)" above.
  gemma_vlm.py                    GemmaVLMClient — Gemma 4 31B via Gemini API (GEMINI_API_KEY),
                                   multi-image query(prompt, images=[...]). No longer used by any live
                                   path (semantic_mapper.py switched to RAM++ + GroundingDINO-tiny, see
                                   above) — left in place, unreferenced, not deleted.
  qwen_vlm.py                     Qwen3VLClient — local vLLM-backed Qwen3-VL, single-image only.
                                   Disabled (not imported by scan_server.py) — its 2B model was prone
                                   to greedy-decoding repetition loops (temperature=0, no repetition
                                   penalty) in grounding_dino_prompt output; kept for reference
  rtabmap_client.py               RtabmapPoseClient — pyzmq REQ/REP bridge to rtabmap_docker/.
                                   track_batch() sends one TRACK request per frame (RGB + DA3-estimated
                                   depth + per-frame intrinsics, no IMU), returns one TrackedFrame
                                   (pose + loop_closure bool + node_id int, -1 if this frame didn't
                                   become a node + inlier_fraction float) per frame — loop_closure
                                   signals that THIS frame's processing closed a loop, so previously-
                                   pulled get_cloud() nodes' poses may have just shifted; node_id lets
                                   scan_session.py correlate its own depth-consistency check with the
                                   specific RTAB-Map node this frame produced (see 3D Scanning
                                   Pipeline's "Depth-consistency gate, extended to RTAB-Map pose
                                   mode"); inlier_fraction (odomInfo.reg.inliers/matches, RTAB-Map's
                                   own frame-to-map registration quality) is the RTAB-Map-pose-mode
                                   novelty signal scan_session.py's gate uses instead of a redundant
                                   Python ORB pass — see "Novelty+blur frame gating and deferred
                                   landmark resolution" above. Reply parsing degrades gracefully
                                   (node_id=-1, inlier_fraction=1.0 i.e. "not novel") against an older
                                   server build without these fields, via the same layered
                                   backward-compat approach used when node_id itself was added.
                                   get_cloud(since_node_id, voxel_size, max_depth) pulls RTAB-Map's OWN reconstructed
                                   surface — each node's stored SensorData re-projected server-side
                                   (util3d::cloudRGBFromSensorData), voxelized, transformed by RTAB-
                                   Map's CURRENT graph-corrected pose — returning ReconstructedNode
                                   (node_id, pose, points, colors) per node; used by scan_session.py
                                   INSTEAD OF its own DA3-depth back-projection, only for RTAB-Map pose
                                   mode. Optional (RTABMAP_ADDR, e.g. tcp://localhost:5556). Replaces
                                   orbslam3_client.py (removed) — RTAB-Map needs no camera-IMU
                                   calibration at all
  rtabmap_docker/                 RTAB-Map RGB-D pose + surface-reconstruction service — own Dockerfile,
                                   no ROS, not part of server/.venv. Replaces orbslam3_docker/ (removed)
                                   — no calibration config/tools subdirectory at all, unlike the old
                                   ORB-SLAM3 service
                                   Dockerfile              FROM introlab3it/rtabmap:noble (official prebuilt image;
                                                            confirmed to ship dev headers + RTABMapConfig.cmake)
                                                            + our ZMQ server; cppzmq-dev apt package for zmq.hpp
                                                            (simpler than vendoring via wget)
                                   src/rtabmap_server.cc  standalone executable — rtabmap::Odometry (pose) +
                                                           rtabmap::Rtabmap (loop closure/graph optimization,
                                                           via getMapCorrection()) directly, custom binary
                                                           protocol over a single ZeroMQ REP socket
                                                           (TRACK/RESET/PING/GET_CLOUD) — RESET reinits both
                                                           objects so a fresh scan never loop-closes against a
                                                           prior one. GET_CLOUD reconstructs each requested
                                                           node's cloud via Memory::getNodeData() + uncompressData()
                                                           + util3d::cloudRGBFromSensorData() (with its
                                                           validIndices output — util3d::voxelize()'s no-indices
                                                           overload silently returns an EMPTY cloud for an
                                                           organized single-camera cloud, a real bug hit and
                                                           fixed during development) + util3d::voxelize() +
                                                           util3d::transformPointCloud() by getLocalOptimizedPoses().
                                                           TRACK's reply carries a loop_closure flag, only set
                                                           when getLoopClosureId()>0 AND the resulting
                                                           getMapCorrection() moved by more than
                                                           SIGNIFICANT_CORRECTION_TRANSLATION_M/_ANGLE_RAD since
                                                           the last one reported — RGBD/ProximityAngle also
                                                           tightened 45°→20° in make_parameters() — both address
                                                           RGBD/ProximityBySpace accepting a loop closure against
                                                           any spatially-nearby node almost every frame during
                                                           slow, close-range scanning (see 3D Scanning Pipeline).
                                                           TRACK's reply also carries new_node_id (int32, -1 if
                                                           this frame didn't become a node) — computed as
                                                           slam.getLastLocationId()==data.id() ? data.id() : -1 —
                                                           letting scan_session.py correlate its own per-frame
                                                           depth-consistency check with the specific RTAB-Map
                                                           node a bad frame produced, so that node can be vetoed
                                                           when later pulled via GET_CLOUD (see 3D Scanning
                                                           Pipeline's "Depth-consistency gate, extended to
                                                           RTAB-Map pose mode"). TRACK's reply also carries a
                                                           4th field, inlier_fraction (float) —
                                                           odomInfo.reg.inliers / max(odomInfo.reg.matches, 1),
                                                           read directly off the RegistrationInfo struct
                                                           odom->process() already populates (no new SLAM
                                                           work) — the RTAB-Map-pose-mode novelty signal
                                                           scan_session.py's gate uses instead of a redundant
                                                           Python-side ORB pass (deliberately NOT new_node_id,
                                                           which reflects keyframe-spacing policy, not visual
                                                           overlap — see "Novelty+blur frame gating and
                                                           deferred landmark resolution" in the 3D Scanning
                                                           Pipeline section). Confirmed reg.inliers/reg.matches
                                                           are the right fields by reading the vendored header
                                                           in the built image
                                                           (/usr/local/include/rtabmap-0.23/rtabmap/core/
                                                           RegistrationInfo.h).
                                   entrypoint.sh            execs rtabmap_server with just a bind address — no
                                                            required config file (intrinsics sent per-frame)
                                   README.md                build/run instructions + corelib-from-source fallback +
                                                            full wire protocol (TRACK/RESET/PING/GET_CLOUD)

server/
  vio → ../scan_server/vio       Symlink so server/ can import the same VIO module

client/
  android/                         The only client — Android Jetpack Compose (Kotlin); Gradle root: client/android/
    audio/
      ContinuousVadRecorder.kt     Replaces PushToTalkRecorder.kt (deleted) — always-on mic capture gated
                                     by a client-side amplitude VAD (onSpeechStart/onSpeechEnd), not a
                                     tap-and-hold gesture. See CLAUDE.md's "Continuous VAD-gated listening"
                                     section.
      StreamingAudioPlayer.kt      Incremental raw PCM playback (24 kHz) for VoiceChatStream response
      TtsPlayer.kt                 WAV playback (24 kHz) for unary VoiceChat / StreamFrame audio
    camera/
      CameraManager.kt             CameraX ImageAnalysis — live JPEG stream (frameFlow) AND, while recording,
                                     periodic frame dump to images/*.jpg + camera.csv (timestamp_ns,filename)
                                     startRecording(outputDir, fps) / stopRecording(): Int (frame count)
                                     frame timestamp = imageInfo.timestamp (boot-time ns, same clock as ImuSensor).
                                     Every incoming frame gets scored via computeSharpness() (variance of the
                                     Laplacian on a 320px-downscaled grayscale copy, same metric orb_novelty_
                                     gate.py's _sharpness_score() uses server-side) — moved here from the
                                     server, see "Client-side frame selection" note for the full rationale.
                                     TWO independent selection policies, chosen per-frame via mappingMode
                                     (String — "", "guiding", "walking", or "scanning" — set every
                                     processed frame by MainViewModel.kt's frame collector, forwarded
                                     verbatim from sessionState.mode; walking rejoined this set once its
                                     beacon moved back to a grid-planned route, see "Grid-planned walking
                                     route" — it had been carved out into policy 2 below for a while, in
                                     between, see "Local reactive HRTF obstacle-dodge"): (1) mapping modes
                                     (guiding/walking/scanning) — NO blur/clarity filtering (removed,
                                     confirmed with the user — see "Blur filtering removed for
                                     scan/walking"): whichever frame arrives once frameIntervalMs (ms,
                                     guiding), walkingIntervalMs (ms, walking), or scanIntervalMs (ms,
                                     scanning) has elapsed since the last send is forwarded directly — no
                                     window, no candidate comparison. All three stay ms internally;
                                     SettingsScreen exposes frameIntervalMs/scanIntervalMs as FPS number
                                     inputs, not sliders ("Mapping FPS" 0.2..10 default 1.0, "Scan FPS"
                                     2..20 default 10.0), converting fps->ms only at doConnect() time;
                                     walkingIntervalMs instead reuses avoidanceIntervalMs directly (set
                                     from MainViewModel.connect(), no separate UI control) since it needs
                                     to stay fast/responsive like the local-avoidance tick, not slow like
                                     guiding's default. activeMappingSubmode tracks which one is in effect
                                     and resets the send-gate on any submode change (a stale timestamp
                                     from a different mode/interval shouldn't suppress the new mode's
                                     first send); (2) recentBufferMs (SettingsScreen slider, 0..1000ms,
                                     50ms steps, default 100) — every other mode (tracking/reading/Q&A/
                                     idle) — a small rolling buffer of the last recentBufferMs of frames,
                                     no window/gap logic; clearestRecentFrame() pulls the sharpest
                                     buffered frame on demand (used by ToolDispatcher.kt's latestFrame()
                                     closure — OCR/run_detection/tracking-init calls, AND walking's own
                                     step-down hazard-check tick, see "Hazard warnings" — all want "the
                                     current frame" right now), while handleRecentEmit() emits that same
                                     clearest-so-far frame into frameFlow at most once per recentBufferMs
                                     for continuous per-frame consumers (hand tracking, local ORB
                                     tracking, UI overlay). Note: recentBuffer is populated unconditionally
                                     on every processed frame regardless of mappingMode, so
                                     clearestRecentFrame() keeps working for walking's hazard tick even
                                     while walking is also in the mapping-mode bucket above — only the
                                     continuous frameFlow emission (policy 2's own emit, not the buffer
                                     itself) is skipped during mapping mode.
                                     clearestRecentFrameWithSharpness() (NEW) — same pull, also returns
                                     the sharpness score, for ToolDispatcher's reading-mode blur
                                     skip/retry (acquireSharpFrame()) — see "Reading-mode OCR" above.
    sensors/
      ImuSensor.kt                 SensorManager wrapper; emits ImuReading Flow at SENSOR_DELAY_FASTEST
      ImuRecorder.kt               Writes imu.csv (header: timestamp_ns,ax,ay,az,gx,gy,gz) alongside images/
    device/
      DeviceToolHandler.kt         Interface for executing device-native tool calls (phone/alarm/calendar)
      AndroidDeviceToolHandler.kt  Implementation: Intent.ACTION_CALL, AlarmClock, CalendarContract, plus
                                     answerPhoneCall() (TelecomManager.acceptRingingCall(), see "Foreground-
                                     service migration + calls/SMS/YouTube" for the accepted dialer-role
                                     risk), sendSms()/checkUnreadSms() (SmsManager/Telephony.Sms.Inbox)
    receivers/                     NEW package — the app's first BroadcastReceivers
      CallBackgroundReceiver.kt     PHONE_STATE -> resolves caller via ContactsContract.PhoneLookup ->
                                     starts LiveAssistantService with ACTION_INCOMING_CALL
      SmsBackgroundReceiver.kt      SMS_RECEIVED -> starts LiveAssistantService with ACTION_SMS_RECEIVED
    live/                          Client-orchestrated Gemini Live session — see CLAUDE.md's
                                     "Client-Orchestrated Live Session" section for the full picture
      LiveAssistantService.kt       NEW — foreground + bound LifecycleService now owning the entire session
                                     graph (gRPC/camera/mic/tool dispatch/uiState) that used to live in
                                     MainViewModel's viewModelScope — see "Foreground-service migration +
                                     calls/SMS/YouTube" for the full design. MainViewModel is now a thin
                                     bound-client facade over this.
      GeminiLiveClient.kt           Raw WebSocket client for Gemini Live's BidiGenerateContent protocol
      ToolDeclarations.kt           SYSTEM_PROMPT + FunctionDeclaration JSON, ported from tool_declarations.py
      YouTubeSearchClient.kt        NEW — direct OkHttp calls to YouTube Data API v3 (search_youtube/
                                     get_video_info); playback itself is the official IFrame player
                                     (play_youtube_video), not a stream URL from this client
      NewsClient.kt                 NEW — direct, keyless Google News RSS calls (get_top_news/search_news),
                                     hardcoded to Vietnam/Vietnamese — see "News / Radio tools" above
      RadioClient.kt                NEW — direct, keyless Radio-Browser API station search (play_radio),
                                     hardcoded to country=Vietnam; resolves a stream URL only, playback
                                     reuses play_video/PlaybackService — see "News / Radio tools" above
      LiveSessionState.kt           Port of server/live_session.py's LiveSessionState. Path PLANNING moved
                                     server-side (see "Server-planned walking path + client-side latency
                                     bridging") — plannedPath/pathConfirmed hold the server-planned route
                                     directly (no more client-side grid/A* state: navWaypoints/
                                     navWaypointIdx/lastMappingGrid/mutableGrid/smoothedBeaconAzimuthDeg
                                     are all gone). guidingGoalXz — GUIDING's FindLandmark-resolved
                                     destination, sent back to the server as the planning goal.
                                     lastMappingPose — the last AUTHORITATIVE server pose (NOT the
                                     current best-estimate, which is derived on demand via
                                     HrtfBeacon.extrapolate()). poseSendHistory (PoseSendSnapshot) — the
                                     send-time accumulator-snapshot buffer latency compensation reads
                                     from, see that section for the full design.
      ToolDispatcher.kt             Port of _dispatch_tool + live_tools/*.py — remote/3rd-party/local/device.
                                     stopActiveModes() — called first by every mode-entry tool, enforces
                                     state.mode exclusivity (see "Mode exclusivity" note above); also stops/
                                     resets angleTracker/pdrStepEstimator and pixieController now, plus the
                                     tracking-mode trackingAxis state machine (see "Pixie + Angle modules").
                                     reportMode() — fire-and-forget StatusService.ReportMode call on every
                                     mode transition. startMappingStream()/feedMappingFrame() are shared by
                                     GUIDING, WALKING, and SCANNING alike. buildMappingChunk() attaches
                                     has_goal/goal_x/goal_z for GUIDING once resolveGuidingGoalIfNeeded()
                                     resolves a destination, and snapshots the local estimators'
                                     accumulators into state.poseSendHistory keyed by the chunk's own
                                     timestamp. The mapping-stream collector no longer runs any path search
                                     at all (recomputeRoute() is gone) — it reconciles the server's
                                     (already slightly stale) pose against the local rotation/PDR
                                     accumulators via HrtfBeacon.extrapolate() (also folding each accepted
                                     fix into angleTracker.setAuthoritativeHeadingDeg() via the new
                                     HrtfBeacon.worldHeadingDeg()), sets state.plannedPath straight from
                                     MappingUpdate.planned_path, fires playDeadEndAlert() on an empty
                                     WALKING path, and checkGuidingArrival() for GUIDING. Steering is
                                     unified for both modes now: runUnifiedAvoidanceTick() (replaces
                                     runAvoidanceTick()/runWalkingAvoidanceTick()) runs the step-down/
                                     drop-off/obstacle hazard check (checkAndWarnHazard() — now also
                                     checks clearance_m, grouped with dropoff_m into one hedged system-note
                                     prompt per tick, see "Hazard warnings" above) then calls
                                     steerBeaconAlongPath() (replaces recomputeRoute()/steerWalkingBeacon()/
                                     checkWaypointProgress()), which projects the extrapolated pose onto
                                     state.plannedPath via PathPursuit and steers Pixie (LEFT/RIGHT +
                                     gainForDeviation-driven volume, see "Pixie + Angle modules") toward a
                                     look-ahead point on it — see "Server-planned walking path" for the
                                     path-pursuit design and what got deleted (LocalPathPlanner.kt,
                                     MutableOccupancyGrid.kt, TraversabilityScorer.kt, all outright), and
                                     "Pixie + Angle modules" for the current beacon-output mechanism.
                                     updateTrackingPixie() (replaces updateTrackingBeacon()) drives
                                     tracking mode's own 2-phase HORIZONTAL/VERTICAL Pixie cue — see that
                                     same section.
      OcrClient.kt                  Direct 3rd-party HTTP client to OCR.space — analyze() chains
                                     filterLinesByRotation() -> filterLinesByNoise() ->
                                     filterLinesByBlur() (TextBlockFilters.kt) before joining survivors
                                     into reading-order text — see "Reading-mode OCR" above.
      TextBlockFilters.kt            OcrLine/OcrWord + OcrBlockFilters: blockSimilarity()/ReadingBlock/
                                     integrateRawBlock() (raw-first fuzzy dedup + boundary stitching +
                                     mid-text realign merge, a Kotlin port of Python difflib's
                                     Ratcliff/Obershelp matching-blocks algorithm at both word and
                                     character granularity) and the rotation/noise/blur line filters
                                     OcrClient.kt calls — see "Reading-mode OCR correction + fuzzy
                                     stitching" above for the full design + thresholds. integrateBlock()
                                     (the old MutableList<String>-based dedup) is deleted outright — no
                                     remaining caller.
      GeminiCorrectionClient.kt      NEW — OCR-error correction (+ translation for non-English text) via
                                     a plain REST call to Gemini's generateContent endpoint
                                     (gemini-3.1-flash-lite), reusing the same geminiApiKey already used
                                     for Gemini Live. Driven by ToolDispatcher's correction queue (one
                                     background coroutine, strictly sequential) — see "Reading-mode OCR
                                     correction + fuzzy stitching" above.
      DebugFrameStore.kt             NEW — annotateOcrFrame(): debug-only, draws each OCR line's
                                     kept/dropped box on a frame copy (see "Reading-mode OCR" above).
                                     Off by default; MainViewModel only wires ToolDispatcher's
                                     saveDebugFrame callback when SettingsScreen's toggle is on.
      LocalMemoryStore.kt           On-device JSON memory store + embedding index + cosine search
      AngleTracker.kt                NEW — replaces RotationTracker.kt (left in place, unreferenced) as the
                                     shared heading/rotation module — see CLAUDE.md's "Pixie + Angle
                                     modules" note. Same accumulatedRotation()/resetAccumulator()/reset()
                                     shapes (drop-in for ToolDispatcher's existing pose-extrapolation math)
                                     plus new luma-direct processLumaFrame() (1000 features/480px default,
                                     fed by CameraManager's new lumaFlow instead of a JPEG round trip),
                                     currentHeadingDeg()/driftedAngleDeg(), and
                                     setAuthoritativeHeadingDeg() (folds in each accepted MappingService
                                     pose fix via the new HrtfBeacon.worldHeadingDeg()). Still rotation-only
                                     (Essential-matrix decomposition) — same NOT-device-verified
                                     composition-convention caveat RotationTracker always carried.
      PdrStepEstimator.kt            NEW — Sensor.TYPE_STEP_DETECTOR + a fixed stride length, bridging the
                                     same gap for walked distance. distanceSinceReset()/resetAccumulator().
                                     Deliberately not ImuSensor.kt/ImuRecorder.kt (confirmed dead code).
      PathPursuit.kt                 NEW — stateless polyline geometry replacing LocalPathPlanner.kt's role:
                                     nearestPointOnPath() (project current position onto the server-planned
                                     path) + advanceAlongPath() (walk forward by a look-ahead distance). No
                                     search — the server already searched; this only follows.
      HrtfBeacon.kt                 directionTo() — egocentric azimuth/elevation from the current Pose to a
                                     world (x, z) target — now used for BOTH modes' unified path-pursuit
                                     steering target (see PathPursuit.kt/ToolDispatcher.steerBeaconAlongPath()).
                                     directionFromBox() — pixel-offset azimuth/elevation from a 2D ORB
                                     tracking box (tracking mode, no pose/depth available). extrapolate()
                                     (new) — combines an authoritative server Pose with RotationTracker's/
                                     PdrStepEstimator's accumulated deltas into a current best-estimate
                                     Pose; quatMultiply() (new, non-private) — Hamilton product, reused by
                                     ToolDispatcher's latency-compensation math. worldPointFrom()/
                                     worldYawRad() are both removed outright now (dead once server-side
                                     planning took over target selection — see "Server-planned walking
                                     path").
    ui/
      MainViewModel.kt             connect(host, port, fps, avoidanceIntervalMs, vadThreshold,
                                     startThreshold, geminiApiKey, ocrServerUrl, locationId) —
                                     doLiveSession() opens a GeminiLiveClient directly (no more
                                     server-relayed VoiceChatStream for this client) and drives
                                     ToolDispatcher.dispatch() for every Gemini function call;
                                     feedMappingFrame() called from the camera-frame collector while
                                     mode is guiding/walking/scanning; also sets
                                     cameraManager.walkingIntervalMs = avoidanceIntervalMs so walking's
                                     mapping-stream send rate stays fast/responsive, not guiding's
                                     slower default. feedRotationFrame() (new) called on EVERY collected
                                     frame regardless of mode (no-ops internally unless walking/guiding
                                     is active) — RotationTracker wants a continuous per-frame trickle,
                                     not the interval-gated mapping-mode push — see "Server-planned
                                     walking path". Owns the rotationTracker/pdrStepEstimator instances,
                                     passed into ToolDispatcher's constructor.

depth-anything-3/                DA3 model package (installed locally)
```

---

## Data Flows

All of these now run through the Android on-device tool-dispatch loop
(`live/ToolDispatcher.kt`) — see "Client-Orchestrated Live Session" for the
full file map. Summarized here as the request-level flow only.

### Object tracking
```
User names a target → start_tracking(target) tool
  → TrackingBackend.kt: TrackingService.DetectObject (one-shot, sends the
    current frame) → ORB reference extracted → continuous local ORB+
    homography tracking every frame, no network call
  → periodic renewal: DetectObject + GetEmbedding re-confirms the target
```

### Live navigation (guiding/walking)

Path PLANNING is entirely server-side now for both modes (see
"Server-planned walking path + client-side latency bridging" for the full
design) — the client only follows what the server returns, bridging the
gap between updates with its own rotation/PDR motion estimate.

```
User says "guide me to the couch" → start_guiding tool
  → ToolDispatcher resolves "the couch" via MappingService.FindLandmark
    (unchanged RPC) → stashes the resulting (x, z) as state.guidingGoalXz
  → opens a MappingService.UpdateMapping bidi stream, feeding camera
    frames (RTAB-Map pose) + the resolved goal (MappingChunk.has_goal/
    goal_x/goal_z) → server plans a route (LiveGridPathPlanner.find_path())
    against its own occupancy grid and streams back Pose + PlannedPath
    (+ OccupancyGrid, still sent but no longer read client-side) every
    update → ToolDispatcher.checkGuidingArrival() announces arrival once
    the extrapolated position nears the path's final point.

User says "start walking" → start_walking tool
  → SAME MappingService.UpdateMapping stream, no goal sent — server
    infers "keep walking forward" from its own RTAB-Map pose's heading and
    plans toward the farthest open direction within a turn budget
    (find_farthest_open_path()) → an empty PlannedPath (nothing walkable
    anywhere) triggers ToolDispatcher.playDeadEndAlert() (ToneGenerator
    tone).

BOTH modes, every avoidanceIntervalMs (ToolDispatcher.runUnifiedAvoidanceTick()):
  → step-down/drop-off hazard check (PerceptionService.AnalyzeFrame
    TRAVERSABILITY, unchanged — see "Hazard warnings" below)
  → HrtfBeacon.extrapolate(state.lastMappingPose, rotationTracker.
    accumulatedRotation(), pdrStepEstimator.distanceSinceReset()) — the
    client's own best-current-estimate position/orientation, bridging the
    gap since the last server update via frame-to-frame Essential-matrix
    rotation tracking + step-detector-based walked distance
  → PathPursuit projects that estimate onto state.plannedPath and steers
    the beacon toward a look-ahead point sliding forward along it —
    azimuth-only, fixed radius, muted only when there's no path at all.
    Purely ambient — no [SYSTEM] messages for the beacon itself, matching
    this project's established behavior (the hazard check above is the
    one thing that DOES speak, additively, when relevant).
```

### Reading a document aloud
```
User says "read this" → enter_reading_mode() then read_aloud(scope="new")
  → acquireSharpFrame() blur skip/retry, then OcrClient.kt POSTs the frame
    directly to OCR.space (no server proxy, no self-hosted OCR service) →
    analyze() drops rotation-mismatched/short-small-isolated-noise/locally-
    blurry lines (TextBlockFilters.kt) → surviving text folded into
    readingBlocks via raw-first fuzzy dedup + boundary stitching
    (OcrBlockFilters.integrateRawBlock() — see "Reading-mode OCR
    correction + fuzzy stitching" above) → "new"/"stitched" spoken
    immediately using the block's RAW text; "new"/"stitched"/"updated" all
    also get queued onto ToolDispatcher's correction worker, which
    corrects/translates the block via GeminiCorrectionClient
    asynchronously and patches `corrected` in place once it lands —
    read_aloud(scope="all")/get_reading_section always read whatever
    correction has landed by then, never blocking on it
  → PerceptionService.Synthesize streams KokoroTTS PCM back → played
    directly by the client — Gemini's own voice is NOT used
  → User asks about already-scanned content → get_reading_section(query)
    → local keyword search over the buffer → Gemini answers in its own voice
```

---

## Frame Extractor (offline tool, `frame_extractor/`)

Standalone Gradio tool (port 7863, `hrtf` conda env) — not part of the
Android/server live pipeline above. Upload a video, get back only the
frames whose ORB features aren't already covered by a previously-accepted
frame (see `extractor.py`'s module docstring for the RTAB-Map-style
novelty-gating algorithm: descriptor match + Essential Matrix RANSAC
against every prior accepted frame, not raw keypoint comparison).

- `extractor.py` — `extract_new_frames()`, `OrbNoveltyGate`. RTAB-Map
  pose/node-id and DA3 depth are optional metadata (`rtabmap_addr`,
  batched via `da3_batch_size`), no longer gate acceptance.
- `tagging.py` — `FrameTagger`: Gemini (`gemini-3.1-flash-lite` via the
  `google-genai` SDK, open-set tagging — supersedes an earlier RAM++-based
  tagging step, see "Semantic mapper adapted to frame_extractor's Gemini ->
  GroundingDINO-tiny pipeline" above) → GroundingDINO tiny (open-vocab
  detection, local), run on every accepted frame. The tagging step's own
  tags become GroundingDINO's per-frame text prompt, so boxes track what
  was actually tagged instead of one fixed prompt for every frame.
  GroundingDINO tiny loaded once and reused; Gemini tagging is one batched
  multi-image API call per flush. Frames are batched (`tag_batch_size`,
  independent of `da3_batch_size` — no chronological dependency, different
  VRAM/latency profile). GroundingDINO tiny resizes to its own preferred
  input resolution (shortest_edge=800/longest_edge=1333 processor config);
  Gemini receives full-resolution images directly. `draw_detections()`
  draws GroundingDINO boxes (yellow) on top of `extractor.py`'s
  ORB-keypoint annotation (green=new/red=old).
- `app.py` — Gradio UI; `_get_tagger()` caches the loaded `FrameTagger` at
  module level across requests (unlike the RTAB-Map client/DA3 estimator,
  which are cheap to reconstruct per call, loading GroundingDINO tiny still
  costs a few seconds — built once and reused, not reloaded per "Extract
  new frames" click). "Gemini API Key"/"Gemini tagging model id" textboxes
  (defaulting from `GEMINI_API_KEY`/`GEMINI_TAGGING_MODEL_ID`) replaced the
  old "RAM++ checkpoint path" textbox.

**Environment note**: `hrtf`'s `transformers` is pinned to `4.46.3` (not
5.13.0, unlike the rest of `hrtf`/`server/.venv`) — `AutoModelForZeroShot
ObjectDetection`'s `post_process_grounded_object_detection` has a different
signature at 4.46.3 (`box_threshold` param, `"labels"` dict key with
pre-decoded phrase strings) vs. 5.13.0's (`threshold`, `"text_labels"`),
which `tagging.py`'s GroundingDINO-tiny usage depends on — `server/tools/
detector.py`'s own (unrelated, full-size) GroundingDINO usage is
unaffected (different venv, `server/.venv`, stays on 5.13.0). The `ram`
(recognize-anything, RAM++) package this pin was ORIGINALLY needed for
(its vendored BERT copy breaks under 5.13.0 — `apply_chunking_to_forward`/
`find_pruneable_heads_and_indices` import paths, `PreTrainedModel` weight
tying internals, `BertTokenizer.additional_special_tokens_ids`, patched via
`frame_extractor/patch_ram_package.py`) is no longer imported by the live
tagging path at all (see the Gemini swap above) — `patch_ram_package.py`
and the `ram` install step are left in the repo, unreferenced, in case
RAM++ tagging is ever revisited, but the `transformers` pin itself stays
for GroundingDINO-tiny's own sake regardless. Verified end-to-end on a real
video with a live RTX 3060 BEFORE the Gemini swap (~4.6GB VRAM reserved for
both models at batch size 2; GroundingDINO's CUDA deformable-attention
kernel fails to JIT-compile against this environment's torch/CUDA
combination and silently falls back to its pure-PyTorch path — functionally
correct, just not the fastest possible path on this particular machine) —
not re-verified against a live GPU/Gemini API call from this environment
after the swap, compile-verified only.

## Pixie HRTF Test Harness (offline tool, `test_module/pixie_hrtf_app/`)

Standalone, throwaway test harness — not part of the main client/server
system — built at the user's request to validate the design behind a
future persistent-companion HRTF beacon ("Pixie") before it's built into
`client/android/` itself. The eventual real design (discussed but
deliberately NOT yet implemented in the main app — see this tool's own
README for the full deferred spec): a beacon entity that exists for the
whole app lifecycle regardless of mode, defaults to a fixed egocentric
position (front-left, a bit upper), exposes a `moveTo(destination, speed?)`
called by navigation/guiding (fly toward open space) or tracking (fly
toward the tracked target), moves gradually with the flapping sound only
audible while actually in transit (fading out shortly after arrival),
rides along with the user's own translation without that counting as
"flying," teleports to the default position on any loss-of-track/reset,
and gets a Settings toggle for whether Gemini Live's own voice is also
routed through the pixie's HRTF position.

This harness tests only the trickiest underlying piece first, per the
user's explicit "need to test first" request: whether an RTAB-Map-derived,
**anchor-relative** head heading (the first tracked frame after connecting
defines "straight ahead") can drive a stable-sounding HRTF direction, with
the server round-trip latency bridged by a local ORB/Essential-matrix
RANSAC drift estimate on the Android side (a duplicate of `client/
android`'s own `RotationTracker.kt`) — the same "server gives an
authoritative but stale fix, client bridges the gap locally" pattern this
project's real walking/guiding beacon already uses (see "Server-planned
walking path + client-side latency bridging" above), applied here to pure
head orientation instead of position-on-a-path.

- `server/pixie_hrtf_server.py` — WebSocket server (reuses `scan_server/
  rtabmap_client.py` + `da3_wrapper.py`, same RTAB-Map docker service the
  real `MappingService` talks to) + a Gradio dashboard. **Second design
  pass, per direct follow-up request**: the pixie now moves autonomously
  server-side (`PixieMotion`) — loops MOVING (travel ALONG the
  circumference of a radius/speed-configurable circle centered on the
  anchor, to a random point) then WAITING (configurable duration) then
  repeats; the dashboard's sliders configure radius/speed/wait/elevation
  instead of setting a fixed azimuth directly.
- `android/` — a separate Gradle project (`com.tracking.pixietest`,
  installable alongside the real client app on one test device), CameraX
  capture (Y-plane luma straight into OpenCV, no Bitmap/JPEG round trip —
  a real ~0.5s-per-update lag traced to that round trip, plus running at
  full sensor resolution instead of a bounded ~640x480 analysis stream) +
  `RotationTracker.kt` (duplicated from `client/android/`) for local
  drift bridging. **ORB detect+match only** — a Lucas-Kanade optical-flow
  alternative was tried and dropped outright per direct request; ORB's
  working resolution (3 dropdown presets) and feature count (dropdown,
  steps of 100) are both live-adjustable from the UI. `PixiePositionView`
  draws two compasses side by side (anchor-relative bearing, which doesn't
  rotate with the head, and facing-relative bearing, which does); a
  synthesized alignment tone (`PingTonePlayer.kt`) replaces the earlier
  continuous HRTF "flapping" beacon (`HrtfConvolver.kt`/
  `HrtfBeaconPlayer.kt`, still present, unreferenced) — silent within ±2°
  of facing the pixie exactly, louder the more deviated, a null-seeking
  cue rather than a peak-seeking one.
- Deliberately simplified vs. the eventual real design: the circle is
  centered on the anchor and assumes the user's own position stays there
  too (no live translation tracking — this harness only exercises head
  ROTATION); no `moveTo(destination, speed)` API called by other
  components, no flying-vs-attached-to-user distinction, no
  teleport-on-reset. See the tool's own README for the full list and
  setup/run instructions.
- Not verified against a live device/RTAB-Map rig from this environment —
  compile-verified only (`./gradlew assembleDebug` succeeded for the
  Android app; `python -m py_compile` succeeded for the server).

---

## Deployment

| Component | Default Port | Command |
|-----------|-------------|---------|
| OCR | — (external) | No server to run — Android calls OCR.space directly (API key entered in Settings); see "Reading-mode OCR" above |
| Main server | 50051 + Gradio 7860 | `python server/grpc_server.py` — imports scan_server/ in-process for MappingService, no separate Scan server process any more |
| RTAB-Map pose service (required) | 5556 (ZeroMQ, no ROS) | `docker compose up rtabmap` — see `scan_server/rtabmap_docker/README.md`; MappingService is disabled (logs and no-ops) without `RTABMAP_ADDR` set |

Docker: `docker build -f server/Dockerfile -t tracking-server .` (repo-root
context — see `server/Dockerfile.dockerignore`). The image bakes in only
the heavy, rarely-changing pieces (CUDA/Python/pip deps, `DA3METRIC-LARGE.onnx`
at `DA3_ONNX_PATH=/opt/models/DA3METRIC-LARGE.onnx`) — **no application code
is copied in at build time at all**. `server/entrypoint.sh` runs on every
`docker run` instead: clones `${REPO_REF}` (default `vi-slam`) fresh into
`/app` if it isn't already a checkout there, or `git fetch --depth 1` +
`git reset --hard` to update it if it is (the `/app/.git` check supports
both a fresh ephemeral container — always a clean clone — and a
persistent-volume-mounted `/app` — incremental update), then `pip install
--no-deps -e` the freshly-cloned `Depth-Anything-3` package before
`exec`-ing the real `CMD`. This means rebuilding the image is only ever
needed for a dependency change; a code change just needs `git push` +
restarting the container. `docker-compose.yml`'s own `streaming-vlm-server`
service is a **stale, unrelated leftover** (builds the root `Dockerfile`, a
different Qwen-VLM setup with a Windows host path) — don't use it for the
main server; build/run `server/Dockerfile` directly as above. The
`rtabmap` service in that same compose file is current and fine (needs no
per-device calibration, unlike the old `orbslam3` service it replaced) but
still isn't brought up automatically by a bare `docker-compose up`
everything-workflow — see `scan_server/rtabmap_docker/README.md` to build
it, or `docker compose up rtabmap`.

### Development Environments

| Component | Python env | Notes |
|-----------|-----------|-------|
| Main server | `server/.venv/` | activate: `source server/.venv/bin/activate` or prefix commands with `server/.venv/bin/python`. Needs the `hrtf` conda env's packages available too for the in-process scan_server/ imports (MappingService) — see below |
| scan_server/ modules (imported in-process by the Main server) | conda env `hrtf` | `conda activate hrtf` is the environment actually used for running/testing this code; `server/.venv/` is kept in sync |
| Android client | — | Gradle project root: `client/android/`; run `./gradlew build` from there. The only client — see "Client-Orchestrated Live Session" |

Environment variables:
- `GEMINI_API_KEY` — used server-side by `MappingService`'s/`scan_server.py`'s `SemanticMapper` for its tagging step (see "Semantic mapper adapted to frame_extractor's Gemini -> GroundingDINO-tiny pipeline" — no separate key needed, reused from this same var); the Gemini Live API key itself is entered in the Android app's Settings screen and never touches the server
- `RTABMAP_ADDR` — e.g. `tcp://localhost:5556` — **required** for `MappingService`; without it, `MappingService` registration is skipped entirely (logged, not fatal)
- `DA3_ONNX_PATH` — ONNX weight path for `PerceptionService.AnalyzeFrame`'s `DEPTH` op, always `DA3DepthDetector` now (default `DA3METRIC-LARGE.onnx`, or `/opt/models/DA3METRIC-LARGE.onnx` inside the Docker image — see Docker note above) — uses the DA3-METRIC checkpoint's own `metric_depth` output directly, no separate scale-alignment step; `SparseObstacleDetector`/`StereoDepthDetector`/the DA3 torch backend were removed, so there's no longer a `DEPTH_MODEL` selector
- `REPO_URL`/`REPO_REF` — Docker-image-only (`server/entrypoint.sh`), default `https://github.com/hungq1205/tracking` / `vi-slam` — which repo/branch the container clones/updates `/app` from on every start; not read by `grpc_server.py` itself
- `SCAN_DA3_TORCH_MODEL_ID` — DA3 torch model for `MappingService`'s live-mapping pipeline (default `depth-anything/DA3METRIC-LARGE`, monocular-metric — was `depth-anything/da3-large` until a live-debugging incident traced total RTAB-Map tracking failure to that non-metric default, see "DA3 model default + per-frame processing + pre-DA3 blur gate") — a separate subsystem (dense reconstruction depth, not obstacle checks), independent of `DA3_ONNX_PATH` above — see "Client-Orchestrated Live Session"
- `GEMINI_TAGGING_MODEL_ID` — Gemini model id for `MappingService`'s/`scan_server.py`'s `SemanticMapper` tagging step (default `gemini-3.1-flash-lite`); `GDINO_TAGGING_MODEL_ID` — GroundingDINO-base model id for the same (default `IDEA-Research/grounding-dino-base`) — see "Semantic mapper adapted to frame_extractor's Gemini -> GroundingDINO-tiny pipeline". `SCAN_GEMMA_MODEL_ID`/`gemma_vlm.py` are no longer read by any live path.
- `MEMORY_STORE_DIR` — on-disk dir for `RagStore`'s text-embedding storage (default `server/data/memory`)
- `RAG_MODEL_ID` — sentence-transformer model id for `RagStore.embed_text()`, backing `PerceptionService.Embed` (default `sentence-transformers/all-MiniLM-L6-v2`)

---

## Agent Instructions — Keeping This File Current

When you make changes to this project, **edit the relevant section of this file** in the same commit/session:

- **New RPC / proto field** → update the gRPC Services table + regeneration note
- **New agent or intent** → update the Agent System section
- **New client or script** → add a row to the Client Implementations table and Key Files Map
- **New AI model or tool** → add to the architecture diagram and Key Files Map
- **Port / env-var change** → update the Deployment table
- **Major refactor of a component** → update the narrative description and file path

Keep entries short and factual. Do not paste code into this file — reference the file path instead.
