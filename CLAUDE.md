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
│  • 3rd-party direct: OCR (paddle_ocr_server), Gemini Live itself     │
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

paddle_ocr_server (port 8100) — called directly by Android, not proxied.
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
  - `AnalyzeFrame(image, ops: {DETECT, EMBED, DEPTH, TRAVERSABILITY}, prompt?,
    box?) → detections[], embedding?, obstacle?, traversability?` — one round
    trip for whatever combo a caller needs (`run_detection`/`check_obstacle`
    on-demand tools; walking mode's own periodic DEPTH-op polling was removed
    long ago, see "Local reactive HRTF obstacle-dodge" below), via
    `detector.detect_all()` (sorted by score, replaces the old single-best
    `detect()` for this path)/`embedder.get_embedding()`/
    `depth_detector.check_obstacle()`/`depth_detector.estimate_traversability()`
    (new — per-angle obstacle-clearance fan, see "Local reactive HRTF
    obstacle-dodge" below; shares the same DA3 inference call `check_obstacle`
    already makes, so requesting both `DEPTH` and `TRAVERSABILITY` in one call
    doesn't pay for DA3 twice).
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
    grid_updated, landmarks, confidence, grid_delta, full_resync}` — bidi
    stream. **RTAB-Map is the ONLY pose source used here** (not IMU+VO) —
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
    create a session) like `GetMapSnapshot`/`ListMappedLocations` below.
    `ToolDispatcher.kt`'s `recomputeRoute()` calls this for EVERY guiding
    destination now (no live-streamed landmarks list to check first).
  - On stream end (client closes/disconnects — `finally` block, so this
    fires on a clean stop or an abrupt drop): `stream.flush()` (last partial
    batch), `session.finalize_landmarks_flat()` (see below), then persists
    an `occupancy_snapshot.json` per location — deliberately NOT using the
    legacy zone-shaped `map_exporter.py`/`ScanSession.export()` path (calling
    both would double-drain the frame store's tag-pending buffer, since both
    finalize methods flush it).
  - `GetMapSnapshot(location_id) → found, grid, landmarks` /
    `ListMappedLocations() → location_ids` — read `occupancy_snapshot.json`
    directly, no live session needed.
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
  case-insensitive substring, either direction) rather than fuzzy-matched
  further, since `FindLandmark`'s own Tier 2 GroundingDINO fallback is
  already the safety net for a near-miss. **Known follow-on, not yet
  built**: nothing memoizes "already tried and failed this session," so an
  unresolved destination retries the RPC on every grid update — worth a
  debounce if it turns out to spam the RPC in practice.

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

### Occupancy-grid persistence across sessions (coarse re-seed)

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
`server_gui.py` panel on the **Perception tab** (not Mapping — walking no
longer touches `MappingService` at all, so its only server traffic is
`PerceptionService.AnalyzeFrame(TRAVERSABILITY)` + `StatusService`;
`_TAB_BY_CLIENT_MODE["walking"]` now points at `tab_perception`
accordingly): `_render_beacon_polar()` draws the last traversability fan as
a Plotly polar bar chart (forward = 12 o'clock, azimuth-right reads
clockwise, matching `HrtfBeacon.kt`'s sign convention) with a marker at the
client-reported final azimuth, greyed out when muted. The old magenta
circle overlays on the Mapping tab's frame/occupancy-map views are removed
(nothing to draw any more — the real beacon direction was never grid-space
to begin with now).

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

### Session-mode pipeline split — SCAN vs. WALKING/GUIDING (`walking_lite`)

Real bug, found via a live device session and fixed: walking mode was
running the exact same FULL pipeline scanning does — RTAB-Map's own
`get_cloud()` reconstruction pull + SOR + server-side voxelize (measured
15s+ and 12s+ respectively on a real batch) AND VLM/semantic tagging — on
every single mini-batch, making walking unusably slow (30-53s stalls
between occupancy updates) for a mode that only ever needed a live
occupancy grid, never a persisted point cloud or landmark discovery.

**Note (post "Local reactive HRTF obstacle-dodge"): `SessionMode.WALKING`
is now effectively dead.** Walking dropped `MappingService` entirely, so
`feedMappingFrame()` is never called while `state.mode == "walking"` any
more — nothing ever sends this enum value in practice. The `walking_lite`
mechanism described below stays fully alive and necessary for `GUIDING`
(still a non-`SCAN` mode), which is the only mode that reaches it now. The
enum value itself was left in the proto rather than removed — harmless to
keep, and removing it would be a wire-compatibility churn for no benefit.

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
- **VLM/semantic tagging (`StoredFrame`/`_frame_store`/`_tag_pending`) is
  skipped entirely** — semantic mapping is scan-only now; walking/guiding
  never discover landmarks live.
- **`FindLandmark` gained a persisted-snapshot fallback**
  (`MappingServiceServicer._find_in_snapshot()`) for exactly this reason —
  a walking/guiding session's `_frame_store` is always empty (VLM tagging
  never ran), so `session.resolve_landmark()` alone would never find
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
the occupancy map via local back-projection, leaves `_frame_store` empty,
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

### Drop-to-latest mapping-chunk ingestion — tried, then reverted (`mapping_servicer.py`)

**Current state: REVERTED.** `UpdateMapping` iterates the raw gRPC
`request_iterator` directly again (`for chunk in request_iterator:`) — the
full queue, no chunk ever silently dropped. Confirmed with the user: bring
the queue back. Documented in full below since the reasoning for trying
the alternative in the first place is still real context (and the "Round
1" reasoning in "DA3 model default + per-frame processing" below, about
this mailbox interacting with `mini_batch`, refers to this same mechanism
— it no longer exists, but the mini_batch conclusions it led to are still
current).

**What was tried**: gRPC's own request iterator queues incoming messages
internally — if the server-side loop falls behind the client's send rate
even briefly (a slow `push_frame()` mini-batch, GPU contention from
another stream, a depth-estimation stall), the default behavior is to keep
working through that backlog in arrival order, so the pose/grid state this
stream reports gets further behind real time and doesn't recover on its
own — the same kind of compounding lag this project has hit before (see
"Mode exclusivity" above for a related GPU-contention incident).
`_latest_only_chunks(request_iterator)` wrapped the raw iterator in a
single-slot mailbox — a background reader thread continuously drained
`request_iterator` and overwrote one shared slot with each new chunk as it
arrived (a chunk not yet consumed by the main processing loop was silently
dropped in favor of whatever was newest), while the main loop blocked on a
`threading.Condition` and always pulled whatever was CURRENTLY in the slot.
This mirrored `CameraManager.kt`'s own `frameFlow`
(`extraBufferCapacity` + `DROP_OLDEST`) — the same "prefer fresh over
complete" policy already used client-side.

**Why it was reverted**: RTAB-Map's own frame-to-frame odometry needs
CONTINUITY, not freshness — dropping frames (even just the ones arriving
while the server is briefly busy) can widen the visual/motion gap between
two frames RTAB-Map actually processes back to back, which is a real
contributor to the tracking-loss investigation in "DA3 model default +
per-frame processing" below (Round 1's reasoning). Restoring the full
queue trades "never falls behind in wall-clock time" for "never
drops a frame RTAB-Map needed to stay tracked" — the latter matters more
for this pipeline. Applies only to `UpdateMapping` — the only streaming
RPC left in this codebase (`TrackingService`/`PerceptionService` are
unary, one frame per call, so they have no equivalent queueing exposure).

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
  buffer, nav waypoints, walking obstacle cache), held in `MainViewModel` now
  instead of server-side.
- `ToolDispatcher.kt` — Kotlin port of `_dispatch_tool` + the `live_tools/*.py`
  implementations. One implementation per tool, routed by weight:
  - **Remote** (`grpc.perceptionStub`/`mappingStub`, new fields on
    `GrpcClientManager`): `run_detection`/`check_obstacle` →
    `AnalyzeFrame`; `read_aloud` → `Synthesize`; `query_memory`/
    `save_memory`/etc.'s vector step → `Embed`; `start_guiding`/
    `start_walking`/`start_scan` → `UpdateMapping` bidi stream (fed by
    `feedMappingFrame()`, called from `MainViewModel`'s existing camera-frame
    collector whenever `mode` is `guiding`/`walking`/`scanning`).
    `recomputeRoute()` (fixed — was a no-op placeholder) resolves
    `guidingDestinationLabel` via `FindLandmark` on every grid update while
    unresolved, then routes with `LocalPathPlanner` below.
  - **3rd-party direct**: `OcrClient.kt` — multipart POST straight to
    `paddle_ocr_server`, bypassing the gRPC server as a proxy.
  - **Local**: `LocalMemoryStore.kt` (on-device JSON files under
    `filesDir/memory/` — labels/notes + a small embedding index, cosine
    search in Kotlin; the `Embed` RPC is the only remote step),
    `LocalPathPlanner.kt` (full Kotlin port of `live_path_planner.py`'s A* —
    same cost model, closest-approach fallback, min-clearance penalty —
    run against the `OccupancyGrid` `MappingService` streams back, so
    routing has zero network round-trip per step), `HrtfBeacon.kt`
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
    changes. Reading-buffer dedup (`MemoryTextUtils` in
    `LocalMemoryStore.kt`) ports `memory_store.py`'s sentence-overlap
    filtering exactly.
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
- `MainViewModel.kt`: `connect()` gained `geminiApiKey`/`ocrServerUrl`/
  `locationId` params (new `SettingsScreen.kt` fields, persisted via
  `SettingsViewModel`); `doLiveSession()` rewritten around
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

### Server-side scan_server.py deletion

`scan_server/scan_server.py` (the FastAPI `/api/upload` entrypoint +
`mount_gradio_app` launcher) was deleted — confirmed zero remaining callers
repo-wide once `ScanViewModel.kt` (its only HTTP client) was removed. The
rest of `scan_server/` (`scan_gui.py`, `stream_simulator.py`,
`zone_labeler.py`, the zone-based `map_exporter.py` export path, etc.) was
**deliberately left in place, not deleted** — `scan_gui.py`'s rich Live
Reconstruction/Occupancy Map/Voxelization/Live Navigation Preview panels
were meant to be merged into the main server's Gradio dashboard
(`server_gui.py`, port 7860) as the debugging/monitoring UI for
`MappingService`, but that richer merge still hasn't happened.
`server_gui.py`'s own "Mapping" tab (see the ActivityMonitor rewrite above)
only shows the last received frame + pose/grid_updated/confidence — a much
thinner view than `scan_gui.py`'s Occupancy Map/Voxelization/Confidence Map
panels, which still have no equivalent on the live dashboard; deleting
`scan_gui.py` now (or the modules it depends on) would remove the only way
to visually inspect the live mapping pipeline in that depth. `scan_gui.py`
currently has no launcher (its only one, `scan_server.py`, is gone) — it's
orphaned, not functional, until either the richer Gradio-dashboard merge
happens or someone launches it with a small ad-hoc script.

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
job now happens via `MappingService` + Android's `LocalPathPlanner.kt`/
`HrtfBeacon.kt` instead, per the new architecture.

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
                                   3 tabs (Tracking/Perception/Mapping), each showing the last frame
                                   + result for that RPC category, plus a rolling Activity Log tab.
                                   Tab auto-selection prefers ActivityMonitor.client_mode
                                   (StatusService.ReportMode, authoritative) over inferring from
                                   whichever RPC category most recently updated (fallback for older
                                   clients) — "walking" now maps to tab_perception, not tab_mapping,
                                   since walking no longer touches MappingService at all (see "Local
                                   reactive HRTF obstacle-dodge"). Mapping tab shows the last frame and
                                   occupancy_map.py's own render_plotly() SIDE BY SIDE in one row,
                                   against the LIVE OccupancyMap reference ActivityMonitor's mapping
                                   bucket holds — occupancy-grid-only (routing use only now, see
                                   below), deliberately no point-cloud/voxel/confidence view (those stay
                                   scan_gui.py's separate, heavier offline debug tool — render_confidence_
                                   plotly() was shown here too originally, dropped as unnecessary for this
                                   at-a-glance live dashboard); wrapped in try/except since that reference
                                   is mutated concurrently by the gRPC streaming thread while this renders
                                   on the Gradio polling thread — a race should skip a tick, not crash the
                                   dashboard. RTAB-Map pose-lost is now surfaced directly (not just
                                   console): a red "RTAB-Map TRACKING LOST (N/M)" burned into the
                                   annotated frame (frame_rgb is RGB order, red=(255,0,0)) plus a line in
                                   the Detail textbox, both driven by ActivityMonitor's rtabmap_lost/
                                   rtabmap_total fields (mapping_servicer.py) — see "Blur filtering
                                   removed for scan/walking" above. Perception tab gained a beacon-
                                   direction panel — _render_beacon_polar()/_beacon_status(), a Plotly
                                   polar chart of the last AnalyzeFrame(TRAVERSABILITY) fan plus a
                                   marker at the client-reported final azimuth (ActivityMonitor.
                                   beacon_azimuth_deg/beacon_muted, fed by StatusService.
                                   ReportBeaconDirection) — see "Local reactive HRTF obstacle-dodge".
                                   The old magenta beacon-position circles on the Mapping tab's frame/
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
    mapping_servicer.py          MappingServiceServicer — UpdateMapping/GetMapSnapshot/
                                   ListMappedLocations/FindLandmark. Records into ActivityMonitor's
                                   mapping bucket on every yielded MappingUpdate (frame, pose,
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
                                   UpdateMapping reads `for chunk in request_iterator:` directly (the full
                                   queue, no dropping) — a drop-to-latest mailbox was tried and reverted,
                                   see "Drop-to-latest mapping-chunk ingestion" below for why.
                                   No configure_novelty_gate() call any more — blur filtering was tried
                                   (SCAN_MIN_SHARPNESS/WALKING_MIN_SHARPNESS) then removed entirely per
                                   "Blur filtering removed for scan/walking" above; self._min_sharpness
                                   stays at DEFAULT_MIN_SHARPNESS (0.0, blur gating off). record_mapping()
                                   now also passes rtabmap_lost/rtabmap_total (from session.last_rtabmap_
                                   lost/last_rtabmap_total) so server_gui.py can show RTAB-Map pose-lost
                                   status directly instead of only in console output.
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
                                   _depth_map() DA3 call between check_obstacle() and the new
                                   estimate_traversability() (delegates to traversability.py, below) —
                                   see "Local reactive HRTF obstacle-dodge".
    traversability.py            (new) estimate_traversability(depth_map, num_bins, max_range_m) —
                                   stateless, single-frame polar obstacle-clearance fan: back-project via
                                   the same pinhole-K fallback used throughout this codebase, RANSAC-fit
                                   a ground plane from the frame's bottom ~40%, classify obstacle vs.
                                   ground by height above it, bucket by azimuth. No IMU, no persisted
                                   ground_y (deliberately NOT occupancy_map.py's accumulated-belief
                                   design — a world map is too slow for reactive per-frame dodging). Two
                                   real bugs found via synthetic ground-truth testing before this
                                   shipped (sign-flip using the wrong reference point; a frontal
                                   obstacle filling the frame getting accepted as "the floor") — see
                                   "Local reactive HRTF obstacle-dodge" for both.
    rag_store.py                 Sentence-transformer text embeddings (embed_text(), backs PerceptionService.Embed);
                                   storage/search methods (add_text/query_global) are now unused server-side —
                                   storage lives on Android (LocalMemoryStore.kt) — kept for reference/DummyRagStore
    embedder.py                  DINOv2Embedder (ViT-S/14) — visual re-ID embeddings
    tts.py                       KokoroTTS.synthesize_pcm_chunks() — backs PerceptionService.Synthesize
  _archived/                     Old orchestrator/, agents/, cloud_vlm, intent_parser (reference only)
  data/
    maps/{location_id}/          occupancy_snapshot.json (MappingService) — see "Client-Orchestrated
                                   Live Session"; the legacy map_geometry.ply/map_labels.json zone-based
                                   export still exists per-location from before this migration

scan_server/
  scan_server.py                 Entry point; FastAPI + Gradio UI, port 7861
                                   POST /api/upload — receives dataset.zip (images/+imu.csv+camera.csv)
                                   from Android, extracts to uploads/<scan_id>/dataset/
                                   GET  /api/uploads — lists available upload scan IDs
  scan_gui.py                    Gradio UI — dataset folder path + segment table → export.
                                   create_scan_ui(scan_manager, upload_dir) — "Load from Android Upload" accordion
                                   pre-fills the dataset folder path; _handle_dataset_change()/
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
                                   same live-tunable pattern as occupancy above); self._frame_store/
                                   _tag_pending (List[StoredFrame], session-scoped, cleared by
                                   reset_cloud()); resolve_landmark()/_resolve_all_frame_store_
                                   landmarks() (deferred GroundingDINO lookup) — see "Novelty+blur frame
                                   gating and deferred landmark resolution" above for the full design.
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
                                   Navigation Preview" note)
  semantic_mapper.py             SemanticMapper — VLM landmark TAGGING (name-only, no GroundingDINO/
                                   coordinates) + deferred backprojection — see "Novelty+blur frame
                                   gating and deferred landmark resolution" above. tag_landmarks_batch
                                   (frames) -> List[List[str]]: ONE VLM call across up to
                                   IMAGES_PER_PROMPT=5 frames, N lines of comma-separated names, one
                                   per image — stateless, ScanSession owns the buffering now (no more
                                   internal _pending/window state, no more consider_frame()/
                                   extract_landmarks()/flush()). _detect_and_backproject() (GroundingDINO
                                   + depth-median-sample + 4-corner backprojection) is UNCHANGED but
                                   only called from ScanSession.resolve_landmark() now, never
                                   proactively. Landmark dataclass; cluster_landmarks() unchanged.
  gemma_vlm.py                    GemmaVLMClient — Gemma 4 31B via Gemini API (GEMINI_API_KEY),
                                   multi-image query(prompt, images=[...]) for semantic mapping
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
      PushToTalkRecorder.kt        PTT recording; onChunkReady emits raw PCM chunks during hold
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
                                     (String — "", "guiding", "scanning" — set every processed frame by
                                     MainViewModel.kt's frame collector, forwarded verbatim from
                                     sessionState.mode; walking is deliberately NOT in this set any more —
                                     it dropped MappingService entirely, see "Local reactive HRTF
                                     obstacle-dodge"): (1) mapping modes (guiding/scanning) — NO
                                     blur/clarity filtering (removed, confirmed with the user — see "Blur
                                     filtering removed for scan/walking"): whichever frame arrives once
                                     frameIntervalMs (ms, guiding) or scanIntervalMs (ms, scanning) has
                                     elapsed since the last send is forwarded directly — no window, no
                                     candidate comparison. Both stay ms internally; SettingsScreen exposes
                                     them as FPS number inputs, not sliders ("Mapping FPS" 0.2..10 default
                                     1.0, "Scan FPS" 2..20 default 10.0), converting fps->ms only at
                                     doConnect() time. activeMappingSubmode tracks which one is in effect and
                                     resets the send-gate on any submode change (a stale timestamp from a
                                     different mode/interval shouldn't suppress the new mode's first send); (2)
                                     recentBufferMs (SettingsScreen slider, 0..1000ms, 50ms steps, default
                                     100) — every other mode (tracking/reading/Q&A/idle/walking) — a small
                                     rolling buffer of the last recentBufferMs of frames, no window/gap logic;
                                     clearestRecentFrame() pulls the sharpest buffered frame on demand (used
                                     by ToolDispatcher.kt's latestFrame() closure — OCR/run_detection/
                                     tracking-init calls, AND now walking/guiding's own local-avoidance tick,
                                     see "Local reactive HRTF obstacle-dodge" — all want "the current frame"
                                     right now), while handleRecentEmit() emits that same clearest-so-far
                                     frame into frameFlow at most once per recentBufferMs for continuous
                                     per-frame consumers (hand tracking, local ORB tracking, UI overlay).
    sensors/
      ImuSensor.kt                 SensorManager wrapper; emits ImuReading Flow at SENSOR_DELAY_FASTEST
      ImuRecorder.kt               Writes imu.csv (header: timestamp_ns,ax,ay,az,gx,gy,gz) alongside images/
    device/
      DeviceToolHandler.kt         Interface for executing device-native tool calls (phone/alarm/calendar)
      AndroidDeviceToolHandler.kt  Implementation: Intent.ACTION_CALL, AlarmClock, CalendarContract
    live/                          Client-orchestrated Gemini Live session — see CLAUDE.md's
                                     "Client-Orchestrated Live Session" section for the full picture
      GeminiLiveClient.kt           Raw WebSocket client for Gemini Live's BidiGenerateContent protocol
      ToolDeclarations.kt           SYSTEM_PROMPT + FunctionDeclaration JSON, ported from tool_declarations.py
      LiveSessionState.kt           Port of server/live_session.py's LiveSessionState. smoothedBeaconAzimuthDeg
                                     (new) — the HRTF beacon's EMA-smoothed azimuth, carried across local-
                                     avoidance ticks (see ToolDispatcher.kt below).
      ToolDispatcher.kt             Port of _dispatch_tool + live_tools/*.py — remote/3rd-party/local/device.
                                     stopActiveModes() — called first by every mode-entry tool, enforces
                                     state.mode exclusivity (see "Mode exclusivity" note above). reportMode()
                                     — fire-and-forget StatusService.ReportMode call on every mode transition.
                                     startLocalAvoidanceTicks()/runAvoidanceTick() (new) — the local reactive
                                     HRTF obstacle-dodge loop for walking AND guiding, see "Local reactive
                                     HRTF obstacle-dodge" above; replaced updateHrtfBeacon() (removed).
      OcrClient.kt                  Direct 3rd-party HTTP client to paddle_ocr_server
      LocalMemoryStore.kt           On-device JSON memory store + embedding index + cosine search
      LocalPathPlanner.kt           Kotlin port of live_path_planner.py's A* (clearance-aware, closest-approach)
                                     for guiding's global route only now — findMostOpenDirection()/
                                     castOpenRay() (walking's old grid-raycast steering) were removed
                                     outright, see "Local reactive HRTF obstacle-dodge" above.
      TraversabilityScorer.kt       (new) pickSteeringAngle() — Vector-Field-Histogram-style local
                                     obstacle-dodge scoring against a per-frame TraversabilityInfo fan:
                                     corridor-windowed clearance minus goal-bearing penalty (guiding only)
                                     minus steering-effort penalty, peak-picked; smoothAzimuth() — EMA
                                     toward the picked angle. See "Local reactive HRTF obstacle-dodge".
      HrtfBeacon.kt                 directionTo() — egocentric waypoint azimuth/elevation from the
                                     current Pose (guiding's goal-bias input only now — its own
                                     elevation/distance are no longer used to place the beacon directly);
                                     directionFromBox() — pixel-offset azimuth/elevation from a 2D ORB
                                     tracking box (tracking mode, no pose/depth available). worldYawRad()
                                     was removed outright (its only caller, findMostOpenDirection(), is
                                     gone — see "Local reactive HRTF obstacle-dodge").
      MutableOccupancyGrid.kt       (new) Persistent, patchable occupancy grid — fromFull()/applyDelta()/
                                     toProto(). Backs LiveSessionState.mutableGrid; ToolDispatcher's
                                     mapping-stream collect loop (guiding/scanning only now) patches it
                                     from MappingUpdate.grid_delta instead of replacing lastMappingGrid
                                     wholesale every update — see "Occupancy grid delta sync" above.
    ui/
      MainViewModel.kt             connect(host, port, fps, avoidanceIntervalMs, vadThreshold,
                                     startThreshold, geminiApiKey, ocrServerUrl, locationId) —
                                     doLiveSession() opens a GeminiLiveClient directly (no more
                                     server-relayed VoiceChatStream for this client) and drives
                                     ToolDispatcher.dispatch() for every Gemini function call;
                                     feedMappingFrame() called from the camera-frame collector while
                                     mode is guiding/scanning only now (walking dropped MappingService
                                     entirely — see "Local reactive HRTF obstacle-dodge" above).

paddle_ocr_server/               Standalone OCR microservice (port 8100), called directly by Android now
  server.py                       FastAPI /ocr endpoint — decode -> _preprocess (grayscale/upscale/
                                    denoise/adaptive-threshold) -> PaddleOCR.predict -> raw per-line
                                    text blocks -> _merge_into_paragraphs. Records every stage into
                                    ocr_monitor (module-level OcrMonitor) for ocr_gui.py to display.
                                    Mounts the Gradio debug UI onto this same FastAPI app via
                                    gr.mount_gradio_app(app, ..., path="/gui") — uvicorn server:app
                                    stays the single entrypoint, dashboard at :8100/gui.
  ocr_monitor.py                  OcrMonitor — thread-safe single-slot ("last request only") snapshot
                                    of every pipeline stage's image/blocks, same pattern as server/
                                    services/activity_monitor.py.
  ocr_gui.py                      Gradio dashboard — 5 panels: received, preprocessed, raw text
                                    blocks (boxed on the orientation-corrected image PaddleOCR itself
                                    used), merged paragraphs (boxed), final text only. Timer-polls
                                    OcrMonitor every 0.5s, same live-monitor pattern as server_gui.py.

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

Two layers now — global route (guiding only, `MappingService`) and local
reactive dodge (both modes, `PerceptionService`) — see "Local reactive
HRTF obstacle-dodge" for the full design.

```
User says "guide me to the couch" → start_guiding tool
  → GLOBAL: ToolDispatcher opens a MappingService.UpdateMapping bidi
    stream, feeding camera frames (RTAB-Map pose only) → server streams
    back Pose + OccupancyGrid (only when changed) + landmarks →
    LocalPathPlanner.findPath() (Kotlin A*) computes/updates a route
    entirely on-device against the received grid; checkWaypointProgress()
    sendSystemNote()s "[SYSTEM] Waypoint reached"/"Arrived at X" for Gemini
    to react to in audio.
  → LOCAL (same as walking below, goal-biased): the actual HRTF beacon
    direction — HrtfBeacon.directionTo()'s azimuth (to the current
    waypoint) feeds in as the goal-bias term, not as the beacon's direction
    directly any more.

User says "start walking" → start_walking tool (no MappingService at all)
  → ToolDispatcher.startLocalAvoidanceTicks(): every avoidanceIntervalMs,
    pull the current frame (clearestRecentFrame()) →
    PerceptionService.AnalyzeFrame(TRAVERSABILITY) → per-frame, ground-
    segmented obstacle-clearance fan, no world state → TraversabilityScorer
    picks the most open direction (pure clearance, no goal bias — walking
    has no destination) → EMA-smoothed → HrtfBeacon plays continuously
    toward that azimuth (elevation 0, fixed radius — pure steering command,
    not a position), muted only when nothing safe is found. Purely
    ambient — no [SYSTEM] messages, no spoken alerts, matching this
    project's established beacon behavior. Replaced the old fixed-interval
    PerceptionService.AnalyzeFrame(DEPTH op) polling + spoken "[SYSTEM]
    Obstacle ~Xm ahead" alert (`quick_label_obstacle` tool,
    `LiveSessionState.walkingObstacleCache`) entirely — both long since
    removed, not deprecated — AND its own successor, the occupancy-grid
    `findMostOpenDirection()` ray-cast, which is now removed too (too slow
    to react — see "Local reactive HRTF obstacle-dodge").
```

### Reading a document aloud
```
User says "read this" → enter_reading_mode() then read_aloud(scope="new")
  → OcrClient.kt POSTs the current frame directly to paddle_ocr_server
    (no server proxy) → new text deduped against the local reading buffer
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
- `tagging.py` — `FrameTagger`: RAM++ (open-set image tagging) →
  GroundingDINO tiny (open-vocab detection), run on every accepted frame.
  RAM++'s own tags become GroundingDINO's per-frame text prompt, so boxes
  track what RAM++ actually saw instead of one fixed prompt for every
  frame. Both models loaded once and reused; frames are batched
  (`tag_batch_size`, independent of `da3_batch_size` — no chronological
  dependency, different VRAM profile). Each model gets resized to its own
  preferred input resolution (RAM++ 384x384; GroundingDINO tiny's own
  shortest_edge=800/longest_edge=1333 processor config) rather than native
  frame resolution. `draw_detections()` draws GroundingDINO boxes (yellow)
  on top of `extractor.py`'s ORB-keypoint annotation (green=new/red=old).
- `app.py` — Gradio UI; `_get_tagger()` caches the loaded `FrameTagger` at
  module level across requests (unlike the RTAB-Map client/DA3 estimator,
  which are cheap to reconstruct per call, a RAM++ checkpoint load is
  expensive — ~1-2 min — so it's built once and reused, not reloaded
  per "Extract new frames" click).

**Environment note**: `hrtf`'s `transformers` is pinned to `4.46.3` (not
5.13.0, unlike the rest of `hrtf`/`server/.venv`) — the `ram`
(recognize-anything, RAM++) package vendors its own BERT copy against a
much older transformers API (`apply_chunking_to_forward`/
`find_pruneable_heads_and_indices` import paths, `PreTrainedModel` weight
tying internals, `BertTokenizer.additional_special_tokens_ids`), all of
which broke under 5.13.0; patched in the installed `ram` package itself
(`site-packages/ram/models/bert.py`, `site-packages/ram/models/utils.py`,
applied via `frame_extractor/patch_ram_package.py` — see
`frame_extractor/requirements.txt`) rather than forking the repo, and
pinning `transformers` down to a version old enough for those internals to
still exist was simpler than chasing further breakage forward.
`AutoModelForZeroShotObjectDetection`'s
`post_process_grounded_object_detection` also has a different signature at
4.46.3 (`box_threshold` param, `"labels"` dict key with pre-decoded phrase
strings) vs. 5.13.0's (`threshold`, `"text_labels"`) — `server/tools/
detector.py`'s GroundingDINO usage is unaffected (different venv,
`server/.venv`, stays on 5.13.0). Verified end-to-end on a real video with
a live RTX 3060 (~4.6GB VRAM reserved for both models at batch size 2;
GroundingDINO's CUDA deformable-attention kernel fails to JIT-compile
against this environment's torch/CUDA combination and silently falls back
to its pure-PyTorch path — functionally correct, just not the fastest
possible path on this particular machine).

## Deployment

| Component | Default Port | Command |
|-----------|-------------|---------|
| OCR server | 8100 (+ Gradio at `:8100/gui`) | `cd paddle_ocr_server && uvicorn server:app --host 0.0.0.0 --port 8100` (called directly by Android; pipeline-stage debug UI mounted on the same port, no separate process) |
| Main server | 50051 + Gradio 7860 | `python server/grpc_server.py` — imports scan_server/ in-process for MappingService, no separate Scan server process any more |
| RTAB-Map pose service (required) | 5556 (ZeroMQ, no ROS) | `docker compose up rtabmap` — see `scan_server/rtabmap_docker/README.md`; MappingService is disabled (logs and no-ops) without `RTABMAP_ADDR` set |

Docker: `docker-compose up` (requires NVIDIA runtime; mounts model volume). The
`rtabmap` service needs no per-device calibration (unlike the old `orbslam3`
service it replaced) but still isn't brought up automatically by a bare
`docker-compose up`-everything workflow — see
`scan_server/rtabmap_docker/README.md` to build it.

### Development Environments

| Component | Python env | Notes |
|-----------|-----------|-------|
| Main server | `server/.venv/` | activate: `source server/.venv/bin/activate` or prefix commands with `server/.venv/bin/python`. Needs the `hrtf` conda env's packages available too for the in-process scan_server/ imports (MappingService) — see below |
| scan_server/ modules (imported in-process by the Main server) | conda env `hrtf` | `conda activate hrtf` is the environment actually used for running/testing this code; `server/.venv/` is kept in sync |
| Android client | — | Gradle project root: `client/android/`; run `./gradlew build` from there. The only client — see "Client-Orchestrated Live Session" |

Environment variables:
- `GEMINI_API_KEY` — used server-side only by `MappingService`'s `SemanticMapper` (Gemma VLM landmark extraction); the Gemini Live API key itself is entered in the Android app's Settings screen and never touches the server
- `RTABMAP_ADDR` — e.g. `tcp://localhost:5556` — **required** for `MappingService`; without it, `MappingService` registration is skipped entirely (logged, not fatal)
- `DA3_ONNX_PATH` — ONNX weight path for `PerceptionService.AnalyzeFrame`'s `DEPTH` op, always `DA3DepthDetector` now (default `DA3METRIC-LARGE.onnx`) — uses the DA3-METRIC checkpoint's own `metric_depth` output directly, no separate scale-alignment step; `SparseObstacleDetector`/`StereoDepthDetector`/the DA3 torch backend were removed, so there's no longer a `DEPTH_MODEL` selector
- `SCAN_DA3_TORCH_MODEL_ID` — DA3 torch model for `MappingService`'s live-mapping pipeline (default `depth-anything/DA3METRIC-LARGE`, monocular-metric — was `depth-anything/da3-large` until a live-debugging incident traced total RTAB-Map tracking failure to that non-metric default, see "DA3 model default + per-frame processing + pre-DA3 blur gate") — a separate subsystem (dense reconstruction depth, not obstacle checks), independent of `DA3_ONNX_PATH` above — see "Client-Orchestrated Live Session"
- `SCAN_GEMMA_MODEL_ID` — Gemma model for `MappingService`'s landmark extraction via Gemini API (default `gemma-4-31b-it`); uses `GEMINI_API_KEY` above
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
