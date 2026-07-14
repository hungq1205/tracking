# Tracking — Vision-Assistance System

> **For AI agents:** This file is the authoritative living document for this project.
> **Every time you add, remove, or significantly change a component, update the relevant section here.**
> Keep it accurate and scannable — future agents rely on it to understand the system without re-reading the whole codebase.

---

## Purpose

This is an AI-powered assistive system for **vision-impaired people**.  
A user wears or carries a camera (Raspberry Pi, Android phone, or webcam). The system:

1. **Tracks objects** in the scene and gives real-time spatial guidance ("move left", "closer")  
2. **Reads text aloud** — screens, documents, labels — using OCR and TTS sentence-by-sentence  
3. **Builds 3D maps** of environments so the user (or caregivers) can label named zones (e.g., "kitchen", "sofa")  
4. **Answers questions** about the scene via voice or text chat  
5. **Stores and recalls memories** so the user can say "where did I put my keys?"  

All modalities (voice input, voice output, vision, memory) combine into one portable, real-time pipeline.

---

## System Architecture
Note that python env is at: server/.venv/ (server/mediator) — scan_server is actually run/tested via conda env `hrtf`; see Development Environments below.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Camera source (Pi / Android / webcam)                              │
│  + IMU (Android SensorManager: accel + gyro via ImuSensor.kt)       │
│    • Offline scan: CameraManager.kt dumps images/*.jpg + camera.csv, │
│      ImuRecorder.kt writes imu.csv — same dataset/ folder            │
│    • Live stream: IMUFrame proto chunks merged into VoiceChatChunk   │
└───────────┬─────────────────────────────────────────────────────────┘
            │ JPEG frames + voice audio + IMUFrame chunks (gRPC / protobuf)
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Mediator Service  ·  client/mediator_gui.py  ·  port 50052         │
│  • Local ORB-based tracking (CPU, Pi-friendly)                      │
│  • MediaPipe hand detection                                          │
│  • Throttled frame forwarding to Main Server (≤3 FPS)               │
│  • Gradio monitor dashboard (port 7862)                              │
└───────────┬─────────────────────────────────────────────────────────┘
            │ gRPC to Main Server  (protobuf)
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Main Server  ·  server/grpc_server.py  ·  port 50051               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │  TrackingService │  │   MapService     │  │  MediatorService  │  │
│  │  (all RPCs)      │  │  (static maps)   │  │  (relay RPCs)     │  │
│  └──────────────────┘  └──────────────────┘  └───────────────────┘  │
│  AI models loaded once on startup:                                   │
│  • Depth detector (tools/depth.py) — selected by DEPTH_MODEL env:   │
│    sparse (default): SparseObstacleDetector — ORB relative depth     │
│    stereo: StereoDepthDetector — plane sweep MVS, metric depth (m)   │
│  • GroundingDINO   — open-vocab object detection                     │
│  • DINOv2 ViT-S/14 — re-ID embeddings (cosine ≥ 0.75 = same target) │
│  • DocLayoutRapidOCR (remote, paddle_ocr_server port 8100)           │
│  Orchestration: LiveAPISession (live_session.py)                     │
│  • Gemini Live API (gemini-3.1-flash-live-preview) — ASR + LLM + TTS│
│  • Per-user WebSocket session; Gemini calls tools via function calls  │
│  Gradio monitor dashboard (port 7860)                                │
└───────────┬─────────────────────────────────────────────────────────┘
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
 LiveAPISession  MapService static
 (below)         server/map_service.py


┌─────────────────────────────────────────────────────────────────────┐
│  Scan Server  ·  scan_server/scan_server.py  ·  port 7861            │
│  OFFLINE MODE — operated by our team before user deployment          │
│  FastAPI + Gradio UI; workflow:                                       │
│  1. Record images/ + imu.csv + camera.csv on Android (ScanScreen) →  │
│     zip as dataset.zip → POST /api/upload (extracted server-side)   │
│  2. Load from Android Upload accordion → pre-fills dataset folder    │
│     path  OR  type/paste a dataset folder path directly in the UI    │
│  3. Fill Segment Table: each row = (start_s, end_s, zone_name)       │
│  4. Click Simulated Live Stream (auto, whole dataset) or step         │
│     through Manual Live Stream (one "Feed Next Frame" click per      │
│     frame, with a preview of the frame about to be fed) — both       │
│     drive the same StreamingScanSession, pose from one of 2          │
│     selectable sources (IMU + VO or RTAB-Map — see 3D Scanning       │
│     Pipeline); dense Plane Sweep MVS depth map per                   │
│     keyframe → back-project to 3D point cloud. There is no batch     │
│     "Scan" button anymore — every run replays frame-by-frame.        │
│  5. Click Export Map (or let a stream run to completion, which       │
│     auto-exports) — writes PLY + JSON + keyframes/index.json         │
│     → served by Main Server MapService at runtime                    │
│  • Gradio UI port 7861  •  REST /api/upload (multipart dataset.zip)  │
│  • Optional: scan_server/rtabmap_docker/ — RTAB-Map RGB-D pose       │
│    service (plain ZeroMQ socket, no ROS, port 5556), robust pose     │
│    source (RGB-D odometry + loop closure, no per-batch anchoring,    │
│    no camera-IMU calibration needed at all — replaces ORB-SLAM3,     │
│    whose calibration proved unreliable); depth from DA3-ONNX either  │
│    way, no IMU used anywhere in this path                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## gRPC Services & Protobuf

**Single source of truth:** `tracking.proto`  
Generated stubs are copied to `server/`, `client/proto/`, `test_module/`.

> **After editing `tracking.proto` regenerate stubs:**
> ```bash
> python -m grpc_tools.protoc -I. \
>   --python_out=<dir> --grpc_python_out=<dir> tracking.proto
> ```
> Run for each directory that holds a copy.

### TrackingService (port 50051)

| RPC | Input | Output | What it does |
|-----|-------|--------|--------------|
| `DetectObject` | prompt string | box_xyxy + score | GroundingDINO detection |
| `GetEmbedding` | box_xyxy | float vector | DINOv2 ViT-S/14 embedding for re-ID |
| `Chat` | text message | text response | One-shot Gemini generate (not Live); simple text stub |
| `VoiceChat` | raw audio bytes | text response | Stub — redirects user to VoiceChatStream |
| `StreamFrame` | JPEG bytes | success | Stores latest frame for DetectObject/GetEmbedding; no ticks |
| `VoiceChatStream` | stream VoiceChatChunk (audio+frames+tracking_data) | stream AudioChunk (raw PCM) | Creates LiveAPISession; forwards mic audio to Gemini Live; frame stored server-side; `VoiceChatChunk.tracking_data` (object+hand boxes from Android's on-device tracker, sent while mode=tracking) updates `state.last_detection`/`state.last_hand_box` live; Gemini decides tool calls; PCM audio streamed back (either Gemini's voice or local TTS via `read_aloud`) |

### MediatorService (port 50052 on mediator host)

| RPC | What it does |
|-----|-------------|
| `StreamFrameWithGuidance` | Local ORB tracking + forward to main server; returns spatial guidance |
| `Chat` | Relay to main server |
| `VoiceChat` | Relay to main server |

### MapService (main server only, port 50051)

| RPC | What it does |
|-----|-------------|
| `ListMaps` | Return known location IDs from `server/data/maps/` |
| `GetMapGeometry` | Stream PLY file in 64 KB chunks |

The scan server has **no gRPC**. Map creation is done entirely in-process via `scan_session.py` + Gradio UI.

---

## Live Session / Tool System

**Entry:** `server/live_session.py` — `LiveAPISession` (one per user `VoiceChatStream` call)  
**Tool declarations:** `server/live_tools/tool_declarations.py` — all `FunctionDeclaration` dicts + `SYSTEM_PROMPT`  
**Device tool declarations:** `server/live_tools/device_tools.py` — `DEVICE_TOOL_DECLARATIONS` + `DEVICE_TOOL_NAMES`; merged into session config only for tools advertised by the client's `capabilities` field  
**Tool implementations:** `server/live_tools/` — dispatched when Gemini makes function calls  
**Device tool routing:** Gemini calls a device tool → `_dispatch_tool` queues it in `_device_tool_q` → servicer yields `AudioChunk(tool_call=...)` to Android → Android executes and sends `VoiceChatChunk(tool_result=...)` → servicer calls `receive_tool_result_sync` → Future resolved → result sent back to Gemini

### Architecture

```
Android mic audio + JPEG frames
        │
  VoiceChatStream RPC (gRPC)
        │
  LiveAPISession
   ├── send_audio_sync(pcm)  ──►  Gemini Live WebSocket  ──►  PCM audio out
   ├── receive_frame_sync(jpeg) → stored as latest_frame + background ticks
   │       • OCR tick (1.5 s, if mode=reading)
   │       • Depth tick (0.5 s, if mode=navigation) → [SYSTEM] obstacle warning
   │       • Localize tick (2.0 s, if mode=navigation) → proximity check
   └── _dispatch_tool(name, args) → live_tools/*.py → local models
```

### Tools (Gemini calls these as function calls)

| Tool | File | What it does |
|------|------|-------------|
| `make_phone_call(contact_name_or_number)` | device_tools (client) | Routed to Android; fires `Intent.ACTION_CALL` |
| `set_alarm(time, label?)` | device_tools (client) | Routed to Android; fires `AlarmClock.ACTION_SET_ALARM` |
| `create_calendar_event(title, start_time, end_time?, description?)` | device_tools (client) | Routed to Android; fires `Intent.ACTION_INSERT` on CalendarContract |
| `get_latest_frame()` | scene_tools | Send stored JPEG to Gemini via `send_realtime_input(video=...)` |
| `start_vision_stream(reason?)` | scene_tools | 1 fps frame stream to Gemini, auto-stops after 15 s |
| `stop_vision_stream()` | scene_tools | Cancel vision stream |
| `run_detection(desc)` | scene_tools | GroundingDINO on latest_frame |
| `check_obstacle()` | scene_tools | depth_detector on latest_frame |
| `enter_reading_mode(label?)` | reading_tools | Set mode=reading, reset buffer; passive OCR accumulation begins |
| `scan_current_view()` | reading_tools | OCR on latest_frame → dedup-append to reading_buffer (silent — not read aloud) |
| `get_reading_section(query)` | reading_tools | Keyword/semantic search over reading_buffer (never feeds full buffer to Gemini) |
| `read_aloud(scope)` | reading_tools | scope=new: scan+speak new text; scope=all: speak full reading_buffer. Uses local KokoroTTS (`tools/tts.py`), streamed straight to `_output_q` — bypasses Gemini Live's voice entirely |
| `flip_reading_direction()` | reading_tools | Toggle ltr/rtl |
| `exit_reading_mode()` | reading_tools | Clear reading state |
| `start_tracking(target)` | tracking_tools | GroundingDINO detect, set mode=tracking |
| `stop_tracking()` | tracking_tools | Clear tracking state |
| `get_object_from_memory(query)` | tracking_tools | rag_store semantic search, threshold 0.5 |
| `query_memory(question)` | memory_tools | rag_store semantic search over all labels, threshold 0.5 |
| `save_memory(label, note)` | memory_tools | memory_store.append + rag_store.add_text |
| `remember_object(label)` | memory_tools | Detect crop + rag_store.add_object |
| `list_memory_labels()` | memory_tools | List known memory labels |
| `start_guiding(dest)` | navigation_tools | Load map; resolves dest via zone label first, then landmark name (find_landmark) + A* (GridPathPlanner) with a fallback to the legacy zone-centroid route; set mode=guiding; route/waypoints injected into response |
| `stop_guiding()` | navigation_tools | Clear guiding state |
| `get_current_location()` | navigation_tools | PnP localize against map keyframes |
| `start_walking()` | walking_tools | Set mode=walking; DINO+DA3 ONNX ticks begin at WalkingConfig.detection_interval |
| `stop_walking()` | walking_tools | Clear walking state |
| `quick_label_obstacle(label)` | walking_tools | Store label in walking_obstacle_cache with 6 s TTL; included in next DINO prompt |

### LiveSessionState

Per-user state held in `LiveAPISession.state` (not in Gemini context window):
- `mode`: idle | reading | tracking | guiding
- `reading_buffer`: full OCR text (server-side only; accessed via `get_reading_section`/`read_aloud`)
- `page_summaries`: brief summaries per scanned page (returned by `scan_current_view`)
- `last_detection`, `last_hand_box`: live per-frame object/hand boxes in tracking mode, updated
  from the client's `VoiceChatChunk.tracking_data` (Android's on-device ORB tracker + MediaPipe
  hand detector) — also drives the server GUI's Tracking tab
- `tracking_guidance_active`, `tracking_last_guidance_at`: gate the Gemini hand-guidance tick
  (fires once when object+hand are first both visible, then every 5 s while both remain visible)
- `nav_route`, `nav_route_idx`, `nav_last_position`: guiding mode progress
- `walking_obstacle_cache`: list of `{label, expires_at}` with 6 s TTL; used to suppress duplicate obstacle alerts
- `live_vision_active`: whether 1 fps frame stream is running

### Navigation / Walking Modes

| Mode | Entry point | Who operates it | What it does |
|------|------------|-----------------|--------------|
| **Offline (Scanning)** | `scan_server/` | Our team, pre-deployment | ORB + VIO/GTSAM → dense point cloud via Plane Sweep MVS per keyframe; label zones; Android ScanScreen records images/ + camera.csv + imu.csv (dataset/ folder) for upload |
| **Online Guiding** | `server/` — LiveAPISession + `start_guiding` tool | End user | Gemini calls `start_guiding(dest)` → route loaded; depth/localize ticks inject `[SYSTEM]` messages to Gemini → Gemini warns user via audio |
| **Online Walking** | `server/` — LiveAPISession + `start_walking` tool | End user | `start_walking()` = guiding mode with no destination; same DINO+DA3 ONNX obstacle detection, no localization tick |

---

## 3D Scanning Pipeline

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
stay per-node (batching those would reproduce the sunburst artifact).

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
Rendered in classic SLAM grayscale (`render_plotly`): white=free/ground,
light gray=low/step-over, black=obstacle, mid-gray=unknown (not enough
agreeing evidence either way) — not a continuous height-gradient heatmap.
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

---

## Client Implementations

| Client | File | Hardware | Role |
|--------|------|----------|------|
| **Pi thin client** | `client/pi_client.py` | Raspberry Pi | Camera capture → Mediator gRPC stream; callback for buzzer/LED |
| **Mediator** | `client/mediator_gui.py` | any host near Pi | Edge proxy; local ORB + Homography; throttled forwarding; Gradio UI port 7862 |
| **Desktop video GUI** | `client/desktop_video_stream_gui.py` | desktop | Operator dashboard; video + overlays + chat |
| **Edge main** | `client/edge_main.py` | desktop/edge | Video file replay for testing |
| **gRPC wrapper** | `client/rpc_client/grpc_client.py` | — | `RemoteTrackingClient` — shared by all Python clients |

---

## Key Files Map

```
tracking.proto                   gRPC + protobuf definitions (edit here, regenerate stubs)
tracking_pb2{,_grpc}.py          Generated — DO NOT EDIT (exists in server/, client/proto/, test_module/)

server/
  grpc_server.py                 Main server entry point; loads models; wires ToolsBundle; starts gRPC + Gradio
  services/servicer.py           TrackingServiceServicer — gRPC RPCs; VoiceChatStream creates LiveAPISession
  map_service.py                 MapServiceServicer — static map retrieval
  live_session.py                LiveAPISession + ToolsBundle + LiveSessionState
                                   One session per VoiceChatStream call; manages Gemini Live WebSocket.
                                   LiveSessionState.nav_waypoints/nav_waypoint_idx (new) track progress along
                                   an A*-planned (x,z) route in parallel to the legacy label-based nav_route —
                                   mutually exclusive, set by navigation_tools.tool_start_navigation.
                                   _check_proximity() branches on which is populated: waypoints use fixed
                                   arrival radii (_WAYPOINT_ARRIVAL_RADIUS/_LANDMARK_ARRIVAL_RADIUS, no AABB),
                                   nav_route keeps the original zone-AABB-contains() logic
  live_tools/
    tool_declarations.py         All FunctionDeclaration dicts + SYSTEM_PROMPT
    device_tools.py              DEVICE_TOOL_DECLARATIONS + DEVICE_TOOL_NAMES (phone/alarm/calendar)
    scene_tools.py               get_latest_frame, start/stop_vision_stream, run_detection, check_obstacle
    reading_tools.py             enter/exit_reading_mode, scan_current_view, get_reading_section, flip_reading_direction
    tracking_tools.py            start_tracking, stop_tracking, get_object_from_memory
    memory_tools.py              query_memory, save_memory, remember_object, list_memory_labels
    navigation_tools.py          start_guiding, stop_guiding, get_current_location (mode=guiding).
                                   _ensure_map_loaded also builds/caches a GridPathPlanner (tools/
                                   grid_path_planner.py) alongside RoutePlanner/LocalizationEngine — None on
                                   maps exported before the height-tiered A* planner existed
    walking_tools.py             start_walking, stop_walking, quick_label_obstacle (mode=walking)
  ARCHITECTURE.md                Detailed server internals (component map, data flows, tool→function map)
  domain/types.py                MemoryDocument + MemoryEntry dataclasses (used by memory_store)
  tools/
    detector.py                  GroundingDINO wrapper
    ocr.py                       Remote OCR client
    depth.py                     Obstacle detectors: SparseObstacleDetector (ORB, relative depth, default)
                                   + StereoDepthDetector (plane sweep MVS, metric metres; set DEPTH_MODEL=stereo)
    memory_store.py              JSON per-label memory; filter_new_sentences()
    rag_store.py                 Sentence-transformer + CLIP embeddings; query_global()
    localization.py              LocalizationEngine — PnP against map keyframes
    route_planner.py             RoutePlanner — zone-based path planning + find_landmark() (parses
                                   zones[].landmarks[] from map_labels.json, previously dropped, into a
                                   name -> (Zone, x, z) lookup — exact/case-insensitive then substring match)
    grid_path_planner.py         GridPathPlanner — A* over the height-tiered whole-map occupancy_grid
                                   (map_labels.json's top-level field, occupancy_map.extract_full_grid()) for
                                   landmark destinations with no zone AABB to route between. Cost model: ground
                                   cheap, low/step-over costlier-but-passable, normal-obstacle blocked, unknown
                                   passable-at-a-premium (not blocked — large legitimately-unscanned patches
                                   would otherwise fragment the map). 8-connected, octile heuristic, corner-
                                   cutting prevented; simplifies the raw path via cost-aware line-of-sight
                                   "string pulling" (checks the shortcut isn't costlier than the original route,
                                   not just that it's unblocked) into a small waypoint list
    embedder.py                  DINOv2Embedder (ViT-S/14) — visual re-ID embeddings
    tts.py                       KokoroTTS.synthesize_pcm_chunks() — local 24 kHz PCM TTS used by
                                   reading_tools.read_aloud() to voice scanned text without Gemini Live
    asr.py                       WhisperASR — kept for non-Live stubs (optional)
  _archived/                     Old orchestrator/, agents/, cloud_vlm, intent_parser (reference only)
  data/
    memory/                      {label}.json memory files
    maps/{location_id}/          map_geometry.ply + map_labels.json

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
                                   new points get coarsened to before feeding the Occupancy Map
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
                                   process_frames_batch() picks one of 2 pose sources (IMU + VO or
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
  stream_session.py              StreamingScanSession — push-based incremental wrapper around ScanSession,
                                   the ONLY way scan_gui.py drives a scan now (no batch path exists anymore):
                                   push_frame()/push_imu()/start_zone()/end_zone()/finish(). Buffers frames
                                   into mini_batch-sized chunks and calls process_frames_batch() once a
                                   chunk is ready. start_zone()/end_zone() are the live replacement for a
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
                                   used as scan_session's default estimator (dense depth for every pose source)
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
                                   "Progressive, Bayesian, SLAM-style Occupancy Map" note. update()
                                   classifies every POINT individually by height above ground (not a
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
                                   entire bed eroding to free/unknown). render_plotly() uses a
                                   classic SLAM grayscale colorscale (white/light-gray/black/
                                   mid-gray), not a continuous height-gradient heatmap. Every
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
  semantic_mapper.py             SemanticMapper — multi-image VLM + GroundingDINO landmark extraction;
                                   buffers IMAGES_PER_PROMPT=5 sampled frames per VLM call (shared
                                   grounding_dino_prompt, per-frame GroundingDINO+backprojection);
                                   Landmark dataclass; cluster_landmarks(); flush() for a partial batch
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
                                   become a node) per frame — loop_closure signals that THIS frame's
                                   processing closed a loop, so previously-pulled get_cloud() nodes'
                                   poses may have just shifted; node_id lets scan_session.py correlate
                                   its own depth-consistency check with the specific RTAB-Map node this
                                   frame produced (see 3D Scanning Pipeline's "Depth-consistency gate,
                                   extended to RTAB-Map pose mode") — reply parsing degrades gracefully
                                   (node_id=-1) against an older server build without this field.
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
                                                           RTAB-Map pose mode")
                                   entrypoint.sh            execs rtabmap_server with just a bind address — no
                                                            required config file (intrinsics sent per-frame)
                                   README.md                build/run instructions + corelib-from-source fallback +
                                                            full wire protocol (TRACK/RESET/PING/GET_CLOUD)

server/
  vio → ../scan_server/vio       Symlink so server/ can import the same VIO module

client/
  pi_client.py                   Pi thin client
  mediator_gui.py                Mediator service + Gradio UI
  desktop_video_stream_gui.py    Operator desktop GUI
  edge_main.py                   Video file test client
  rpc_client/grpc_client.py      RemoteTrackingClient wrapper
  core/
    local_models.py              LocalHandDetector + GPUVIOAnchorBackend (ORB tracking)
    guidance_engine.py           Spatial guidance output (instruction, delta_x/y, distance)
  android/                         Android Jetpack Compose client (Kotlin); Gradle root: client/android/
    audio/
      PushToTalkRecorder.kt        PTT recording; onChunkReady emits raw PCM chunks during hold
      StreamingAudioPlayer.kt      Incremental raw PCM playback (24 kHz) for VoiceChatStream response
      TtsPlayer.kt                 WAV playback (24 kHz) for unary VoiceChat / StreamFrame audio
    camera/
      CameraManager.kt             CameraX ImageAnalysis — live JPEG stream (frameFlow) AND, while recording,
                                     periodic frame dump to images/*.jpg + camera.csv (timestamp_ns,filename)
                                     startRecording(outputDir, fps) / stopRecording(): Int (frame count)
                                     frame timestamp = imageInfo.timestamp (boot-time ns, same clock as ImuSensor)
    sensors/
      ImuSensor.kt                 SensorManager wrapper; emits ImuReading Flow at SENSOR_DELAY_FASTEST
      ImuRecorder.kt               Writes imu.csv (header: timestamp_ns,ax,ay,az,gx,gy,gz) alongside images/
    device/
      DeviceToolHandler.kt         Interface for executing device-native tool calls (phone/alarm/calendar)
      AndroidDeviceToolHandler.kt  Implementation: Intent.ACTION_CALL, AlarmClock, CalendarContract
    ui/
      MainViewModel.kt             doVoiceChatStream() — merges audio+frame+IMU Flow → VoiceChatStream RPC
                                     Sends capabilities handshake on first chunk; handles AudioChunk.tool_call
                                     by delegating to DeviceToolHandler and returning VoiceChatChunk.tool_result
      ScanViewModel.kt             Live gRPC scan stream + offline recording (startRecording/stopRecording);
                                     uploadToScanServer() zips the dataset dir (images/+imu.csv+camera.csv)
                                     and POSTs it as dataset.zip to /api/upload
      ScanScreen.kt                3D Scan UI — Record section (images+IMU to dataset folder, upload) + Live Stream section
      ScanUiState.kt               State for both live-stream and offline recording modes (datasetDir, imageCount, imuFile)

scan_app/
  scan_client.py                 Thin client for the scan server

paddle_ocr_server/               Standalone OCR microservice (port 8100)

test_module/
  mock_frame_server.py           MockMapServicer for unit tests

depth-anything-3/                DA3 model package (installed locally)
```

---

## Data Flows

### Real-time object tracking (Pi → voice guidance)
```
Pi camera → mediator_gui.py (ORB track, hand detect)
  → if 1.5 s elapsed: Main Server DetectObject + GetEmbedding
  → Guidance engine: delta_x, delta_y, distance, instruction
  → audio callback on Pi (buzzer / speaker)
```

### Online navigation (user walking through a mapped venue)
```
User says "navigate to kitchen"
  → Gemini Live detects intent → calls start_navigation("kitchen")
  → route computed, injected into Gemini context as function response
  → Gemini announces route in audio

Each received JPEG frame triggers background ticks:
  depth tick (0.5 s): SparseObstacleDetector → if obstacle:
    inject "[SYSTEM] Obstacle ~Xm ahead. Warn user immediately."
    → Gemini responds in audio (15 s cooldown)
  localize tick (2.0 s): PnP → check proximity → if at waypoint/destination:
    inject "[SYSTEM] Passed X, now heading to Y."
    → Gemini announces in audio
```

### Reading a document aloud
```
User says "read this" or "enter reading mode"
  → Gemini Live calls enter_reading_mode()
  → User points camera at text; OCR tick (1.5 s) accumulates text in reading_buffer
  → User says "read this" / "scan and read" → Gemini calls read_aloud(scope="new")
    → OCR on latest_frame → new text → KokoroTTS synthesizes 24 kHz PCM
    → streamed directly into _output_q → client hears it — Gemini's own voice is NOT used
  → User says "read all of it" → Gemini calls read_aloud(scope="all")
    → entire reading_buffer synthesized and streamed the same way
  → User asks a question about content → Gemini calls get_reading_section(query)
    → keyword search over reading_buffer → returns relevant passage
    → Gemini answers from passage in its own voice (only path that still uses Gemini's TTS)
```

### Environment scanning (offline, team-operated)
```
Android ScanScreen (operator walks through venue):
  → tap Record → CameraManager.startRecording() writes images/*.jpg + camera.csv
                  ImuRecorder.start() writes imu.csv (100 Hz)
  → tap Stop  → tap "Upload Files" → zips dataset dir → HTTP multipart
                                      POST /api/upload (dataset.zip) → extracted to
                                      scan_server/uploads/<scan_id>/dataset/

Scan server Gradio UI (port 7861):
  → "Load from Android Upload" dropdown → pre-fills the dataset folder path
     OR type/paste a dataset folder path manually
  → Segment Table: [(start_s, end_s, zone_name), ...]
  → Click "Simulated Live Stream" (auto, whole dataset) OR click "Start /
    Reset Manual Stream" then "Feed Next Frame" repeatedly (one frame per
    click, with a preview of the frame about to be fed) — both replay the
    dataset through the same StreamingScanSession: ORB VO + IMU gyro-integrated
    rotation, or RTAB-Map RGB-D odometry → triangulation point cloud
  → zone AABB = camera path AABB + margin per segment (start_zone/end_zone
    fired at Segment Table boundaries crossed during replay)
  → Export Map (or let the stream finish, which auto-exports) → PLY + JSON
    saved to maps/{id}/
  → Main Server MapService serves static map to any client
```

---

## Deployment

| Component | Default Port | Command |
|-----------|-------------|---------|
| OCR server | 8100 | `cd paddle_ocr_server && uvicorn server:app --host 0.0.0.0 --port 8100` |
| Main server | 50051 + Gradio 7860 | `python server/grpc_server.py` |
| Mediator | 50052 + Gradio 7862 | `python client/mediator_gui.py` |
| Scan server | 7861 (FastAPI+Gradio) | `python scan_server/scan_server.py [--da3-model torch\|onnx] [--da3-onnx-path PATH]` |
| RTAB-Map pose service (optional) | 5556 (ZeroMQ, no ROS) | `docker compose up rtabmap` — see `scan_server/rtabmap_docker/README.md` |

Docker: `docker-compose up` (requires NVIDIA runtime; mounts model volume). The
`rtabmap` service needs no per-device calibration (unlike the old `orbslam3`
service it replaced) but still isn't brought up automatically by a bare
`docker-compose up`-everything workflow — see
`scan_server/rtabmap_docker/README.md` to build it.

### Development Environments

| Component | Python env | Notes |
|-----------|-----------|-------|
| Server / Mediator | `server/.venv/` | activate: `source server/.venv/bin/activate` or prefix commands with `server/.venv/bin/python` |
| Scan server | conda env `hrtf` | activate: `conda activate hrtf`. `server/.venv/` also works (kept in sync) but `hrtf` is the one actually used for running/testing scan_server |
| Android client | — | Gradle project root: `client/android/`; run `./gradlew build` from there |

Environment variables:
- `GEMINI_API_KEY` — **required** — API key for Gemini Live API (used by `LiveAPISession`)
- `OCR_SERVER_URL` — default `http://localhost:8100`
- `DEPTH_MODEL` — `sparse` (default, ORB relative depth), `stereo` (plane sweep MVS), or `da3` (Depth Anything 3 + VIO scale alignment)
- `DA3_MODEL_ID` — DA3 model name (default `depth-anything/da3-large`; also `da3-giant`, `da3metric-large`)
- `DA3_ONNX_PATH` — path to DA3METRIC ONNX file for walking mode (default `../DA3METRIC-LARGE.onnx`); walking mode is silently disabled if file not found
- `SCAN_GRADIO_PORT` — scan server Gradio port (default 7861)
- `SCAN_DEVICE` — `cpu` or `cuda` for scan server's DA3 estimator (default `cuda`)
- `SCAN_DA3_TORCH_MODEL_ID` — DA3 torch model for scan server (default `depth-anything/da3-large`), used for depth estimation regardless of pose source (only applies when `--da3-model torch`, the default)

Scan server CLI options (`python scan_server/scan_server.py --help`):
- `--da3-model {torch,onnx}` — main dense-depth estimator backend (default `torch`), used for depth estimation regardless of pose source (IMU + VO or RTAB-Map). `torch` loads a `DA3Estimator`; `onnx` loads a `DA3OnnxEstimator` (DA3-METRIC ONNX, metric depth, lighter/faster).
- `--da3-onnx-path PATH` — path to the DA3-METRIC ONNX file (only used with `--da3-model onnx`; default `../DA3METRIC-LARGE.onnx` relative to `scan_server/`)
- `SCAN_GEMMA_MODEL_ID` — Gemma model for scan server semantic mapping via Gemini API (default `gemma-4-31b-it`); uses `GEMINI_API_KEY` above
- `RTABMAP_ADDR` — ZeroMQ address for the RTAB-Map pose service, e.g. `tcp://localhost:5556`; when unset, "RTAB-Map" pose source is disabled and the Scan UI falls back to IMU + VO
- `OPENROUTER_API_KEY` — unused by the current scan server; kept for reference (was used by the old pre-Qwen `OpenRouterVLMClient`, see `server/_archived/tools/cloud_vlm.py`)
- `YTDLP_COOKIES_FILE` — path to a Netscape-format cookies file passed as `--cookies` to all `yt-dlp` calls in `live_tools/music_tools.py`; needed when YouTube blocks a request with "Sign in to confirm you're not a bot" (export cookies from a logged-in browser session). When unset, `yt-dlp` runs unauthenticated as before.

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
