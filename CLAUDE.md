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
Note that python env is at: server/.venv/

```
┌─────────────────────────────────────────────────────────────────────┐
│  Camera source (Pi / Android / webcam)                              │
│  + IMU (Android SensorManager: accel + gyro via ImuSensor.kt)       │
│    • Offline scan: ImuRecorder.kt writes imu_data.csv alongside video│
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
│  1. Record video + IMU on Android (ScanScreen) → upload to           │
│     POST /api/upload  OR  drag-drop in Gradio UI                     │
│  2. Load from Android Upload accordion → pre-fills video + IMU       │
│  3. Fill Segment Table: each row = (start_s, end_s, zone_name)       │
│  4. Click Scan — ORB + VIO/GTSAM per segment; dense Plane Sweep MVS  │
│     depth map per keyframe → back-project to 3D point cloud          │
│  5. Click Export Map — writes PLY + JSON + keyframes/index.json      │
│     → served by Main Server MapService at runtime                    │
│  • Gradio UI port 7861  •  REST /api/upload (multipart video+imu)    │
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
| `start_guiding(dest)` | navigation_tools | Load map, compute route, set mode=guiding; route injected into response |
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
| **Offline (Scanning)** | `scan_server/` | Our team, pre-deployment | ORB + VIO/GTSAM → dense point cloud via Plane Sweep MVS per keyframe; label zones; Android ScanScreen records video + IMU CSV for upload |
| **Online Guiding** | `server/` — LiveAPISession + `start_guiding` tool | End user | Gemini calls `start_guiding(dest)` → route loaded; depth/localize ticks inject `[SYSTEM]` messages to Gemini → Gemini warns user via audio |
| **Online Walking** | `server/` — LiveAPISession + `start_walking` tool | End user | `start_walking()` = guiding mode with no destination; same DINO+DA3 ONNX obstacle detection, no localization tick |

---

## 3D Scanning Pipeline

```
Android ScanScreen records video.mp4 + imu_data.csv simultaneously
  → POST /api/upload to scan server  OR  drag-drop in Gradio UI
      │
Video upload + imu_data.csv (optional)
      │
      ├─ imu_data.csv → scan_session.add_imu_file()
      │    → IMUPreintegrator: gravity/bias init from stationary period
      │
      ▼
feature_tracker.py — FeatureTracker
  • ORB + PnP → (world_pose, rel_pose) per frame
  • rel_pose fed as visual factor to VIO estimator
      │
      ▼
vio/ — VIOEstimator (GTSAM iSAM2)
  • CombinedImuFactor: IMU pre-integration between keyframes
  • BetweenFactor<Pose3>: visual VO constraint
  • Returns smoothed keyframe poses with metric scale from IMU
      │
      ▼
mvs.py — DA3 + Sparse Alignment (dense depth per keyframe)
  • ORB triangulation between consecutive keyframes → sparse metric 3D anchors
  • DA3Estimator.estimate(rgb) → dense relative depth (Depth Anything 3)
  • align_metric_depth(): RANSAC linear fit d_metric = s·d_relative + t
  • backproject_depth(): X=(u-cx)d/fx, Y=(v-cy)d/fy, Z=d → world via VIO pose
  • First keyframe (no prior): skipped — DA3 starts at second keyframe
      │
      ▼
point_cloud_fusion (in scan_session.py)
  • Accumulate back-projected dense points per keyframe
  • Voxel downsample 0.02 m
      │
      ▼
zone_labeler.py  (called from scan_session after each segment)
  • Named AABB from camera path in that segment (set_label_from_positions)
      │
      ▼
map_exporter.py
  • maps/{location_id}/map_geometry.ply  (binary Open3D PLY)
  • maps/{location_id}/map_labels.json   (metadata + zones[])
```

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
                                   One session per VoiceChatStream call; manages Gemini Live WebSocket
  live_tools/
    tool_declarations.py         All FunctionDeclaration dicts + SYSTEM_PROMPT
    device_tools.py              DEVICE_TOOL_DECLARATIONS + DEVICE_TOOL_NAMES (phone/alarm/calendar)
    scene_tools.py               get_latest_frame, start/stop_vision_stream, run_detection, check_obstacle
    reading_tools.py             enter/exit_reading_mode, scan_current_view, get_reading_section, flip_reading_direction
    tracking_tools.py            start_tracking, stop_tracking, get_object_from_memory
    memory_tools.py              query_memory, save_memory, remember_object, list_memory_labels
    navigation_tools.py          start_guiding, stop_guiding, get_current_location (mode=guiding)
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
    route_planner.py             RoutePlanner — zone-based path planning
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
                                   POST /api/upload — receives video + optional imu from Android
                                   GET  /api/uploads — lists available upload IDs
                                   uploads/<scan_id>/ — saved files from Android uploads
  scan_gui.py                    Gradio UI — video + imu_data.csv upload + segment table → export
                                   create_scan_ui(scan_manager, upload_dir) — "Load from Android Upload" accordion
  scan_css.py                    CSS + theme + header/description HTML for Gradio UI
  scan_session.py                ScanSession + ScanSessionManager (per-location state)
                                   add_imu_file() loads IMU CSV; set_imu_distribution(n) distributes IMU evenly;
                                   process_frames_batch() routes to VIO; keyframe MVS dense depth accumulation
  mvs.py                         Depth utilities: align_metric_depth() (RANSAC DA3 scale alignment),
                                   backproject_depth(), classify_traversable(), plane_sweep_stereo() (kept)
  feature_tracker.py             ORB + PnP pose estimator; returns (world_pose, rel_pose) tuple
  da3_wrapper.py                 Unused — kept for reference (DA3/DepthAnythingV2 removed from pipeline)
  vio/                           Visual-Inertial Odometry module (GTSAM iSAM2)
    __init__.py                  Exports IMUPreintegrator, VIOEstimator
    defaults.py                  Sensor noise model constants (accel/gyro sigmas)
    imu_preintegrator.py         GTSAM PreintegratedCombinedMeasurements; gravity/bias init
    vio_estimator.py             GTSAM ISAM2 + CombinedImuFactor + BetweenFactor<Pose3>
  zone_labeler.py                Zone AABB management (Zone dataclass has landmarks: List[Landmark])
  map_exporter.py                PLY + JSON export; map_labels.json includes zone_type, landmarks[], occupancy_grid per area
  semantic_mapper.py             SemanticMapper — VLM (OpenRouter) + GroundingDINO landmark extraction;
                                   Landmark dataclass; cluster_landmarks(); only active when OPENROUTER_API_KEY is set

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
      CameraManager.kt             CameraX ImageAnalysis (JPEG stream) + VideoCapture (MP4 file recording)
                                     startRecording(outputFile, onFinalized) / stopRecording()
    sensors/
      ImuSensor.kt                 SensorManager wrapper; emits ImuReading Flow at SENSOR_DELAY_FASTEST
      ImuRecorder.kt               Writes imu_data.csv alongside video for offline scan upload
    device/
      DeviceToolHandler.kt         Interface for executing device-native tool calls (phone/alarm/calendar)
      AndroidDeviceToolHandler.kt  Implementation: Intent.ACTION_CALL, AlarmClock, CalendarContract
    ui/
      MainViewModel.kt             doVoiceChatStream() — merges audio+frame+IMU Flow → VoiceChatStream RPC
                                     Sends capabilities handshake on first chunk; handles AudioChunk.tool_call
                                     by delegating to DeviceToolHandler and returning VoiceChatChunk.tool_result
      ScanViewModel.kt             Live gRPC scan stream + offline recording (startRecording/stopRecording/uploadToScanServer)
      ScanScreen.kt                3D Scan UI — Record section (video+IMU to file, upload) + Live Stream section
      ScanUiState.kt               State for both live-stream and offline recording modes

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
  → tap Record → CameraManager.startRecording() writes video.mp4
                  ImuRecorder.start() writes imu_data.csv (100 Hz)
  → tap Stop  → tap "Upload Files" → HTTP multipart POST /api/upload
                                      saves to scan_server/uploads/<scan_id>/

Scan server Gradio UI (port 7861):
  → "Load from Android Upload" dropdown → pre-fills video + IMU inputs
     OR drag-drop video + imu_data.csv manually
  → Segment Table: [(start_s, end_s, zone_name), ...]
  → Click Scan: ORB VO + VIO/GTSAM refinement → triangulation point cloud
  → zone AABB = camera path AABB + margin per segment
  → Export Map → PLY + JSON saved to maps/{id}/
  → Main Server MapService serves static map to any client
```

---

## Deployment

| Component | Default Port | Command |
|-----------|-------------|---------|
| OCR server | 8100 | `cd paddle_ocr_server && python server.py` |
| Main server | 50051 + Gradio 7860 | `python server/grpc_server.py` |
| Mediator | 50052 + Gradio 7862 | `python client/mediator_gui.py` |
| Scan server | 7861 (FastAPI+Gradio) | `python scan_server/scan_server.py` |

Docker: `docker-compose up` (requires NVIDIA runtime; mounts model volume).

### Development Environments

| Component | Python env | Notes |
|-----------|-----------|-------|
| Server / Mediator / Scan server | `server/.venv/` | activate: `source server/.venv/bin/activate` or prefix commands with `server/.venv/bin/python` |
| Android client | — | Gradle project root: `client/android/`; run `./gradlew build` from there |

Environment variables:
- `GEMINI_API_KEY` — **required** — API key for Gemini Live API (used by `LiveAPISession`)
- `OCR_SERVER_URL` — default `http://localhost:8100`
- `DEPTH_MODEL` — `sparse` (default, ORB relative depth), `stereo` (plane sweep MVS), or `da3` (Depth Anything 3 + VIO scale alignment)
- `DA3_MODEL_ID` — DA3 model name (default `depth-anything/da3-large`; also `da3-giant`, `da3metric-large`)
- `DA3_ONNX_PATH` — path to DA3METRIC ONNX file for walking mode (default `../DA3METRIC-LARGE.onnx`); walking mode is silently disabled if file not found
- `SCAN_GRADIO_PORT` — scan server Gradio port (default 7861)
- `SCAN_DEVICE` — `cpu` or `cuda` for scan server (unused since DA3 removal)
- `OPENROUTER_API_KEY` — API key for OpenRouter (scan server offline semantic mapping; when absent, semantic mapping is silently skipped)

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
