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
│  • EfficientNetLite — re-ID embeddings (cosine ≥ 0.75 = same target)│
│  • Whisper ASR     — speech-to-text                                  │
│  • Kokoro TTS      — text-to-speech                                  │
│  • DocLayoutRapidOCR (remote, paddle_ocr_server port 8100)           │
│  Cloud VLM (server/tools/cloud_vlm.py) — obstacle ID + scene Q&A   │
│  • GeminiVLMClient — Gemini Flash with streaming AUDIO output        │
│  Gradio monitor dashboard (port 7860)                                │
└───────────┬─────────────────────────────────────────────────────────┘
            │
      ┌─────┴─────┐
      │           │
      ▼           ▼
 Agent system   MapService static
 (below)        server/map_service.py


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
| `GetEmbedding` | box_xyxy | float vector | EfficientNetLite embedding for re-ID |
| `Chat` | text message | text + audio + agent metadata | Text chat → Orchestrator → TTS reply |
| `VoiceChat` | raw audio bytes | text + audio + agent metadata | Whisper ASR then same as Chat |
| `StreamFrame` | JPEG bytes | success + optional audio | Real-time frame ingestion; VLM obstacle alerts |
| `VoiceChatStream` | stream VoiceChatChunk (audio+frames+IMUFrame) | stream AudioChunk (raw PCM) | PTT streaming: accumulate audio+frames+IMU while button held → ASR → intent → INFO: Gemini Flash audio stream; other: Kokoro TTS; IMU frames feed VIOEstimator → `ctx.current_pose` |

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

## Agent System

**Orchestrator:** `server/orchestrator/orchestrator.py`  
**Router:** `server/orchestrator/router.py` — picks agent based on detected intent  
**Session:** `server/orchestrator/session.py` — per-user state (active_agent, scan_buffer, reading_state, …)

### Intent → Agent routing

```
user utterance / frame
      │
      ▼
Intent parsers (server/tools/intent_parser.py)
  GeneralIntentParser  · ReadingIntentParser  · TrackingIntentParser
      │                  NavigationIntentParser (new)
      ▼
Intent enum (server/domain/intents.py)
  START_TRACKING, STOP_TRACKING
  START_READING, SCAN_PAGE, READ_ALOUD, PAUSE, CONTINUE, BACK, FORWARD,
  FLIP_DIRECTION
  SAVE_MEMORY, READ_MEMORY
  START_NAVIGATION, STOP_NAVIGATION, SET_DESTINATION   ← NAVIGATE_INTENTS
  INFO, ALERT, …
      │
      ▼
Router selects agent
      │
    ┌─┴──────────────────────────────────────────────────────────┐
    ▼                ▼               ▼            ▼               ▼
TrackingAgent   ReadingAgent   MemoryAgent  NavigationAgent   InfoAgent
```

### Agents

| Agent | File | Purpose |
|-------|------|---------|
| **TrackingAgent** | `server/agents/tracking_agent.py` | GroundingDINO detect → cosine re-ID verify → spatial guidance payload |
| **ReadingAgent** | `server/agents/reading_agent.py` | OCR accumulation (SCANNING) → tokenize sentences → sentence-by-sentence TTS (READ_ALOUD); supports direction toggle (LTR/RTL) |
| **MemoryAgent** | `server/agents/memory_agent.py` | SAVE: OCR buffer → `JsonMemoryStore`; RECALL: RAG semantic search → top-3 hits |
| **NavigationAgent** | `server/agents/navigation_agent.py` | Online navigation: SET_DESTINATION → route to zone; depth-obstacle loop fires cloud VLM to identify obstacle + alert user; proximity alert on arrival |
| **InfoAgent** | `server/agents/info_agent.py` | Scene Q&A via `CloudVLMClient`; no local VLM |

All agents share the signature:
```python
def handle(self, request: AgentRequest) -> AgentResult
```
`AgentResult` carries: `agent_name`, `state`, `payload` (JSON), `reply_text`, `speak` flag.

### Navigation Modes

Navigation has two distinct modes — see `navigation-state.md` for the full state machine.

| Mode | Entry point | Who operates it | What it does |
|------|------------|-----------------|--------------|
| **Offline (Scanning)** | `scan_server/` | Our team, pre-deployment | ORB + VIO/GTSAM → dense point cloud via Plane Sweep MVS per keyframe; label zones; Android ScanScreen records video + IMU CSV for upload |
| **Online (Navigation)** | `server/` — NavigationAgent | End user | Continuous depth obstacle detection → cloud VLM obstacle ID → TTS alert; zone proximity tracking → destination arrival alert |

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
  grpc_server.py                 Main server entry point; loads all models; starts gRPC + Gradio
  services/servicer.py           TrackingServiceServicer — all main-server RPCs
  map_service.py                 MapServiceServicer — static map retrieval
  orchestrator/
    orchestrator.py              Routes request to agent; manages session
    router.py                    Intent → agent selection logic
    session.py                   Per-user session state
  agents/
    base.py                      AgentRequest / AgentResult types
    tracking_agent.py            Object tracking
    reading_agent.py             OCR + TTS reading
    memory_agent.py              JsonMemoryStore + RAG save/recall
    navigation_agent.py          Online navigation: destination routing, depth-obstacle loop, proximity alerts
    info_agent.py                Scene Q&A via CloudVLMClient
  domain/intents.py              Intent enum + TRACKING/READING/MEMORY/NAVIGATE_INTENTS sets
  tools/
    detector.py                  GroundingDINO wrapper
    ocr.py                       Remote OCR client
    tts.py                       Kokoro TTS
    asr.py                       Whisper ASR
    depth.py                     Obstacle detectors: SparseObstacleDetector (ORB, relative depth, default)
                                   + StereoDepthDetector (plane sweep MVS, metric metres; set DEPTH_MODEL=stereo)
    cloud_vlm.py                 CloudVLMClient abstract base + vendor impls; GeminiVLMClient.query_stream() yields raw PCM chunks
    intent_parser.py             LLM-based intent classifiers (incl. NavigationIntentParser)
    memory_store.py              JSON per-label memory
    rag_store.py                 Sentence-transformer + CLIP embeddings
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
  zone_labeler.py                Zone AABB management
  map_exporter.py                PLY + JSON export

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
    ui/
      MainViewModel.kt             doVoiceChatStream() — merges audio+frame+IMU Flow → VoiceChatStream RPC
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
User: "navigate to kitchen" → NavigationAgent (SET_DESTINATION → NAVIGATING)
  │
  ├─ Depth loop (each StreamFrame):
  │    depth.py → forward corridor depth map
  │    → obstacle below threshold → cloud_vlm.py → describe obstacle
  │    → Kokoro TTS alert (15 s cooldown per unique obstacle)
  │
  └─ Proximity check (each StreamFrame when map loaded):
       user position vs zone centroids
       → distance < arrival_radius → "You have arrived at the kitchen"
       → NavigationAgent state → DESTINATION_REACHED
```

### Reading a document aloud
```
"read this" → Whisper ASR → Chat RPC → Orchestrator
  → Intent: START_READING → ReadingAgent (SCANNING)
  → each StreamFrame: OCR accumulates into scan_buffer
  → "continue" → Intent: READ_ALOUD → tokenize sentences
  → TTS sentence 1 → ChatResponse (audio_response)
  → auto-continue timer → next sentence …
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
- `OCR_SERVER_URL` — default `http://localhost:8100`
- `SCAN_GRADIO_PORT` — scan server Gradio port (default 7861)
- `SCAN_DEVICE` — `cpu` or `cuda` for scan server (unused since DA3 removal)
- `DEPTH_MODEL` — `sparse` (default, ORB relative depth), `stereo` (plane sweep MVS), or `da3` (Depth Anything 3 + VIO scale alignment)
- `DA3_MODEL_ID` — DA3 model name (default `depth-anything/da3-large`; also `da3-giant`, `da3metric-large`)
- `CLOUD_VLM_VENDOR` — cloud VLM provider (`stub` default; `anthropic`, `gemini` for live)
- `CLOUD_VLM_API_KEY` — API key for the selected vendor (Anthropic)
- `GEMINI_API_KEY` — API key for Gemini Flash (used when `CLOUD_VLM_VENDOR=gemini`)

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
