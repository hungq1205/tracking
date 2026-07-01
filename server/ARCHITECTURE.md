# Server Architecture

> **This documents the pre-Gemini-Live-migration server internals.**
> The current implementation uses `LiveAPISession` (see `live_session.py`).

---

## Component Map

```
grpc_server.py          Entry point — loads models, wires servicer, starts gRPC (50051) + Gradio (7860)
services/servicer.py    TrackingServiceServicer — all gRPC RPC implementations
live_session.py         LiveAPISession — Gemini Live WebSocket session per user (replaces Orchestrator)
live_tools/             Tool implementations dispatched by Gemini function calls
  tool_declarations.py  All FunctionDeclaration dicts + SYSTEM_PROMPT
  reading_tools.py      OCR accumulation, sentence navigation
  tracking_tools.py     GroundingDINO detection, object memory lookup
  memory_tools.py       RAG save/query, object memory
  navigation_tools.py   Route planning, localization
  scene_tools.py        get_latest_frame, vision stream, obstacle check

tools/
  detector.py           GroundingDINODetector — zero-shot detection
  depth.py              SparseObstacleDetector / StereoDepthDetector / DA3DepthDetector
  ocr.py                DocLayoutRapidOCRTool — HTTP client to OCR microservice (port 8100)
  rag_store.py          RagStore — sentence-transformer (384d) + CLIP (512d) embeddings
  memory_store.py       JsonMemoryStore — per-label JSON files; filter_new_sentences()
  localization.py       LocalizationEngine — PnP against map keyframes
  route_planner.py      RoutePlanner — zone-based path planning from map_labels.json
  embedder.py           DINOv2Embedder (ViT-S/14) — visual re-ID embeddings

map_service.py          MapServiceServicer — streams PLY files from data/maps/
interfaces.py           IObjectDetector abstract base
```

---

## gRPC Data Flow

### VoiceChatStream (primary path)

```
Android ──PCM chunks + JPEG frames──► VoiceChatStream RPC
                                              │
                                    LiveAPISession.send_audio_sync(pcm)
                                    LiveAPISession.receive_frame_sync(jpeg)
                                              │
                                  ┌───────────▼───────────┐
                                  │  Async Gemini Live     │
                                  │  WebSocket loop        │
                                  │  (background thread)   │
                                  │                        │
                                  │  • Audio → Gemini ASR  │
                                  │  • Gemini decides tool │◄── [SYSTEM] obstacle alerts
                                  │  • _dispatch_tool()    │    injected via send_realtime_input
                                  │  • Gemini TTS output   │
                                  └───────────┬───────────┘
                                              │ PCM chunks
                                    output_queue (thread-safe)
                                              │
                            yield AudioChunk(pcm_data=...) ──► Android
```

### Background frame tick (per received frame)

```
receive_frame_sync(jpeg) → _on_frame_tick(jpeg)
  if mode=="reading" and 1.5s elapsed:
    OCR frame → filter_new_sentences → append to reading_buffer
  if mode=="navigation" and 0.5s elapsed:
    depth_detector.check_obstacle → if obstacle: inject [SYSTEM] text to Gemini
  if mode=="navigation" and 2.0s elapsed:
    localizer.localize → update nav_last_position → check proximity
```

### StreamFrame / Chat / VoiceChat

These RPCs are kept for compatibility but are secondary paths:
- `DetectObject` / `GetEmbedding` — call detector/embedder directly, unchanged
- `StreamFrame` — stores `latest_frame` only (OCR/nav ticks moved to LiveAPISession)
- `Chat` / `VoiceChat` — simple text stubs using one-shot Gemini generate call

---

## Tool → Function Map

| Gemini calls... | Server does... |
|-----------------|----------------|
| `get_latest_frame()` | Sends JPEG via `send_realtime_input(video=...)` |
| `start_vision_stream()` | Spawns `_vision_stream_loop()` at 1 fps, auto-stops 15 s |
| `stop_vision_stream()` | Cancels vision stream loop |
| `run_detection(desc)` | GroundingDINO on `latest_frame` |
| `check_obstacle()` | depth_detector on `latest_frame` |
| `enter_reading_mode(label?)` | Sets `state.mode="reading"`, resets reading_buffer |
| `scan_current_view()` | OCR on `latest_frame`, dedup-appends to reading_buffer |
| `get_reading_section(query)` | Semantic search over reading_buffer chunks |
| `flip_reading_direction()` | Toggles ltr/rtl |
| `exit_reading_mode()` | Clears reading state |
| `start_tracking(target)` | GroundingDINO detect, sets mode="tracking" |
| `stop_tracking()` | Clears tracking state |
| `get_object_from_memory(query)` | rag_store.query_global, threshold 0.5 |
| `query_memory(question)` | rag_store.query_global, threshold 0.5 |
| `save_memory(label, note)` | memory_store.append + rag_store.add_text |
| `remember_object(label)` | Detect crop + save to rag_store.add_object |
| `list_memory_labels()` | List known JSON memory files |
| `start_navigation(dest)` | Load map, compute route, set mode="navigation" |
| `stop_navigation()` | Clear nav state |
| `get_current_location()` | PnP localize against map keyframes |

---

## State Machine (LiveSessionState)

```
mode: idle
  │
  ├─[enter_reading_mode]──► reading
  │   │ passive OCR tick (1.5s), reading_buffer accumulates
  │   └─[exit_reading_mode]──► idle
  │
  ├─[start_tracking]──────► tracking
  │   └─[stop_tracking]───► idle
  │
  └─[start_navigation]────► navigation
      │ depth tick (0.5s), localize tick (2.0s), obstacle injection
      └─[stop_navigation / destination reached]──► idle
```

---

## Key Data Directories

```
server/data/
  memory/          {label}.json + {label}.npy + images/{label}/ — RAG memory store
  maps/{id}/       map_geometry.ply + map_labels.json — served by MapService
```
