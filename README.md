# Tracking — Vision-Assistance System

An AI-powered assistive system for vision-impaired people: real-time object
tracking with spatial guidance, text-to-speech reading, live 3D venue
mapping and landmark-based navigation, voice/text Q&A, and memory recall —
all through the Android app, running Gemini Live directly on the phone.

This file covers **how to set up and run** every component. For system
architecture, data flows, and the full file map, see [`CLAUDE.md`](CLAUDE.md)
(and [`server/ARCHITECTURE.md`](server/ARCHITECTURE.md) for main-server
internals).

---

## Quick start

One workflow: OCR server + Main server + RTAB-Map pose service + Android app
(the only client this project ships — Android talks to Gemini Live
directly). Mapping runs live, automatically, during guiding/walking mode —
there's no separate offline scanning step any more. Everything needs
`GEMINI_API_KEY` set. See [Workflows](#workflows) below for the step-by-step.

---

## Prerequisites

| Need | Why |
|------|-----|
| Python 3.12 | Main server (`server/.venv`) |
| Python 3.11 (via conda) | Scan server (`conda env hrtf`) — see [Development Environments](CLAUDE.md#development-environments) |
| `GEMINI_API_KEY` | Gemini Live API — required by the Main server; Scan server's semantic mapper also uses it |
| Docker + NVIDIA Container Toolkit | OCR server (GPU PaddleOCR image), and the optional RTAB-Map pose service |
| NVIDIA GPU | DA3 depth models (Main server, Scan server) — CPU fallback exists but is slow |
| JDK 17 + Android SDK | Building the Android client |

---

## Python environments

### `server/.venv` — Main server

```bash
python3.12 -m venv server/.venv
source server/.venv/bin/activate
pip install -r server/requirements.txt
```

### conda env `hrtf` — Scan server

The Scan server is developed/tested against this conda env specifically (not
`server/.venv`, even though the venv is kept roughly in sync):

```bash
conda create -n hrtf python=3.11
conda activate hrtf
pip install -r scan_server/requirements.txt
pip install -e scan_server/Depth-Anything-3   # DA3 model package, editable install
```

### `paddle_ocr_server` — usually run via Docker (see below) instead of a
local env, since PaddlePaddle's GPU wheels are picky about CUDA/cuDNN
versions; the Dockerfile pins a known-good base image.

---

## Environment variables

Required:

- `GEMINI_API_KEY` — Gemini Live API key. Export it in every shell that runs
  `server/grpc_server.py` or `scan_server/scan_server.py`, or add it to
  `docker-compose.yml`'s `environment:` block if running that way (it is
  **not** passed through automatically).

Everything below is read by the Main server (`server/grpc_server.py`) — there
is no separate Scan server process any more, and OCR is called directly from
Android now (its URL is entered in the app's Settings screen, not an env var
here). Full narrative in
[CLAUDE.md's Deployment section](CLAUDE.md#deployment).

| Variable | Default | Used for |
|----------|---------|---------|
| `DA3_ONNX_PATH` | `DA3METRIC-LARGE.onnx` | `DA3DepthDetector`'s ONNX model — walking-mode obstacle depth checks (`PerceptionService.AnalyzeFrame`'s `DEPTH` op) |
| `RTABMAP_ADDR` | unset (disabled) | RTAB-Map pose service address, e.g. `tcp://localhost:5556` — **required** to enable `MappingService` (live mapping/guiding); without it, `MappingService` registration is skipped entirely (logged, not fatal) |
| `SCAN_DA3_TORCH_MODEL_ID` | `depth-anything/da3-large` | `MappingService`'s live-mapping dense-reconstruction depth (torch backend) — a separate model instance from `DA3_ONNX_PATH` above |
| `SCAN_GEMMA_MODEL_ID` | `gemma-4-31b-it` | `MappingService`'s landmark-tagging VLM, via Gemini API (`GEMINI_API_KEY`) |
| `MEMORY_STORE_DIR` | `server/data/memory` | `RagStore`'s on-disk text-embedding storage dir |
| `RAG_MODEL_ID` | `sentence-transformers/all-MiniLM-L6-v2` | `RagStore`'s sentence-transformer model (backs `PerceptionService.Embed`) |

---

## Running the components

### OCR server

Called directly by Android for reading mode (not proxied through the Main server).

```bash
cd paddle_ocr_server
uvicorn server:app --host 0.0.0.0 --port 8100
```

Pipeline-stage debug UI (received → preprocessed → raw text blocks →
merged paragraphs → final text) at `http://<host>:8100/gui` — same process,
no extra command.

Or via Docker (recommended — pins the CUDA/cuDNN-matched PaddlePaddle base image):

```bash
docker build -t tracking-ocr paddle_ocr_server
docker run --rm --gpus all -p 8100:8100 tracking-ocr
```

### Main server

```bash
source server/.venv/bin/activate
export GEMINI_API_KEY=your-key-here
python server/grpc_server.py
```

- gRPC on port `50051`, Gradio monitor dashboard on port `7860`.
- Loads GroundingDINO, DINOv2, and a DA3 depth detector on startup — first
  run downloads model weights, expect a delay.

### Scan server

There is no longer a standalone Scan server process — live mapping
(`scan_server/`'s occupancy-map pipeline) now runs in-process inside the
Main server (`server/grpc_server.py` imports it directly), driven live by
the Android client's guiding/walking mode instead of a recorded-and-uploaded
dataset. `scan_server/scan_gui.py`'s Gradio debugging UI still exists but
currently has no launcher (its old entrypoint, `scan_server.py`, was
removed) — see `CLAUDE.md`'s "Client-Orchestrated Live Session" section.

### RTAB-Map pose service (required — the only pose source MappingService supports)

No per-device calibration needed (unlike the ORB-SLAM3 service it replaced —
see [`scan_server/rtabmap_docker/README.md`](scan_server/rtabmap_docker/README.md)):

```bash
docker build -t tracking-rtabmap scan_server/rtabmap_docker
docker run --rm -p 5556:5556 tracking-rtabmap
# then point the Main server at it:
export RTABMAP_ADDR=tcp://localhost:5556
```

Or via Compose: `docker compose up rtabmap` (see below).

### Android client

```bash
cd client/android
./setup_and_build.sh   # one-shot: installs JDK 17 + Android SDK, builds the APK
```

Installs to `app/build/outputs/apk/debug/`. It's the only client this
project ships — on first launch, open Settings and enter your Gemini API
key, the Main server's IP, and the OCR server's IP (all on your LAN).

---

## Workflows

Mapping is no longer a separate offline step — it runs live, automatically,
whenever guiding/walking mode is on (see `CLAUDE.md`'s
"Client-Orchestrated Live Session" section). There's just one workflow now:

1. Start the [OCR server](#ocr-server), the [Main server](#main-server), and
   the [RTAB-Map pose service](#rtab-map-pose-service-optional--a-more-robust-pose-source-for-scanning)
   (required — MappingService is RTAB-Map-only).
2. Launch the Android app, enter your Gemini API key + the Main server's
   address in Settings, connect.
3. Speak naturally — Gemini Live (running directly on the phone) handles
   intent routing: tracking, reading, navigating to a landmark, memory
   recall, device actions like calls/alarms/calendar. Say "guide me to
   <landmark>" or "start walking" to begin — the occupancy map for that
   `location_id` builds up live as you walk and persists for next time.

---

## Docker Compose

```bash
docker-compose up
```

Brings up the Main server (`streaming-vlm-server`, ports `7860`/`50051`).
The `rtabmap` service is defined but **not** started by a bare
`docker-compose up` — bring it up explicitly:

```bash
docker compose up rtabmap
```

Remember to export `GEMINI_API_KEY` before running `docker-compose up`, or
add it under the `streaming-vlm-server` service's `environment:` block —
Compose does not pass host environment variables through automatically.

---

## More documentation

- [`CLAUDE.md`](CLAUDE.md) — full architecture, gRPC services, data flows, file map
- [`server/ARCHITECTURE.md`](server/ARCHITECTURE.md) — Main server internals
- [`scan_server/rtabmap_docker/README.md`](scan_server/rtabmap_docker/README.md) — RTAB-Map service build/run details
