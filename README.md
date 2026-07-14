# Tracking — Vision-Assistance System

An AI-powered assistive system for vision-impaired people: real-time object
tracking with spatial guidance, text-to-speech reading, 3D venue mapping with
named zones, voice/text Q&A, and memory recall — all through a phone or
Pi-mounted camera.

This file covers **how to set up and run** every component. For system
architecture, data flows, and the full file map, see [`CLAUDE.md`](CLAUDE.md)
(and [`server/ARCHITECTURE.md`](server/ARCHITECTURE.md) for main-server
internals).

---

## Quick start

Two independent workflows:

- **Build a map of a venue (offline, done once per location)** →
  [Scan server](#scan-server) + Android app's Scan screen.
- **Use the assistant live (online, day-to-day)** → OCR server + Main server
  (+ Mediator if using a Raspberry Pi camera instead of the phone) + Android
  app.

Both need `GEMINI_API_KEY` set. See [Workflows](#workflows) below for the
step-by-step for each.

---

## Prerequisites

| Need | Why |
|------|-----|
| Python 3.12 | Main server / Mediator (`server/.venv`) |
| Python 3.11 (via conda) | Scan server (`conda env hrtf`) — see [Development Environments](CLAUDE.md#development-environments) |
| `GEMINI_API_KEY` | Gemini Live API — required by the Main server; Scan server's semantic mapper also uses it |
| Docker + NVIDIA Container Toolkit | OCR server (GPU PaddleOCR image), and the optional RTAB-Map pose service |
| NVIDIA GPU | DA3 depth models (Main server, Scan server) — CPU fallback exists but is slow |
| JDK 17 + Android SDK | Building the Android client |

---

## Python environments

### `server/.venv` — Main server + Mediator

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

Commonly-tuned optional ones (full list in
[CLAUDE.md's Deployment section](CLAUDE.md)):

| Variable | Default | Used by |
|----------|---------|---------|
| `OCR_SERVER_URL` | `http://localhost:8100` | Main server |
| `DEPTH_MODEL` | `sparse` | Main server — `sparse`\|`stereo`\|`da3` |
| `SCAN_DEVICE` | `cuda` | Scan server — `cpu`\|`cuda` |
| `SCAN_GRADIO_PORT` | `7861` | Scan server |
| `RTABMAP_ADDR` | unset (disabled) | Scan server — e.g. `tcp://localhost:5556` |
| `ORBSLAM3_ADDR` (removed) | — | no longer used — RTAB-Map replaced ORB-SLAM3 |

---

## Running the components

### OCR server

Needed by the Main server's reading-mode OCR.

```bash
cd paddle_ocr_server
uvicorn server:app --host 0.0.0.0 --port 8100
```

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

### Mediator (optional — only if using a Raspberry Pi camera, not a phone)

```bash
source server/.venv/bin/activate
python client/mediator_gui.py --server-ip <main-server-ip>
```

- gRPC on port `50052`, Gradio dashboard on port `7862`.
- Runs local ORB tracking + hand detection on the Pi-adjacent host, forwards
  throttled frames to the Main server.

### Scan server

Team-operated, offline — builds 3D maps from a recorded walkthrough.

```bash
conda activate hrtf
export GEMINI_API_KEY=your-key-here
python scan_server/scan_server.py [--da3-model torch|onnx] [--da3-onnx-path PATH]
```

- FastAPI + Gradio UI on port `7861` (`SCAN_GRADIO_PORT` to change it).
- `--da3-model torch` (default) also enables the "DA3 poses" pose source;
  `--da3-model onnx` is lighter/faster but loses that pose option.

### RTAB-Map pose service (optional — a more robust pose source for scanning)

No per-device calibration needed (unlike the ORB-SLAM3 service it replaced —
see [`scan_server/rtabmap_docker/README.md`](scan_server/rtabmap_docker/README.md)):

```bash
docker build -t tracking-rtabmap scan_server/rtabmap_docker
docker run --rm -p 5556:5556 tracking-rtabmap
# then point the Scan server at it:
export RTABMAP_ADDR=tcp://localhost:5556
```

Or via Compose: `docker compose up rtabmap` (see below).

### Android client

```bash
cd client/android
./setup_and_build.sh   # one-shot: installs JDK 17 + Android SDK, builds the APK
```

Installs to `app/build/outputs/apk/debug/`. On first launch, point it at the
Main server's IP (voice/live-assist) and/or the Scan server's IP (Scan
screen), both on your LAN.

---

## Workflows

### A) Build a map of a venue (offline)

1. Start the [Scan server](#scan-server) (optionally with
   [RTAB-Map](#rtab-map-pose-service-optional--a-more-robust-pose-source-for-scanning) running too).
2. On the Android app's Scan screen: tap **Record**, walk the venue, tap
   **Stop**, then **Upload Files** — this zips `images/ + camera.csv +
   imu.csv` and POSTs it to the Scan server.
3. In the Scan server's Gradio UI (`http://<host>:7861`): "Load from Android
   Upload" → pick the upload → fill the **Segment Table** (one row per named
   zone, e.g. `0, 45, kitchen`) → choose a **Pose source** (`Auto` is a
   reasonable default; `RTAB-Map` if the service above is running) → **Scan**.
4. **Export Map** — writes `scan_server/data/maps/<location_id>/` (PLY +
   `map_labels.json` + keyframes), which the Main server's `MapService`
   serves at runtime.

### B) Live assistive use (online)

1. Start the [OCR server](#ocr-server) and [Main server](#main-server) (+
   [Mediator](#mediator-optional--only-if-using-a-raspberry-pi-camera-not-a-phone)
   if using a Pi camera instead of the phone directly).
2. Launch the Android app, connect to the Main server.
3. Speak naturally — Gemini Live handles intent routing (tracking, reading,
   navigating to a mapped zone/landmark, memory recall, device actions like
   calls/alarms/calendar).

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
