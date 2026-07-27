# Pixie HRTF head-tracking test harness

Standalone test harness — **not** part of the main client/server system —
built to validate a design before it lands in the real app (see
`CLAUDE.md`'s eventual "Pixie" persistent-companion beacon note). Answers:
does an RTAB-Map-derived, anchor-relative head heading let a moving pixie's
bearing track a real head turn correctly, with the server's round-trip
latency bridged smoothly by a local ORB/Essential-matrix drift estimate on
the Android side?

## Pieces

- `server/pixie_hrtf_server.py` — Python. Runs a WebSocket server that
  receives camera frames from the test Android app, estimates depth (DA3)
  and pose (RTAB-Map, via `scan_server/rtabmap_client.py`), and reports an
  **anchor-relative heading** back per frame. Also autonomously moves the
  pixie around a circle (`PixieMotion` — see "Pixie motion" below) and runs
  a Gradio dashboard (live status + last frame + radius/speed/wait/
  elevation controls).
- `android/` — a separate Gradle project (package `com.tracking.pixietest`),
  **not** a module of `client/android/`. Streams camera frames to the
  server, and locally bridges the gap between server responses via
  `RotationTracker.kt` (ORB detect + BFMatcher match + Essential-matrix
  RANSAC rotation — **ORB only**, no scale/translation needed here; an
  earlier pass also had a Lucas-Kanade optical-flow path, dropped outright
  per direct request). Two on-screen compasses (`PixiePositionView.kt`)
  show the pixie's bearing both anchor-relative and facing-relative; a
  synthesized tone (`PingTonePlayer.kt`) gets louder the more you're turned
  away from it and silent once aligned.

## Pixie motion

The pixie moves on its own, autonomously, on the server (`PixieMotion` in
`pixie_hrtf_server.py`) — no Android involvement in the motion itself:

- Loops **MOVING** (travel ALONG the circumference of a circle — never a
  straight chord through the middle — to a randomly picked point, at a
  configurable speed) then **WAITING** (a configurable duration) then
  repeats with a new random target.
- The circle's radius, the travel speed, and the wait duration are all
  live-configurable from the Gradio dashboard.
- **Deliberate simplification**: the circle is centered on the anchor's own
  position, and the user's position is assumed to stay there too (RTAB-Map
  reports translation, but this harness never reads it) — this only
  exercises head ROTATION tracking, not walking. Walking during a session
  will make the pixie's apparent bearing drift by however far you moved.

## Dual compass + ping tone

`PixiePositionView` draws two side-by-side compasses every UI tick:

- **ANCHOR** (left) — the pixie's bearing relative to the anchor's original
  forward direction. Does NOT rotate as you turn your head — a fixed
  world-frame bearing, directly `pixie_azimuth_deg` off the wire.
- **FACING** (right) — the pixie's bearing relative to where you're
  CURRENTLY looking. Rotates continuously as you turn your head:
  `pixie_azimuth_deg - currentHeadingDeg` (currentHeadingDeg being the
  locally-bridged anchor-relative heading estimate — see "Wire protocol").

The FACING value (an unsigned angular deviation, 0 = looking exactly at the
pixie) also drives `PingTonePlayer`: silent within ±2°, ramping up to full
volume by ±90° of deviation — a null-seeking alignment cue (turn until it
goes quiet, not until it peaks — a null is sharper than a hunt-for-the-peak
cue). This replaced the earlier continuous HRTF "flapping" beacon
(`HrtfConvolver.kt`/`HrtfBeaconPlayer.kt`, still present in this app,
unreferenced, in case direction-spatialized audio is revisited) once the
pixie gained real autonomous motion.

## Wire protocol

```
Android -> server   {"type":"frame","ts_ns":<int>,"jpeg_b64":<str>}
Android -> server   {"type":"reset"}
server  -> Android  {"type":"heading","frame_ts_ns":<int>,"tracking_ok":<bool>,
                      "relative_heading_deg":<float>,
                      "pixie_azimuth_deg":<float>,"pixie_elevation_deg":<float>}
server  -> Android  {"type":"error","message":<str>}
```

`pixie_azimuth_deg` is now `PixieMotion`'s current angle on the circle
(server-autonomous) rather than a GUI-set value — same wire semantics as
before, different source. The **anchor** is the first frame that yields a
valid RTAB-Map pose after connecting (or after a reset) — its heading
becomes 0 deg, and it's also the circle's center. "Reset anchor" (Gradio
button or the Android app's own button, which also sends `{"type":"reset"}`)
clears it so the NEXT valid frame re-anchors (and re-centers the circle).

## Running it

1. **RTAB-Map docker service** must already be running and reachable (see
   `scan_server/rtabmap_docker/README.md` — same service the main app's
   `MappingService` uses): `docker build -t tracking-rtabmap -f
   scan_server/rtabmap_docker/Dockerfile scan_server/rtabmap_docker` then
   run it, exposing port 5556.
2. **Server** — uses the `hrtf` conda env (same one `scan_server/` code
   already runs under — has `websockets`/`gradio`/`zmq`/`onnxruntime`
   installed):
   ```
   conda activate hrtf
   cd test_module/pixie_hrtf_app/server
   RTABMAP_ADDR=tcp://localhost:5556 python pixie_hrtf_server.py
   ```
   Env vars: `RTABMAP_ADDR` (required), `DA3_ONNX_PATH` (default: the
   repo-root `DA3METRIC-LARGE.onnx`, same file `server/grpc_server.py`
   defaults to), `PIXIE_DA3_DEVICE` (`cpu` default, set `cuda` if a GPU is
   available — depth estimation is the slow part per-frame), `PIXIE_WS_PORT`
   (default 8765), `PIXIE_GRADIO_PORT` (default 7864).
3. **Android app** — separate install from the main client app (different
   `applicationId`, `com.tracking.pixietest`, so both can be installed on
   the same test device at once): `cd test_module/pixie_hrtf_app/android &&
   ./gradlew installDebug` (or `./build.sh` for a from-scratch JDK/SDK
   setup, mirroring `client/android/setup_and_build.sh`).
4. On the phone: enter the server machine's LAN IP + the WS port (8765),
   tap **Connect**, grant the camera permission, then look in whatever
   direction you want to count as "straight ahead" for a moment. Pick a
   **Resolution** (px, long edge of the working image ORB runs on) and
   **ORB points** (feature count, steps of 100) from the dropdowns, and a
   **Target FPS** — all live-adjustable, no reconnect needed. Open the
   Gradio dashboard (`http://<server-host>:7864`) on a computer to watch
   the pixie move and tune its circle. Wearing headphones, confirm the
   ping tone goes quiet exactly when you turn to face the pixie (per the
   FACING compass) and gets louder the more you're turned away, tracking
   smoothly with no noticeable lag between server updates.

## Deliberate simplifications (out of scope for this pass)

This harness only tests the head-tracking/ping-alignment machinery — not
the actual "Pixie" companion entity described in the main design
discussion. Left out on purpose, to keep this test focused:

- No `move(destination, speed)` API called by other app components (the
  motion here is a fixed autonomous circle, not "fly toward open space" or
  "fly toward the tracked target").
- No flying-vs-attached-to-user distinction, no flapping-only-while-flying
  + fade-out behavior — the pixie here always moves the same way regardless
  of any app "mode."
- No "lost track / reset -> teleport to default position" state machine —
  a lost RTAB-Map frame here just means the local drift estimate free-runs
  off the last known-good anchor fix until tracking resumes or someone
  hits reset.
- The circle assumes the user stays in place (see "Pixie motion" above) —
  no live translation tracking.
- Only one Android client at a time — RTAB-Map's own docker service is a
  single active session (see `rtabmap_docker/README.md`), so this harness
  doesn't try to multiplex it.
- **Not verified against a live device/RTAB-Map rig from this
  environment** — compile-verified only (`./gradlew assembleDebug`
  succeeded; `python -m py_compile` succeeded on the server), same standing
  caveat as the rest of this project's Android/RTAB-Map work.

## Next step

Once this validates the anchor + heading + local-drift-bridge + alignment-
cue approach, the real work is building the persistent "Pixie" entity into
`client/android/` itself: a position that survives across mode/state
changes, `moveTo(destination, speed?)` called by navigation/guiding
(fly toward open space) and tracking (fly toward the tracked target),
gradual movement with a sound only while in transit (fading out shortly
after arrival), the pixie riding along with the user's own translation
(not counted as "flying"), teleport-on-reset, and a Settings toggle for
whether Gemini Live's own voice is also routed through the pixie's HRTF
position. Deferred until this harness confirms the tracking approach
actually works — see the user's own request to test first.
