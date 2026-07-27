# Task: Pi Zero 2 W edge-device server

## Goal

Write a Python program that runs on a **Raspberry Pi Zero 2 W** with an
attached **camera** and an **earphone with mic** (assume a standard
USB or 3.5mm analog headset, ALSA-visible as a normal capture+playback
device — adjust the capture/playback backend if the actual hardware turns
out to be different, e.g. Bluetooth or a USB audio dongle with a different
device name).

The Pi acts as the **camera + mic + speaker** for an existing Android app
(`client/android/` in this same repo) that normally uses the phone's own
camera/mic/speaker. When the user enables "Use Remote Edge Device (Pi)" in
the Android app's Settings screen, the Android app instead expects a
device on the same LAN speaking the exact wire protocol below. **Your
program is that device.** The Android side (`RemoteEdgeDevice.kt`) is
already fully implemented and will not change — your job is to build the
Pi side to match it exactly, not the other way around.

This is a from-scratch program — there is no existing Pi-side code in this
repo yet. A close reference implementation (mocking the Pi side, for
Android-side testing) already exists at
`test_module/edge_mock_app/android/app/src/main/java/.../EdgeMockZmqServer.kt`
— read it for the general shape (it's Kotlin, not Python, but the socket
roles/binding/framing are identical to what you need to build), but treat
the protocol spec below as authoritative over that file's exact byte
values (its frame_out cadence/resolution are just mock/test values, not a
requirement).

## Network model

- Pi and Android phone are on the **same LAN** (plain Wi-Fi, no VPN/relay).
- The **Pi always BINDs**, the **Android app always CONNECTs**. Never the
  other way around, and never a REQ/REP handshake — this is a pure
  fire-and-forget PUSH/PULL media pipe, matching every other ZMQ usage
  already in this codebase (see `scan_server/rtabmap_docker/` for a
  precedent, though that one uses REP for a different reason — the edge
  link is PUSH/PULL only, no request/response anywhere).
- No authentication, no TLS. This is assumed to be a private home/trusted
  LAN. Do not add auth/encryption unless explicitly asked — it is out of
  scope and would break compatibility with the already-built Android
  client.
- No discovery mechanism. The Android user manually types the Pi's IP
  address into a Settings text field. Your program should bind to
  `0.0.0.0` (all interfaces) on the fixed ports below, and it's fine (in
  fact expected) for the Pi to have a static/DHCP-reserved LAN IP the user
  knows ahead of time.
- Library: **ZeroMQ** (`pyzmq` — `pip install pyzmq`). Do not use raw
  sockets, gRPC, WebSockets, or any other transport — the Android side is
  hardcoded to ZMQ PUSH/PULL.

## The four sockets (fixed ports, no negotiation)

| Socket | ZMQ type (Pi side) | Port | Direction | Payload |
|---|---|---|---|---|
| `frame_out` | `PUSH` (bind) | **5602** | Pi → Android | JPEG-encoded camera frame |
| `luma_out` | `PUSH` (bind) | **5604** | Pi → Android | Raw Y-plane (luma-only) camera frame + small sub-header |
| `mic_out` | `PUSH` (bind) | **5601** | Pi → Android | Raw mic PCM chunk |
| `audio_in` | `PULL` (bind) | **5603** | Android → Pi | Rendered PCM to play on the earphone speaker |

All four ports are **fixed constants today** — there is no control/config
channel yet (see "Known gaps" below). Bind all four unconditionally at
startup and keep them open for the process's whole lifetime; don't try to
open/close them per "mode."

## Wire framing (applies to `frame_out`, `luma_out`, `mic_out`, `audio_in`)

Every message on every one of these four sockets is a **2-part ZMQ
multipart message**:

1. **Frame 1 — 16-byte header**, little-endian:
   - bytes 0–7: `int64` sequence number (monotonically increasing per
     socket, starting at 0, one counter per socket — doesn't need to be
     shared/synced across sockets)
   - bytes 8–15: `float64` Unix timestamp in **seconds** (i.e.
     `time.time()`, not milliseconds)
2. **Frame 2 — the raw payload** (format depends on which socket — see
   below). Sent as a second ZMQ frame in the same multipart message (i.e.
   `socket.send_multipart([header_bytes, payload_bytes])` in pyzmq) — NOT
   length-prefixed manually; ZMQ's own multipart framing is what separates
   the two.

The Android side currently **ignores** the header's actual seq/timestamp
values (it reads and discards frame 1, see `RemoteEdgeDevice.recvLoop()`)
— but you must still send it as a well-formed 16-byte frame on every
message, on every socket including `audio_in`-bound acks... actually
`audio_in` is Android→Pi, so the header on THAT socket is written by the
**Android side**, not you — your Pi program only needs to (a) send this
header format on `frame_out`/`luma_out`/`mic_out`, and (b) be able to
receive and skip/ignore this same 16-byte header on the *first* frame of
each `audio_in` message before reading the actual PCM payload from the
second frame.

Python example for sending on `frame_out`/`luma_out`/`mic_out`:

```python
import struct, time

def send_framed(sock, seq: int, payload: bytes):
    header = struct.pack("<qd", seq, time.time())  # <qd = little-endian int64 + float64
    sock.send_multipart([header, payload])
```

And for receiving on `audio_in`:

```python
def recv_framed(sock):
    header, payload = sock.recv_multipart()
    seq, ts = struct.unpack("<qd", header)
    return payload  # seq/ts not currently needed by anything downstream
```

## Per-socket payload formats

### `frame_out` (JPEG camera frames)

Payload = a complete, standard JPEG-encoded image, exactly as
`cv2.imencode('.jpg', frame)[1].tobytes()` or Pillow's `.save(buf,
format='JPEG')` would produce. No sub-header, no extra framing — just the
raw JPEG bytes as frame 2.

- **Recommended**: downscale to **640px on the long edge** (matches the
  phone's own local-camera convention in `CameraManager.kt`, so Gemini
  Live/the detection pipeline sees a consistent resolution regardless of
  which edge device is active), JPEG quality ~50.
- **Recommended rate**: since there's no control channel yet (see below),
  pick a single fixed rate that's reasonable across all app modes —
  **~2 fps** is a safe default (matches the phone's own slowest mode,
  "guiding," at 1000ms; walking/scanning on the phone go faster but this
  socket isn't the only frame source — see `luma_out` below for the
  higher-rate stream those modes actually depend on more).
- These frames get shown to the user's AI assistant (Gemini) for scene
  understanding and object detection — image quality/exposure matters
  more here than for `luma_out`.

### `luma_out` (raw Y-plane frames, for on-device visual rotation tracking)

Payload = **16-byte sub-header + raw luma bytes**, little-endian:

```
[4-byte int32 width][4-byte int32 height][4-byte int32 rowStride][4-byte int32 rotationDegrees][raw luma bytes...]
```

- `width`/`height`: the luma image's actual pixel dimensions.
- `rowStride`: bytes per row in the luma buffer (usually equal to
  `width` for a tightly-packed 8-bit grayscale buffer — set `rowStride =
  width` unless your camera capture pipeline pads rows, in which case set
  it to the real stride so the Android side can skip padding correctly).
- `rotationDegrees`: one of `0`/`90`/`180`/`270` — how much the raw sensor
  buffer needs to be rotated to appear upright. If your camera mount is
  fixed and you don't know/care about orientation, `0` is fine as a
  starting point, but get this right if rotation tracking looks wrong
  later (see "How the Android side uses this" below).
- The luma bytes themselves: **one byte per pixel**, 8-bit grayscale,
  `height * rowStride` bytes total, row-major, top-to-bottom — i.e.
  exactly the Y-plane of a YUV420 frame, or equivalently
  `cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)`'s output if you only have
  an RGB/BGR camera pipeline available (a converted grayscale frame is an
  acceptable stand-in for a true camera Y-plane — the Android side only
  does ORB feature matching on it, it doesn't need "real" YUV).

- **Recommended**: **480px on the long edge**, **~15fps**. This is a
  *separate, independent* capture/encode from `frame_out` — different
  resolution, different (faster) rate, no JPEG compression (raw bytes
  only). It does not need to be extracted from the exact same physical
  frame as `frame_out`, though it can be if that's simpler for your camera
  pipeline (e.g. capture once, produce both a downscaled-JPEG output and a
  separately-downscaled grayscale output from the same sensor frame).

**How the Android side uses this**: it feeds a continuous ORB
(feature-point) matcher (`AngleTracker.kt`) that tracks frame-to-frame
camera rotation for HRTF steering-cue purposes during "walking"/"guiding"
modes. It wants a **steady, frequent trickle** more than it wants
individual frame quality — prioritize consistent ~15fps delivery over
resolution/sharpness if you have to trade off given the Pi Zero 2's
limited CPU.

### `mic_out` (raw mic audio)

Payload = raw **PCM16 little-endian, mono, 16000 Hz**, no header/wrapping
beyond the standard 16-byte seq+timestamp frame already described above.

- **Recommended chunk size: 512 samples per message** (i.e. 1024 bytes of
  payload) — this matches the phone's own local capture chunk size
  exactly (`ContinuousVadRecorder.kt`'s `CHUNK = 512`), so the Android
  side's voice-activity-detection state machine (which runs once per
  received chunk) behaves the same way it does with local capture. Sending
  smaller or larger chunks won't break anything functionally, but keep
  each chunk in the same rough ballpark (a few tens of ms) — don't batch
  up multiple seconds of audio into one message, and don't send
  individual-sample messages either.
- Stream **continuously** for the whole time your program is running —
  there's no "start/stop listening" signal from Android today. Just always
  be capturing and pushing mic chunks; Android's own voice-activity-
  detection logic (amplitude-threshold gating) decides what to actually do
  with the audio, not you.
- Use a real mic capture API (`sounddevice`, `pyaudio`, or raw
  `arecord`/ALSA via a subprocess) at native 16kHz mono if the hardware
  supports it directly; if your mic hardware only supports a different
  native rate (e.g. 44.1kHz or 48kHz, common for USB headsets), capture at
  the native rate and **downsample to 16kHz mono PCM16** before sending
  (a simple linear-interpolation resample, or `scipy.signal.resample`, is
  sufficient — this doesn't need to be broadcast-quality).

### `audio_in` (rendered audio to play on the earphone speaker) — Pi RECEIVES this

This is the one socket where **you PULL and the Android app PUSHes**.

Payload format = raw **PCM16 little-endian, STEREO (interleaved
L,R,L,R,...), 44100 Hz**.

- Chunk sizes will vary (Android sends whatever it currently has ready,
  typically ~2048 frames / 8192 bytes per message, but don't hardcode that
  assumption — just decode-and-play whatever arrives, in order, as it
  arrives).
- **Play these chunks back-to-back through the earphone speaker with no
  gaps** — this is a combined stream (Gemini's spoken voice + a
  continuous spatial-audio steering cue mixed together on the Android
  side), so treat it as one continuous audio feed to play, not
  discrete/interruptible clips. Use a streaming playback API
  (`sounddevice.OutputStream`/`pyaudio` in callback or blocking-write
  mode) that can be fed chunk-by-chunk without re-opening the audio device
  each time — reopening per-chunk will cause audible clicks/gaps.
- Messages may **not arrive continuously** — Android only pushes when
  there's actually something to play (see "Design notes" below). Expect
  silence/no messages for stretches of time; don't treat a gap as an error
  or try to reconnect.
- Note the earphone's speaker and mic are almost certainly different
  physical/logical devices in ALSA (or the same USB device exposing
  separate playback/capture sub-devices) — make sure your program opens
  the correct one for each direction; test both independently first if
  unsure which ALSA device name/index is which.

## Suggested Python structure

A single long-running process is sufficient — no need for a supervisor/
multi-process architecture on a Pi Zero 2's limited resources. Rough shape:

```
main.py
├── zmq context + 4 sockets, bound at startup
├── thread/task: camera capture loop → encodes JPEG → sends on frame_out
├── thread/task: camera capture loop → encodes/grayscales luma → sends on luma_out
│     (can share a camera capture source with frame_out if convenient —
│      see picamera2's request/array API for grabbing one frame and
│      deriving multiple outputs from it)
├── thread/task: mic capture loop → sends on mic_out
└── thread/task: audio_in recv loop → plays through speaker, back-to-back
```

- **Camera**: if it's the official Raspberry Pi Camera Module,
  `picamera2` (the modern libcamera-based Python API) is the right choice
  on a current Raspberry Pi OS. If it's a USB webcam instead, `opencv-
  python`'s `cv2.VideoCapture` is simpler and fine. Confirm which kind of
  camera is actually attached before committing to a library — do not
  assume; ask if genuinely ambiguous from the hardware description.
- **JPEG encode**: `cv2.imencode` or Pillow, either is fine.
- **Threading**: plain Python `threading` (with a shared `ZMQ.Context`
  instantiated once, one socket per thread — pyzmq sockets are not
  thread-safe to share across threads, so each capture/send loop should
  own its own socket instance from the shared context, matching how the
  Android side gives each direction its own dedicated thread too).
- Keep resource usage light — this is a Pi Zero 2 W (512MB RAM, quad-core
  Cortex-A53 ~1GHz). Avoid heavyweight frameworks; avoid buffering more
  than a frame or two of anything (drop old frames rather than let a queue
  grow, same "prefer fresh over complete" policy the Android side already
  uses for its own camera pipeline).

## Testing / verification

There's no live server to test against from a dev machine that doesn't
have the real Android app — but you can sanity-check your protocol
implementation two ways:

1. **Read `test_module/edge_mock_app/`'s Kotlin `EdgeMockZmqServer.kt`** —
   it implements this exact same wire framing (minus the `luma_out`
   socket/sub-header, which postdates that file) as a mock/test server, so
   its `sendFramed()`/socket-binding code is a working cross-check for the
   framing byte layout.
2. Write a small standalone Python test client (not part of your deliverable, just for your own verification) that connects to your bound ports as PULL/PUSH sockets and confirms: `frame_out` messages decode as valid JPEGs, `luma_out` messages parse to a sensible width/height with `payload_len - 16 == height * rowStride`, `mic_out` messages are even-length (whole PCM16 samples) at roughly the expected chunk rate, and that pushing synthetic PCM16 stereo data to your `audio_in` port results in audible playback.

## Known gaps — do not try to solve these, they're explicitly out of scope for this task

- **No dynamic control channel.** The Android app has per-mode frame
  rates/resolutions internally (e.g. it wants frames faster during
  "walking" mode than "idle") but has no way to communicate that to a
  remote edge device yet — pick fixed, reasonable rates per the
  recommendations above and don't try to build a control/config socket
  unless separately asked to.
- **No reconnect/liveness signaling.** If the Wi-Fi drops, ZMQ's PUSH/PULL
  sockets will just silently stop delivering on both ends with no
  explicit error — this is a known, accepted limitation of the whole
  design (see the Android-side code comments in `RemoteEdgeDevice.kt`),
  not something your Pi-side program needs to detect/recover from
  specially. Standard ZMQ behavior (auto-reconnect on the underlying TCP
  connection) is sufficient.
- **No authentication/encryption** (see "Network model" above).

## Design notes for context (you don't need to change any of this — it's already built and working on the Android side; included so the "why" behind the format choices above makes sense)

- The Android app mixes Gemini's spoken voice (mono 24kHz) and a spatial
  audio steering cue (stereo 44.1kHz) into one combined stereo 44.1kHz
  stream (`AudioMixer.kt`) before pushing to `audio_in` — that's why
  you'll only ever receive ONE stereo 44.1kHz stream, never two separate
  ones to mix yourself.
- The Android app only pushes to `audio_in` when the mixed result isn't
  pure silence — it doesn't send a continuous stream of silent frames when
  nothing is playing, so absence of messages is the normal "nothing to
  play right now" state, not a dropped connection.
