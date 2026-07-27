"""Raspberry Pi Zero 2 W edge server: streams camera + mic to the Android app
and plays back whatever audio the app sends, over four fixed ZeroMQ
PUSH/PULL sockets. The wire protocol below is NOT ours to change — it
matches the already-built Android-side client exactly
(client/android/app/src/main/java/com/tracking/client/edge/RemoteEdgeDevice.kt
+ AudioMixer.kt). This process only ever BINDs; the Android app CONNECTs out
to whatever IP the user types in, so there is no discovery/auth/TLS here.

WIRE FRAMING (every message, on all 4 sockets)
-----------------------------------------------
Every message is a 2-part ZMQ multipart send:

    Frame 1 -- 16-byte header, little-endian:
        bytes 0-7   int64   sequence number (per-socket counter, starts at 0)
        bytes 8-15  float64 unix timestamp (time.time())
    Frame 2 -- the payload (format depends on the socket, see below)

    header = struct.pack("<qd", seq, time.time())
    sock.send_multipart([header, payload])

PAYLOAD FORMATS
---------------
1. frame_out (PUSH, port 5602, us -> Android)
   Raw JPEG bytes. 640px long edge, quality ~50, ~2 fps. Physically rotated
   (see ROTATION NOTE below) -- NOT just tagged with rotation metadata,
   because unlike luma_out this channel carries no rotation field at all,
   and the Android side decodes it straight into a Bitmap with no
   compensation applied anywhere downstream.

2. luma_out (PUSH, port 5604, us -> Android)
   16-byte sub-header + raw Y-plane bytes:
       bytes 0-3   int32  width
       bytes 4-7   int32  height
       bytes 8-11  int32  rowStride
       bytes 12-15 int32  rotationDegrees (0/90/180/270)
   followed by height*rowStride bytes of 8-bit grayscale, row-major,
   RAW (not rotated -- Android's AngleTracker rotates it itself using the
   rotationDegrees field). 480px long edge, ~15 fps.

3. mic_out (PUSH, port 5601, us -> Android)
   Raw PCM16 little-endian, mono, 16000 Hz. Recommended 512-sample
   (1024-byte) chunks, continuous.

4. audio_in (PULL, port 5603, Android -> us)
   Raw PCM16 little-endian, STEREO interleaved, 44100 Hz. Variable chunk
   sizes; arrival is bursty (silence = no messages at all), so playback
   pads with digital silence rather than blocking/gapping when nothing has
   arrived recently.

ROTATION NOTE
-------------
The Pi Camera Module 3 (Sony IMX708, 4608x2592, 16:9) is physically mounted
rotated. --rotation-degrees (default 90) is the single source of truth for
BOTH channels: luma_out reports it as metadata (Android applies the
rotation itself), while frame_out has the SAME rotation baked into the JPEG
pixels directly with cv2.rotate before encoding. If the image looks
sideways/upside-down on first real hardware test, flip this value
(0/90/180/270) rather than editing code -- it drives both channels
identically by construction.

THREADING MODEL
----------------
One shared zmq.Context(), but every socket is created and used by exactly
one thread -- ZMQ sockets are never shared across threads. Camera capture
is a single thread (concurrent capture_request() calls against one
Picamera2 instance from two different threads is not a pattern picamera2
is designed for) that distributes frames into two small drop-oldest queues;
two separate sender threads each own their own PUSH socket and do the
(comparatively expensive) JPEG encode / sub-header packing at their own
pace. Mic capture -> mic_out and audio_in -> speaker are each their own
producer/consumer thread pair.

Queues intentionally behave differently for video vs. audio: frame/luma
queues are maxsize=1 and drop the oldest entry under backlog (a fresher
frame is strictly better than a stale one, and it costs nothing to skip a
video frame). The mic queue is deliberately NOT drop-first: dropping a
chunk mid-utterance corrupts what Gemini hears, so it's sized generously
(about 30s of audio) and only drops-oldest as an absolute last resort if
something has been broken for a long time -- under normal load it should
never need to.
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import struct
import sys
import threading
import time

import numpy as np
import zmq

try:
    import cv2
except ImportError:  # pragma: no cover - dependency is required to run
    cv2 = None

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - dependency is required to run
    sd = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_FMT = "<qd"  # int64 seq, float64 unix timestamp -- 16 bytes total
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 16

LUMA_SUBHEADER_FMT = "<iiii"  # width, height, rowStride, rotationDegrees
LUMA_SUBHEADER_SIZE = struct.calcsize(LUMA_SUBHEADER_FMT)
assert LUMA_SUBHEADER_SIZE == 16

# Camera Module 3 (IMX708) native sensor aspect is 16:9 (4608x2592).
CAMERA_ASPECT_W = 16
CAMERA_ASPECT_H = 9

MIC_QUEUE_MAXSIZE = 1000  # ~30s at 512 samples/16kHz -- see module docstring
PLAYBACK_BUFFER_SECONDS = 2.0

LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def pack_header(seq: int) -> bytes:
    return struct.pack(HEADER_FMT, seq, time.time())


def send_framed(sock: "zmq.Socket", seq: int, payload: bytes) -> None:
    sock.send_multipart([pack_header(seq), payload])


def put_drop_oldest(q: "queue.Queue", item) -> None:
    """Non-blocking put that discards the oldest queued item on overflow."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def cv2_rotate_code(rotation_degrees: int):
    return {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(rotation_degrees % 360)


class RateStats:
    """Thread-safe per-stream throughput counter, logged periodically so a
    headless Pi's console/journal shows liveness without a display attached.
    """

    def __init__(self, label: str, log_every_s: float = 10.0):
        self.label = label
        self.log_every_s = log_every_s
        self._count = 0
        self._window_count = 0
        self._last_log = time.monotonic()
        self._lock = threading.Lock()

    def record(self) -> None:
        with self._lock:
            self._count += 1
            self._window_count += 1
            now = time.monotonic()
            elapsed = now - self._last_log
            if elapsed >= self.log_every_s:
                rate = self._window_count / elapsed
                logging.info("[stats] %-10s total=%-8d %.1f/s", self.label, self._count, rate)
                self._window_count = 0
                self._last_log = now


# ---------------------------------------------------------------------------
# ALSA device auto-detection (sounddevice / PortAudio)
# ---------------------------------------------------------------------------

def list_audio_devices() -> None:
    devices = sd.query_devices()
    logging.info("ALSA/PortAudio devices:")
    for i, d in enumerate(devices):
        logging.info(
            "  [%2d] %-40s in=%d out=%d default_rate=%.0fHz",
            i, d["name"], d["max_input_channels"], d["max_output_channels"], d["default_samplerate"],
        )


def _match_device(preferred: str, need_input: bool) -> int:
    devices = sd.query_devices()
    try:
        return int(preferred)
    except ValueError:
        pass
    for i, d in enumerate(devices):
        if preferred.lower() in d["name"].lower():
            channels = d["max_input_channels"] if need_input else d["max_output_channels"]
            if channels > 0:
                return i
    raise RuntimeError(f"No audio device matching '{preferred}' with the required channels found.")


def resolve_input_device(preferred):
    if preferred is not None:
        idx = _match_device(preferred, need_input=True)
        logging.info("[audio] using explicitly requested input device [%d] %s", idx, sd.query_devices(idx)["name"])
        return idx
    devices = sd.query_devices()
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0 and devices[default_in]["max_input_channels"] > 0:
            logging.info("[audio] using system default input device [%d] %s", default_in, devices[default_in]["name"])
            return default_in
    except Exception:
        pass
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            logging.info("[audio] auto-picked input device [%d] %s", i, d["name"])
            return i
    raise RuntimeError("No input-capable ALSA device found -- plug in the mic adapter and check `arecord -l`.")


def resolve_output_device(preferred):
    if preferred is not None:
        idx = _match_device(preferred, need_input=False)
        logging.info("[audio] using explicitly requested output device [%d] %s", idx, sd.query_devices(idx)["name"])
        return idx
    devices = sd.query_devices()
    try:
        default_out = sd.default.device[1]
        if default_out is not None and default_out >= 0 and devices[default_out]["max_output_channels"] > 0:
            logging.info("[audio] using system default output device [%d] %s", default_out, devices[default_out]["name"])
            return default_out
    except Exception:
        pass
    for i, d in enumerate(devices):
        if d["max_output_channels"] > 0:
            logging.info("[audio] auto-picked output device [%d] %s", i, d["name"])
            return i
    raise RuntimeError("No output-capable ALSA device found -- plug in the earphone adapter and check `aplay -l`.")


# ---------------------------------------------------------------------------
# Mic resampling (only exercised if the device can't open at 16kHz directly)
# ---------------------------------------------------------------------------

class MicResampler:
    """Buffers native-rate int16 mono samples and yields fixed-size chunks
    at the target rate. A no-op passthrough when native_rate == target_rate.

    Resamples in fixed-size blocks (not a continuous-state filter), so there
    is a tiny phase discontinuity at each block boundary -- the same
    accepted, inaudible-in-practice tradeoff this project's Android side
    already makes for its own per-chunk resampling (see AudioMixer.kt's
    resampleMonoToStereo).
    """

    RESAMPLE_BLOCK = 2048

    def __init__(self, native_rate: int, target_rate: int, chunk_samples: int):
        self.native_rate = native_rate
        self.target_rate = target_rate
        self.chunk_samples = chunk_samples
        self._resample = native_rate != target_rate
        self._native_buf = np.zeros(0, dtype=np.int16)
        self._out_buf = np.zeros(0, dtype=np.int16)
        if self._resample:
            import scipy.signal  # noqa: F401 -- import here to fail loudly only when actually needed

    def push(self, samples: np.ndarray) -> list:
        if not self._resample:
            self._out_buf = np.concatenate([self._out_buf, samples])
        else:
            import scipy.signal
            self._native_buf = np.concatenate([self._native_buf, samples])
            if len(self._native_buf) >= self.RESAMPLE_BLOCK:
                resampled = scipy.signal.resample_poly(
                    self._native_buf.astype(np.float32), self.target_rate, self.native_rate
                )
                self._out_buf = np.concatenate([self._out_buf, resampled.astype(np.int16)])
                self._native_buf = np.zeros(0, dtype=np.int16)

        chunks = []
        while len(self._out_buf) >= self.chunk_samples:
            chunks.append(self._out_buf[: self.chunk_samples].copy())
            self._out_buf = self._out_buf[self.chunk_samples:]
        return chunks


# ---------------------------------------------------------------------------
# Camera: one capture thread feeding two send threads
# ---------------------------------------------------------------------------

def camera_capture_thread(frame_raw_queue, luma_queue, stop_event, args, stats_frame_cap, stats_luma_cap):
    """Owns the single Picamera2 instance. Configures a dual stream --
    "main" (RGB888, for frame_out's JPEG) and "lores" (YUV420, for
    luma_out's raw Y-plane) -- and pulls both from the same capture request
    each cycle so the ISP does resolution scaling in hardware rather than
    software resize on this box's weak CPU. Never touches a ZMQ socket
    itself; just distributes frames into two drop-oldest queues for the
    sender threads below to encode/send at their own pace.
    """
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        logging.error(
            "picamera2 is not importable. On Raspberry Pi OS install it via "
            "'sudo apt install -y python3-picamera2' (it wraps system libcamera "
            "bindings, it is not a plain pip package). (%s)", e,
        )
        stop_event.set()
        return

    main_size = (args.frame_long_edge, round(args.frame_long_edge * CAMERA_ASPECT_H / CAMERA_ASPECT_W))
    lores_size = (args.luma_long_edge, round(args.luma_long_edge * CAMERA_ASPECT_H / CAMERA_ASPECT_W))
    frame_duration_us = int(1_000_000 / args.luma_fps) if args.luma_fps > 0 else 0

    backoff = 1.0
    while not stop_event.is_set():
        picam2 = None
        try:
            picam2 = Picamera2()
            controls = {"FrameDurationLimits": (frame_duration_us, frame_duration_us)} if frame_duration_us else {}
            config = picam2.create_video_configuration(
                # picamera2's "RGB888" format is stored in BGR byte order on
                # purpose, specifically so arrays can be handed to OpenCV
                # (cv2.imencode etc.) with no color conversion needed.
                main={"size": main_size, "format": "RGB888"},
                lores={"size": lores_size, "format": "YUV420"},
                controls=controls,
            )
            picam2.configure(config)
            picam2.start()

            lores_cfg = picam2.camera_configuration()["lores"]
            stride = lores_cfg["stride"]
            width, height = lores_cfg["size"]
            logging.info("camera started: main=%s lores=%s stride=%d", main_size, lores_size, stride)
            backoff = 1.0

            while not stop_event.is_set():
                request = picam2.capture_request()
                try:
                    capture_ts = time.time()
                    main_arr = request.make_array("main")
                    put_drop_oldest(frame_raw_queue, (main_arr, capture_ts))
                    stats_frame_cap.record()

                    # make_buffer gives the raw, stride-padded plane bytes --
                    # exactly what the wire format wants (height*rowStride
                    # bytes, row-major, padding included), so no reshape is
                    # needed: the Y-plane is simply the first stride*height
                    # bytes of a planar YUV420 buffer.
                    lores_buf = bytes(request.make_buffer("lores"))[: stride * height]
                    put_drop_oldest(luma_queue, (lores_buf, width, height, stride, args.rotation_degrees))
                    stats_luma_cap.record()
                finally:
                    request.release()
        except Exception:
            logging.exception("camera capture loop failed, retrying in %.1fs", backoff)
            if picam2 is not None:
                try:
                    picam2.close()
                except Exception:
                    pass
            time.sleep(backoff)
            backoff = min(backoff * 2, 15.0)
        else:
            if picam2 is not None:
                try:
                    picam2.stop()
                    picam2.close()
                except Exception:
                    pass


def frame_sender_thread(ctx, frame_raw_queue, stop_event, port, quality, fps, rotation_degrees, stats):
    """Owns the frame_out PUSH socket. Pulls the latest captured main-stream
    array, throttles to ~fps, rotates (see ROTATION NOTE in the module
    docstring), JPEG-encodes, and sends.
    """
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("frame_out PUSH bound on 0.0.0.0:%d", port)

    rotate_code = cv2_rotate_code(rotation_degrees)
    interval = 1.0 / fps if fps > 0 else 0.0
    next_send = time.monotonic()
    seq = 0
    try:
        while not stop_event.is_set():
            try:
                frame_bgr, _capture_ts = frame_raw_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.monotonic()
            if now < next_send:
                continue  # not our turn yet; a fresher frame will arrive shortly
            next_send = now + interval
            try:
                if rotate_code is not None:
                    frame_bgr = cv2.rotate(frame_bgr, rotate_code)
                ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if not ok:
                    logging.warning("JPEG encode failed, skipping this frame")
                    continue
                send_framed(sock, seq, jpg.tobytes())
                seq += 1
                stats.record()
            except Exception:
                logging.exception("frame_out send failed")
                time.sleep(0.1)
    finally:
        sock.close(0)


def luma_sender_thread(ctx, luma_queue, stop_event, port, stats):
    """Owns the luma_out PUSH socket. Packs the 16-byte sub-header ahead of
    the raw Y-plane bytes and forwards essentially every captured luma frame
    (already rate-limited upstream by the camera's FrameDurationLimits).
    """
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("luma_out PUSH bound on 0.0.0.0:%d", port)

    seq = 0
    try:
        while not stop_event.is_set():
            try:
                luma_bytes, width, height, stride, rotation = luma_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                sub_header = struct.pack(LUMA_SUBHEADER_FMT, width, height, stride, rotation)
                send_framed(sock, seq, sub_header + luma_bytes)
                seq += 1
                stats.record()
            except Exception:
                logging.exception("luma_out send failed")
                time.sleep(0.1)
    finally:
        sock.close(0)


# ---------------------------------------------------------------------------
# Mic capture -> mic_out
# ---------------------------------------------------------------------------

def mic_capture_thread(mic_queue, stop_event, args, stats_cap):
    """Owns the mic InputStream. Tries to open the resolved device directly
    at the target 16kHz; if the hardware only offers e.g. 44.1/48kHz (common
    on cheap USB audio dongles), falls back to the device's native rate and
    resamples down via MicResampler. Never blocks inside the audio callback
    (a hard real-time constraint) -- pushes onto mic_queue non-blocking and
    only drops-oldest as a last resort if ~30s has backed up, which should
    never happen under normal load on this light a workload.
    """
    device = resolve_input_device(args.input_device)
    resampler_box = {}

    def callback(indata, frames, time_info, status):
        if status:
            logging.warning("mic input status: %s", status)
        mono = indata[:, 0].copy()
        resampler = resampler_box.get("resampler")
        if resampler is None:
            return
        for chunk in resampler.push(mono):
            try:
                mic_queue.put_nowait(chunk.tobytes())
            except queue.Full:
                logging.warning("mic queue full (~30s buffered) -- dropping oldest chunk as a last resort")
                try:
                    mic_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    mic_queue.put_nowait(chunk.tobytes())
                except queue.Full:
                    pass
            stats_cap.record()

    backoff = 1.0
    while not stop_event.is_set():
        stream = None
        try:
            try:
                stream = sd.InputStream(
                    device=device, samplerate=args.mic_sample_rate, channels=1,
                    dtype="int16", blocksize=args.mic_chunk_samples, callback=callback,
                )
                stream.start()
                native_rate = args.mic_sample_rate
                logging.info("mic opened device=%s directly at target rate %dHz", device, native_rate)
            except Exception as e:
                logging.warning(
                    "mic device can't open at %dHz directly (%s) -- falling back to its native rate + resampling",
                    args.mic_sample_rate, e,
                )
                native_rate = int(round(sd.query_devices(device)["default_samplerate"]))
                stream = sd.InputStream(
                    device=device, samplerate=native_rate, channels=1,
                    dtype="int16", blocksize=0, callback=callback,
                )
                stream.start()
                logging.info("mic opened device=%s at native rate %dHz, resampling to %dHz",
                             device, native_rate, args.mic_sample_rate)

            resampler_box["resampler"] = MicResampler(native_rate, args.mic_sample_rate, args.mic_chunk_samples)
            backoff = 1.0
            while not stop_event.is_set():
                time.sleep(0.2)
        except Exception:
            logging.exception("mic capture failed, retrying in %.1fs", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 15.0)
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass


def mic_sender_thread(ctx, mic_queue, stop_event, port, stats):
    """Owns the mic_out PUSH socket. SNDTIMEO is set so a vanished peer
    times out the send instead of blocking this thread (and therefore the
    upstream mic_queue) forever.
    """
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 200)
    sock.setsockopt(zmq.SNDTIMEO, 2000)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("mic_out PUSH bound on 0.0.0.0:%d", port)

    seq = 0
    try:
        while not stop_event.is_set():
            try:
                chunk = mic_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                send_framed(sock, seq, chunk)
                seq += 1
                stats.record()
            except zmq.Again:
                logging.warning("mic_out send timed out (peer not consuming?) -- dropping this chunk")
            except Exception:
                logging.exception("mic_out send failed")
                time.sleep(0.1)
    finally:
        sock.close(0)


# ---------------------------------------------------------------------------
# audio_in -> speaker playback
# ---------------------------------------------------------------------------

class PlaybackBuffer:
    """Thread-safe byte ring fed by the audio_in recv thread and drained by
    the sounddevice output callback. Reads pad with digital silence when
    starved (arrival is bursty by design -- silence really does mean
    nothing to play), and the buffer is capped so a stalled output device
    can't grow it unboundedly -- unlike mic capture, dropping stale
    already-rendered playback audio under a sustained backlog is harmless.
    """

    def __init__(self, max_bytes: int):
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._max_bytes = max_bytes

    def append(self, data: bytes) -> None:
        with self._lock:
            self._buf.extend(data)
            if len(self._buf) > self._max_bytes:
                del self._buf[: len(self._buf) - self._max_bytes]

    def read(self, n_bytes: int) -> bytes:
        with self._lock:
            n = min(n_bytes, len(self._buf))
            out = bytes(self._buf[:n])
            del self._buf[:n]
        if len(out) < n_bytes:
            out += b"\x00" * (n_bytes - len(out))
        return out


def audio_in_recv_thread(ctx, playback_buffer, stop_event, port, stats):
    """Owns the audio_in PULL socket. Just appends whatever arrives into the
    shared playback buffer; the output stream's callback (a different
    thread) does the actual pacing.
    """
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 64)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("audio_in PULL bound on 0.0.0.0:%d", port)

    try:
        while not stop_event.is_set():
            try:
                _header, payload = sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception:
                logging.exception("audio_in recv failed")
                continue
            playback_buffer.append(payload)
            stats.record()
    finally:
        sock.close(0)


def audio_playback_thread(playback_buffer, stop_event, args):
    """Owns the speaker OutputStream. Runs continuously for the process's
    whole lifetime (not opened/closed per chunk) so consecutive chunks that
    DO arrive play back-to-back with no gap; the callback reads whatever is
    buffered and pads with zeros when nothing has arrived recently.
    """
    device = resolve_output_device(args.output_device)
    bytes_per_frame = args.audio_in_channels * 2  # int16

    def callback(outdata, frames, time_info, status):
        if status:
            logging.warning("audio_in output status: %s", status)
        data = playback_buffer.read(frames * bytes_per_frame)
        outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, args.audio_in_channels)

    backoff = 1.0
    while not stop_event.is_set():
        try:
            with sd.OutputStream(
                device=device, samplerate=args.audio_in_sample_rate,
                channels=args.audio_in_channels, dtype="int16", callback=callback,
            ):
                logging.info("audio_in playback stream open on device=%s", device)
                while not stop_event.is_set():
                    time.sleep(0.2)
            backoff = 1.0
        except Exception:
            logging.exception("audio_in playback stream failed, retrying in %.1fs", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 15.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame-port", type=int, default=5602)
    p.add_argument("--luma-port", type=int, default=5604)
    p.add_argument("--mic-port", type=int, default=5601)
    p.add_argument("--audio-in-port", type=int, default=5603)

    p.add_argument("--frame-long-edge", type=int, default=640)
    p.add_argument("--frame-quality", type=int, default=50)
    p.add_argument("--frame-fps", type=float, default=2.0)

    p.add_argument("--luma-long-edge", type=int, default=480)
    p.add_argument("--luma-fps", type=float, default=15.0)

    p.add_argument("--rotation-degrees", type=int, default=90, choices=[0, 90, 180, 270],
                    help="Physical camera mount rotation; drives frame_out's baked-in "
                         "pixel rotation and luma_out's sub-header identically.")

    p.add_argument("--mic-sample-rate", type=int, default=16000)
    p.add_argument("--mic-chunk-samples", type=int, default=512)

    p.add_argument("--audio-in-sample-rate", type=int, default=44100)
    p.add_argument("--audio-in-channels", type=int, default=2)

    p.add_argument("--input-device", default=None,
                    help="ALSA input device index or name substring (default: auto-pick)")
    p.add_argument("--output-device", default=None,
                    help="ALSA output device index or name substring (default: auto-pick)")
    p.add_argument("--list-audio-devices", action="store_true", help="List ALSA devices and exit")

    p.add_argument("--log-level", default="INFO")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format=LOG_FORMAT)

    if cv2 is None or sd is None:
        logging.error("Missing required dependency (opencv-python-headless and/or sounddevice). "
                       "Run: pip install -r requirements.txt")
        sys.exit(1)

    if args.list_audio_devices:
        list_audio_devices()
        return

    stop_event = threading.Event()

    def handle_signal(signum, _frame):
        logging.info("received signal %s, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    ctx = zmq.Context()

    frame_raw_queue: "queue.Queue" = queue.Queue(maxsize=1)
    luma_queue: "queue.Queue" = queue.Queue(maxsize=1)
    mic_queue: "queue.Queue" = queue.Queue(maxsize=MIC_QUEUE_MAXSIZE)
    playback_buffer = PlaybackBuffer(
        max_bytes=int(args.audio_in_sample_rate * args.audio_in_channels * 2 * PLAYBACK_BUFFER_SECONDS)
    )

    stat_names = ("frame_cap", "luma_cap", "frame_out", "luma_out", "mic_cap", "mic_out", "audio_in")
    stats = {name: RateStats(name) for name in stat_names}

    threads = [
        threading.Thread(
            target=camera_capture_thread,
            args=(frame_raw_queue, luma_queue, stop_event, args, stats["frame_cap"], stats["luma_cap"]),
            name="camera-capture", daemon=True,
        ),
        threading.Thread(
            target=frame_sender_thread,
            args=(ctx, frame_raw_queue, stop_event, args.frame_port, args.frame_quality,
                  args.frame_fps, args.rotation_degrees, stats["frame_out"]),
            name="frame-sender", daemon=True,
        ),
        threading.Thread(
            target=luma_sender_thread,
            args=(ctx, luma_queue, stop_event, args.luma_port, stats["luma_out"]),
            name="luma-sender", daemon=True,
        ),
        threading.Thread(
            target=mic_capture_thread,
            args=(mic_queue, stop_event, args, stats["mic_cap"]),
            name="mic-capture", daemon=True,
        ),
        threading.Thread(
            target=mic_sender_thread,
            args=(ctx, mic_queue, stop_event, args.mic_port, stats["mic_out"]),
            name="mic-sender", daemon=True,
        ),
        threading.Thread(
            target=audio_in_recv_thread,
            args=(ctx, playback_buffer, stop_event, args.audio_in_port, stats["audio_in"]),
            name="audio-in-recv", daemon=True,
        ),
        threading.Thread(
            target=audio_playback_thread,
            args=(playback_buffer, stop_event, args),
            name="audio-playback", daemon=True,
        ),
    ]

    for t in threads:
        t.start()
    logging.info(
        "all threads started -- frame_out:%d luma_out:%d mic_out:%d audio_in:%d",
        args.frame_port, args.luma_port, args.mic_port, args.audio_in_port,
    )

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=3.0)
        ctx.term()
        logging.info("shutdown complete")


if __name__ == "__main__":
    main()
