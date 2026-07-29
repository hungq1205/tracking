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
   Raw JPEG bytes, at the SENSOR'S OWN FULL NATIVE RESOLUTION (e.g. 4608x2592
   for the IMX708 -- picam2.sensor_resolution, see camera_capture_thread),
   quality ~50, ~2 fps. Independent of luma_out's own (much smaller)
   resolution. Physically rotated (see ROTATION NOTE below) -- NOT just
   tagged with rotation metadata, because unlike luma_out this channel
   carries no rotation field at all, and the Android side decodes it
   straight into a Bitmap with no compensation applied anywhere
   downstream. NOTE: the Android device's own screen aspect ratio is very
   unlikely to match the sensor's (e.g. 16:9 vs. a phone's own aspect) --
   see MainScreen.kt's ContentScale.Fit, which shows the WHOLE frame
   letterboxed rather than cropping any of it away.

2. luma_out (PUSH, port 5604, us -> Android)
   16-byte sub-header + raw Y-plane bytes:
       bytes 0-3   int32  width
       bytes 4-7   int32  height
       bytes 8-11  int32  rowStride
       bytes 12-15 int32  rotationDegrees (0/90/180/270)
   followed by height*rowStride bytes of 8-bit grayscale, row-major,
   RAW (not rotated -- Android's AngleTracker rotates it itself using the
   rotationDegrees field). 360px long edge, ~15 fps.

3. mic_out (PUSH, port 5601, us -> Android)
   Raw PCM16 little-endian, mono, 16000 Hz. Recommended 512-sample
   (1024-byte) chunks, continuous. Digital gain (--mic-gain, default 4.0x)
   is applied before sending -- see mic_capture_thread's docstring for why
   (raw ALSA capture has no AGC, unlike the phone's own local mic path).

4. audio_in (PULL, port 5603, Android -> us)
   Raw PCM16 little-endian, STEREO interleaved, 44100 Hz. Variable chunk
   sizes; arrival is bursty (silence = no messages at all), so playback
   pads with digital silence rather than blocking/gapping when nothing has
   arrived recently.

5. control (PULL, port 5605, Android -> us) -- NOT part of the original
   fixed 4-socket spec; added so the phone can report its current mode
   (piggybacked on the same reportMode() call already fired on every mode
   change, see RemoteEdgeDevice.kt). Payload is just a UTF-8 mode string
   ("walking"/"guiding"/"idle"/etc, same 16-byte header framing as the other
   4 sockets). Only ever used to gate luma_out sending -- see
   control_recv_thread's own docstring. Silence here (an older Android
   build that never sends anything) degrades to today's always-on luma_out
   behavior, not silent data loss.

6. ocr_request (PULL, port 5606, Android -> us) -- triggers a one-shot
   FULL-RESOLUTION still capture, for OCR specifically. Payload is empty
   (a pure trigger, same 16-byte header framing as everything else) --
   see RemoteEdgeDevice.kt's requestOcrFrame(). frame_out/luma_out are
   deliberately low-res (tied to --luma-long-edge) for bandwidth/latency's
   sake on every OTHER consumer (tracking, mapping, hazard checks); OCR
   needs real detail that resolution can't give it, so it gets its own
   on-demand channel instead of raising the continuous streams' resolution
   for everyone.

7. ocr_frame_out (PUSH, port 5607, us -> Android) -- the JPEG result of
   the ocr_request above, at the sensor's own FULL native resolution
   (picam2.sensor_resolution -- NOT --luma-long-edge), same rotation
   handling as frame_out. While a request is being handled,
   frame_out/luma_out sending is PAUSED (see StreamPauseGate,
   ocr_request_thread, camera_capture_thread's OCR branch, and
   ocr_frame_sender_thread) -- both to give the still capture the camera's
   full attention (Picamera2 can't do two capture-mode things at once) and
   to keep the full-res JPEG's own send from competing for bandwidth with
   the continuous streams. Auto-resumes even if something goes wrong
   (encode failure, no Android peer connected, an unexpected exception) --
   see StreamPauseGate's own docstring for the hard timeout backstop.

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
PLAYBACK_BUFFER_SECONDS = 2.0  # hard backstop against a genuinely stalled output device

# Adaptive jitter-buffer tuning for audio_in playback (see PlaybackBuffer) --
# only the initial/floor target is CLI-exposed (--audio-in-jitter-ms); these
# secondary knobs are reasonable fixed defaults, not yet independently tunable.
JITTER_MIN_MS = 40.0
JITTER_MAX_MS = 500.0
JITTER_GROW_MS = 60.0
JITTER_SHRINK_MS = 30.0
JITTER_SHRINK_AFTER_S = 20.0

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


class StreamPauseGate:
    """Pauses frame_out/luma_out SENDING while a full-resolution OCR still
    capture (ocr_request -> ocr_frame_out) is in flight -- see
    camera_capture_thread's OCR branch (which calls pause()) and
    ocr_frame_sender_thread (which calls resume() once the JPEG has
    actually been pushed onto the wire, success or failure). frame_sender_
    thread/luma_sender_thread check is_paused() at the top of their loop
    and simply skip sending while it's true (their queues are already
    drop-oldest, so nothing backs up -- they just resume with whatever's
    freshest once unpaused).

    Auto-expires after `max_pause_s` regardless of whether resume() was
    ever called, so an unexpected crash/hang while handling one OCR
    request can't permanently stall the live video stream -- a live view
    resuming (possibly slightly early, mid-send) is far preferable to it
    silently dying for good.
    """

    def __init__(self, max_pause_s: float = 8.0):
        self._max_pause_s = max_pause_s
        self._paused_since: float | None = None
        self._lock = threading.Lock()

    def pause(self) -> None:
        with self._lock:
            self._paused_since = time.monotonic()

    def resume(self) -> None:
        with self._lock:
            self._paused_since = None

    def is_paused(self) -> bool:
        with self._lock:
            started = self._paused_since
        if started is None:
            return False
        if time.monotonic() - started > self._max_pause_s:
            return False
        return True


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

def camera_capture_thread(frame_raw_queue, luma_queue, stop_event, args, stats_frame_cap, stats_luma_cap, luma_enabled,
                           ocr_request_queue, ocr_result_queue, stream_pause):
    """Owns the single Picamera2 instance. Configures a dual stream --
    "main" (RGB888, for frame_out's JPEG) and "lores" (YUV420, for
    luma_out's raw Y-plane) -- and pulls both from the same capture request
    each cycle so the ISP does resolution scaling in hardware rather than
    software resize on this box's weak CPU. Never touches a ZMQ socket
    itself; just distributes frames into two drop-oldest queues for the
    sender threads below to encode/send at their own pace.

    [luma_enabled] (a threading.Event, see control_recv_thread) gates only
    the lores buffer copy + queue put below, NOT the dual-stream capture
    config itself -- picamera2/libcamera still produces both streams every
    request regardless, since dropping a stream requires a stop/reconfigure/
    start cycle that would glitch every mode transition. This still saves
    the per-frame buffer copy and, more importantly, the actual network
    send (luma_out's only real consumer is AngleTracker, walking/guiding-
    only -- see ToolDispatcher.feedAngleLumaFrame() -- so every other mode
    was paying full 15fps/~130KB-per-frame network cost for data the phone
    just discarded).

    Also owns the OCR full-resolution still capture (see ocr_request_thread/
    ocr_frame_sender_thread) -- MUST happen on this same thread, not a
    separate one, since concurrent capture_request()/switch_mode calls
    against one Picamera2 instance from two different threads is not a
    pattern picamera2 supports (see the module docstring's threading
    model). Checked non-blockingly once per normal video-capture iteration,
    so it interleaves with (rather than interrupts) the regular loop below.
    """
    try:
        from picamera2 import Picamera2
        from libcamera import controls as libcamera_controls
    except ImportError as e:
        logging.error(
            "picamera2 is not importable. On Raspberry Pi OS install it via "
            "'sudo apt install -y python3-picamera2' (it wraps system libcamera "
            "bindings, it is not a plain pip package). (%s)", e,
        )
        stop_event.set()
        return

    # lores (luma_out, only consumed locally by AngleTracker's ORB rotation
    # tracking) keeps its own modest resolution -- more pixels there is more
    # ORB-detection work per frame with no benefit to what's displayed.
    lores_size = (args.luma_long_edge, round(args.luma_long_edge * CAMERA_ASPECT_H / CAMERA_ASPECT_W))
    frame_duration_us = int(1_000_000 / args.luma_fps) if args.luma_fps > 0 else 0
    # Same rotation this thread applies to nothing itself normally (that's
    # frame_sender_thread's job for the continuous stream) -- needed here
    # too for the OCR still branch below, since that JPEG is built directly
    # in THIS thread, not handed off to frame_sender_thread.
    rotate_code = cv2_rotate_code(args.rotation_degrees)

    backoff = 1.0
    while not stop_event.is_set():
        picam2 = None
        try:
            picam2 = Picamera2()
            # Continuous AF for the normal video stream -- the IMX708 (Camera
            # Module 3) has autofocus hardware, but picamera2 doesn't drive it
            # at all unless AfMode is explicitly set. Without this, the lens
            # sits wherever it last was (often a default/hyperfocal distance),
            # which is exactly why a document held close for OCR came back
            # genuinely out of focus regardless of how steady it was held --
            # steadiness only helps with motion blur, not wrong focus
            # distance. This keeps the lens tracking whatever's in frame
            # during ordinary streaming; the OCR still branch below ALSO
            # explicitly triggers+waits for a fresh autofocus_cycle() of its
            # own, since switch_mode() resets into a fresh capture session and
            # continuous AF may not have re-converged yet at the instant of
            # that mode switch.
            controls = {
                "FrameDurationLimits": (frame_duration_us, frame_duration_us),
                "AfMode": libcamera_controls.AfModeEnum.Continuous,
            } if frame_duration_us else {"AfMode": libcamera_controls.AfModeEnum.Continuous}
            # main (frame_out, what the Android client displays) is
            # requested at the SENSOR'S OWN FULL NATIVE RESOLUTION directly
            # (e.g. 4608x2592 for the IMX708) -- --frame-long-edge is gone;
            # there's no separate "raw" stream pin any more either (an
            # earlier round used one purely to steer libcamera's sensor-
            # mode selection toward full-FOV while `main` itself stayed
            # small -- now that `main` IS the full-resolution request,
            # nothing else needs to be pinned: requesting the sensor's own
            # native size for `main` already forces the full-FOV mode by
            # construction, no separate hint required). `main` and `lores`
            # no longer share one aspect-locked pair either, since `main`'s
            # size now comes straight from the sensor, not
            # CAMERA_ASPECT_W/H arithmetic.
            try:
                main_size = picam2.sensor_resolution
            except Exception:
                # Fallback only if the sensor's own resolution can't be
                # queried at all (shouldn't happen on real hardware) --
                # matches CAMERA_ASPECT_W/H's assumed 16:9 at a size that's
                # at least still a real request, not a crash.
                main_size = (args.luma_long_edge * 4, round(args.luma_long_edge * 4 * CAMERA_ASPECT_H / CAMERA_ASPECT_W))
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
            logging.info("camera started: main=%s (sensor full res) lores=%s stride=%d", main_size, lores_size, stride)
            backoff = 1.0

            while not stop_event.is_set():
                try:
                    ocr_req = ocr_request_queue.get_nowait()
                except queue.Empty:
                    ocr_req = None
                if ocr_req is not None:
                    # Manually split what switch_mode_and_capture_array() would
                    # otherwise do in one call (switch -> capture -> switch
                    # back), specifically to insert a blocking autofocus_cycle()
                    # in between the mode switch and the actual capture. This
                    # is the fix for a real reported bug: an OCR still of a
                    # document held close up came back genuinely blurry even
                    # when held perfectly steady -- switch_mode() starts a
                    # fresh capture session, and continuous AF (see the
                    # AfMode control set above) has no guarantee of having
                    # already re-converged for the new distance at the exact
                    # instant a bare capture would otherwise fire. Captures
                    # ONE still at the still config's own resolution (full
                    # sensor resolution here -- no "size" key means picamera2
                    # defaults to the sensor's max). Normal main/lores CAPTURE
                    # is naturally paused for this whole window since it's the
                    # same camera/thread; the stream_pause gate below
                    # additionally blocks SENDING any already-queued main/lores
                    # frame while this is in flight.
                    stream_pause.pause()
                    try:
                        logging.info("OCR full-res capture requested -- switching camera mode")
                        still_config = picam2.create_still_configuration(main={"format": "RGB888"})
                        picam2.switch_mode(still_config)
                        try:
                            try:
                                focused = picam2.autofocus_cycle()
                                if not focused:
                                    logging.warning("OCR still: autofocus_cycle() did not converge -- capturing anyway")
                            except Exception:
                                logging.exception("OCR still: autofocus_cycle() failed -- capturing anyway")
                            still_arr = picam2.capture_array("main")
                        finally:
                            # Always restore the video config, even if autofocus
                            # or the capture itself raised -- leaving picam2
                            # parked in still mode would silently break every
                            # subsequent frame_out/lores capture for the rest
                            # of this camera session.
                            picam2.switch_mode(config)
                        if rotate_code is not None:
                            still_arr = cv2.rotate(still_arr, rotate_code)
                        ok, jpg = cv2.imencode(
                            ".jpg", still_arr, [int(cv2.IMWRITE_JPEG_QUALITY), args.ocr_frame_quality],
                        )
                        if ok:
                            put_drop_oldest(ocr_result_queue, jpg.tobytes())
                            logging.info(
                                "OCR full-res capture done: %s -> %d JPEG bytes",
                                still_arr.shape, len(jpg.tobytes()),
                            )
                            # stream_pause.resume() is NOT called here on
                            # success -- ocr_frame_sender_thread owns clearing
                            # it, once the JPEG has actually been sent (not
                            # merely captured/encoded) -- see that thread's
                            # own docstring for why "done" means "sent."
                        else:
                            logging.warning("OCR full-res JPEG encode failed")
                            stream_pause.resume()
                    except Exception:
                        logging.exception("OCR full-res capture failed")
                        stream_pause.resume()
                    continue

                request = picam2.capture_request()
                try:
                    capture_ts = time.time()
                    main_arr = request.make_array("main")
                    put_drop_oldest(frame_raw_queue, (main_arr, capture_ts))
                    stats_frame_cap.record()

                    if luma_enabled.is_set():
                        # make_buffer gives the raw, stride-padded plane bytes
                        # -- exactly what the wire format wants (height*
                        # rowStride bytes, row-major, padding included), so no
                        # reshape is needed: the Y-plane is simply the first
                        # stride*height bytes of a planar YUV420 buffer.
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


def frame_sender_thread(ctx, frame_raw_queue, stop_event, port, quality, fps, rotation_degrees, stats, stream_pause):
    """Owns the frame_out PUSH socket. Pulls the latest captured main-stream
    array, throttles to ~fps, rotates (see ROTATION NOTE in the module
    docstring), JPEG-encodes, and sends.

    Skips sending entirely while `stream_pause.is_paused()` (a full-res OCR
    still capture is in flight, see StreamPauseGate/ocr_request_thread) --
    frame_raw_queue is drop-oldest anyway, so nothing backs up; it just
    resumes with whatever's freshest once unpaused.
    """
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 2)
    # Without this, send() blocks indefinitely once no peer is connected and
    # the HWM-bounded queue fills -- confirmed live: with nobody pulling,
    # this thread hung forever inside send_multipart() and never logged
    # another frame again. A short timeout + dropping the frame on zmq.Again
    # matches this stream's own "a fresher frame beats a stale one" policy.
    sock.setsockopt(zmq.SNDTIMEO, 300)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("frame_out PUSH bound on 0.0.0.0:%d", port)

    rotate_code = cv2_rotate_code(rotation_degrees)
    interval = 1.0 / fps if fps > 0 else 0.0
    next_send = time.monotonic()
    seq = 0
    try:
        while not stop_event.is_set():
            if stream_pause.is_paused():
                time.sleep(0.02)
                continue
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
            except zmq.Again:
                logging.debug("frame_out send timed out (no peer connected yet?) -- dropping this frame")
            except Exception:
                logging.exception("frame_out send failed")
                time.sleep(0.1)
    finally:
        sock.close(0)


def luma_sender_thread(ctx, luma_queue, stop_event, port, stats, stream_pause):
    """Owns the luma_out PUSH socket. Packs the 16-byte sub-header ahead of
    the raw Y-plane bytes and forwards essentially every captured luma frame
    (already rate-limited upstream by the camera's FrameDurationLimits).

    Skips sending entirely while `stream_pause.is_paused()` -- see
    frame_sender_thread's own comment on why.
    """
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.setsockopt(zmq.SNDTIMEO, 300)  # see frame_sender_thread's comment on why this is required, not optional
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("luma_out PUSH bound on 0.0.0.0:%d", port)

    seq = 0
    try:
        while not stop_event.is_set():
            if stream_pause.is_paused():
                time.sleep(0.02)
                continue
            try:
                luma_bytes, width, height, stride, rotation = luma_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                sub_header = struct.pack(LUMA_SUBHEADER_FMT, width, height, stride, rotation)
                send_framed(sock, seq, sub_header + luma_bytes)
                seq += 1
                stats.record()
            except zmq.Again:
                logging.debug("luma_out send timed out (no peer connected yet?) -- dropping this frame")
            except Exception:
                logging.exception("luma_out send failed")
                time.sleep(0.1)
    finally:
        sock.close(0)


# ---------------------------------------------------------------------------
# Control channel: the phone reports its current mode, we gate luma_out
# ---------------------------------------------------------------------------

def control_recv_thread(ctx, luma_enabled, stop_event, port):
    """Owns the control PULL socket. The phone pushes its current mode
    string here on every transition (see RemoteEdgeDevice.kt's reportMode(),
    piggybacked on the same reportMode() call ToolDispatcher.kt already made
    on every mode change for the dashboard) -- this closes the gap
    RemoteEdgeDevice.kt's own docstring flagged ("No dynamic frame-rate/
    resolution control channel yet"). Only ever flips [luma_enabled] (a
    threading.Event shared with camera_capture_thread): luma_out's only real
    consumer is AngleTracker, active only during walking/guiding (see
    ToolDispatcher.feedAngleLumaFrame()) -- every other mode doesn't need it
    sent at all.

    [luma_enabled] starts SET (see main()) so an older Android build that
    never sends a mode report here degrades to today's always-on behavior,
    not silent data loss -- same backward-compatible-by-default precedent
    this wire protocol already uses elsewhere (e.g. rtabmap_client.py's
    node_id/inlier_fraction fallback).
    """
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 8)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("control PULL bound on 0.0.0.0:%d", port)

    try:
        while not stop_event.is_set():
            try:
                _header, payload = sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception:
                logging.exception("control recv failed")
                continue
            mode = payload.decode("utf-8", errors="replace")
            wants_luma = mode in ("walking", "guiding")
            if wants_luma and not luma_enabled.is_set():
                luma_enabled.set()
                logging.info("control: mode='%s' -> luma_out ENABLED", mode)
            elif not wants_luma and luma_enabled.is_set():
                luma_enabled.clear()
                logging.info("control: mode='%s' -> luma_out DISABLED (camera capture continues, just not sent)", mode)
    finally:
        sock.close(0)


# ---------------------------------------------------------------------------
# Full-resolution OCR capture: ocr_request -> (camera_capture_thread) -> ocr_frame_out
# ---------------------------------------------------------------------------

def ocr_request_thread(ctx, ocr_request_queue, stop_event, port):
    """Owns the ocr_request PULL socket. Payload is empty -- this is a pure
    trigger (see RemoteEdgeDevice.kt's requestOcrFrame()), nothing to parse.
    Hands the request off to camera_capture_thread (the only thread allowed
    to touch the Picamera2 instance -- see the module docstring's threading
    model) via a maxsize=1 drop-oldest queue: a second request arriving
    before the first has been picked up just coalesces into "handle one full-
    res capture soon," not two queued captures back to back.
    """
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 4)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("ocr_request PULL bound on 0.0.0.0:%d", port)

    try:
        while not stop_event.is_set():
            try:
                sock.recv_multipart()
            except zmq.Again:
                continue
            except Exception:
                logging.exception("ocr_request recv failed")
                continue
            put_drop_oldest(ocr_request_queue, True)
            logging.info("ocr_request received -- queued for camera_capture_thread")
    finally:
        sock.close(0)


def ocr_frame_sender_thread(ctx, ocr_result_queue, stop_event, port, stats, stream_pause):
    """Owns the ocr_frame_out PUSH socket. Pulls a completed full-res JPEG
    (put there by camera_capture_thread's OCR branch) and sends it -- a
    LONGER SNDTIMEO than the continuous streams (this is a rare, important,
    already-paced-by-request payload, not a droppable video frame) and
    always calls stream_pause.resume() in `finally`, success or failure, so
    a send that fails (no Android peer connected, etc.) can't leave
    frame_out/luma_out paused until StreamPauseGate's own hard timeout.
    """
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.bind(f"tcp://0.0.0.0:{port}")
    logging.info("ocr_frame_out PUSH bound on 0.0.0.0:%d", port)

    seq = 0
    try:
        while not stop_event.is_set():
            try:
                jpg_bytes = ocr_result_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                send_framed(sock, seq, jpg_bytes)
                seq += 1
                stats.record()
                logging.info("ocr_frame_out sent (%d bytes)", len(jpg_bytes))
            except zmq.Again:
                logging.warning("ocr_frame_out send timed out (no Android peer connected?) -- dropping this capture")
            except Exception:
                logging.exception("ocr_frame_out send failed")
            finally:
                stream_pause.resume()
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

    Applies a fixed digital gain (--mic-gain, default 4.0) before anything
    else touches the samples. Real, confirmed cause behind "have to shout
    for the edge device to hear me": the phone's own local mic path
    (ContinuousVadRecorder.kt) captures via AudioSource.VOICE_COMMUNICATION,
    which most Android hardware boosts with HAL/driver-level AGC before the
    app ever sees a sample -- the VAD thresholds in Settings (default
    noiseGate=0.012/startThreshold=0.018) were tuned against THAT already-
    boosted level. A bare ALSA capture here (plain sounddevice.InputStream,
    no AGC at all -- cheap USB/HAT mic capsules in particular run quiet) has
    no equivalent boost, so the identical thresholds effectively require
    much louder real-world speech to cross. Gain is applied here (source),
    not client-side, so it also benefits mic_out's actual audio quality/
    intelligibility for Gemini, not just the VAD's amplitude reading.
    """
    device = resolve_input_device(args.input_device)
    resampler_box = {}
    gain = args.mic_gain

    def callback(indata, frames, time_info, status):
        if status:
            logging.warning("mic input status: %s", status)
        mono = indata[:, 0]
        if gain != 1.0:
            mono = np.clip(mono.astype(np.int32) * gain, -32768, 32767).astype(np.int16)
        else:
            mono = mono.copy()
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
    the sounddevice output callback, with a small adaptive jitter buffer on
    top of the plain ring.

    Naively starting playback the instant any byte arrives (the original
    design) meant a normal momentary gap between network packets -- Wi-Fi
    jitter, AudioMixer.kt's own ~46ms tick-batching on the Android side --
    showed up as an audible stutter (confirmed live: repeated "output
    underflow" warnings). Real-time streaming players solve this with a
    small PREBUFFER: don't start pulling real audio out until at least
    `target_bytes` have accumulated, trading a little latency for
    smoothness. `target_bytes` starts at `initial_bytes` and adapts:

    - Grows (by `grow_bytes`, capped at `max_bytes_target`) when underruns
      happen REPEATEDLY in a short window (`_UNDERRUN_WINDOW_S`,
      `_UNDERRUN_GROW_THRESHOLD` occurrences) -- deliberately NOT on every
      single drain-to-empty event, because a normal conversational pause
      (Gemini stops talking, buffer empties, nothing more is coming for a
      while) looks identical to a real jitter underrun from a single event
      alone; only a cluster of them close together is real evidence of
      network jitter worth reacting to.
    - Shrinks (by `shrink_bytes`, floored at `min_bytes_target`) after a
      long stretch (`shrink_after_s`) with ZERO underruns at all, so
      latency drifts back down once the network's behaving rather than
      staying pinned at whatever it grew to during a rough patch.

    The hard `max_bytes` cap (unchanged from before) is a separate, much
    larger backstop against a genuinely stalled output device -- unrelated
    to the jitter-buffer target, which stays far below it.
    """

    _UNDERRUN_WINDOW_S = 5.0
    _UNDERRUN_GROW_THRESHOLD = 2

    def __init__(self, max_bytes: int, initial_target_bytes: int,
                 min_target_bytes: int, max_target_bytes: int,
                 grow_bytes: int, shrink_bytes: int, shrink_after_s: float):
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._target_bytes = initial_target_bytes
        self._min_target_bytes = min_target_bytes
        self._max_target_bytes = max_target_bytes
        self._grow_bytes = grow_bytes
        self._shrink_bytes = shrink_bytes
        self._shrink_after_s = shrink_after_s
        self._primed = False
        self._underrun_times: list = []
        self._last_underrun_at = time.monotonic()

    def append(self, data: bytes) -> None:
        with self._lock:
            self._buf.extend(data)
            if len(self._buf) > self._max_bytes:
                del self._buf[: len(self._buf) - self._max_bytes]
            if not self._primed and len(self._buf) >= self._target_bytes:
                self._primed = True

    def read(self, n_bytes: int) -> bytes:
        with self._lock:
            if not self._primed:
                # Still accumulating the initial cushion (or re-cushioning
                # after a real underrun) -- silence, not counted as a new
                # underrun itself, so priming-in-progress doesn't compound.
                return b"\x00" * n_bytes
            n = min(n_bytes, len(self._buf))
            out = bytes(self._buf[:n])
            del self._buf[:n]
            starved = n < n_bytes
            if starved:
                self._primed = False  # ran dry mid-stream -> re-buffer before resuming
            self._adapt(starved)
        if len(out) < n_bytes:
            out += b"\x00" * (n_bytes - len(out))
        return out

    def _adapt(self, starved: bool) -> None:
        # Caller already holds self._lock.
        now = time.monotonic()
        if starved:
            self._underrun_times.append(now)
            self._underrun_times = [t for t in self._underrun_times if now - t <= self._UNDERRUN_WINDOW_S]
            self._last_underrun_at = now
            if len(self._underrun_times) >= self._UNDERRUN_GROW_THRESHOLD and self._target_bytes < self._max_target_bytes:
                self._target_bytes = min(self._target_bytes + self._grow_bytes, self._max_target_bytes)
                self._underrun_times.clear()  # this cluster has already been "spent" on one grow step
                logging.info("audio_in jitter buffer grew to target=%dB after repeated underruns", self._target_bytes)
        elif (now - self._last_underrun_at >= self._shrink_after_s
              and self._target_bytes > self._min_target_bytes):
            self._target_bytes = max(self._target_bytes - self._shrink_bytes, self._min_target_bytes)
            self._last_underrun_at = now  # restart the stable-stretch timer for the next, smaller step
            logging.info("audio_in jitter buffer shrank to target=%dB after a stable stretch", self._target_bytes)


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
    p.add_argument("--control-port", type=int, default=5605,
                    help="Phone -> Pi mode reports, used only to gate luma_out sending (see control_recv_thread).")
    p.add_argument("--ocr-request-port", type=int, default=5606,
                    help="Phone -> Pi trigger for a one-shot full-resolution OCR still capture.")
    p.add_argument("--ocr-frame-port", type=int, default=5607,
                    help="Pi -> Phone JPEG result of --ocr-request-port's trigger, at full sensor resolution.")
    p.add_argument("--ocr-frame-quality", type=int, default=85,
                    help="JPEG quality for the full-resolution OCR still (higher than --frame-quality's "
                         "50 -- OCR wants real detail, and this channel is on-demand/infrequent, not "
                         "continuous, so the extra bytes-per-frame cost doesn't matter the way it would "
                         "for frame_out/luma_out.")

    # frame_out (main, RGB888/JPEG -- what the Android client actually
    # displays) has its OWN resolution, separate from --luma-long-edge
    # (lores, only consumed locally by AngleTracker's ORB tracking) -- see
    # camera_capture_thread's own comment on why these were un-tied again
    # after a previous round briefly merged them.
    # No --frame-long-edge any more: frame_out's (main) resolution is always
    # the sensor's own full native resolution now -- see
    # camera_capture_thread's own comment.
    p.add_argument("--frame-quality", type=int, default=50)
    p.add_argument("--frame-fps", type=float, default=2.0)

    p.add_argument("--luma-long-edge", type=int, default=360,
                    help="Long-edge resolution for luma_out (Y-plane, AngleTracker's ORB rotation "
                         "tracking input during walking/guiding only) -- independent of --frame-long-edge "
                         "above. Kept modest on purpose: more pixels here means more ORB-detection work "
                         "per frame with no benefit to what the Android client actually displays.")
    p.add_argument("--luma-fps", type=float, default=13.0,
                    help="Was 15.0 -- lowered to stay under the IMX708's own ~14fps full-FOV sensor "
                         "mode ceiling. Above that, libcamera's automatic sensor-mode selection can "
                         "silently switch to a narrower, cropped-FOV high-speed mode instead -- raise "
                         "this only if you don't mind trading FOV for frame rate.")

    p.add_argument("--rotation-degrees", type=int, default=90, choices=[0, 90, 180, 270],
                    help="Physical camera mount rotation; drives frame_out's baked-in "
                         "pixel rotation and luma_out's sub-header identically.")

    p.add_argument("--mic-sample-rate", type=int, default=16000)
    p.add_argument("--mic-chunk-samples", type=int, default=512)
    p.add_argument("--mic-gain", type=float, default=4.0,
                    help="Digital gain applied to captured mic samples before resampling/sending "
                         "(1.0 = off). Raw ALSA capture has no AGC, unlike the phone's own local mic "
                         "path -- see mic_capture_thread's docstring. Raise if the app still requires "
                         "shouting; lower if audio sounds clipped/distorted.")

    p.add_argument("--audio-in-sample-rate", type=int, default=44100)
    p.add_argument("--audio-in-channels", type=int, default=2)
    p.add_argument("--audio-in-jitter-ms", type=float, default=100.0,
                    help="Initial/floor playback prebuffer, in ms of audio, absorbing network jitter "
                         "before the speaker starts pulling real audio -- grows automatically on "
                         "repeated underruns and shrinks back down after a stable stretch.")

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
    ocr_request_queue: "queue.Queue" = queue.Queue(maxsize=1)
    ocr_result_queue: "queue.Queue" = queue.Queue(maxsize=1)
    stream_pause = StreamPauseGate()
    luma_enabled = threading.Event()
    luma_enabled.set()  # default on -- see control_recv_thread's own docstring for why
    bytes_per_ms = args.audio_in_sample_rate * args.audio_in_channels * 2 / 1000.0
    playback_buffer = PlaybackBuffer(
        max_bytes=int(args.audio_in_sample_rate * args.audio_in_channels * 2 * PLAYBACK_BUFFER_SECONDS),
        initial_target_bytes=int(args.audio_in_jitter_ms * bytes_per_ms),
        min_target_bytes=int(JITTER_MIN_MS * bytes_per_ms),
        max_target_bytes=int(JITTER_MAX_MS * bytes_per_ms),
        grow_bytes=int(JITTER_GROW_MS * bytes_per_ms),
        shrink_bytes=int(JITTER_SHRINK_MS * bytes_per_ms),
        shrink_after_s=JITTER_SHRINK_AFTER_S,
    )

    stat_names = ("frame_cap", "luma_cap", "frame_out", "luma_out", "mic_cap", "mic_out", "audio_in", "ocr_frame_out")
    stats = {name: RateStats(name) for name in stat_names}

    threads = [
        threading.Thread(
            target=camera_capture_thread,
            args=(frame_raw_queue, luma_queue, stop_event, args, stats["frame_cap"], stats["luma_cap"], luma_enabled,
                  ocr_request_queue, ocr_result_queue, stream_pause),
            name="camera-capture", daemon=True,
        ),
        threading.Thread(
            target=control_recv_thread,
            args=(ctx, luma_enabled, stop_event, args.control_port),
            name="control-recv", daemon=True,
        ),
        threading.Thread(
            target=frame_sender_thread,
            args=(ctx, frame_raw_queue, stop_event, args.frame_port, args.frame_quality,
                  args.frame_fps, args.rotation_degrees, stats["frame_out"], stream_pause),
            name="frame-sender", daemon=True,
        ),
        threading.Thread(
            target=luma_sender_thread,
            args=(ctx, luma_queue, stop_event, args.luma_port, stats["luma_out"], stream_pause),
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
        threading.Thread(
            target=ocr_request_thread,
            args=(ctx, ocr_request_queue, stop_event, args.ocr_request_port),
            name="ocr-request-recv", daemon=True,
        ),
        threading.Thread(
            target=ocr_frame_sender_thread,
            args=(ctx, ocr_result_queue, stop_event, args.ocr_frame_port, stats["ocr_frame_out"], stream_pause),
            name="ocr-frame-sender", daemon=True,
        ),
    ]

    for t in threads:
        t.start()
    logging.info(
        "all threads started -- frame_out:%d luma_out:%d mic_out:%d audio_in:%d control:%d "
        "ocr_request:%d ocr_frame_out:%d",
        args.frame_port, args.luma_port, args.mic_port, args.audio_in_port, args.control_port,
        args.ocr_request_port, args.ocr_frame_port,
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
