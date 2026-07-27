"""Mock Pi-Zero-2 edge device, for testing the Android <-> edge ZeroMQ link.

Stands in for the real edge device end to end so the Android app's ZeroMQ
client can be exercised without real Pi hardware. Three PUSH/PULL sockets,
matching the real edge-device data flow described in CLAUDE.md's
"Client-Orchestrated Live Session" (EdgeDevice interface):

    mic_out    (this server PUSHes, Android PULLs)   -- mock mic audio, as if
               captured by the edge device's microphone and forwarded to the
               phone. Android should play it back locally so you can
               audibly confirm the right bytes arrived.
    frame_out  (this server PUSHes, Android PULLs)   -- mock JPEG frames, as
               if captured by the edge device's camera. Android should
               display them.
    audio_in   (this server PULLs, Android PUSHes)   -- mock rendered audio
               (e.g. the HRTF beacon / TTS) as Android would send it to the
               edge device's speaker. This server just counts/logs what
               arrives (and plays it locally if `sounddevice` is installed,
               so you can confirm round-trip by ear too).

Run: python mock_edge_server.py [--host 0.0.0.0] [--audio-port 5601]
     [--frame-port 5602] [--audio-in-port 5603]

Deliberately brokerless PUSH/PULL (not REQ/REP, not pub/sub) -- these are
three independent, continuous, unidirectional streams; see CLAUDE.md
discussion on why ZeroMQ PUSH/PULL over MQTT/gRPC/WebSocket for this link.
"""

import argparse
import io
import struct
import threading
import time

import numpy as np
import zmq

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # ~32ms per chunk, same granularity ContinuousVadRecorder uses
FRAME_FPS = 2.0


def make_tone_chunk(phase: float, freq_hz: float) -> tuple[bytes, float]:
    """One PCM16 mono chunk of a sine tone, continuing smoothly from `phase`."""
    t = (np.arange(CHUNK_SAMPLES) + phase) / SAMPLE_RATE
    samples = (np.sin(2 * np.pi * freq_hz * t) * 0.3 * 32767).astype(np.int16)
    return samples.tobytes(), phase + CHUNK_SAMPLES


def mic_out_loop(ctx: zmq.Context, bind_addr: str, stop: threading.Event) -> None:
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 32)
    sock.bind(bind_addr)
    print(f"[mic_out]   PUSH bound  {bind_addr}  (mock mic audio -> Android)")
    phase = 0.0
    freq = 440.0
    seq = 0
    next_send = time.monotonic()
    interval = CHUNK_SAMPLES / SAMPLE_RATE
    while not stop.is_set():
        # Slowly sweep the tone so it's obviously "live", not a static beep.
        freq = 440.0 + 220.0 * np.sin(seq / 200.0)
        chunk, phase = make_tone_chunk(phase, freq)
        header = struct.pack("<Qd", seq, time.time())
        sock.send_multipart([header, chunk])
        seq += 1
        next_send += interval
        sleep_for = next_send - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    sock.close(0)


def frame_out_loop(ctx: zmq.Context, bind_addr: str, stop: threading.Event) -> None:
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.bind(bind_addr)
    print(f"[frame_out] PUSH bound  {bind_addr}  (mock camera frames -> Android)")
    seq = 0
    interval = 1.0 / FRAME_FPS
    colors = [(220, 60, 60), (60, 160, 220), (60, 200, 100), (230, 190, 40)]
    while not stop.is_set():
        jpg = render_mock_frame(seq, colors[seq % len(colors)])
        header = struct.pack("<Qd", seq, time.time())
        sock.send_multipart([header, jpg])
        seq += 1
        time.sleep(interval)
    sock.close(0)


def render_mock_frame(seq: int, color: tuple[int, int, int]) -> bytes:
    if Image is None:
        # No Pillow available -- send a tiny valid-JPEG-shaped placeholder
        # instead of crashing; real testing should install pillow.
        return b"\xff\xd8\xff\xd9"
    img = Image.new("RGB", (640, 480), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 620, 460], outline=(255, 255, 255), width=4)
    draw.text((40, 40), f"MOCK EDGE FRAME #{seq}", fill=(255, 255, 255))
    draw.text((40, 80), time.strftime("%H:%M:%S"), fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def audio_in_loop(ctx: zmq.Context, bind_addr: str, stop: threading.Event) -> None:
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM, 64)
    sock.bind(bind_addr)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    print(f"[audio_in]  PULL bound  {bind_addr}  (Android -> mock edge speaker)")

    stream = None
    if sd is not None:
        try:
            stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
            stream.start()
            print("[audio_in]  sounddevice available -- will play received audio locally")
        except Exception as exc:  # no output device on this host, etc.
            print(f"[audio_in]  sounddevice present but unusable ({exc}); logging only")
            stream = None
    else:
        print("[audio_in]  sounddevice not installed -- logging byte counts only")

    total_bytes = 0
    last_log = time.monotonic()
    while not stop.is_set():
        try:
            parts = sock.recv_multipart()
        except zmq.Again:
            continue
        payload = parts[-1]
        total_bytes += len(payload)
        if stream is not None:
            try:
                stream.write(payload)
            except Exception:
                pass
        now = time.monotonic()
        if now - last_log >= 1.0:
            print(f"[audio_in]  received {total_bytes} bytes total from Android so far")
            last_log = now
    if stream is not None:
        stream.stop()
        stream.close()
    sock.close(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--audio-port", type=int, default=5601)
    p.add_argument("--frame-port", type=int, default=5602)
    p.add_argument("--audio-in-port", type=int, default=5603)
    args = p.parse_args()

    ctx = zmq.Context()
    stop = threading.Event()

    threads = [
        threading.Thread(target=mic_out_loop, args=(ctx, f"tcp://{args.host}:{args.audio_port}", stop), daemon=True),
        threading.Thread(target=frame_out_loop, args=(ctx, f"tcp://{args.host}:{args.frame_port}", stop), daemon=True),
        threading.Thread(target=audio_in_loop, args=(ctx, f"tcp://{args.host}:{args.audio_in_port}", stop), daemon=True),
    ]
    for t in threads:
        t.start()

    print("Mock edge server running. Point the Android app's Edge ZMQ Test screen at this host.")
    print("Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
        ctx.term()


if __name__ == "__main__":
    main()
