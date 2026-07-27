"""Standalone verification client for main.py's ZMQ wire protocol.

Connects to frame_out/luma_out/mic_out as a PULL client (mirroring how the
real Android app's RemoteEdgeDevice.kt connects) and to audio_in as a PUSH
client, then validates the wire format documented in main.py's own module
docstring:

    - every message: 16-byte little-endian header (int64 seq + float64
      unix timestamp), then a payload frame -- a 2-part ZMQ multipart
    - frame_out : JPEG bytes (checked via SOI/EOI magic bytes + a real
      cv2.imdecode)
    - luma_out  : 16-byte sub-header (width/height/rowStride/
      rotationDegrees, little-endian int32 each) + raw Y-plane bytes whose
      length must equal height*rowStride
    - mic_out   : even-length PCM16 bytes (mono 16kHz assumed, not
      independently checkable from the client side without a known tone)
    - audio_in  : sends synthetic PCM16 stereo 44.1kHz chunks of varying
      sizes to confirm the send path itself doesn't raise/time out -- there
      is no ack in this protocol, so this is a connectivity check, not a
      round-trip verification; confirm on the Pi's own console/journal that
      main.py logged "[stats] audio_in" incrementing during this run.

Run against a live main.py instance:

    python test_client.py --host 192.168.1.50
    python test_client.py --host 127.0.0.1   # main.py running on the same box
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

import numpy as np
import zmq

try:
    import cv2
except ImportError:
    cv2 = None

HEADER_FMT = "<qd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 16

LUMA_SUBHEADER_FMT = "<iiii"
LUMA_SUBHEADER_SIZE = struct.calcsize(LUMA_SUBHEADER_FMT)
assert LUMA_SUBHEADER_SIZE == 16


def recv_framed(sock: "zmq.Socket", timeout_ms: int = 2000):
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    header, payload = sock.recv_multipart()
    seq, ts = struct.unpack(HEADER_FMT, header)
    return seq, ts, payload


def test_frame_out(ctx, host, port, n_messages=5, timeout_s=10.0) -> bool:
    print(f"\n=== frame_out  (PULL tcp://{host}:{port}) ===")
    if cv2 is None:
        print("  [FAIL] opencv-python is required to validate JPEG payloads (pip install opencv-python-headless)")
        return False

    sock = ctx.socket(zmq.PULL)
    sock.connect(f"tcp://{host}:{port}")
    ok = True
    received = 0
    sizes = []
    deadline = time.monotonic() + timeout_s
    try:
        while received < n_messages and time.monotonic() < deadline:
            try:
                seq, ts, payload = recv_framed(sock)
            except zmq.Again:
                continue
            if not (payload[:2] == b"\xff\xd8" and payload[-2:] == b"\xff\xd9"):
                print(f"  [FAIL] seq={seq} payload missing JPEG SOI/EOI markers")
                ok = False
                continue
            arr = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [FAIL] seq={seq} cv2 could not decode the JPEG payload")
                ok = False
                continue
            sizes.append(len(payload))
            received += 1
            print(f"  seq={seq} ts={ts:.3f} bytes={len(payload)} decoded_shape={img.shape}")
    finally:
        sock.close(0)

    if received == 0:
        print("  [FAIL] no messages received within timeout -- is main.py running and reachable?")
        return False
    print(f"  {'PASS' if ok else 'FAIL'} -- {received} valid JPEG frame(s), avg size {sum(sizes)/len(sizes):.0f} bytes")
    return ok


def test_luma_out(ctx, host, port, n_messages=5, timeout_s=10.0) -> bool:
    print(f"\n=== luma_out   (PULL tcp://{host}:{port}) ===")
    sock = ctx.socket(zmq.PULL)
    sock.connect(f"tcp://{host}:{port}")
    ok = True
    received = 0
    deadline = time.monotonic() + timeout_s
    try:
        while received < n_messages and time.monotonic() < deadline:
            try:
                seq, ts, payload = recv_framed(sock)
            except zmq.Again:
                continue
            if len(payload) < LUMA_SUBHEADER_SIZE:
                print(f"  [FAIL] seq={seq} payload too short for the 16-byte sub-header ({len(payload)} bytes)")
                ok = False
                continue
            width, height, row_stride, rotation = struct.unpack(
                LUMA_SUBHEADER_FMT, payload[:LUMA_SUBHEADER_SIZE]
            )
            luma = payload[LUMA_SUBHEADER_SIZE:]
            expected_len = height * row_stride
            if len(luma) != expected_len:
                print(
                    f"  [FAIL] seq={seq} luma length {len(luma)} != height*rowStride ({expected_len}) "
                    f"(w={width} h={height} stride={row_stride})"
                )
                ok = False
                continue
            if rotation not in (0, 90, 180, 270):
                print(f"  [FAIL] seq={seq} rotationDegrees={rotation} is not one of 0/90/180/270")
                ok = False
                continue
            received += 1
            print(f"  seq={seq} ts={ts:.3f} w={width} h={height} stride={row_stride} rot={rotation} bytes={len(luma)}")
    finally:
        sock.close(0)

    if received == 0:
        print("  [FAIL] no messages received within timeout")
        return False
    print(f"  {'PASS' if ok else 'FAIL'} -- {received} valid luma frame(s)")
    return ok


def test_mic_out(ctx, host, port, n_messages=10, timeout_s=10.0) -> bool:
    print(f"\n=== mic_out    (PULL tcp://{host}:{port}) ===")
    sock = ctx.socket(zmq.PULL)
    sock.connect(f"tcp://{host}:{port}")
    ok = True
    received = 0
    timestamps = []
    deadline = time.monotonic() + timeout_s
    try:
        while received < n_messages and time.monotonic() < deadline:
            try:
                seq, ts, payload = recv_framed(sock)
            except zmq.Again:
                continue
            if len(payload) % 2 != 0:
                print(f"  [FAIL] seq={seq} odd-length payload ({len(payload)} bytes) -- not valid PCM16")
                ok = False
                continue
            timestamps.append(ts)
            received += 1
            print(f"  seq={seq} ts={ts:.3f} bytes={len(payload)} samples={len(payload)//2}")
    finally:
        sock.close(0)

    if received == 0:
        print("  [FAIL] no messages received within timeout")
        return False
    if len(timestamps) >= 2:
        span = timestamps[-1] - timestamps[0]
        rate_hz = (received - 1) / span if span > 0 else 0.0
        print(f"  observed ~{rate_hz:.1f} chunks/s over {span:.2f}s (expect ~16000/512 = 31.25/s at defaults)")
    print(f"  {'PASS' if ok else 'FAIL'} -- {received} valid PCM16 chunk(s)")
    return ok


def test_audio_in(ctx, host, port) -> bool:
    print(f"\n=== audio_in   (PUSH tcp://{host}:{port}) ===")
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDTIMEO, 3000)
    sock.connect(f"tcp://{host}:{port}")
    ok = True
    sample_rate = 44100
    try:
        # Deliberately varying chunk durations -- the protocol explicitly
        # allows variable chunk sizes, so exercise more than one.
        for i, duration_s in enumerate([0.05, 0.2, 0.01]):
            n = int(sample_rate * duration_s)
            t = np.arange(n) / sample_rate
            tone = (np.sin(2 * np.pi * 440 * t) * 0.2 * 32767).astype(np.int16)
            stereo = np.repeat(tone[:, None], 2, axis=1).astype(np.int16).tobytes()
            header = struct.pack(HEADER_FMT, i, time.time())
            try:
                sock.send_multipart([header, stereo])
                print(f"  sent chunk {i}: {len(stereo)} bytes ({n} stereo frames, {duration_s*1000:.0f}ms)")
            except zmq.Again:
                print(f"  [FAIL] send timed out for chunk {i} -- is main.py's audio_in socket bound and reachable?")
                ok = False
            time.sleep(0.05)
    finally:
        sock.close(0)

    print(
        f"  {'PASS' if ok else 'FAIL'} -- synthetic PCM16 stereo chunks sent "
        f"(no ack exists in this protocol; confirm on the Pi's console/journal "
        f"that '[stats] audio_in' logged an incrementing count during this run)"
    )
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--frame-port", type=int, default=5602)
    p.add_argument("--luma-port", type=int, default=5604)
    p.add_argument("--mic-port", type=int, default=5601)
    p.add_argument("--audio-in-port", type=int, default=5603)
    p.add_argument("--timeout", type=float, default=10.0, help="seconds to wait for messages on each PULL socket")
    args = p.parse_args()

    ctx = zmq.Context()
    results = {}
    try:
        results["frame_out"] = test_frame_out(ctx, args.host, args.frame_port, timeout_s=args.timeout)
        results["luma_out"] = test_luma_out(ctx, args.host, args.luma_port, timeout_s=args.timeout)
        results["mic_out"] = test_mic_out(ctx, args.host, args.mic_port, timeout_s=args.timeout)
        results["audio_in"] = test_audio_in(ctx, args.host, args.audio_in_port)
    finally:
        ctx.term()

    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"  {name:10s} {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
