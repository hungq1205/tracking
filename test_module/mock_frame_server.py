#!/usr/bin/env python3
"""
Mock gRPC server that receives ScanFrameRequest from the Android scan screen
and shows frames in a Gradio UI.

Usage:
    cd /home/hungq/projects/tracking
    ./server/.venv/bin/python3 test_module/mock_frame_server.py

Point the scan screen at this machine's IP on port 50052.
"""

import sys, os, time, threading
from concurrent import futures

import cv2
import numpy as np
import grpc
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tracking_pb2
import tracking_pb2_grpc

SCAN_PORT = 50052

state_lock = threading.Lock()
_state = {
    "frame": None,
    "count": 0,
    "fps": 0.0,
    "last_ts": time.time(),
    "fps_count": 0,
    "fps_ts": time.time(),
    "location_id": "",
}


def _record_frame(image_data: bytes, location_id: str = "") -> bool:
    arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[MOCK] received {len(image_data)} bytes but cv2.imdecode returned None", flush=True)
        return False
    h, w = frame.shape[:2]
    print(f"[MOCK] received {len(image_data)} bytes → decoded {w}x{h}", flush=True)
    now = time.time()
    with state_lock:
        _state["frame"] = frame
        _state["count"] += 1
        _state["fps_count"] += 1
        _state["location_id"] = location_id
        elapsed = now - _state["fps_ts"]
        if elapsed >= 1.0:
            _state["fps"] = _state["fps_count"] / elapsed
            _state["fps_count"] = 0
            _state["fps_ts"] = now
        _state["last_ts"] = now
    return True


class MockMapServicer(tracking_pb2_grpc.MapServiceServicer):
    def ScanFrame(self, request, context):
        ok = _record_frame(request.image_data, request.location_id)
        with state_lock:
            count = _state["count"]
        return tracking_pb2.ScanFrameResponse(
            success=ok,
            point_count=count,
            camera_position=[0.0, 0.0, 0.0],
        )

    def ListMaps(self, *_):
        return tracking_pb2.ListMapsResponse(location_ids=[])

    def GetMapGeometry(self, *_):
        return iter([])

    def SetZoneLabel(self, *_):
        return tracking_pb2.SetZoneLabelResponse(success=True, message="[mock]")

    def ExportScanMap(self, *_):
        return tracking_pb2.ExportScanMapResponse(success=True, output_path="[mock]", point_count=0, zone_count=0)


def start_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    tracking_pb2_grpc.add_MapServiceServicer_to_server(MockMapServicer(), server)
    server.add_insecure_port(f"[::]:{SCAN_PORT}")
    server.start()
    print(f"[MOCK] MapService listening on port {SCAN_PORT}", flush=True)
    server.wait_for_termination()


def get_frame_rgb():
    with state_lock:
        frame = _state["frame"]
        count = _state["count"]
        fps = _state["fps"]
        age = time.time() - _state["last_ts"]
        loc = _state["location_id"]

    if frame is None:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for frames...", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return blank, "No frames yet"

    display = frame.copy()
    cv2.putText(display, f"FPS: {fps:.1f}  frames: {count}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    status = (
        f"Frames received: {count} | "
        f"FPS: {fps:.1f} | "
        f"Last frame: {age:.1f}s ago | "
        f"Size: {frame.shape[1]}x{frame.shape[0]} | "
        f"Location: {loc}"
    )
    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), status


def build_ui():
    with gr.Blocks(title="Mock Scan Server") as demo:
        gr.Markdown(f"## Mock gRPC Scan Frame Receiver  (port {SCAN_PORT})")
        gr.Markdown(
            "Point the Android scan screen at this machine's IP on port **50052**. "
            "Frames arriving here means the phone side is working."
        )
        with gr.Row():
            img = gr.Image(label="Latest frame", height=480)
        status = gr.Textbox(label="Stats", interactive=False)
        btn = gr.Button("Refresh", variant="primary")

        def refresh():
            return get_frame_rgb()

        btn.click(refresh, outputs=[img, status])
        gr.Timer(value=0.5).tick(refresh, outputs=[img, status])

    return demo


if __name__ == "__main__":
    grpc_thread = threading.Thread(target=start_grpc, daemon=True)
    grpc_thread.start()

    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7861)
