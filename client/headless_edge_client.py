import time

from rpc_client.grpc_client import RemoteGroundingDINO, RemoteEmbedder
from core.local_models import LocalHandDetector
from core.object_tracker import GPUVIOAnchorBackend
from core.guidance_engine import GuidanceEngine
from core.helpers import CameraEmulator
from core.renderer import GUIRenderer # Still needed for server-side streaming
from core.interfaces import IClientApp

class HeadlessEdgeClient(IClientApp):
    def run(self, video_source: str, prompt: str, server_ip: str, stream_to_server: bool):
        server_addr = f"{server_ip}:50051"
        
        # Initialize Remote Proxies
        remote_detector = RemoteGroundingDINO(server_addr)
        remote_embedder = RemoteEmbedder(server_addr)
        local_hand = LocalHandDetector()
        
        # Local Backend (ORB & Homography run on the Pi)
        backend = GPUVIOAnchorBackend(
            detector=remote_detector, 
            embedder=remote_embedder, 
            nfeatures=800  # Lowered features for Pi Zero 2 CPU constraints
        )
        
        engine = GuidanceEngine(object_tracker=backend, hand_detector=local_hand)
        emulator = CameraEmulator(video_source, target_fps=15) # Targeting 15 FPS for stability
        
        print(f"[HEADLESS EDGE] Connecting to Server at {server_addr}...")
        last_gui_send = 0
        initialized = False

        for frame, _ in emulator.stream():
            t_start = time.perf_counter()
            
            if not initialized:
                print(f"[HEADLESS EDGE] Initializing target: {prompt}")
                obj_track = backend.initialize(frame, prompt)
                if obj_track.visible:
                    initialized = True
                    state = engine._build_state(obj_track, local_hand.detect(frame))
                else:
                    print(f"[HEADLESS EDGE] Initial detection failed for '{prompt}'. Retrying...")
                    continue
            else:
                state = engine.update(frame)
            
            latency = (time.perf_counter() - t_start) * 1000
            print(f"[HEADLESS EDGE] Latency: {latency:.1f}ms, Status: {state.object_track.status}, Instruction: {state.instruction}")
            
            # Push to Server GUI if enabled (Throttled to ~20 FPS)
            if stream_to_server:
                now = time.time()
                if (now - last_gui_send) > 0.2: # 5 FPS default
                    remote_detector.send_gui_frame(frame)
                    last_gui_send = now

if __name__ == "__main__":
    client_app = HeadlessEdgeClient()
    client_app.run(video_source="test_video.mp4", prompt="Calculator", server_ip="127.0.0.1", stream_to_server=True)