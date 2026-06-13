import cv2
import time
from rpc_client.grpc_client import RemoteGroundingDINO, RemoteEmbedder
from core.local_models import LocalHandDetector
from core.object_tracker import GPUVIOAnchorBackend
from core.guidance_engine import GuidanceEngine
from core.helpers import CameraEmulator
from core.renderer import GUIRenderer

def run_edge_client(video_path, prompt, server_ip="192.168.1.100", stream_to_server=False, stream_fps=5):
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
    emulator = CameraEmulator(video_path, target_fps=15) # Targeting 15 FPS for stability
    
    print(f"[EDGE] Connecting to Server at {server_addr}...")
    last_gui_send = 0
    last_loop_time = time.perf_counter()
    initialized = False

    for frame, _ in emulator.stream():
        t_start = time.perf_counter()
        
        # Calculate real-time FPS
        current_time = time.perf_counter()
        fps = 1.0 / (current_time - last_loop_time) if current_time > last_loop_time else 0.0
        last_loop_time = current_time
        
        # Push to Server GUI if enabled (Tweakable FPS)
        if stream_to_server:
            now = time.time()
            if (now - last_gui_send) > (1.0 / stream_fps):
                remote_detector.send_gui_frame(frame) # Send clean frame for server processing
                last_gui_send = now

        if not initialized:
            print(f"[EDGE] Initializing target: {prompt}")
            obj_track = backend.initialize(frame, prompt)
            if obj_track.visible:
                initialized = True
                state = engine._build_state(obj_track, local_hand.detect(frame), fps=fps)
            else:
                continue
        else:
            # Main tracking loop:
            # backend.update uses local ORB (Fast)
            # engine.update calls local hand detector
            state = engine.update(frame, fps=fps)
        
        latency = (time.perf_counter() - t_start) * 1000
        
        # Visualization
        vis_frame = GUIRenderer.render(frame, state)
        
        cv2.imshow("Edge Tracker View", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your actual server IP and video file
    run_edge_client("test_video.mp4", "Calculator", server_ip="127.0.0.1", stream_to_server=True)