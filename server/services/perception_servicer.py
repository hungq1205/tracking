"""
PerceptionServiceServicer — stateless heavy-compute RPCs for the Android
client's on-device tool-dispatch loop (see CLAUDE.md's "Client-Orchestrated
Live Session" section). Each RPC is a thin wrapper around an existing
tools/*.py model wrapper — no new model code, just a new wire boundary that
replaces the old pattern of Gemini calling a server-side tool which
internally called these same models.
"""

import traceback

import cv2
import grpc
import numpy as np

import tracking_pb2
import tracking_pb2_grpc


class PerceptionServiceServicer(tracking_pb2_grpc.PerceptionServiceServicer):
    def __init__(self, detector, embedder, depth_detector, tts=None, rag_store=None, activity_monitor=None):
        self.detector = detector
        self.embedder = embedder
        self.depth_detector = depth_detector
        self.tts = tts
        self.rag_store = rag_store
        self.activity_monitor = activity_monitor

    def _decode_image(self, data: bytes):
        if not data:
            return None
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def AnalyzeFrame(self, request, context):
        frame = self._decode_image(request.image_data)
        ops_names = [tracking_pb2.AnalysisOp.Name(op) for op in request.ops]
        print(f"[PerceptionService] AnalyzeFrame <- {context.peer()} ops={ops_names} "
              f"prompt='{request.prompt}' image_bytes={len(request.image_data)}")
        response = tracking_pb2.AnalyzeFrameResponse()
        if frame is None:
            print("[PerceptionService] AnalyzeFrame: failed to decode image_data")
            return response

        ops = set(request.ops)
        try:
            if tracking_pb2.DETECT in ops and request.prompt:
                labeled = self.detector.detect_all(frame, request.prompt)
                for d in sorted(labeled, key=lambda x: x.score, reverse=True):
                    response.detections.append(
                        tracking_pb2.Detection(
                            box_xyxy=list(d.box_xyxy), score=d.score, label=d.label
                        )
                    )

            if tracking_pb2.EMBED in ops and len(request.box_xyxy) == 4:
                emb = self.embedder.get_embedding(frame, tuple(request.box_xyxy))
                if emb is not None:
                    response.embedding.extend(emb.detach().cpu().numpy().flatten().tolist())

            if tracking_pb2.DEPTH in ops and self.depth_detector is not None:
                detected, value = self.depth_detector.check_obstacle(frame)
                response.obstacle.CopyFrom(
                    tracking_pb2.ObstacleInfo(
                        detected=bool(detected),
                        distance_m=float(value),
                        angle_deg=0.0,  # corridor-center check only, no angular sweep
                        description=(
                            f"{type(self.depth_detector).__name__}: {value:.2f}"
                            if detected else ""
                        ),
                    )
                )

            if tracking_pb2.TRAVERSABILITY in ops and self.depth_detector is not None:
                trav = self.depth_detector.estimate_traversability(frame)
                response.traversability.CopyFrom(
                    tracking_pb2.TraversabilityInfo(
                        clearance_m=trav.clearance_m,
                        min_angle_deg=trav.min_angle_deg,
                        max_angle_deg=trav.max_angle_deg,
                        angle_step_deg=trav.angle_step_deg,
                        max_range_m=trav.max_range_m,
                    )
                )
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

        if self.activity_monitor is not None:
            obstacle = response.obstacle if response.HasField("obstacle") else None
            self.activity_monitor.record_perception(
                f"AnalyzeFrame ops={ops_names} detections={len(response.detections)}",
                op="AnalyzeFrame", frame_bgr=frame, ops=ops_names, prompt=request.prompt,
                detections=[
                    {"box_xyxy": list(d.box_xyxy), "score": d.score, "label": d.label}
                    for d in response.detections
                ],
                obstacle=(
                    {"detected": obstacle.detected, "distance_m": obstacle.distance_m}
                    if obstacle is not None else None
                ),
                traversability=(
                    {
                        "clearance_m": list(response.traversability.clearance_m),
                        "min_angle_deg": response.traversability.min_angle_deg,
                        "max_angle_deg": response.traversability.max_angle_deg,
                        "angle_step_deg": response.traversability.angle_step_deg,
                        "max_range_m": response.traversability.max_range_m,
                    }
                    if response.HasField("traversability") else None
                ),
            )
        return response

    def Synthesize(self, request, context):
        print(f"[PerceptionService] Synthesize <- {context.peer()} chars={len(request.text)}")
        if self.activity_monitor is not None:
            self.activity_monitor.record_perception(
                f"Synthesize chars={len(request.text)}",
                op="Synthesize", text=request.text,
            )
        if self.tts is None or not request.text:
            return
        try:
            for pcm in self.tts.synthesize_pcm_chunks(request.text):
                yield tracking_pb2.PcmChunk(pcm_data=pcm)
        except Exception as e:
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def Embed(self, request, context):
        print(f"[PerceptionService] Embed <- {context.peer()} text='{request.text}'")
        if self.activity_monitor is not None:
            self.activity_monitor.record_perception(
                f"Embed text='{request.text}'",
                op="Embed", text=request.text,
            )
        if self.rag_store is None or not request.text:
            return tracking_pb2.EmbedResponse(vector=[])
        vec = self.rag_store.embed_text(request.text)
        return tracking_pb2.EmbedResponse(vector=vec.tolist())
