import cv2
import numpy as np

import tracking_pb2
import tracking_pb2_grpc


class TrackingServiceServicer(tracking_pb2_grpc.TrackingServiceServicer):
    """Heavy-model primitives used directly by Android's TrackingBackend.kt
    for its local-ORB-tracking init/renewal (see CLAUDE.md's
    "Client-Orchestrated Live Session" section) — the only remaining
    consumer, now that Gemini Live orchestration and every other client
    (Pi thin client, Mediator, Desktop operator GUI) are gone. Each request
    carries its own frame (DetectionRequest/EmbeddingRequest.image_data) —
    there's no continuous frame stream anymore to keep a server-side
    "latest frame" warm the way the old StreamFrame/VoiceChatStream RPCs did."""

    def __init__(self, detector, embedder, activity_monitor=None):
        self.detector = detector
        self.embedder = embedder
        self.activity_monitor = activity_monitor

    def _decode_image(self, data):
        if not data:
            return None
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def DetectObject(self, request, context):
        frame = self._decode_image(request.image_data)
        print(f"[TrackingService] DetectObject <- {context.peer()} prompt='{request.prompt}' "
              f"image_bytes={len(request.image_data)}")
        if frame is None:
            print("[TrackingService] DetectObject: failed to decode image_data")
            return tracking_pb2.DetectionResponse()
        det = self.detector.detect(frame, request.prompt)
        print(f"[TrackingService] DetectObject -> score={det.score:.3f} box={list(det.box_xyxy)}")
        if self.activity_monitor is not None:
            self.activity_monitor.record_tracking(
                f"DetectObject prompt='{request.prompt}' score={det.score:.2f}",
                op="DetectObject", frame_bgr=frame, prompt=request.prompt,
                box_xyxy=list(det.box_xyxy), score=float(det.score),
            )
        return tracking_pb2.DetectionResponse(box_xyxy=list(det.box_xyxy), score=det.score)

    def GetEmbedding(self, request, context):
        frame = self._decode_image(request.image_data)
        print(f"[TrackingService] GetEmbedding <- {context.peer()} box={list(request.box_xyxy)} "
              f"image_bytes={len(request.image_data)}")
        if frame is None:
            print("[TrackingService] GetEmbedding: failed to decode image_data")
            return tracking_pb2.EmbeddingResponse(embedding=[])
        emb = self.embedder.get_embedding(frame, tuple(request.box_xyxy))
        if emb is not None:
            vec = emb.detach().cpu().numpy().flatten().tolist()
            if self.activity_monitor is not None:
                self.activity_monitor.record_tracking(
                    f"GetEmbedding box={list(request.box_xyxy)} dim={len(vec)}",
                    op="GetEmbedding", frame_bgr=frame, box_xyxy=list(request.box_xyxy),
                    embedding_dim=len(vec),
                )
            return tracking_pb2.EmbeddingResponse(embedding=vec)
        print("[TrackingService] GetEmbedding: embedder returned None")
        return tracking_pb2.EmbeddingResponse(embedding=[])
