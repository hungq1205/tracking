import json
import os

import grpc

import tracking_pb2
import tracking_pb2_grpc

MAPS_ROOT = os.path.join(os.path.dirname(__file__), "data", "maps")
_CHUNK_SIZE = 64 * 1024  # 64 KB per stream chunk


class MapServiceServicer(tracking_pb2_grpc.MapServiceServicer):
    """Serves pre-built maps (PLY + JSON) to Android/edge clients."""

    def ListMaps(self, request, context):
        location_ids = []
        if os.path.isdir(MAPS_ROOT):
            for name in sorted(os.listdir(MAPS_ROOT)):
                if os.path.isfile(os.path.join(MAPS_ROOT, name, "map_labels.json")):
                    location_ids.append(name)
        return tracking_pb2.ListMapsResponse(location_ids=location_ids)

    def GetMapMetadata(self, request, context):
        labels_path = os.path.join(MAPS_ROOT, request.location_id, "map_labels.json")
        if not os.path.isfile(labels_path):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Map '{request.location_id}' not found")
            return tracking_pb2.MapMetadata()

        with open(labels_path) as f:
            data = json.load(f)

        zones = [
            tracking_pb2.ZoneMetadata(
                label=z["label"],
                bbox_min=z["bbox_min"],
                bbox_max=z["bbox_max"],
            )
            for z in data.get("zones", [])
        ]
        return tracking_pb2.MapMetadata(
            location_id=data.get("location_id", request.location_id),
            created_at=data.get("created_at", ""),
            point_count=data.get("point_count", 0),
            zones=zones,
        )

    def GetMapGeometry(self, request, context):
        ply_path = os.path.join(MAPS_ROOT, request.location_id, "map_geometry.ply")
        if not os.path.isfile(ply_path):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Geometry for '{request.location_id}' not found")
            return

        with open(ply_path, "rb") as f:
            while True:
                chunk = f.read(_CHUNK_SIZE)
                if not chunk:
                    break
                yield tracking_pb2.MapGeometryChunk(data=chunk)
