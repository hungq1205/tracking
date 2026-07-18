"""
StatusServiceServicer — the only RPC whose sole purpose is telling the
server what mode the Android client is currently in. Everything else this
server does is heavy-compute or mapping; this one call carries no data any
of that needs — it exists purely so server_gui.py's dashboard can select
the correct tab directly instead of inferring it from whichever RPC
category most recently fired (see ActivityMonitor).
"""
import tracking_pb2
import tracking_pb2_grpc


class StatusServiceServicer(tracking_pb2_grpc.StatusServiceServicer):
    def __init__(self, activity_monitor=None):
        self.activity_monitor = activity_monitor

    def ReportMode(self, request, context):
        print(f"[StatusService] ReportMode <- {context.peer()} mode='{request.mode}' target='{request.target}'")
        if self.activity_monitor is not None:
            self.activity_monitor.record_client_mode(request.mode, request.target)
        return tracking_pb2.ReportModeResponse()
