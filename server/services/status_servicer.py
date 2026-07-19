"""
StatusServiceServicer — dashboard-only RPCs whose sole purpose is telling
the server things the wire protocol otherwise carries no data for.
ReportMode: which mode the Android client is currently in, so
server_gui.py's dashboard can select the correct tab directly instead of
inferring it from whichever RPC category most recently fired (see
ActivityMonitor). ReportBeaconDirection: the HRTF beacon's actual final
azimuth (post goal-bias, post EMA smoothing — all computed client-side, see
CLAUDE.md's "Local reactive HRTF obstacle-dodge" note), called once per
avoidance tick while walking/guiding is active — no console print here
(unlike ReportMode), since that cadence would spam the log.
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

    def ReportBeaconDirection(self, request, context):
        if self.activity_monitor is not None:
            self.activity_monitor.record_beacon_direction(request.azimuth_deg, request.muted)
        return tracking_pb2.ReportBeaconDirectionResponse()
