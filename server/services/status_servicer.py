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
(unlike ReportMode), since that cadence would spam the log. ResetSession:
called once per fresh client connection (see MainViewModel.connect()) —
the only place a brand-new connection is distinguishable server-side from
this same client continuing an existing one.
"""
import tracking_pb2
import tracking_pb2_grpc


class StatusServiceServicer(tracking_pb2_grpc.StatusServiceServicer):
    def __init__(self, activity_monitor=None, scan_manager=None):
        self.activity_monitor = activity_monitor
        # None whenever MappingService itself is disabled (no RTABMAP_ADDR)
        # — ResetSession just skips the scan/RTAB-Map half in that case,
        # same "servicer works with whatever's actually available" pattern
        # every other optional dependency in this codebase already follows.
        self.scan_manager = scan_manager

    def ReportMode(self, request, context):
        print(f"[StatusService] ReportMode <- {context.peer()} mode='{request.mode}' target='{request.target}'")
        if self.activity_monitor is not None:
            self.activity_monitor.record_client_mode(request.mode, request.target)
        return tracking_pb2.ReportModeResponse()

    def ReportBeaconDirection(self, request, context):
        if self.activity_monitor is not None:
            self.activity_monitor.record_beacon_direction(request.azimuth_deg, request.muted)
        return tracking_pb2.ReportBeaconDirectionResponse()

    def ResetSession(self, request, context):
        print(f"[StatusService] ResetSession <- {context.peer()} — clearing all scan/mapping state.")
        if self.scan_manager is not None:
            self.scan_manager.reset_all()
        if self.activity_monitor is not None:
            self.activity_monitor.reset()
        return tracking_pb2.ResetSessionResponse()
