package com.tracking.client.grpc

import android.util.Log
import com.tracking.client.model.ConnectionState
import io.grpc.ConnectivityState
import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import tracking.MappingServiceGrpcKt
import tracking.PerceptionServiceGrpcKt
import tracking.StatusServiceGrpcKt
import tracking.TrackingServiceGrpcKt
import java.util.concurrent.TimeUnit

typealias TrackingStub = TrackingServiceGrpcKt.TrackingServiceCoroutineStub
typealias PerceptionStub = PerceptionServiceGrpcKt.PerceptionServiceCoroutineStub
typealias MappingStub = MappingServiceGrpcKt.MappingServiceCoroutineStub
typealias StatusStub = StatusServiceGrpcKt.StatusServiceCoroutineStub

class GrpcClientManager {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var channel: ManagedChannel? = null
    private var monitorJob: Job? = null

    // trackingStub kept for TrackingBackend's existing detectObject/
    // getEmbedding calls (local ORB tracking init/renewal) — not folded
    // into PerceptionService since that call site predates it and still
    // works unchanged (see CLAUDE.md's "Client-Orchestrated Live Session"
    // section).
    var trackingStub: TrackingStub? = null
        private set

    // Consolidated heavy-compute surface used by the on-device Gemini Live
    // tool-dispatch loop (live/ToolDispatcher.kt).
    var perceptionStub: PerceptionStub? = null
        private set
    var mappingStub: MappingStub? = null
        private set

    // Reports LiveSessionState.mode transitions purely so server_gui.py's
    // dashboard can select the correct tab directly (ToolDispatcher.
    // reportMode()) — carries no data any other service needs.
    var statusStub: StatusStub? = null
        private set

    private val _connectionState = kotlinx.coroutines.flow.MutableStateFlow(ConnectionState.DISCONNECTED)
    val connectionState: kotlinx.coroutines.flow.StateFlow<ConnectionState> = _connectionState

    fun connect(host: String, port: Int) {
        disconnect()
        _connectionState.value = ConnectionState.CONNECTING
        val ch = ManagedChannelBuilder.forAddress(host, port)
            .usePlaintext()
            .build()
        channel = ch
        trackingStub = TrackingStub(ch)
        perceptionStub = PerceptionStub(ch)
        mappingStub = MappingStub(ch)
        statusStub = StatusStub(ch)
        startConnectivityMonitor(ch)
    }

    fun disconnect() {
        monitorJob?.cancel()
        monitorJob = null
        channel?.shutdown()?.awaitTermination(3, TimeUnit.SECONDS)
        channel = null
        trackingStub = null
        perceptionStub = null
        mappingStub = null
        statusStub = null
        _connectionState.value = ConnectionState.DISCONNECTED
    }

    private fun startConnectivityMonitor(ch: ManagedChannel) {
        var lastLogged: ConnectivityState? = null
        monitorJob = scope.launch {
            while (isActive) {
                try {
                    // requestConnection=true: a freshly-built channel starts IDLE and
                    // stays that way until something makes an RPC call — without this,
                    // the badge can sit on "Connecting..." indefinitely even once real
                    // traffic (e.g. TrackingBackend's DetectObject calls) has already
                    // pushed the channel to READY on its own, because the state read
                    // here would otherwise depend on being polled at the right moment
                    // relative to unrelated RPC activity elsewhere in the app.
                    val grpcState = ch.getState(true)
                    if (grpcState != lastLogged) {
                        Log.d(TAG, "gRPC connectivity state: $grpcState")
                        lastLogged = grpcState
                    }
                    _connectionState.value = when (grpcState) {
                        ConnectivityState.READY -> ConnectionState.CONNECTED
                        ConnectivityState.CONNECTING, ConnectivityState.IDLE -> ConnectionState.CONNECTING
                        ConnectivityState.TRANSIENT_FAILURE, ConnectivityState.SHUTDOWN -> ConnectionState.ERROR
                    }
                } catch (e: Exception) {
                    // Never let an unexpected getState()/mapping failure silently kill
                    // this loop — that would freeze the badge forever (the bug this
                    // guard replaces: the badge stuck on "Connecting..." while the
                    // channel itself kept working fine for real RPCs).
                    Log.w(TAG, "Connectivity monitor tick failed: ${e.message}")
                }
                delay(1000)
            }
        }
    }

    companion object {
        private const val TAG = "GrpcClientManager"
    }
}
