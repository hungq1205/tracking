package com.tracking.client.grpc

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
import tracking.MapServiceGrpcKt
import tracking.MediatorServiceGrpcKt
import tracking.TrackingServiceGrpcKt
import java.util.concurrent.TimeUnit

typealias MediatorStub = MediatorServiceGrpcKt.MediatorServiceCoroutineStub
typealias TrackingStub = TrackingServiceGrpcKt.TrackingServiceCoroutineStub
typealias MapStub = MapServiceGrpcKt.MapServiceCoroutineStub

class GrpcClientManager {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var channel: ManagedChannel? = null
    private var monitorJob: Job? = null

    var mediatorStub: MediatorStub? = null
        private set
    var trackingStub: TrackingStub? = null
        private set

    private var scanChannel: ManagedChannel? = null
    var mapStub: MapStub? = null
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
        mediatorStub = MediatorStub(ch)
        trackingStub = TrackingStub(ch)
        startConnectivityMonitor(ch)
    }

    fun disconnect() {
        monitorJob?.cancel()
        monitorJob = null
        channel?.shutdown()?.awaitTermination(3, TimeUnit.SECONDS)
        channel = null
        mediatorStub = null
        trackingStub = null
        disconnectScan()
        _connectionState.value = ConnectionState.DISCONNECTED
    }

    fun connectScan(host: String, port: Int) {
        scanChannel?.shutdown()?.awaitTermination(3, TimeUnit.SECONDS)
        val ch = ManagedChannelBuilder.forAddress(host, port)
            .usePlaintext()
            .build()
        scanChannel = ch
        mapStub = MapStub(ch)
    }

    fun disconnectScan() {
        scanChannel?.shutdown()?.awaitTermination(3, TimeUnit.SECONDS)
        scanChannel = null
        mapStub = null
    }

    private fun startConnectivityMonitor(ch: ManagedChannel) {
        monitorJob = scope.launch {
            while (isActive) {
                val grpcState = ch.getState(false)
                _connectionState.value = when (grpcState) {
                    ConnectivityState.READY -> ConnectionState.CONNECTED
                    ConnectivityState.CONNECTING, ConnectivityState.IDLE -> ConnectionState.CONNECTING
                    ConnectivityState.TRANSIENT_FAILURE, ConnectivityState.SHUTDOWN -> ConnectionState.ERROR
                }
                delay(1000)
            }
        }
    }
}
