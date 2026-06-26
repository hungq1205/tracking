package com.tracking.client.model

import java.io.File

data class ScanUiState(
    // gRPC live-stream state
    val connectionState: ConnectionState = ConnectionState.DISCONNECTED,
    val isStreaming: Boolean = false,
    val locationId: String = "location_01",
    val host: String = "192.168.1.100",
    val port: Int = 50052,
    val pointCount: Int = 0,
    val cameraPosition: List<Float> = emptyList(),
    val statusMessage: String = "Idle",
    val zoneLabel: String = "",
    val zoneRadius: Float = 1.5f,
    val lastExportPath: String = "",
    val lastExportPointCount: Int = 0,
    val lastExportZoneCount: Int = 0,

    // Offline recording state
    val isRecording: Boolean = false,
    val recordingElapsedMs: Long = 0L,
    val videoFile: File? = null,
    val imuFile: File? = null,
    val scanServerHost: String = "192.168.1.100",
    val scanServerPort: Int = 7861,
    val isUploading: Boolean = false,
    val uploadStatus: String? = null,

    val error: String? = null,
)
