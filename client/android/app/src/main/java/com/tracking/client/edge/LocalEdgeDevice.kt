package com.tracking.client.edge

import com.tracking.client.camera.CameraManager
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

/**
 * Placeholder [EdgeDevice] where the Android phone itself acts as the edge device:
 * - Camera frames come from [CameraManager] (the local camera).
 * - Audio to play back is emitted via [audioFlow] to [StreamingAudioPlayer] in MainViewModel.
 *
 * To switch to a real remote edge device (Pi, etc.), implement [EdgeDevice] with a
 * network transport (e.g., WebSocket/gRPC to the local device) and inject it in place of this class.
 */
class LocalEdgeDevice(
    private val cameraManager: CameraManager,
) : EdgeDevice {

    override val frameFlow: SharedFlow<ByteArray> = cameraManager.frameFlow
    override val lumaFlow: SharedFlow<CameraManager.LumaFrame> = cameraManager.lumaFlow

    // Unused/empty: the phone's own mic is captured directly by
    // ContinuousVadRecorder's AudioRecord loop (VOICE_COMMUNICATION source +
    // AEC/NS), not routed through EdgeDevice at all — only RemoteEdgeDevice
    // actually populates this.
    override val micFlow: SharedFlow<ByteArray> = MutableSharedFlow()

    private val _audioFlow = MutableSharedFlow<ByteArray>(
        extraBufferCapacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    override val audioFlow: SharedFlow<ByteArray> = _audioFlow

    override fun emitAudio(pcm: ByteArray) { _audioFlow.tryEmit(pcm) }

    override fun connect() = Unit
    override fun disconnect() = Unit
}
