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

    private val _audioFlow = MutableSharedFlow<ByteArray>(
        extraBufferCapacity = 64,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    override val audioFlow: SharedFlow<ByteArray> = _audioFlow

    /** Called by MainViewModel to deliver a PCM chunk to be played on this device's speaker. */
    fun emitAudio(pcm: ByteArray) { _audioFlow.tryEmit(pcm) }

    override fun connect() = Unit
    override fun disconnect() = Unit
}
