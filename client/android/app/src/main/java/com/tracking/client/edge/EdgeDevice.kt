package com.tracking.client.edge

import kotlinx.coroutines.flow.SharedFlow

/** Metadata returned at the end of a voice session. */
data class SessionResult(
    val agentName: String = "",
    val agentState: String = "",
    val agentPayload: String = "",
)

/**
 * Abstraction over the physical edge device — the unit that owns the camera and speaker.
 *
 * Responsibilities:
 *   - [frameFlow]: stream JPEG frames captured by the edge device's camera
 *   - [audioFlow]: emit raw PCM audio chunks to be played back on the edge device's speaker
 *
 * The Android app ([MainViewModel]) sits between the edge device and the server:
 *   edge device → frames → Android (local processing) → server → result → Android → audio → edge device
 *
 * [LocalEdgeDevice] is the placeholder where Android itself acts as the edge device.
 * A future RemoteEdgeDevice would communicate with a Raspberry Pi or similar unit over a network transport.
 */
interface EdgeDevice {
    /** JPEG frames from the edge device's camera. */
    val frameFlow: SharedFlow<ByteArray>

    /**
     * Raw PCM audio chunks to be played back on the edge device's speaker.
     * Emit chunks here to deliver audio output to the physical device.
     */
    val audioFlow: SharedFlow<ByteArray>

    fun connect()
    fun disconnect()
}
