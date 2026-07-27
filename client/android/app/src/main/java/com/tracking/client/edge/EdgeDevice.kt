package com.tracking.client.edge

import com.tracking.client.camera.CameraManager
import kotlinx.coroutines.flow.SharedFlow

/** Metadata returned at the end of a voice session. */
data class SessionResult(
    val agentName: String = "",
    val agentState: String = "",
    val agentPayload: String = "",
)

/**
 * Abstraction over the physical edge device — the unit that owns the camera,
 * mic, and speaker. Two implementations: [LocalEdgeDevice] (the phone itself
 * acts as the edge device — camera/mic/speaker all local) and
 * [RemoteEdgeDevice] (a Raspberry Pi or similar, talked to over ZeroMQ) — see
 * CLAUDE.md's "Edge device: local vs. remote (ZMQ)" note for the wire
 * protocol and the reasoning behind PUSH/PULL-not-REQ/REP.
 *
 * Responsibilities:
 *   - [frameFlow]: JPEG frames from the edge device's camera
 *   - [lumaFlow]: raw Y-plane-only frames from the same camera, for
 *     AngleTracker's ORB rotation tracking (guiding/walking only) — kept
 *     separate from [frameFlow] since it's a different resolution/format,
 *     not a derivative of the JPEG stream
 *   - [micFlow]: raw PCM chunks captured by the edge device's mic (16kHz
 *     mono, matching [com.tracking.client.audio.ContinuousVadRecorder]'s own
 *     capture format) — [LocalEdgeDevice] leaves this unused/empty, since
 *     the phone's own mic is captured directly by ContinuousVadRecorder's
 *     AudioRecord loop, not through this abstraction
 *   - [audioFlow] / [emitAudio]: deliver a rendered PCM chunk (Gemini's
 *     voice, reading TTS, etc.) to be played on the edge device's speaker
 *
 * The Android app ([MainViewModel]/[com.tracking.client.live.LiveAssistantService])
 * sits between the edge device and the server:
 *   edge device → frames/mic → Android (local processing + Gemini Live) → audio → edge device
 */
interface EdgeDevice {
    /** JPEG frames from the edge device's camera. */
    val frameFlow: SharedFlow<ByteArray>

    /** Raw Y-plane-only frames from the edge device's camera, for ORB rotation tracking. */
    val lumaFlow: SharedFlow<CameraManager.LumaFrame>

    /** Raw PCM mic chunks captured by the edge device (16kHz mono PCM16). */
    val micFlow: SharedFlow<ByteArray>

    /**
     * Raw PCM audio chunks to be played back on the edge device's speaker.
     * Populated as a side effect of [emitAudio] — collect this, or just call
     * [emitAudio] directly; both exist for symmetry with the other flows.
     */
    val audioFlow: SharedFlow<ByteArray>

    /** Deliver a rendered PCM chunk to be played on this device's speaker. */
    fun emitAudio(pcm: ByteArray)

    fun connect()
    fun disconnect()
}
