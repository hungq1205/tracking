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
     * JPEG frames from a one-shot FULL-RESOLUTION still capture, requested
     * via [requestOcrFrame] -- distinct from [frameFlow]'s continuous,
     * lower-resolution (same size as [lumaFlow] now) video stream. OCR needs
     * real detail a live-preview-resolution frame can't give it; the
     * continuous streams stay low-res for bandwidth/latency's sake. See
     * CLAUDE.md's "Full-resolution OCR capture channel" note for the wire
     * protocol (ocr_request/ocr_frame_out) and why frame_out/luma_out are
     * paused server-side for the duration of a still capture+send.
     */
    val ocrFrameFlow: SharedFlow<ByteArray>

    /**
     * Requests one full-resolution capture (fire-and-forget -- the result
     * arrives asynchronously on [ocrFrameFlow], or never if the device
     * doesn't support this or the request/response is lost; callers must
     * time out on their own). No-op default: [LocalEdgeDevice] doesn't need
     * this at all -- the phone's own camera already provides a
     * full-resolution frame on demand via [CameraManager] with no separate
     * request/response round trip needed.
     */
    fun requestOcrFrame() {}

    /**
     * Raw PCM audio chunks to be played back on the edge device's speaker.
     * Populated as a side effect of [emitAudio] — collect this, or just call
     * [emitAudio] directly; both exist for symmetry with the other flows.
     */
    val audioFlow: SharedFlow<ByteArray>

    /** Deliver a rendered PCM chunk to be played on this device's speaker. */
    fun emitAudio(pcm: ByteArray)

    /**
     * Tells the edge device which mode the app is currently in, so it can
     * skip capturing/sending [lumaFlow] when nothing will consume it --
     * only walking/guiding actually process luma client-side (see
     * ToolDispatcher.feedAngleLumaFrame()), so every other mode was paying
     * the Pi's full 15fps camera/network cost for data the phone just
     * discarded. Piggybacked on the same call site ToolDispatcher's
     * existing reportMode() already fires on every mode change (for the
     * server dashboard), no new call sites needed. No-op default --
     * [LocalEdgeDevice] needs no equivalent, since local capture already
     * gates lumaFlow emission via CameraManager.mappingMode without any
     * network round trip.
     */
    fun reportMode(mode: String) {}

    fun connect()
    fun disconnect()
}
