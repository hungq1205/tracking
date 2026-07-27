package com.tracking.pixietest

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch

/**
 * Fuses pixie_hrtf_server.py's RTAB-Map-based, anchor-relative heading
 * fixes with [RotationTracker]'s fast local drift estimate — this is the
 * "local view drift detector" the user asked for, so the HRTF sourcing
 * angle stays responsive during the gap while a frame is in flight to the
 * server and back, exactly mirroring how the main client app's
 * ToolDispatcher.kt/HrtfBeacon.extrapolate() bridge MappingService's ~1Hz
 * update gap (see CLAUDE.md's "Server-planned walking path + client-side
 * latency bridging" — same design, reduced to rotation-only since this
 * test app has no translation/path to bridge, only a steering angle).
 *
 * The trick: [RotationTracker]'s accumulator is never reset except when a
 * fresh server fix is folded in, so a snapshot taken at frame-capture time
 * and the accumulator's value when the matching response later arrives are
 * both measured from the SAME baseline — their quaternion difference is
 * exactly the rotation that happened during that specific frame's round
 * trip, regardless of how many other frames were sent in between.
 *
 * Luma frames from the camera are submitted via [submitLumaFrame] into a
 * CONFLATED channel drained by a dedicated Dispatchers.Default coroutine —
 * the ORB/RANSAC work only ever runs on the MOST RECENT frame, never
 * queues a backlog. This (plus RotationTracker's own faster
 * luma-direct path) is what fixes the ~0.5s perceived lag: previously
 * this work ran synchronously on the camera analysis thread, so a slow
 * call stalled capture itself and let frames pile up behind it.
 */
class HeadingEstimator {

    val rotationTracker = RotationTracker()

    // Anchor-relative heading (degrees) as of the last accepted server fix —
    // the baseline currentHeadingDeg() extrapolates forward from using
    // whatever local rotation has accumulated since that fix.
    @Volatile private var headingBaselineDeg = 0f

    // frame_ts_ns -> rotationTracker.accumulatedRotation() snapshot taken at
    // the moment that frame was captured (see onFrameSentToServer). Pruned
    // on every accepted server fix (see onServerHeading) and capped so a
    // server that stops responding can't leak memory here forever.
    // onFrameSentToServer() runs on the camera analysis thread while
    // onServerHeading() runs on OkHttp's callback thread — both touch this
    // map, so every access goes through [pendingLock].
    private val pendingLock = Any()
    private val pendingSnapshots = LinkedHashMap<Long, FloatArray>()

    @Volatile var latestPixieAzimuthDeg: Float = 0f
        private set
    @Volatile var latestPixieElevationDeg: Float = 0f
        private set
    @Volatile var lastTrackingOk: Boolean = false
        private set

    /** Caps how often [submitLumaFrame]'d frames actually get processed —
     * independent of whatever rate the camera delivers or how fast a call
     * itself completes. 0 or negative disables the cap (process every
     * frame the conflated channel hands over). Mutable — MainActivity
     * applies its "target FPS" input field live, no reconnect needed. */
    @Volatile var targetFps: Int = 20
    private var lastProcessedMs = 0L

    private data class LumaFrame(val luma: ByteArray, val width: Int, val height: Int, val rowStride: Int, val rotationDegrees: Int)

    private val lumaScope = CoroutineScope(Dispatchers.Default)
    private val lumaChannel = Channel<LumaFrame>(capacity = Channel.CONFLATED)

    init {
        lumaScope.launch {
            for (frame in lumaChannel) {
                val now = System.currentTimeMillis()
                val minIntervalMs = if (targetFps > 0) 1000L / targetFps else 0L
                if (minIntervalMs > 0 && now - lastProcessedMs < minIntervalMs) continue
                lastProcessedMs = now
                rotationTracker.processFrameLuma(frame.luma, frame.width, frame.height, frame.rowStride, frame.rotationDegrees)
            }
        }
    }

    /** Call on EVERY captured camera frame — cheap, non-blocking (just
     * posts into a conflated channel; a busy consumer simply overwrites
     * the pending entry with whatever's newest). */
    fun submitLumaFrame(luma: ByteArray, width: Int, height: Int, rowStride: Int, rotationDegrees: Int) {
        lumaChannel.trySend(LumaFrame(luma, width, height, rowStride, rotationDegrees))
    }

    /** Call right when a frame is ALSO forwarded to the server — snapshots
     * the accumulator so the matching response can later measure exactly
     * how much local rotation happened during that frame's round trip.
     * Safe to call even if the matching luma frame's own tracking update
     * hasn't finished processing yet (see class doc) — the snapshot only
     * needs to share the same continuous accumulator timeline as whatever
     * value exists when the response arrives, not reflect this specific
     * frame's own contribution. */
    fun onFrameSentToServer(frameTsNs: Long) {
        synchronized(pendingLock) {
            pendingSnapshots[frameTsNs] = rotationTracker.accumulatedRotation()
            while (pendingSnapshots.size > MAX_PENDING) {
                val oldest = pendingSnapshots.keys.firstOrNull() ?: break
                pendingSnapshots.remove(oldest)
            }
        }
    }

    fun onServerHeading(update: WsHeadTrackClient.HeadingUpdate) {
        latestPixieAzimuthDeg = update.pixieAzimuthDeg
        latestPixieElevationDeg = update.pixieElevationDeg
        lastTrackingOk = update.trackingOk

        val accumAtSend = synchronized(pendingLock) { pendingSnapshots.remove(update.frameTsNs) }
        if (!update.trackingOk) {
            // Tracking lost for this specific frame — the server has
            // nothing authoritative to offer, so leave headingBaselineDeg/
            // the accumulator untouched and keep extrapolating from
            // whatever the last good fix was. Error grows the longer this
            // persists — an accepted limitation for a test harness with no
            // world map to fall back on (see this app's README).
            return
        }

        val accumNow = rotationTracker.accumulatedRotation()
        val deltaSinceSend = if (accumAtSend != null) {
            RotationTracker.quatMultiply(RotationTracker.quatConjugate(accumAtSend), accumNow)
        } else {
            // No matching snapshot (e.g. right after a reconnect) — best we
            // can do is trust the server value outright with no bridge.
            floatArrayOf(0f, 0f, 0f, 1f)
        }
        val deltaHeadingDeg = RotationTracker.headingDegOf(deltaSinceSend)
        headingBaselineDeg = wrapDeg(update.relativeHeadingDeg + deltaHeadingDeg)
        rotationTracker.resetAccumulator()
        // Older in-flight snapshots are now stale relative to the fresh
        // baseline/reset accumulator window — drop them rather than let a
        // late, out-of-order response apply a delta computed against a
        // baseline that no longer exists.
        synchronized(pendingLock) { pendingSnapshots.clear() }
    }

    /** Current best real-time estimate of the anchor-relative heading
     * (degrees, positive = turned right), extrapolated from the last
     * accepted server fix by whatever local rotation has accumulated
     * since. Call this as often as you like (e.g. every UI/audio tick) —
     * it's a pure read, no network/vision work. */
    fun currentHeadingDeg(): Float =
        wrapDeg(headingBaselineDeg + RotationTracker.headingDegOf(rotationTracker.accumulatedRotation()))

    /** Last ORB/RANSAC processing call's duration, ms — see
     * RotationTracker.lastProcessingMs. */
    fun lastProcessingMs(): Long = rotationTracker.lastProcessingMs

    /** Achieved local tracking rate, calls/sec — see RotationTracker.lastFps. */
    fun lastFps(): Double = rotationTracker.lastFps

    /** "Reset anchor" — also send a {"type":"reset"} to the server (see
     * MainActivity) so both sides re-anchor on the next tracked frame. */
    fun resetLocal() {
        headingBaselineDeg = 0f
        rotationTracker.reset()
        synchronized(pendingLock) { pendingSnapshots.clear() }
        lastTrackingOk = false
    }

    fun shutdown() {
        lumaScope.cancel()
        lumaChannel.close()
    }

    companion object {
        private const val MAX_PENDING = 32

        private fun wrapDeg(deg: Float): Float {
            var d = deg % 360f
            if (d > 180f) d -= 360f
            if (d < -180f) d += 360f
            return d
        }
    }
}
