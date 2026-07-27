package com.tracking.client.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** The 4 fixed positions Pixie can teleport to, plus a silent CENTER
 * placeholder (used whenever a caller's own deadzone logic decides nothing
 * should be audible right now — see PixieController's class doc). */
enum class PixiePoint { LEFT, RIGHT, UP, DOWN, CENTER }

/**
 * Generic 4-point HRTF cue player — validated first in
 * test_module/pixie_hrtf_app/ before being ported here (see CLAUDE.md's
 * "Pixie + Angle modules" note). Deliberately a DUMB mechanism with NO
 * angle/deviation logic baked in: tracking mode (screen-space hand
 * guidance) and guiding/walking mode (world-heading turn guidance) each
 * compute their own point/volume from completely different signals, so
 * that mapping stays with the caller, not here — see ToolDispatcher.kt's
 * `updateTrackingPixie()` vs. `steerBeaconAlongPath()`.
 *
 * Reuses [HrtfConvolver] (real MIT KEMAR HRTF, unchanged) and the same
 * `assets/fluttering.mp3` loop [HrtfBeaconPlayer] already decodes (via
 * [decodeAssetToMonoPcm], shared — see that file). [HrtfBeaconPlayer]
 * itself stays in the codebase, unreferenced once tracking/guiding/walking
 * switch to this — kept in case its continuous-panning/distance-gain
 * design is wanted again later, same "leave unreferenced, don't delete"
 * precedent this codebase already uses elsewhere (e.g. `beacon_preview.py`,
 * `gemma_vlm.py`).
 *
 * Unlike `HrtfBeaconPlayer.updateDirection()`'s continuously-varying
 * `nearestIndex()` lookup (a fresh brute-force scan over all 710 HRIR
 * directions every call), this only ever needs 4 fixed directions —
 * resolved ONCE at [start], never re-searched. "Teleport" per the class
 * name: [move] is an instantaneous filter switch (crossfaded over one
 * audio chunk to avoid a click, same mechanism `HrtfBeaconPlayer` already
 * uses for its own filter changes) — no gradual flight between points,
 * unlike the test harness's autonomous `PixieMotion` (which doesn't apply
 * here at all: this Pixie's position is 100% caller-commanded every tick).
 */
class PixieController(private val context: Context) {

    companion object {
        private const val SAMPLE_RATE = 44100
        private const val ASSET_PATH = "fluttering.mp3"
        private const val CHUNK_FRAMES = 2048
        private const val GAIN_SMOOTHING = 0.15f  // per-chunk exponential smoothing toward setVolume()'s target

        // Fixed cardinal directions — HrtfBeacon's azimuth/elevation
        // convention (0=ahead/+right, 0=level/+above).
        private const val LEFT_AZIMUTH_DEG = -90f
        private const val RIGHT_AZIMUTH_DEG = 90f
        private const val UP_ELEVATION_DEG = 45f
        private const val DOWN_ELEVATION_DEG = -45f
    }

    private val convolver = HrtfConvolver(context)
    private val scope = CoroutineScope(Dispatchers.IO)
    private var job: Job? = null
    private var track: AudioTrack? = null

    @Volatile private var point: PixiePoint = PixiePoint.CENTER
    @Volatile private var targetGain = 0f
    @Volatile private var currentGain = 0f  // smoothed toward targetGain, read by isEmitting

    /** User-configurable master volume multiplier (Settings' "Cue Volume"
     * slider) — same role as HrtfBeaconPlayer.cueVolume, applied on top of
     * whatever gain a caller's own deviation mapping computes via
     * [setVolume]. */
    @Volatile var cueVolume = 1f

    /** True whenever this is actually producing audible output — same
     * "output-aware VAD gating" role as HrtfBeaconPlayer.isEmitting (see
     * ContinuousVadRecorder's isOutputActive()), now checked against
     * Pixie's own smoothed current gain instead. */
    val isEmitting: Boolean get() = currentGain > 0.02f

    /** Fired with each already-rendered stereo 16-bit PCM chunk (same bytes
     * about to be written to the local [AudioTrack]) — lets [AudioMixer]
     * fold this cue into a combined stream for a remote edge device's
     * speaker, without this class knowing anything about mixing/ZMQ/remote
     * devices. Additive only: local playback below is completely
     * unaffected whether or not this is set. */
    var onRenderedChunk: ((ByteArray) -> Unit)? = null

    fun start() {
        if (job != null) return
        job = scope.launch {
            val monoPcm = decodeAssetToMonoPcm(context, ASSET_PATH, SAMPLE_RATE)
            if (monoPcm.isEmpty()) return@launch  // missing/undecodable asset — silently no-op

            // Resolve all 4 fixed filter pairs ONCE — never searched again
            // (contrast HrtfBeaconPlayer's per-call nearestIndex()).
            val leftIdx = if (convolver.isLoaded) convolver.nearestIndex(LEFT_AZIMUTH_DEG, 0f) else -1
            val rightIdx = if (convolver.isLoaded) convolver.nearestIndex(RIGHT_AZIMUTH_DEG, 0f) else -1
            val upIdx = if (convolver.isLoaded) convolver.nearestIndex(0f, UP_ELEVATION_DEG) else -1
            val downIdx = if (convolver.isLoaded) convolver.nearestIndex(0f, DOWN_ELEVATION_DEG) else -1

            val minBuf = AudioTrack.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_OUT_STEREO, AudioFormat.ENCODING_PCM_16BIT
            )
            val t = AudioTrack.Builder()
                .setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(SAMPLE_RATE)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
                        .build()
                )
                .setTransferMode(AudioTrack.MODE_STREAM)
                .setBufferSizeInBytes(maxOf(minBuf, CHUNK_FRAMES * 4))
                .build()
            t.play()
            track = t

            var readPos = 0
            var lastFilterIndex = -1
            val outL = FloatArray(CHUNK_FRAMES)
            val outR = FloatArray(CHUNK_FRAMES)
            val prevOutL = FloatArray(CHUNK_FRAMES)
            val prevOutR = FloatArray(CHUNK_FRAMES)
            val byteBuf = ByteBuffer.allocate(CHUNK_FRAMES * 4).order(ByteOrder.LITTLE_ENDIAN)

            while (isActive) {
                currentGain += (targetGain - currentGain) * GAIN_SMOOTHING
                byteBuf.clear()

                val idx = when (point) {
                    PixiePoint.LEFT -> leftIdx
                    PixiePoint.RIGHT -> rightIdx
                    PixiePoint.UP -> upIdx
                    PixiePoint.DOWN -> downIdx
                    PixiePoint.CENTER -> leftIdx  // silent whenever CENTER is used — arbitrary choice
                }

                if (convolver.isLoaded && idx >= 0) {
                    convolver.convolveChunk(
                        monoPcm, readPos, CHUNK_FRAMES,
                        convolver.leftFilter(idx), convolver.rightFilter(idx), outL, outR,
                    )
                    if (lastFilterIndex != -1 && lastFilterIndex != idx) {
                        // Point changed since the last chunk — crossfade
                        // from the previous filter's output instead of an
                        // instantaneous switch (which would click).
                        convolver.convolveChunk(
                            monoPcm, readPos, CHUNK_FRAMES,
                            convolver.leftFilter(lastFilterIndex), convolver.rightFilter(lastFilterIndex),
                            prevOutL, prevOutR,
                        )
                        for (i in 0 until CHUNK_FRAMES) {
                            val f = i / CHUNK_FRAMES.toFloat()
                            outL[i] = prevOutL[i] * (1f - f) + outL[i] * f
                            outR[i] = prevOutR[i] * (1f - f) + outR[i] * f
                        }
                    }
                    lastFilterIndex = idx
                } else {
                    // HRIR asset missing/corrupt — plain hard-pan fallback.
                    // No elevation possible here (panning is azimuth-only),
                    // so UP/DOWN/CENTER just split evenly.
                    val lg: Float
                    val rg: Float
                    when (point) {
                        PixiePoint.LEFT -> { lg = 1f; rg = 0f }
                        PixiePoint.RIGHT -> { lg = 0f; rg = 1f }
                        else -> { lg = 0.71f; rg = 0.71f }
                    }
                    for (i in 0 until CHUNK_FRAMES) {
                        val s = monoPcm[(readPos + i) % monoPcm.size]
                        outL[i] = s * lg
                        outR[i] = s * rg
                    }
                }

                for (i in 0 until CHUNK_FRAMES) {
                    byteBuf.putShort(clipToShort(outL[i] * currentGain))
                    byteBuf.putShort(clipToShort(outR[i] * currentGain))
                }
                readPos = (readPos + CHUNK_FRAMES) % monoPcm.size
                val bytes = byteBuf.array().copyOf(byteBuf.position())
                t.write(bytes, 0, bytes.size)
                onRenderedChunk?.invoke(bytes)
            }
        }
    }

    /** Teleports to [point] — see class doc; never a gradual flight. */
    fun move(point: PixiePoint) {
        this.point = point
    }

    /** 0f..1f, smoothed internally (see GAIN_SMOOTHING) so a sudden jump
     * doesn't click. What this value MEANS is entirely up to the caller —
     * see class doc (tracking vs. guiding/walking use completely
     * different deviation-to-volume mappings). */
    fun setVolume(gain: Float) {
        targetGain = gain.coerceIn(0f, 1f) * cueVolume.coerceIn(0f, 1f)
    }

    /** Convenience for setVolume(0f) — matches HrtfBeaconPlayer.mute()'s
     * naming so call sites read the same way. */
    fun mute() = setVolume(0f)

    fun stop() {
        job?.cancel()
        job = null
        try { track?.stop(); track?.release() } catch (_: Exception) {}
        track = null
    }
}
