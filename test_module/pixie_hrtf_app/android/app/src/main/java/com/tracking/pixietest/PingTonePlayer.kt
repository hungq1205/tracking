package com.tracking.pixietest

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/** Which fixed cue position is currently active. CENTER is silent (the
 * aligned dead zone). */
enum class PingSide { LEFT, RIGHT, CENTER }

/**
 * Alignment cue — "which way to turn," not a full spatial-position cue.
 *
 * **Seventh pass — HRTF is back**, per direct follow-up: up/down
 * (elevation) genuinely CANNOT be done with plain stereo panning (panning
 * only ever moves a sound between the two ears, i.e. azimuth only — it
 * has no way to express height at all), unlike left/right alone, which
 * panning handles just fine. So real HRTF convolution
 * ([HrtfConvolver]/the MIT KEMAR HRIR set) is back specifically to carry
 * elevation.
 *
 * **"4 points"**: LEFT+up, LEFT+down, RIGHT+up, RIGHT+down — reached via
 * just 2 CACHED filters (azimuth fixed at ±90°, elevation = whatever
 * [elevationDeg] currently is, live from MainActivity's elevation slider),
 * recomputed only when elevation actually changes (see
 * [ensureFilterCache]), not searched every audio chunk. Up vs. down isn't
 * a separate selectable axis in [sideFor] — it falls out naturally from
 * whatever elevation is currently configured, exactly the same way this
 * app's earlier passes already used elevation.
 *
 * Loudness: silent for 0-5°, then linear 5-180° (see [gainForDeviation])
 * — full volume only at facing directly away, not at a lower cutoff.
 *
 * **Source signal**: `assets/fluttering.mp3` (same loop
 * HrtfBeaconPlayer.kt uses) — a real recorded broadband texture localizes
 * far better through HRTF convolution than a synthesized pure tone (tried
 * and dropped in an earlier pass). `decodeAssetToMonoPcm`/`resampleLinear`
 * below are a direct duplicate of HrtfBeaconPlayer.kt's own (private,
 * file-scoped) versions — not shared, this file is meant to be
 * self-contained (see this app's README).
 */
class PingTonePlayer(private val context: Context) {

    companion object {
        private const val SAMPLE_RATE = 44100
        private const val ASSET_PATH = "fluttering.mp3"
        private const val CHUNK_FRAMES = 2048
        private const val DEAD_ZONE_DEG = 5f
        private const val RAMP_END_DEG = 180f  // full volume only at facing directly away
        private const val GAIN_SMOOTHING = 0.08f
        private const val MAX_GAIN = 0.7f  // peak volume when fully deviated — still well short of full scale
        private const val LEFT_AZIMUTH_DEG = -90f
        private const val RIGHT_AZIMUTH_DEG = 90f

        /** 0-5 deg: silent. 5-180 deg: linear 0..1 (100% only reached at
         * facing directly away). */
        fun gainForDeviation(azimuthDeg: Float): Float {
            val d = abs(azimuthDeg)
            return when {
                d <= DEAD_ZONE_DEG -> 0f
                d >= RAMP_END_DEG -> 1f
                else -> (d - DEAD_ZONE_DEG) / (RAMP_END_DEG - DEAD_ZONE_DEG)
            }
        }

        fun sideFor(azimuthDeg: Float): PingSide = when {
            abs(azimuthDeg) <= DEAD_ZONE_DEG -> PingSide.CENTER
            azimuthDeg < 0f -> PingSide.LEFT
            else -> PingSide.RIGHT
        }
    }

    private val convolver = HrtfConvolver(context)
    private val scope = CoroutineScope(Dispatchers.IO)
    private var job: Job? = null
    private var track: AudioTrack? = null

    @Volatile private var elevationDeg = 0f
    @Volatile private var overallGain = 0f
    @Volatile private var side: PingSide = PingSide.CENTER

    // The 2 cached filter indices ("4 points" — LEFT/RIGHT x whatever
    // up/down elevationDeg currently is) — recomputed only when elevation
    // actually changes, not once per audio chunk. -1 = not yet resolved
    // (before the HRIR asset finishes loading).
    private var cachedElevationDeg = Float.NaN
    private var leftIdx = -1
    private var rightIdx = -1

    private fun ensureFilterCache(elevation: Float) {
        if (elevation == cachedElevationDeg && leftIdx != -1) return
        cachedElevationDeg = elevation
        leftIdx = convolver.nearestIndex(LEFT_AZIMUTH_DEG, elevation)
        rightIdx = convolver.nearestIndex(RIGHT_AZIMUTH_DEG, elevation)
    }

    fun start() {
        if (job != null) return
        job = scope.launch {
            val monoPcm = decodeAssetToMonoPcm(context, ASSET_PATH, SAMPLE_RATE)
            if (monoPcm.isEmpty()) return@launch  // missing/undecodable asset — silently no-op

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
            var currentGain = 0f
            val outL = FloatArray(CHUNK_FRAMES)
            val outR = FloatArray(CHUNK_FRAMES)
            val prevOutL = FloatArray(CHUNK_FRAMES)
            val prevOutR = FloatArray(CHUNK_FRAMES)
            val byteBuf = ByteBuffer.allocate(CHUNK_FRAMES * 4).order(ByteOrder.LITTLE_ENDIAN)

            while (isActive) {
                currentGain += (overallGain - currentGain) * GAIN_SMOOTHING
                byteBuf.clear()

                if (convolver.isLoaded) {
                    ensureFilterCache(elevationDeg)
                    val idx = when (side) {
                        PingSide.LEFT -> leftIdx
                        PingSide.RIGHT -> rightIdx
                        PingSide.CENTER -> leftIdx  // silent anyway (gain=0) — arbitrary choice
                    }
                    convolver.convolveChunk(
                        monoPcm, readPos, CHUNK_FRAMES,
                        convolver.leftFilter(idx), convolver.rightFilter(idx), outL, outR,
                    )
                    if (lastFilterIndex != -1 && lastFilterIndex != idx) {
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
                    // HRIR asset missing/corrupt — plain hard-pan fallback
                    // (no elevation possible here, but at least LEFT/RIGHT
                    // still works).
                    val lg: Float; val rg: Float
                    when (side) {
                        PingSide.LEFT -> { lg = 1f; rg = 0f }
                        PingSide.RIGHT -> { lg = 0f; rg = 1f }
                        PingSide.CENTER -> { lg = 0f; rg = 0f }
                    }
                    for (i in 0 until CHUNK_FRAMES) {
                        val s = monoPcm[(readPos + i) % monoPcm.size]
                        outL[i] = s * lg
                        outR[i] = s * rg
                    }
                }

                for (i in 0 until CHUNK_FRAMES) {
                    byteBuf.putShort(clipToShort(outL[i] * currentGain * MAX_GAIN))
                    byteBuf.putShort(clipToShort(outR[i] * currentGain * MAX_GAIN))
                }
                readPos = (readPos + CHUNK_FRAMES) % monoPcm.size
                val bytes = byteBuf.array().copyOf(byteBuf.position())
                t.write(bytes, 0, bytes.size)
            }
        }
    }

    /** [azimuthDeg]: HrtfBeacon.directionTo()'s convention (0=ahead/+right)
     * — the pixie's EGO-relative direction, i.e. where to actually turn
     * your head to face it. Already wrapped to (-180, 180] by the caller,
     * so its sign alone already picks the shortest-path turn direction —
     * see [sideFor]. [elevationDeg] feeds the HRTF filter cache — real
     * up/down cueing, see class doc's "4 points" note. */
    fun updateDirection(azimuthDeg: Float, elevationDeg: Float) {
        this.elevationDeg = elevationDeg
        this.side = sideFor(azimuthDeg)
        this.overallGain = gainForDeviation(azimuthDeg)
    }

    fun mute() {
        overallGain = 0f
    }

    fun stop() {
        job?.cancel()
        job = null
        try { track?.stop(); track?.release() } catch (_: Exception) {}
        track = null
    }
}

private fun clipToShort(v: Float): Short =
    v.toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()

/** Direct duplicate of HrtfBeaconPlayer.kt's own (private, file-scoped)
 * asset decoder — decodes a compressed audio asset to mono PCM16 at
 * [targetSampleRate], downmixing multi-channel sources. Returns an empty
 * array on any failure. */
private fun decodeAssetToMonoPcm(context: Context, assetPath: String, targetSampleRate: Int): ShortArray {
    return try {
        val afd = context.assets.openFd(assetPath)
        val extractor = MediaExtractor()
        extractor.setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)

        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) { trackIndex = i; format = f; break }
        }
        if (trackIndex < 0 || format == null) { afd.close(); return ShortArray(0) }
        extractor.selectTrack(trackIndex)

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val srcChannels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val srcSampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val out = ArrayList<Short>()
        val bufferInfo = MediaCodec.BufferInfo()
        var sawInputEos = false
        var sawOutputEos = false

        while (!sawOutputEos) {
            if (!sawInputEos) {
                val inIndex = codec.dequeueInputBuffer(10_000)
                if (inIndex >= 0) {
                    val inBuf = codec.getInputBuffer(inIndex)!!
                    val sampleSize = extractor.readSampleData(inBuf, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEos = true
                    } else {
                        codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }
            val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            if (outIndex >= 0) {
                val outBuf = codec.getOutputBuffer(outIndex)!!
                val chunk = ShortArray(bufferInfo.size / 2)
                outBuf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(chunk)
                if (srcChannels > 1) {
                    var i = 0
                    while (i + srcChannels <= chunk.size) {
                        var sum = 0
                        for (c in 0 until srcChannels) sum += chunk[i + c]
                        out.add((sum / srcChannels).toShort())
                        i += srcChannels
                    }
                } else {
                    for (s in chunk) out.add(s)
                }
                codec.releaseOutputBuffer(outIndex, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEos = true
            }
        }
        codec.stop(); codec.release(); extractor.release(); afd.close()

        var pcm = out.toShortArray()
        if (srcSampleRate != targetSampleRate && pcm.isNotEmpty()) {
            pcm = resampleLinear(pcm, srcSampleRate, targetSampleRate)
        }
        pcm
    } catch (e: Exception) {
        ShortArray(0)
    }
}

private fun resampleLinear(input: ShortArray, srcRate: Int, dstRate: Int): ShortArray {
    val ratio = dstRate.toDouble() / srcRate.toDouble()
    val outLen = (input.size * ratio).toInt().coerceAtLeast(1)
    val out = ShortArray(outLen)
    for (i in 0 until outLen) {
        val srcPos = i / ratio
        val i0 = srcPos.toInt().coerceIn(0, input.size - 1)
        val i1 = (i0 + 1).coerceAtMost(input.size - 1)
        val frac = srcPos - i0
        out[i] = (input[i0] * (1 - frac) + input[i1] * frac).toInt().toShort()
    }
    return out
}
