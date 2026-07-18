package com.tracking.client.audio

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
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Continuous binaural beacon sound (assets/fluttering.mp3, looped) —
 * spatialized toward the current waypoint (HrtfBeacon.directionTo) via real
 * HRTF convolution (HrtfConvolver, MIT KEMAR measured impulse responses),
 * not a parametric pan approximation. Falls back to plain equal-power
 * stereo panning only if the HRIR asset failed to load (missing/corrupt
 * assets/hrtf_kemar.bin) — same "degrade, don't crash" style used elsewhere
 * in this codebase.
 *
 * Deliberately NOT Android's Spatializer API (API 32+, and it renders
 * straight to THIS device's own output — no way to capture the result and
 * forward it elsewhere): this project's audio output is moving to a
 * separate low-power edge device connected to earbuds (see EdgeDevice.kt)
 * that should do as little DSP as possible, so ALL rendering (convolution
 * or the pan fallback) happens HERE, phone-side, and [onChunk] exposes
 * every already-rendered stereo PCM16 chunk so a future RemoteEdgeDevice
 * only has to stream bytes to the earbuds, never compute anything itself.
 * updateDirection() only changes which precomputed filter pair / gain is
 * used — the audio asset and the HRIR set are each decoded/loaded once.
 *
 * Filter changes are crossfaded across one chunk (~46ms) to avoid an
 * audible click when the nearest HRIR direction changes.
 */
class HrtfBeaconPlayer(private val context: Context) {

    companion object {
        private const val SAMPLE_RATE = 44100
        private const val ASSET_PATH = "fluttering.mp3"
        private const val CHUNK_FRAMES = 2048
        private const val MAX_RANGE_M = 15f    // beacon gain floors out beyond this distance
        private const val MIN_GAIN = 0.15f     // never fully silent once a target exists
        private const val MAX_GAIN = 1.0f
    }

    /** Fires with every already-rendered stereo PCM16 (little-endian) chunk,
     * in addition to local playback — hook point for a future edge device. */
    var onChunk: ((ByteArray) -> Unit)? = null

    private val convolver = HrtfConvolver(context)

    private val scope = CoroutineScope(Dispatchers.IO)
    private var job: Job? = null
    private var track: AudioTrack? = null

    @Volatile private var filterIndex = 0
    @Volatile private var overallGain = 0f
    @Volatile private var leftGain = 0f    // pan-fallback path only
    @Volatile private var rightGain = 0f   // pan-fallback path only

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
            val byteBuf = ByteBuffer.allocate(CHUNK_FRAMES * 4).order(ByteOrder.LITTLE_ENDIAN)
            var lastFilterIndex = -1
            val outL = FloatArray(CHUNK_FRAMES)
            val outR = FloatArray(CHUNK_FRAMES)
            val prevOutL = FloatArray(CHUNK_FRAMES)
            val prevOutR = FloatArray(CHUNK_FRAMES)

            while (isActive) {
                val gain = overallGain
                byteBuf.clear()

                if (convolver.isLoaded) {
                    val idx = filterIndex
                    convolver.convolveChunk(
                        monoPcm, readPos, CHUNK_FRAMES,
                        convolver.leftFilter(idx), convolver.rightFilter(idx), outL, outR,
                    )
                    if (lastFilterIndex != -1 && lastFilterIndex != idx) {
                        // Direction changed since the last chunk — crossfade from the
                        // previous filter's output over this chunk instead of switching
                        // filters instantaneously (which would click).
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
                    for (i in 0 until CHUNK_FRAMES) {
                        byteBuf.putShort(clipToShort(outL[i] * gain))
                        byteBuf.putShort(clipToShort(outR[i] * gain))
                    }
                } else {
                    val lg = leftGain
                    val rg = rightGain
                    for (i in 0 until CHUNK_FRAMES) {
                        val s = monoPcm[(readPos + i) % monoPcm.size]
                        byteBuf.putShort(clipToShort(s * lg))
                        byteBuf.putShort(clipToShort(s * rg))
                    }
                }

                readPos = (readPos + CHUNK_FRAMES) % monoPcm.size
                val bytes = byteBuf.array().copyOf(byteBuf.position())
                t.write(bytes, 0, bytes.size)
                onChunk?.invoke(bytes)
            }
        }
    }

    /** azimuthDeg/elevationDeg: HrtfBeacon.directionTo()'s convention
     * (0=ahead/+right, 0=level/+above). distanceM: flat-plane distance. */
    fun updateDirection(azimuthDeg: Float, elevationDeg: Float, distanceM: Float) {
        overallGain = (1f - (distanceM / MAX_RANGE_M)).coerceIn(MIN_GAIN, MAX_GAIN)
        if (convolver.isLoaded) {
            filterIndex = convolver.nearestIndex(azimuthDeg, elevationDeg)
        } else {
            val pan = (azimuthDeg / 90f).coerceIn(-1f, 1f)
            val angle = (pan + 1f) * (PI.toFloat() / 4f)  // equal-power pan law
            leftGain = cos(angle) * overallGain
            rightGain = sin(angle) * overallGain
        }
    }

    /** Silences the beacon (e.g. no active waypoint) without stopping the loop/track. */
    fun mute() {
        overallGain = 0f
        leftGain = 0f
        rightGain = 0f
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

/** Decodes a compressed audio asset (mp3, etc.) to mono PCM16 at [targetSampleRate],
 * downmixing multi-channel sources. Returns an empty array on any failure — the
 * caller treats that as "beacon unavailable," never crashes. */
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
