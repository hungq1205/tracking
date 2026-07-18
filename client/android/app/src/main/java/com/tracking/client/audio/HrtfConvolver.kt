package com.tracking.client.audio

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.sin

/**
 * Real binaural HRTF rendering — direct time-domain FIR convolution of a
 * mono source against the pair of ear filters nearest the requested
 * azimuth/elevation, loaded from assets/hrtf_kemar.bin (a compact binary
 * conversion of the public MIT KEMAR HRIR set, SOFA "SimpleFreeFieldHRIR"
 * convention — 710 measured directions, 512-tap filters, 44.1kHz; see
 * CLAUDE.md's HrtfBeaconPlayer.kt entry for the one-off conversion script).
 *
 * This is real HRTF (measured impulse responses, not a parametric pan/ITD
 * approximation) — deliberately implemented as our own convolution rather
 * than Android's Spatializer API (API 32+, and it renders straight to THIS
 * device's own output, giving no way to capture the result and forward it
 * elsewhere) so the exact same already-rendered PCM this produces can later
 * be shipped to a separate low-power edge device that just plays bytes,
 * doing no DSP of its own — see HrtfBeaconPlayer.kt.
 *
 * The 710 measurement points are NOT a uniform grid (coarser azimuth
 * spacing near the poles, standard for this dataset) — nearestIndex() does
 * a brute-force nearest-neighbor search over precomputed unit vectors
 * (dot-product/cosine-similarity, avoids azimuth-wraparound edge cases a
 * naive angle-difference search would need to handle) rather than assuming
 * any particular grid structure.
 */
class HrtfConvolver(context: Context, assetPath: String = "hrtf_kemar.bin") {

    private var sampleRate = 44100
    private var tapCount = 0
    private var unitX: FloatArray = FloatArray(0)
    private var unitY: FloatArray = FloatArray(0)
    private var unitZ: FloatArray = FloatArray(0)
    private var leftIRs: Array<ShortArray> = arrayOf()
    private var rightIRs: Array<ShortArray> = arrayOf()

    val isLoaded: Boolean get() = tapCount > 0
    val sampleRateHz: Int get() = sampleRate
    val tapCountVal: Int get() = tapCount

    init {
        try {
            val bytes = context.assets.open(assetPath).use { it.readBytes() }
            parse(bytes)
        } catch (e: Exception) {
            tapCount = 0  // missing/corrupt asset — isLoaded stays false, caller falls back
        }
    }

    private fun parse(bytes: ByteArray) {
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { buf.get(it) }
        if (String(magic, Charsets.US_ASCII) != "HRIR") return
        sampleRate = buf.int
        val n = buf.int
        tapCount = buf.int
        if (n <= 0 || tapCount <= 0) { tapCount = 0; return }

        unitX = FloatArray(n); unitY = FloatArray(n); unitZ = FloatArray(n)
        leftIRs = Array(n) { ShortArray(tapCount) }
        rightIRs = Array(n) { ShortArray(tapCount) }
        for (i in 0 until n) {
            val az = buf.float
            val el = buf.float
            val azR = Math.toRadians(az.toDouble())
            val elR = Math.toRadians(el.toDouble())
            unitX[i] = (cos(elR) * cos(azR)).toFloat()
            unitY[i] = (cos(elR) * sin(azR)).toFloat()
            unitZ[i] = sin(elR).toFloat()
            val l = leftIRs[i]; val r = rightIRs[i]
            for (k in 0 until tapCount) l[k] = buf.short
            for (k in 0 until tapCount) r[k] = buf.short
        }
    }

    /**
     * Index of the measured direction nearest [azimuthDeg]/[elevationDeg]
     * — HrtfBeacon.directionTo()'s convention (0=ahead, +right, +above) —
     * converted internally to the SOFA source convention (0=front, +90=LEFT,
     * counterclockwise) before comparing: our +azimuth (right) is SOFA's
     * -azimuth. Returns -1 if the asset failed to load.
     */
    fun nearestIndex(azimuthDeg: Float, elevationDeg: Float): Int {
        if (!isLoaded) return -1
        val sofaAz = -azimuthDeg
        val azR = Math.toRadians(sofaAz.toDouble())
        val elR = Math.toRadians(elevationDeg.toDouble())
        val qx = (cos(elR) * cos(azR)).toFloat()
        val qy = (cos(elR) * sin(azR)).toFloat()
        val qz = sin(elR).toFloat()
        var best = 0
        var bestDot = -2f
        for (i in unitX.indices) {
            val dot = unitX[i] * qx + unitY[i] * qy + unitZ[i] * qz
            if (dot > bestDot) { bestDot = dot; best = i }
        }
        return best
    }

    fun leftFilter(index: Int): ShortArray = leftIRs[index]
    fun rightFilter(index: Int): ShortArray = rightIRs[index]

    /**
     * Convolves [frameCount] samples starting at [startReadPos] in the
     * circular mono [source] buffer against [leftIR]/[rightIR] (both
     * fixed-point, *32767 scale — see the conversion script), writing
     * de-scaled float output into [outLeft]/[outRight] (caller applies any
     * further overall gain and converts to PCM16).
     */
    fun convolveChunk(
        source: ShortArray, startReadPos: Int, frameCount: Int,
        leftIR: ShortArray, rightIR: ShortArray,
        outLeft: FloatArray, outRight: FloatArray,
    ) {
        val srcLen = source.size
        val taps = leftIR.size
        for (n in 0 until frameCount) {
            val p = (startReadPos + n) % srcLen
            var l = 0f
            var r = 0f
            for (k in 0 until taps) {
                val s = source[Math.floorMod(p - k, srcLen)].toInt()
                l += s * leftIR[k]
                r += s * rightIR[k]
            }
            outLeft[n] = l / 32767f
            outRight[n] = r / 32767f
        }
    }
}
