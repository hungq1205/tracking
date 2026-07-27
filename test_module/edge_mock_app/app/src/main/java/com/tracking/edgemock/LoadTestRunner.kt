package com.tracking.edgemock

import kotlin.concurrent.thread
import kotlin.math.sin

/**
 * Drives an N-second (default 4s, per direct spec) concurrent load test
 * over an already-`connect()`ed [EdgeDeviceClient] — all three channels
 * running AT THE SAME TIME, mirroring what would actually happen once this
 * real edge device is wired into `client/android` for real:
 *
 *   - this app continuously PUSHes mock "audio to render" to the edge
 *     (audio_in) — standing in for whatever client/android would actually
 *     send. Mono 24kHz PCM16, 512-sample chunks (~21ms) — this is the ONE
 *     format that's ACTUALLY wired in client/android today
 *     (`LiveAssistantService.emitAudio()` forwards Gemini's own raw voice
 *     PCM, mono/24kHz, to `EdgeDevice.audioFlow`). NOTE: the HRTF/Pixie
 *     steering cue (stereo, 44.1kHz) does NOT currently reach the edge
 *     device at all — see CLAUDE.md's edge_mock_app note — so this
 *     generator deliberately matches the format that's real today, not a
 *     hypothetical future one; revisit this if/when HRTF output is wired
 *     through too (that would need either a second channel or a documented
 *     mixed/resampled format, a real open design question, not decided
 *     here).
 *   - the real edge device is expected to continuously PUSH real mic audio
 *     (mic_out) and real camera frames (frame_out) back — this class only
 *     VALIDATES structural type-correctness (even-length PCM16 bytes;
 *     JPEG magic-byte framing) and MEASURES achieved rates, deliberately
 *     NOT asserting a fixed target fps/sample-rate — the real edge
 *     hardware's actual camera resolution/rate isn't known ahead of time
 *     (confirmed directly with the user), so this is a "does it look like
 *     real structured data, at whatever rate it actually runs" check, not
 *     a conformance-to-a-guessed-spec check.
 */
class LoadTestRunner(private val client: EdgeDeviceClient) {

    data class Report(
        val durationS: Double,
        val micChunks: Int, val micBytes: Long, val micInvalid: Int,
        val frames: Int, val frameBytes: Long, val frameInvalid: Int,
        val audioSent: Int, val audioBytesSent: Long,
    ) {
        val micRateHz: Double get() = if (durationS > 0) micChunks / durationS else 0.0
        val frameFps: Double get() = if (durationS > 0) frames / durationS else 0.0
        val audioSendRateHz: Double get() = if (durationS > 0) audioSent / durationS else 0.0

        fun summary(): String = buildString {
            appendLine("Load test complete — ${"%.1f".format(durationS)}s")
            appendLine()
            appendLine("mic_out  (edge -> this app):")
            appendLine("  chunks: $micChunks  (${"%.1f".format(micRateHz)}/s)   bytes: $micBytes")
            appendLine("  invalid (odd-length/empty): $micInvalid ${if (micChunks == 0) "  [NO DATA RECEIVED]" else if (micInvalid == 0) "  OK" else "  FAIL"}")
            appendLine()
            appendLine("frame_out (edge -> this app):")
            appendLine("  frames: $frames  (${"%.1f".format(frameFps)} fps)   bytes: $frameBytes")
            appendLine("  invalid (not JPEG-framed): $frameInvalid ${if (frames == 0) "  [NO DATA RECEIVED]" else if (frameInvalid == 0) "  OK" else "  FAIL"}")
            appendLine()
            appendLine("audio_in (this app -> edge):")
            append("  sent: $audioSent  (${"%.1f".format(audioSendRateHz)}/s)   bytes: $audioBytesSent")
        }
    }

    @Volatile private var running = false
    @Volatile private var micChunks = 0
    @Volatile private var micBytes = 0L
    @Volatile private var micInvalid = 0
    @Volatile private var frames = 0
    @Volatile private var frameBytes = 0L
    @Volatile private var frameInvalid = 0
    @Volatile private var audioSent = 0
    @Volatile private var audioBytesSent = 0L

    /** Live snapshot while a test is running — for a UI progress poller. */
    fun liveReport(elapsedS: Double): Report = Report(
        elapsedS, micChunks, micBytes, micInvalid, frames, frameBytes, frameInvalid, audioSent, audioBytesSent,
    )

    fun run(durationSeconds: Double, onDone: (Report) -> Unit) {
        if (running) return
        running = true
        micChunks = 0; micBytes = 0; micInvalid = 0
        frames = 0; frameBytes = 0; frameInvalid = 0
        audioSent = 0; audioBytesSent = 0

        client.onMicChunk = { r ->
            micChunks++
            micBytes += r.payload.size
            if (r.payload.isEmpty() || r.payload.size % 2 != 0) micInvalid++
        }
        client.onFrame = { r ->
            frames++
            frameBytes += r.payload.size
            if (!looksLikeJpeg(r.payload)) frameInvalid++
        }

        thread(name = "edge-loadtest-audio-out", isDaemon = true) {
            val sampleRate = 24000
            val chunkSamples = 512
            var phase = 0.0
            val intervalMs = (chunkSamples * 1000L) / sampleRate
            val deadline = System.currentTimeMillis() + (durationSeconds * 1000).toLong()
            while (running && System.currentTimeMillis() < deadline) {
                val chunk = makeToneChunk(phase, 440.0, sampleRate, chunkSamples)
                phase += chunkSamples
                client.sendAudioOut(chunk)
                audioSent++
                audioBytesSent += chunk.size
                Thread.sleep(intervalMs)
            }
            running = false
            client.onMicChunk = null
            client.onFrame = null
            onDone(Report(durationSeconds, micChunks, micBytes, micInvalid, frames, frameBytes, frameInvalid, audioSent, audioBytesSent))
        }
    }

    fun stop() { running = false }
    val isRunning: Boolean get() = running

    companion object {
        private fun looksLikeJpeg(b: ByteArray): Boolean =
            b.size >= 4 && b[0] == 0xFF.toByte() && b[1] == 0xD8.toByte() &&
                b[b.size - 2] == 0xFF.toByte() && b[b.size - 1] == 0xD9.toByte()

        private fun makeToneChunk(phase: Double, freqHz: Double, sampleRate: Int, samples: Int): ByteArray {
            val out = ByteArray(samples * 2)
            for (i in 0 until samples) {
                val t = (phase + i) / sampleRate
                val v = (sin(2.0 * Math.PI * freqHz * t) * 0.3 * Short.MAX_VALUE).toInt().toShort()
                out[i * 2] = (v.toInt() and 0xFF).toByte()
                out[i * 2 + 1] = ((v.toInt() shr 8) and 0xFF).toByte()
            }
            return out
        }
    }
}
