package com.tracking.client.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.media.audiofx.AcousticEchoCanceler
import android.media.audiofx.NoiseSuppressor
import android.os.Process
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.runBlocking
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Replaces PushToTalkRecorder — continuous mic capture (no tap-and-hold),
 * gated by a client-side amplitude VAD instead. Runs for the whole
 * connected-session lifetime (start() in LiveAssistantService.connect(),
 * stop() in disconnect()/onDestroy()), not per-gesture.
 *
 * Reuses [startThreshold]/[noiseGate] from the SAME (previously dead)
 * Settings fields ("Start Volume"/"Noise Gate") the old PTT design never
 * actually consumed — see CLAUDE.md's "Continuous VAD-gated listening"
 * note.
 *
 * **Self-echo defense — see CLAUDE.md's "Self-echo / output-aware VAD
 * gating" note for the full incident.** Continuous capture means the mic is
 * live at the exact same time the device's own speaker is producing sound
 * (Gemini's spoken reply, reading-mode TTS, the HRTF beacon's continuous
 * guidance tone during walking/guiding, music/YouTube) — without any
 * defense the VAD would hear the phone's own output and misinterpret it as
 * the user talking, feed it back to Gemini, and loop. Two layers:
 *  1. **Real acoustic echo cancellation** — `AudioSource.VOICE_COMMUNICATION`
 *     (not `MIC`) enables platform AEC/NS on most devices for the capture
 *     session, and `AcousticEchoCanceler`/`NoiseSuppressor` are explicitly
 *     attached when available as a belt-and-suspenders measure (some
 *     devices need the explicit attach even with this source). This is the
 *     primary defense and the only one that doesn't sacrifice barge-in —
 *     it works whether or not [isOutputActive] correctly enumerates every
 *     output source, and is essential for the HRTF beacon case specifically
 *     (it plays continuously through all of walking/guiding, so gating the
 *     VAD off entirely whenever it's audible would make the assistant deaf
 *     for the whole navigation session — not acceptable).
 *  2. **Output-aware threshold raise, defense-in-depth** — [isOutputActive]
 *     (polled once per chunk, cheap) reports whether ANY tracked output
 *     (see LiveAssistantService.connect()'s wiring — streamingPlayer,
 *     hrtfBeacon, PlaybackService, the YouTube overlay) is currently
 *     producing sound; while true, the effective start threshold is
 *     multiplied by [OUTPUT_ACTIVE_THRESHOLD_MULTIPLIER] so residual echo
 *     that leaks past AEC needs to be well above ambient level to
 *     mis-register as speech — a genuinely louder direct interruption still
 *     gets through.
 *
 * State machine per chunk:
 *  - IDLE: RMS >= effective start threshold -> transition to SPEAKING, fire
 *    [onSpeechStart], forward this chunk. Otherwise the chunk is discarded
 *    (never sent to Gemini) — only [onVolumeChange] fires, for the UI meter.
 *  - SPEAKING: every chunk is forwarded via [onChunkReady]. Once RMS has
 *    stayed below [noiseGate] for [hangoverMs], transition back to IDLE and
 *    fire [onSpeechEnd] — "the user's line has been registered."
 */
class ContinuousVadRecorder {

    companion object {
        private const val TAG = "ContinuousVadRecorder"
        private const val SAMPLE_RATE = 16000
        private const val CHUNK = 512
        private const val DEFAULT_HANGOVER_MS = 800L
        // Was 3.0x — too aggressive combined with hrtfBeacon.isEmitting being
        // true for the ENTIRE duration of a walking/guiding session (the
        // beacon tone plays continuously, not in short bursts), which meant
        // the effective start threshold was tripled for nearly the whole
        // session, not just around real playback bursts. Real AEC
        // (AudioSource.VOICE_COMMUNICATION + AcousticEchoCanceler, see this
        // class's own header comment) is the primary defense against
        // self-echo; this multiplier is only meant to be a modest
        // defense-in-depth margin on top of that, not a second gate.
        private const val OUTPUT_ACTIVE_THRESHOLD_MULTIPLIER = 1.5f
    }

    var onVolumeChange: ((Float) -> Unit)? = null
    var onChunkReady: ((ByteArray) -> Unit)? = null
    var onSpeechStart: (() -> Unit)? = null
    var onSpeechEnd: (() -> Unit)? = null

    @Volatile private var recording = false
    private var thread: Thread? = null

    /**
     * @param externalChunks when non-null, mic capture is NOT done locally
     *   (no [AudioRecord]/AEC/NS at all) — instead this VAD's state machine
     *   runs over whatever 16kHz mono PCM16 chunks arrive on this flow, e.g.
     *   [com.tracking.client.edge.RemoteEdgeDevice.micFlow] when a remote
     *   edge device's mic is in use. Same chunking/state-machine semantics
     *   either way — only where the raw PCM comes from differs.
     */
    fun start(
        startThreshold: Float,
        noiseGate: Float,
        hangoverMs: Long = DEFAULT_HANGOVER_MS,
        isOutputActive: () -> Boolean = { false },
        externalChunks: SharedFlow<ByteArray>? = null,
    ) {
        if (recording) return
        recording = true
        thread = if (externalChunks != null) {
            Thread({ externalLoop(externalChunks, startThreshold, noiseGate, hangoverMs, isOutputActive) }, "VAD-Thread-Remote")
                .also { it.isDaemon = true; it.start() }
        } else {
            Thread({ recordLoop(startThreshold, noiseGate, hangoverMs, isOutputActive) }, "VAD-Thread")
                .also { it.isDaemon = true; it.start() }
        }
    }

    fun stop() {
        recording = false
        thread?.join(600)
        thread = null
    }

    private fun recordLoop(startThreshold: Float, noiseGate: Float, hangoverMs: Long, isOutputActive: () -> Boolean) {
        // Real bug found via a user report: mic capture could go silent
        // entirely during SCAN mode (queues every camera frame unbounded,
        // see ToolDispatcher.startMappingStream()'s Channel.UNLIMITED for
        // "scanning") — this thread ran at normal/default priority with
        // nothing telling the OS scheduler it's latency-critical, so under
        // SCAN's own CPU/GC pressure it could get starved long enough for
        // the small AudioRecord ring buffer (~256ms, see maxOf below) to
        // overrun before a speech-shaped RMS pattern was ever read.
        // Process.setThreadPriority (not Thread.setPriority — Android's
        // scheduler only honors the former) with THREAD_PRIORITY_URGENT_AUDIO
        // is the standard fix for a raw-audio-capture thread.
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, CHUNK * 8)
        )
        var aec: AcousticEchoCanceler? = null
        var ns: NoiseSuppressor? = null
        try {
            if (AcousticEchoCanceler.isAvailable()) {
                aec = AcousticEchoCanceler.create(recorder.audioSessionId)?.also { it.setEnabled(true) }
            }
            if (NoiseSuppressor.isAvailable()) {
                ns = NoiseSuppressor.create(recorder.audioSessionId)?.also { it.setEnabled(true) }
            }
        } catch (e: Exception) {
            Log.w(TAG, "AEC/NS setup failed: ${e.message}")
        }
        recorder.startRecording()
        val chunk = ShortArray(CHUNK)
        var speaking = false
        var lastAboveGateAtMs = 0L
        try {
            while (recording) {
                val read = recorder.read(chunk, 0, CHUNK)
                if (read <= 0) continue
                val rms = computeRms(chunk, read)
                onVolumeChange?.invoke(rms)
                val now = System.currentTimeMillis()
                val pcm = shortsToPcmBytes(chunk, read)
                if (!speaking) {
                    val effectiveStart = if (isOutputActive()) startThreshold * OUTPUT_ACTIVE_THRESHOLD_MULTIPLIER else startThreshold
                    if (rms >= effectiveStart) {
                        speaking = true
                        lastAboveGateAtMs = now
                        onSpeechStart?.invoke()
                        onChunkReady?.invoke(pcm)
                    }
                    // else: below start threshold — ambient noise (or,
                    // while output is active, residual echo), discard.
                } else {
                    onChunkReady?.invoke(pcm)
                    if (rms >= noiseGate) {
                        lastAboveGateAtMs = now
                    } else if (now - lastAboveGateAtMs >= hangoverMs) {
                        speaking = false
                        onSpeechEnd?.invoke()
                    }
                }
            }
        } finally {
            recorder.stop()
            recorder.release()
            aec?.release()
            ns?.release()
            onVolumeChange?.invoke(0f)
        }
    }

    private fun shortsToPcmBytes(samples: ShortArray, count: Int): ByteArray {
        val buf = ByteBuffer.allocate(count * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until count) buf.putShort(samples[i])
        return buf.array()
    }

    private fun computeRms(samples: ShortArray, count: Int): Float {
        var sumSq = 0.0
        for (i in 0 until count) { val n = samples[i] / 32768.0; sumSq += n * n }
        return sqrt(sumSq / count).toFloat()
    }

    /**
     * Same per-chunk state machine as [recordLoop]'s body, driven by an
     * external PCM source instead of a local [AudioRecord] — no AEC/NS here
     * (there's nothing local to cancel echo against; a remote edge device's
     * own mic hardware/firmware is responsible for its own audio quality).
     */
    private fun externalLoop(
        chunks: SharedFlow<ByteArray>,
        startThreshold: Float,
        noiseGate: Float,
        hangoverMs: Long,
        isOutputActive: () -> Boolean,
    ) {
        Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO)
        var speaking = false
        var lastAboveGateAtMs = 0L
        try {
            runBlocking {
                chunks.collect { pcm ->
                    if (!recording) throw CancellationException("VAD stopped")
                    val rms = computeRmsBytes(pcm)
                    onVolumeChange?.invoke(rms)
                    val now = System.currentTimeMillis()
                    if (!speaking) {
                        val effectiveStart = if (isOutputActive()) startThreshold * OUTPUT_ACTIVE_THRESHOLD_MULTIPLIER else startThreshold
                        if (rms >= effectiveStart) {
                            speaking = true
                            lastAboveGateAtMs = now
                            onSpeechStart?.invoke()
                            onChunkReady?.invoke(pcm)
                        }
                    } else {
                        onChunkReady?.invoke(pcm)
                        if (rms >= noiseGate) {
                            lastAboveGateAtMs = now
                        } else if (now - lastAboveGateAtMs >= hangoverMs) {
                            speaking = false
                            onSpeechEnd?.invoke()
                        }
                    }
                }
            }
        } catch (_: CancellationException) {
            // normal stop() path
        } finally {
            onVolumeChange?.invoke(0f)
        }
    }

    private fun computeRmsBytes(pcm: ByteArray): Float {
        val buf = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN)
        val count = pcm.size / 2
        if (count == 0) return 0f
        var sumSq = 0.0
        for (i in 0 until count) { val n = buf.short / 32768.0; sumSq += n * n }
        return sqrt(sumSq / count).toFloat()
    }
}
