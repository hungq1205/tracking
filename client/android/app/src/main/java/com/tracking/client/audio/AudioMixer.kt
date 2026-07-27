package com.tracking.client.audio

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

/**
 * Combines two independent PCM sources — Gemini Live's own spoken voice
 * (mono 16-bit @24kHz, irregular server-timed chunks, fed via [feedVoice])
 * and Pixie's rendered HRTF cue (stereo 16-bit @44.1kHz, fixed ~46ms
 * chunks, fed via [feedPixie]) — into one stereo 16-bit @44.1kHz stream for
 * [emitMixed], so a remote edge device's single speaker gets ONE combined
 * feed instead of two independently-timed ones. See CLAUDE.md's "Edge
 * device: local vs. remote (ZMQ)" note for why this exists — a real
 * Raspberry-Pi-class edge device only has one speaker, so the phone (which
 * already renders both sources for its own local dual-AudioTrack playback)
 * has to mix before forwarding.
 *
 * Deliberately NOT a replacement for local playback — [StreamingAudioPlayer]/
 * [PixieController] keep playing locally exactly as before; this only taps
 * the same chunks additively via [PixieController.onRenderedChunk] and a
 * direct [feedVoice] call site, and is only started at all while a remote
 * edge device is active (see LiveAssistantService.configureEdgeDevice()).
 *
 * Fixed-rate mixer thread, ticking on [CHUNK_FRAMES]/[SAMPLE_RATE] — the
 * SAME cadence PixieController's own render loop already uses, so most
 * ticks consume exactly one just-arrived Pixie chunk with no
 * resample/rebuffer needed on that side. Each tick pulls up to one chunk's
 * worth of samples from each source's queue (zero-filled if a source has
 * nothing buffered — e.g. Pixie muted, or no voice chunk arrived that
 * tick), sums with clipping, and calls [onMixedChunk] — but only when the
 * result isn't pure silence, matching this codebase's "PUSH only when
 * there's something to send" ZMQ convention (see RemoteEdgeDevice.kt) —
 * no point spamming silent frames to the Pi when nothing is playing.
 *
 * Voice chunks are upsampled 24kHz->44.1kHz and duplicated mono->stereo on
 * ingest (linear interpolation, same technique HrtfBeaconPlayer's asset
 * loader uses for resampleLinear) — a fresh per-chunk resample, not a
 * continuous-phase one; a tiny phase discontinuity between chunks is an
 * accepted, inaudible-in-practice tradeoff for a voice-quality bridge, not
 * a mastered audio path.
 */
class AudioMixer {

    companion object {
        private const val SAMPLE_RATE = 44100
        private const val VOICE_SAMPLE_RATE = 24000
        private const val CHUNK_FRAMES = 2048 // matches PixieController's own cadence
        private const val TICK_MS = (CHUNK_FRAMES * 1000L) / SAMPLE_RATE
        private const val MAX_QUEUED_FRAMES = SAMPLE_RATE * 2 // ~2s backstop against a stalled consumer
    }

    var onMixedChunk: ((ByteArray) -> Unit)? = null

    // Interleaved stereo shorts, one queue per source — synchronized
    // ArrayDeque is more than sufficient at this size/rate (a handful of
    // feed() calls/sec, one drain/tick).
    private val voiceQueue = ArrayDeque<Short>()
    private val pixieQueue = ArrayDeque<Short>()
    private val lock = Any()

    @Volatile private var running = false
    private var thread: Thread? = null

    fun start() {
        if (running) return
        running = true
        synchronized(lock) { voiceQueue.clear(); pixieQueue.clear() }
        thread = Thread({ tickLoop() }, "AudioMixer").also { it.isDaemon = true; it.start() }
    }

    fun stop() {
        running = false
        thread?.join(300)
        thread = null
        synchronized(lock) { voiceQueue.clear(); pixieQueue.clear() }
    }

    /** [pcm]: mono 16-bit LE PCM @[VOICE_SAMPLE_RATE] (Gemini's own voice chunk). */
    fun feedVoice(pcm: ByteArray) {
        if (!running) return
        val stereo = resampleMonoToStereo(pcm, VOICE_SAMPLE_RATE, SAMPLE_RATE)
        synchronized(lock) {
            for (s in stereo) voiceQueue.addLast(s)
            dropExcess(voiceQueue)
        }
    }

    /** [pcm]: stereo 16-bit LE PCM @[SAMPLE_RATE] (Pixie's already-rendered chunk). */
    fun feedPixie(pcm: ByteArray) {
        if (!running) return
        val buf = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN)
        synchronized(lock) {
            while (buf.remaining() >= 2) pixieQueue.addLast(buf.short)
            dropExcess(pixieQueue)
        }
    }

    private fun dropExcess(q: ArrayDeque<Short>) {
        while (q.size > MAX_QUEUED_FRAMES * 2) q.removeFirst() // *2: interleaved stereo shorts
    }

    private fun tickLoop() {
        val framesPerTick = CHUNK_FRAMES * 2 // interleaved stereo
        while (running) {
            val start = System.currentTimeMillis()
            val voice = drain(voiceQueue, framesPerTick)
            val pixie = drain(pixieQueue, framesPerTick)

            var anyNonZero = false
            val out = ByteBuffer.allocate(framesPerTick * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until framesPerTick) {
                val v = (voice.getOrElse(i) { 0 }) + (pixie.getOrElse(i) { 0 })
                if (v != 0) anyNonZero = true
                out.putShort(v.coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort())
            }
            if (anyNonZero) onMixedChunk?.invoke(out.array())

            val elapsed = System.currentTimeMillis() - start
            val sleepMs = TICK_MS - elapsed
            if (sleepMs > 0) try { Thread.sleep(sleepMs) } catch (_: InterruptedException) {}
        }
    }

    private fun drain(q: ArrayDeque<Short>, count: Int): IntArray {
        val out = IntArray(count)
        synchronized(lock) {
            val n = min(count, q.size)
            for (i in 0 until n) out[i] = q.removeFirst().toInt()
        }
        return out
    }

    /** Mono @[srcRate] -> stereo (duplicated) @[dstRate], linear interpolation. */
    private fun resampleMonoToStereo(pcm: ByteArray, srcRate: Int, dstRate: Int): ShortArray {
        val inBuf = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN)
        val inCount = pcm.size / 2
        if (inCount == 0) return ShortArray(0)
        val mono = ShortArray(inCount) { inBuf.short }
        val ratio = srcRate.toDouble() / dstRate
        val outCount = (inCount / ratio).toInt().coerceAtLeast(0)
        val out = ShortArray(outCount * 2)
        for (i in 0 until outCount) {
            val srcPos = i * ratio
            val idx = srcPos.toInt().coerceIn(0, inCount - 1)
            val frac = srcPos - idx
            val s0 = mono[idx]
            val s1 = mono[min(idx + 1, inCount - 1)]
            val v = (s0 + (s1 - s0) * frac).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            out[2 * i] = v
            out[2 * i + 1] = v
        }
        return out
    }
}
