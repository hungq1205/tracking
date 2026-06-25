package com.tracking.client.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class PushToTalkRecorder {

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val CHUNK = 512
    }

    var onAudioReady: ((ByteArray) -> Unit)? = null
    var onVolumeChange: ((Float) -> Unit)? = null

    @Volatile private var recording = false
    private val audioChunks = mutableListOf<ShortArray>()
    private val lock = Any()
    private var thread: Thread? = null

    fun startRecording() {
        if (recording) return
        recording = true
        synchronized(lock) { audioChunks.clear() }
        thread = Thread(::recordLoop, "PTT-Thread").also { it.isDaemon = true; it.start() }
    }

    fun stopRecording() {
        recording = false
        thread?.join(600)
        thread = null
        val chunks = synchronized(lock) { audioChunks.toList().also { audioChunks.clear() } }
        onVolumeChange?.invoke(0f)
        if (chunks.isNotEmpty()) {
            onAudioReady?.invoke(buildWav(chunks))
        }
    }

    private fun recordLoop() {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, CHUNK * 8)
        )
        recorder.startRecording()
        val chunk = ShortArray(CHUNK)
        try {
            while (recording) {
                val read = recorder.read(chunk, 0, CHUNK)
                if (read > 0) {
                    synchronized(lock) { audioChunks.add(chunk.copyOfRange(0, read)) }
                    onVolumeChange?.invoke(computeRms(chunk, read))
                }
            }
        } finally {
            recorder.stop()
            recorder.release()
            onVolumeChange?.invoke(0f)
        }
    }

    private fun computeRms(samples: ShortArray, count: Int): Float {
        var sumSq = 0.0
        for (i in 0 until count) { val n = samples[i] / 32768.0; sumSq += n * n }
        return sqrt(sumSq / count).toFloat()
    }

    private fun buildWav(chunks: List<ShortArray>): ByteArray {
        val all = ShortArray(chunks.sumOf { it.size })
        var off = 0; for (c in chunks) { c.copyInto(all, off); off += c.size }
        val pcmSize = all.size * 2
        val buf = ByteBuffer.allocate(44 + pcmSize).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray()); buf.putInt(36 + pcmSize); buf.put("WAVE".toByteArray())
        buf.put("fmt ".toByteArray()); buf.putInt(16)
        buf.putShort(1); buf.putShort(1); buf.putInt(SAMPLE_RATE)
        buf.putInt(SAMPLE_RATE * 2); buf.putShort(2); buf.putShort(16)
        buf.put("data".toByteArray()); buf.putInt(pcmSize)
        for (s in all) buf.putShort(s)
        return buf.array()
    }
}
