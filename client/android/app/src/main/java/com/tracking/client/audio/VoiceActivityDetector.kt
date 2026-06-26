package com.tracking.client.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class VoiceActivityDetector {

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val VAD_CHUNK = 512
        private const val VAD_SILENCE_CHUNKS = 47
        private const val VAD_MIN_SPEECH_CHUNKS = 5
        private const val MAX_SPEECH_CHUNKS = 500  // ~16 s; force-submit if noise never stops
    }

    @Volatile var threshold: Float = 0.03f       // noise gate: min RMS to keep buffering
    @Volatile var startThreshold: Float = 0.05f  // peak RMS required to START speech

    var onAudioReady: ((ByteArray) -> Unit)? = null
    var onStatusChange: ((String) -> Unit)? = null
    var onVolumeChange: ((Float) -> Unit)? = null

    @Volatile private var running = false
    private var thread: Thread? = null

    fun start() {
        if (running) return
        running = true
        thread = Thread(::vadLoop, "VAD-Thread").also { it.isDaemon = true; it.start() }
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
    }

    private fun vadLoop() {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = maxOf(minBuf, VAD_CHUNK * 2 * 4)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufSize
        )

        recorder.startRecording()
        onStatusChange?.invoke("Listening")

        val speechBuffer = mutableListOf<ShortArray>()
        var inSpeech = false
        var speechChunks = 0
        var silentChunks = 0
        var peakRms = 0f
        val chunk = ShortArray(VAD_CHUNK)

        fun resetSpeech() {
            speechBuffer.clear(); inSpeech = false; speechChunks = 0; silentChunks = 0; peakRms = 0f
        }

        try {
            while (running) {
                val read = recorder.read(chunk, 0, VAD_CHUNK)
                if (read <= 0) continue

                val rms = computeRms(chunk, read)
                onVolumeChange?.invoke(rms)

                if (rms >= threshold) {
                    peakRms = maxOf(peakRms, rms)
                    speechBuffer.add(chunk.copyOf())
                    speechChunks++
                    silentChunks = 0
                    // only transition to speaking if peak has crossed the start threshold
                    if (!inSpeech && speechChunks >= VAD_MIN_SPEECH_CHUNKS && peakRms >= startThreshold) {
                        inSpeech = true
                        onStatusChange?.invoke("Speaking")
                    }
                    if (speechBuffer.size >= MAX_SPEECH_CHUNKS) {
                        if (inSpeech) submitAudio(speechBuffer)
                        resetSpeech()
                        onStatusChange?.invoke("Listening")
                    }
                } else {
                    if (inSpeech) {
                        speechBuffer.add(chunk.copyOf())
                        silentChunks++
                        if (silentChunks >= VAD_SILENCE_CHUNKS) {
                            submitAudio(speechBuffer)
                            resetSpeech()
                            onStatusChange?.invoke("Listening")
                        }
                    } else {
                        speechChunks = maxOf(0, speechChunks - 1)
                        if (speechChunks == 0) peakRms = 0f
                    }
                }
            }
        } finally {
            recorder.stop()
            recorder.release()
            onStatusChange?.invoke("Idle")
        }
    }

    private fun computeRms(samples: ShortArray, count: Int): Float {
        var sumSq = 0.0
        for (i in 0 until count) {
            val n = samples[i] / 32768.0
            sumSq += n * n
        }
        return sqrt(sumSq / count).toFloat()
    }

    private fun submitAudio(chunks: List<ShortArray>) {
        val allShorts = ShortArray(chunks.sumOf { it.size })
        var offset = 0
        for (c in chunks) {
            c.copyInto(allShorts, offset)
            offset += c.size
        }
        val wavBytes = buildWav(allShorts)
        onAudioReady?.invoke(wavBytes)
    }

    private fun buildWav(samples: ShortArray): ByteArray {
        val pcmSize = samples.size * 2
        val totalSize = 44 + pcmSize
        val buf = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header
        buf.put("RIFF".toByteArray())
        buf.putInt(totalSize - 8)
        buf.put("WAVE".toByteArray())

        // fmt chunk
        buf.put("fmt ".toByteArray())
        buf.putInt(16)
        buf.putShort(1)                        // PCM
        buf.putShort(1)                        // mono
        buf.putInt(SAMPLE_RATE)
        buf.putInt(SAMPLE_RATE * 2)            // byteRate
        buf.putShort(2)                        // blockAlign
        buf.putShort(16)                       // bitsPerSample

        // data chunk
        buf.put("data".toByteArray())
        buf.putInt(pcmSize)
        for (s in samples) buf.putShort(s)

        return buf.array()
    }
}
