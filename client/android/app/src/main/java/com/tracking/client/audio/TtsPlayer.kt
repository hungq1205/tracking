package com.tracking.client.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext

class TtsPlayer {

    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying

    @Volatile private var currentTrack: AudioTrack? = null

    suspend fun play(wavBytes: ByteArray) = withContext(Dispatchers.IO) {
        if (wavBytes.size <= 44) return@withContext

        stop()

        val pcmData = wavBytes.copyOfRange(44, wavBytes.size)
        val sampleRate = 24000
        val minBufSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(minBufSize, 4096))
            .build()

        currentTrack = track
        _isPlaying.value = true

        try {
            track.play()
            var offset = 0
            while (offset < pcmData.size) {
                val chunk = minOf(4096, pcmData.size - offset)
                val written = track.write(pcmData, offset, chunk)
                if (written < 0) break
                offset += written
            }
            track.stop()
        } finally {
            track.release()
            if (currentTrack === track) currentTrack = null
            _isPlaying.value = false
        }
    }

    fun stop() {
        val track = currentTrack
        currentTrack = null
        try {
            track?.stop()
            track?.release()
        } catch (_: Exception) {}
        _isPlaying.value = false
    }

    fun computeDurationMs(wavBytes: ByteArray): Long {
        val pcmBytes = (wavBytes.size - 44).coerceAtLeast(0)
        return ((pcmBytes / (24000.0 * 2)) * 1000).toLong()
    }
}
