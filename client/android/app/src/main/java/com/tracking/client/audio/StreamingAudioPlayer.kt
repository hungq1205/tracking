package com.tracking.client.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class StreamingAudioPlayer {

    companion object {
        private const val TAG = "StreamingAudioPlayer"
        private const val SAMPLE_RATE = 24000
    }

    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying

    private var track: AudioTrack? = null

    // Requested directly by the user: Gemini Live's own spoken voice played
    // noticeably louder than Pixie's fluttering cue (which already has its
    // own Settings-configurable cueVolume, see PixieController.kt) with no
    // way to balance the two. AudioTrack.setVolume() does the actual gain
    // — no manual PCM-sample scaling needed — applied both on the track
    // that's about to start AND retroactively via setVolume() below so a
    // mid-session Settings change (next connect()) takes effect immediately
    // rather than only on the NEXT start().
    @Volatile private var volume = 1f

    fun start() {
        stop()
        val minBuf = AudioTrack.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(SAMPLE_RATE)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(maxOf(minBuf, 8192))
            .build()
            .also { it.setVolume(volume); it.play() }
        _isPlaying.value = true
    }

    /** 0f..1f — the Settings "Gemini Voice Volume" slider. Safe to call
     * whether or not a track is currently playing (applied immediately if
     * one exists, remembered for the next start() otherwise). */
    fun setVolume(gain: Float) {
        volume = gain.coerceIn(0f, 1f)
        track?.setVolume(volume)
    }

    fun writeChunk(pcmBytes: ByteArray) {
        // AudioTrack.write() can throw IllegalStateException if the track
        // is concurrently stopped/released (e.g. a mode switch/disconnect
        // racing an in-flight Gemini-voice chunk) — previously uncaught
        // here, unlike stop() below which already guards its own AudioTrack
        // calls the same way.
        try {
            track?.write(pcmBytes, 0, pcmBytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "writeChunk failed: ${e.message}", e)
        }
    }

    fun stop() {
        try {
            track?.stop()
            track?.release()
        } catch (_: Exception) {}
        track = null
        _isPlaying.value = false
    }

    /** Discards whatever's currently queued in the track's own buffer (the
     * stale tail of an in-progress response) WITHOUT tearing the track
     * down — unlike [stop], the track stays open and ready for the next
     * turn's chunks to play immediately, clean. Requested directly by the
     * user: every [SYSTEM]-tagged note is now an INTERRUPTING send (see
     * GeminiLiveClient.onInterrupt), and this is what makes that actually
     * silence whatever Gemini's own voice was still saying rather than
     * letting it keep talking over/after the new response. Pause+flush+
     * play on an already-idle/paused track is a safe no-op (guarded the
     * same way [stop] already guards its own AudioTrack calls). */
    fun interrupt() {
        try {
            track?.pause()
            track?.flush()
            track?.play()
        } catch (_: Exception) {}
    }
}
