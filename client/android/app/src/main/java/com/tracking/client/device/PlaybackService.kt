package com.tracking.client.device

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.media3.common.AudioAttributes
import androidx.media3.common.C
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.DefaultRenderersFactory
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.audio.AudioSink
import androidx.media3.exoplayer.audio.DefaultAudioSink
import androidx.media3.exoplayer.audio.TeeAudioProcessor
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.nio.ByteBuffer

class PlaybackService : Service() {

    private var player: ExoPlayer? = null

    /** Taps ExoPlayer's own decoded PCM output — bypasses the unreliable
     * system-wide MediaProjection/AudioPlaybackCaptureConfiguration capture
     * path entirely for THIS source (radio/music/resolved-YouTube-stream
     * playback, all of which run through this Service's ExoPlayer). Real,
     * fully-controlled tap point vs. capturing a copy of whatever the OS
     * decides matches a USAGE filter system-wide — see LiveAssistantService's
     * startSystemAudioCapture() for the (separate, still-needed-for-the-
     * WebView-based-YouTube-IFrame-player) MediaProjection path, which has
     * no equivalent direct tap since a WebView exposes no raw PCM callback. */
    private inner class RemoteEdgeTeeSink : TeeAudioProcessor.AudioBufferSink {
        private var rate = 0
        private var channels = 0
        override fun flush(sampleRateHz: Int, channelCount: Int, encoding: Int) {
            rate = sampleRateHz
            channels = channelCount
        }
        override fun handleBuffer(buffer: ByteBuffer) {
            if (rate <= 0 || channels <= 0) return
            val cb = onPcmTapped ?: return
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            cb(bytes, rate, channels)
        }
    }

    private inner class TeeRenderersFactory : DefaultRenderersFactory(this) {
        override fun buildAudioSink(
            context: Context,
            enableFloatOutput: Boolean,
            enableAudioTrackPlaybackParams: Boolean,
        ): AudioSink {
            val tee = TeeAudioProcessor(RemoteEdgeTeeSink())
            return DefaultAudioSink.Builder(context)
                .setEnableFloatOutput(enableFloatOutput)
                .setEnableAudioTrackPlaybackParams(enableAudioTrackPlaybackParams)
                .setAudioProcessorChain(
                    DefaultAudioSink.DefaultAudioProcessorChain(tee)
                )
                .build()
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Ducking while the user speaks (PTT held) — see
        // LiveAssistantService.startPtt()/stopPtt(). No-op (and stops this
        // service if it wasn't already playing anything) rather than a
        // dedicated bound API, since this Service has always been
        // started-only, never bound (onBind() stub below).
        when (intent?.action) {
            ACTION_PAUSE -> {
                if (player == null) stopSelf() else player?.pause()
                _isPlaying.value = false
                return START_NOT_STICKY
            }
            ACTION_RESUME -> {
                if (player == null) stopSelf() else player?.play()
                _isPlaying.value = player != null
                return START_NOT_STICKY
            }
        }

        val streamUrl = intent?.getStringExtra("stream_url") ?: return START_NOT_STICKY
        val title     = intent.getStringExtra("title") ?: ""
        val channel   = intent.getStringExtra("channel") ?: ""

        player?.release()
        val newPlayer = ExoPlayer.Builder(this, TeeRenderersFactory())
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(C.USAGE_MEDIA)
                    .setContentType(C.AUDIO_CONTENT_TYPE_MUSIC)
                    .build(),
                /* handleAudioFocus = */ true
            )
            .setHandleAudioBecomingNoisy(true)
            .build()
        newPlayer.addListener(object : Player.Listener {
            override fun onPlayerError(error: PlaybackException) {
                Log.e(TAG, "Playback error: ${error.message}", error)
                _isPlaying.value = false
                stopSelf()
            }
        })
        player = newPlayer

        // Settings "Other Sound Volume" slider — read directly from prefs
        // (same convention LiveAssistantService.restoreSessionFromPrefsIfAvailable()
        // already uses) rather than threading a param through every call
        // site that starts playback (play_video/play_radio/resolved
        // YouTube stream all funnel through here) — this Service is a
        // process-wide singleton with no other config-passing path.
        val volume = java.lang.Float.intBitsToFloat(
            getSharedPreferences("tracking_prefs", MODE_PRIVATE)
                .getInt("other_sound_volume_bits", java.lang.Float.floatToIntBits(1f))
        ).coerceIn(0f, 1f)
        newPlayer.volume = volume

        // setMediaItem()/prepare() can throw SYNCHRONOUSLY, not just report
        // an async onPlayerError — e.g. IllegalStateException("No suitable
        // media source factory found for content type: N") for a stream
        // format this build has no extension registered for (real incident:
        // an HLS radio stream crashed the whole app here before this
        // try/catch existed, since an uncaught exception inside
        // onStartCommand kills the process, not just this Service).
        try {
            newPlayer.setMediaItem(MediaItem.fromUri(streamUrl))
            newPlayer.prepare()
            newPlayer.play()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start playback for '$streamUrl': ${e.message}", e)
            newPlayer.release()
            player = null
            _isPlaying.value = false
            stopSelf()
            return START_NOT_STICKY
        }
        _isPlaying.value = true

        startForeground(NOTIF_ID, buildNotification(title, channel))
        return START_NOT_STICKY
    }

    private fun buildNotification(title: String, channel: String): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val notifChannel = NotificationChannel(
                CHANNEL_ID, "Playback", NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java)
                .createNotificationChannel(notifChannel)
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title.ifBlank { "Playing" })
            .setContentText(channel)
            .setSmallIcon(android.R.drawable.ic_media_play)
            .setOngoing(true)
            .build()
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        player?.release()
        player = null
        _isPlaying.value = false
        stopSelf()
        super.onTaskRemoved(rootIntent)
    }

    override fun onDestroy() {
        player?.release()
        player = null
        _isPlaying.value = false
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    companion object {
        private const val TAG        = "PlaybackService"
        private const val NOTIF_ID  = 42
        private const val CHANNEL_ID = "playback"

        const val ACTION_PAUSE = "com.tracking.client.action.PLAYBACK_PAUSE"
        const val ACTION_RESUME = "com.tracking.client.action.PLAYBACK_RESUME"

        // Process-wide — there's only ever one PlaybackService instance —
        // so ContinuousVadRecorder's output-aware VAD gating (see its own
        // doc comment) can check this without needing a bound connection.
        private val _isPlaying = MutableStateFlow(false)
        val isPlaying: StateFlow<Boolean> = _isPlaying

        // Process-wide, same reasoning as _isPlaying above — set by
        // LiveAssistantService.configureEdgeDevice() only while a remote
        // edge device is active; (pcm, sampleRateHz, channelCount) straight
        // off TeeRenderersFactory's tap, forwarded into AudioMixer.feedExoAudio().
        @Volatile var onPcmTapped: ((ByteArray, Int, Int) -> Unit)? = null
    }
}
