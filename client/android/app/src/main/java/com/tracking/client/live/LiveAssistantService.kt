package com.tracking.client.live

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Intent
import android.graphics.BitmapFactory
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import com.tracking.client.audio.AudioMixer
import com.tracking.client.audio.ContinuousVadRecorder
import com.tracking.client.audio.PixieController
import com.tracking.client.audio.ReadingTtsPlayer
import com.tracking.client.audio.StreamingAudioPlayer
import com.tracking.client.camera.CameraManager
import com.tracking.client.device.AndroidDeviceToolHandler
import com.tracking.client.device.DeviceToolHandler
import com.tracking.client.device.PlaybackService
import com.tracking.client.edge.EdgeDevice
import com.tracking.client.edge.LocalEdgeDevice
import com.tracking.client.edge.RemoteEdgeDevice
import com.tracking.client.grpc.GrpcClientManager
import com.tracking.client.model.AppUiState
import com.tracking.client.model.ChatMessage
import com.tracking.client.model.ConnectionState
import com.tracking.client.model.ObjectTrack
import com.tracking.client.tracking.HandTracker
import com.tracking.client.tracking.TrackingBackend
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File

/**
 * Foreground bound Service hosting the entire Gemini Live session graph —
 * gRPC, camera frames, mic capture, tool dispatch — that used to live in
 * MainViewModel's viewModelScope (Activity/ViewModel-scoped, died on
 * backgrounding/task-swipe). MainViewModel is now a thin bound-client facade
 * (see MainViewModel.kt) so the assistant keeps running — including
 * reacting to incoming calls/SMS via CallBackgroundReceiver/
 * SmsBackgroundReceiver — with the phone screen off or the app out of
 * Recents. See CLAUDE.md's "Client-Orchestrated Live Session" section.
 *
 * Deliberately does NOT stop itself in onTaskRemoved (the one divergence
 * from PlaybackService.kt's pattern, which this otherwise mirrors for the
 * foreground-notification shape) — the whole point of this Service is to
 * survive task removal.
 */
class LiveAssistantService : LifecycleService() {

    inner class LocalBinder : Binder() {
        fun getService(): LiveAssistantService = this@LiveAssistantService
    }

    private val binder = LocalBinder()

    val grpcManager = GrpcClientManager()
    val cameraManager = CameraManager(this)
    private val vadRecorder = ContinuousVadRecorder()
    private val streamingPlayer = StreamingAudioPlayer()
    private val pixieController = PixieController(this)
    // Combines Gemini's voice + Pixie's HRTF cue into one stream for a
    // remote edge device's single speaker — see AudioMixer.kt. Only
    // started/stopped by configureEdgeDevice(); feedVoice()/feedPixie() are
    // themselves no-ops while stopped, so wiring pixieController's tap
    // unconditionally (below) is harmless when Local is active.
    private val audioMixer = AudioMixer()
    @Volatile private var remoteEdgeActive = false

    // On-demand "give me the current frame right now" consumers (OCR,
    // run_detection, tracking-mode initialization, walking/guiding's hazard
    // check) historically all pulled straight from cameraManager's own
    // local-camera buffer (clearestRecentFrame()/clearestRecentFrameWithSharpness()),
    // predating RemoteEdgeDevice — only the CONTINUOUS frame collector below
    // (startLocalProcessing()) was ever wired to edgeDevice.frameFlow. That
    // meant tracking mode (and OCR, etc.) silently used the phone's own
    // camera even with "Use Remote Edge Device" fully active and connected —
    // a real bug, found from a live device report. This cache is updated
    // every frame that collector already receives from edgeDevice.frameFlow,
    // and the closures below/startLocalTracking() now prefer it whenever
    // remoteEdgeActive, instead of ever touching cameraManager in that case.
    @Volatile private var lastEdgeFrame: ByteArray? = null

    // UI-facing mirror of lastEdgeFrame, populated ONLY while remoteEdgeActive
    // — MainScreen shows this in place of the phone's own CameraX preview
    // when a remote edge device supplies the camera. Local mode leaves this
    // null so MainScreen falls back to the normal PreviewView.
    private val _edgeFrame = MutableStateFlow<ByteArray?>(null)
    val edgeFrame: StateFlow<ByteArray?> = _edgeFrame
    private val _isRemoteEdgeActive = MutableStateFlow(false)
    val isRemoteEdgeActive: StateFlow<Boolean> = _isRemoteEdgeActive

    // Voice-message-sent confirmation cue — a synthesized ToneGenerator
    // tone, not a bundled audio asset, same "works with no sound file
    // supplied" precedent as ToolDispatcher's playDeadEndAlert(). Played
    // once per fully-registered utterance (vadRecorder.onSpeechEnd, right
    // after sendAudioStreamEnd() confirms the socket actually took it) —
    // NOT per audio chunk, since chunks stream continuously at ~512-sample
    // granularity while SPEAKING and a pop per chunk would be constant
    // noise; "a voice message" is the whole utterance, not a chunk.
    private var voiceSentToneGenerator: ToneGenerator? = null

    /** Short confirmation tone — TONE_PROP_BEEP2 is the shortest tone in
     * ToneGenerator's DTMF-derived set. Played at full ToneGenerator volume
     * (bumped up from 40% per direct user feedback — the original level was
     * too quiet to reliably notice). Deliberately NOT tracked by
     * ContinuousVadRecorder's isOutputActive() output-aware gating (unlike
     * streamingPlayer/pixieController/PlaybackService/YouTube) — it's a single
     * ~50ms blip immediately after the mic just finished an utterance and
     * transitioned back to IDLE, not a sustained output that could
     * plausibly be mistaken for new speech. */
    private fun playVoiceSentCue() {
        try {
            val tg = voiceSentToneGenerator
                ?: ToneGenerator(AudioManager.STREAM_MUSIC, ToneGenerator.MAX_VOLUME)
                    .also { voiceSentToneGenerator = it }
            tg.startTone(ToneGenerator.TONE_PROP_BEEP2, 50)
        } catch (e: Exception) {
            Log.w(TAG, "voice-sent cue tone failed: ${e.message}")
        }
    }
    // Bundled beep.mp3 asset — the obstacle-ahead alert (ToolDispatcher's
    // pollObstacleAheadOnce()) switched from a synthesized ToneGenerator
    // tone to this per direct user feedback ("not that lightly beep sound,
    // but a like a warning beep" — the ToneGenerator tone wasn't audible at
    // all in practice). SoundPool (not MediaPlayer) — short one-shot SFX,
    // low playback latency, loaded once and replayed on every trigger
    // rather than re-decoding the file per call.
    // USAGE_MEDIA (not USAGE_ASSISTANCE_SONIFICATION) — routes through
    // STREAM_MUSIC, the SAME stream every other audible sound in this app
    // uses (streamingPlayer/pixieController/readingTts/ToneGenerator calls
    // elsewhere all use STREAM_MUSIC) and is controlled by the media volume
    // slider. USAGE_ASSISTANCE_SONIFICATION routes through a system/
    // notification-adjacent stream on many OEMs, which can be silenced
    // independently of media volume (e.g. "touch sounds" off) — the
    // likeliest reason this was silent the first time.
    private val obstacleBeepSoundPool: android.media.SoundPool by lazy {
        android.media.SoundPool.Builder()
            .setMaxStreams(1)
            .setAudioAttributes(
                android.media.AudioAttributes.Builder()
                    .setUsage(android.media.AudioAttributes.USAGE_MEDIA)
                    .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .build()
            .apply {
                setOnLoadCompleteListener { _, sampleId, status ->
                    Log.d(TAG, "obstacle beep asset load complete: sampleId=$sampleId status=$status")
                    if (status == 0) obstacleBeepLoaded = true
                }
            }
    }
    private var obstacleBeepSoundId: Int = 0
    @Volatile private var obstacleBeepLoaded = false

    private fun playObstacleBeep() {
        try {
            if (!obstacleBeepLoaded) {
                Log.w(TAG, "obstacle beep asset not loaded yet — skipping this trigger")
                return
            }
            val streamId = obstacleBeepSoundPool.play(obstacleBeepSoundId, 1f, 1f, 1, 0, 1f)
            if (streamId == 0) Log.w(TAG, "obstacle beep SoundPool.play() returned 0 (failed to play)")
        } catch (e: Exception) {
            Log.w(TAG, "obstacle beep asset playback failed: ${e.message}")
        }
    }

    private val trackingBackend by lazy { TrackingBackend(grpcManager) }
    private val handTracker by lazy { HandTracker(this) }
    // On-device TextToSpeech for reading mode — see ReadingTtsPlayer.kt's
    // own doc comment. Service-scoped (reused across reconnects, same as
    // pixieController/streamingPlayer) — ToolDispatcher.shutdown() only stop()s
    // it per connection; only onDestroy() below actually release()s it.
    private val readingTts by lazy { ReadingTtsPlayer(this) }
    // Client-side latency bridging for walking/guiding's HRTF beacon, AND
    // local-only heading tracking for tracking mode — see CLAUDE.md's
    // "Pixie + Angle modules" note. 1000 features / 480px, heavier than the
    // pixie_hrtf_app test harness's own 300/320 defaults (this app's own
    // deliberate, separate production choice).
    private val angleTracker by lazy { AngleTracker(maxFeatures = 1000, processResolution = 480) }
    private val pdrStepEstimator by lazy { PdrStepEstimator(this) }
    private val deviceToolHandler: DeviceToolHandler = AndroidDeviceToolHandler(this)
    private val memoryStore by lazy { LocalMemoryStore(this) }
    private val sessionState = LiveSessionState()

    // Local (phone camera/mic/speaker) by default; connect() swaps in a
    // RemoteEdgeDevice when the "Use Remote Edge Device" Settings toggle is
    // on. localProcessingJob/angleLumaJob and vadRecorder all read from
    // whichever implementation is currently assigned here, so swapping this
    // field is the single point of control for the toggle.
    var edgeDevice: EdgeDevice = LocalEdgeDevice(cameraManager)
        private set

    private val _uiState = MutableStateFlow(AppUiState())
    val uiState: StateFlow<AppUiState> = _uiState

    private var isLocalTrackingActive = false
    private var lastTrackingUpdateMs = 0L
    private val trackingIntervalMs = 143L // cap local ORB tracking at ~7 fps

    private var trackingGuidanceLastAtMs = 0L
    private val trackingGuidanceIntervalMs = 5000L
    // One-shot: announces the object's rough left/middle/right position
    // ONCE, the first time it's visible with no hand detected yet — see
    // the frame collector's own comment. Reset whenever tracking (re)starts.
    private var trackingInitialPositionAnnounced = false

    private var localProcessingJob: Job? = null
    private var angleLumaJob: Job? = null
    private var initJob: Job? = null
    private var liveSessionJob: Job? = null
    private var liveClient: GeminiLiveClient? = null
    private var toolDispatcher: ToolDispatcher? = null
    private var wakeLock: PowerManager.WakeLock? = null

    // Response-drain timing — see doLiveSession()'s TurnComplete handling
    // and resumeAfterResponse(). Only meaningful while
    // uiState.isAwaitingResponse is true (armed in the VAD's onSpeechEnd).
    private var responseBytesWritten = 0L
    private var responseFirstChunkAtMs = 0L
    private var responseWaitJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        lifecycleScope.launch {
            grpcManager.connectionState.collect { state ->
                _uiState.update { it.copy(connectionState = state) }
            }
        }
        lifecycleScope.launch {
            streamingPlayer.isPlaying.collect { playing ->
                _uiState.update { it.copy(isTtsPlaying = playing) }
            }
        }
        vadRecorder.onVolumeChange = { rms -> _uiState.update { it.copy(micVolume = rms) } }
        // Unconditional — AudioMixer.feedPixie() is itself a no-op while
        // the mixer isn't running (Local edge device), so this tap costs
        // nothing when a remote device isn't in use.
        pixieController.onRenderedChunk = { chunk -> audioMixer.feedPixie(chunk) }
        cameraManager.bind(this)
        // Preload here rather than lazily on first playObstacleBeep() call —
        // SoundPool.load() decodes asynchronously, so a sample requested for
        // the first time right as an alert fires could silently no-op
        // (play() on a not-yet-loaded sound is a documented no-op).
        try {
            assets.openFd("beep.mp3").use { afd ->
                obstacleBeepSoundId = obstacleBeepSoundPool.load(afd, 1)
            }
        } catch (e: Exception) {
            Log.w(TAG, "obstacle beep asset preload failed: ${e.message}")
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        startForeground(
            NOTIF_ID, buildNotification(),
            android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE or
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA,
        )
        if (intent == null) {
            // A null Intent here specifically means the system restarted
            // this Service after the process was killed (START_STICKY) —
            // MainViewModel's own initial start always passes a real Intent
            // object (even with a null action), so this is unambiguous.
            // Nothing else calls connect() automatically, so without this
            // the assistant would silently stop responding until the app
            // is manually reopened. See CLAUDE.md's "Self-echo /
            // output-aware VAD gating" note's issue #3.
            restoreSessionFromPrefsIfAvailable()
        } else {
            when (intent.action) {
                ACTION_INCOMING_CALL -> {
                    val caller = intent.getStringExtra(EXTRA_CALLER_LABEL) ?: "unknown number"
                    liveClient?.sendSystemNote("[SYSTEM] Incoming call from $caller")
                }
                ACTION_SMS_RECEIVED -> {
                    val sender = intent.getStringExtra(EXTRA_SMS_SENDER) ?: "unknown"
                    val body = intent.getStringExtra(EXTRA_SMS_BODY) ?: ""
                    liveClient?.sendSystemNote("[SYSTEM] New SMS from $sender: $body")
                }
            }
        }
        return START_STICKY
    }

    /** Only reconnects if a session was genuinely active when the process
     * died (a persisted flag, not just "prefs happen to have an API key
     * saved") — a user who explicitly disconnected shouldn't be silently
     * reconnected just because the OS happened to kill and restart this
     * Service afterward. */
    private fun restoreSessionFromPrefsIfAvailable() {
        val prefs = getSharedPreferences("tracking_prefs", MODE_PRIVATE)
        if (!prefs.getBoolean("session_was_active", false)) return
        val apiKey = prefs.getString("gemini_api_key", "") ?: ""
        if (apiKey.isBlank()) return
        Log.i(TAG, "Restoring session after Service restart (process was killed)")
        connect(
            host = prefs.getString("server_host", "192.168.1.15") ?: "192.168.1.15",
            port = prefs.getInt("server_port", 50051),
            frameIntervalMs = prefs.getInt("frame_interval_ms", 1000),
            scanIntervalMs = prefs.getInt("scan_interval_ms", 200),
            recentBufferMs = prefs.getInt("recent_buffer_ms", 100),
            avoidanceIntervalMs = prefs.getInt("avoidance_interval_ms", 350),
            beaconElevationDeg = java.lang.Float.intBitsToFloat(prefs.getInt("beacon_elevation_deg_bits", java.lang.Float.floatToIntBits(-20f))),
            beaconRadiusM = java.lang.Float.intBitsToFloat(prefs.getInt("beacon_radius_m_bits", java.lang.Float.floatToIntBits(6f))),
            vadThreshold = java.lang.Float.intBitsToFloat(prefs.getInt("vad_threshold_bits", java.lang.Float.floatToIntBits(0.012f))),
            startThreshold = java.lang.Float.intBitsToFloat(prefs.getInt("start_threshold_bits", java.lang.Float.floatToIntBits(0.018f))),
            geminiApiKey = apiKey,
            ocrApiKey = prefs.getString("ocr_api_key", "helloworld") ?: "helloworld",
            locationId = prefs.getString("location_id", "default") ?: "default",
            blurSharpnessThreshold = java.lang.Float.intBitsToFloat(prefs.getInt("blur_sharpness_threshold_bits", java.lang.Float.floatToIntBits(40f))),
            saveDebugOcrFrames = prefs.getBoolean("save_debug_ocr_frames", false),
            youtubeApiKey = prefs.getString("youtube_api_key", "") ?: "",
            cueVolume = java.lang.Float.intBitsToFloat(prefs.getInt("cue_volume_bits", java.lang.Float.floatToIntBits(1f))),
            geminiVoiceVolume = java.lang.Float.intBitsToFloat(prefs.getInt("gemini_voice_volume_bits", java.lang.Float.floatToIntBits(1f))),
            otherSoundVolume = java.lang.Float.intBitsToFloat(prefs.getInt("other_sound_volume_bits", java.lang.Float.floatToIntBits(1f))),
            useRemoteEdgeDevice = prefs.getBoolean("use_remote_edge_device", false),
            edgeDeviceHost = prefs.getString("edge_device_host", "") ?: "",
        )
    }

    override fun onBind(intent: Intent): IBinder {
        super.onBind(intent)
        return binder
    }

    /** Deliberately a no-op beyond the default — this Service must survive
     * the app being swiped from Recents, unlike PlaybackService's
     * stopSelf()-on-task-removed behavior. */
    override fun onTaskRemoved(rootIntent: Intent?) {
        super.onTaskRemoved(rootIntent)
    }

    private fun buildNotification(): Notification {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "Live Assistant", NotificationManager.IMPORTANCE_LOW
            )
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Assistant running")
            .setContentText("Listening in the background")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .build()
    }

    // ── Bound-client API (mirrors the old MainViewModel surface) ──────────────

    /**
     * Swaps [edgeDevice] to match the "Use Remote Edge Device" toggle, called
     * once at the top of every [connect()]. Tears down whichever
     * implementation was previously active (a no-op for [LocalEdgeDevice],
     * a real ZMQ socket teardown for [RemoteEdgeDevice]) before assigning
     * and connecting the new one — covers both a genuinely fresh connect and
     * a reconnect where the toggle flipped since last time.
     */
    private fun configureEdgeDevice(useRemote: Boolean, edgeHost: String) {
        edgeDevice.disconnect()
        audioMixer.stop()
        remoteEdgeActive = useRemote && edgeHost.isNotBlank()
        _isRemoteEdgeActive.value = remoteEdgeActive
        _edgeFrame.value = null
        edgeDevice = if (remoteEdgeActive) {
            RemoteEdgeDevice(edgeHost)
        } else {
            LocalEdgeDevice(cameraManager)
        }
        edgeDevice.connect()
        if (remoteEdgeActive) {
            audioMixer.onMixedChunk = { bytes -> edgeDevice.emitAudio(bytes) }
            audioMixer.start()
            // Radio/music/resolved-YouTube-stream playback (PlaybackService's
            // own ExoPlayer, including play_youtube_video now — see
            // ToolDispatcher.toolPlayYoutubeVideo()/YouTubeStreamResolver.kt;
            // the old WebView-based IFrame player, and the MediaProjection/
            // AudioPlaybackCaptureConfiguration system-audio-capture path it
            // needed, were both removed outright per direct user request)
            // taps its PCM directly — a real, fully-controlled path, no OS
            // consent dialog, no uncertainty about which AudioAttributes
            // usage a capture filter needs to match. See PlaybackService.kt's
            // TeeRenderersFactory.
            PlaybackService.onPcmTapped = { pcm, rate, channels ->
                audioMixer.feedExoAudio(pcm, rate, channels)
            }
        } else {
            PlaybackService.onPcmTapped = null
        }
    }

    fun connect(
        host: String, port: Int, frameIntervalMs: Int, scanIntervalMs: Int, recentBufferMs: Int,
        avoidanceIntervalMs: Int = 350,
        // Dead params, kept only so MainActivity/MainViewModel/SettingsScreen's
        // existing call signature + persisted prefs don't need touching —
        // HrtfBeaconPlayer's continuous elevation/radius knobs have no
        // equivalent in PixieController's fixed 4-point design (see CLAUDE.md's
        // "Pixie + Angle modules" note). No longer forwarded to ToolDispatcher.
        beaconElevationDeg: Float = -20f, beaconRadiusM: Float = 6f,
        vadThreshold: Float = 0.012f, startThreshold: Float = 0.018f,
        geminiApiKey: String, ocrApiKey: String, locationId: String,
        blurSharpnessThreshold: Float = 40f, saveDebugOcrFrames: Boolean = false,
        youtubeApiKey: String = "",
        cueVolume: Float = 1f,
        geminiVoiceVolume: Float = 1f,
        otherSoundVolume: Float = 1f,
        useRemoteEdgeDevice: Boolean = false,
        edgeDeviceHost: String = "",
    ) {
        acquireWakeLock()
        configureEdgeDevice(useRemoteEdgeDevice, edgeDeviceHost)
        grpcManager.connect(host, port)
        cameraManager.frameIntervalMs = frameIntervalMs
        cameraManager.scanIntervalMs = scanIntervalMs
        cameraManager.recentBufferMs = recentBufferMs
        cameraManager.walkingIntervalMs = avoidanceIntervalMs
        pixieController.cueVolume = cueVolume.coerceIn(0f, 1f)
        streamingPlayer.setVolume(geminiVoiceVolume)
        readingTts.setVolume(otherSoundVolume)

        val ocrClient = OcrClient(ocrApiKey)
        val geminiCorrectionClient = if (geminiApiKey.isNotBlank()) GeminiCorrectionClient(geminiApiKey) else null
        val geminiObjectDescriptionClient = if (geminiApiKey.isNotBlank()) GeminiObjectDescriptionClient(geminiApiKey) else null
        val youtubeSearchClient = if (youtubeApiKey.isNotBlank()) YouTubeSearchClient(youtubeApiKey) else null
        toolDispatcher = ToolDispatcher(
            grpc = grpcManager,
            ocrClient = ocrClient,
            memoryStore = memoryStore,
            deviceToolHandler = deviceToolHandler,
            state = sessionState,
            locationId = locationId,
            avoidanceIntervalMs = avoidanceIntervalMs,
            scope = lifecycleScope,
            latestFrame = { if (remoteEdgeActive) lastEdgeFrame else cameraManager.clearestRecentFrame() },
            sendVideoFrame = { jpeg -> liveClient?.sendVideoFrame(jpeg) },
            sendSystemNote = { text -> liveClient?.sendSystemNote(text) },
            readingTts = readingTts,
            onTrackingStateChanged = { active, target, detectionPrompt ->
                if (active) startLocalTracking(target, detectionPrompt) else stopLocalTracking()
            },
            onGuidanceUpdate = { mode, waypoints ->
                _uiState.update {
                    it.copy(
                        isWalkingMode = mode == "walking",
                        guidingDestination = if (mode == "guiding") sessionState.guidingDestinationLabel else "",
                        guidingRoute = waypoints.map { (x, z) -> "($x, $z)" },
                    )
                }
            },
            pixieController = pixieController,
            angleTracker = angleTracker,
            pdrStepEstimator = pdrStepEstimator,
            latestFrameWithSharpness = {
                if (remoteEdgeActive) {
                    // The Pi doesn't send its own sharpness score over frame_out
                    // (see client/pi_edge/main.py), and re-decoding+scoring the
                    // JPEG here just to blur-filter is more than this needs for
                    // now — treat every edge frame as sharp enough, never
                    // blur-reject/retry it. Real, accepted limitation.
                    lastEdgeFrame?.let { it to Double.MAX_VALUE }
                } else {
                    cameraManager.clearestRecentFrameWithSharpness()
                }
            },
            blurSharpnessThreshold = blurSharpnessThreshold.toDouble(),
            // Full-resolution OCR capture channel — only meaningful (and
            // only wired) for a remote edge device; the phone's own local
            // camera already gives acquireSharpFrame() a real frame via
            // latestFrame()/latestFrameWithSharpness() above with no
            // separate request/response round trip needed. Passing null
            // here for the local case means acquireSharpFrame() skips this
            // path entirely rather than awaiting-and-timing-out on every
            // single OCR call for nothing.
            ocrFrameFlow = if (remoteEdgeActive) edgeDevice.ocrFrameFlow else null,
            requestOcrFrame = { edgeDevice.requestOcrFrame() },
            saveDebugFrame = if (saveDebugOcrFrames) { jpeg ->
                lifecycleScope.launch(Dispatchers.IO) {
                    try {
                        val dir = File(filesDir, "debug_ocr_frames").apply { mkdirs() }
                        File(dir, "frame_${System.currentTimeMillis()}.jpg").writeBytes(jpeg)
                    } catch (e: Exception) {
                        Log.w(TAG, "debug OCR frame save failed: ${e.message}")
                    }
                }
            } else null,
            geminiCorrectionClient = geminiCorrectionClient,
            geminiObjectDescriptionClient = geminiObjectDescriptionClient,
            youtubeSearchClient = youtubeSearchClient,
            reportModeToEdge = { mode -> edgeDevice.reportMode(mode) },
            playObstacleBeep = { playObstacleBeep() },
        )
        // Fresh connection — tell the server to drop any scan/mapping state
        // left over from a previous connection (see ToolDispatcher.
        // resetSession()'s own docstring for why the server can't tell
        // these two cases apart on its own).
        toolDispatcher?.resetSession()

        // Continuous VAD-gated listening — replaces push-to-talk entirely.
        // No ducking on speech START: music/reading keep playing normally
        // through the whole capture window (per the user's explicit spec).
        // Only once the utterance is fully REGISTERED (onSpeechEnd) do we
        // interrupt reading and pause music, since Gemini's response is
        // about to arrive and would otherwise overlap it. See CLAUDE.md's
        // "Continuous VAD-gated listening" note.
        vadRecorder.onSpeechStart = {
            _uiState.update { it.copy(isRecording = true) }
        }
        vadRecorder.onChunkReady = { pcm -> liveClient?.sendAudioChunk(pcm) }
        vadRecorder.onSpeechEnd = {
            val sent = liveClient?.sendAudioStreamEnd() ?: false
            if (sent) playVoiceSentCue()
            _uiState.update { it.copy(isRecording = false, micVolume = 0f, isAwaitingResponse = true) }
            toolDispatcher?.interruptReadingForUserTurn()
            responseBytesWritten = 0L
            responseWaitJob?.cancel(); responseWaitJob = null
            startService(Intent(this, com.tracking.client.device.PlaybackService::class.java).setAction(
                com.tracking.client.device.PlaybackService.ACTION_PAUSE
            ))
        }
        // Output-aware VAD gating (defense-in-depth on top of the real
        // AEC ContinuousVadRecorder itself sets up) — covers every audio
        // output source in this app: Gemini's own spoken reply + reading
        // TTS (streamingPlayer), Pixie's own audio cue, and PlaybackService's
        // music/radio/YouTube playback (the old separate _isYoutubePlaying
        // signal is gone — YouTube now plays through PlaybackService like
        // everything else, so PlaybackService.isPlaying already covers it).
        // See CLAUDE.md's "Self-echo / output-aware VAD gating" note.
        vadRecorder.start(
            startThreshold, vadThreshold,
            isOutputActive = {
                streamingPlayer.isPlaying.value ||
                    pixieController.isEmitting ||
                    com.tracking.client.device.PlaybackService.isPlaying.value
            },
            // Remote edge device's mic replaces local AudioRecord capture
            // entirely when active — LocalEdgeDevice.micFlow is an unused
            // empty flow, so this only actually changes behavior for
            // RemoteEdgeDevice (see ContinuousVadRecorder.start()'s doc).
            externalChunks = if (useRemoteEdgeDevice) edgeDevice.micFlow else null,
        )

        startLocalProcessing()
        startLiveSession(geminiApiKey)
        _uiState.update { it.copy(isVadActive = true) }
        appendSystemMessage("Connecting to $host:$port …")
        getSharedPreferences("tracking_prefs", MODE_PRIVATE).edit()
            .putBoolean("session_was_active", true).apply()
    }

    fun disconnect() {
        localProcessingJob?.cancel(); localProcessingJob = null
        angleLumaJob?.cancel(); angleLumaJob = null
        liveSessionJob?.cancel(); liveSessionJob = null
        liveClient?.close(); liveClient = null
        toolDispatcher?.shutdown(); toolDispatcher = null
        initJob?.cancel()
        vadRecorder.stop()
        edgeDevice.disconnect()
        audioMixer.stop()
        responseWaitJob?.cancel(); responseWaitJob = null
        grpcManager.disconnect()
        sessionState.reset()
        _edgeFrame.value = null
        _isRemoteEdgeActive.value = false
        _uiState.update {
            it.copy(
                isVadActive = false, isRecording = false, isAwaitingResponse = false,
                connectionState = ConnectionState.DISCONNECTED,
            )
        }
        appendSystemMessage("Disconnected")
        releaseWakeLock()
        getSharedPreferences("tracking_prefs", MODE_PRIVATE).edit()
            .putBoolean("session_was_active", false).apply()
    }

    /** Called once Gemini's response has finished (estimated) playing out —
     * see doLiveSession()'s TurnComplete handling. Resumes music, but
     * deliberately NOT reading — reading only resumes via the explicit
     * continue_reading() tool call. */
    private fun resumeAfterResponse() {
        _uiState.update { it.copy(isAwaitingResponse = false) }
        responseBytesWritten = 0L
        startService(Intent(this, com.tracking.client.device.PlaybackService::class.java).setAction(
            com.tracking.client.device.PlaybackService.ACTION_RESUME
        ))
    }

    /** [label] is the plain target name (display/status only); [detectionPrompt]
     * is what actually gets sent to GroundingDINO — a richer appearance
     * description when start_tracking() was called with one (see
     * ToolDispatcher.toolStartTracking()'s own doc comment), otherwise the
     * same as [label]. */
    fun startLocalTracking(label: String, detectionPrompt: String = label) {
        isLocalTrackingActive = true
        trackingInitialPositionAnnounced = false
        _uiState.update { it.copy(agentName = "tracking", agentState = "INITIALIZING") }
        appendSystemMessage("Searching for '$label'…")

        initJob?.cancel()
        initJob = lifecycleScope.launch(Dispatchers.IO) {
            while (isActive) {
                val frame = if (remoteEdgeActive) lastEdgeFrame else cameraManager.clearestRecentFrame()
                if (frame == null) { delay(100); continue }
                Log.d(TAG, "initialize attempt for '$label' (prompt='$detectionPrompt')")
                val track = trackingBackend.initialize(frame, detectionPrompt)
                if (track != null) {
                    Log.d(TAG, "Tracking initialized: box=${track.boxXyxy.toList()}")
                    _uiState.update { it.copy(agentState = "TRACKING") }
                    break
                }
                Log.d(TAG, "Detection failed, retrying in 1s")
                delay(1000L)
            }
        }
    }

    fun stopLocalTracking() {
        initJob?.cancel()
        initJob = null
        isLocalTrackingActive = false
        trackingBackend.stop()
        _uiState.update { it.copy(agentState = "STOPPED", guidanceData = ObjectTrack(status = "Local tracking stopped")) }
        appendSystemMessage("Local tracking stopped")
    }

    // ── Local frame processing (tracking + hand detection + UI + mapping feed) ──

    private fun startLocalProcessing() {
        localProcessingJob?.cancel()
        localProcessingJob = lifecycleScope.launch(Dispatchers.IO) {
            edgeDevice.frameFlow
                .conflate()
                .catch { e -> appendSystemMessage("[Flow error] ${e.message}") }
                .collect { jpegBytes ->
                    lastEdgeFrame = jpegBytes
                    if (remoteEdgeActive) _edgeFrame.value = jpegBytes
                    val mappingModeActive = sessionState.mode == "guiding" || sessionState.mode == "walking" || sessionState.mode == "scanning"
                    cameraManager.mappingMode = if (mappingModeActive) sessionState.mode else ""
                    if (mappingModeActive) {
                        toolDispatcher?.feedMappingFrame(jpegBytes)
                    }

                    val jpegOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, jpegOpts)
                    val frameWidth = jpegOpts.outWidth
                    val frameHeight = jpegOpts.outHeight

                    // Hand detection hoisted ahead of the tracking-object
                    // update below (pure computation only here — the
                    // _uiState merge stays in its original position further
                    // down, unchanged) so updateTrackingPixie() can steer
                    // relative to the HAND's actual position from this same
                    // frame, not the frame center — see that function's own
                    // doc comment for why this matters (a real reported bug:
                    // tracking mode is meant to guide the hand to the
                    // object, not the view/camera direction).
                    // Only actually run MediaPipe hand detection while tracking
                    // mode is active -- every consumer of handBox/handLmX/handLmY
                    // below (TrackingBackend's occlusion check, updateTrackingPixie(),
                    // the one-shot position announcement) is tracking-mode-only,
                    // and the UI overlay that used to display this data was
                    // already removed from MainScreen.kt (see CLAUDE.md's "UI
                    // simplified" note) -- so running an ML inference pass on
                    // EVERY incoming frame regardless of mode was pure wasted
                    // CPU, a real contributor to reported lag outside tracking
                    // mode (Q&A/idle), found via a live device report.
                    val handLmX: List<List<Float>>
                    val handLmY: List<List<Float>>
                    val handBox: List<Float>
                    if (isLocalTrackingActive) {
                        val handResult = try { handTracker.detect(jpegBytes) } catch (e: Exception) { null }
                        if (handResult != null && handResult.hands.isNotEmpty() && frameWidth > 0 && frameHeight > 0) {
                            handLmX = handResult.hands.map { hand -> hand.map { it.first * frameWidth } }
                            handLmY = handResult.hands.map { hand -> hand.map { it.second * frameHeight } }
                            val allX = handLmX.flatten(); val allY = handLmY.flatten()
                            handBox = listOf(allX.min(), allY.min(), allX.max(), allY.max())
                        } else {
                            handLmX = emptyList(); handLmY = emptyList(); handBox = emptyList()
                        }
                    } else {
                        handLmX = emptyList(); handLmY = emptyList(); handBox = emptyList()
                    }

                    val now = System.currentTimeMillis()
                    if (isLocalTrackingActive && _uiState.value.agentState == "TRACKING" &&
                        now - lastTrackingUpdateMs >= trackingIntervalMs
                    ) {
                        lastTrackingUpdateMs = now
                        try {
                            // handBox (computed above, same frame) — skips
                            // this cycle's re-identify DetectObject call
                            // while the hand overlaps the target (likely
                            // occluding it, so the redetect would just see a
                            // wrong/blocked view). See TrackingBackend.
                            // update()'s own doc comment.
                            val track = trackingBackend.update(jpegBytes, handBox)
                            if (track != null) {
                                Log.d(TAG, "TrackingBackend: visible=${track.visible} conf=${track.confidence} box=${track.boxXyxy.toList()}")
                                val guidance = track.copy(
                                    instruction = when {
                                        !track.visible -> "Target lost"
                                        track.centerX < track.frameWidth * 0.2f -> "Move right"
                                        track.centerX > track.frameWidth * 0.8f -> "Move left"
                                        track.centerY < track.frameHeight * 0.2f -> "Move down"
                                        track.centerY > track.frameHeight * 0.8f -> "Move up"
                                        else -> "On target"
                                    },
                                    objectBoxXyxy = track.boxXyxy.toList(),
                                    deltaX = track.centerX - (track.frameWidth / 2f),
                                    deltaY = track.centerY - (track.frameHeight / 2f)
                                )
                                _uiState.update { it.copy(guidanceData = guidance) }
                                val handCenterX = if (handBox.size == 4) (handBox[0] + handBox[2]) / 2f else 0f
                                val handCenterY = if (handBox.size == 4) (handBox[1] + handBox[3]) / 2f else 0f
                                toolDispatcher?.updateTrackingPixie(
                                    objectVisible = track.visible, objectCenterX = track.centerX, objectCenterY = track.centerY,
                                    handVisible = handBox.size == 4, handCenterX = handCenterX, handCenterY = handCenterY,
                                    frameWidth = track.frameWidth, frameHeight = track.frameHeight,
                                )

                                // One-shot: the FIRST time the object becomes
                                // visible this tracking session, if no hand is
                                // in frame yet, tell the user its rough
                                // left/middle/right position so they know
                                // which way to start reaching — requested
                                // directly by the user, in place of the old
                                // clock-position phrasing. Marked "announced"
                                // on this first sighting regardless (even if a
                                // hand IS already visible) so it never fires
                                // again later this session.
                                if (track.visible && !trackingInitialPositionAnnounced) {
                                    trackingInitialPositionAnnounced = true
                                    if (handBox.isEmpty()) {
                                        val third = track.frameWidth / 3f
                                        val pos = when {
                                            track.centerX < third -> "left"
                                            track.centerX > third * 2 -> "right"
                                            else -> "middle"
                                        }
                                        liveClient?.sendSystemNote(
                                            "[SYSTEM] Target object located, roughly at your $pos (no hand " +
                                                "detected yet). Briefly tell the user its general position " +
                                                "(left/middle/right) — plain words only, no clock positions — " +
                                                "so they know which way to start reaching. Say this once."
                                        )
                                    }
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Tracking error: ${e.message}", e)
                        }
                    }

                    _uiState.update { s ->
                        s.copy(guidanceData = s.guidanceData.copy(
                            frameWidth = frameWidth,
                            frameHeight = frameHeight,
                            handBoxXyxy = handBox,
                            handLandmarksX = handLmX,
                            handLandmarksY = handLmY,
                        ))
                    }

                    if (sessionState.mode == "tracking" && handBox.isNotEmpty()) {
                        val g = _uiState.value.guidanceData
                        if (g.visible && g.objectBoxXyxy.size == 4 &&
                            now - trackingGuidanceLastAtMs >= trackingGuidanceIntervalMs
                        ) {
                            trackingGuidanceLastAtMs = now
                            // Real bug fix: this used to hand Gemini the raw
                            // box coordinates and ask IT to work out whether
                            // the hand had arrived / which way to move — LLMs
                            // are unreliable at that kind of precise box-
                            // overlap arithmetic from two lists of numbers,
                            // which is exactly why it kept saying "move" even
                            // once the hand was already on the target. Now
                            // the arrived/direction judgment is computed HERE
                            // (same box-overlap + offset approach Pixie's own
                            // audio cue already uses) and Gemini is just told
                            // the answer, not asked to derive it.
                            val obj = g.objectBoxXyxy
                            val handCenterX = (handBox[0] + handBox[2]) / 2f
                            val handCenterY = (handBox[1] + handBox[3]) / 2f
                            val objCenterX = (obj[0] + obj[2]) / 2f
                            val objCenterY = (obj[1] + obj[3]) / 2f
                            val boxesOverlap = obj[0] < handBox[2] && obj[2] > handBox[0] &&
                                obj[1] < handBox[3] && obj[3] > handBox[1]

                            val dx = objCenterX - handCenterX
                            val dy = objCenterY - handCenterY
                            val hDeadzone = frameWidth * 0.05f
                            val vDeadzone = frameHeight * 0.05f
                            val parts = mutableListOf<String>()
                            if (kotlin.math.abs(dx) > hDeadzone) {
                                val mag = if (kotlin.math.abs(dx) > frameWidth * 0.25f) "a lot" else "a bit"
                                parts.add("${if (dx > 0) "right" else "left"} ($mag)")
                            }
                            if (kotlin.math.abs(dy) > vDeadzone) {
                                val mag = if (kotlin.math.abs(dy) > frameHeight * 0.25f) "a lot" else "a bit"
                                // +dy = target below hand in image coords (y grows down) -> hand must move down
                                parts.add("${if (dy > 0) "down" else "up"} ($mag)")
                            }
                            val arrived = boxesOverlap || parts.isEmpty()

                            val note = if (arrived) {
                                "[SYSTEM] Hand-to-target: ARRIVED — the hand is now on the target. Say a " +
                                    "short confirmation like \"got it!\" or \"right there\", nothing more."
                            } else {
                                "[SYSTEM] Hand-to-target direction (already computed — do not recompute from " +
                                    "coordinates): move " + parts.joinToString(" and ") + ". Say ONLY a short " +
                                    "cue matching this exactly (e.g. \"move left\", \"a bit down\"), plain " +
                                    "words, NEVER clock positions."
                            }
                            liveClient?.sendSystemNote(note)
                        }
                    }
                }
        }

        // AngleTracker wants a continuous per-frame trickle (good frame-to-
        // frame ORB matching), independent of the interval-gated mapping-mode
        // push above — same reasoning the old feedRotationFrame() call had,
        // just fed from CameraManager's additive lumaFlow (raw Y-plane, no
        // JPEG round trip — see AngleTracker.kt's own docstring) instead of
        // frameFlow's JPEGs. No-ops internally unless WALKING/GUIDING is active.
        angleLumaJob?.cancel()
        angleLumaJob = lifecycleScope.launch(Dispatchers.IO) {
            edgeDevice.lumaFlow
                .catch { e -> Log.w(TAG, "lumaFlow error: ${e.message}") }
                .collect { f ->
                    toolDispatcher?.feedAngleLumaFrame(f.luma, f.width, f.height, f.rowStride, f.rotationDegrees)
                }
        }
    }

    // ── Persistent Gemini Live session (direct on-device connection) ─────────

    private fun startLiveSession(apiKey: String) {
        liveSessionJob?.cancel()
        liveSessionJob = lifecycleScope.launch(Dispatchers.IO) {
            while (isActive) {
                if (apiKey.isBlank()) {
                    appendSystemMessage("[Voice] No Gemini API key configured — set one in Settings.")
                    delay(5000L)
                    continue
                }
                doLiveSession(apiKey)
                if (!isActive) break
                delay(2000L)
            }
        }
    }

    private suspend fun doLiveSession(apiKey: String) {
        val client = GeminiLiveClient(apiKey)
        // Every [SYSTEM]-tagged note is now an interrupting send — see
        // GeminiLiveClient.onInterrupt's own doc comment. Flushing here
        // (not tearing the track down) is what stops a still-playing
        // response from talking over/after the new one.
        client.onInterrupt = { streamingPlayer.interrupt() }
        liveClient = client
        streamingPlayer.start()
        try {
            client.events(ToolDeclarations.SYSTEM_PROMPT, ToolDeclarations.buildDeclarations()).collect { event ->
                when (event) {
                    is LiveServerEvent.SetupComplete -> {
                        Log.d(TAG, "Gemini Live setup complete")
                        appendSystemMessage("Connected to Gemini Live")
                        // Requested directly by the user: give the user an
                        // audible cue the session is actually live, not
                        // just silence until they happen to speak first.
                        // Same [SYSTEM]-event convention as incoming-call/
                        // SMS notes (CORE RULES: respond to these
                        // immediately) — sendSystemNote() forces immediate
                        // generation (turnComplete=true) regardless of VAD
                        // state, so this reliably triggers a spoken turn.
                        client.sendSystemNote(
                            "[SYSTEM] The session just connected and is ready. Briefly say you're ready " +
                                "to help (just a few words), then wait for the user."
                        )
                    }
                    is LiveServerEvent.Audio -> {
                        streamingPlayer.writeChunk(event.pcm)
                        // Remote: fold into AudioMixer's combined stream
                        // (see configureEdgeDevice()) instead of forwarding
                        // this voice-only chunk directly — the mixer is
                        // what actually calls edgeDevice.emitAudio() once
                        // it's combined with Pixie's cue. Local: unchanged,
                        // direct emitAudio() (feeds an unused flow, kept
                        // for interface symmetry — see LocalEdgeDevice.kt).
                        if (remoteEdgeActive) audioMixer.feedVoice(event.pcm) else edgeDevice.emitAudio(event.pcm)
                        if (_uiState.value.isAwaitingResponse) {
                            if (responseBytesWritten == 0L) responseFirstChunkAtMs = System.currentTimeMillis()
                            responseBytesWritten += event.pcm.size
                        }
                    }
                    is LiveServerEvent.ToolCall -> {
                        for (call in event.calls) {
                            lifecycleScope.launch(Dispatchers.IO) {
                                val dispatcher = toolDispatcher ?: return@launch
                                val response = dispatcher.dispatch(call.name, call.args)
                                Log.d(TAG, "tool ${call.name} -> $response")
                                client.sendToolResponse(call.id, call.name, response)
                            }
                        }
                    }
                    is LiveServerEvent.TurnComplete -> {
                        // See resumeAfterResponse() — estimates remaining
                        // playback time from bytes-written/sample-rate
                        // rather than polling the AudioTrack's real
                        // playback-head position (a documented, accepted
                        // approximation — see CLAUDE.md).
                        if (_uiState.value.isAwaitingResponse) {
                            val totalDurationMs = responseBytesWritten * 1000L / BYTES_PER_SEC_24K
                            val elapsedMs = System.currentTimeMillis() - responseFirstChunkAtMs
                            val remainingMs = (totalDurationMs - elapsedMs).coerceAtLeast(0L) + DRAIN_MARGIN_MS
                            responseWaitJob?.cancel()
                            responseWaitJob = lifecycleScope.launch {
                                delay(remainingMs)
                                resumeAfterResponse()
                            }
                        }
                    }
                    is LiveServerEvent.Interrupted -> { /* no-op */ }
                    is LiveServerEvent.Error -> {
                        Log.e(TAG, "Live session error: ${event.message}")
                        appendSystemMessage("[Session] Reconnecting…")
                    }
                    is LiveServerEvent.Closed -> {
                        Log.d(TAG, "Live session closed")
                    }
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "liveSession error: ${e.message}")
            appendSystemMessage("[Session] Reconnecting…")
        } finally {
            client.close()
            if (liveClient === client) liveClient = null
            streamingPlayer.stop()
            _uiState.update { it.copy(isTtsPlaying = false) }
        }
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    private fun appendSystemMessage(text: String) {
        _uiState.update { state -> state.copy(chatHistory = state.chatHistory + ChatMessage("system", text)) }
    }

    fun clearError() { _uiState.update { it.copy(error = null) } }


    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "LiveAssistantService:session").apply {
            setReferenceCounted(false)
            acquire(12 * 60 * 60 * 1000L /* 12h safety cap */)
        }
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    override fun onDestroy() {
        initJob?.cancel()
        liveSessionJob?.cancel()
        localProcessingJob?.cancel()
        angleLumaJob?.cancel()
        liveClient?.close()
        toolDispatcher?.shutdown()
        vadRecorder.stop()
        edgeDevice.disconnect()
        audioMixer.stop()
        responseWaitJob?.cancel()
        if (isLocalTrackingActive) trackingBackend.stop()
        handTracker.close()
        streamingPlayer.stop()
        readingTts.release()
        cameraManager.shutdown()
        grpcManager.disconnect()
        releaseWakeLock()
        voiceSentToneGenerator?.release()
        voiceSentToneGenerator = null
        obstacleBeepSoundPool.release()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "LiveAssistantService"
        private const val NOTIF_ID = 101
        private const val CHANNEL_ID = "live_assistant"

        // Gemini Live audio output: 24kHz, 16-bit mono — see
        // resumeAfterResponse()'s response-drain estimate. Real AEC (see
        // ContinuousVadRecorder) is now the primary defense against this
        // window's tail-end echo being misheard as a new utterance — this
        // margin is just a small extra cushion, not load-bearing on its own.
        private const val BYTES_PER_SEC_24K = 24000L * 2L
        private const val DRAIN_MARGIN_MS = 350L

        const val ACTION_INCOMING_CALL = "com.tracking.client.action.INCOMING_CALL"
        const val EXTRA_CALLER_LABEL = "caller_label"
        const val ACTION_SMS_RECEIVED = "com.tracking.client.action.SMS_RECEIVED"
        const val EXTRA_SMS_SENDER = "sms_sender"
        const val EXTRA_SMS_BODY = "sms_body"
    }
}
