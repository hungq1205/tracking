package com.tracking.client.ui

import android.app.Application
import android.content.Context
import android.graphics.BitmapFactory
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.tracking.client.audio.HrtfBeaconPlayer
import com.tracking.client.audio.PushToTalkRecorder
import com.tracking.client.audio.StreamingAudioPlayer
import com.tracking.client.camera.CameraManager
import com.tracking.client.device.AndroidDeviceToolHandler
import com.tracking.client.device.DeviceToolHandler
import com.tracking.client.edge.LocalEdgeDevice
import com.tracking.client.grpc.GrpcClientManager
import com.tracking.client.live.GeminiLiveClient
import com.tracking.client.live.LiveServerEvent
import com.tracking.client.live.LiveSessionState
import com.tracking.client.live.LocalMemoryStore
import com.tracking.client.live.OcrClient
import com.tracking.client.live.ToolDeclarations
import com.tracking.client.live.ToolDispatcher
import com.tracking.client.model.AppUiState
import com.tracking.client.model.ChatMessage
import com.tracking.client.model.ConnectionState
import com.tracking.client.model.ObjectTrack
import com.tracking.client.tracking.HandTracker
import com.tracking.client.tracking.TrackingBackend
import android.util.Log
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

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences("tracking_prefs", Context.MODE_PRIVATE)

    val grpcManager = GrpcClientManager()
    val cameraManager = CameraManager(app)
    private val ptt = PushToTalkRecorder()
    private val streamingPlayer = StreamingAudioPlayer()
    private val hrtfBeacon = HrtfBeaconPlayer(app)
    private val trackingBackend by lazy { TrackingBackend(grpcManager) }
    private val handTracker by lazy { HandTracker(getApplication()) }
    private val deviceToolHandler: DeviceToolHandler = AndroidDeviceToolHandler(app)
    private val memoryStore by lazy { LocalMemoryStore(app) }
    private val sessionState = LiveSessionState()

    val edgeDevice: LocalEdgeDevice = LocalEdgeDevice(cameraManager)

    private val _uiState = MutableStateFlow(AppUiState())
    val uiState: StateFlow<AppUiState> = _uiState

    private var isLocalTrackingActive = false
    private var localTrackingPrompt = ""
    private var lastTrackingUpdateMs = 0L
    private val trackingIntervalMs = 143L // cap local ORB tracking at ~7 fps

    // Hand-guidance [SYSTEM] tick while tracking (mirrors the old server's
    // tracking_guidance_active/tracking_last_guidance_at — now client-side
    // since Gemini Live runs here) — fires once both target+hand are visible,
    // then every 5s while both remain visible.
    private var trackingGuidanceLastAtMs = 0L
    private val trackingGuidanceIntervalMs = 5000L

    private var localProcessingJob: Job? = null
    private var initJob: Job? = null
    private var liveSessionJob: Job? = null
    private var liveClient: GeminiLiveClient? = null
    private var toolDispatcher: ToolDispatcher? = null

    init {
        viewModelScope.launch {
            grpcManager.connectionState.collect { state ->
                _uiState.update { it.copy(connectionState = state) }
            }
        }
        viewModelScope.launch {
            streamingPlayer.isPlaying.collect { playing ->
                _uiState.update { it.copy(isTtsPlaying = playing) }
            }
        }
        ptt.onVolumeChange = { rms -> _uiState.update { it.copy(micVolume = rms) } }
    }

    fun connect(
        host: String, port: Int, frameIntervalMs: Int, scanIntervalMs: Int, recentBufferMs: Int,
        avoidanceIntervalMs: Int = 350,
        vadThreshold: Float = 0.03f, startThreshold: Float = 0.05f,
        geminiApiKey: String, ocrServerUrl: String, locationId: String,
    ) {
        grpcManager.connect(host, port)
        cameraManager.frameIntervalMs = frameIntervalMs
        cameraManager.scanIntervalMs = scanIntervalMs
        cameraManager.recentBufferMs = recentBufferMs

        val ocrClient = OcrClient(ocrServerUrl)
        toolDispatcher = ToolDispatcher(
            grpc = grpcManager,
            ocrClient = ocrClient,
            memoryStore = memoryStore,
            deviceToolHandler = deviceToolHandler,
            state = sessionState,
            locationId = locationId,
            avoidanceIntervalMs = avoidanceIntervalMs,
            scope = viewModelScope,
            latestFrame = { cameraManager.clearestRecentFrame() },
            sendVideoFrame = { jpeg -> liveClient?.sendVideoFrame(jpeg) },
            sendSystemNote = { text -> liveClient?.sendSystemNote(text) },
            playPcm = { pcm -> streamingPlayer.writeChunk(pcm); edgeDevice.emitAudio(pcm) },
            onTrackingStateChanged = { active, target ->
                if (active) startLocalTracking(target) else stopLocalTracking()
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
            hrtfBeacon = hrtfBeacon,
        )

        startLocalProcessing()
        startLiveSession(geminiApiKey)
        _uiState.update { it.copy(isVadActive = true) }
        appendSystemMessage("Connecting to $host:$port …")
    }

    fun disconnect() {
        localProcessingJob?.cancel(); localProcessingJob = null
        liveSessionJob?.cancel(); liveSessionJob = null
        liveClient?.close(); liveClient = null
        toolDispatcher?.shutdown(); toolDispatcher = null
        initJob?.cancel()
        if (_uiState.value.isRecording) ptt.stopRecording()
        grpcManager.disconnect()
        sessionState.reset()
        _uiState.update { it.copy(isVadActive = false, isRecording = false, connectionState = ConnectionState.DISCONNECTED) }
        appendSystemMessage("Disconnected")
    }

    fun startPtt() {
        _uiState.update { it.copy(isRecording = true) }
        ptt.onChunkReady = { pcm -> liveClient?.sendAudioChunk(pcm) }
        ptt.startRecording()
    }

    fun stopPtt() {
        ptt.stopRecording()
        ptt.onChunkReady = null
        liveClient?.sendAudioStreamEnd()
        _uiState.update { it.copy(isRecording = false, micVolume = 0f) }
    }

    fun startLocalTracking(prompt: String) {
        isLocalTrackingActive = true
        localTrackingPrompt = prompt
        _uiState.update { it.copy(agentName = "tracking", agentState = "INITIALIZING") }
        appendSystemMessage("Searching for '$prompt'…")

        initJob?.cancel()
        initJob = viewModelScope.launch(Dispatchers.IO) {
            while (isActive) {
                val frame = cameraManager.clearestRecentFrame()
                if (frame == null) { delay(100); continue }
                Log.d(TAG, "initialize attempt for '$prompt'")
                val track = trackingBackend.initialize(frame, prompt)
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
        localProcessingJob = viewModelScope.launch(Dispatchers.IO) {
            cameraManager.frameFlow
                .conflate()
                .catch { e -> appendSystemMessage("[Flow error] ${e.message}") }
                .collect { jpegBytes ->
                    // Guiding/scanning mode: feed frames into the live mapping
                    // stream (MappingService.UpdateMapping) — see
                    // ToolDispatcher.feedMappingFrame. Walking is NOT included
                    // any more — it dropped MappingService/RTAB-Map entirely
                    // in favor of a local per-frame reactive obstacle-dodge
                    // (see ToolDispatcher.runAvoidanceTick(), which pulls its
                    // own frames on demand via clearestRecentFrame() instead).
                    // sessionState.mode is forwarded as-is into
                    // CameraManager.mappingMode so it can pick the right
                    // window size per submode (frameIntervalMs for guiding,
                    // scanIntervalMs for scanning) — see CameraManager.kt's
                    // frame-selection note. Set here rather than from
                    // ToolDispatcher so no new callback wiring is needed;
                    // cheap to re-check every processed frame.
                    val mappingModeActive = sessionState.mode == "guiding" || sessionState.mode == "scanning"
                    cameraManager.mappingMode = if (mappingModeActive) sessionState.mode else ""
                    if (mappingModeActive) {
                        toolDispatcher?.feedMappingFrame(jpegBytes)
                    }

                    val jpegOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, jpegOpts)
                    val frameWidth = jpegOpts.outWidth
                    val frameHeight = jpegOpts.outHeight

                    // Local ORB tracking — throttled to ~7 fps
                    val now = System.currentTimeMillis()
                    if (isLocalTrackingActive && _uiState.value.agentState == "TRACKING" &&
                        now - lastTrackingUpdateMs >= trackingIntervalMs
                    ) {
                        lastTrackingUpdateMs = now
                        try {
                            val track = trackingBackend.update(jpegBytes)
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
                                toolDispatcher?.updateTrackingBeacon(
                                    track.visible, track.centerX, track.centerY, track.frameWidth, track.frameHeight
                                )
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Tracking error: ${e.message}", e)
                        }
                    }

                    // Hand detection — MediaPipe normalized [0,1] coords
                    val handResult = try { handTracker.detect(jpegBytes) } catch (e: Exception) { null }
                    val handLmX: List<List<Float>>
                    val handLmY: List<List<Float>>
                    val handBox: List<Float>
                    if (handResult != null && handResult.hands.isNotEmpty() && frameWidth > 0 && frameHeight > 0) {
                        handLmX = handResult.hands.map { hand -> hand.map { it.first * frameWidth } }
                        handLmY = handResult.hands.map { hand -> hand.map { it.second * frameHeight } }
                        val allX = handLmX.flatten(); val allY = handLmY.flatten()
                        handBox = listOf(allX.min(), allY.min(), allX.max(), allY.max())
                    } else {
                        handLmX = emptyList(); handLmY = emptyList(); handBox = emptyList()
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

                    // Tracking mode hand-guidance [SYSTEM] tick — once both
                    // target and hand are visible, then every 5s while both
                    // remain visible (mirrors the old server's
                    // tracking_guidance_active behavior, now computed
                    // entirely on-device).
                    if (sessionState.mode == "tracking" && handBox.isNotEmpty()) {
                        val g = _uiState.value.guidanceData
                        if (g.visible && g.objectBoxXyxy.size == 4 &&
                            now - trackingGuidanceLastAtMs >= trackingGuidanceIntervalMs
                        ) {
                            trackingGuidanceLastAtMs = now
                            liveClient?.sendSystemNote(
                                "[SYSTEM] Target box=${g.objectBoxXyxy}, hand box=$handBox. " +
                                    "Give brief directional guidance to move the hand toward the target."
                            )
                        }
                    }
                }
        }
    }

    // ── Persistent Gemini Live session (direct on-device connection) ─────────

    private fun startLiveSession(apiKey: String) {
        liveSessionJob?.cancel()
        liveSessionJob = viewModelScope.launch(Dispatchers.IO) {
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
        liveClient = client
        streamingPlayer.start()
        try {
            client.events(ToolDeclarations.SYSTEM_PROMPT, ToolDeclarations.buildDeclarations()).collect { event ->
                when (event) {
                    is LiveServerEvent.SetupComplete -> {
                        Log.d(TAG, "Gemini Live setup complete")
                        appendSystemMessage("Connected to Gemini Live")
                    }
                    is LiveServerEvent.Audio -> {
                        streamingPlayer.writeChunk(event.pcm)
                        edgeDevice.emitAudio(event.pcm)
                    }
                    is LiveServerEvent.ToolCall -> {
                        for (call in event.calls) {
                            viewModelScope.launch(Dispatchers.IO) {
                                val dispatcher = toolDispatcher ?: return@launch
                                val response = dispatcher.dispatch(call.name, call.args)
                                Log.d(TAG, "tool ${call.name} -> $response")
                                client.sendToolResponse(call.id, call.name, response)
                            }
                        }
                    }
                    is LiveServerEvent.TurnComplete, is LiveServerEvent.Interrupted -> { /* no-op */ }
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

    companion object {
        private const val TAG = "MainViewModel"
    }

    override fun onCleared() {
        super.onCleared()
        initJob?.cancel()
        liveSessionJob?.cancel()
        localProcessingJob?.cancel()
        liveClient?.close()
        toolDispatcher?.shutdown()
        if (_uiState.value.isRecording) ptt.stopRecording()
        if (isLocalTrackingActive) trackingBackend.stop()
        handTracker.close()
        streamingPlayer.stop()
        cameraManager.shutdown()
        grpcManager.disconnect()
    }
}
