package com.tracking.client.ui

import android.app.Application
import android.content.Context
import android.graphics.BitmapFactory
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.protobuf.ByteString
import com.tracking.client.audio.PushToTalkRecorder
import com.tracking.client.audio.StreamingAudioPlayer
import com.tracking.client.camera.CameraManager
import com.tracking.client.edge.LocalEdgeDevice
import com.tracking.client.edge.SessionResult
import com.tracking.client.grpc.GrpcClientManager
import com.tracking.client.model.AppUiState
import com.tracking.client.model.ChatMessage
import com.tracking.client.model.ConnectionState
import com.tracking.client.model.ObjectTrack
import com.tracking.client.sensors.ImuSensor
import com.tracking.client.tracking.HandTracker
import com.tracking.client.tracking.TrackingBackend
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import tracking.Tracking

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences("tracking_prefs", Context.MODE_PRIVATE)

    val grpcManager = GrpcClientManager()
    val cameraManager = CameraManager(app)
    private val imuSensor = ImuSensor(app)
    private val ptt = PushToTalkRecorder()
    private val streamingPlayer = StreamingAudioPlayer()
    private val trackingBackend by lazy { TrackingBackend(grpcManager) }
    private val handTracker by lazy { HandTracker(getApplication()) }

    val edgeDevice: LocalEdgeDevice = LocalEdgeDevice(cameraManager)

    private val _uiState = MutableStateFlow(AppUiState())
    val uiState: StateFlow<AppUiState> = _uiState

    private var isLocalTrackingActive = false
    private var localTrackingPrompt = ""

    // Latest JPEG bytes for local tracking initialization retries
    @Volatile private var latestFrameBytes: ByteArray? = null

    private var localProcessingJob: Job? = null
    private var initJob: Job? = null
    private var liveSessionJob: Job? = null

    // Persistent audio channel for the lifetime of a connection.
    // PTT writes PCM chunks; the live session reads them. Closed on disconnect.
    private var audioChannel: Channel<ByteArray>? = null

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

    fun connect(host: String, port: Int, fps: Int, vadThreshold: Float = 0.03f, startThreshold: Float = 0.05f) {
        grpcManager.connect(host, port)
        cameraManager.targetFps = fps
        val ch = Channel<ByteArray>(Channel.UNLIMITED)
        audioChannel = ch
        startLocalProcessing()
        startLiveSession(ch)
        _uiState.update { it.copy(isVadActive = true) }
        appendSystemMessage("Connecting to $host:$port …")
    }

    fun disconnect() {
        localProcessingJob?.cancel(); localProcessingJob = null
        liveSessionJob?.cancel(); liveSessionJob = null
        audioChannel?.close(); audioChannel = null
        initJob?.cancel()
        if (_uiState.value.isRecording) ptt.stopRecording()
        grpcManager.disconnect()
        _uiState.update { it.copy(isVadActive = false, isRecording = false, connectionState = ConnectionState.DISCONNECTED) }
        appendSystemMessage("Disconnected")
    }

    fun startPtt() {
        val ch = audioChannel ?: return
        _uiState.update { it.copy(isRecording = true) }
        ptt.onChunkReady = { pcm -> ch.trySend(pcm) }
        ptt.startRecording()
    }

    fun stopPtt() {
        ptt.stopRecording()
        ptt.onChunkReady = null
        // Empty chunk signals audio_stream_end to Gemini Live (flushes cached audio)
        audioChannel?.trySend(ByteArray(0))
        _uiState.update { it.copy(isRecording = false, micVolume = 0f) }
        // audioChannel stays open — the live session continues between PTT presses
    }

    fun startLocalTracking(prompt: String) {
        isLocalTrackingActive = true
        localTrackingPrompt = prompt
        _uiState.update { it.copy(agentName = "tracking", agentState = "INITIALIZING") }
        appendSystemMessage("Searching for '$prompt'…")

        initJob?.cancel()
        initJob = viewModelScope.launch(Dispatchers.IO) {
            while (isActive) {
                val frame = latestFrameBytes
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

    // ── Local frame processing (tracking + hand detection + UI) ──────────────

    private fun startLocalProcessing() {
        localProcessingJob?.cancel()
        localProcessingJob = viewModelScope.launch(Dispatchers.IO) {
            cameraManager.frameFlow
                .conflate()
                .catch { e -> appendSystemMessage("[Flow error] ${e.message}") }
                .collect { jpegBytes ->
                    latestFrameBytes = jpegBytes

                    val jpegOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, jpegOpts)
                    val frameWidth = jpegOpts.outWidth
                    val frameHeight = jpegOpts.outHeight

                    // Local ORB tracking — runs every frame without blocking
                    if (isLocalTrackingActive && _uiState.value.agentState == "TRACKING") {
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
                }
        }
    }

    // ── Persistent Gemini Live session ────────────────────────────────────────

    private fun startLiveSession(audioChannel: Channel<ByteArray>) {
        liveSessionJob?.cancel()
        liveSessionJob = viewModelScope.launch(Dispatchers.IO) {
            while (isActive && !audioChannel.isClosedForReceive) {
                doLiveSession(audioChannel)
                if (!isActive || audioChannel.isClosedForReceive) break
                delay(2000L)
            }
        }
    }

    private suspend fun doLiveSession(audioChannel: Channel<ByteArray>) {
        val stub = grpcManager.trackingStub ?: run {
            appendSystemMessage("[Voice] Not connected")
            delay(3000L)
            return
        }

        // requestFlow stays open until audioChannel is closed (on disconnect).
        // Video uses send() — suspends if gRPC is backpressured, conflate() keeps only latest.
        // Audio uses send() — suspends to guarantee delivery.
        val requestFlow = channelFlow<Tracking.VoiceChatChunk> {
            val videoJob = launch {
                edgeDevice.frameFlow.conflate().collect { jpeg ->
                    send(Tracking.VoiceChatChunk.newBuilder()
                        .setVideoFrame(ByteString.copyFrom(jpeg)).build())
                }
            }
            val imuJob = if (imuSensor.isAvailable) launch {
                imuSensor.readings().collect { r ->
                    trySend(
                        Tracking.VoiceChatChunk.newBuilder().setImuFrame(
                            Tracking.IMUFrame.newBuilder()
                                .setTimestampNs(r.timestampNs)
                                .setAccelX(r.accelX).setAccelY(r.accelY).setAccelZ(r.accelZ)
                                .setGyroX(r.gyroX).setGyroY(r.gyroY).setGyroZ(r.gyroZ)
                                .build()
                        ).build()
                    )
                }
            } else null

            audioChannel.receiveAsFlow().collect { pcm ->
                send(Tracking.VoiceChatChunk.newBuilder()
                    .setAudioChunk(ByteString.copyFrom(pcm)).build())
            }
            // Reaches here only when audioChannel is closed (disconnect)
            videoJob.cancel()
            imuJob?.cancel()
        }

        streamingPlayer.start()
        try {
            stub.voiceChatStream(requestFlow).collect { chunk ->
                val pcm = chunk.pcmData.toByteArray()
                if (pcm.isNotEmpty()) {
                    streamingPlayer.writeChunk(pcm)
                    edgeDevice.emitAudio(pcm)
                }
                if (pcm.isEmpty() && chunk.agentState.isNotEmpty()) {
                    Log.d(TAG, "state chunk: state=${chunk.agentState} payload=${chunk.agentPayload}")
                    applySessionResult(SessionResult("", chunk.agentState, chunk.agentPayload))
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "liveSession error: ${e.message}")
            appendSystemMessage("[Session] Reconnecting…")
        } finally {
            streamingPlayer.stop()
            _uiState.update { it.copy(isTtsPlaying = false) }
        }
    }

    // ── State application ─────────────────────────────────────────────────────

    private fun applySessionResult(result: SessionResult) {
        _uiState.update { it.copy(agentState = result.agentState) }
        when (result.agentState) {
            "TRACKING" -> {
                val target = extractTarget(result.agentPayload)
                Log.d(TAG, "Starting local tracking for target='$target'")
                startLocalTracking(target)
            }
            "GUIDING" -> {
                val destination = extractDestination(result.agentPayload)
                val route = extractRoute(result.agentPayload)
                Log.d(TAG, "Guiding to '$destination' via $route")
                _uiState.update { it.copy(guidingDestination = destination, guidingRoute = route, isWalkingMode = false) }
                appendSystemMessage("Navigating to $destination")
            }
            "WALKING" -> {
                Log.d(TAG, "Walking mode started")
                _uiState.update { it.copy(isWalkingMode = true, guidingDestination = "", guidingRoute = emptyList()) }
                appendSystemMessage("Walking mode — obstacle detection active")
            }
            "IDLE" -> {
                if (isLocalTrackingActive) {
                    Log.d(TAG, "Stopping local tracking (IDLE)")
                    stopLocalTracking()
                }
                if (_uiState.value.guidingDestination.isNotEmpty() || _uiState.value.isWalkingMode) {
                    _uiState.update { it.copy(guidingDestination = "", guidingRoute = emptyList(), isWalkingMode = false) }
                }
            }
        }
    }

    private fun extractTarget(payload: String): String =
        try { org.json.JSONObject(payload).optString("target", "") } catch (_: Exception) { "" }

    private fun extractDestination(payload: String): String =
        try { org.json.JSONObject(payload).optString("destination", "") } catch (_: Exception) { "" }

    private fun extractRoute(payload: String): List<String> {
        return try {
            val arr = org.json.JSONObject(payload).optJSONArray("route") ?: return emptyList()
            (0 until arr.length()).map { arr.getString(it) }
        } catch (_: Exception) { emptyList() }
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
        audioChannel?.close()
        if (_uiState.value.isRecording) ptt.stopRecording()
        if (isLocalTrackingActive) trackingBackend.stop()
        handTracker.close()
        streamingPlayer.stop()
        cameraManager.shutdown()
        grpcManager.disconnect()
    }
}
