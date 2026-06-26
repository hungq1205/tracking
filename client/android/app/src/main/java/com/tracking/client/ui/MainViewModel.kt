package com.tracking.client.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.protobuf.ByteString
import com.tracking.client.audio.PushToTalkRecorder
import com.tracking.client.audio.StreamingAudioPlayer
import com.tracking.client.audio.TtsPlayer
import com.tracking.client.camera.CameraManager
import com.tracking.client.grpc.GrpcClientManager
import com.tracking.client.model.AppUiState
import com.tracking.client.model.ChatMessage
import com.tracking.client.model.ConnectionState
import com.tracking.client.model.GuidanceData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import com.tracking.client.sensors.ImuSensor
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.conflate
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import tracking.Tracking

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences("tracking_prefs", Context.MODE_PRIVATE)

    val grpcManager = GrpcClientManager()
    val cameraManager = CameraManager(app)
    private val imuSensor = ImuSensor(app)
    private val ptt = PushToTalkRecorder()
    private val ttsPlayer = TtsPlayer()
    private val streamingPlayer = StreamingAudioPlayer()

    private val _uiState = MutableStateFlow(AppUiState())
    val uiState: StateFlow<AppUiState> = _uiState

    private var frameStreamJob: Job? = null
    private var readingContinueJob: Job? = null
    private var lastFrameErrorMs = 0L
    private var activePttChannel: Channel<ByteArray>? = null
    private var pttStreamJob: Job? = null

    init {
        viewModelScope.launch {
            grpcManager.connectionState.collect { state ->
                _uiState.update { it.copy(connectionState = state) }
            }
        }
        viewModelScope.launch {
            ttsPlayer.isPlaying.collect { playing ->
                _uiState.update { it.copy(isTtsPlaying = playing) }
            }
        }

        ptt.onVolumeChange = { rms ->
            _uiState.update { it.copy(micVolume = rms) }
        }
        viewModelScope.launch {
            streamingPlayer.isPlaying.collect { playing ->
                if (playing) _uiState.update { it.copy(isTtsPlaying = true) }
            }
        }
    }

    fun connect(host: String, port: Int, fps: Int, vadThreshold: Float = 0.03f, startThreshold: Float = 0.05f) {
        grpcManager.connect(host, port)
        cameraManager.targetFps = fps
        startFrameStreaming()
        _uiState.update { it.copy(isVadActive = true) }
        appendSystemMessage("Connecting to $host:$port …")
    }

    fun disconnect() {
        frameStreamJob?.cancel()
        if (_uiState.value.isRecording) ptt.stopRecording()
        grpcManager.disconnect()
        _uiState.update { it.copy(isVadActive = false, isRecording = false, connectionState = ConnectionState.DISCONNECTED) }
        appendSystemMessage("Disconnected")
    }

    fun startPtt() {
        _uiState.update { it.copy(isRecording = true, isWaitingResponse = true) }
        frameStreamJob?.cancel()  // pause regular frame stream; VoiceChatStream sends frames instead
        val channel = Channel<ByteArray>(Channel.UNLIMITED)
        activePttChannel = channel
        ptt.onChunkReady = { pcm -> channel.trySend(pcm) }
        pttStreamJob = viewModelScope.launch(Dispatchers.IO) { doVoiceChatStream(channel) }
        ptt.startRecording()
    }

    fun stopPtt() {
        ptt.stopRecording()
        ptt.onChunkReady = null
        activePttChannel?.close()
        activePttChannel = null
        _uiState.update { it.copy(isRecording = false, micVolume = 0f) }
        viewModelScope.launch(Dispatchers.IO) {
            pttStreamJob?.join()
            pttStreamJob = null
            startFrameStreaming()  // resume regular frame stream after response completes
        }
    }

    private suspend fun doVoiceChatStream(pttChannel: Channel<ByteArray>) {
        val stub = grpcManager.trackingStub ?: return
        // Single channelFlow so child video/IMU coroutines are cancelled the moment
        // the audio channel closes (PTT button released), which closes the gRPC stream
        // and lets the server process the buffered audio.
        val requestFlow: Flow<Tracking.VoiceChatChunk> = channelFlow {
            val videoJob = launch {
                cameraManager.frameFlow.conflate().collect { jpeg ->
                    trySend(Tracking.VoiceChatChunk.newBuilder()
                        .setVideoFrame(ByteString.copyFrom(jpeg)).build())
                }
            }
            val isNavigating = _uiState.value.agentName.equals("NavigationAgent", ignoreCase = true)
            val imuJob = if (imuSensor.isAvailable && isNavigating) launch {
                imuSensor.readings().collect { r ->
                    val imuProto = Tracking.IMUFrame.newBuilder()
                        .setTimestampNs(r.timestampNs)
                        .setAccelX(r.accelX).setAccelY(r.accelY).setAccelZ(r.accelZ)
                        .setGyroX(r.gyroX).setGyroY(r.gyroY).setGyroZ(r.gyroZ)
                        .build()
                    trySend(Tracking.VoiceChatChunk.newBuilder().setImuFrame(imuProto).build())
                }
            } else null
            // Drain audio until PTT is released (channel closed by stopPtt)
            for (pcm in pttChannel) {
                send(Tracking.VoiceChatChunk.newBuilder()
                    .setAudioChunk(ByteString.copyFrom(pcm)).build())
            }
            // Audio done — cancel video/IMU so the stream closes
            videoJob.cancel()
            imuJob?.cancel()
        }
        streamingPlayer.start()
        try {
            stub.voiceChatStream(requestFlow).collect { chunk ->
                val pcm = chunk.pcmData.toByteArray()
                if (pcm.isNotEmpty()) streamingPlayer.writeChunk(pcm)
            }
        } catch (e: Exception) {
            appendSystemMessage("[VoiceStream error] ${e.message}")
        } finally {
            streamingPlayer.stop()
            _uiState.update { it.copy(isWaitingResponse = false, isTtsPlaying = false) }
        }
    }

    private fun startFrameStreaming() {
        frameStreamJob?.cancel()
        frameStreamJob = viewModelScope.launch(Dispatchers.IO) {
            cameraManager.frameFlow
                .conflate()
                .catch { e -> appendSystemMessage("[Flow error] ${e.message}") }
                .collect { jpegBytes ->
                    val stub = grpcManager.trackingStub
                    if (stub == null) {
                        appendSystemMessage("[Frame] stub is null — not connected")
                        return@collect
                    }
                    try {
                        val req = Tracking.FrameRequest.newBuilder()
                            .setImageData(ByteString.copyFrom(jpegBytes))
                            .build()
                        val resp = stub.streamFrame(req)
                        if (resp.audioResponse.size() > 44) {
                            viewModelScope.launch(Dispatchers.IO) { ttsPlayer.play(resp.audioResponse.toByteArray()) }
                        }
                        lastFrameErrorMs = 0L
                    } catch (e: Exception) {
                        val now = System.currentTimeMillis()
                        if (now - lastFrameErrorMs > 3000) {
                            lastFrameErrorMs = now
                            appendSystemMessage("[Frame error] ${e.message}")
                        }
                    }
                }
        }
    }

    private suspend fun sendVoiceChat(wavBytes: ByteArray) {
        val stub = grpcManager.trackingStub
        if (stub == null) {
            appendSystemMessage("[Voice] stub is null — not connected")
            return
        }
        try {
            val req = Tracking.VoiceChatRequest.newBuilder()
                .setAudioData(ByteString.copyFrom(wavBytes))
                .build()
            val resp = stub.voiceChat(req)
            withContext(Dispatchers.Main) {
                handleChatResponse(resp.response, resp.agentName, resp.agentState, resp.audioResponse.toByteArray())
            }
        } catch (e: Exception) {
            appendSystemMessage("[Voice error] ${e.message}")
        }
    }

    private fun handleChatResponse(text: String, agentName: String, agentState: String, audioBytes: ByteArray) {
        _uiState.update { state ->
            val newHistory = if (text.isNotBlank()) {
                state.chatHistory + ChatMessage("assistant", text)
            } else state.chatHistory
            state.copy(chatHistory = newHistory, agentState = agentState, agentName = agentName)
        }
        if (audioBytes.size > 44) {
            viewModelScope.launch(Dispatchers.IO) { ttsPlayer.play(audioBytes) }
            scheduleReadingContinue(audioBytes, agentState)
        }
    }

    private fun scheduleReadingContinue(audioBytes: ByteArray, agentState: String) {
        readingContinueJob?.cancel()
        if (agentState != "READING_ALOUD") return
        val durationMs = ttsPlayer.computeDurationMs(audioBytes) + 300L
        readingContinueJob = viewModelScope.launch(Dispatchers.IO) {
            delay(durationMs)
            if (isActive && _uiState.value.agentState == "READING_ALOUD") {
                sendContinueReadingInternal()
            }
        }
    }

    private suspend fun sendContinueReadingInternal() {
        val stub = grpcManager.trackingStub ?: return
        try {
            val req = Tracking.ChatRequest.newBuilder().setMessage("continue reading").build()
            val resp = stub.chat(req)
            withContext(Dispatchers.Main) {
                handleChatResponse(resp.response, resp.agentName, resp.agentState, resp.audioResponse.toByteArray())
            }
        } catch (e: Exception) {
            appendSystemMessage("[Continue error] ${e.message}")
        }
    }

    private fun appendSystemMessage(text: String) {
        _uiState.update { state ->
            state.copy(chatHistory = state.chatHistory + ChatMessage("system", text))
        }
    }

    fun clearError() { _uiState.update { it.copy(error = null) } }

    override fun onCleared() {
        super.onCleared()
        if (_uiState.value.isRecording) ptt.stopRecording()
        ttsPlayer.stop()
        streamingPlayer.stop()
        cameraManager.shutdown()
        grpcManager.disconnect()
    }
}
