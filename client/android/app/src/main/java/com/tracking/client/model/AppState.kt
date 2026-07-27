package com.tracking.client.model

enum class ConnectionState { DISCONNECTED, CONNECTING, CONNECTED, ERROR }

data class ChatMessage(
    val role: String,
    val content: String,
    val isVoice: Boolean = false
)

data class AppUiState(
    val connectionState: ConnectionState = ConnectionState.DISCONNECTED,
    val guidanceData: ObjectTrack = ObjectTrack(),
    val chatHistory: List<ChatMessage> = emptyList(),
    val agentState: String = "",
    val agentName: String = "",
    val isVadActive: Boolean = false,
    val isTtsPlaying: Boolean = false,
    val isRecording: Boolean = false,
    // True from the VAD registering a user utterance (onSpeechEnd) until
    // Gemini's response has finished playing — distinct from isRecording,
    // which only covers active speech capture. Drives music/YouTube
    // ducking (NOT capture-time ducking — see CLAUDE.md's "Continuous
    // VAD-gated listening" note).
    val isAwaitingResponse: Boolean = false,
    val micVolume: Float = 0f,
    val error: String? = null,
    val guidingDestination: String = "",
    val guidingRoute: List<String> = emptyList(),
    val isWalkingMode: Boolean = false,
)
