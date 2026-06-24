package com.tracking.client.model

enum class ConnectionState { DISCONNECTED, CONNECTING, CONNECTED, ERROR }

data class ChatMessage(
    val role: String,
    val content: String,
    val isVoice: Boolean = false
)

data class AppUiState(
    val connectionState: ConnectionState = ConnectionState.DISCONNECTED,
    val guidanceData: GuidanceData = GuidanceData(),
    val chatHistory: List<ChatMessage> = emptyList(),
    val agentState: String = "",
    val agentName: String = "",
    val isVadActive: Boolean = false,
    val isTtsPlaying: Boolean = false,
    val micStatus: String = "Idle",
    val micVolume: Float = 0f,
    val error: String? = null
)
