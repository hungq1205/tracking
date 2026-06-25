package com.tracking.client.ui

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.tracking.client.model.ChatMessage

@Composable
fun ChatPanel(
    chatHistory: List<ChatMessage>,
    micVolume: Float,
    isRecording: Boolean,
    agentState: String,
    isTtsPlaying: Boolean,
    onStartRecording: () -> Unit,
    onStopRecording: () -> Unit,
    modifier: Modifier = Modifier
) {
    val listState = rememberLazyListState()

    LaunchedEffect(chatHistory.size) {
        if (chatHistory.isNotEmpty()) listState.scrollToItem(chatHistory.size - 1)
    }

    Column(
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.75f))
            .padding(8.dp)
    ) {
        // Agent state row
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 4.dp)
        ) {
            Text(
                text = "Agent: $agentState",
                color = if (agentState == "READING_ALOUD") Color(0xFF69F0AE) else Color(0xFFFFCC00),
                fontSize = 11.sp
            )
            if (isTtsPlaying) Text("  ♪", color = Color(0xFF69F0AE), fontSize = 11.sp)
        }

        // Chat history
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            items(chatHistory) { msg -> ChatBubble(msg) }
        }

        // Push-to-talk button
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 10.dp),
            contentAlignment = Alignment.Center
        ) {
            val scale by animateFloatAsState(
                targetValue = if (isRecording) 1.4f + (micVolume * 8f).coerceIn(0f, 0.6f) else 1f,
                animationSpec = spring(stiffness = Spring.StiffnessLow),
                label = "pttScale"
            )
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .scale(scale)
                    .background(
                        color = if (isRecording) Color(0xFFB71C1C) else Color(0xFF1A1A2E),
                        shape = CircleShape
                    )
                    .pointerInput(Unit) {
                        detectTapGestures(
                            onPress = {
                                onStartRecording()
                                try { awaitRelease() } finally { onStopRecording() }
                            }
                        )
                    },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Mic,
                    contentDescription = if (isRecording) "Recording…" else "Hold to talk",
                    tint = if (isRecording) Color.White else Color(0xFF00E5FF),
                    modifier = Modifier.size(32.dp)
                )
            }
            if (isRecording) {
                Text(
                    "Release to send",
                    color = Color(0xFFFF8A80),
                    fontSize = 9.sp,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(top = 72.dp)
                )
            }
        }
    }
}

@Composable
private fun ChatBubble(msg: ChatMessage) {
    val isSystem = msg.role == "system"
    val isUser = msg.role == "user"
    Box(
        modifier = Modifier.fillMaxWidth(),
        contentAlignment = when {
            isSystem -> Alignment.CenterStart
            isUser -> Alignment.CenterEnd
            else -> Alignment.CenterStart
        }
    ) {
        Text(
            text = msg.content,
            color = when {
                isSystem -> Color(0xFFFF8C00)
                isUser -> Color.White
                else -> Color(0xFFCCEEFF)
            },
            fontSize = if (isSystem) 10.sp else 12.sp,
            modifier = Modifier
                .widthIn(max = 260.dp)
                .background(
                    color = when {
                        isSystem -> Color(0xFF2A1A00)
                        isUser -> Color(0xFF1A3A5C)
                        else -> Color(0xFF1A1A2E)
                    },
                    shape = RoundedCornerShape(6.dp)
                )
                .padding(horizontal = 8.dp, vertical = 4.dp)
        )
    }
}
