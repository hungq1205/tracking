package com.tracking.client.ui

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.tracking.client.model.ChatMessage

private const val VAD_THRESHOLD = 0.01f

@Composable
fun ChatPanel(
    chatHistory: List<ChatMessage>,
    micStatus: String,
    micVolume: Float,
    agentState: String,
    isTtsPlaying: Boolean,
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
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 4.dp)) {
            Text(
                text = "Agent: $agentState",
                color = if (agentState == "READING_ALOUD") Color(0xFF69F0AE) else Color(0xFFFFCC00),
                fontSize = 11.sp
            )
            if (isTtsPlaying) {
                Text("  ♪", color = Color(0xFF69F0AE), fontSize = 11.sp)
            }
        }

        // Chat history
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            items(chatHistory) { msg ->
                ChatBubble(msg)
            }
        }

        // Animated mic icon
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            contentAlignment = Alignment.Center
        ) {
            val scale by animateFloatAsState(
                targetValue = 1f + (micVolume * 15f).coerceIn(0f, 2f),
                animationSpec = spring(stiffness = Spring.StiffnessLow),
                label = "micScale"
            )
            val tint = when {
                micVolume > VAD_THRESHOLD -> Color(0xFF69F0AE)
                micStatus == "Listening" -> Color(0xFF00E5FF)
                else -> Color.Gray
            }
            Icon(
                imageVector = Icons.Default.Mic,
                contentDescription = "Microphone",
                tint = tint,
                modifier = Modifier
                    .size(48.dp)
                    .scale(scale)
            )
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
