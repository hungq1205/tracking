package com.tracking.client.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.tracking.client.model.ConnectionState
import com.tracking.client.model.GuidanceData

private const val FRAME_W = 640f
private const val FRAME_H = 480f

@Composable
fun MainScreen(
    viewModel: MainViewModel,
    onOpenSettings: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsState()
    val lifecycleOwner = LocalLifecycleOwner.current
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(uiState.error) {
        uiState.error?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).also { previewView ->
                    viewModel.cameraManager.bind(lifecycleOwner, previewView)
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // Bounding box overlay
        Canvas(modifier = Modifier.fillMaxSize()) {
            val sx = size.width / FRAME_W
            val sy = size.height / FRAME_H

            val obj = uiState.guidanceData.objectBoxXyxy
            if (obj.size == 4) {
                drawRect(
                    color = Color.Green,
                    topLeft = Offset(obj[0] * sx, obj[1] * sy),
                    size = Size((obj[2] - obj[0]) * sx, (obj[3] - obj[1]) * sy),
                    style = Stroke(width = 4f)
                )
            }

            val hand = uiState.guidanceData.handBoxXyxy
            if (hand.size == 4) {
                drawRect(
                    color = Color.Blue,
                    topLeft = Offset(hand[0] * sx, hand[1] * sy),
                    size = Size((hand[2] - hand[0]) * sx, (hand[3] - hand[1]) * sy),
                    style = Stroke(width = 3f)
                )
            }
        }

        // HUD — top-left
        Column(
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(8.dp)
        ) {
            ConnectionChip(uiState.connectionState)
            GuidanceHud(uiState.guidanceData, uiState.agentState, uiState.agentName)
        }

        // Chat panel — right side
        ChatPanel(
            chatHistory = uiState.chatHistory,
            micStatus = uiState.micStatus,
            micVolume = uiState.micVolume,
            agentState = uiState.agentState,
            isTtsPlaying = uiState.isTtsPlaying,
            modifier = Modifier
                .align(Alignment.CenterEnd)
                .width(300.dp)
                .fillMaxHeight()
        )

        // Settings button — top-right
        IconButton(
            onClick = onOpenSettings,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(end = 308.dp, top = 4.dp)
        ) {
            Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color.White)
        }

        SnackbarHost(
            hostState = snackbarHostState,
            modifier = Modifier.align(Alignment.BottomStart)
        )
    }
}

@Composable
private fun ConnectionChip(state: ConnectionState) {
    val (label, color) = when (state) {
        ConnectionState.CONNECTED -> "Connected" to Color(0xFF69F0AE)
        ConnectionState.CONNECTING -> "Connecting..." to Color(0xFFFFCC00)
        ConnectionState.ERROR -> "Error" to Color(0xFFFF5252)
        ConnectionState.DISCONNECTED -> "Disconnected" to Color.Gray
    }
    AssistChip(
        onClick = {},
        label = { Text(label, fontSize = 11.sp) },
        colors = AssistChipDefaults.assistChipColors(containerColor = color.copy(alpha = 0.2f)),
        border = AssistChipDefaults.assistChipBorder(enabled = true, borderColor = color)
    )
}

@Composable
private fun GuidanceHud(data: GuidanceData, agentState: String, agentName: String) {
    if (agentState.isNotBlank()) {
        Text("[$agentName] $agentState", color = Color(0xFFFFCC00), fontSize = 11.sp,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(2.dp))
    }
    if (data.trackingStatus.isNotBlank()) {
        Text(data.trackingStatus, color = Color.White, fontSize = 11.sp,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(2.dp))
    }
    if (data.instruction.isNotBlank()) {
        Text(data.instruction, color = Color(0xFF00E5FF), fontSize = 12.sp,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(2.dp))
    }
    if (data.objectConfidence > 0f) {
        Text("Conf: ${"%.2f".format(data.objectConfidence)}", color = Color.White, fontSize = 10.sp,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(2.dp))
    }
}

