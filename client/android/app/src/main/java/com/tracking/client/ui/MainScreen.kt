package com.tracking.client.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Map
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
import com.tracking.client.model.ObjectTrack

@Composable
fun MainScreen(
    viewModel: MainViewModel,
    onOpenSettings: () -> Unit,
    onOpenScan: () -> Unit = {}
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

            // Bounding box overlay — match PreviewView FIT_CENTER transform
            Canvas(modifier = Modifier.fillMaxSize()) {
                val frameW = uiState.guidanceData.frameWidth.toFloat()
                val frameH = uiState.guidanceData.frameHeight.toFloat()
                if (frameW <= 0f || frameH <= 0f) return@Canvas
                // FIT_CENTER: uniform scale so full frame fits, letterbox offsets
                val scale = minOf(size.width / frameW, size.height / frameH)
                val ox = (size.width - frameW * scale) / 2f
                val oy = (size.height - frameH * scale) / 2f

                val obj = uiState.guidanceData.objectBoxXyxy
                if (obj.size == 4) {
                    drawRect(
                        color = Color.Green,
                        topLeft = Offset(obj[0] * scale + ox, obj[1] * scale + oy),
                        size = Size((obj[2] - obj[0]) * scale, (obj[3] - obj[1]) * scale),
                        style = Stroke(width = 4f)
                    )
                }

                val hand = uiState.guidanceData.handBoxXyxy
                if (hand.size == 4) {
                    drawRect(
                        color = Color.Blue,
                        topLeft = Offset(hand[0] * scale + ox, hand[1] * scale + oy),
                        size = Size((hand[2] - hand[0]) * scale, (hand[3] - hand[1]) * scale),
                        style = Stroke(width = 3f)
                    )
                }

                val kx = uiState.guidanceData.matchedKeypointsX
                val ky = uiState.guidanceData.matchedKeypointsY
                if (kx.size == ky.size && kx.isNotEmpty()) {
                    for (i in kx.indices) {
                        drawCircle(
                            color = Color.Yellow,
                            radius = 5f,
                            center = Offset(kx[i] * scale + ox, ky[i] * scale + oy)
                        )
                    }
                }

                uiState.guidanceData.handLandmarksX.zip(uiState.guidanceData.handLandmarksY)
                    .forEach { (lx, ly) ->
                        lx.zip(ly).forEach { (x, y) ->
                            drawCircle(
                                color = Color.Cyan,
                                radius = 4f,
                                center = Offset(x * scale + ox, y * scale + oy)
                            )
                        }
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

            // Guiding / walking banner — top-center
            if (uiState.guidingDestination.isNotEmpty() || uiState.isWalkingMode) {
                val bannerText = if (uiState.guidingDestination.isNotEmpty())
                    "Navigating → ${uiState.guidingDestination}"
                else
                    "Walking — obstacle detection active"
                Text(
                    text = bannerText,
                    color = Color(0xFFFFCC00),
                    fontSize = 13.sp,
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(top = 8.dp)
                        .background(Color.Black.copy(alpha = 0.6f))
                        .padding(horizontal = 12.dp, vertical = 4.dp)
                        .fillMaxWidth(0.5f)
                )
            }

            // Chat panel — right side
            ChatPanel(
                chatHistory = uiState.chatHistory,
                micVolume = uiState.micVolume,
                isRecording = uiState.isRecording,
                agentState = uiState.agentState,
                isTtsPlaying = uiState.isTtsPlaying,
                onStartRecording = { viewModel.startPtt() },
                onStopRecording = { viewModel.stopPtt() },
                modifier = Modifier
                    .align(Alignment.CenterEnd)
                    .width(300.dp)
                    .fillMaxHeight()
            )

            // Settings + Scan buttons — top-right
            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(end = 308.dp, top = 4.dp)
            ) {
                IconButton(onClick = onOpenScan) {
                    Icon(Icons.Default.Map, contentDescription = "Scan", tint = Color.White)
                }
                IconButton(onClick = onOpenSettings) {
                    Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color.White)
                }
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
private fun GuidanceHud(data: ObjectTrack, agentState: String, agentName: String) {
    if (agentState.isNotBlank()) {
        Text("[$agentName] $agentState", color = Color(0xFFFFCC00), fontSize = 11.sp,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(2.dp))
    }
    if (data.status.isNotBlank()) {
        Text(data.status, color = Color.White, fontSize = 11.sp,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(2.dp))
    }
    if (data.instruction.isNotBlank()) {
        Text(data.instruction, color = Color(0xFF00E5FF), fontSize = 12.sp,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(2.dp))
    }
    if (data.confidence > 0f) {
        Text("Conf: ${"%.2f".format(data.confidence)}", color = Color.White, fontSize = 10.sp,
            modifier = Modifier.background(Color.Black.copy(alpha = 0.5f)).padding(2.dp))
    }
}

