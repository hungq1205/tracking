package com.tracking.client.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.tracking.client.model.ConnectionState
import java.io.File

@Composable
fun ScanScreen(
    viewModel: ScanViewModel,
    onBack: () -> Unit
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

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { padding ->
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // Left 2/3 — camera preview
            Box(
                modifier = Modifier
                    .weight(2f)
                    .fillMaxHeight()
            ) {
                AndroidView(
                    factory = { ctx ->
                        PreviewView(ctx).also { pv ->
                            viewModel.cameraManager.bind(lifecycleOwner, pv)
                        }
                    },
                    modifier = Modifier.fillMaxSize()
                )
                ScanConnectionChip(
                    state = uiState.connectionState,
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(8.dp)
                )

                // Recording indicator overlay
                if (uiState.isRecording) {
                    val elapsed = uiState.recordingElapsedMs
                    val mm = elapsed / 60_000
                    val ss = (elapsed % 60_000) / 1000
                    Text(
                        text = "● REC  %02d:%02d".format(mm, ss),
                        color = Color(0xFFFF1744),
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier
                            .align(Alignment.TopEnd)
                            .padding(8.dp)
                            .background(Color.Black.copy(alpha = 0.6f))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }

            // Right 1/3 — controls
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .background(Color(0xDD121212))
                    .verticalScroll(rememberScrollState())
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                // Title row
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Color.White)
                    }
                    Text("3D Scan", color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                }

                // ── Offline Recording section ─────────────────────────────────
                ScanSectionLabel("Record to File")
                Text(
                    "Walk through the venue. Video + IMU saved locally, then upload to scan server.",
                    color = Color.Gray, fontSize = 10.sp
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { viewModel.startRecording() },
                        enabled = !uiState.isRecording && !uiState.isUploading,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFB71C1C)),
                        modifier = Modifier.weight(1f)
                    ) { Text("Record", fontSize = 12.sp) }
                    Button(
                        onClick = { viewModel.stopRecording() },
                        enabled = uiState.isRecording,
                        modifier = Modifier.weight(1f)
                    ) { Text("Stop", fontSize = 12.sp) }
                }

                if (uiState.videoFile != null) {
                    RecordingFileInfo(uiState.videoFile, uiState.imuFile)
                }

                if (uiState.videoFile != null && !uiState.isRecording) {
                    ScanSectionLabel("Upload to Scan Server")
                    OutlinedTextField(
                        value = uiState.scanServerHost,
                        onValueChange = { viewModel.updateScanServerHost(it) },
                        label = { Text("Scan Server Host") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !uiState.isUploading,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri)
                    )
                    OutlinedTextField(
                        value = uiState.scanServerPort.toString(),
                        onValueChange = { viewModel.updateScanServerPort(it) },
                        label = { Text("Port (Gradio)") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !uiState.isUploading,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
                    )
                    if (uiState.isUploading) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            CircularProgressIndicator(modifier = Modifier.padding(4.dp), strokeWidth = 2.dp)
                            Text("Uploading…", color = Color.White, fontSize = 11.sp)
                        }
                    } else {
                        Button(
                            onClick = { viewModel.uploadToScanServer() },
                            modifier = Modifier.fillMaxWidth()
                        ) { Text("Upload Files", fontSize = 12.sp) }
                    }
                    uiState.uploadStatus?.let { status ->
                        Text(
                            status,
                            color = if (status.startsWith("Uploaded")) Color(0xFF69F0AE) else Color(0xFFFF5252),
                            fontSize = 10.sp, fontFamily = FontFamily.Monospace
                        )
                    }
                }

                Divider(color = Color(0xFF333333))

                // ── Live gRPC stream section ──────────────────────────────────
                ScanSectionLabel("Live Stream (gRPC)")
                OutlinedTextField(
                    value = uiState.host,
                    onValueChange = { viewModel.updateHost(it) },
                    label = { Text("Host") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !uiState.isStreaming,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri)
                )
                OutlinedTextField(
                    value = uiState.port.toString(),
                    onValueChange = { viewModel.updatePort(it) },
                    label = { Text("Port") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !uiState.isStreaming,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
                )

                ScanSectionLabel("Scan")
                OutlinedTextField(
                    value = uiState.locationId,
                    onValueChange = { viewModel.updateLocationId(it) },
                    label = { Text("Location ID") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !uiState.isStreaming
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { viewModel.startStreaming() },
                        enabled = !uiState.isStreaming,
                        modifier = Modifier.weight(1f)
                    ) { Text("Start", fontSize = 12.sp) }
                    Button(
                        onClick = { viewModel.stopStreaming() },
                        enabled = uiState.isStreaming,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFB71C1C))
                    ) { Text("Stop", fontSize = 12.sp) }
                }

                // Zone labeling section
                ScanSectionLabel("Zone Label")
                Text(
                    "Stand in the zone, enter its name, then tap Set.",
                    color = Color.Gray, fontSize = 10.sp
                )
                OutlinedTextField(
                    value = uiState.zoneLabel,
                    onValueChange = { viewModel.updateZoneLabel(it) },
                    label = { Text("Zone name") },
                    placeholder = { Text("e.g. Kitchen") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text("Radius:", color = Color.White, fontSize = 11.sp)
                    Slider(
                        value = uiState.zoneRadius,
                        onValueChange = { viewModel.updateZoneRadius(it) },
                        valueRange = 0.5f..5f,
                        modifier = Modifier.weight(1f)
                    )
                    Text("${"%.1f".format(uiState.zoneRadius)}m", color = Color.White, fontSize = 11.sp)
                }
                Button(
                    onClick = { viewModel.setZoneLabel() },
                    enabled = uiState.isStreaming && uiState.zoneLabel.isNotBlank(),
                    modifier = Modifier.fillMaxWidth()
                ) { Text("Set Zone Label", fontSize = 12.sp) }

                // Export section
                ScanSectionLabel("Export")
                Button(
                    onClick = { viewModel.exportMap() },
                    enabled = uiState.isStreaming,
                    modifier = Modifier.fillMaxWidth()
                ) { Text("Export Map to Server", fontSize = 12.sp) }
                if (uiState.lastExportPath.isNotBlank()) {
                    Text(
                        "${uiState.lastExportPath}\n${uiState.lastExportPointCount} pts | ${uiState.lastExportZoneCount} zones",
                        color = Color(0xFF69F0AE), fontSize = 9.sp, fontFamily = FontFamily.Monospace
                    )
                }

                // Stats section
                ScanSectionLabel("Stats")
                Text("Points: ${uiState.pointCount}", color = Color.White, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                val posStr = if (uiState.cameraPosition.size >= 3) {
                    "x=%.2f  y=%.2f  z=%.2f".format(
                        uiState.cameraPosition[0], uiState.cameraPosition[1], uiState.cameraPosition[2]
                    )
                } else "—"
                Text("Pos: $posStr", color = Color.White, fontSize = 11.sp, fontFamily = FontFamily.Monospace)

                Divider(color = Color.DarkGray)
                Text(uiState.statusMessage, color = Color(0xFF888888), fontSize = 10.sp)
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

@Composable
private fun RecordingFileInfo(videoFile: File?, imuFile: File?) {
    if (videoFile == null) return
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(
            "Video: ${videoFile.name}  (${videoFile.length() / 1024} KB)",
            color = Color(0xFF69F0AE), fontSize = 10.sp, fontFamily = FontFamily.Monospace
        )
        if (imuFile != null && imuFile.exists()) {
            val lines = imuFile.bufferedReader().use { it.readLines().size }
            Text(
                "IMU:   ${imuFile.name}  ($lines samples)",
                color = Color(0xFF69F0AE), fontSize = 10.sp, fontFamily = FontFamily.Monospace
            )
        } else {
            Text("IMU: not available", color = Color.Gray, fontSize = 10.sp)
        }
    }
}

@Composable
private fun ScanSectionLabel(text: String) {
    Text(text, color = Color(0xFF00E5FF), fontSize = 10.sp, fontWeight = FontWeight.SemiBold)
}

@Composable
private fun ScanConnectionChip(state: ConnectionState, modifier: Modifier = Modifier) {
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
        border = AssistChipDefaults.assistChipBorder(enabled = true, borderColor = color),
        modifier = modifier
    )
}
