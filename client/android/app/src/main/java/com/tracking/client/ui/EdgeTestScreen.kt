package com.tracking.client.ui

import android.graphics.BitmapFactory
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import com.tracking.client.edge.EdgeZmqTestClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.sin

/**
 * Manual test screen for the Android <-> edge-device ZeroMQ link, against
 * test_module/edge_mock/mock_edge_server.py — not part of the real app flow,
 * just a way to confirm the PUSH/PULL protocol works before wiring a real
 * Pi-Zero-2. See EdgeZmqTestClient.kt for the socket setup.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EdgeTestScreen(onBack: () -> Unit) {
    val scope = rememberCoroutineScope()
    val client = remember { EdgeZmqTestClient() }
    var host by remember { mutableStateOf("192.168.1.100") }
    var connected by remember { mutableStateOf(false) }
    var framesReceived by remember { mutableStateOf(0) }
    var audioChunksReceived by remember { mutableStateOf(0) }
    var audioChunksSent by remember { mutableStateOf(0) }
    var lastFrameBytes by remember { mutableStateOf<ByteArray?>(null) }

    val audioTrack = remember {
        val minBuf = AudioTrack.getMinBufferSize(16000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(16000)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(minBuf.coerceAtLeast(4096) * 4)
            .build()
    }

    DisposableEffect(Unit) {
        onDispose {
            client.disconnect()
            audioTrack.stop()
            audioTrack.release()
        }
    }

    // Play back whatever mock "mic" audio arrives from the edge -- this is
    // the "receive audio, speak it back" leg.
    LaunchedEffect(Unit) {
        audioTrack.play()
        client.micAudioFlow.collect { pcm ->
            audioTrack.write(pcm, 0, pcm.size)
            audioChunksReceived = client.stats.audioChunksReceived
        }
    }

    // Display whatever mock camera frame arrives from the edge.
    LaunchedEffect(Unit) {
        client.frameFlow.collect { jpg ->
            lastFrameBytes = jpg
            framesReceived = client.stats.framesReceived
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Edge ZMQ Test") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
        ) {
            Text(
                "Point this at the machine running " +
                    "test_module/edge_mock/mock_edge_server.py",
                style = MaterialTheme.typography.bodySmall,
            )
            Spacer(Modifier.height(12.dp))
            OutlinedTextField(
                value = host,
                onValueChange = { host = it },
                label = { Text("Mock edge server host/IP") },
                modifier = Modifier.fillMaxWidth(),
                enabled = !connected,
            )
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        if (connected) {
                            client.disconnect()
                            connected = false
                        } else {
                            client.connect(host)
                            connected = true
                        }
                    },
                    colors = if (connected) ButtonDefaults.buttonColors(containerColor = androidx.compose.ui.graphics.Color(0xFFB71C1C))
                             else ButtonDefaults.buttonColors(),
                ) {
                    Text(if (connected) "Disconnect" else "Connect")
                }
                Button(
                    enabled = connected,
                    onClick = {
                        scope.launch {
                            // Simulate the phone sending rendered HRTF/TTS audio
                            // out to the edge speaker: a short 440Hz test tone.
                            withContext(Dispatchers.IO) {
                                repeat(20) { i ->
                                    val chunk = ShortArray(512) { s ->
                                        (sin(2.0 * Math.PI * 440.0 * (i * 512 + s) / 16000.0) * 0.3 * Short.MAX_VALUE).toInt().toShort()
                                    }
                                    val bytes = ByteArray(chunk.size * 2)
                                    for (j in chunk.indices) {
                                        bytes[j * 2] = (chunk[j].toInt() and 0xFF).toByte()
                                        bytes[j * 2 + 1] = ((chunk[j].toInt() shr 8) and 0xFF).toByte()
                                    }
                                    client.sendAudioOut(bytes)
                                }
                            }
                            audioChunksSent = client.stats.audioChunksSent
                        }
                    },
                ) {
                    Text("Send test tone to edge")
                }
            }
            Spacer(Modifier.height(16.dp))
            Text("Frames received: $framesReceived", style = MaterialTheme.typography.bodyMedium)
            Text("Mic audio chunks received: $audioChunksReceived", style = MaterialTheme.typography.bodyMedium)
            Text("Audio chunks sent to edge: $audioChunksSent", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(16.dp))
            lastFrameBytes?.let { jpg ->
                val bmp = remember(jpg) { BitmapFactory.decodeByteArray(jpg, 0, jpg.size) }
                bmp?.let {
                    Image(
                        bitmap = it.asImageBitmap(),
                        contentDescription = "Last mock edge frame",
                        modifier = Modifier.fillMaxWidth().aspectRatio(it.width.toFloat() / it.height.toFloat()),
                    )
                }
            }
        }
    }
}
