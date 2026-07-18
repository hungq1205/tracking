package com.tracking.client.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusDirection
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.tracking.client.model.ConnectionState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    mainViewModel: MainViewModel,
    onConnect: (String, Int, Int, Int, Int, Float, Float, String, String, String) -> Unit,
    onBack: () -> Unit
) {
    val settingsVm: SettingsViewModel = viewModel()
    val savedHost by settingsVm.serverHost.collectAsState()
    val savedPort by settingsVm.serverPort.collectAsState()
    val savedFrameIntervalMs by settingsVm.frameIntervalMs.collectAsState()
    val savedScanIntervalMs by settingsVm.scanIntervalMs.collectAsState()
    val savedRecentBufferMs by settingsVm.recentBufferMs.collectAsState()
    val savedVad by settingsVm.vadThreshold.collectAsState()
    val savedStart by settingsVm.startThreshold.collectAsState()
    val savedApiKey by settingsVm.geminiApiKey.collectAsState()
    val savedOcrUrl by settingsVm.ocrServerUrl.collectAsState()
    val savedLocationId by settingsVm.locationId.collectAsState()

    val uiState by mainViewModel.uiState.collectAsState()
    val isConnected = uiState.connectionState == ConnectionState.CONNECTED ||
                      uiState.connectionState == ConnectionState.CONNECTING

    var host by rememberSaveable { mutableStateOf(savedHost) }
    var portStr by rememberSaveable { mutableStateOf(savedPort.toString()) }
    var frameIntervalMs by rememberSaveable { mutableStateOf(savedFrameIntervalMs.toFloat()) }
    var scanIntervalMs by rememberSaveable { mutableStateOf(savedScanIntervalMs.toFloat()) }
    var recentBufferMs by rememberSaveable { mutableStateOf(savedRecentBufferMs.toFloat()) }
    var noiseGateStr by rememberSaveable { mutableStateOf("%.3f".format(savedVad)) }
    var startVolStr by rememberSaveable { mutableStateOf("%.3f".format(savedStart)) }
    var apiKey by rememberSaveable { mutableStateOf(savedApiKey) }
    var ocrUrl by rememberSaveable { mutableStateOf(savedOcrUrl) }
    var locationId by rememberSaveable { mutableStateOf(savedLocationId) }

    val focusManager = LocalFocusManager.current

    fun doConnect() {
        val port = portStr.toIntOrNull() ?: 50051
        val intervalMs = frameIntervalMs.toInt()
        val scanMs = scanIntervalMs.toInt()
        val bufferMs = recentBufferMs.toInt()
        val noiseGate = noiseGateStr.toFloatOrNull()?.coerceIn(0.001f, 1f) ?: 0.03f
        val startVol = startVolStr.toFloatOrNull()?.coerceIn(0.001f, 1f) ?: 0.05f
        settingsVm.setServerHost(host)
        settingsVm.setServerPort(port)
        settingsVm.setFrameIntervalMs(intervalMs)
        settingsVm.setScanIntervalMs(scanMs)
        settingsVm.setRecentBufferMs(bufferMs)
        settingsVm.setVadThreshold(noiseGate)
        settingsVm.setStartThreshold(startVol)
        settingsVm.setGeminiApiKey(apiKey)
        settingsVm.setOcrServerUrl(ocrUrl)
        settingsVm.setLocationId(locationId)
        settingsVm.save()
        focusManager.clearFocus()
        onConnect(host, port, intervalMs, scanMs, bufferMs, noiseGate, startVol, apiKey, ocrUrl, locationId)
        onBack()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Server Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 20.dp, vertical = 12.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Row 1: Noise gate + Start volume text fields
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = noiseGateStr,
                    onValueChange = { noiseGateStr = it },
                    label = { Text("Noise Gate") },
                    placeholder = { Text("0.030") },
                    supportingText = { Text("min RMS to keep buffering") },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Decimal,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Right) })
                )
                Spacer(Modifier.width(12.dp))
                OutlinedTextField(
                    value = startVolStr,
                    onValueChange = { startVolStr = it },
                    label = { Text("Start Volume") },
                    placeholder = { Text("0.050") },
                    supportingText = { Text("peak needed to start") },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Decimal,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
                )
            }

            // Row 2: mapping-mode send interval — governs walking/guiding
            // only. No blur/clarity filtering (removed client- and
            // server-side) — whichever frame arrives once this interval has
            // elapsed since the last send is forwarded directly.
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Mapping frame interval: ${frameIntervalMs.toInt()} ms — walking/guiding")
                Slider(
                    value = frameIntervalMs,
                    onValueChange = { frameIntervalMs = it },
                    valueRange = 100f..5000f,
                    steps = 48,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            // Row 2a: scan mode's own, much tighter send interval — same
            // no-filtering behavior as above, just a separate slider since a
            // scan pass wants denser coverage than ambient walking/guiding
            // steering needs.
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Scan frame interval: ${scanIntervalMs.toInt()} ms — scanning only")
                Slider(
                    value = scanIntervalMs,
                    onValueChange = { scanIntervalMs = it },
                    valueRange = 50f..500f,
                    steps = 8,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            // Row 2b: rolling recent-frame buffer — governs tracking/reading/
            // Q&A/idle modes instead. No timespan-boundary or send-gap logic
            // here: the buffer just always holds the last recentBufferMs of
            // frames, and the clearest one in it is used whenever a frame is
            // actually needed (OCR, run_detection, tracking init) — see
            // CameraManager.kt's clearestRecentFrame().
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Recent-frame buffer: ${recentBufferMs.toInt()} ms — tracking/reading/Q&A")
                Slider(
                    value = recentBufferMs,
                    onValueChange = { recentBufferMs = it },
                    valueRange = 0f..1000f,
                    steps = 19,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            // Row 2.5: Gemini API key (Live session now runs directly on-device)
            OutlinedTextField(
                value = apiKey,
                onValueChange = { apiKey = it },
                label = { Text("Gemini API Key") },
                placeholder = { Text("AIza...") },
                supportingText = { Text("Used to connect directly to Gemini Live from this device") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
                keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
            )

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = ocrUrl,
                    onValueChange = { ocrUrl = it },
                    label = { Text("OCR Server URL") },
                    placeholder = { Text("http://192.168.1.100:8100") },
                    modifier = Modifier.weight(2f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, imeAction = ImeAction.Next),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Right) })
                )
                Spacer(Modifier.width(8.dp))
                OutlinedTextField(
                    value = locationId,
                    onValueChange = { locationId = it },
                    label = { Text("Location") },
                    placeholder = { Text("default") },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
                )
            }

            // Row 3: IP + Port + Connect/Disconnect toggle
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = host,
                    onValueChange = { host = it },
                    label = { Text("Server IP") },
                    placeholder = { Text("192.168.1.100") },
                    modifier = Modifier.weight(2.5f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Uri,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Right) })
                )
                Spacer(Modifier.width(8.dp))
                OutlinedTextField(
                    value = portStr,
                    onValueChange = { portStr = it.filter { c -> c.isDigit() } },
                    label = { Text("Port") },
                    placeholder = { Text("50051") },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Number,
                        imeAction = ImeAction.Done
                    ),
                    keyboardActions = KeyboardActions(onDone = { doConnect() })
                )
                Spacer(Modifier.width(8.dp))
                Button(
                    onClick = {
                        if (isConnected) {
                            mainViewModel.disconnect()
                            onBack()
                        } else {
                            doConnect()
                        }
                    },
                    modifier = Modifier.weight(1.2f),
                    colors = if (isConnected) ButtonDefaults.buttonColors(containerColor = Color(0xFFB71C1C))
                             else ButtonDefaults.buttonColors()
                ) {
                    Text(if (isConnected) "Disconnect" else "Connect")
                }
            }
            Spacer(Modifier.height(12.dp))
        }
    }
}
