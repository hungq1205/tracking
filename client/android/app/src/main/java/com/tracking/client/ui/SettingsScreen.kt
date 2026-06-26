package com.tracking.client.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
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
    onConnect: (String, Int, Int, Float, Float) -> Unit,
    onBack: () -> Unit
) {
    val settingsVm: SettingsViewModel = viewModel()
    val savedHost by settingsVm.serverHost.collectAsState()
    val savedPort by settingsVm.serverPort.collectAsState()
    val savedFps by settingsVm.targetFps.collectAsState()
    val savedVad by settingsVm.vadThreshold.collectAsState()
    val savedStart by settingsVm.startThreshold.collectAsState()

    val uiState by mainViewModel.uiState.collectAsState()
    val isConnected = uiState.connectionState == ConnectionState.CONNECTED ||
                      uiState.connectionState == ConnectionState.CONNECTING

    var host by rememberSaveable { mutableStateOf(savedHost) }
    var portStr by rememberSaveable { mutableStateOf(savedPort.toString()) }
    var fps by rememberSaveable { mutableStateOf(savedFps.toFloat()) }
    var noiseGateStr by rememberSaveable { mutableStateOf("%.3f".format(savedVad)) }
    var startVolStr by rememberSaveable { mutableStateOf("%.3f".format(savedStart)) }

    val focusManager = LocalFocusManager.current

    fun doConnect() {
        val port = portStr.toIntOrNull() ?: 50051
        val fpsInt = fps.toInt()
        val noiseGate = noiseGateStr.toFloatOrNull()?.coerceIn(0.001f, 1f) ?: 0.03f
        val startVol = startVolStr.toFloatOrNull()?.coerceIn(0.001f, 1f) ?: 0.05f
        settingsVm.setServerHost(host)
        settingsVm.setServerPort(port)
        settingsVm.setTargetFps(fpsInt)
        settingsVm.setVadThreshold(noiseGate)
        settingsVm.setStartThreshold(startVol)
        settingsVm.save()
        focusManager.clearFocus()
        onConnect(host, port, fpsInt, noiseGate, startVol)
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
                .padding(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.SpaceBetween
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

            // Row 2: FPS slider
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Camera FPS: ${fps.toInt()}")
                Slider(
                    value = fps,
                    onValueChange = { fps = it },
                    valueRange = 1f..30f,
                    steps = 28,
                    modifier = Modifier.fillMaxWidth()
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
        }
    }
}
