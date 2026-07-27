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
import androidx.compose.material3.Switch
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings as AndroidSettings
import com.tracking.client.model.ConnectionState
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    mainViewModel: MainViewModel,
    onConnect: (String, Int, Int, Int, Int, Int, Float, Float, Float, Float, String, String, String, Float, Boolean, String, Float, Float, Float, Boolean, String) -> Unit,
    onBack: () -> Unit,
    onOpenEdgeTest: () -> Unit = {},
) {
    val settingsVm: SettingsViewModel = viewModel()
    val savedHost by settingsVm.serverHost.collectAsState()
    val savedPort by settingsVm.serverPort.collectAsState()
    val savedFrameIntervalMs by settingsVm.frameIntervalMs.collectAsState()
    val savedScanIntervalMs by settingsVm.scanIntervalMs.collectAsState()
    val savedRecentBufferMs by settingsVm.recentBufferMs.collectAsState()
    val savedAvoidanceIntervalMs by settingsVm.avoidanceIntervalMs.collectAsState()
    val savedBeaconElevationDeg by settingsVm.beaconElevationDeg.collectAsState()
    val savedBeaconRadiusM by settingsVm.beaconRadiusM.collectAsState()
    val savedVad by settingsVm.vadThreshold.collectAsState()
    val savedStart by settingsVm.startThreshold.collectAsState()
    val savedCueVolume by settingsVm.cueVolume.collectAsState()
    val savedGeminiVoiceVolume by settingsVm.geminiVoiceVolume.collectAsState()
    val savedOtherSoundVolume by settingsVm.otherSoundVolume.collectAsState()
    val savedApiKey by settingsVm.geminiApiKey.collectAsState()
    val savedOcrApiKey by settingsVm.ocrApiKey.collectAsState()
    val savedYoutubeApiKey by settingsVm.youtubeApiKey.collectAsState()
    val savedBlurThreshold by settingsVm.blurSharpnessThreshold.collectAsState()
    val savedSaveDebugFrames by settingsVm.saveDebugOcrFrames.collectAsState()
    val savedLocationId by settingsVm.locationId.collectAsState()
    val savedUseRemoteEdgeDevice by settingsVm.useRemoteEdgeDevice.collectAsState()
    val savedEdgeDeviceHost by settingsVm.edgeDeviceHost.collectAsState()

    val uiState by mainViewModel.uiState.collectAsState()
    val isConnected = uiState.connectionState == ConnectionState.CONNECTED ||
                      uiState.connectionState == ConnectionState.CONNECTING

    var host by rememberSaveable { mutableStateOf(savedHost) }
    var portStr by rememberSaveable { mutableStateOf(savedPort.toString()) }
    // FPS in the UI, milliseconds internally (CameraManager.kt/persistence
    // both stay ms-based — see the two number-input fields below for the
    // fps<->ms conversion, kept local to this screen).
    var frameFpsStr by rememberSaveable { mutableStateOf("%.2f".format(1000f / savedFrameIntervalMs)) }
    var scanFpsStr by rememberSaveable { mutableStateOf("%.2f".format(1000f / savedScanIntervalMs)) }
    var avoidanceFpsStr by rememberSaveable { mutableStateOf("%.2f".format(1000f / savedAvoidanceIntervalMs)) }
    var beaconElevationStr by rememberSaveable { mutableStateOf("%.0f".format(savedBeaconElevationDeg)) }
    var beaconRadiusStr by rememberSaveable { mutableStateOf("%.1f".format(savedBeaconRadiusM)) }
    var recentBufferMs by rememberSaveable { mutableStateOf(savedRecentBufferMs.toFloat()) }
    var cueVolume by rememberSaveable { mutableStateOf(savedCueVolume) }
    var geminiVoiceVolume by rememberSaveable { mutableStateOf(savedGeminiVoiceVolume) }
    var otherSoundVolume by rememberSaveable { mutableStateOf(savedOtherSoundVolume) }
    var noiseGateStr by rememberSaveable { mutableStateOf("%.3f".format(savedVad)) }
    var startVolStr by rememberSaveable { mutableStateOf("%.3f".format(savedStart)) }
    var apiKey by rememberSaveable { mutableStateOf(savedApiKey) }
    var ocrApiKey by rememberSaveable { mutableStateOf(savedOcrApiKey) }
    var youtubeApiKey by rememberSaveable { mutableStateOf(savedYoutubeApiKey) }
    var blurThresholdStr by rememberSaveable { mutableStateOf("%.0f".format(savedBlurThreshold)) }
    var saveDebugFrames by rememberSaveable { mutableStateOf(savedSaveDebugFrames) }
    var locationId by rememberSaveable { mutableStateOf(savedLocationId) }
    var useRemoteEdgeDevice by rememberSaveable { mutableStateOf(savedUseRemoteEdgeDevice) }
    var edgeDeviceHost by rememberSaveable { mutableStateOf(savedEdgeDeviceHost) }

    val focusManager = LocalFocusManager.current

    fun doConnect() {
        val port = portStr.toIntOrNull() ?: 50051
        // fps -> ms, no artificial min/max — only guarding against a
        // zero/negative/unparseable value, which would otherwise divide
        // into an infinite or nonsensical interval.
        val frameFps = frameFpsStr.toFloatOrNull()?.takeIf { it > 0f } ?: 1f
        val scanFps = scanFpsStr.toFloatOrNull()?.takeIf { it > 0f } ?: 5f
        val avoidanceFps = avoidanceFpsStr.toFloatOrNull()?.takeIf { it > 0f } ?: 2.86f
        val intervalMs = (1000f / frameFps).roundToInt()
        val scanMs = (1000f / scanFps).roundToInt()
        val avoidanceMs = (1000f / avoidanceFps).roundToInt()
        val bufferMs = recentBufferMs.toInt()
        val beaconElevation = beaconElevationStr.toFloatOrNull()?.coerceIn(-90f, 90f) ?: -20f
        val beaconRadius = beaconRadiusStr.toFloatOrNull()?.takeIf { it > 0f } ?: 6f
        val noiseGate = noiseGateStr.toFloatOrNull()?.coerceIn(0.001f, 1f) ?: 0.012f
        val startVol = startVolStr.toFloatOrNull()?.coerceIn(0.001f, 1f) ?: 0.018f
        val blurThreshold = blurThresholdStr.toFloatOrNull()?.coerceAtLeast(0f) ?: 40f
        settingsVm.setServerHost(host)
        settingsVm.setServerPort(port)
        settingsVm.setFrameIntervalMs(intervalMs)
        settingsVm.setScanIntervalMs(scanMs)
        settingsVm.setRecentBufferMs(bufferMs)
        settingsVm.setAvoidanceIntervalMs(avoidanceMs)
        settingsVm.setBeaconElevationDeg(beaconElevation)
        settingsVm.setBeaconRadiusM(beaconRadius)
        settingsVm.setVadThreshold(noiseGate)
        settingsVm.setStartThreshold(startVol)
        settingsVm.setCueVolume(cueVolume)
        settingsVm.setGeminiVoiceVolume(geminiVoiceVolume)
        settingsVm.setOtherSoundVolume(otherSoundVolume)
        settingsVm.setGeminiApiKey(apiKey)
        settingsVm.setOcrApiKey(ocrApiKey)
        settingsVm.setYoutubeApiKey(youtubeApiKey)
        settingsVm.setBlurSharpnessThreshold(blurThreshold)
        settingsVm.setSaveDebugOcrFrames(saveDebugFrames)
        settingsVm.setLocationId(locationId)
        settingsVm.setUseRemoteEdgeDevice(useRemoteEdgeDevice)
        settingsVm.setEdgeDeviceHost(edgeDeviceHost)
        settingsVm.save()
        focusManager.clearFocus()
        onConnect(host, port, intervalMs, scanMs, bufferMs, avoidanceMs, beaconElevation, beaconRadius, noiseGate, startVol, apiKey, ocrApiKey, locationId, blurThreshold, saveDebugFrames, youtubeApiKey, cueVolume, geminiVoiceVolume, otherSoundVolume, useRemoteEdgeDevice, edgeDeviceHost)
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
                    placeholder = { Text("0.012") },
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
                    placeholder = { Text("0.018") },
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

            // Row 2: mapping-mode send rate — governs guiding's MappingService
            // route stream only now (walking dropped MappingService
            // entirely, see the Avoidance FPS field below for its actual
            // steering-signal rate). No blur/clarity filtering (removed
            // client- and server-side) — whichever frame arrives once 1/fps
            // seconds have elapsed since the last send is forwarded
            // directly. Entered as FPS (0.2..10), converted to
            // CameraManager's internal ms interval at connect.
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = frameFpsStr,
                    onValueChange = { frameFpsStr = it },
                    label = { Text("Mapping FPS") },
                    placeholder = { Text("1.0") },
                    supportingText = { Text("guiding route only") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Decimal,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
                )
            }

            // Row 2a: scan mode's own, much higher send rate — same
            // no-filtering behavior as above, just a separate field since a
            // scan pass wants denser coverage than ambient walking/guiding
            // steering needs.
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = scanFpsStr,
                    onValueChange = { scanFpsStr = it },
                    label = { Text("Scan FPS") },
                    placeholder = { Text("5.0") },
                    supportingText = { Text("scanning only") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Decimal,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
                )
            }

            // Row 2c: local reactive HRTF obstacle-dodge tick rate — governs
            // walking AND guiding's actual beacon-steering signal now (see
            // CLAUDE.md's "Local reactive HRTF obstacle-dodge" note), fully
            // independent from Mapping FPS above (which now only drives
            // guiding's separate, slower MappingService route stream).
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = avoidanceFpsStr,
                    onValueChange = { avoidanceFpsStr = it },
                    label = { Text("Avoidance FPS") },
                    placeholder = { Text("10") },
                    supportingText = { Text("walking/guiding HRTF steering") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Decimal,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
                )
            }

            // Row 2d: the walking/guiding HRTF beacon's fixed circle around
            // the user's head — elevation (negative = below ear level,
            // toward the torso) and radius (gain-falloff distance only;
            // HRTF convolution has no real distance rendering — see
            // HrtfBeaconPlayer.kt). Threaded into ToolDispatcher's
            // beaconElevationDeg/beaconRadiusM.
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = beaconElevationStr,
                    onValueChange = { beaconElevationStr = it },
                    label = { Text("Beacon Elevation (°)") },
                    placeholder = { Text("-20") },
                    supportingText = { Text("negative = toward torso") },
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
                    value = beaconRadiusStr,
                    onValueChange = { beaconRadiusStr = it },
                    label = { Text("Beacon Radius (m)") },
                    placeholder = { Text("1.0") },
                    supportingText = { Text("gain falloff only") },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Decimal,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
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

            // Row 2c: walking/guiding HRTF cue (fluttering.mp3 loop) master
            // volume — multiplies on top of HrtfBeaconPlayer's own
            // distance-based gain falloff, see HrtfBeaconPlayer.cueVolume.
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Cue Volume: ${(cueVolume * 100).roundToInt()}%")
                Slider(
                    value = cueVolume,
                    onValueChange = { cueVolume = it },
                    valueRange = 0f..1f,
                    steps = 9,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            // Row 2c-i: Gemini Live's own spoken voice — requested directly
            // by the user after finding it noticeably louder than the Pixie
            // cue above, with no way to balance the two. See
            // StreamingAudioPlayer.setVolume().
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Gemini Voice Volume: ${(geminiVoiceVolume * 100).roundToInt()}%")
                Slider(
                    value = geminiVoiceVolume,
                    onValueChange = { geminiVoiceVolume = it },
                    valueRange = 0f..1f,
                    steps = 9,
                    modifier = Modifier.fillMaxWidth()
                )
            }

            // Row 2c-ii: everything else — reading-mode TTS and music/
            // radio/resolved-YouTube-stream playback (NOT the embedded
            // YouTube IFrame player itself, which has its own independent
            // volume). See ReadingTtsPlayer.setVolume()/PlaybackService.kt.
            Column(modifier = Modifier.fillMaxWidth()) {
                Text("Other Sound Volume: ${(otherSoundVolume * 100).roundToInt()}%")
                Slider(
                    value = otherSoundVolume,
                    onValueChange = { otherSoundVolume = it },
                    valueRange = 0f..1f,
                    steps = 9,
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

            // OCR.space — free hosted OCR, replaces paddle_ocr_server as
            // this client's direct 3rd-party OCR call (see OcrClient.kt).
            // "helloworld" is OCR.space's own public rate-limited test key.
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = ocrApiKey,
                    onValueChange = { ocrApiKey = it },
                    label = { Text("OCR.space API Key") },
                    placeholder = { Text("helloworld") },
                    modifier = Modifier.weight(2f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
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

            // YouTube Data API v3 key — search_youtube/get_video_info
            // (direct 3rd-party call, see YouTubeSearchClient.kt). Playback
            // itself uses the official android-youtube-player library, not
            // this key.
            OutlinedTextField(
                value = youtubeApiKey,
                onValueChange = { youtubeApiKey = it },
                label = { Text("YouTube API Key") },
                placeholder = { Text("AIza...") },
                supportingText = { Text("For search_youtube — leave blank to disable") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Next),
                keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
            )

            // Reading mode's blur skip/retry (see
            // ToolDispatcher.acquireSharpFrame()) — a frame scoring below
            // this Laplacian-variance threshold is re-sampled (up to 2
            // retries, 0.5s apart) instead of OCR'd outright. 0 disables.
            // "Save debug OCR frames" (off by default — real storage/
            // privacy cost) writes each scanned frame with its kept/dropped
            // text boxes drawn on to app-private storage, for diagnosing
            // the rotation/noise/blur filters — see DebugFrameStore.kt.
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                OutlinedTextField(
                    value = blurThresholdStr,
                    onValueChange = { blurThresholdStr = it },
                    label = { Text("Blur Threshold") },
                    placeholder = { Text("40") },
                    supportingText = { Text("0 = disabled") },
                    modifier = Modifier.weight(1f),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Decimal,
                        imeAction = ImeAction.Next
                    ),
                    keyboardActions = KeyboardActions(onNext = { focusManager.moveFocus(FocusDirection.Down) })
                )
                Spacer(Modifier.width(12.dp))
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Save debug OCR frames", modifier = Modifier.weight(1f))
                    Switch(checked = saveDebugFrames, onCheckedChange = { saveDebugFrames = it })
                }
            }

            // Battery optimization exemption — LiveAssistantService needs
            // this to reliably keep running in the background (screen off,
            // app swiped from Recents). A special-intent flow, not a
            // runtime permission, so it's a deliberate user action here
            // rather than an automatic prompt on launch.
            run {
                val context = LocalContext.current
                val powerManager = context.getSystemService(PowerManager::class.java)
                val alreadyExempt = powerManager?.isIgnoringBatteryOptimizations(context.packageName) == true
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        if (alreadyExempt) "Battery optimization: exempt" else "Battery optimization: not exempt",
                        modifier = Modifier.weight(1f)
                    )
                    Button(
                        enabled = !alreadyExempt,
                        onClick = {
                            @Suppress("BatteryLife")
                            context.startActivity(
                                Intent(
                                    AndroidSettings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                                    Uri.parse("package:${context.packageName}")
                                )
                            )
                        }
                    ) {
                        Text("Exempt app")
                    }
                }
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

            // Camera/mic/speaker source — off (default) means the phone
            // itself is the edge device; on means a remote Raspberry-Pi-
            // class device is, talked to over ZeroMQ (RemoteEdgeDevice) —
            // see EdgeDevice.kt. Reconnect required for a change to take
            // effect (read once in LiveAssistantService.connect()).
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Use Remote Edge Device (Pi)", modifier = Modifier.weight(1f))
                Switch(checked = useRemoteEdgeDevice, onCheckedChange = { useRemoteEdgeDevice = it })
            }
            if (useRemoteEdgeDevice) {
                OutlinedTextField(
                    value = edgeDeviceHost,
                    onValueChange = { edgeDeviceHost = it },
                    label = { Text("Edge Device IP") },
                    placeholder = { Text("192.168.1.50") },
                    supportingText = { Text("Fixed ports 5601-5604 (mic/frame/audio/luma)") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(
                        keyboardType = KeyboardType.Uri,
                        imeAction = ImeAction.Done
                    ),
                    keyboardActions = KeyboardActions(onDone = { focusManager.clearFocus() })
                )
            }
            Spacer(Modifier.height(12.dp))
            Button(onClick = onOpenEdgeTest, modifier = Modifier.fillMaxWidth()) {
                Text("Edge ZMQ Test (mock Pi device)")
            }
            Spacer(Modifier.height(12.dp))
        }
    }
}
