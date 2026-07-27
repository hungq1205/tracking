package com.tracking.client.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class SettingsViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences("tracking_prefs", Context.MODE_PRIVATE)

    private val _serverHost = MutableStateFlow(prefs.getString("server_host", "192.168.1.15") ?: "192.168.1.15")
    val serverHost: StateFlow<String> = _serverHost

    private val _serverPort = MutableStateFlow(prefs.getInt("server_port", 50051))
    val serverPort: StateFlow<Int> = _serverPort

    // "Timespan" for client-side blur-aware frame selection (see
    // CameraManager.kt) — replaces the old fixed target_fps. Frames are
    // accumulated within this window and the sharpest one is sent, never
    // more often than every frameIntervalMs/2.
    private val _frameIntervalMs = MutableStateFlow(prefs.getInt("frame_interval_ms", 1000))
    val frameIntervalMs: StateFlow<Int> = _frameIntervalMs

    // Scan mode's own, much tighter window (see CameraManager.kt) — kept
    // separate from frameIntervalMs above (walking/guiding only) since a
    // scan pass wants denser frame coverage than ambient steering does.
    private val _scanIntervalMs = MutableStateFlow(prefs.getInt("scan_interval_ms", 200))
    val scanIntervalMs: StateFlow<Int> = _scanIntervalMs

    // Rolling recent-frame buffer for tracking/reading/Q&A modes (see
    // CameraManager.kt) — separate from frameIntervalMs/scanIntervalMs
    // above, which only govern mapping-mode (guiding/scanning).
    private val _recentBufferMs = MutableStateFlow(prefs.getInt("recent_buffer_ms", 100))
    val recentBufferMs: StateFlow<Int> = _recentBufferMs

    // Local reactive HRTF obstacle-dodge tick rate (walking AND guiding —
    // see ToolDispatcher.runAvoidanceTick()/CLAUDE.md's "Local reactive
    // HRTF obstacle-dodge" note). Independent of frameIntervalMs above,
    // which only governs guiding's separate, slower MappingService route
    // stream — this is the faster, no-world-map reactive layer.
    private val _avoidanceIntervalMs = MutableStateFlow(prefs.getInt("avoidance_interval_ms", 350))
    val avoidanceIntervalMs: StateFlow<Int> = _avoidanceIntervalMs

    // The walking/guiding HRTF beacon's fixed circle — elevation (negative =
    // below ear level, toward the torso) and radius (gain-falloff distance
    // only; HRTF convolution has no real distance rendering — see
    // HrtfBeaconPlayer.kt). See ToolDispatcher.kt's beaconElevationDeg/
    // beaconRadiusM.
    private val _beaconElevationDeg = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("beacon_elevation_deg_bits", java.lang.Float.floatToIntBits(-20f)))
    )
    val beaconElevationDeg: StateFlow<Float> = _beaconElevationDeg

    private val _beaconRadiusM = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("beacon_radius_m_bits", java.lang.Float.floatToIntBits(6f)))
    )
    val beaconRadiusM: StateFlow<Float> = _beaconRadiusM

    private val _vadThreshold = MutableStateFlow(
        // Was 0.03f — too high relative to computeRms()'s actual normalized-
        // RMS output range for typical phone-mic speech (roughly 0.005-0.05),
        // meaning ordinary speaking volume rarely crossed it and the user
        // had to shout close to the mic. Lowered alongside startThreshold
        // below and OUTPUT_ACTIVE_THRESHOLD_MULTIPLIER's own reduction (see
        // ContinuousVadRecorder.kt).
        java.lang.Float.intBitsToFloat(prefs.getInt("vad_threshold_bits", java.lang.Float.floatToIntBits(0.012f)))
    )
    val vadThreshold: StateFlow<Float> = _vadThreshold

    private val _startThreshold = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("start_threshold_bits", java.lang.Float.floatToIntBits(0.018f)))
    )
    val startThreshold: StateFlow<Float> = _startThreshold

    // Master volume multiplier (0f..1f) for the walking/guiding HRTF "cue"
    // (the looped fluttering.mp3 beacon) — applied on top of HrtfBeaconPlayer's
    // own distance-based gain falloff, see HrtfBeaconPlayer.cueVolume.
    private val _cueVolume = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("cue_volume_bits", java.lang.Float.floatToIntBits(1f)))
    )
    val cueVolume: StateFlow<Float> = _cueVolume

    // Master volume for Gemini Live's own spoken voice (StreamingAudioPlayer)
    // — requested directly by the user after finding it noticeably louder
    // than the Pixie cue, with no way to balance the two.
    private val _geminiVoiceVolume = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("gemini_voice_volume_bits", java.lang.Float.floatToIntBits(1f)))
    )
    val geminiVoiceVolume: StateFlow<Float> = _geminiVoiceVolume

    // Master volume for everything else that isn't Gemini's voice or the
    // Pixie cue — reading-mode TTS (ReadingTtsPlayer) and music/radio/
    // resolved-YouTube-stream playback (PlaybackService's ExoPlayer). Does
    // NOT cover the embedded YouTube IFrame player itself (a WebView with
    // its own independent volume, not reachable from here).
    private val _otherSoundVolume = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("other_sound_volume_bits", java.lang.Float.floatToIntBits(1f)))
    )
    val otherSoundVolume: StateFlow<Float> = _otherSoundVolume

    // Gemini Live now runs directly on-device (see CLAUDE.md's
    // "Client-Orchestrated Live Session" section) — the API key is embedded
    // in app-local storage rather than a broker, per the accepted tradeoff.
    private val _geminiApiKey = MutableStateFlow(prefs.getString("gemini_api_key", "") ?: "")
    val geminiApiKey: StateFlow<String> = _geminiApiKey

    // OCR is a direct 3rd-party call from the client (bypasses the gRPC
    // server as a proxy) — OCR.space (free hosted OCR), replacing the
    // earlier self-hosted paddle_ocr_server call. "helloworld" is
    // OCR.space's own public rate-limited test key.
    private val _ocrApiKey = MutableStateFlow(prefs.getString("ocr_api_key", "helloworld") ?: "helloworld")
    val ocrApiKey: StateFlow<String> = _ocrApiKey

    // YouTube Data API v3 key — used by YouTubeSearchClient.kt for
    // search_youtube/get_video_info (direct 3rd-party call, same "no server
    // proxy" convention as OCR above). Playback itself uses the official
    // android-youtube-player (IFrame) library, not this key.
    private val _youtubeApiKey = MutableStateFlow(prefs.getString("youtube_api_key", "") ?: "")
    val youtubeApiKey: StateFlow<String> = _youtubeApiKey

    // Reading mode's blur skip/retry (see ToolDispatcher.acquireSharpFrame())
    // — a frame scoring below this Laplacian-variance threshold is
    // re-sampled instead of OCR'd. 0 disables the check entirely.
    private val _blurSharpnessThreshold = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("blur_sharpness_threshold_bits", java.lang.Float.floatToIntBits(40f)))
    )
    val blurSharpnessThreshold: StateFlow<Float> = _blurSharpnessThreshold

    // Debug-only: writes every OCR'd frame (with its kept/dropped text
    // boxes drawn on) to app-private storage — see DebugFrameStore.kt. Off
    // by default; real storage/privacy cost if left on.
    private val _saveDebugOcrFrames = MutableStateFlow(prefs.getBoolean("save_debug_ocr_frames", false))
    val saveDebugOcrFrames: StateFlow<Boolean> = _saveDebugOcrFrames

    // Identifies which MappingService location this device is guiding/
    // walking in — matches server/data/maps/{location_id}.
    private val _locationId = MutableStateFlow(prefs.getString("location_id", "default") ?: "default")
    val locationId: StateFlow<String> = _locationId

    // Camera/mic/speaker source toggle — off (default) means the phone
    // itself is the edge device (LocalEdgeDevice); on means a remote
    // Raspberry-Pi-class device is, talked to over ZeroMQ (RemoteEdgeDevice)
    // — see EdgeDevice.kt/RemoteEdgeDevice.kt.
    private val _useRemoteEdgeDevice = MutableStateFlow(prefs.getBoolean("use_remote_edge_device", false))
    val useRemoteEdgeDevice: StateFlow<Boolean> = _useRemoteEdgeDevice

    private val _edgeDeviceHost = MutableStateFlow(prefs.getString("edge_device_host", "") ?: "")
    val edgeDeviceHost: StateFlow<String> = _edgeDeviceHost

    fun setServerHost(host: String) { _serverHost.value = host }
    fun setServerPort(port: Int) { _serverPort.value = port }
    fun setFrameIntervalMs(ms: Int) { _frameIntervalMs.value = ms }
    fun setScanIntervalMs(ms: Int) { _scanIntervalMs.value = ms }
    fun setRecentBufferMs(ms: Int) { _recentBufferMs.value = ms }
    fun setAvoidanceIntervalMs(ms: Int) { _avoidanceIntervalMs.value = ms }
    fun setBeaconElevationDeg(deg: Float) { _beaconElevationDeg.value = deg }
    fun setBeaconRadiusM(m: Float) { _beaconRadiusM.value = m }
    fun setVadThreshold(v: Float) { _vadThreshold.value = v }
    fun setStartThreshold(v: Float) { _startThreshold.value = v }
    fun setCueVolume(v: Float) { _cueVolume.value = v }
    fun setGeminiVoiceVolume(v: Float) { _geminiVoiceVolume.value = v }
    fun setOtherSoundVolume(v: Float) { _otherSoundVolume.value = v }
    fun setGeminiApiKey(key: String) { _geminiApiKey.value = key }
    fun setOcrApiKey(key: String) { _ocrApiKey.value = key }
    fun setYoutubeApiKey(key: String) { _youtubeApiKey.value = key }
    fun setBlurSharpnessThreshold(v: Float) { _blurSharpnessThreshold.value = v }
    fun setSaveDebugOcrFrames(v: Boolean) { _saveDebugOcrFrames.value = v }
    fun setLocationId(id: String) { _locationId.value = id }
    fun setUseRemoteEdgeDevice(v: Boolean) { _useRemoteEdgeDevice.value = v }
    fun setEdgeDeviceHost(host: String) { _edgeDeviceHost.value = host }

    fun save() {
        prefs.edit()
            .putString("server_host", _serverHost.value)
            .putInt("server_port", _serverPort.value)
            .putInt("frame_interval_ms", _frameIntervalMs.value)
            .putInt("scan_interval_ms", _scanIntervalMs.value)
            .putInt("recent_buffer_ms", _recentBufferMs.value)
            .putInt("avoidance_interval_ms", _avoidanceIntervalMs.value)
            .putInt("beacon_elevation_deg_bits", java.lang.Float.floatToIntBits(_beaconElevationDeg.value))
            .putInt("beacon_radius_m_bits", java.lang.Float.floatToIntBits(_beaconRadiusM.value))
            .putInt("vad_threshold_bits", java.lang.Float.floatToIntBits(_vadThreshold.value))
            .putInt("start_threshold_bits", java.lang.Float.floatToIntBits(_startThreshold.value))
            .putInt("cue_volume_bits", java.lang.Float.floatToIntBits(_cueVolume.value))
            .putInt("gemini_voice_volume_bits", java.lang.Float.floatToIntBits(_geminiVoiceVolume.value))
            .putInt("other_sound_volume_bits", java.lang.Float.floatToIntBits(_otherSoundVolume.value))
            .putString("gemini_api_key", _geminiApiKey.value)
            .putString("ocr_api_key", _ocrApiKey.value)
            .putString("youtube_api_key", _youtubeApiKey.value)
            .putInt("blur_sharpness_threshold_bits", java.lang.Float.floatToIntBits(_blurSharpnessThreshold.value))
            .putBoolean("save_debug_ocr_frames", _saveDebugOcrFrames.value)
            .putString("location_id", _locationId.value)
            .putBoolean("use_remote_edge_device", _useRemoteEdgeDevice.value)
            .putString("edge_device_host", _edgeDeviceHost.value)
            .apply()
    }
}
