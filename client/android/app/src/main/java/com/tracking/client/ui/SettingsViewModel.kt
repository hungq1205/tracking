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
    private val _scanIntervalMs = MutableStateFlow(prefs.getInt("scan_interval_ms", 100))
    val scanIntervalMs: StateFlow<Int> = _scanIntervalMs

    // Rolling recent-frame buffer for tracking/reading/Q&A modes (see
    // CameraManager.kt) — separate from frameIntervalMs/scanIntervalMs
    // above, which only govern mapping-mode (walking/guiding/scanning).
    private val _recentBufferMs = MutableStateFlow(prefs.getInt("recent_buffer_ms", 100))
    val recentBufferMs: StateFlow<Int> = _recentBufferMs

    private val _vadThreshold = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("vad_threshold_bits", java.lang.Float.floatToIntBits(0.03f)))
    )
    val vadThreshold: StateFlow<Float> = _vadThreshold

    private val _startThreshold = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("start_threshold_bits", java.lang.Float.floatToIntBits(0.05f)))
    )
    val startThreshold: StateFlow<Float> = _startThreshold

    // Gemini Live now runs directly on-device (see CLAUDE.md's
    // "Client-Orchestrated Live Session" section) — the API key is embedded
    // in app-local storage rather than a broker, per the accepted tradeoff.
    private val _geminiApiKey = MutableStateFlow(prefs.getString("gemini_api_key", "") ?: "")
    val geminiApiKey: StateFlow<String> = _geminiApiKey

    // OCR is now a direct 3rd-party call from the client (bypasses the gRPC
    // server as a proxy) — same paddle_ocr_server microservice, just called
    // straight from here.
    private val _ocrServerUrl = MutableStateFlow(prefs.getString("ocr_server_url", "http://192.168.1.15:8100") ?: "http://192.168.1.15:8100")
    val ocrServerUrl: StateFlow<String> = _ocrServerUrl

    // Identifies which MappingService location this device is guiding/
    // walking in — matches server/data/maps/{location_id}.
    private val _locationId = MutableStateFlow(prefs.getString("location_id", "default") ?: "default")
    val locationId: StateFlow<String> = _locationId

    fun setServerHost(host: String) { _serverHost.value = host }
    fun setServerPort(port: Int) { _serverPort.value = port }
    fun setFrameIntervalMs(ms: Int) { _frameIntervalMs.value = ms }
    fun setScanIntervalMs(ms: Int) { _scanIntervalMs.value = ms }
    fun setRecentBufferMs(ms: Int) { _recentBufferMs.value = ms }
    fun setVadThreshold(v: Float) { _vadThreshold.value = v }
    fun setStartThreshold(v: Float) { _startThreshold.value = v }
    fun setGeminiApiKey(key: String) { _geminiApiKey.value = key }
    fun setOcrServerUrl(url: String) { _ocrServerUrl.value = url }
    fun setLocationId(id: String) { _locationId.value = id }

    fun save() {
        prefs.edit()
            .putString("server_host", _serverHost.value)
            .putInt("server_port", _serverPort.value)
            .putInt("frame_interval_ms", _frameIntervalMs.value)
            .putInt("scan_interval_ms", _scanIntervalMs.value)
            .putInt("recent_buffer_ms", _recentBufferMs.value)
            .putInt("vad_threshold_bits", java.lang.Float.floatToIntBits(_vadThreshold.value))
            .putInt("start_threshold_bits", java.lang.Float.floatToIntBits(_startThreshold.value))
            .putString("gemini_api_key", _geminiApiKey.value)
            .putString("ocr_server_url", _ocrServerUrl.value)
            .putString("location_id", _locationId.value)
            .apply()
    }
}
