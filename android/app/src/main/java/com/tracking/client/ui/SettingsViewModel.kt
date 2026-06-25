package com.tracking.client.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class SettingsViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences("tracking_prefs", Context.MODE_PRIVATE)

    private val _serverHost = MutableStateFlow(prefs.getString("server_host", "192.168.1.100") ?: "192.168.1.100")
    val serverHost: StateFlow<String> = _serverHost

    private val _serverPort = MutableStateFlow(prefs.getInt("server_port", 50051))
    val serverPort: StateFlow<Int> = _serverPort

    private val _targetFps = MutableStateFlow(prefs.getInt("target_fps", 10))
    val targetFps: StateFlow<Int> = _targetFps

    private val _vadThreshold = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("vad_threshold_bits", java.lang.Float.floatToIntBits(0.03f)))
    )
    val vadThreshold: StateFlow<Float> = _vadThreshold

    private val _startThreshold = MutableStateFlow(
        java.lang.Float.intBitsToFloat(prefs.getInt("start_threshold_bits", java.lang.Float.floatToIntBits(0.05f)))
    )
    val startThreshold: StateFlow<Float> = _startThreshold

    fun setServerHost(host: String) { _serverHost.value = host }
    fun setServerPort(port: Int) { _serverPort.value = port }
    fun setTargetFps(fps: Int) { _targetFps.value = fps }
    fun setVadThreshold(v: Float) { _vadThreshold.value = v }
    fun setStartThreshold(v: Float) { _startThreshold.value = v }

    fun save() {
        prefs.edit()
            .putString("server_host", _serverHost.value)
            .putInt("server_port", _serverPort.value)
            .putInt("target_fps", _targetFps.value)
            .putInt("vad_threshold_bits", java.lang.Float.floatToIntBits(_vadThreshold.value))
            .putInt("start_threshold_bits", java.lang.Float.floatToIntBits(_startThreshold.value))
            .apply()
    }
}
