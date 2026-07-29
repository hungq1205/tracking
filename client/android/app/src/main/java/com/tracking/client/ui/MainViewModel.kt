package com.tracking.client.ui

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.tracking.client.camera.CameraManager
import com.tracking.client.live.LiveAssistantService
import com.tracking.client.model.AppUiState
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

/**
 * Thin bound-client facade over [LiveAssistantService], which now owns the
 * entire Gemini Live session graph (gRPC, camera, mic, tool dispatch) — see
 * CLAUDE.md's "Client-Orchestrated Live Session" section. This class used to
 * own that graph directly inside viewModelScope, which meant the whole
 * session died whenever the Activity/ViewModelStore was torn down (screen
 * off, app swiped from Recents). It now just binds to the Service (started
 * as a foreground service so it survives independently of this binding) and
 * delegates every call, re-exposing the Service's own StateFlow as its own.
 */
class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val _boundService = MutableStateFlow<LiveAssistantService?>(null)

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val service = (binder as? LiveAssistantService.LocalBinder)?.getService()
            _boundService.value = service
            pendingPreviewView?.let { service?.cameraManager?.attachPreviewSurface(it) }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            _boundService.value = null
        }
    }

    // Remembered in case a PreviewView attaches before the Service binding
    // completes — see the same race/fallback in CameraManager's own
    // pendingPreviewView.
    private var pendingPreviewView: PreviewView? = null

    @OptIn(ExperimentalCoroutinesApi::class)
    val uiState: StateFlow<AppUiState> = _boundService
        .flatMapLatest { service -> service?.uiState ?: flowOf(AppUiState()) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, AppUiState())

    /** Latest JPEG received from a remote edge device's camera (null unless
     * "Use Remote Edge Device" is active) — MainScreen shows this in place
     * of the phone's own CameraX preview when non-null. */
    @OptIn(ExperimentalCoroutinesApi::class)
    val edgeFrame: StateFlow<ByteArray?> = _boundService
        .flatMapLatest { service -> service?.edgeFrame ?: flowOf(null) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    @OptIn(ExperimentalCoroutinesApi::class)
    val isRemoteEdgeActive: StateFlow<Boolean> = _boundService
        .flatMapLatest { service -> service?.isRemoteEdgeActive ?: flowOf(false) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, false)

    /** Latest JPEG frame actually sent to OCR.space (scan_current_view()/
     * live-reading) — null when nothing has been scanned yet this session.
     * MainScreen overlays this so reading-mode scans are visible on screen
     * instead of happening invisibly. */
    @OptIn(ExperimentalCoroutinesApi::class)
    val ocrFrame: StateFlow<ByteArray?> = _boundService
        .flatMapLatest { service -> service?.ocrFrame ?: flowOf(null) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, null)

    /** Exposed only for [attachCameraPreview]'s reuse of CameraManager's own
     * pending-attach fallback — not read directly by the UI layer any more. */
    private val cameraManager: CameraManager?
        get() = _boundService.value?.cameraManager

    init {
        val intent = Intent(app, LiveAssistantService::class.java)
        // Start it as a genuine foreground service independent of this
        // binding — the Activity going away must not stop it.
        app.startForegroundService(intent)
        app.bindService(intent, connection, Context.BIND_AUTO_CREATE)
    }

    /** Replaces the old direct `cameraManager.bind(lifecycleOwner, previewView)`
     * call site — camera binding itself now happens once, inside the
     * Service's own onCreate(), against the Service's lifecycle. This just
     * plugs the Activity's on-screen PreviewView into the already-bound
     * Preview use case (see CameraManager.attachPreviewSurface). */
    fun attachCameraPreview(previewView: PreviewView) {
        pendingPreviewView = previewView
        cameraManager?.attachPreviewSurface(previewView)
    }

    fun detachCameraPreview() {
        pendingPreviewView = null
        cameraManager?.detachPreviewSurface()
    }

    fun connect(
        host: String, port: Int, frameIntervalMs: Int, scanIntervalMs: Int, recentBufferMs: Int,
        avoidanceIntervalMs: Int = 350,
        beaconElevationDeg: Float = -20f, beaconRadiusM: Float = 6f,
        vadThreshold: Float = 0.012f, startThreshold: Float = 0.018f,
        geminiApiKey: String, ocrApiKey: String, locationId: String,
        blurSharpnessThreshold: Float = 40f, saveDebugOcrFrames: Boolean = false,
        youtubeApiKey: String = "",
        cueVolume: Float = 1f,
        geminiVoiceVolume: Float = 1f,
        otherSoundVolume: Float = 1f,
        useRemoteEdgeDevice: Boolean = false,
        edgeDeviceHost: String = "",
    ) {
        _boundService.value?.connect(
            host, port, frameIntervalMs, scanIntervalMs, recentBufferMs,
            avoidanceIntervalMs, beaconElevationDeg, beaconRadiusM,
            vadThreshold, startThreshold, geminiApiKey, ocrApiKey, locationId,
            blurSharpnessThreshold, saveDebugOcrFrames, youtubeApiKey, cueVolume,
            geminiVoiceVolume, otherSoundVolume, useRemoteEdgeDevice, edgeDeviceHost,
        )
    }

    fun disconnect() {
        _boundService.value?.disconnect()
    }

    fun clearError() { _boundService.value?.clearError() }

    override fun onCleared() {
        super.onCleared()
        // Deliberately does NOT call disconnect() — the whole point of the
        // Service migration is that the session outlives this ViewModel.
        // Only the binding itself is torn down.
        getApplication<Application>().unbindService(connection)
    }
}
