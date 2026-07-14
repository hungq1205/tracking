package com.tracking.client.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.protobuf.ByteString
import com.tracking.client.camera.CameraManager
import com.tracking.client.grpc.GrpcClientManager
import com.tracking.client.model.ScanUiState
import com.tracking.client.sensors.ImuRecorder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import tracking.Tracking
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class ScanViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences("scan_prefs", Context.MODE_PRIVATE)

    val grpcManager = GrpcClientManager()
    val cameraManager = CameraManager(app)

    private val _uiState = MutableStateFlow(
        ScanUiState(
            host = prefs.getString("scan_host", "192.168.1.100") ?: "192.168.1.100",
            port = prefs.getInt("scan_port", 50052),
            locationId = prefs.getString("scan_location_id", "location_01") ?: "location_01",
            scanServerHost = prefs.getString("scan_upload_host", "192.168.1.100") ?: "192.168.1.100",
            scanServerPort = prefs.getInt("scan_upload_port", 7861),
        )
    )
    val uiState: StateFlow<ScanUiState> = _uiState

    private var frameStreamJob: Job? = null
    private var elapsedTickJob: Job? = null
    private var lastFrameErrorMs = 0L
    private var recordingStartMs = 0L

    private var imuRecorder: ImuRecorder? = null
    private var currentOutDir: File? = null

    // ── gRPC live-stream controls ─────────────────────────────────────────────

    fun updateHost(host: String) = _uiState.update { it.copy(host = host) }

    fun updatePort(portStr: String) {
        portStr.toIntOrNull()?.let { port -> _uiState.update { it.copy(port = port) } }
    }

    fun updateLocationId(id: String) = _uiState.update { it.copy(locationId = id) }

    fun updateZoneLabel(label: String) = _uiState.update { it.copy(zoneLabel = label) }

    fun updateZoneRadius(radius: Float) = _uiState.update { it.copy(zoneRadius = radius) }

    fun startStreaming() {
        val state = _uiState.value
        grpcManager.connectScan(state.host, state.port)
        savePrefs(state.host, state.port, state.locationId)
        cameraManager.targetFps = 5
        frameStreamJob?.cancel()
        frameStreamJob = viewModelScope.launch(Dispatchers.IO) {
            cameraManager.frameFlow
                .catch { e -> setStatus("Flow error: ${e.message}") }
                .collect { jpegBytes ->
                    val stub = grpcManager.mapStub ?: return@collect
                    try {
                        val req = Tracking.ScanFrameRequest.newBuilder()
                            .setImageData(ByteString.copyFrom(jpegBytes))
                            .setLocationId(_uiState.value.locationId)
                            .build()
                        val resp = stub.scanFrame(req)
                        if (resp.success) {
                            _uiState.update {
                                it.copy(
                                    pointCount = resp.pointCount,
                                    cameraPosition = resp.cameraPositionList
                                )
                            }
                        }
                        lastFrameErrorMs = 0L
                    } catch (e: Exception) {
                        val now = System.currentTimeMillis()
                        if (now - lastFrameErrorMs > 3000) {
                            lastFrameErrorMs = now
                            setStatus("Frame error: ${e.message}")
                        }
                    }
                }
        }
        _uiState.update {
            it.copy(
                isStreaming = true,
                statusMessage = "Streaming to ${state.host}:${state.port}"
            )
        }
    }

    fun stopStreaming() {
        frameStreamJob?.cancel()
        frameStreamJob = null
        grpcManager.disconnectScan()
        cameraManager.targetFps = 10
        _uiState.update { it.copy(isStreaming = false, statusMessage = "Stopped") }
    }

    fun setZoneLabel() {
        val state = _uiState.value
        if (state.zoneLabel.isBlank()) {
            setStatus("Enter a zone name first")
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val req = Tracking.SetZoneLabelRequest.newBuilder()
                    .setLocationId(state.locationId)
                    .setLabel(state.zoneLabel)
                    .setRadius(state.zoneRadius)
                    .build()
                val resp = grpcManager.mapStub?.setZoneLabel(req)
                if (resp?.success == true) {
                    _uiState.update { it.copy(zoneLabel = "", statusMessage = "Zone saved: ${resp.message}") }
                } else {
                    setStatus("Zone failed: ${resp?.message}")
                }
            } catch (e: Exception) {
                setStatus("Zone error: ${e.message}")
            }
        }
    }

    fun exportMap() {
        val state = _uiState.value
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val req = Tracking.ExportScanMapRequest.newBuilder()
                    .setLocationId(state.locationId)
                    .build()
                val resp = grpcManager.mapStub?.exportScanMap(req)
                if (resp?.success == true) {
                    _uiState.update {
                        it.copy(
                            lastExportPath = resp.outputPath,
                            lastExportPointCount = resp.pointCount,
                            lastExportZoneCount = resp.zoneCount,
                            statusMessage = "Exported: ${resp.pointCount} pts, ${resp.zoneCount} zones"
                        )
                    }
                } else {
                    setStatus("Export failed")
                }
            } catch (e: Exception) {
                setStatus("Export error: ${e.message}")
            }
        }
    }

    // ── Offline recording ─────────────────────────────────────────────────────

    fun updateScanServerHost(host: String) {
        _uiState.update { it.copy(scanServerHost = host) }
        prefs.edit().putString("scan_upload_host", host).apply()
    }

    fun updateScanServerPort(portStr: String) {
        portStr.toIntOrNull()?.let { port ->
            _uiState.update { it.copy(scanServerPort = port) }
            prefs.edit().putInt("scan_upload_port", port).apply()
        }
    }

    fun startRecording() {
        val ctx = getApplication<Application>()
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val outDir = File(ctx.getExternalFilesDir("scans"), "scan_$ts")
        outDir.mkdirs()
        currentOutDir = outDir

        val recorder = ImuRecorder(ctx, outDir)
        imuRecorder = recorder
        if (recorder.isAvailable) {
            recorder.start(viewModelScope)
        }

        cameraManager.startRecording(outDir, fps = 5)

        recordingStartMs = System.currentTimeMillis()
        _uiState.update {
            it.copy(
                isRecording = true,
                datasetDir = null,
                imageCount = 0,
                imuFile = null,
                recordingElapsedMs = 0L,
                uploadStatus = null,
                statusMessage = "Recording…"
            )
        }

        elapsedTickJob = viewModelScope.launch {
            while (true) {
                delay(500)
                _uiState.update {
                    it.copy(recordingElapsedMs = System.currentTimeMillis() - recordingStartMs)
                }
            }
        }
    }

    fun stopRecording() {
        val imageCount = cameraManager.stopRecording()
        val imuFile = imuRecorder?.stop()
        elapsedTickJob?.cancel()
        _uiState.update {
            it.copy(
                isRecording = false,
                datasetDir = currentOutDir,
                imageCount = imageCount,
                imuFile = imuFile,
                statusMessage = "Recording saved: $imageCount frames"
            )
        }
    }

    fun uploadToScanServer() {
        val state = _uiState.value
        val datasetDir = state.datasetDir ?: return
        _uiState.update { it.copy(isUploading = true, uploadStatus = "Zipping…") }

        viewModelScope.launch(Dispatchers.IO) {
            try {
                val zipFile = File(datasetDir.parentFile, "${datasetDir.name}.zip")
                zipDirectory(datasetDir, zipFile)

                _uiState.update { it.copy(uploadStatus = "Uploading…") }

                val boundary = "----Boundary${System.currentTimeMillis()}"
                val url = URL("http://${state.scanServerHost}:${state.scanServerPort}/api/upload")
                val conn = url.openConnection() as HttpURLConnection
                conn.requestMethod = "POST"
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
                conn.connectTimeout = 10_000
                conn.readTimeout = 300_000

                conn.outputStream.buffered().use { out ->
                    out.write("--$boundary\r\n".toByteArray())
                    out.write("Content-Disposition: form-data; name=\"dataset\"; filename=\"dataset.zip\"\r\n".toByteArray())
                    out.write("Content-Type: application/zip\r\n\r\n".toByteArray())
                    zipFile.inputStream().use { it.copyTo(out) }
                    out.write("\r\n--$boundary--\r\n".toByteArray())
                }

                val code = conn.responseCode
                val body = runCatching { conn.inputStream.bufferedReader().readText() }.getOrElse { "" }
                _uiState.update {
                    it.copy(
                        isUploading = false,
                        uploadStatus = if (code == 200) "Uploaded! $body" else "Upload failed: HTTP $code"
                    )
                }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(isUploading = false, uploadStatus = "Upload error: ${e.message}")
                }
            }
        }
    }

    /** Zips [dir]'s contents (images/, imu.csv, camera.csv) at the zip root into [zipFile]. */
    private fun zipDirectory(dir: File, zipFile: File) {
        ZipOutputStream(BufferedOutputStream(FileOutputStream(zipFile))).use { zos ->
            fun addFile(file: File, entryName: String) {
                if (file.isDirectory) {
                    file.listFiles()?.sortedBy { it.name }?.forEach { addFile(it, "$entryName/${it.name}") }
                } else {
                    zos.putNextEntry(ZipEntry(entryName))
                    file.inputStream().use { it.copyTo(zos) }
                    zos.closeEntry()
                }
            }
            dir.listFiles()?.sortedBy { it.name }?.forEach { addFile(it, it.name) }
        }
    }

    // ── Misc ──────────────────────────────────────────────────────────────────

    fun clearError() = _uiState.update { it.copy(error = null) }

    private fun setStatus(msg: String) = _uiState.update { it.copy(statusMessage = msg) }

    private fun savePrefs(host: String, port: Int, locationId: String) {
        prefs.edit()
            .putString("scan_host", host)
            .putInt("scan_port", port)
            .putString("scan_location_id", locationId)
            .apply()
    }

    override fun onCleared() {
        super.onCleared()
        stopStreaming()
        imuRecorder?.stop()
        cameraManager.shutdown()
        grpcManager.disconnect()
    }
}
