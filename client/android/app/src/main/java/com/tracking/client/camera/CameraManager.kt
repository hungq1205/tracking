package com.tracking.client.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.*
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.BufferedWriter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileWriter
import java.util.concurrent.Executors

class CameraManager(private val context: Context) {

    private val _frameFlow = MutableSharedFlow<ByteArray>(
        extraBufferCapacity = 16,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val frameFlow: SharedFlow<ByteArray> = _frameFlow

    /** Raw Y-plane (luma) straight off the sensor, for AngleTracker's
     * luma-direct ORB rotation tracking — see CLAUDE.md's "Pixie + Angle
     * modules" note. Additive: does not touch [frameFlow]/[streamJpeg] at
     * all. Only extracted while walking/guiding is active (gated on
     * [mappingMode] the same way the mapping-mode JPEG send-interval logic
     * already is below) since AngleTracker is the only consumer and it's
     * only fed during those two modes. Known, accepted limitation: this
     * reuses whichever resolution `bind()`'s ImageAnalysis is currently
     * bound at (HIGHEST_AVAILABLE_STRATEGY, i.e. often full sensor
     * resolution) — the same resolution the existing JPEG path already
     * pays for `toBitmap()` at. If this turns out to be a real per-frame
     * cost in practice (the exact lesson from this project's own
     * pixie_hrtf_app test harness — see its "Change strategy" notes on
     * ImageAnalysis resolution), the fix is bounding this stream the same
     * way that harness's SimpleCameraSource was fixed — not attempted here
     * since it would also affect every other consumer of this analysis
     * stream (recording, tracking JPEG frames), out of scope for this
     * change. */
    data class LumaFrame(val luma: ByteArray, val width: Int, val height: Int, val rowStride: Int, val rotationDegrees: Int)
    private val _lumaFlow = MutableSharedFlow<LumaFrame>(
        extraBufferCapacity = 4,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val lumaFlow: SharedFlow<LumaFrame> = _lumaFlow

    // Two independent client-side frame-selection policies, replacing the
    // old fixed targetFps.
    //
    // 1. Mapping-mode (guiding/walking/scanning — walking rejoined this
    //    bucket after its beacon moved back to a grid-planned route via
    //    MappingService/RTAB-Map, see CLAUDE.md's walking-mode local-map
    //    note) — no blur/clarity filtering (confirmed with the user —
    //    removed both here and server-side): whichever frame arrives once
    //    frameIntervalMs (guiding), walkingIntervalMs (walking), or
    //    scanIntervalMs (scanning) has elapsed since the last one sent is
    //    forwarded directly — see the mapping-mode branch in
    //    processFrame(). Each has its OWN interval — walking wants RTAB-Map
    //    pose updates fast enough for a responsive route (much tighter than
    //    guiding's default), scanning wants denser coverage for
    //    reconstruction/landmark tagging, guiding can afford to be the
    //    slowest since its beacon's own local-avoidance tick (not this
    //    stream) is what needs to be fast — so the three are independently
    //    tunable rather than sharing one slider. mappingMode tracks which
    //    one is in effect (also used to reset the send-gate on any submode
    //    change, so switching mid-interval doesn't inherit a stale gate
    //    from a different window size).
    // 2. Everything else (tracking/reading/Q&A/idle) — still blur-aware: a
    //    small rolling buffer of the last recentBufferMs of frames;
    //    consumers that need "the current frame" pull clearestRecentFrame()
    //    on demand, and continuous per-frame consumers (hand tracking,
    //    local ORB tracking) still get a steady trickle via frameFlow,
    //    emitted at most once per recentBufferMs from whatever's currently
    //    sharpest in the buffer.
    var frameIntervalMs: Int = 1000       // SettingsScreen slider: 100..5000 — guiding only
    var walkingIntervalMs: Int = 350      // matches ToolDispatcher's avoidanceIntervalMs default — walking only
    var scanIntervalMs: Int = 200         // SettingsScreen slider: 50..500, 50ms steps — scanning only (5fps default)
    var recentBufferMs: Int = 100         // SettingsScreen slider: 0..1000, 50ms steps
    /** "", "guiding", "walking", or "scanning" — set every processed frame
     * by MainViewModel.kt from sessionState.mode. Empty means non-mapping
     * (tracking/reading/Q&A/idle), which uses the rolling buffer instead. */
    @Volatile var mappingMode: String = ""

    private data class TimedFrame(val jpeg: ByteArray, val sharpness: Double, val atMs: Long)

    private var lastMappingSentMs = -1L
    private var activeMappingSubmode: String? = null  // null = not mapping

    private val recentBuffer = ArrayDeque<TimedFrame>()
    private var lastRecentEmitMs = 0L

    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private var boundPreview: Preview? = null
    private var boundAnalysis: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var boundLifecycleOwner: LifecycleOwner? = null
    // bind()'s ProcessCameraProvider.getInstance() future is async — a
    // PreviewView can attach before boundPreview exists yet (e.g. the
    // Activity's Compose UI creating a PreviewView while the Service is
    // still completing its initial bind()). Remembered here and (re)applied
    // once bind()'s listener actually sets boundPreview.
    private var pendingPreviewView: PreviewView? = null

    // ── Dataset recording (images/ + camera.csv) ───────────────────────────────
    private val recordingLock = Any()
    private var recordingImagesDir: File? = null
    private var cameraCsvWriter: BufferedWriter? = null
    private var recordingFrameIndex = 0
    private var recordingFps: Int = 5
    private var lastRecordFrameTimeMs = 0L

    /**
     * Binds Preview + ImageAnalysis ONCE against [lifecycleOwner] — now always
     * the hosting LiveAssistantService (a LifecycleService that outlives the
     * Activity), not the Activity itself, so frame capture keeps running
     * while the app is backgrounded/the task is swiped away. No PreviewView is
     * passed here any more: the Preview use case is built with no surface
     * provider attached yet — see [attachPreviewSurface]/[detachPreviewSurface]
     * for how the Activity-side on-screen preview plugs in and out
     * independently, without ever calling bindToLifecycle() a second time
     * (which replaces the whole bound use-case set — see the note below).
     */
    fun bind(lifecycleOwner: LifecycleOwner) {
        boundLifecycleOwner = lifecycleOwner

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider

            val toUnbind = listOfNotNull(boundPreview, boundAnalysis)
            if (toUnbind.isNotEmpty()) provider.unbind(*toUnbind.toTypedArray())

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()

            val resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                .build()

            val imageAnalysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            // Bind everything the app will ever need — Preview + ImageAnalysis —
            // in ONE bindToLifecycle() call, once, against the Service's own
            // long-lived lifecycle. Calling bindToLifecycle() a *second* time
            // later replaces the entire bound use-case set — even re-passing
            // the same Preview instance was silently dropping its rendered
            // output, which is what caused the preview to go black when a
            // second bind happened. So this is now called exactly once for
            // the Service's whole lifetime; the on-screen preview attaches/
            // detaches via setSurfaceProvider() instead of ever rebinding.
            try {
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis,
                )
                boundPreview = preview
                boundAnalysis = imageAnalysis
                pendingPreviewView?.let { attachPreviewSurface(it) }
            } catch (e: Exception) {
                Log.e("CameraManager", "Failed to bind camera use cases: ${e.message}")
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /** Plugs the given on-screen [previewView] into the already-bound Preview
     * use case — safe to call repeatedly (e.g. every time the Activity's UI
     * becomes visible) without ever re-invoking bindToLifecycle(). If
     * bind()'s async camera-provider setup hasn't completed yet, remembers
     * [previewView] and applies it once it does. */
    fun attachPreviewSurface(previewView: PreviewView) {
        // FIT_CENTER: shows the full camera frame without cropping.
        // The overlay transform uses the same min-scale fit so boxes align exactly.
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        pendingPreviewView = previewView
        boundPreview?.setSurfaceProvider(previewView.surfaceProvider)
    }

    /** Detaches the on-screen preview surface (e.g. the Activity going into
     * the background) without affecting ImageAnalysis/frame capture, which
     * keeps running against the Service's lifecycle regardless. */
    fun detachPreviewSurface() {
        pendingPreviewView = null
        boundPreview?.setSurfaceProvider(null)
    }

    fun unbind() {
        val toUnbind = listOfNotNull(boundPreview, boundAnalysis)
        if (toUnbind.isNotEmpty()) cameraProvider?.unbind(*toUnbind.toTypedArray())
        boundPreview = null
        boundAnalysis = null
    }

    /**
     * Start dumping analyzed frames to `outputDir/images/NNN.jpg` at [fps],
     * alongside `outputDir/camera.csv` (header: timestamp_ns,filename) —
     * the dataset layout the scan server ingests.
     */
    fun startRecording(outputDir: File, fps: Int = 5) {
        val imagesDir = File(outputDir, "images").also { it.mkdirs() }
        val writer = BufferedWriter(FileWriter(File(outputDir, "camera.csv")))
        writer.write("timestamp_ns,filename\n")
        synchronized(recordingLock) {
            recordingImagesDir = imagesDir
            cameraCsvWriter = writer
            recordingFrameIndex = 0
            recordingFps = fps
        }
        Log.d("CameraManager", "Recording started → ${imagesDir.absolutePath}")
    }

    /** Stops dataset recording and returns the number of frames saved. */
    fun stopRecording(): Int {
        return synchronized(recordingLock) {
            val count = recordingFrameIndex
            cameraCsvWriter?.flush()
            cameraCsvWriter?.close()
            cameraCsvWriter = null
            recordingImagesDir = null
            count
        }
    }

    val isRecording: Boolean get() = synchronized(recordingLock) { recordingImagesDir != null }

    private var loggedOnce = false

    private fun processFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        val doRecord = isRecording && now - lastRecordFrameTimeMs >= 1000L / recordingFps
        try {
            if (mappingMode == "guiding" || mappingMode == "walking") {
                val yPlane = imageProxy.planes[0]
                val rowStride = yPlane.rowStride
                val buffer = yPlane.buffer
                val luma = ByteArray(buffer.remaining())
                buffer.get(luma)
                _lumaFlow.tryEmit(
                    LumaFrame(luma, imageProxy.width, imageProxy.height, rowStride, imageProxy.imageInfo.rotationDegrees)
                )
            }

            val crop = imageProxy.cropRect
            val bitmap = imageProxy.toBitmap()

            if (!loggedOnce) {
                Log.d("CameraManager",
                    "ImageProxy buffer: ${imageProxy.width}x${imageProxy.height} " +
                    "cropRect: ${crop.width()}x${crop.height()} @(${crop.left},${crop.top}) " +
                    "rotation: ${imageProxy.imageInfo.rotationDegrees} " +
                    "format: ${imageProxy.format}")
                Log.d("CameraManager", "toBitmap: ${bitmap.width}x${bitmap.height}")
            }

            val rotated = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
            // The pre-rotation bitmap is never needed again past this point,
            // regardless of what happens to `rotated` below (which is either
            // this same object, when rotation was 0, or a freshly-created one).
            if (rotated !== bitmap) bitmap.recycle()

            if (doRecord) {
                lastRecordFrameTimeMs = now
                // imageInfo.timestamp is the sensor's boot-time nanosecond clock —
                // the same clock domain as SensorEvent.timestamp (see ImuSensor.kt),
                // so camera.csv and imu.csv stay directly comparable.
                saveRecordingFrame(rotated, imageProxy.imageInfo.timestamp)
            }

            // Bitmap's job ends here — everything downstream (window
            // accumulation, the rolling buffer, both send paths) works off
            // the encoded JPEG + its sharpness score, jpeg/sharpness are
            // computed once and shared by both selection policies below.
            val sharpness = computeSharpness(rotated)
            val jpeg = streamJpeg(rotated)
            rotated.recycle()
            if (!loggedOnce) {
                Log.d("CameraManager", "First frame encoded: ${jpeg.size} bytes (sharpness=$sharpness)")
                loggedOnce = true
            }
            val tf = TimedFrame(jpeg, sharpness, now)

            // Always kept fresh regardless of mode — the pull-based source
            // for clearestRecentFrame() (OCR/run_detection/tracking-init
            // "give me the current frame" tool calls).
            recentBuffer.addLast(tf)
            while (recentBuffer.isNotEmpty() && now - recentBuffer.first().atMs > recentBufferMs) {
                recentBuffer.removeFirst()
            }

            val submode = mappingMode.ifEmpty { null }
            if (submode != activeMappingSubmode) {
                // Entering/leaving mapping mode, OR switching between
                // walking/guiding/scanning — reset the send-gate so a
                // stale timestamp from a different mode/interval doesn't
                // suppress this mode's first send.
                lastMappingSentMs = -1L
                activeMappingSubmode = submode
            }

            if (submode != null) {
                // No blur/clarity filtering here — confirmed with the user
                // (removed both client- and server-side). Whichever frame
                // happens to arrive once intervalMs has elapsed since the
                // last send is forwarded directly, no window/candidate
                // comparison at all.
                val intervalMs = when (submode) {
                    "scanning" -> scanIntervalMs
                    "walking" -> walkingIntervalMs
                    else -> frameIntervalMs
                }
                if (lastMappingSentMs < 0 || now - lastMappingSentMs >= intervalMs) {
                    _frameFlow.tryEmit(tf.jpeg)
                    lastMappingSentMs = now
                }
            } else {
                handleRecentEmit(now)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        imageProxy.close()
    }

    /** Non-mapping modes: drive continuous per-frame consumers (hand
     * tracking, local ORB tracking, UI overlay) off whatever's currently
     * sharpest in the rolling recentBufferMs buffer, at most once per
     * recentBufferMs — a real-time counterpart to clearestRecentFrame()'s
     * on-demand pull for one-shot tool calls. */
    private fun handleRecentEmit(now: Long) {
        val tickMs = recentBufferMs.coerceAtLeast(1)
        if (now - lastRecentEmitMs < tickMs) return
        recentBuffer.maxByOrNull { it.sharpness }?.let { best ->
            _frameFlow.tryEmit(best.jpeg)
            lastRecentEmitMs = now
        }
    }

    /** Pull-based accessor: the sharpest frame currently sitting in the
     * rolling recentBufferMs window — for tool calls (OCR, run_detection,
     * tracking init retries) that want a fresh best-quality frame on demand
     * rather than waiting on frameFlow's periodic emission. */
    fun clearestRecentFrame(): ByteArray? = recentBuffer.maxByOrNull { it.sharpness }?.jpeg

    /** Same pull as [clearestRecentFrame], but also exposes the sharpness
     * score — for ToolDispatcher's reading-mode blur skip/retry
     * (acquireSharpFrame()), which needs to compare it against a threshold
     * before deciding whether to re-sample instead of OCR'ing a blurry
     * frame outright. */
    fun clearestRecentFrameWithSharpness(): Pair<ByteArray, Double>? =
        recentBuffer.maxByOrNull { it.sharpness }?.let { it.jpeg to it.sharpness }

    /** Variance of the Laplacian on a downscaled grayscale copy — same blur
     * metric orb_novelty_gate.py's _sharpness_score() uses server-side
     * (cv2.Laplacian(gray, CV_64F).var()), computed via OpenCV's
     * meanStdDev (variance = stddev²) to avoid a second pass over the
     * pixels. Downscaled first since this now runs on every incoming
     * frame, not just ones about to be sent — blur detection doesn't need
     * full resolution to be accurate. */
    private fun computeSharpness(bitmap: Bitmap): Double {
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        val small = Mat()
        val longEdge = maxOf(gray.width(), gray.height()).coerceAtLeast(1)
        if (longEdge > 320) {
            val scale = 320.0 / longEdge
            Imgproc.resize(gray, small, Size(gray.width() * scale, gray.height() * scale))
        } else {
            gray.copyTo(small)
        }
        val laplacian = Mat()
        Imgproc.Laplacian(small, laplacian, CvType.CV_64F)
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(laplacian, mean, stddev)
        val sd = stddev.toArray().getOrElse(0) { 0.0 }
        rgba.release(); gray.release(); small.release(); laplacian.release()
        mean.release(); stddev.release()
        return sd * sd
    }

    private fun rotateBitmap(src: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return src
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
    }

    private fun streamJpeg(rotated: Bitmap): ByteArray {
        // Downscale to max 640px on the long edge, preserving full frame (no crop)
        val maxLongEdge = 640
        val longEdge = maxOf(rotated.width, rotated.height)
        val scaled = if (longEdge > maxLongEdge) {
            val scale = maxLongEdge.toFloat() / longEdge
            Bitmap.createScaledBitmap(
                rotated,
                (rotated.width * scale).toInt(),
                (rotated.height * scale).toInt(),
                true
            )
        } else rotated

        val baos = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 50, baos)

        if (scaled !== rotated) scaled.recycle()

        return baos.toByteArray()
    }

    private fun saveRecordingFrame(rotated: Bitmap, timestampNs: Long) {
        synchronized(recordingLock) {
            val dir = recordingImagesDir ?: return
            val writer = cameraCsvWriter ?: return
            val filename = "%09d.jpg".format(recordingFrameIndex)
            val baos = ByteArrayOutputStream()
            rotated.compress(Bitmap.CompressFormat.JPEG, 90, baos)
            File(dir, filename).writeBytes(baos.toByteArray())
            writer.write("$timestampNs,$filename\n")
            recordingFrameIndex++
        }
    }

    fun shutdown() {
        stopRecording()
        unbind()
        analysisExecutor.shutdown()
    }
}
