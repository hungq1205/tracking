package com.tracking.pixietest

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

/**
 * Minimal CameraX capture for this test app — ImageAnalysis only, no
 * Preview use case (nothing renders the raw camera feed on screen; see
 * PixiePositionView for what the UI shows instead).
 *
 * Every frame's raw Y-plane (luma, already what RotationTracker needs) is
 * handed to [onLumaFrame] with essentially no processing cost — no Bitmap
 * conversion, no JPEG round trip. The considerably more expensive
 * Bitmap+JPEG path (needed only for what gets sent to the server) runs at
 * its own much lower [jpegIntervalMs] cadence via [onFrame], instead of on
 * every analyzed frame like the original design — that redundant per-frame
 * JPEG encode (paid for even on frames nothing ever read) was part of what
 * made rotation tracking feel laggy.
 */
class SimpleCameraSource(private val context: Context) {

    /** (jpegBytes, frameTimestampNs) — imageInfo.timestamp, boot-time clock.
     * Fires at most once every [jpegIntervalMs]. */
    var onFrame: ((ByteArray, Long) -> Unit)? = null

    /** Raw Y-plane bytes straight off the sensor — fires on EVERY analyzed
     * frame. (luma, width, height, rowStride, rotationDegrees). */
    var onLumaFrame: ((ByteArray, Int, Int, Int, Int) -> Unit)? = null

    var jpegIntervalMs: Long = 300

    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null
    private var lastJpegMs = 0L

    fun start(lifecycleOwner: LifecycleOwner) {
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider

            // Bound the ANALYSIS stream to ~VGA — HIGHEST_AVAILABLE_STRATEGY
            // was handing every frame in at full sensor resolution (often
            // 12MP+ on a real phone), so the Mat build/rotate that runs
            // BEFORE the 320px downscale in RotationTracker was operating on
            // a huge image every call — the actual dominant cost behind the
            // reported >500ms, not the optical-flow tracking itself. Asking
            // CameraX for a small analysis size directly avoids paying for
            // any of that: a smaller buffer to copy, smaller Mat to
            // allocate/rotate, and no downscale-from-huge step at all.
            val resolutionSelector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor) { proxy -> processFrame(proxy) } }

            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        cameraProvider?.unbindAll()
        cameraProvider = null
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val yPlane = imageProxy.planes[0]
            val rowStride = yPlane.rowStride
            val buffer = yPlane.buffer
            val luma = ByteArray(buffer.remaining())
            buffer.get(luma)
            onLumaFrame?.invoke(luma, imageProxy.width, imageProxy.height, rowStride, imageProxy.imageInfo.rotationDegrees)

            val now = System.currentTimeMillis()
            if (onFrame != null && now - lastJpegMs >= jpegIntervalMs) {
                lastJpegMs = now
                val bitmap = imageProxy.toBitmap()
                val rotated = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
                if (rotated !== bitmap) bitmap.recycle()
                val jpeg = toJpeg(rotated)
                rotated.recycle()
                onFrame?.invoke(jpeg, imageProxy.imageInfo.timestamp)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            imageProxy.close()
        }
    }

    private fun rotateBitmap(src: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return src
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
    }

    private fun toJpeg(bitmap: Bitmap): ByteArray {
        val maxLongEdge = 640
        val longEdge = maxOf(bitmap.width, bitmap.height)
        val scaled = if (longEdge > maxLongEdge) {
            val scale = maxLongEdge.toFloat() / longEdge
            Bitmap.createScaledBitmap(
                bitmap, (bitmap.width * scale).toInt(), (bitmap.height * scale).toInt(), true,
            )
        } else bitmap
        val baos = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 70, baos)
        if (scaled !== bitmap) scaled.recycle()
        return baos.toByteArray()
    }
}
