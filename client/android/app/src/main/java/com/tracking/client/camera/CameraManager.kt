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

    var targetFps: Int = 10
    private var lastFrameTimeMs = 0L
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private var boundPreview: Preview? = null
    private var boundAnalysis: ImageAnalysis? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var boundLifecycleOwner: LifecycleOwner? = null

    // ── Dataset recording (images/ + camera.csv) ───────────────────────────────
    private val recordingLock = Any()
    private var recordingImagesDir: File? = null
    private var cameraCsvWriter: BufferedWriter? = null
    private var recordingFrameIndex = 0
    private var recordingFps: Int = 5
    private var lastRecordFrameTimeMs = 0L

    fun bind(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        boundLifecycleOwner = lifecycleOwner
        // FIT_CENTER: shows the full camera frame without cropping.
        // The overlay transform uses the same min-scale fit so boxes align exactly.
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider

            val toUnbind = listOfNotNull(boundPreview, boundAnalysis)
            if (toUnbind.isNotEmpty()) provider.unbind(*toUnbind.toTypedArray())

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build().also { it.setSurfaceProvider(previewView.surfaceProvider) }

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
            // in ONE bindToLifecycle() call, once. Calling bindToLifecycle() a
            // *second* time later replaces the entire bound use-case set — even
            // re-passing the same Preview instance was silently dropping its
            // rendered output, which is what caused the preview to go black
            // when a second bind happened.
            try {
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis,
                )
                boundPreview = preview
                boundAnalysis = imageAnalysis
            } catch (e: Exception) {
                Log.e("CameraManager", "Failed to bind camera use cases: ${e.message}")
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(context))
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
        val doStream = now - lastFrameTimeMs >= 1000L / targetFps
        val doRecord = isRecording && now - lastRecordFrameTimeMs >= 1000L / recordingFps
        if (!doStream && !doRecord) {
            imageProxy.close()
            return
        }
        try {
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

            if (doStream) {
                lastFrameTimeMs = now
                val jpeg = streamJpeg(rotated)
                if (!loggedOnce) {
                    Log.d("CameraManager", "JPEG sent: ${jpeg.size} bytes")
                    loggedOnce = true
                }
                _frameFlow.tryEmit(jpeg)
            }

            if (doRecord) {
                lastRecordFrameTimeMs = now
                // imageInfo.timestamp is the sensor's boot-time nanosecond clock —
                // the same clock domain as SensorEvent.timestamp (see ImuSensor.kt),
                // so camera.csv and imu.csv stay directly comparable.
                saveRecordingFrame(rotated, imageProxy.imageInfo.timestamp)
            }

            if (rotated !== bitmap) bitmap.recycle()
            rotated.recycle()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        imageProxy.close()
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
