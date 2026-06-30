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
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.io.ByteArrayOutputStream
import java.io.File
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

    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null

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

            try {
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis
                )
                boundPreview = preview
                boundAnalysis = imageAnalysis
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun unbind() {
        val toUnbind = listOfNotNull(boundPreview, boundAnalysis, videoCapture)
        if (toUnbind.isNotEmpty()) cameraProvider?.unbind(*toUnbind.toTypedArray())
        boundPreview = null
        boundAnalysis = null
        videoCapture = null
    }

    fun startRecording(outputFile: File, onFinalized: (File) -> Unit) {
        val provider = cameraProvider ?: return
        val lco = boundLifecycleOwner ?: return

        val recorder = Recorder.Builder()
            .setQualitySelector(
                QualitySelector.from(
                    Quality.HD,
                    androidx.camera.video.FallbackStrategy.lowerQualityOrHigherThan(Quality.SD)
                )
            )
            .build()
        val vc = VideoCapture.withOutput(recorder)
        videoCapture = vc

        try {
            provider.bindToLifecycle(lco, CameraSelector.DEFAULT_BACK_CAMERA, vc)
        } catch (e: Exception) {
            Log.e("CameraManager", "Failed to bind VideoCapture: ${e.message}")
            videoCapture = null
            return
        }

        outputFile.parentFile?.mkdirs()
        activeRecording = vc.output
            .prepareRecording(context, FileOutputOptions.Builder(outputFile).build())
            .start(ContextCompat.getMainExecutor(context)) { event: VideoRecordEvent ->
                if (event is VideoRecordEvent.Finalize) {
                    activeRecording = null
                    onFinalized(outputFile)
                }
            }

        Log.d("CameraManager", "Recording started → ${outputFile.absolutePath}")
    }

    fun stopRecording() {
        activeRecording?.stop()
        activeRecording = null
    }

    val isRecordingVideo: Boolean get() = activeRecording != null

    private var loggedOnce = false

    private fun processFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastFrameTimeMs >= 1000L / targetFps) {
            lastFrameTimeMs = now
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

                val jpeg = bitmapToJpeg(bitmap, imageProxy.imageInfo.rotationDegrees)

                if (!loggedOnce) {
                    Log.d("CameraManager", "JPEG sent: ${jpeg.size} bytes")
                    loggedOnce = true
                }

                _frameFlow.tryEmit(jpeg)
                bitmap.recycle()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        imageProxy.close()
    }

    private fun bitmapToJpeg(src: Bitmap, rotationDegrees: Int): ByteArray {
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)

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

        Log.d("CameraManager", "sending: ${scaled.width}x${scaled.height}")

        val baos = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 50, baos)

        if (scaled !== rotated) scaled.recycle()
        if (rotated !== src) rotated.recycle()

        return baos.toByteArray()
    }

    fun shutdown() {
        stopRecording()
        unbind()
        analysisExecutor.shutdown()
    }
}
