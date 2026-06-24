package com.tracking.client.camera

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class CameraManager(private val context: Context) {

    private val _frameFlow = MutableSharedFlow<ByteArray>(extraBufferCapacity = 2)
    val frameFlow: SharedFlow<ByteArray> = _frameFlow

    var targetFps: Int = 10
    private var lastFrameTimeMs = 0L
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    fun bind(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastFrameTimeMs >= 1000L / targetFps) {
            lastFrameTimeMs = now
            try {
                val jpeg = imageProxy.toJpegByteArray()
                if (jpeg.isNotEmpty()) _frameFlow.tryEmit(jpeg)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        imageProxy.close()
    }

    private fun ImageProxy.toJpegByteArray(): ByteArray {
        val raw: Bitmap = toBitmap()
        val bitmap = if (raw.width == 640 && raw.height == 480) raw
                     else Bitmap.createScaledBitmap(raw, 640, 480, true).also { raw.recycle() }
        val baos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 75, baos)
        bitmap.recycle()
        return baos.toByteArray()
    }

    fun shutdown() {
        analysisExecutor.shutdown()
    }
}
