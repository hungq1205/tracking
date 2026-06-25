package com.tracking.client.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.*
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

            // 1. Configure Preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build().also { it.setSurfaceProvider(previewView.surfaceProvider) }

            // 2. Configure Analysis - REMOVED RGBA_8888 FORCE
            // By NOT specifying a format, we use YUV_420_888, 
            // which prevents the rowStride hardware padding issue.
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
                // toBitmap handles the YUV-to-RGBA conversion internally,
                // calculating row strides correctly.
                val bitmap = imageProxy.toBitmap()
                
                // Rotate and convert to JPEG for the network
                val jpeg = bitmapToByteArray(bitmap, imageProxy.imageInfo.rotationDegrees)
                _frameFlow.tryEmit(jpeg)
                
                bitmap.recycle()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        imageProxy.close()
    }

    private fun bitmapToByteArray(src: Bitmap, rotationDegrees: Int): ByteArray {
        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, true)
        
        // Scale down to a manageable size (e.g., 640x480)
        val scaled = Bitmap.createScaledBitmap(rotated, 640, 480, true)
        
        val baos = ByteArrayOutputStream()
        scaled.compress(Bitmap.CompressFormat.JPEG, 75, baos)
        
        if (rotated !== src) rotated.recycle()
        scaled.recycle()
        
        return baos.toByteArray()
    }

    fun shutdown() {
        analysisExecutor.shutdown()
    }
}