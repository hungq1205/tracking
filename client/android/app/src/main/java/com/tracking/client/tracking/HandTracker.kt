package com.tracking.client.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext

data class HandLandmarks(
    /** Normalized [0,1] landmark positions (x, y, z) per hand. */
    val hands: List<List<Triple<Float, Float, Float>>>,
)

class HandTracker(context: Context) {

    private val _resultFlow = MutableSharedFlow<HandLandmarks>(
        extraBufferCapacity = 8,
        onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val resultFlow: SharedFlow<HandLandmarks> = _resultFlow

    private val landmarker: HandLandmarker? = try {
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(MODEL_ASSET)
                    .build()
            )
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setRunningMode(RunningMode.IMAGE)
            .build()
        HandLandmarker.createFromOptions(context, options)
    } catch (e: Exception) {
        Log.e(TAG, "Failed to initialize HandLandmarker: ${e.message}")
        null
    }

    /** Run hand detection on a JPEG frame. Call from a background thread or IO dispatcher. */
    suspend fun detect(jpeg: ByteArray): HandLandmarks? = withContext(Dispatchers.Default) {
        val lm = landmarker ?: return@withContext null
        val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
            ?.copy(Bitmap.Config.ARGB_8888, false)
            ?: return@withContext null

        val mpImage = BitmapImageBuilder(bitmap).build()
        val result: HandLandmarkerResult = try {
            lm.detect(mpImage)
        } catch (e: Exception) {
            Log.w(TAG, "detect error: ${e.message}")
            bitmap.recycle()
            return@withContext null
        }
        bitmap.recycle()

        val hands = result.landmarks().map { hand ->
            hand.map { lp -> Triple(lp.x(), lp.y(), lp.z()) }
        }
        val landmarks = HandLandmarks(hands)
        _resultFlow.tryEmit(landmarks)
        landmarks
    }

    fun close() {
        landmarker?.close()
    }

    companion object {
        private const val TAG = "HandTracker"
        private const val MODEL_ASSET = "hand_landmarker.task"
    }
}
