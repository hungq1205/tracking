package com.tracking.client.model

data class ObjectTrack(
    // Backend tracking state
    val boxXyxy: FloatArray = floatArrayOf(0f, 0f, 0f, 0f),
    val centerX: Float = 0f,
    val centerY: Float = 0f,
    val confidence: Float = 0f,
    val visible: Boolean = false,
    val status: String = "",
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    // UI guidance fields
    val instruction: String = "",
    val objectBoxXyxy: List<Float> = emptyList(),
    val handBoxXyxy: List<Float> = emptyList(),
    val deltaX: Float = 0f,
    val deltaY: Float = 0f,
    val distancePx: Float = 0f,
    val matchedKeypointsX: List<Float> = emptyList(),
    val matchedKeypointsY: List<Float> = emptyList(),
    val handLandmarksX: List<List<Float>> = emptyList(),
    val handLandmarksY: List<List<Float>> = emptyList(),
)
