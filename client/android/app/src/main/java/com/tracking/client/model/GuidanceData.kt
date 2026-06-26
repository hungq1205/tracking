package com.tracking.client.model

data class GuidanceData(
    val instruction: String = "",
    val trackingStatus: String = "",
    val objectConfidence: Float = 0f,
    val objectBoxXyxy: List<Float> = emptyList(),
    val handBoxXyxy: List<Float> = emptyList(),
    val deltaX: Float = 0f,
    val deltaY: Float = 0f,
    val distancePx: Float = 0f
)
