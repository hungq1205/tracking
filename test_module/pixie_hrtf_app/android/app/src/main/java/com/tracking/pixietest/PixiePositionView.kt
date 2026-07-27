package com.tracking.pixietest

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * Replaces the camera preview in this test app's UI — there's nothing
 * useful to look at in the raw feed (it only ever drives ORB tracking, see
 * SimpleCameraSource), so instead this draws two related but different
 * things side by side:
 *
 * - Left circle ("ANCHOR"): the pixie's bearing relative to the anchor's
 *   original forward direction — does NOT rotate as you turn your head,
 *   since it's a fixed world-frame bearing (PixieMotion's thetaDeg,
 *   running locally regardless of server connection — see MainActivity).
 *   Dot color encodes the pixie's elevation (warm/high = above, cool/low
 *   = below).
 * - Right circle ("YOUR DRIFT"): how far YOUR OWN current facing
 *   direction has drifted from the anchor's original forward direction —
 *   HeadingEstimator.currentHeadingDeg() directly, nothing pixie-related.
 *   Always shown regardless of [muted] (there's no "pixie alignment"
 *   concept for this one — it's just your own orientation) in a fixed
 *   neutral color.
 *
 * Both share the same "straight up = 0 deg, clockwise = +" convention as
 * HrtfBeacon's azimuth.
 */
class PixiePositionView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var anchorAzimuthDeg = 0f
    private var elevationDeg = 0f
    private var userDriftDeg = 0f
    private var muted = true

    private val circlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.DKGRAY
    }
    private val forwardTickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.LTGRAY
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val centerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.GRAY
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.DKGRAY
        textSize = 32f
        textAlign = Paint.Align.CENTER
    }

    private val userDriftColor = Color.rgb(90, 160, 255)  // fixed neutral blue — not elevation-coded

    fun update(anchorAzimuthDeg: Float, elevationDeg: Float, userDriftDeg: Float, muted: Boolean) {
        this.anchorAzimuthDeg = anchorAzimuthDeg
        this.elevationDeg = elevationDeg
        this.userDriftDeg = userDriftDeg
        this.muted = muted
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val halfW = width / 2f
        drawCompass(canvas, halfW / 2f, height / 2f, halfW, "ANCHOR", anchorAzimuthDeg, elevationColor(elevationDeg), muted)
        drawCompass(canvas, halfW + halfW / 2f, height / 2f, halfW, "YOUR DRIFT", userDriftDeg, userDriftColor, muted = false)
    }

    private fun drawCompass(
        canvas: Canvas, cx: Float, cyIn: Float, columnWidth: Float, label: String,
        azimuthDeg: Float, color: Int, muted: Boolean,
    ) {
        val cy = cyIn + 20f
        val r = min(columnWidth, height.toFloat()) * 0.38f

        canvas.drawText(label, cx, cy - r - 24f, labelPaint)
        canvas.drawCircle(cx, cy, r, circlePaint)
        canvas.drawLine(cx, cy - r, cx, cy - r * 1.15f, forwardTickPaint)  // "ahead" tick
        canvas.drawCircle(cx, cy, 8f, centerPaint)

        if (!muted) {
            val theta = Math.toRadians(azimuthDeg.toDouble())
            val px = cx + r * sin(theta).toFloat()
            val py = cy - r * cos(theta).toFloat()
            linePaint.color = color
            dotPaint.color = color
            canvas.drawLine(cx, cy, px, py, linePaint)
            canvas.drawCircle(px, py, 16f, dotPaint)
        }
    }

    private fun elevationColor(elevationDeg: Float): Int {
        // Warm (above) <-> cool (below), 0 deg = neutral white-ish.
        val frac = (elevationDeg / 90f).coerceIn(-1f, 1f)
        return if (frac >= 0) {
            Color.rgb(255, (255 - frac * 120).toInt().coerceIn(0, 255), (255 - frac * 200).toInt().coerceIn(0, 255))
        } else {
            Color.rgb((255 + frac * 200).toInt().coerceIn(0, 255), (255 + frac * 120).toInt().coerceIn(0, 255), 255)
        }
    }
}
