package com.tracking.client.live

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.io.ByteArrayOutputStream

/**
 * Debug-only: draws each OCR line's box on a copy of the frame — green =
 * kept, red = dropped for rotation mismatch, orange = dropped as
 * short/small/isolated noise, magenta = dropped as a locally-blurry patch
 * — mirroring gt.py's annotate_frame(). Off by default (see
 * SettingsScreen's "Save debug OCR frames" toggle, wired through
 * ToolDispatcher's saveDebugFrame callback): writing every scanned camera
 * frame to device storage has real storage-growth and privacy cost in a
 * production assistive app, so this exists purely for diagnosing the
 * rotation/noise/blur filters on-device, not as a standing feature.
 */
fun annotateOcrFrame(
    jpeg: ByteArray,
    kept: List<OcrLine>,
    droppedRotation: List<OcrLine>,
    droppedNoise: List<OcrLine>,
    droppedBlur: List<OcrLine>,
): ByteArray {
    val decoded = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return jpeg
    val bitmap = decoded.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(bitmap)
    val paint = Paint().apply { style = Paint.Style.STROKE; strokeWidth = 4f }

    fun draw(lines: List<OcrLine>, color: Int) {
        paint.color = color
        for (line in lines) {
            canvas.drawRect(
                line.left.toFloat(), line.top.toFloat(),
                (line.left + line.width).toFloat(), (line.top + line.height).toFloat(),
                paint,
            )
        }
    }
    draw(droppedRotation, Color.RED)
    draw(droppedNoise, Color.rgb(255, 165, 0))
    draw(droppedBlur, Color.MAGENTA)
    draw(kept, Color.GREEN)

    val out = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
    return out.toByteArray()
}
