package com.tracking.client.live

import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/** One [analyze] call's result: the already-filtered, reading-order text
 * plus the raw kept/dropped line sets (for the optional debug frame
 * overlay — see DebugFrameStore.kt). */
data class OcrAnalysisResult(
    val text: String,
    val kept: List<OcrLine>,
    val droppedRotation: List<OcrLine>,
    val droppedNoise: List<OcrLine>,
    val droppedBlur: List<OcrLine>,
)

/**
 * Direct 3rd-party call to OCR.space (free hosted OCR) — bypasses the gRPC
 * server as a proxy, per the local/remote/3rd-party split (see CLAUDE.md's
 * "Client-Orchestrated Live Session" section). Replaces the earlier direct
 * call to the self-hosted paddle_ocr_server microservice. Requests
 * isOverlayRequired=true so per-word bounding boxes come back — needed for
 * OcrBlockFilters' rotation/noise filtering, not just flat text.
 */
class OcrClient(private val apiKey: String) {
    private val client = OkHttpClient.Builder()
        .callTimeout(30, TimeUnit.SECONDS)
        .build()

    private val effectiveKey get() = apiKey.ifBlank { "helloworld" }

    /** Fetches OCR.space's raw per-line boxes. Empty list on any failure —
     * callers should treat that the same as "no text detected". */
    private suspend fun readLines(jpeg: ByteArray): List<OcrLine> = withContext(Dispatchers.IO) {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("file", "frame.jpg", jpeg.toRequestBody("image/jpeg".toMediaType()))
            .addFormDataPart("language", "auto")
            .addFormDataPart("OCREngine", "2")
            .addFormDataPart("isOverlayRequired", "true")
            .addFormDataPart("detectOrientation", "true")
            .addFormDataPart("scale", "true")
            .build()
        val request = Request.Builder()
            .url("https://api.ocr.space/parse/image")
            .header("apikey", effectiveKey)
            .post(body)
            .build()
        try {
            client.newCall(request).execute().use { resp ->
                if (!resp.isSuccessful) return@withContext emptyList()
                val json = JSONObject(resp.body?.string() ?: "{}")
                if (json.optBoolean("IsErroredOnProcessing", false)) return@withContext emptyList()
                parseLines(json)
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun parseLines(json: JSONObject): List<OcrLine> {
        val lines = mutableListOf<OcrLine>()
        val results = json.optJSONArray("ParsedResults") ?: return lines
        for (r in 0 until results.length()) {
            val overlay = results.getJSONObject(r).optJSONObject("TextOverlay") ?: continue
            val overlayLines = overlay.optJSONArray("Lines") ?: continue
            for (l in 0 until overlayLines.length()) {
                val lineObj = overlayLines.getJSONObject(l)
                val text = lineObj.optString("LineText", "").trim()
                if (text.isEmpty()) continue
                val wordsArr = lineObj.optJSONArray("Words")
                val words = mutableListOf<OcrWord>()
                var left = Double.MAX_VALUE
                var top = Double.MAX_VALUE
                var right = -Double.MAX_VALUE
                var bottom = -Double.MAX_VALUE
                if (wordsArr != null) {
                    for (w in 0 until wordsArr.length()) {
                        val wo = wordsArr.getJSONObject(w)
                        val wl = wo.optDouble("Left", 0.0)
                        val wt = wo.optDouble("Top", 0.0)
                        val ww = wo.optDouble("Width", 0.0)
                        val wh = wo.optDouble("Height", 0.0)
                        words.add(OcrWord(wo.optString("WordText", ""), wl, wt, ww, wh))
                        left = minOf(left, wl)
                        top = minOf(top, wt)
                        right = maxOf(right, wl + ww)
                        bottom = maxOf(bottom, wt + wh)
                    }
                }
                if (words.isEmpty()) {
                    left = 0.0; top = 0.0; right = 0.0; bottom = 0.0
                }
                lines.add(OcrLine(text, left, top, right - left, bottom - top, words))
            }
        }
        return lines
    }

    /** Full pipeline: fetch -> drop rotation-mismatched lines -> drop
     * short/small/isolated noise lines -> drop locally-blurry lines (each
     * line's OWN cropped patch, independent of the whole-frame blur check
     * acquireSharpFrame() already did before this call — that one is blind
     * to a frame that's only PARTLY blurry, e.g. a page mid-turn with one
     * side still sharp) -> join survivors in reading order. */
    suspend fun analyze(jpeg: ByteArray): OcrAnalysisResult = withContext(Dispatchers.Default) {
        val lines = readLines(jpeg)
        if (lines.isEmpty()) return@withContext OcrAnalysisResult("", emptyList(), emptyList(), emptyList(), emptyList())
        val (afterRotation, droppedRotation) = OcrBlockFilters.filterLinesByRotation(lines)
        val (afterNoise, droppedNoise) = OcrBlockFilters.filterLinesByNoise(afterRotation)
        val bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
        val (kept, droppedBlur) = if (bitmap != null) {
            OcrBlockFilters.filterLinesByBlur(bitmap, afterNoise)
        } else {
            afterNoise to emptyList()
        }
        val text = kept.sortedWith(compareBy({ it.top }, { it.left })).joinToString("\n") { it.text }
        OcrAnalysisResult(text, kept, droppedRotation, droppedNoise, droppedBlur)
    }
}
