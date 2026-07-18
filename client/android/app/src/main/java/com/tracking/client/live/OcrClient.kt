package com.tracking.client.live

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Direct 3rd-party call to the OCR microservice (paddle_ocr_server) —
 * bypasses the gRPC server as a proxy, per the local/remote/3rd-party split
 * (see CLAUDE.md's "Client-Orchestrated Live Session" section: things that
 * were "calling an outside API via our server" become direct client calls).
 * Mirrors server/tools/ocr.py's DocLayoutRapidOCRTool.read_text exactly:
 * POST multipart JPEG to /ocr, join the returned blocks' text with spaces
 * (the server-side `direction` param was already a no-op there — reading
 * order comes from the OCR service itself, not client-side reordering).
 */
class OcrClient(private val baseUrl: String) {
    private val client = OkHttpClient.Builder()
        .callTimeout(30, TimeUnit.SECONDS)
        .build()

    suspend fun readText(jpeg: ByteArray): String = withContext(Dispatchers.IO) {
        val body = MultipartBody.Builder().setType(MultipartBody.FORM)
            .addFormDataPart("image", "frame.jpg", jpeg.toRequestBody("image/jpeg".toMediaType()))
            .build()
        val request = Request.Builder().url("$baseUrl/ocr").post(body).build()
        try {
            client.newCall(request).execute().use { resp ->
                if (!resp.isSuccessful) return@withContext ""
                val json = JSONObject(resp.body?.string() ?: "{}")
                val blocks = json.optJSONArray("blocks") ?: return@withContext ""
                (0 until blocks.length()).joinToString(" ") { i -> blocks.getJSONObject(i).optString("text", "") }
            }
        } catch (e: Exception) {
            ""
        }
    }
}
