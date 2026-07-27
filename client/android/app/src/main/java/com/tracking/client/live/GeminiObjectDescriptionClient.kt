package com.tracking.client.live

import android.util.Base64
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Generates a SHORT appearance description for `remember_object()` via a
 * dedicated one-shot Gemini Flash-Lite vision call, instead of trusting
 * whatever free-form text Gemini Live's own conversational turn happened to
 * compose. Same rationale/pattern as GeminiCorrectionClient.kt (plain REST
 * call to generateContent, not the BidiGenerateContent WebSocket protocol,
 * not a Google SDK) — this is a different task (vision captioning, not text
 * correction), so it's its own small class rather than a new method bolted
 * onto GeminiCorrectionClient.
 *
 * Requested directly by the user: descriptions Gemini Live was writing on
 * its own for remember_object() (e.g. "A small, white, fluffy, short-haired
 * dog with pointy ears, sitting patiently.") were too long/scene-y for two
 * things this app actually uses the stored description for — a
 * GroundingDINO detection prompt (start_tracking's detectionPrompt, see
 * ToolDispatcher.toolStartTracking()) and a spoken answer to "what does X
 * look like?" — both want a terse "color + shape + entity type" phrase, not
 * a full sentence about pose/setting.
 */
class GeminiObjectDescriptionClient(private val apiKey: String, private val model: String = DEFAULT_MODEL) {
    private val client = OkHttpClient.Builder()
        .callTimeout(30, TimeUnit.SECONDS)
        .build()

    /** [label] is passed as context only (which object in a possibly-
     * cluttered frame to describe), never echoed back into the result.
     * Returns a short plain-text description; throws on failure — same
     * "never silently degrade" precedent GeminiCorrectionClient.kt already
     * established (a bad key/network error should be visibly different
     * from "nothing to describe"), left for the caller (ToolDispatcher) to
     * catch and fall back to whatever description Gemini Live itself
     * supplied. */
    suspend fun describe(imageJpeg: ByteArray, label: String): String = withContext(Dispatchers.IO) {
        val base64Image = Base64.encodeToString(imageJpeg, Base64.NO_WRAP)
        val userPrompt = "The object to describe is: \"$label\" (may be one of several things in view)."

        val requestBody = JSONObject().apply {
            put("systemInstruction", JSONObject().put("parts", JSONArray().put(JSONObject().put("text", SYSTEM_PROMPT))))
            put("contents", JSONArray().put(
                JSONObject().put("role", "user").put("parts", JSONArray().apply {
                    put(JSONObject().put("inlineData", JSONObject().put("mimeType", "image/jpeg").put("data", base64Image)))
                    put(JSONObject().put("text", userPrompt))
                })
            ))
            put("generationConfig", JSONObject().apply {
                put("temperature", 0.2)
                put("maxOutputTokens", MAX_OUTPUT_TOKENS)
                put("responseMimeType", "text/plain")
            })
        }

        val request = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$apiKey")
            .post(requestBody.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val description = try {
            client.newCall(request).execute().use { resp ->
                val bodyStr = resp.body?.string() ?: ""
                if (!resp.isSuccessful) {
                    throw java.io.IOException("Gemini object-description HTTP ${resp.code}: $bodyStr")
                }
                val json = JSONObject(bodyStr)
                val candidates = json.optJSONArray("candidates")
                val finishReason = candidates?.optJSONObject(0)?.optString("finishReason", "?")
                val parts = candidates?.optJSONObject(0)?.optJSONObject("content")?.optJSONArray("parts")
                val out = StringBuilder()
                if (parts != null) {
                    for (i in 0 until parts.length()) out.append(parts.optJSONObject(i)?.optString("text", "") ?: "")
                }
                android.util.Log.d(TAG, "[gemini-object-desc] model=$model label=$label finishReason=$finishReason out=$out")
                out.toString().trim()
            }
        } catch (e: Exception) {
            android.util.Log.w(TAG, "[gemini-object-desc] FAILED model=$model label=$label", e)
            throw e
        }

        if (description.isEmpty()) throw java.io.IOException("Gemini object-description returned empty content")
        description
    }

    companion object {
        private const val TAG = "GeminiObjectDesc"
        const val DEFAULT_MODEL = "gemini-3.1-flash-lite"
        private const val MAX_OUTPUT_TOKENS = 128 // a short phrase, not a paragraph

        private const val SYSTEM_PROMPT = (
            "You describe an object's appearance for a blind/low-vision assistive app, in as FEW " +
            "words as possible. This description is mainly used to compare objects to each other " +
            "via embeddings, not read verbatim to a user — so it only needs to be distinctive " +
            "enough for that, not exhaustive.\n\n" +
            "Output EXACTLY one short phrase (3-6 words), covering ONLY:\n" +
            "- object/entity type\n" +
            "- shape/size\n" +
            "- color(s)\n\n" +
            "Nothing else. Do NOT include material, pattern, logos/labels, pose/posture, " +
            "background/setting, actions, or any sentence-level narration. Do NOT use the object's " +
            "given name/label in the description. Two different objects of the same type/shape/color " +
            "describing near-identically is fine and expected.\n\n" +
            "Output only the phrase itself — no punctuation-only sentence wrapper, no explanations, " +
            "no markdown, no quotes."
        )
    }
}
