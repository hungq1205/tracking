package com.tracking.client.live

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
 * Kotlin port of gt.py's correct_text_with_gemini() — OCR-error correction
 * (and, for non-English text, translation into English) via a plain REST
 * call to Gemini's generateContent endpoint. Deliberately NOT the
 * BidiGenerateContent WebSocket protocol GeminiLiveClient.kt uses (this is
 * a one-shot request/response, no session/turn state needed) and
 * deliberately NOT a Google SDK (same reasoning GeminiLiveClient.kt's own
 * comment gives for avoiding Firebase AI Logic: a hand-rolled OkHttp +
 * org.json call needs no extra project setup). Runs on ToolDispatcher's
 * correction queue — see that file — never on the tool-call path itself,
 * so a slow/failed correction never blocks OCR or reading.
 *
 * Deliberately does NOT swallow failures — gt.py's own predecessor of this
 * function originally caught every exception and silently fell back to the
 * uncorrected text, which made a missing/bad API key indistinguishable
 * from "nothing needed correcting" (a real, confirmed-confusing failure
 * mode during that harness's own development). Failures are logged with
 * the real exception here and then thrown; ToolDispatcher's correction
 * worker is the layer that catches and logs at the per-block level, so one
 * bad correction doesn't kill the whole worker loop.
 */
class GeminiCorrectionClient(private val apiKey: String, private val model: String = DEFAULT_MODEL) {
    private val client = OkHttpClient.Builder()
        .callTimeout(30, TimeUnit.SECONDS)
        .build()

    /** Returns the corrected (and, if [langCode] isn't English, translated)
     * text. Throws on any failure — see class doc. */
    suspend fun correct(text: String, langCode: String): String = withContext(Dispatchers.IO) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return@withContext trimmed

        val isEnglish = langCode.trim().lowercase() == "en"
        val systemInstruction = if (isEnglish) FIX_SYSTEM_PROMPT else translateSystemPrompt(langCode)
        val userPrompt = "Input:\n---\n$trimmed\n---"

        val requestBody = JSONObject().apply {
            put("systemInstruction", JSONObject().put("parts", JSONArray().put(JSONObject().put("text", systemInstruction))))
            put("contents", JSONArray().put(
                JSONObject().put("role", "user").put("parts", JSONArray().put(JSONObject().put("text", userPrompt)))
            ))
            put("generationConfig", JSONObject().apply {
                put("temperature", 0.1)
                put("maxOutputTokens", MAX_OUTPUT_TOKENS)
                put("responseMimeType", "text/plain")
            })
        }

        val request = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$apiKey")
            .post(requestBody.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val corrected = try {
            client.newCall(request).execute().use { resp ->
                val bodyStr = resp.body?.string() ?: ""
                if (!resp.isSuccessful) {
                    throw java.io.IOException("Gemini correction HTTP ${resp.code}: $bodyStr")
                }
                val json = JSONObject(bodyStr)
                val candidates = json.optJSONArray("candidates")
                val finishReason = candidates?.optJSONObject(0)?.optString("finishReason", "?")
                val parts = candidates?.optJSONObject(0)?.optJSONObject("content")?.optJSONArray("parts")
                val out = StringBuilder()
                if (parts != null) {
                    for (i in 0 until parts.length()) out.append(parts.optJSONObject(i)?.optString("text", "") ?: "")
                }
                android.util.Log.d(
                    TAG,
                    "[gemini-correction] model=$model lang=$langCode finishReason=$finishReason in=$trimmed out=$out",
                )
                out.toString().trim()
            }
        } catch (e: Exception) {
            android.util.Log.w(TAG, "[gemini-correction] FAILED model=$model lang=$langCode text=$trimmed", e)
            throw e
        }

        corrected.ifEmpty {
            android.util.Log.w(TAG, "[gemini-correction] WARNING: empty content despite a successful call — falling back to uncorrected text")
            trimmed
        }
    }

    companion object {
        private const val TAG = "GeminiCorrection"
        const val DEFAULT_MODEL = "gemini-3.1-flash-lite"
        // Generous ceiling so a long OCR page can't get silently truncated —
        // ported straight from gt.py's own GEMINI_MAX_OUTPUT_TOKENS, chosen
        // after a real incident there where a smaller budget on a
        // reasoning-heavy backend returned empty content on harder
        // (translate) blocks.
        private const val MAX_OUTPUT_TOKENS = 8192

        private const val FIX_SYSTEM_PROMPT = (
            "You are an OCR correction assistant.\n\n" +
            "Task:\n" +
            "1. Correct all OCR errors, typos, and broken formatting.\n" +
            "2. Restore punctuation, spacing, and paragraph breaks where appropriate.\n" +
            "3. Recover garbled words using context when reasonably certain.\n" +
            "4. Preserve the original language, wording, meaning, tone, dialogue, and paragraph structure.\n" +
            "5. Do not rewrite, summarize, censor, or translate the text.\n" +
            "6. If any word or phrase cannot be determined with reasonable confidence, write [unclear].\n\n" +
            "Output requirements:\n" +
            "- Output only the corrected text.\n" +
            "- Do not include explanations, notes, markdown, or the original text."
        )

        private const val TRANSLATE_SYSTEM_PROMPT_TEMPLATE = (
            "You are an OCR correction and translation assistant.\n\n" +
            "Task:\n" +
            "1. Correct all OCR errors, typos, and broken formatting.\n" +
            "2. Restore punctuation, spacing, and paragraph breaks where appropriate.\n" +
            "3. Recover garbled words using context when reasonably certain.\n" +
            "4. Translate the corrected text from %s into natural, fluent English.\n" +
            "5. Preserve the original meaning, tone, dialogue, names, and paragraph structure.\n" +
            "6. Do not summarize, omit, censor, or add information.\n" +
            "7. If any word or phrase cannot be determined with reasonable confidence, write [unclear].\n\n" +
            "Output requirements:\n" +
            "- Output only the final English text.\n" +
            "- Do not include explanations, notes, markdown, or the original text."
        )

        // ML Kit LanguageIdentification's ISO 639-1 codes -> a human-readable
        // name for the %s slot above ("translate from zh" reads far worse to
        // the model than "from Chinese") — falls back to the raw code for
        // anything not listed rather than guessing.
        private val LANG_NAMES = mapOf(
            "en" to "English", "vi" to "Vietnamese", "zh" to "Chinese", "ja" to "Japanese",
            "ko" to "Korean", "fr" to "French", "de" to "German", "es" to "Spanish",
            "pt" to "Portuguese", "ru" to "Russian", "it" to "Italian", "th" to "Thai",
            "ar" to "Arabic", "hi" to "Hindi", "id" to "Indonesian", "nl" to "Dutch",
            "pl" to "Polish", "tr" to "Turkish", "sv" to "Swedish", "uk" to "Ukrainian",
        )

        private fun translateSystemPrompt(langCode: String): String {
            val name = LANG_NAMES[langCode.trim().lowercase()] ?: langCode.ifBlank { "the source language" }
            return TRANSLATE_SYSTEM_PROMPT_TEMPLATE.format(name)
        }
    }
}
