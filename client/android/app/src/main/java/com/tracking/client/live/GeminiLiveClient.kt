package com.tracking.client.live

import android.util.Base64
import android.util.Log
import kotlinx.coroutines.channels.ProducerScope
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Direct Kotlin client for the Gemini Live API's raw WebSocket protocol
 * (BidiGenerateContent) — replaces the old server-side google-genai Python
 * session (server/live_session.py's LiveAPISession) now that orchestration
 * runs on-device. No official Kotlin SDK is used here deliberately: Firebase
 * AI Logic's LiveSession wraps this same protocol but requires a Firebase
 * project + google-services.json, a heavier integration than "embed an API
 * key and call Gemini directly" (the chosen tradeoff — see CLAUDE.md's
 * "Client-Orchestrated Live Session" section).
 *
 * Wire protocol per https://ai.google.dev/api/live — untested against a
 * live backend from this environment (no network path to verify), built
 * strictly from the documented JSON schema. Treat as the first thing to
 * check if a real device fails to connect.
 */
data class FunctionCallEvent(val id: String, val name: String, val args: JSONObject)

sealed class LiveServerEvent {
    data class Audio(val pcm: ByteArray) : LiveServerEvent()
    data class ToolCall(val calls: List<FunctionCallEvent>) : LiveServerEvent()
    object TurnComplete : LiveServerEvent()
    object Interrupted : LiveServerEvent()
    object SetupComplete : LiveServerEvent()
    data class Error(val message: String) : LiveServerEvent()
    object Closed : LiveServerEvent()
}

class GeminiLiveClient(
    private val apiKey: String,
    private val model: String = "gemini-3.1-flash-live-preview",
) {
    private var webSocket: WebSocket? = null
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS) // persistent stream, no read timeout
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    /**
     * Opens the WebSocket, sends the setup message, and emits every server
     * event as a cold Flow — collecting it drives the connection's lifetime;
     * cancelling the collector closes the socket (awaitClose).
     */
    fun events(systemPrompt: String, toolDeclarations: JSONArray): Flow<LiveServerEvent> = callbackFlow {
        val scope: ProducerScope<LiveServerEvent> = this
        val url = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=$apiKey"
        val listener = object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                val setup = JSONObject().put("setup", JSONObject().apply {
                    put("model", "models/$model")
                    put("generationConfig", JSONObject().apply {
                        put("responseModalities", JSONArray().put("AUDIO"))
                    })
                    put(
                        "systemInstruction",
                        JSONObject().put("parts", JSONArray().put(JSONObject().put("text", systemPrompt)))
                    )
                    put("tools", JSONArray().put(JSONObject().put("functionDeclarations", toolDeclarations)))
                })
                ws.send(setup.toString())
            }

            override fun onMessage(ws: WebSocket, text: String) = handleServerMessage(text, scope)
            override fun onMessage(ws: WebSocket, bytes: ByteString) = handleServerMessage(bytes.utf8(), scope)

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure: ${t.message}", t)
                scope.trySend(LiveServerEvent.Error(t.message ?: "WebSocket failure"))
                scope.close(t)
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                scope.trySend(LiveServerEvent.Closed)
                scope.close()
            }
        }
        webSocket = client.newWebSocket(Request.Builder().url(url).build(), listener)
        awaitClose {
            webSocket?.close(1000, "session ended")
            webSocket = null
        }
    }

    private fun handleServerMessage(text: String, scope: ProducerScope<LiveServerEvent>) {
        val obj = try { JSONObject(text) } catch (e: Exception) {
            Log.w(TAG, "Unparseable server message: $text")
            return
        }
        if (obj.has("setupComplete")) {
            scope.trySend(LiveServerEvent.SetupComplete)
        }
        obj.optJSONObject("serverContent")?.let { sc ->
            sc.optJSONObject("modelTurn")?.optJSONArray("parts")?.let { parts ->
                for (i in 0 until parts.length()) {
                    val inlineData = parts.optJSONObject(i)?.optJSONObject("inlineData") ?: continue
                    val mime = inlineData.optString("mimeType", "")
                    if (!mime.startsWith("audio/")) continue
                    val pcm = Base64.decode(inlineData.optString("data", ""), Base64.DEFAULT)
                    scope.trySend(LiveServerEvent.Audio(pcm))
                }
            }
            if (sc.optBoolean("interrupted", false)) scope.trySend(LiveServerEvent.Interrupted)
            if (sc.optBoolean("turnComplete", false)) scope.trySend(LiveServerEvent.TurnComplete)
        }
        obj.optJSONObject("toolCall")?.optJSONArray("functionCalls")?.let { calls ->
            val events = (0 until calls.length()).mapNotNull { i ->
                val c = calls.optJSONObject(i) ?: return@mapNotNull null
                FunctionCallEvent(
                    id = c.optString("id", ""),
                    name = c.optString("name", ""),
                    args = c.optJSONObject("args") ?: JSONObject(),
                )
            }
            if (events.isNotEmpty()) scope.trySend(LiveServerEvent.ToolCall(events))
        }
    }

    fun sendAudioChunk(pcm: ByteArray) {
        send(JSONObject().put("realtimeInput", JSONObject().put(
            "audio", JSONObject().put("mimeType", "audio/pcm;rate=16000")
                .put("data", Base64.encodeToString(pcm, Base64.NO_WRAP))
        )))
    }

    fun sendAudioStreamEnd() {
        send(JSONObject().put("realtimeInput", JSONObject().put("audioStreamEnd", true)))
    }

    fun sendVideoFrame(jpeg: ByteArray) {
        send(JSONObject().put("realtimeInput", JSONObject().put(
            "video", JSONObject().put("mimeType", "image/jpeg")
                .put("data", Base64.encodeToString(jpeg, Base64.NO_WRAP))
        )))
    }

    fun sendSystemNote(text: String) {
        send(JSONObject().put("clientContent", JSONObject()
            .put("turns", JSONArray().put(
                JSONObject().put("role", "user").put("parts", JSONArray().put(
                    JSONObject().put("text", text)
                ))
            ))
            .put("turnComplete", true)
        ))
    }

    fun sendToolResponse(id: String, name: String, response: JSONObject) {
        send(JSONObject().put("toolResponse", JSONObject().put(
            "functionResponses", JSONArray().put(
                JSONObject().put("id", id).put("name", name).put("response", response)
            )
        )))
    }

    fun close() {
        webSocket?.close(1000, "closed by client")
        webSocket = null
    }

    private fun send(obj: JSONObject) {
        webSocket?.send(obj.toString()) ?: Log.w(TAG, "send() called with no open WebSocket")
    }

    companion object {
        private const val TAG = "GeminiLiveClient"
    }
}
