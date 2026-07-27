package com.tracking.pixietest

import android.util.Base64
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Plain WebSocket client to pixie_hrtf_server.py — same OkHttp-direct
 * pattern the main client app's GeminiLiveClient.kt uses for its own raw
 * WebSocket protocol, just callback-based instead of a Flow (this is a
 * throwaway test-harness UI, not something composing multiple event
 * streams).
 *
 * Wire protocol (see pixie_hrtf_server.py's own docstring for the
 * server side):
 *   -> {"type":"frame","ts_ns":<Long>,"jpeg_b64":<String>}
 *   -> {"type":"reset"}
 *   <- {"type":"heading","frame_ts_ns":<Long>,"tracking_ok":<Bool>,
 *       "relative_heading_deg":<Float>,"pixie_azimuth_deg":<Float>,
 *       "pixie_elevation_deg":<Float>}
 *   <- {"type":"error","message":<String>}
 */
class WsHeadTrackClient {

    data class HeadingUpdate(
        val frameTsNs: Long,
        val trackingOk: Boolean,
        val relativeHeadingDeg: Float,
        val pixieAzimuthDeg: Float,
        val pixieElevationDeg: Float,
    )

    var onHeadingUpdate: ((HeadingUpdate) -> Unit)? = null
    var onStatus: ((String) -> Unit)? = null

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .build()
    private var webSocket: WebSocket? = null

    val isConnected: Boolean get() = webSocket != null

    fun connect(host: String, port: Int) {
        val url = "ws://$host:$port"
        val listener = object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                onStatus?.invoke("Connected to $url")
            }

            override fun onMessage(ws: WebSocket, text: String) = handleMessage(text)
            override fun onMessage(ws: WebSocket, bytes: ByteString) = handleMessage(bytes.utf8())

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure: ${t.message}", t)
                onStatus?.invoke("Connection failed: ${t.message}")
                webSocket = null
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                onStatus?.invoke("Connection closed: $reason")
                webSocket = null
            }
        }
        webSocket = client.newWebSocket(Request.Builder().url(url).build(), listener)
    }

    fun disconnect() {
        webSocket?.close(1000, "user disconnected")
        webSocket = null
    }

    fun sendFrame(jpeg: ByteArray, frameTsNs: Long): Boolean {
        val ws = webSocket ?: return false
        val msg = JSONObject().apply {
            put("type", "frame")
            put("ts_ns", frameTsNs)
            put("jpeg_b64", Base64.encodeToString(jpeg, Base64.NO_WRAP))
        }
        return ws.send(msg.toString())
    }

    fun sendReset() {
        webSocket?.send(JSONObject().put("type", "reset").toString())
    }

    private fun handleMessage(text: String) {
        val obj = try { JSONObject(text) } catch (e: Exception) {
            Log.w(TAG, "Unparseable server message: $text")
            return
        }
        when (obj.optString("type")) {
            "heading" -> onHeadingUpdate?.invoke(
                HeadingUpdate(
                    frameTsNs = obj.optLong("frame_ts_ns"),
                    trackingOk = obj.optBoolean("tracking_ok", false),
                    relativeHeadingDeg = obj.optDouble("relative_heading_deg", 0.0).toFloat(),
                    pixieAzimuthDeg = obj.optDouble("pixie_azimuth_deg", 0.0).toFloat(),
                    pixieElevationDeg = obj.optDouble("pixie_elevation_deg", 0.0).toFloat(),
                )
            )
            "error" -> onStatus?.invoke("Server error: ${obj.optString("message")}")
            else -> Log.w(TAG, "Unknown message type in: $text")
        }
    }

    companion object {
        private const val TAG = "WsHeadTrackClient"
    }
}
