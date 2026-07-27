package com.tracking.client.live

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import java.util.concurrent.TimeUnit

private const val TAG = "RadioClient"

/** One match from [RadioClient.searchStation]. */
data class RadioStation(val name: String, val streamUrl: String, val codec: String)

/**
 * Direct 3rd-party call to the free, keyless Radio-Browser API
 * (radio-browser.info) — same "no server proxy" pattern as
 * YouTubeSearchClient.kt/NewsClient.kt. Hardcoded to Vietnam
 * (country=Vietnam) per this feature's scope. This client only ever
 * resolves a station name to a real stream URL — playback itself reuses
 * the EXISTING play_video/PlaybackService path (ExoPlayer, see
 * AndroidDeviceToolHandler.playVideo()), not a separate player, since
 * "a live radio stream" and "a resolved video/audio stream URL" are the
 * exact same problem this codebase already solves.
 */
class RadioClient {
    private val client = OkHttpClient.Builder().callTimeout(15, TimeUnit.SECONDS).build()

    /** Best (highest-voted) match for [query] among Vietnamese stations, or
     * null if none found/the request failed — same degrade-gracefully
     * convention as YouTubeSearchClient.kt. */
    suspend fun searchStation(query: String): RadioStation? = withContext(Dispatchers.IO) {
        if (query.isBlank()) return@withContext null
        val url = "https://de1.api.radio-browser.info/json/stations/search".toHttpUrl().newBuilder()
            .addQueryParameter("name", query)
            .addQueryParameter("country", "Vietnam")
            .addQueryParameter("limit", "5")
            .addQueryParameter("hidebroken", "true")
            .addQueryParameter("order", "votes")
            .addQueryParameter("reverse", "true")
            .build()
        // Radio-Browser's usage policy asks clients identify themselves via
        // User-Agent — a real header, not decorative.
        val request = Request.Builder().url(url).header("User-Agent", "TrackingAssistiveClient/1.0").build()
        try {
            client.newCall(request).execute().use { resp ->
                val bodyStr = resp.body?.string() ?: "[]"
                if (!resp.isSuccessful) {
                    Log.e(TAG, "searchStation('$query') HTTP ${resp.code}: $bodyStr")
                    return@withContext null
                }
                val items = JSONArray(bodyStr)
                if (items.length() == 0) {
                    Log.d(TAG, "searchStation('$query') -> no stations found")
                    return@withContext null
                }
                val item = items.getJSONObject(0)
                // url_resolved is the redirect-followed real stream URL —
                // preferred over the raw "url" field, which is sometimes a
                // playlist/redirect wrapper ExoPlayer can't play directly.
                val streamUrl = item.optString("url_resolved").ifBlank { item.optString("url") }
                if (streamUrl.isBlank()) {
                    Log.w(TAG, "searchStation('$query') matched '${item.optString("name")}' but has no stream URL")
                    return@withContext null
                }
                RadioStation(
                    name = item.optString("name"),
                    streamUrl = streamUrl,
                    codec = item.optString("codec"),
                ).also { Log.d(TAG, "searchStation('$query') -> ${it.name} (${it.streamUrl})") }
            }
        } catch (e: Exception) {
            Log.e(TAG, "searchStation('$query') failed", e)
            null
        }
    }
}
