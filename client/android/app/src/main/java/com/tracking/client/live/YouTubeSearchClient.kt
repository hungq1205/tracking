package com.tracking.client.live

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit

private const val TAG = "YouTubeSearchClient"

/** One result from [YouTubeSearchClient.search]/[YouTubeSearchClient.getVideoInfo]. */
data class YouTubeVideoResult(
    val videoId: String,
    val title: String,
    val channel: String,
    val durationIso8601: String? = null,
)

/**
 * Direct 3rd-party call to the YouTube Data API v3 — same "no server proxy"
 * pattern as OcrClient.kt. Only resolves search queries/video ids to
 * metadata (search_youtube/get_video_info); actual playback is a separate
 * concern handled by the official android-youtube-player (IFrame) library
 * via the play_youtube_video tool (see ToolDispatcher.kt) — this client
 * never returns a raw stream URL, since the IFrame player needs only a
 * video id.
 */
class YouTubeSearchClient(private val apiKey: String) {
    private val client = OkHttpClient.Builder()
        .callTimeout(15, TimeUnit.SECONDS)
        .build()

    /** Empty list if [apiKey] is blank or the request fails — callers treat
     * that the same as "no results found", per this project's existing
     * degrade-gracefully convention (see OcrClient.kt). */
    suspend fun search(query: String, maxResults: Int = 5): List<YouTubeVideoResult> = withContext(Dispatchers.IO) {
        if (apiKey.isBlank() || query.isBlank()) {
            Log.w(TAG, "search('$query') skipped — apiKey.isBlank()=${apiKey.isBlank()}")
            return@withContext emptyList()
        }
        val url = "https://www.googleapis.com/youtube/v3/search".toHttpUrl().newBuilder()
            .addQueryParameter("part", "snippet")
            .addQueryParameter("type", "video")
            .addQueryParameter("maxResults", maxResults.toString())
            .addQueryParameter("q", query)
            .addQueryParameter("key", apiKey)
            .build()
        try {
            client.newCall(Request.Builder().url(url).build()).execute().use { resp ->
                val bodyStr = resp.body?.string() ?: "{}"
                if (!resp.isSuccessful) {
                    Log.e(TAG, "search('$query') HTTP ${resp.code}: $bodyStr")
                    return@withContext emptyList()
                }
                val json = JSONObject(bodyStr)
                val items = json.optJSONArray("items") ?: return@withContext emptyList()
                (0 until items.length()).mapNotNull { i ->
                    val item = items.getJSONObject(i)
                    val videoId = item.optJSONObject("id")?.optString("videoId") ?: return@mapNotNull null
                    val snippet = item.optJSONObject("snippet") ?: return@mapNotNull null
                    YouTubeVideoResult(
                        videoId = videoId,
                        title = snippet.optString("title"),
                        channel = snippet.optString("channelTitle"),
                    )
                }.also { Log.d(TAG, "search('$query') -> ${it.size} result(s)") }
            }
        } catch (e: Exception) {
            Log.e(TAG, "search('$query') failed", e)
            emptyList()
        }
    }

    /** Full metadata (including duration) for specific video ids. */
    suspend fun getVideoInfo(videoIds: List<String>): List<YouTubeVideoResult> = withContext(Dispatchers.IO) {
        if (apiKey.isBlank() || videoIds.isEmpty()) {
            Log.w(TAG, "getVideoInfo($videoIds) skipped — apiKey.isBlank()=${apiKey.isBlank()}")
            return@withContext emptyList()
        }
        val url = "https://www.googleapis.com/youtube/v3/videos".toHttpUrl().newBuilder()
            .addQueryParameter("part", "snippet,contentDetails")
            .addQueryParameter("id", videoIds.joinToString(","))
            .addQueryParameter("key", apiKey)
            .build()
        try {
            client.newCall(Request.Builder().url(url).build()).execute().use { resp ->
                val bodyStr = resp.body?.string() ?: "{}"
                if (!resp.isSuccessful) {
                    Log.e(TAG, "getVideoInfo($videoIds) HTTP ${resp.code}: $bodyStr")
                    return@withContext emptyList()
                }
                val json = JSONObject(bodyStr)
                val items = json.optJSONArray("items") ?: return@withContext emptyList()
                (0 until items.length()).mapNotNull { i ->
                    val item = items.getJSONObject(i)
                    val videoId = item.optString("id").ifBlank { return@mapNotNull null }
                    val snippet = item.optJSONObject("snippet") ?: return@mapNotNull null
                    val contentDetails = item.optJSONObject("contentDetails")
                    YouTubeVideoResult(
                        videoId = videoId,
                        title = snippet.optString("title"),
                        channel = snippet.optString("channelTitle"),
                        durationIso8601 = contentDetails?.optString("duration"),
                    )
                }.also { Log.d(TAG, "getVideoInfo($videoIds) -> ${it.size} result(s)") }
            }
        } catch (e: Exception) {
            Log.e(TAG, "getVideoInfo($videoIds) failed", e)
            emptyList()
        }
    }
}
