package com.tracking.client.live

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import org.schabi.newpipe.extractor.NewPipe
import org.schabi.newpipe.extractor.ServiceList
import org.schabi.newpipe.extractor.downloader.Downloader
import org.schabi.newpipe.extractor.downloader.Request as NewPipeRequest
import org.schabi.newpipe.extractor.downloader.Response as NewPipeResponse

/**
 * Resolves a YouTube video id to a direct, playable AUDIO stream URL via
 * NewPipeExtractor — replaces the official WebView-based IFrame player
 * (deleted outright) as `play_youtube_video`'s playback mechanism, per
 * direct user request: the IFrame player's audio could only ever reach a
 * remote edge device through the unreliable MediaProjection/
 * AudioPlaybackCaptureConfiguration system-audio-capture path (uncertain
 * whether a WebView's internal player even tags its AudioTrack with a
 * usage that capture filter matches — see CLAUDE.md's "Remote edge device
 * follow-up round"); a resolved stream URL instead plays through
 * PlaybackService's own ExoPlayer, whose PCM output is tapped directly
 * (TeeRenderersFactory, see PlaybackService.kt) — a real, fully-controlled
 * path, same as radio/music already use.
 *
 * A KNOWN, ACCEPTED TRADEOFF, same as this codebase's other yt-dlp-style
 * resolver considerations: this deliberately steps away from the ToS
 * compliance the IFrame player was originally chosen for (see the now-
 * removed player's own doc comment history in CLAUDE.md) — accepted
 * directly by the user in exchange for reliable edge-device audio.
 *
 * Audio-only, not audio+video — PlaybackService only ever plays audio
 * (see its own doc comment), so there's no reason to resolve/download a
 * combined stream. Prefers a direct HTTP URL stream (`isUrl`) over a DASH/
 * HLS manifest one, since ExoPlayer's `MediaItem.fromUri()` (the exact
 * mechanism PlaybackService already uses for every stream_url) plays a
 * plain progressive URL with no extra manifest-parsing extension needed —
 * media3-exoplayer-hls is already a dependency (see build.gradle.kts) for
 * HLS radio streams, so an HLS manifest URL would also work if that's ever
 * the only kind NewPipeExtractor returns for a given video, but a direct
 * URL is preferred when available since it needs no extension at all.
 */
object YouTubeStreamResolver {

    @Volatile private var initialized = false

    private fun ensureInit() {
        if (initialized) return
        synchronized(this) {
            if (initialized) return
            NewPipe.init(OkHttpNewPipeDownloader(OkHttpClient()))
            initialized = true
        }
    }

    /** Returns the highest-average-bitrate playable audio stream URL for
     * [videoId], or null if resolution failed (age-restricted, region-
     * blocked, deleted/private video, or NewPipeExtractor itself failing
     * to parse YouTube's current page format — the standard risk of any
     * unofficial extractor, distinct from the IFrame player's own,
     * documented origin/error-handling bugs). Blocking network I/O,
     * dispatched onto [Dispatchers.IO]. */
    suspend fun resolveAudioStreamUrl(videoId: String): String? = withContext(Dispatchers.IO) {
        ensureInit()
        try {
            val extractor = ServiceList.YouTube.getStreamExtractor(
                "https://www.youtube.com/watch?v=$videoId"
            )
            extractor.fetchPage()
            val streams = extractor.audioStreams
            (streams.filter { it.isUrl }.maxByOrNull { it.averageBitrate }
                ?: streams.maxByOrNull { it.averageBitrate })
                ?.content
        } catch (e: Exception) {
            null
        }
    }

    /** Minimal [Downloader] backing NewPipeExtractor's own HTTP calls —
     * reuses this app's existing OkHttp dependency (already pulled in for
     * GeminiLiveClient's WebSocket, see build.gradle.kts) rather than
     * adding a second HTTP client. Only [execute] is abstract on
     * NewPipeExtractor's own Downloader base class — get/head/post all
     * have default implementations that funnel through it. */
    private class OkHttpNewPipeDownloader(private val client: OkHttpClient) : Downloader() {
        override fun execute(request: NewPipeRequest): NewPipeResponse {
            val builder = okhttp3.Request.Builder().url(request.url())
            for ((key, values) in request.headers()) {
                for (value in values) builder.addHeader(key, value)
            }
            val method = request.httpMethod()
            val dataToSend = request.dataToSend()
            when {
                method.equals("GET", ignoreCase = true) -> builder.get()
                method.equals("HEAD", ignoreCase = true) -> builder.head()
                dataToSend != null -> builder.method(method, dataToSend.toRequestBody())
                else -> builder.method(method, ByteArray(0).toRequestBody())
            }
            client.newCall(builder.build()).execute().use { resp ->
                val bodyBytes = resp.body?.bytes()
                val bodyString = bodyBytes?.toString(Charsets.UTF_8) ?: ""
                return NewPipeResponse(
                    resp.code,
                    resp.message,
                    resp.headers.toMultimap(),
                    bodyString,
                    resp.request.url.toString(),
                )
            }
        }
    }
}
