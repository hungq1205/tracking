package com.tracking.client.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import android.util.Log
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.viewinterop.AndroidView
import com.pierfrancescosoffritti.androidyoutubeplayer.core.player.PlayerConstants
import com.pierfrancescosoffritti.androidyoutubeplayer.core.player.YouTubePlayer
import com.pierfrancescosoffritti.androidyoutubeplayer.core.player.listeners.AbstractYouTubePlayerListener
import com.pierfrancescosoffritti.androidyoutubeplayer.core.player.options.IFramePlayerOptions
import com.pierfrancescosoffritti.androidyoutubeplayer.core.player.views.YouTubePlayerView

private const val YT_TAG = "YouTubePlayerOverlay"

@Composable
fun MainScreen(
    viewModel: MainViewModel,
    onOpenSettings: () -> Unit,
) {
    val uiState by viewModel.uiState.collectAsState()
    val pendingYoutubeVideoId by viewModel.pendingYoutubeVideoId.collectAsState()

    // Frame capture runs continuously against LiveAssistantService's own
    // lifecycle regardless (see CameraManager.bind()) — this only plugs/
    // unplugs the on-screen preview surface while this screen is actually
    // visible, per CLAUDE.md's "Client-Orchestrated Live Session" note on
    // the Preview/ImageAnalysis binding split.
    DisposableEffect(Unit) {
        onDispose { viewModel.detachCameraPreview() }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // Camera preview
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).also { previewView ->
                    viewModel.attachCameraPreview(previewView)
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // Settings button — top-right
        Row(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(end = 4.dp, top = 4.dp)
        ) {
            IconButton(onClick = onOpenSettings) {
                Icon(Icons.Default.Settings, contentDescription = "Settings", tint = Color.White)
            }
        }

        // Embedded YouTube (IFrame) player — play_youtube_video tool.
        // Always mounted (not conditionally composed on a pending video id)
        // so it's ready instantly rather than being re-created per
        // playback — deliberately the ONLY player surface for YouTube
        // specifically (play_video's resolved-stream path stays headless
        // via PlaybackService) — the official player requires being
        // attached/rendering to play at all, a real ToS-driven limitation,
        // not an oversight. See CLAUDE.md's YouTube playback note.
        //
        // Per direct user request: visually hidden (near-zero size + fully
        // transparent) while audio keeps playing — the WebView stays
        // attached and rendering (alpha/size don't pause a WebView's own
        // media playback), it's just drawn with zero opacity into a 1.dp
        // box, so nothing shows on screen. No on-screen close button any
        // more, since there's nothing visible left to tap — dismiss via the
        // voice `stop_music`/`stop_radio` tools instead, consistent with
        // this app's voice-first design (dismissYoutubeVideo() is still
        // reachable programmatically if a future affordance needs it).
        YouTubePlayerOverlay(
            videoId = pendingYoutubeVideoId,
            isAwaitingResponse = uiState.isAwaitingResponse,
            onPlaybackStateChanged = { isPlaying -> viewModel.reportYoutubePlaybackState(isPlaying) },
            modifier = Modifier
                .size(1.dp)
                .alpha(0f)
        )
    }
}

@Composable
private fun YouTubePlayerOverlay(
    videoId: String?,
    isAwaitingResponse: Boolean,
    onPlaybackStateChanged: (Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    var player by remember { mutableStateOf<YouTubePlayer?>(null) }
    // Backs the listener's log lines below with the LATEST videoId, not
    // whichever value happened to be current when the AndroidView factory
    // ran (factory only runs once per view instance, so a plain closure
    // over the `videoId` parameter would otherwise always log the FIRST
    // composition's value, not the one actually being played later).
    val currentVideoId = rememberUpdatedState(videoId)
    val currentOnPlaybackStateChanged = rememberUpdatedState(onPlaybackStateChanged)

    LaunchedEffect(videoId, player) {
        if (videoId != null) player?.loadVideo(videoId, 0f) else player?.pause()
    }

    // Ducking only once the user's utterance is fully registered and
    // Gemini's response is about to arrive — NOT during capture itself
    // (music/reading keep playing while the user is being recorded, per
    // the user's explicit spec). This player's instance lives in Compose,
    // not the Service, so it reacts to isAwaitingResponse directly here
    // rather than through LiveAssistantService's PlaybackService-pause
    // path. See CLAUDE.md's "Continuous VAD-gated listening" note.
    LaunchedEffect(isAwaitingResponse, player) {
        if (videoId == null) return@LaunchedEffect
        if (isAwaitingResponse) player?.pause() else player?.play()
    }

    Box(modifier = modifier) {
        AndroidView(
            factory = { ctx ->
                YouTubePlayerView(ctx).also { view ->
                    lifecycleOwner.lifecycle.addObserver(view)
                    // Error 152-4 firing IMMEDIATELY on load (before any
                    // playback attempt) is YouTube's anti-bot/referrer
                    // verification rejecting a WebView with no valid
                    // Referer/origin at all — automatic initialization (no
                    // origin set) is the WORST case for this, not a safe
                    // default. A prior attempt set origin to
                    // "https://www.youtube.com" itself, which is wrong (origin
                    // must identify the EMBEDDING app/page, not claim to BE
                    // youtube.com) and plausibly made it worse. Correct value
                    // per YouTube's own documented workaround for WebView/app
                    // embeds: "https://www.youtube-nocookie.com".
                    view.enableAutomaticInitialization = false
                    view.initialize(
                        object : AbstractYouTubePlayerListener() {
                            override fun onReady(youTubePlayer: YouTubePlayer) {
                                Log.d(YT_TAG, "onReady, pending videoId=${currentVideoId.value}")
                                player = youTubePlayer
                                currentVideoId.value?.let { youTubePlayer.loadVideo(it, 0f) }
                            }
                            override fun onStateChange(
                                youTubePlayer: YouTubePlayer,
                                state: PlayerConstants.PlayerState,
                            ) {
                                Log.d(YT_TAG, "onStateChange: $state (videoId=${currentVideoId.value})")
                                // Real playing/not-playing signal for VAD
                                // output-aware gating (LiveAssistantService's
                                // isOutputActive) — must reflect ACTUAL
                                // audio output, not just "a video is loaded,"
                                // since a paused-but-loaded video was found to
                                // keep the VAD's 3x threshold multiplier stuck
                                // on indefinitely (see that field's doc).
                                currentOnPlaybackStateChanged.value(state == PlayerConstants.PlayerState.PLAYING)
                            }
                            override fun onError(
                                youTubePlayer: YouTubePlayer,
                                error: PlayerConstants.PlayerError,
                            ) {
                                // This is the signal that was previously completely
                                // silent — toolPlayYoutubeVideo() reports "playing"
                                // to Gemini before the WebView even attempts to load,
                                // so an invalid video id / embedding-disabled /
                                // region-blocked video would fail with zero trace.
                                Log.e(YT_TAG, "onError: $error (videoId=${currentVideoId.value})")
                            }
                        },
                        IFramePlayerOptions.Builder().origin("https://www.youtube-nocookie.com").build(),
                    )
                }
            },
            modifier = Modifier.fillMaxWidth()
        )
    }
}
