package com.tracking.client.live

import android.media.AudioManager
import android.media.ToneGenerator
import android.util.Log
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.tracking.client.audio.PixieController
import com.tracking.client.audio.PixiePoint
import com.tracking.client.audio.ReadingTtsPlayer
import com.tracking.client.device.DeviceToolHandler
import com.tracking.client.grpc.GrpcClientManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONObject
import tracking.Tracking
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.math.abs

/** Result of one dispatched tool call, plus optional side effects the caller
 * (MainViewModel) needs to react to (starting/stopping local tracking, UI state). */
data class DispatchResult(val response: JSONObject)

/**
 * Kotlin port of server/live_session.py's `_dispatch_tool` + the
 * server/live_tools tool implementations — the on-device tool-calling loop
 * now that Gemini Live orchestration runs
 * here (see CLAUDE.md's "Client-Orchestrated Live Session" section). Each
 * tool routes to exactly one implementation, by weight (heavy models
 * remote, plain bookkeeping local, device-native calls in-process, OCR
 * direct 3rd-party) — not a swappable interface for now, per the migration
 * plan.
 */
class ToolDispatcher(
    private val grpc: GrpcClientManager,
    private val ocrClient: OcrClient,
    private val memoryStore: LocalMemoryStore,
    private val deviceToolHandler: DeviceToolHandler,
    private val state: LiveSessionState,
    private val locationId: String,
    private val avoidanceIntervalMs: Int = 350,
    private val scope: CoroutineScope,
    private val latestFrame: () -> ByteArray?,
    private val sendVideoFrame: (ByteArray) -> Unit,
    private val sendSystemNote: (String) -> Unit,
    // On-device TextToSpeech for reading mode — replaces the old
    // gRPC-based PerceptionService.Synthesize (KokoroTTS) round trip +
    // shared StreamingAudioPlayer PCM path (playPcm/flushPlayback, both
    // removed) per direct user request, so reading keeps working with the
    // server unreachable. See ReadingTtsPlayer.kt's own doc comment.
    private val readingTts: ReadingTtsPlayer,
    // detectionPrompt is what actually gets sent to DetectObject/GroundingDINO
    // — the memory-resolved description when start_tracking() was called
    // with one, otherwise just target again. See toolStartTracking()'s own
    // comment for why these two can legitimately differ (a proper-noun
    // label like "Cutie Pie" is a poor open-vocab detection prompt).
    private val onTrackingStateChanged: (active: Boolean, target: String, detectionPrompt: String) -> Unit,
    private val onGuidanceUpdate: (mode: String, waypoints: List<Pair<Float, Float>>) -> Unit,
    // Generic 4-point HRTF cue player — replaces HrtfBeaconPlayer for
    // tracking/guiding/walking (see CLAUDE.md's "Pixie + Angle modules"
    // note); HrtfBeaconPlayer itself stays in the codebase, unreferenced.
    private val pixieController: PixieController,
    // Client-side latency bridging between MappingService's ~1Hz updates
    // (see CLAUDE.md's "Server-planned walking path" note) — AngleTracker
    // (a drop-in replacement for the old RotationTracker, see its own doc
    // comment) estimates head rotation from consecutive camera frames
    // (Essential matrix decomposition, no scale ambiguity for rotation);
    // PdrStepEstimator estimates walked distance from the phone's
    // step-detector + a fixed stride length. Neither runs unless
    // WALKING/GUIDING is active.
    private val angleTracker: AngleTracker,
    private val pdrStepEstimator: PdrStepEstimator,
    // Reading-mode blur skip/retry (see acquireSharpFrame()) — distinct from
    // latestFrame() (used everywhere else: OCR's own frame source is still
    // the sharpest-in-window recentBufferMs pull, see CameraManager.kt) in
    // that it also exposes the sharpness score so a below-threshold frame
    // can be retried instead of OCR'd outright. threshold<=0 disables the
    // check entirely (acquireSharpFrame() falls back to latestFrame()).
    private val latestFrameWithSharpness: () -> Pair<ByteArray, Double>? = { null },
    private val blurSharpnessThreshold: Double = 40.0,
    // Full-resolution, one-shot OCR capture channel — a remote edge device's
    // continuous frameFlow/lumaFlow are deliberately low-res (see
    // CLAUDE.md's "Full-resolution OCR capture channel" note), so OCR asks
    // for a dedicated full-res still instead. null (the default, and always
    // the case for the phone's own local camera) means "no such channel" —
    // acquireSharpFrame() then falls straight through to the existing
    // latestFrame()/latestFrameWithSharpness() path, unchanged.
    private val ocrFrameFlow: SharedFlow<ByteArray>? = null,
    private val requestOcrFrame: () -> Unit = {},
    // Debug-only: when non-null, toolScanCurrentView() draws each OCR
    // line's kept/dropped box onto the frame (see DebugFrameStore.kt) and
    // hands the annotated JPEG to this callback — MainViewModel wires it to
    // write into app-private storage only when the user has explicitly
    // enabled "Save debug OCR frames" in Settings (off by default: writing
    // every scanned frame to disk has real storage/privacy cost).
    private val saveDebugFrame: ((ByteArray) -> Unit)? = null,
    // Live "what did OCR just look at" preview — every frame actually sent
    // to OCR.space (one-shot scan_current_view() AND the continuous
    // live-reading capture loop) is handed to this callback so the phone's
    // own screen can show it, instead of the scan happening invisibly.
    // Distinct from saveDebugFrame above (which persists to disk, gated
    // behind a Settings toggle, off by default) — this is always-on,
    // in-memory-only UI feedback, wired by MainViewModel/
    // LiveAssistantService the same way the remote-edge-device camera
    // preview (LiveAssistantService.edgeFrame) already works.
    private val onOcrFrame: ((ByteArray) -> Unit)? = null,
    // OCR-error correction (+ translation for non-English text) — ported
    // from gt.py's Gemini-correction pipeline (see GeminiCorrectionClient's
    // own doc comment). Reuses the same geminiApiKey already used for
    // Gemini Live — no separate Settings field. Null (e.g. blank API key)
    // disables correction entirely: toolScanCurrentView() then behaves
    // exactly as before, storing/speaking raw OCR text with corrected
    // always mirroring it.
    private val geminiCorrectionClient: GeminiCorrectionClient? = null,
    // Generates remember_object()'s stored description via a dedicated
    // one-shot vision call instead of trusting Gemini Live's own free-form
    // text (see GeminiObjectDescriptionClient's own doc comment — a direct
    // user request for terser, more consistent descriptions). Same null-
    // when-no-key convention as geminiCorrectionClient: toolRememberObject()
    // then falls back to whatever description Gemini Live itself passed in.
    private val geminiObjectDescriptionClient: GeminiObjectDescriptionClient? = null,
    // YouTube search/metadata — direct 3rd-party call (see
    // YouTubeSearchClient.kt), null when no API key is configured (degrades
    // to a clear "not configured" error, same convention as
    // geminiCorrectionClient above). Playback itself no longer goes through
    // a callback into an on-screen player — toolPlayYoutubeVideo() resolves
    // a real stream URL (YouTubeStreamResolver.kt) and dispatches straight
    // to play_video/PlaybackService, same as radio/music.
    private val youtubeSearchClient: YouTubeSearchClient? = null,
    // Piggybacked on reportMode()'s own existing call sites (every mode
    // transition) -- wired to edgeDevice.reportMode() so a remote Pi can
    // skip capturing/sending luma_out outside walking/guiding, its only
    // real consumer (see EdgeDevice.reportMode()'s own doc comment).
    // No-op default -- LocalEdgeDevice needs no equivalent.
    private val reportModeToEdge: (String) -> Unit = {},
    // Obstacle-ahead alert (pollObstacleAheadOnce()) — plays the bundled
    // assets/beep.mp3 asset via a Service-owned SoundPool. A callback
    // (not a raw Context/SoundPool field here) to keep this class free of
    // Android framework audio plumbing, same convention as
    // reportModeToEdge above.
    private val playObstacleBeep: () -> Unit = {},
) {
    // News/radio — both keyless, no-config 3rd-party APIs hardcoded to
    // Vietnam/Vietnamese (see NewsClient.kt/RadioClient.kt's own doc
    // comments), so unlike youtubeSearchClient above they need no Settings
    // field or constructor injection — always available.
    private val newsClient = NewsClient()
    private val radioClient = RadioClient()

    private var visionStreamJob: Job? = null
    private var mappingJob: Job? = null
    private var mappingChunkChannel: Channel<Tracking.MappingChunk>? = null
    private var avoidanceJob: Job? = null
    private var obstacleAheadJob: Job? = null
    private var guidingArrivalAnnounced = false
    // One-shot per WALKING/GUIDING session — see the mapping-stream
    // collector's own comment (announces the first main-path joint's clock
    // direction once, subsequent announcements come from
    // advanceMainPathJoint() instead). Reset in stopActiveModes() so it's
    // fresh for the next mode-entry.
    private var mainJointAnnounced = false

    // Walking-mode cold-start warm-up gate — requested directly by the
    // user after real on-device lag at walking start: the mapping stream
    // used to open and immediately start feeding a continuous stream of
    // frames + running avoidance ticks/Pixie, all before the server's
    // FIRST RTAB-Map/occupancy round trip had actually landed. Now:
    // feedMappingFrame() sends exactly ONE frame (walkingFirstFrameSent
    // guards this) and then withholds further frames until walkingReady
    // flips true — set by the mapping-stream collector the moment ANY
    // real MappingUpdate arrives for this session (see startMappingStream()).
    // Only once ready does toolStartWalking() actually activate ticks/
    // Pixie/PDR and announce readiness — see activateWalkingOnceReady().
    private var walkingReady = false
    private var walkingFirstFrameSent = false

    // Tracking-mode Pixie state machine — see updateTrackingPixie(). Phase
    // HORIZONTAL guides left/right first; once centered it switches to
    // VERTICAL (up/down); slipping back out of horizontal alignment while
    // in VERTICAL drops back to HORIZONTAL. Reset in toolStartTracking().
    private enum class TrackingAxis { HORIZONTAL, VERTICAL }
    private var trackingAxis = TrackingAxis.HORIZONTAL

    // ── OCR correction queue (reading mode) ─────────────────────────────
    // Kotlin port of gt.py's ReadingPipeline correction_queue/
    // _correction_worker: OCR results are stored (and, for "new"/"stitched",
    // spoken) IMMEDIATELY via toolScanCurrentView() — correction runs
    // strictly SEQUENTIALLY (one Gemini call finishes before the next
    // starts, same contract as gt.py's worker) on its own background
    // coroutine, entirely decoupled from the tool-call path, so a slow or
    // failed correction never blocks scanning/reading. Unlimited capacity:
    // a burst of scans should never block on this queue being full.
    // onComplete (new) — fires once this specific job's correction attempt
    // finishes, success or failure, so the caller can cue "the scan/
    // correct/store pipeline for THIS capture is fully done" (see
    // playScanCompleteCue()) at the right moment instead of guessing when
    // an async correction landed.
    private data class CorrectionJob(val blockId: Int, val rawText: String, val langCode: String, val onComplete: (() -> Unit)? = null)
    private val correctionChannel = Channel<CorrectionJob>(Channel.UNLIMITED)
    private var correctionWorkerJob: Job? = null

    init {
        if (geminiCorrectionClient != null) {
            correctionWorkerJob = scope.launch(Dispatchers.IO) {
                for (job in correctionChannel) {
                    try {
                        val corrected = geminiCorrectionClient.correct(job.rawText, job.langCode)
                        val applied = OcrBlockFilters.applyCorrection(state.readingBlocks, job.blockId, job.rawText, corrected)
                        Log.d(TAG, "[correction] block #${job.blockId} -> ${if (applied) "applied" else "discarded (raw changed since request)"}")
                    } catch (e: Exception) {
                        // Never let one bad correction (bad key, network
                        // error, quota) kill the worker loop — future
                        // frames must still get corrected once the root
                        // cause is fixed. GeminiCorrectionClient already
                        // logged the real exception; this is just the
                        // block-level "it failed" fact.
                        Log.w(TAG, "[correction] block #${job.blockId} error: ${e.message}")
                    } finally {
                        // Fires even on failure — the pipeline attempt for
                        // this capture is over either way, and staying
                        // silent forever on a correction failure would be
                        // worse than confirming a (possibly uncorrected)
                        // completion.
                        job.onComplete?.invoke()
                    }
                }
            }
        }
    }

    /** ML Kit's on-device Language Identification — same role gt.py's
     * fastText detector played (picking which correction prompt to use),
     * chosen over trying to port/bundle fastText itself since ML Kit is a
     * standard, well-supported Android library with no model-management
     * code needed here (it lazily downloads its own small model). Returns
     * "" (treated as "unknown, use the fix+translate prompt as a safe
     * default") on failure or "und" (undetermined) — never throws. */
    private suspend fun detectLanguageCode(text: String): String = suspendCancellableCoroutine { cont ->
        val identifier = LanguageIdentification.getClient()
        identifier.identifyLanguage(text)
            .addOnSuccessListener { code -> if (cont.isActive) cont.resume(if (code == "und") "" else code) }
            .addOnFailureListener { e ->
                Log.w(TAG, "language detection failed: ${e.message}")
                if (cont.isActive) cont.resume("")
            }
    }

    // Dead-end alert (see playDeadEndAlert()) — a synthesized tone, not a
    // bundled asset, so it works without needing an audio file supplied.
    // Rate-limited so it doesn't fire every single avoidance tick.
    private var toneGenerator: ToneGenerator? = null
    private var lastDeadEndAlertAtMs = 0L

    // Periodic hazard/narration vision check (see runPeriodicVisionCheck())
    // — shared by both WALKING and GUIDING's avoidance tick. Fires every
    // PERIODIC_ALERT_INTERVAL_MS since the LAST time THIS sent a request —
    // a flat send-to-send cadence, not gated by a local depth heuristic or
    // by whether a response is still playing — see that function's own
    // docstring for why. Independent of the VAD/user-speech pipeline — a
    // real user utterance is handled by ContinuousVadRecorder/
    // LiveAssistantService as normal and is never delayed or blocked by
    // this cadence.
    private var lastPeriodicAlertAtMs = 0L

    // Depth-map-based obstacle-directly-ahead alert (see
    // checkAndWarnObstacleAhead()) — its own independent rate limit,
    // separate from lastPeriodicAlertAtMs above (different trigger,
    // different urgency).
    private var lastObstacleAheadWarnedAtMs = 0L
    // Trend tracking for pollObstacleAheadOnce()'s "getting closer" check —
    // null means either no reading yet this session, or the obstacle was
    // last seen out of range (reset there so a later encounter starts its
    // own fresh trend instead of comparing against a stale distance).
    private var lastObstacleDistanceM: Float? = null

    // Ambient WALKING-only frame feed (see sendWalkingAmbientFrame()) — gives
    // Gemini continuous, up-to-date visual context of what's ahead while
    // walking, independent of the hazard alert path below. Sent as a plain
    // realtimeInput video frame (no turnComplete — see GeminiLiveClient.
    // sendVideoFrame()), so it never forces a response/interrupts anything;
    // it's just buffered context for whenever Gemini does speak next.
    private var lastWalkingAmbientFrameSentAtMs = 0L

    /** Ensures mode exclusivity — every mode-entry tool calls this first.
     * Found via a real bug: state.mode is documented as a single exclusive
     * value, but nothing actually enforced that — toolStartGuiding/
     * toolStartWalking/toolStartScan/toolEnterReadingMode never stopped a
     * still-active tracking session, so its local-tracking init-retry loop
     * (MainViewModel.startLocalTracking(), retries every 1s until the
     * target is found) kept hammering the server with DetectObject calls
     * indefinitely alongside whatever new mode had started, contending for
     * GPU/model resources with mapping's own DA3/RTAB-Map calls. */
    private fun stopActiveModes() {
        if (state.mode == "tracking") {
            state.trackingTarget = ""
            onTrackingStateChanged(false, "", "")
        }
        if (state.mode == "guiding" || state.mode == "walking" || state.mode == "scanning") {
            stopMappingStream()
        }
        stopLiveReadingPipeline()
        // Switching to another mode mid-read-aloud must stop the speech too
        // — previously only the live-reading capture/OCR loop was stopped
        // here, not in-progress TTS.
        readingTts.stop()
        stopLocalAvoidanceTicks()
        stopObstacleAheadPolling()
        pixieController.stop()
        pdrStepEstimator.stop()
        pdrStepEstimator.resetAccumulator()
        angleTracker.reset()
        trackingAxis = TrackingAxis.HORIZONTAL
        // In case walking was interrupted mid-warm-up (switched modes
        // before activateWalkingOnceReady() ever fired) — see the
        // class-level walkingReady doc comment.
        walkingReady = false
        walkingFirstFrameSent = false
        mainJointAnnounced = false
    }

    /** Auto-enters reading mode on demand — replaces the old explicit
     * enter_reading_mode() tool call Gemini used to have to make first.
     * Only resets the buffer/cursor when actually TRANSITIONING into
     * reading from something else; repeated scan/read_aloud/live-reading
     * calls while ALREADY in reading mode must not wipe progress made so
     * far. Called by every real reading entry point (scan_current_view,
     * read_aloud, start_live_reading). */
    private fun ensureReadingMode() {
        if (state.mode == "reading") return
        Log.d(TAG, "[reading] entering reading mode from '${state.mode}'")
        stopActiveModes()
        state.mode = "reading"
        state.readingBlocks.clear()
        state.pageSummaries.clear()
        state.readingCursorSentenceIndex = 0
        state.readingLabel = "reading"
        state.readingDirection = "ltr"
        reportMode("reading", state.readingLabel)
    }

    /** Tells the server which mode state.mode just became — the server has
     * no other way to know (no server-side session any more, see
     * CLAUDE.md's "Client-Orchestrated Live Session" section), and
     * server_gui.py's dashboard uses this to select the correct tab
     * directly instead of inferring it from whichever RPC category most
     * recently happened to fire. Fire-and-forget on IO: a dropped/failed
     * report only means the dashboard falls back to inference for a
     * moment, never something the tool-call flow itself should fail over. */
    private fun reportMode(mode: String, target: String = "") {
        reportModeToEdge(mode)
        val stub = grpc.statusStub ?: return
        scope.launch(Dispatchers.IO) {
            try {
                stub.reportMode(
                    Tracking.ReportModeRequest.newBuilder().setMode(mode).setTarget(target).build()
                )
            } catch (e: Exception) {
                Log.w(TAG, "reportMode('$mode') failed: ${e.message}")
            }
        }
    }

    /** Called once, right after a fresh connection is established (see
     * LiveAssistantService.connect()) — clears every accumulated server-side
     * scan/mapping session (all location_ids) and resets the shared
     * RTAB-Map docker session, so a new connection never silently resumes a
     * stale map/pose left over from a previous one. Same fire-and-forget
     * precedent as reportMode()/reportBeaconDirection(): a dropped/failed
     * reset just means the server keeps whatever state it already had,
     * never something connect() itself should fail over. */
    fun resetSession() {
        val stub = grpc.statusStub ?: return
        scope.launch(Dispatchers.IO) {
            try {
                stub.resetSession(Tracking.ResetSessionRequest.newBuilder().build())
            } catch (e: Exception) {
                Log.w(TAG, "resetSession failed: ${e.message}")
            }
        }
    }

    /** Dashboard-only, same fire-and-forget precedent as reportMode() above
     * — the server has no other way to learn the beacon's actual final
     * azimuth, since goal-biasing + EMA smoothing now happen entirely
     * client-side (see runAvoidanceTick()). Called once per avoidance tick
     * while walking/guiding is active. */
    private fun reportBeaconDirection(azimuthDeg: Float, muted: Boolean) {
        val stub = grpc.statusStub ?: return
        scope.launch(Dispatchers.IO) {
            try {
                stub.reportBeaconDirection(
                    Tracking.ReportBeaconDirectionRequest.newBuilder()
                        .setAzimuthDeg(azimuthDeg).setMuted(muted).build()
                )
            } catch (e: Exception) {
                Log.w(TAG, "reportBeaconDirection failed: ${e.message}")
            }
        }
    }

    suspend fun dispatch(name: String, args: JSONObject): JSONObject = try {
        when (name) {
            "get_current_time" -> toolGetCurrentTime()
            "get_latest_frame" -> toolGetLatestFrame()
            "start_vision_stream" -> toolStartVisionStream()
            "stop_vision_stream" -> toolStopVisionStream()
            "run_detection" -> toolRunDetection(args.optString("object_description", ""))
            "check_obstacle" -> toolCheckObstacle()

            "scan_current_view" -> toolScanCurrentView()
            "get_reading_section" -> toolGetReadingSection(args.optString("query", ""))
            "read_aloud" -> toolReadAloud()
            "flip_reading_direction" -> toolFlipReadingDirection()
            "exit_reading_mode" -> toolExitReadingMode()
            "start_live_reading" -> toolStartLiveReading()
            "stop_live_reading" -> toolStopLiveReading()
            "continue_reading" -> toolContinueReading()

            "start_tracking" -> toolStartTracking(args.optString("target", ""), args.optString("description", ""))
            "stop_tracking" -> toolStopTracking()
            "get_object_from_memory" -> toolGetObjectFromMemory(args.optString("query", ""))
            "is_this_object" -> toolIsThisObject(args.optString("label", ""))
            "search_objects" -> toolSearchObjects(args.optJSONArray("targets"))

            "query_memory" -> toolQueryMemory(args.optString("question", ""))
            "save_memory" -> toolSaveMemory(args.optString("label", ""), args.optString("note", ""))
            "save_reading_buffer" -> toolSaveReadingBuffer(args.optString("label", ""))
            "remember_object" -> toolRememberObject(args.optString("label", ""), args.optString("description", ""))
            "list_memory_labels" -> toolListMemoryLabels()
            "clear_memory" -> toolClearMemory(args.optString("label", ""))

            "start_guiding" -> toolStartGuiding(args.optString("destination", ""))
            "stop_guiding" -> toolStopGuiding()
            "get_current_location" -> toolGetCurrentLocation()

            "start_walking" -> toolStartWalking()
            "stop_walking" -> toolStopWalking()

            "start_scan" -> toolStartScan()
            "stop_scan" -> toolStopScan()

            "search_youtube" -> toolSearchYoutube(args.optString("query", ""))
            "get_video_info" -> toolGetVideoInfo(args)
            "play_youtube_video" -> toolPlayYoutubeVideo(args.optString("video_id", ""))

            "make_phone_call", "search_contacts", "set_alarm", "create_calendar_event", "play_video",
            "answer_phone_call", "send_sms", "check_unread_sms" ->
                dispatchDeviceTool(name, args)

            "stop_music" -> toolStopMusic(args)

            "get_top_news" -> toolGetTopNews()
            "search_news" -> toolSearchNews(args.optString("query", ""))
            "play_radio" -> toolPlayRadio(args.optString("station_query", ""))
            "stop_radio" -> toolStopMusic(args)

            else -> JSONObject().put("error", "Unknown tool: $name")
        }
    } catch (e: Exception) {
        Log.e(TAG, "Tool '$name' failed", e)
        JSONObject().put("error", e.message ?: "unknown error")
    }

    private suspend fun dispatchDeviceTool(name: String, args: JSONObject): JSONObject {
        val call = com.tracking.client.device.DeviceToolCall(callId = "local", name = name, argsJson = args.toString())
        val resultJson = deviceToolHandler.execute(call)
        return try { JSONObject(resultJson) } catch (e: Exception) { JSONObject().put("raw", resultJson) }
    }

    // ── YouTube ──────────────────────────────────────────────────────────

    private suspend fun toolSearchYoutube(query: String): JSONObject {
        Log.d(TAG, "search_youtube(query='$query') youtubeSearchClient=${youtubeSearchClient != null}")
        if (query.isBlank()) return JSONObject().put("error", "query required")
        val client = youtubeSearchClient
            ?: return JSONObject().put("error", "YouTube search is not configured — set a YouTube API Key in Settings.")
        val results = client.search(query)
        Log.d(TAG, "search_youtube(query='$query') -> ${results.size} result(s): ${results.map { it.videoId }}")
        if (results.isEmpty()) return JSONObject().put("found", false)
        val array = org.json.JSONArray()
        results.forEach { r ->
            array.put(JSONObject().put("video_id", r.videoId).put("title", r.title).put("channel", r.channel))
        }
        return JSONObject().put("found", true).put("results", array)
    }

    private suspend fun toolGetVideoInfo(args: JSONObject): JSONObject {
        val idsArg = args.optJSONArray("video_ids")
        val ids = if (idsArg != null) (0 until idsArg.length()).map { idsArg.getString(it) } else emptyList()
        if (ids.isEmpty()) return JSONObject().put("error", "video_ids required")
        val client = youtubeSearchClient
            ?: return JSONObject().put("error", "YouTube search is not configured — set a YouTube API Key in Settings.")
        val results = client.getVideoInfo(ids)
        val array = org.json.JSONArray()
        results.forEach { r ->
            array.put(
                JSONObject().put("video_id", r.videoId).put("title", r.title)
                    .put("channel", r.channel).put("duration", r.durationIso8601 ?: "")
            )
        }
        return JSONObject().put("results", array)
    }

    /** Resolves [videoId] to a direct audio stream URL via
     * YouTubeStreamResolver (NewPipeExtractor), then dispatches to the SAME
     * play_video/PlaybackService path radio/music already use — replaces
     * the old WebView-based IFrame player entirely (removed outright) per
     * direct user request, so YouTube audio reaches a remote edge device
     * through PlaybackService's own PCM tap (TeeRenderersFactory) instead
     * of the unreliable MediaProjection system-audio-capture path a WebView
     * player left as the only option. See YouTubeStreamResolver.kt's own
     * doc comment for the accepted ToS tradeoff. */
    private suspend fun toolPlayYoutubeVideo(videoId: String): JSONObject {
        Log.d(TAG, "play_youtube_video(video_id='$videoId')")
        if (videoId.isBlank()) return JSONObject().put("error", "video_id required")
        val streamUrl = YouTubeStreamResolver.resolveAudioStreamUrl(videoId)
            ?: return JSONObject().put("error", "Could not resolve a playable stream for that video (it may be age-restricted, region-blocked, or removed).")
        dispatchDeviceTool(
            "play_video",
            JSONObject().put("stream_url", streamUrl).put("video_id", videoId)
        )
        return JSONObject().put("status", "playing").put("video_id", videoId)
    }

    /** Stops playback (play_video/play_radio/play_youtube_video all funnel
     * through the same PlaybackService now, so one stop covers all of
     * them). */
    private suspend fun toolStopMusic(args: JSONObject): JSONObject = dispatchDeviceTool("stop_music", args)

    // ── News / Radio (Vietnam only, hardcoded — see NewsClient.kt/RadioClient.kt) ──

    private fun newsArticlesToJson(articles: List<NewsArticle>): JSONObject {
        if (articles.isEmpty()) return JSONObject().put("found", false)
        val array = org.json.JSONArray()
        articles.forEach { a ->
            array.put(
                JSONObject().put("title", a.title).put("source", a.source)
                    .put("pub_date", a.pubDate).put("link", a.link)
            )
        }
        return JSONObject().put("found", true).put("results", array)
    }

    private suspend fun toolGetTopNews(): JSONObject = newsArticlesToJson(newsClient.getTopHeadlines())

    private suspend fun toolSearchNews(query: String): JSONObject {
        if (query.isBlank()) return JSONObject().put("error", "query required")
        return newsArticlesToJson(newsClient.search(query))
    }

    /** Resolves [stationQuery] to a real stream URL via RadioClient, then
     * dispatches to the SAME play_video/PlaybackService path YouTube's
     * resolved-stream fallback and any other stream source already use —
     * a live radio stream is just another stream URL, no separate player
     * needed. */
    private suspend fun toolPlayRadio(stationQuery: String): JSONObject {
        if (stationQuery.isBlank()) return JSONObject().put("error", "station_query required")
        val station = radioClient.searchStation(stationQuery)
            ?: return JSONObject().put("error", "Radio station '$stationQuery' not found.")
        dispatchDeviceTool(
            "play_video",
            JSONObject().put("stream_url", station.streamUrl).put("title", station.name).put("channel", "Radio")
        )
        return JSONObject().put("status", "playing").put("station", station.name)
    }

    // ── Time ─────────────────────────────────────────────────────────────

    private fun toolGetCurrentTime(): JSONObject {
        val now = java.util.Calendar.getInstance()
        val fmt = java.text.SimpleDateFormat("EEEE, MMMM d, yyyy HH:mm", java.util.Locale.US)
        return JSONObject().put("time", fmt.format(now.time))
    }

    // ── Scene / Vision ───────────────────────────────────────────────────

    private fun toolGetLatestFrame(): JSONObject {
        val frame = latestFrame() ?: return JSONObject().put("error", "No frame available yet.")
        sendVideoFrame(frame)
        return JSONObject().put("status", "frame_sent")
    }

    private fun toolStartVisionStream(): JSONObject {
        visionStreamJob?.cancel()
        visionStreamJob = scope.launch(Dispatchers.IO) {
            val deadline = System.currentTimeMillis() + 15_000L
            while (isActive && System.currentTimeMillis() < deadline) {
                latestFrame()?.let(sendVideoFrame)
                delay(1000L)
            }
        }
        return JSONObject().put("status", "vision_stream_started")
    }

    private fun toolStopVisionStream(): JSONObject {
        visionStreamJob?.cancel(); visionStreamJob = null
        return JSONObject().put("status", "vision_stream_stopped")
    }

    private suspend fun toolRunDetection(prompt: String): JSONObject {
        if (prompt.isBlank()) return JSONObject().put("error", "object_description required")
        val frame = latestFrame() ?: return JSONObject().put("error", "No frame available.")
        val stub = grpc.perceptionStub ?: return JSONObject().put("error", "Not connected.")
        val resp = stub.analyzeFrame(
            Tracking.AnalyzeFrameRequest.newBuilder()
                .setImageData(com.google.protobuf.ByteString.copyFrom(frame))
                .addOps(Tracking.AnalysisOp.DETECT)
                .setPrompt(prompt)
                .build()
        )
        if (resp.detectionsCount == 0) return JSONObject().put("found", false)
        val best = resp.detectionsList.maxByOrNull { it.score }!!
        return JSONObject().put("found", true).put("score", best.score).put("box_xyxy", org.json.JSONArray(best.boxXyxyList))
    }

    private suspend fun toolCheckObstacle(): JSONObject {
        val frame = latestFrame() ?: return JSONObject().put("error", "No frame available.")
        val stub = grpc.perceptionStub ?: return JSONObject().put("error", "Not connected.")
        val resp = stub.analyzeFrame(
            Tracking.AnalyzeFrameRequest.newBuilder()
                .setImageData(com.google.protobuf.ByteString.copyFrom(frame))
                .addOps(Tracking.AnalysisOp.DEPTH)
                .build()
        )
        return JSONObject()
            .put("obstacle_detected", resp.obstacle.detected)
            .put("distance", resp.obstacle.distanceM)
            .put("description", resp.obstacle.description)
    }

    // ── Reading ──────────────────────────────────────────────────────────

    // ── Live reading — continuous capture/OCR/speak, decoupled from
    // correction ─────────────────────────────────────────────────────────
    // Kotlin port of gt.py's ReadingPipeline (the "Live Reading" tab) —
    // THREE decoupled stages instead of one call-and-return tool:
    //   capture loop --> ocrChannel --> OCR worker --> correctionChannel --> correction worker
    // - Capture: every LIVE_READING_INTERVAL_MS, grabs the current frame
    //   (via the SAME acquireSharpFrame() blur skip/retry the one-shot
    //   scan_current_view() uses) and sends it to the OCR worker. Runs on
    //   its own fixed cadence, never blocked by how long OCR/correction
    //   are taking — a still-blurry frame after retries waits
    //   LIVE_READING_BLUR_GIVEUP_MS before the next attempt instead of the
    //   normal interval (mirrors gt.py's BLUR_GIVEUP_WAIT_S).
    // - OCR worker: drains ocrChannel ONE AT A TIME (never starts a new OCR
    //   call until the previous one's response is back) — OcrClient, then
    //   OcrBlockFilters.integrateRawBlock() against RAW text. "new"/
    //   "stitched" are spoken IMMEDIATELY (raw text — correction hasn't
    //   run yet); "new"/"stitched"/"updated" all also get queued onto the
    //   correction worker (shared with the one-shot scan_current_view()
    //   path — see the class-level correctionChannel/correctionWorkerJob).
    private var liveReadingCaptureJob: Job? = null
    private var liveReadingOcrJob: Job? = null
    private var liveReadingOcrChannel: Channel<ByteArray>? = null

    // Sentence-level reading pipeline — see speakSentencesFrom()/
    // interruptReadingForUserTurn()/toolContinueReading() below. Position
    // tracking lives in state.readingCursorSentenceIndex (a persistent
    // cursor into the full, possibly-growing buffer), not a frozen
    // "remaining sentences" snapshot — see that field's own doc comment.
    // Playback itself is readingTts (ReadingTtsPlayer, on-device
    // TextToSpeech) — no local Job/coroutine needed for it any more, since
    // ReadingTtsPlayer manages its own queueing/callbacks internally.

    private fun stopLiveReadingPipeline() {
        liveReadingCaptureJob?.cancel(); liveReadingCaptureJob = null
        liveReadingOcrJob?.cancel(); liveReadingOcrJob = null
        liveReadingOcrChannel?.close(); liveReadingOcrChannel = null
    }

    private fun toolStartLiveReading(): JSONObject {
        ensureReadingMode()

        val ocrChannel = Channel<ByteArray>(Channel.UNLIMITED)
        liveReadingOcrChannel = ocrChannel

        liveReadingCaptureJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                try {
                    val blur = acquireSharpFrame()
                    if (blur.skipped) {
                        delay(LIVE_READING_BLUR_GIVEUP_MS)
                        continue
                    }
                    blur.jpeg?.let { ocrChannel.trySend(it) }
                } catch (e: Exception) {
                    Log.w(TAG, "[live-reading] capture cycle failed: ${e.message}")
                }
                delay(LIVE_READING_INTERVAL_MS)
            }
        }

        liveReadingOcrJob = scope.launch(Dispatchers.IO) {
            for (frame in ocrChannel) {
                try {
                    sendVideoFrame(frame) // same visual-context feed toolScanCurrentView() gives Gemini
                    onOcrFrame?.invoke(frame)
                    val ocrResult = ocrClient.analyze(frame)
                    if (ocrResult.text.isBlank()) continue
                    val (kind, block) = OcrBlockFilters.integrateRawBlock(state.readingBlocks, ocrResult.text)
                    if (block != null && kind != "duplicate") {
                        // Only cue completion for genuinely NEW material —
                        // unlike the one-shot toolScanCurrentView() (which
                        // pops on every explicit user-triggered scan
                        // regardless of outcome), this loop runs
                        // unattended every LIVE_READING_INTERVAL_MS and
                        // would otherwise pop constantly while the camera
                        // just holds steady on an already-seen page.
                        if (geminiCorrectionClient != null) {
                            val blockId = block.id
                            val rawText = block.raw
                            launch(Dispatchers.IO) {
                                val langCode = detectLanguageCode(rawText)
                                correctionChannel.send(CorrectionJob(blockId, rawText, langCode, onComplete = { playScanCompleteCue() }))
                            }
                        } else {
                            playScanCompleteCue()
                        }
                        // Speak from the cursor to the current end of the
                        // buffer (not just this one block's own raw text) —
                        // the same "speak whatever's unread so far" path
                        // read_aloud/continue_reading use, so live reading's
                        // automatic announcements and a manual resume can
                        // never disagree about what's already been spoken.
                        if (kind == "new" || kind == "stitched") {
                            speakUnreadBuffer()
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "[live-reading] OCR cycle failed: ${e.message}")
                }
            }
        }

        return JSONObject().put("status", "live_reading_started").put("label", state.readingLabel)
    }

    private fun toolStopLiveReading(): JSONObject {
        stopLiveReadingPipeline()
        return JSONObject().put("status", "live_reading_stopped").put("chars", state.readingBufferText().length)
    }

    /** Splits reading-mode text at sentence boundaries — the same
     * `(?<=[.!?])\s+` lookbehind splitChunks() already uses below, just
     * without repacking into ~500-char groups (each element here is one
     * sentence, the actual unit speakSentences() synthesizes/plays). */
    private fun splitIntoSentences(text: String): List<String> =
        text.split(Regex("(?<=[.!?])\\s+")).map { it.trim() }.filter { it.isNotEmpty() }

    /** All sentences in the current (possibly still-growing) reading
     * buffer — the single source of truth `state.readingCursorSentenceIndex`
     * is an index into. */
    private fun currentBufferSentences(): List<String> = splitIntoSentences(state.readingBufferText())

    /** Speaks everything from the cursor to the current end of the buffer —
     * the shared "read whatever's unread so far" behavior used by
     * read_aloud, live reading's automatic announcements, and
     * continue_reading alike, so they can never disagree about what's
     * already been spoken. No-ops if the cursor has already caught up to
     * the end. */
    private fun speakUnreadBuffer() {
        val all = currentBufferSentences()
        val start = state.readingCursorSentenceIndex.coerceIn(0, all.size)
        if (start >= all.size) return
        speakSentencesFrom(start, all.subList(start, all.size))
    }

    /** Hands [sentences] off to the on-device ReadingTtsPlayer (native
     * android.speech.tts.TextToSpeech — see that class's own doc comment
     * for the bounded-lookahead queueing design). [globalStartIndex] is
     * [sentences][0]'s position in `currentBufferSentences()`; after each
     * sentence actually finishes playing (or the session is stopped early),
     * `state.readingCursorSentenceIndex` is advanced to match exactly how
     * far playback got — a single persistent cursor, continuously updated,
     * rather than a frozen "remaining sentences" snapshot captured only on
     * interruption, so it stays correct even as live reading keeps growing
     * the buffer underneath it.
     *
     * Fire-and-forget and non-blocking (ReadingTtsPlayer.speakFrom() returns
     * immediately; playback/queueing happens via its own callbacks) — this
     * is why toolReadAloud()/the live-reading OCR worker never await the
     * "speak" step itself, only kick it off; see CLAUDE.md's "Continuous
     * VAD-gated listening" note on why this matters (Gemini Live's function
     * calling is synchronous-only — a tool call that blocked until a whole
     * page finished speaking would prevent Gemini from reacting to the
     * user's interrupting speech at all). */
    private fun speakSentencesFrom(globalStartIndex: Int, sentences: List<String>) {
        Log.d(TAG, "[reading] speakSentencesFrom(globalStartIndex=$globalStartIndex, ${sentences.size} sentence(s))")
        readingTts.speakFrom(
            globalStartIndex,
            sentences,
            onSentenceDone = { newCursor ->
                state.readingCursorSentenceIndex = newCursor
                Log.d(TAG, "[reading] sentence finished, cursor now $newCursor")
            },
            onAllDone = { Log.d(TAG, "[reading] speakSentencesFrom finished, cursor now ${state.readingCursorSentenceIndex}") },
        )
    }

    /** Called by LiveAssistantService the instant the VAD registers a user
     * utterance (onSpeechEnd) — stops any in-progress reading immediately
     * so Gemini's response starts clean. No-ops if nothing was reading. See
     * continue_reading()/toolContinueReading() for resuming.
     *
     * Unlike the old gRPC-PCM design, this is a plain synchronous call —
     * android.speech.tts.TextToSpeech.stop() halts audio and clears its
     * queue immediately, no coroutine cancellation/join race to guard
     * against any more (ReadingTtsPlayer owns its own callback-driven state
     * entirely; there's no separate AudioTrack this needs to flush, since
     * reading-mode TTS no longer shares StreamingAudioPlayer with Gemini's
     * own voice at all). */
    fun interruptReadingForUserTurn() {
        readingTts.stop()
    }

    /** continue_reading tool — resumes from state.readingCursorSentenceIndex
     * against the CURRENT buffer (which may have grown via live reading
     * since the interruption, unlike the old frozen-snapshot design).
     * Deliberately never auto-triggered (only resumes on this explicit tool
     * call) — per the user's requirement that reading only continues if
     * they ask for it. */
    private fun toolContinueReading(): JSONObject {
        val all = currentBufferSentences()
        val start = state.readingCursorSentenceIndex.coerceIn(0, all.size)
        if (start >= all.size) {
            return JSONObject().put("status", "nothing_to_continue")
        }
        speakSentencesFrom(start, all.subList(start, all.size))
        return JSONObject().put("status", "continuing")
    }

    /** One OCR pass, paced by "wait for the blur check/OCR response, don't
     * fire on a fixed timer regardless of frame quality" — before spending
     * an OCR call, checks the current frame's sharpness (see
     * CameraManager.computeSharpness()); if it's blurry, re-samples up to
     * BLUR_MAX_RETRIES times (waiting BLUR_RETRY_WAIT_MS between each,
     * standing in for "let the live camera produce a fresher frame") before
     * giving up on this cycle entirely rather than OCR'ing known-bad input. */
    private data class SharpFrameResult(val jpeg: ByteArray?, val sharpness: Double?, val attempts: Int, val skipped: Boolean)

    private suspend fun acquireSharpFrame(): SharpFrameResult {
        // Full-resolution edge-device path, tried first when available — a
        // real still capture is worth waiting a bit for; OCR_FULL_RES_
        // TIMEOUT_MS bounds how long, so a lost request/response (or a
        // Local/non-edge session, where ocrFrameFlow is always null) falls
        // straight through to the existing live-stream-frame path below
        // instead of hanging.
        if (ocrFrameFlow != null) {
            requestOcrFrame()
            val fullRes = try {
                withTimeoutOrNull(OCR_FULL_RES_TIMEOUT_MS) { ocrFrameFlow.first() }
            } catch (e: Exception) {
                Log.w(TAG, "[reading] full-res OCR frame request failed: ${e.message}")
                null
            }
            if (fullRes != null) return SharpFrameResult(fullRes, null, 0, false)
            Log.w(TAG, "[reading] full-res OCR frame request timed out after ${OCR_FULL_RES_TIMEOUT_MS}ms, falling back to live-stream frame")
        }
        if (blurSharpnessThreshold <= 0.0) {
            val f = latestFrame() ?: return SharpFrameResult(null, null, 0, false)
            return SharpFrameResult(f, null, 0, false)
        }
        var sample = latestFrameWithSharpness() ?: return SharpFrameResult(null, null, 0, true)
        var attempts = 0
        while (sample.second < blurSharpnessThreshold && attempts < BLUR_MAX_RETRIES) {
            attempts++
            delay(BLUR_RETRY_WAIT_MS)
            sample = latestFrameWithSharpness() ?: return SharpFrameResult(null, sample.second, attempts, true)
        }
        val skipped = sample.second < blurSharpnessThreshold
        return SharpFrameResult(if (skipped) null else sample.first, sample.second, attempts, skipped)
    }

    private suspend fun toolScanCurrentView(): JSONObject {
        ensureReadingMode()
        val blur = acquireSharpFrame()
        if (blur.skipped) {
            val retries = blur.attempts
            Log.d(TAG, "[reading] scan skipped — too blurry after $retries retries")
            return JSONObject().put("found", false).put("message",
                "Frame too blurry after $retries retr${if (retries == 1) "y" else "ies"} — hold the camera steadier.")
        }
        val frame = blur.jpeg ?: return JSONObject().put("error", "No frame available. Point the camera at the text.")

        // Give Gemini Live actual visual context of what got scanned, not
        // just the extracted OCR text — requested directly by the user.
        // Plain realtimeInput video (no turnComplete, see sendVideoFrame's
        // own contract), same as the walking-mode ambient frame feed — this
        // never forces a spoken response on its own, it's just buffered
        // context for whatever Gemini says next (e.g. if the user then asks
        // "what does this say" / "what am I looking at").
        sendVideoFrame(frame)
        onOcrFrame?.invoke(frame)

        val ocrResult = ocrClient.analyze(frame)
        Log.d(TAG, "[reading] OCR result: ${ocrResult.text.length} char(s)")
        if (ocrResult.text.isBlank()) return JSONObject().put("found", false).put("message", "No text detected in current view.")

        saveDebugFrame?.let { save ->
            try {
                save(annotateOcrFrame(frame, ocrResult.kept, ocrResult.droppedRotation, ocrResult.droppedNoise, ocrResult.droppedBlur))
            } catch (e: Exception) {
                Log.w(TAG, "debug OCR frame save failed: ${e.message}")
            }
        }

        val (kind, block) = OcrBlockFilters.integrateRawBlock(state.readingBlocks, ocrResult.text)
        Log.d(TAG, "[reading] integrateRawBlock -> kind=$kind, blockId=${block?.id}, totalBlocks=${state.readingBlocks.size}")

        // Correction runs async regardless of whether this capture is
        // "spoken" this cycle — "updated" (a longer/cleaner reread) still
        // needs its own fresh correction even though it isn't announced as
        // new text, same as gt.py's kind in ("new","updated","stitched")
        // push condition. playScanCompleteCue() fires once THIS capture's
        // pipeline is actually done — deferred to the correction job's
        // onComplete when one gets queued, played immediately here
        // otherwise (correction disabled, or nothing new to correct at
        // all) since no later completion event will ever arrive for it.
        if (geminiCorrectionClient != null && block != null && kind != "duplicate") {
            val blockId = block.id
            val rawText = block.raw
            scope.launch(Dispatchers.IO) {
                val langCode = detectLanguageCode(rawText)
                correctionChannel.send(CorrectionJob(blockId, rawText, langCode, onComplete = { playScanCompleteCue() }))
            }
        } else {
            playScanCompleteCue()
        }

        if (kind != "new" && kind != "stitched") {
            return JSONObject().put("found", false).put("message", "No new text (already scanned — $kind).")
        }
        val newText = block!!.corrected
        val summary = newText.split(" ").take(30).joinToString(" ") + if (newText.split(" ").size > 30) "..." else ""
        state.pageSummaries.add(summary)
        return JSONObject()
            .put("found", true).put("new_text", newText)
            .put("page_count", state.pageSummaries.size).put("summary", summary)
    }

    private fun toolGetReadingSection(query: String): JSONObject {
        val buffer = state.readingBufferText()
        if (buffer.isEmpty()) return JSONObject().put("error", "No text has been scanned yet. Use scan_current_view() first.")
        if (buffer.length < 1200) return JSONObject().put("text", buffer).put("source", "full_buffer")
        val chunks = splitChunks(buffer)
        val keywords = query.lowercase().split(" ").filter { it.length > 3 }
        val scored = chunks.map { c -> c to keywords.count { kw -> c.lowercase().contains(kw) } }
            .filter { it.second > 0 }.sortedByDescending { it.second }
        if (scored.isNotEmpty()) {
            return JSONObject().put("text", scored.take(2).joinToString("\n\n") { it.first }).put("source", "keyword_match")
        }
        return JSONObject().put("text", chunks.firstOrNull() ?: "").put("source", "first_chunk")
    }

    private fun splitChunks(text: String, size: Int = 500): List<String> {
        val sentences = text.split(Regex("(?<=[.!?])\\s+"))
        val chunks = mutableListOf<String>()
        var current = mutableListOf<String>(); var curLen = 0
        for (raw in sentences) {
            val s = raw.trim()
            if (s.isEmpty()) continue
            if (curLen + s.length > size && current.isNotEmpty()) {
                chunks.add(current.joinToString(" ")); current = mutableListOf(); curLen = 0
            }
            current.add(s); curLen += s.length + 1
        }
        if (current.isNotEmpty()) chunks.add(current.joinToString(" "))
        return chunks.ifEmpty { listOf(text.take(size)) }
    }

    /** The `read_aloud`/"read this to me" trigger — no scope param any
     * more: always (1) scans the current view into the buffer (best
     * effort — a blurry frame or a view with no new text doesn't block
     * reading whatever's already buffered), then (2) speaks the buffer from
     * the start, resetting the cursor to 0 first — UNLESS this capture
     * turned out to be long, per the word-count gate below.
     *
     * Requested directly by the user: a short capture (a label, a sign, a
     * couple sentences — under READ_ALOUD_IMMEDIATE_MAX_WORDS) is read
     * straight away, same as before; a LONGER capture (a full page) is
     * saved to the reading session instead of being spoken in full
     * unprompted — the cursor stays put and Gemini is told to offer to
     * read it rather than dumping a whole page of speech on the user the
     * instant they said "read this". The gate only looks at what THIS scan
     * actually captured (`new_text`), not the whole historical buffer — a
     * scan that found nothing new (duplicate reread) still falls through
     * to the pre-existing "speak whatever's unread so far" behavior below,
     * same fix as the "scope='new' silently did nothing" bug this function
     * already fixed once.
     *
     * Fire-and-forget once the text to speak is known — kicks off
     * speakSentencesFrom() and returns immediately rather than awaiting the
     * whole reading, so Gemini stays free to react to a VAD-registered user
     * turn (interruptReadingForUserTurn()) instead of being blocked inside
     * this synchronous tool call for as long as a whole page takes to read.
     * See speakSentencesFrom()'s own doc comment for why this matters. */
    private suspend fun toolReadAloud(): JSONObject {
        Log.d(TAG, "[reading] read_aloud() called, mode=${state.mode}")
        ensureReadingMode()
        val scanResult = toolScanCurrentView()
        val newText = scanResult.optString("new_text", "")
        val sentences = currentBufferSentences()
        Log.d(TAG, "[reading] read_aloud() buffer has ${sentences.size} sentence(s), ${state.readingBufferText().length} char(s)")
        if (sentences.isEmpty()) return JSONObject().put("error", "No text has been scanned yet.")

        if (newText.isNotBlank()) {
            val wordCount = newText.trim().split(Regex("\\s+")).size
            if (wordCount >= READ_ALOUD_IMMEDIATE_MAX_WORDS) {
                Log.d(TAG, "[reading] read_aloud() captured $wordCount word(s) — storing to session instead of speaking immediately")
                return JSONObject().put("status", "stored_long_text").put("word_count", wordCount)
                    .put("chars", state.readingBufferText().length)
                    .put("message", "Captured a longer passage ($wordCount words) — saved to the reading " +
                        "session instead of reading it all aloud immediately. Tell the user it's ready and " +
                        "offer to read it (call continue_reading), or use get_reading_section for a specific part.")
            }
        }

        state.readingCursorSentenceIndex = 0
        speakSentencesFrom(0, sentences)
        return JSONObject().put("status", "reading_started").put("chars", state.readingBufferText().length)
    }

    private fun toolFlipReadingDirection(): JSONObject {
        state.readingDirection = if (state.readingDirection == "rtl") "ltr" else "rtl"
        return JSONObject().put("direction", state.readingDirection)
    }

    /** The "clear" trigger — stops any in-progress reading, discards the
     * buffer, and exits reading mode. Not a mandatory cleanup step any
     * more (see ensureReadingMode()/stopActiveModes() — switching to
     * another mode already stops live capture and speech on its own) —
     * this is only for an explicit "clear what you've read"/"forget that"
     * request while still in reading mode.
     *
     * Also tells Gemini Live to disregard the reading session's visual
     * context — requested directly by the user ("clear on end reading
     * session, if possible"). The Live API's BidiGenerateContent protocol
     * has no operation to selectively purge already-sent realtimeInput
     * video frames from a session's context (only automatic context-window
     * compression and full session resumption, neither of which is a
     * targeted "forget these specific frames" — checked against this
     * project's own Gemini Live API reference); every scanned/read frame
     * this session sent via sendVideoFrame() (toolScanCurrentView()/the
     * live-reading OCR worker) stays in context regardless. The practical
     * realization of "if possible" is this [SYSTEM] note — a prompted,
     * not a true, clear — same convention this codebase already uses for
     * every other behavior signal Gemini needs to react to. */
    private fun toolExitReadingMode(): JSONObject {
        stopLiveReadingPipeline()
        readingTts.stop()
        val label = state.readingLabel
        val chars = state.readingBufferText().length
        state.mode = "idle"
        state.readingBlocks.clear(); state.pageSummaries.clear(); state.readingLabel = ""
        state.readingCursorSentenceIndex = 0
        reportMode("idle")
        if (chars > 0) {
            sendSystemNote(
                "[SYSTEM] The reading session just ended and its buffer was cleared — disregard any " +
                    "text/pages you saw from it as current context from now on."
            )
        }
        return JSONObject().put("status", "exited").put("label", label).put("chars_discarded", chars)
    }

    // ── Tracking ─────────────────────────────────────────────────────────

    /** [description], when non-blank, is a real bug fix: it used to be
     * declared on the start_tracking tool schema but never actually read
     * here — every call fell back to using [target] itself (often a proper
     * noun/label like "Cutie Pie") as the GroundingDINO detection prompt,
     * which is a poor open-vocab prompt compared to an actual appearance
     * description ("small white fluffy dog"). Now: the detection prompt
     * sent to TrackingBackend.initialize() is [description] when the
     * caller supplied one (typically from get_object_from_memory's result),
     * falling back to [target] otherwise — while [target]/state.
     * trackingTarget/reportMode() still use the plain label for display and
     * server-side status, unaffected. */
    private fun toolStartTracking(target: String, description: String = ""): JSONObject {
        if (target.isBlank()) return JSONObject().put("error", "target required")
        stopActiveModes()
        state.mode = "tracking"; state.trackingTarget = target
        val detectionPrompt = description.ifBlank { target }
        onTrackingStateChanged(true, target, detectionPrompt)
        pixieController.start()  // plays constantly but muted until the target is visible — see updateTrackingPixie()
        trackingAxis = TrackingAxis.HORIZONTAL
        reportMode("tracking", target)
        return JSONObject().put("status", "tracking_started").put("target", target)
    }

    private fun toolStopTracking(): JSONObject {
        state.mode = "idle"; state.trackingTarget = ""
        onTrackingStateChanged(false, "", "")
        pixieController.stop()
        reportMode("idle")
        return JSONObject().put("status", "tracking_stopped")
    }

    // ── Tracking — periodic verbal hand-guidance toward the target ────────
    //
    // Real bug, found and removed: this used to run its own periodic
    // (every TRACKING_GUIDANCE_INTERVAL_MS) spoken-guidance trigger,
    // UNCONDITIONALLY — it sent the current frame and asked Gemini to
    // guide the user's hand toward the target regardless of whether a
    // hand was actually visible anywhere in frame, which is exactly what
    // caused Gemini to narrate nonsense directional guidance ("move left",
    // etc.) at a hand that wasn't there. This duplicated — worse, without
    // the visibility gate — a SECOND, ALREADY-CORRECT mechanism living in
    // LiveAssistantService's own frame collector (`trackingGuidanceLastAtMs`/
    // `trackingGuidanceIntervalMs`), which only fires once BOTH the target
    // object AND a detected hand box are actually visible together in the
    // same frame, and sends real box coordinates instead of a raw image
    // (cheaper, and gives Gemini exact spatial data instead of asking it
    // to eyeball an image). That mechanism is now the ONLY periodic
    // tracking-guidance trigger — this whole duplicate block was deleted
    // outright, not fixed in place, since the correct version already
    // existed elsewhere. See LiveAssistantService.kt's frame collector.

    /** Steers Pixie to move the user's HAND toward the currently-tracked
     * object — NOT to turn/look toward it. Real bug fix, from a direct
     * user report: this used to compute the object's position relative to
     * the FRAME CENTER (HrtfBeacon.directionFromBox), which only guides
     * "look/turn this way to center the object" — the whole point of
     * tracking mode is hand guidance, and the hand's actual position was
     * never part of that computation at all. Now uses
     * HrtfBeacon.directionBetweenPoints(object, HAND) instead — the cue is
     * the object's position relative to WHERE THE HAND CURRENTLY IS, so
     * "move right" genuinely means "move your hand right," regardless of
     * where in the frame the hand happens to be. [handCenterX]/[handCenterY]
     * come from MediaPipe hand detection (LiveAssistantService's frame
     * collector, same frame as the object track) — mutes whenever either
     * the object or a hand isn't currently visible, since there's nothing
     * meaningful to steer without both.
     *
     * Two-phase state machine (confirmed with the user): phase HORIZONTAL
     * (left/right) runs first; once the target is centered horizontally
     * relative to the hand (within TRACKING_H_DEADZONE_DEG), phase
     * VERTICAL (up/down, screen-space — NOT depth/distance) takes over. If
     * it drifts back out past the SAME horizontal deadzone while in
     * VERTICAL, phase drops back to HORIZONTAL first — re-centering always
     * takes priority over the forward/back cue. Volume follows the same
     * "quiet when correctly positioned" convention as guiding/walking's own
     * Pixie cue (see steerAlongMainPath()). */
    fun updateTrackingPixie(
        objectVisible: Boolean, objectCenterX: Float, objectCenterY: Float,
        handVisible: Boolean, handCenterX: Float, handCenterY: Float,
        frameWidth: Int, frameHeight: Int,
    ) {
        if (!objectVisible || !handVisible || frameWidth <= 0 || frameHeight <= 0) {
            pixieController.mute()
            return
        }
        val dir = HrtfBeacon.directionBetweenPoints(objectCenterX, objectCenterY, handCenterX, handCenterY, frameWidth, frameHeight)

        if (trackingAxis == TrackingAxis.VERTICAL && abs(dir.azimuthDeg) > TRACKING_H_DEADZONE_DEG) {
            trackingAxis = TrackingAxis.HORIZONTAL  // slipped back off horizontally — re-center first
        }
        if (trackingAxis == TrackingAxis.HORIZONTAL && abs(dir.azimuthDeg) <= TRACKING_H_DEADZONE_DEG) {
            trackingAxis = TrackingAxis.VERTICAL
        }

        when (trackingAxis) {
            TrackingAxis.HORIZONTAL -> {
                pixieController.move(if (dir.azimuthDeg < 0) PixiePoint.LEFT else PixiePoint.RIGHT)
                pixieController.setVolume(
                    gainForDeviation(dir.azimuthDeg, TRACKING_H_DEADZONE_DEG, TRACKING_H_RAMP_END_DEG)
                )
            }
            TrackingAxis.VERTICAL -> {
                pixieController.move(if (dir.elevationDeg > 0) PixiePoint.UP else PixiePoint.DOWN)
                pixieController.setVolume(
                    gainForDeviation(dir.elevationDeg, TRACKING_V_DEADZONE_DEG, TRACKING_V_RAMP_END_DEG)
                )
            }
        }
    }

    /** Runs GroundingDINO (open-vocab, prompted with [prompt]) against
     * [frame] and returns every detection, sorted by score — thin wrapper
     * around PerceptionService.AnalyzeFrame(DETECT), same RPC
     * toolRunDetection() already uses. Empty (not an error) if not
     * connected or the call fails — callers treat that as "nothing found". */
    private suspend fun detectAll(prompt: String, frame: ByteArray): List<Tracking.Detection> {
        val stub = grpc.perceptionStub ?: return emptyList()
        return try {
            stub.analyzeFrame(
                Tracking.AnalyzeFrameRequest.newBuilder()
                    .setImageData(com.google.protobuf.ByteString.copyFrom(frame))
                    .addOps(Tracking.AnalysisOp.DETECT)
                    .setPrompt(prompt)
                    .build()
            ).detectionsList
        } catch (e: Exception) { emptyList() }
    }

    /** DINOv2 re-ID embedding for one box in [frame] — same PerceptionService.
     * AnalyzeFrame(EMBED) op the EMBED-only path already exposes, avoiding a
     * separate TrackingService.GetEmbedding round trip. */
    private suspend fun embedBox(box: List<Float>, frame: ByteArray): FloatArray? {
        val stub = grpc.perceptionStub ?: return null
        return try {
            val resp = stub.analyzeFrame(
                Tracking.AnalyzeFrameRequest.newBuilder()
                    .setImageData(com.google.protobuf.ByteString.copyFrom(frame))
                    .addOps(Tracking.AnalysisOp.EMBED)
                    .addAllBoxXyxy(box)
                    .build()
            )
            if (resp.embeddingCount == 0) null else resp.embeddingList.toFloatArray()
        } catch (e: Exception) { null }
    }

    /** Resolves a POSSESSIVE reference ("my water bottle") to a saved
     * label — first via text semantic search (unchanged), then, if the
     * label has a stored visual (DINOv2) reference and a current frame is
     * available, confirms/disambiguates against what's ACTUALLY in view
     * right now: detect every box GroundingDINO finds for the stored
     * description, embed each, and compare against the label's own stored
     * embedding(s) via cosine similarity (OBJECT_MATCH_MIN_SIM=0.55).
     * Deliberately does NOT guess between two near-equally-good instances
     * (e.g. two bottles on a table) — within AMBIGUITY_MARGIN of each
     * other, this returns ambiguous=true instead of picking one, so the
     * caller can ask the user to clarify rather than silently tracking the
     * wrong physical object. */
    private suspend fun toolGetObjectFromMemory(query: String): JSONObject {
        // Try a direct LABEL-NAME match first — a real, reported bug fix:
        // a proper-noun query like "Cutie Patootie" (the object's own
        // saved name) has near-zero semantic-embedding similarity to its
        // own stored DESCRIPTION text ("small yellow plush bird..."), so
        // the embedding search below routinely missed the exact case of
        // "the query just IS the label" entirely, reporting found=false
        // for an object that had in fact just been saved. See
        // LocalMemoryStore.findLabelsMatching()'s own doc comment.
        val labelMatches = memoryStore.findLabelsMatching(query)
        val m: MemoryMatch = when {
            labelMatches.size == 1 -> {
                val label = labelMatches[0]
                MemoryMatch(label, memoryStore.getFullText(label), 1f)
            }
            labelMatches.size > 1 -> {
                return JSONObject().put("found", true).put("ambiguous", true)
                    .put("candidate_count", labelMatches.size)
                    .put("message", "Multiple saved memories match '$query' by name (${labelMatches.joinToString()}) — ask the user which one they mean.")
            }
            else -> {
                val perceptionStub = grpc.perceptionStub ?: return JSONObject().put("error", "Not connected.")
                val queryVec = try {
                    perceptionStub.embed(Tracking.EmbedRequest.newBuilder().setText(query).build()).vectorList.toFloatArray()
                } catch (e: Exception) { return JSONObject().put("error", "Embed failed: ${e.message}") }
                val matches = memoryStore.queryGlobal(queryVec, topK = 1).filter { it.score > 0.3f }
                if (matches.isEmpty()) return JSONObject().put("found", false)
                matches.first()
            }
        }

        val frame = latestFrame()
        if (frame == null || !memoryStore.hasObjectEmbeddings(m.label)) {
            // No frame to verify against, or this label was never visually
            // captured (remember_object() saved a description only) — fall
            // back to the text-only result, same as before this change.
            return JSONObject().put("found", true).put("label", m.label).put("description", m.text).put("score", m.score)
        }

        val detections = detectAll(m.text, frame)
        if (detections.isEmpty()) {
            return JSONObject().put("found", true).put("label", m.label).put("description", m.text)
                .put("score", m.score).put("visual_confirmed", false)
        }

        data class Candidate(val boxXyxy: List<Float>, val sim: Float)
        val candidates = detections.mapNotNull { d ->
            val vec = embedBox(d.boxXyxyList, frame) ?: return@mapNotNull null
            Candidate(d.boxXyxyList, memoryStore.bestObjectSimilarity(m.label, vec))
        }.sortedByDescending { it.sim }

        val best = candidates.firstOrNull()
        if (best == null || best.sim < OBJECT_MATCH_MIN_SIM) {
            return JSONObject().put("found", true).put("label", m.label).put("description", m.text)
                .put("score", m.score).put("visual_confirmed", false)
        }

        if (candidates.size > 1 && (best.sim - candidates[1].sim) < AMBIGUITY_MARGIN) {
            return JSONObject().put("found", true).put("label", m.label).put("description", m.text)
                .put("ambiguous", true).put("candidate_count", candidates.size)
                .put("message", "Multiple similar-looking objects match '${m.label}' in view — ask the user to point more directly at the one they mean, or describe which one (e.g. left/right/color), before tracking.")
        }

        return JSONObject().put("found", true).put("label", m.label).put("description", m.text)
            .put("visual_confirmed", true).put("similarity", best.sim)
            .put("box_xyxy", org.json.JSONArray(best.boxXyxy))
    }

    /** Answers "is this my <label>?" — detects every box GroundingDINO
     * matches against the label's stored description, picks the LARGEST
     * bounding box in view (the object the user is most likely pointing
     * the camera at, not necessarily the highest-scoring detection), embeds
     * it, and compares against the label's stored visual reference(s).
     * Stricter threshold than get_object_from_memory's disambiguation
     * (IS_THIS_OBJECT_MIN_SIM=0.65) since this is a direct yes/no
     * confirmation, not a best-of-several-candidates pick. */
    private suspend fun toolIsThisObject(label: String): JSONObject {
        if (label.isBlank()) return JSONObject().put("error", "label required")
        val description = memoryStore.getFullText(label)
        if (description.isBlank()) return JSONObject().put("error", "No memory found for label '$label'.")
        if (!memoryStore.hasObjectEmbeddings(label)) {
            return JSONObject().put("error", "No visual reference stored for '$label' yet — it needs to have been visible when remembered.")
        }
        if (grpc.perceptionStub == null) return JSONObject().put("error", "Not connected.")
        val frame = latestFrame() ?: return JSONObject().put("error", "No frame available.")

        val detections = detectAll(description, frame)
        if (detections.isEmpty()) return JSONObject().put("is_match", false).put("label", label).put("reason", "no_matching_object_in_view")

        val biggest = detections.maxByOrNull { d ->
            val b = d.boxXyxyList
            (b[2] - b[0]) * (b[3] - b[1])
        }!!

        val vec = embedBox(biggest.boxXyxyList, frame)
            ?: return JSONObject().put("is_match", false).put("label", label).put("reason", "embedding_failed")

        val sim = memoryStore.bestObjectSimilarity(label, vec)
        return JSONObject().put("is_match", sim >= IS_THIS_OBJECT_MIN_SIM).put("label", label)
            .put("similarity", sim).put("box_xyxy", org.json.JSONArray(biggest.boxXyxyList))
    }

    /** Q&A helper for "which one is my X" / "what's on the table, any of my
     * belongings" — distinct from start_tracking/get_object_from_memory:
     * this ANSWERS a question about what's currently in view, it doesn't
     * initiate tracking. Requested directly by the user.
     *
     * With [targetsArg] given: each target string is resolved to saved
     * memory label(s) via findLabelsMatching(), and each matched label
     * gets its OWN dedicated GroundingDINO call against its own stored
     * description — simple, no ambiguity about which detection belongs to
     * which label, and cheap enough for the handful of targets a real
     * question names. A target matching no saved label at all is reported
     * separately (not_in_memory), not silently dropped.
     *
     * With [targetsArg] null/empty: every saved label that has a stored
     * visual reference is checked via ONE combined multi-phrase
     * GroundingDINO call (all of their own descriptions joined by " . ",
     * same phrase-grounding convention frame_extractor/tagging.py already
     * established for multi-tag prompts) — efficient for however many
     * labels are saved, rather than one call per label. Each returned
     * detection is attributed back to whichever candidate label's own
     * description contains (or is contained by) its decoded label text —
     * safe because GroundingDINO only ever decodes a literal span of
     * whichever prompt phrase it matched.
     *
     * Per-label verdict (classifySearchObjectMatch()): best embedding
     * similarity >= SEARCH_OBJECTS_CONFIRM_SIM (0.5) -> "found" (box
     * included); >= SEARCH_OBJECTS_RESEMBLE_SIM (0.3) only -> "resembles"
     * (a possible, not confident, match); below that (or no detection at
     * all) -> omitted from the response entirely — a long "not found" list
     * for an all-labels search_objects() call would be noise, not help. */
    private suspend fun toolSearchObjects(targetsArg: org.json.JSONArray?): JSONObject {
        val frame = latestFrame() ?: return JSONObject().put("error", "No frame available.")
        if (grpc.perceptionStub == null) return JSONObject().put("error", "Not connected.")

        val results = org.json.JSONArray()
        val notInMemory = org.json.JSONArray()

        if (targetsArg != null && targetsArg.length() > 0) {
            for (i in 0 until targetsArg.length()) {
                val target = targetsArg.optString(i, "")
                if (target.isBlank()) continue
                val labels = memoryStore.findLabelsMatching(target)
                if (labels.isEmpty()) {
                    notInMemory.put(target)
                    continue
                }
                for (label in labels) {
                    // NOTE: previously required hasObjectEmbeddings(label) here,
                    // which silently dropped any label whose remember_object()
                    // call never captured a visual reference (e.g. detectAll()
                    // found nothing at save time — see toolRememberObject()'s
                    // visual_captured flag) — the label was found by
                    // findLabelsMatching() but then vanished with no result and
                    // no notInMemory entry either. bestSearchObjectMatch() now
                    // falls back to a description-only (unconfirmed) match when
                    // there's no stored embedding to re-ID against.
                    val description = memoryStore.getFullText(label)
                    if (description.isBlank()) continue
                    val verdict = bestSearchObjectMatch(label, description, frame)
                    if (verdict != null) results.put(verdict)
                }
            }
        } else {
            data class Candidate(val label: String, val description: String)
            val candidates = memoryStore.listLabels()
                .mapNotNull { label ->
                    val text = memoryStore.getFullText(label)
                    if (text.isBlank()) null else Candidate(label, text)
                }
            if (candidates.isNotEmpty()) {
                val combinedPrompt = candidates.joinToString(" . ") { it.description.trim().trimEnd('.').lowercase() }
                val detections = detectAll(combinedPrompt, frame)
                for (candidate in candidates) {
                    val desc = candidate.description.trim().lowercase()
                    val matches = detections.filter { d ->
                        val detLabel = d.label.trim().lowercase()
                        detLabel.isNotEmpty() && (desc.contains(detLabel) || detLabel.contains(desc))
                    }
                    if (matches.isEmpty()) continue
                    val verdict = if (memoryStore.hasObjectEmbeddings(candidate.label)) {
                        var bestSim = -1f
                        var bestBox: List<Float>? = null
                        for (d in matches) {
                            val vec = embedBox(d.boxXyxyList, frame) ?: continue
                            val sim = memoryStore.bestObjectSimilarity(candidate.label, vec)
                            if (sim > bestSim) { bestSim = sim; bestBox = d.boxXyxyList }
                        }
                        classifySearchObjectMatch(candidate.label, bestSim, bestBox)
                    } else {
                        unconfirmedMatch(candidate.label, matches)
                    }
                    if (verdict != null) results.put(verdict)
                }
            }
        }

        return JSONObject().put("found", results.length() > 0).put("results", results)
            .apply { if (notInMemory.length() > 0) put("not_in_memory", notInMemory) }
    }

    /** One dedicated GroundingDINO call for a single already-resolved
     * memory [label] — picks whichever detection best matches the label's
     * own stored visual embedding(s) (highest cosine similarity wins, same
     * "best of several candidates" convention get_object_from_memory()
     * already uses), then classifies it via classifySearchObjectMatch(). */
    private suspend fun bestSearchObjectMatch(label: String, description: String, frame: ByteArray): JSONObject? {
        val detections = detectAll(description, frame)
        if (detections.isEmpty()) return null
        if (!memoryStore.hasObjectEmbeddings(label)) return unconfirmedMatch(label, detections)
        var bestSim = -1f
        var bestBox: List<Float>? = null
        for (d in detections) {
            val vec = embedBox(d.boxXyxyList, frame) ?: continue
            val sim = memoryStore.bestObjectSimilarity(label, vec)
            if (sim > bestSim) { bestSim = sim; bestBox = d.boxXyxyList }
        }
        return classifySearchObjectMatch(label, bestSim, bestBox)
    }

    /** Applies search_objects()'s found(>=0.5)/resembles(>=0.3)/omitted
     * thresholds — see toolSearchObjects()'s own doc comment. */
    private fun classifySearchObjectMatch(label: String, sim: Float, box: List<Float>?): JSONObject? {
        if (box == null || sim < SEARCH_OBJECTS_RESEMBLE_SIM) return null
        val match = if (sim >= SEARCH_OBJECTS_CONFIRM_SIM) "found" else "resembles"
        return JSONObject().put("label", label).put("match", match)
            .put("similarity", sim).put("box_xyxy", org.json.JSONArray(box))
    }

    /** Fallback for a label with a stored text description but no DINOv2
     * visual reference to re-ID against (remember_object() never captured
     * one) — reports the best-scoring GroundingDINO detection matching the
     * label's own description as an unconfirmed "found" rather than
     * silently dropping the label from every search_objects() result. */
    private fun unconfirmedMatch(label: String, detections: List<Tracking.Detection>): JSONObject? {
        val best = detections.maxByOrNull { it.score } ?: return null
        return JSONObject().put("label", label).put("match", "found")
            .put("box_xyxy", org.json.JSONArray(best.boxXyxyList))
            .put("note", "no stored visual reference for this label; matched by description only, not re-ID confirmed")
    }

    // ── Memory ───────────────────────────────────────────────────────────

    private suspend fun toolQueryMemory(question: String): JSONObject {
        val stub = grpc.perceptionStub ?: return JSONObject().put("error", "Not connected.")
        val queryVec = try {
            stub.embed(Tracking.EmbedRequest.newBuilder().setText(question).build()).vectorList.toFloatArray()
        } catch (e: Exception) { return JSONObject().put("error", "Embed failed: ${e.message}") }
        val semanticMatches = memoryStore.queryGlobal(queryVec, topK = 5).filter { it.score > 0.5f }
        // Same label-name-match boost as toolGetObjectFromMemory() — a
        // question that directly names a saved label ("what does Cutie
        // Patootie look like?") deserves that exact memory even if its
        // stored description text has little semantic overlap with the
        // question's own wording. Prepended (label hits are a stronger
        // signal than a cosine score) and deduped against the semantic
        // results by label.
        val labelHits = memoryStore.findLabelsMatching(question)
            .map { label -> MemoryMatch(label, memoryStore.getFullText(label), 1f) }
        val matches = labelHits + semanticMatches.filter { sem -> labelHits.none { it.label == sem.label } }
        val arr = org.json.JSONArray()
        matches.forEach { arr.put(JSONObject().put("label", it.label).put("text", it.text).put("score", it.score)) }
        return JSONObject().put("results", arr).put("found", matches.isNotEmpty())
    }

    private suspend fun toolSaveMemory(label: String, note: String): JSONObject {
        if (label.isBlank() || note.isBlank()) return JSONObject().put("error", "label and note required")
        memoryStore.append(label, note, source = "note")
        embedAndStore(label, note)
        return JSONObject().put("status", "saved").put("label", label)
    }

    /** Saves whatever's currently in the reading buffer (state.
     * readingBufferText() — CORRECTED text, whatever's landed so far)
     * directly, instead of routing through save_memory(label, note) with
     * Gemini retyping the scanned text into `note` itself. Avoids Gemini
     * having to faithfully reproduce a whole scanned document into a
     * function-call argument (wasteful, and risks truncation/paraphrasing
     * for anything beyond a short note) — works identically regardless of
     * whether the buffer was filled by one-shot scan_current_view() calls
     * or a live-reading session. */
    private suspend fun toolSaveReadingBuffer(label: String): JSONObject {
        if (label.isBlank()) return JSONObject().put("error", "label required")
        val text = state.readingBufferText()
        if (text.isBlank()) return JSONObject().put("error", "Reading buffer is empty — nothing to save.")
        memoryStore.append(label, text, source = "reading")
        embedAndStore(label, text)
        return JSONObject().put("status", "saved").put("label", label).put("chars", text.length)
    }

    /** Saves a text description AND, if the object is actually visible in
     * the current frame, a DINOv2 visual embedding of it — a separate index
     * from the text embedding (see LocalMemoryStore's objEmbPath) used
     * later by get_object_from_memory()/is_this_object() for real visual
     * re-ID against the camera view, not just a text-label match. Visual
     * capture is best-effort: a failed/low-confidence detection still saves
     * the text description, just without a visual reference (those two
     * later tools then degrade to text-only/error, respectively — see
     * their own comments).
     *
     * [description] (Gemini Live's own composed text, passed in by the
     * caller) is only ever a FALLBACK now — when a frame is available and
     * geminiObjectDescriptionClient is configured, a dedicated one-shot
     * vision call regenerates a terser, more consistent description
     * (color/shape/entity-type only, see that class's own doc comment)
     * from the ACTUAL image and that's what gets stored/embedded/used as
     * the detection prompt below — requested directly by the user after
     * Gemini Live's own free-form descriptions ran long and scene-y. */
    private suspend fun toolRememberObject(label: String, description: String): JSONObject {
        if (label.isBlank() || description.isBlank()) return JSONObject().put("error", "label and description required")
        val frame = latestFrame()

        val finalDescription = if (frame != null && geminiObjectDescriptionClient != null) {
            try {
                geminiObjectDescriptionClient.describe(frame, label)
            } catch (e: Exception) {
                Log.w(TAG, "geminiObjectDescriptionClient.describe failed for '$label', falling back: ${e.message}")
                description
            }
        } else description

        memoryStore.append(label, finalDescription, source = "object_description")
        embedAndStore(label, finalDescription)

        var visualCaptured = false
        if (frame != null) {
            val best = detectAll(finalDescription, frame).maxByOrNull { it.score }
            if (best != null) {
                val vec = embedBox(best.boxXyxyList, frame)
                if (vec != null) {
                    memoryStore.addObjectEmbedding(label, vec)
                    visualCaptured = true
                }
            }
        }
        return JSONObject().put("status", "remembered").put("label", label)
            .put("description", finalDescription).put("visual_captured", visualCaptured)
    }

    private suspend fun embedAndStore(label: String, text: String) {
        val stub = grpc.perceptionStub ?: return
        try {
            val vec = stub.embed(Tracking.EmbedRequest.newBuilder().setText(text).build()).vectorList.toFloatArray()
            memoryStore.addEmbedding(label, text, vec)
        } catch (e: Exception) {
            Log.w(TAG, "embedAndStore failed for '$label': ${e.message}")
        }
    }

    private fun toolListMemoryLabels(): JSONObject =
        JSONObject().put("labels", org.json.JSONArray(memoryStore.listLabels()))

    /** Deletes a saved memory label outright — note/description, text
     * embedding index, and visual embedding index, all of it. The user-
     * facing escape hatch for a mislabeled/duplicate save (e.g.
     * remember_object() saving the same physical thing twice under two
     * slightly different names) — "forget X"/"delete the memory for X". */
    private fun toolClearMemory(label: String): JSONObject {
        if (label.isBlank()) return JSONObject().put("error", "label required")
        val deleted = memoryStore.delete(label)
        return JSONObject().put("status", if (deleted) "cleared" else "not_found").put("label", label)
    }

    // ── Guiding (live MappingService-backed navigation) ─────────────────

    private suspend fun toolStartGuiding(destination: String): JSONObject {
        if (destination.isBlank()) return JSONObject().put("error", "destination required")
        stopActiveModes()
        state.mode = "guiding"
        state.guidingDestinationLabel = destination
        state.guidingGoalXz = null
        guidingArrivalAnnounced = false
        pdrStepEstimator.resetAccumulator()
        pdrStepEstimator.start()
        startMappingStream()  // server-planned route + pose, via RTAB-Map — see UpdateMapping's collector
        startLocalAvoidanceTicks()  // path-pursuit beacon steering — see runUnifiedAvoidanceTick()
        startObstacleAheadPolling()  // own fixed-rate depth beep, independent of the mapping stream above
        pixieController.start()
        onGuidanceUpdate("guiding", state.plannedPath)
        reportMode("guiding", destination)
        return JSONObject().put("status", "guiding_started").put("destination", destination)
            .put("note", "Route will be announced once the destination landmark has been located in the live map.")
    }

    private fun toolStopGuiding(): JSONObject {
        stopMappingStream()
        stopLocalAvoidanceTicks()
        stopObstacleAheadPolling()
        pixieController.stop()
        pdrStepEstimator.stop()
        state.mode = "idle"; state.guidingDestinationLabel = ""; state.guidingGoalXz = null
        state.plannedPath = emptyList()
        onGuidanceUpdate("idle", emptyList())
        reportMode("idle")
        return JSONObject().put("status", "guiding_stopped")
    }

    private fun toolGetCurrentLocation(): JSONObject {
        val pose = state.lastMappingPose ?: return JSONObject().put("error", "No location fix yet.")
        return JSONObject().put("x", pose.x).put("z", pose.z)
    }

    private fun startMappingStream() {
        stopMappingStream()
        val stub = grpc.mappingStub ?: return
        // SCAN is the one exception to "drop-to-latest" (see CLAUDE.md's
        // "Drop-to-latest mapping-chunk ingestion" note for why WALKING/
        // GUIDING want freshness over completeness): a scan wants EVERY
        // frame processed, in order, even if the server falls behind —
        // requested directly by the user, who wants scanning to behave
        // like a real work queue (buffer while busy, keep draining after
        // the user stops) rather than silently discarding whatever it
        // didn't get to. UNLIMITED here is what makes that possible on the
        // client side; mapping_servicer.py's matching server-side change
        // (skip the _latest_only_chunks() mailbox for SCAN, let gRPC's own
        // request-iterator queue do the buffering) is what makes it
        // possible end-to-end — this alone would do nothing if the server
        // still dropped chunks upstream.
        val channel = if (state.mode == "scanning") {
            Channel<Tracking.MappingChunk>(Channel.UNLIMITED)
        } else {
            Channel<Tracking.MappingChunk>(Channel.CONFLATED)
        }
        mappingChunkChannel = channel
        mappingJob = scope.launch(Dispatchers.IO) {
            try {
                stub.updateMapping(channel.receiveAsFlow()).collect { update ->
                    if (update.resetOccurred) {
                        // pure-walking total-tracking-loss reset (see
                        // scan_session.py's PURE_WALKING_LOST_RESET_S) —
                        // the server's own pose/grid state was just wiped,
                        // so our locally-tracked route and motion estimates
                        // since the last fix are now stale garbage. Silent
                        // — matches this mode's "no spoken alerts" design;
                        // the beacon just mutes until a fresh pose/path
                        // arrives on a later update.
                        state.plannedPath = emptyList()
                        state.pathConfirmed = true
                        state.lastMappingPose = null
                        state.lastSentSnapshot = null
                        angleTracker.resetAccumulator()
                        pdrStepEstimator.resetAccumulator()
                        pixieController.mute()
                        reportBeaconDirection(0f, muted = true)
                        return@collect
                    }

                    if (state.mode == "guiding") resolveGuidingGoalIfNeeded()
                    // See the class-level walkingReady doc comment — the
                    // very first real MappingUpdate for a walking session
                    // is the "server actually processed a frame" signal
                    // this gate is waiting on.
                    if (state.mode == "walking" && !walkingReady) {
                        activateWalkingOnceReady()
                    }

                    // Latency compensation: the server's `update.pose` describes
                    // an already-slightly-stale frame (network + RTAB-Map/DA3
                    // processing time). Fast-forward it by however much MORE
                    // rotation/distance the local estimators have accumulated
                    // since the LATEST frame we sent — not necessarily the exact
                    // one this update was computed from (the server may have
                    // dropped several in between via _latest_only_chunks(), so
                    // an exact timestamp match isn't guaranteed); an accepted
                    // approximation, see LiveSessionState.lastSentSnapshot's own
                    // doc comment.
                    val snapshot = state.lastSentSnapshot?.second
                    val correctedPose = if (snapshot != null) {
                        val deltaSinceSend = HrtfBeacon.quatMultiply(
                            floatArrayOf(-snapshot.rotationAccum[0], -snapshot.rotationAccum[1],
                                -snapshot.rotationAccum[2], snapshot.rotationAccum[3]),
                            angleTracker.accumulatedRotation(),
                        )
                        val distanceSinceSend = pdrStepEstimator.distanceSinceReset() - snapshot.distanceAccum
                        HrtfBeacon.extrapolate(update.pose, deltaSinceSend, distanceSinceSend)
                    } else {
                        update.pose
                    }
                    state.lastMappingPose = correctedPose
                    // Also folds this fix in as AngleTracker's new authoritative
                    // heading baseline (the "update direction when direction info
                    // sent from RTAB-Map through the server" half of that module)
                    // — replaces the old bare resetAccumulator() call here.
                    angleTracker.setAuthoritativeHeadingDeg(HrtfBeacon.worldHeadingDeg(correctedPose))
                    pdrStepEstimator.resetAccumulator()

                    // The MAIN path's joints, straight from the server — no
                    // prepended pose any more (that was only ever needed for
                    // PathPursuit's whole-path arc-length projection, which
                    // this design no longer uses — see "Main-path/sub-path
                    // joint navigation" below).
                    //
                    // Re-indexed to joint 0 only when the incoming path's
                    // CONTENT actually changed from what's already tracked
                    // — NOT unconditionally on every update any more. The
                    // server now often resends the exact same route several
                    // updates in a row (path-stability: it only replans
                    // when the near-term route is genuinely blocked, see
                    // CLAUDE.md's path-stability note) specifically so
                    // mainPathIdx's own local advancement (real arrival
                    // events, see advanceMainPathJoint()) isn't disturbed —
                    // resetting to 0 on every resend of an UNCHANGED path
                    // would silently discard that progress every ~1Hz
                    // update, which is exactly the bug this guards against.
                    // A genuinely different path (a real replan) still
                    // resets to 0, since the server always plans a fresh
                    // route FROM the current pose forward, so a new path's
                    // own first joint is the correct next target.
                    val rawPath = update.plannedPath.pointsList.map { it.x to it.z }
                    val pathChanged = !pathsRoughlyEqual(rawPath, state.plannedPath)
                    state.plannedPath = rawPath
                    state.pathConfirmed = update.plannedPath.confirmed
                    if (pathChanged) {
                        state.mainPathIdx = 0
                    }

                    if (state.mode == "walking" && rawPath.isEmpty()) {
                        playDeadEndAlert()
                    }
                    if (state.mode == "guiding") {
                        checkGuidingArrival(correctedPose)
                    }
                    // One-shot per session: the very first time a real main
                    // path arrives, announce its clock direction — see
                    // announceClockDirection()'s own doc comment. Later
                    // announcements come from advanceMainPathJoint(), a
                    // purely local event, not from subsequent server
                    // updates (which would otherwise re-announce roughly
                    // the same direction on every ~1Hz update).
                    if (!mainJointAnnounced && rawPath.isNotEmpty()) {
                        mainJointAnnounced = true
                        announceClockDirection(correctedPose, rawPath[0])
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Mapping stream ended: ${e.message}")
            }
        }
    }

    /** GUIDING only — resolves guidingDestinationLabel to a world point via
     * FindLandmark (unchanged RPC), then stashes it so feedMappingFrame()
     * sends it to the server on every subsequent chunk as the path-planning
     * goal (see MappingChunk.has_goal's proto comment). Retried on every
     * mapping update until it succeeds; no memoization of "already tried
     * and failed" — same known, accepted gap the old recomputeRoute() had. */
    private suspend fun resolveGuidingGoalIfNeeded() {
        if (state.guidingGoalXz != null) return
        val destination = state.guidingDestinationLabel
        if (destination.isBlank()) return
        val stub = grpc.mappingStub ?: return
        try {
            val resp = stub.findLandmark(
                Tracking.FindLandmarkRequest.newBuilder()
                    .setLocationId(locationId)
                    .setQuery(destination)
                    .build()
            )
            if (resp.found) state.guidingGoalXz = resp.x to resp.z
        } catch (e: Exception) {
            Log.w(TAG, "FindLandmark failed for '$destination': ${e.message}")
        }
    }

    /** GUIDING only — once the extrapolated position is within
     * ARRIVAL_RADIUS_M of the server-planned path's final point, announce
     * arrival exactly once (guidingArrivalAnnounced guards repeats; reset
     * in toolStartGuiding()). WALKING has no destination, so no equivalent
     * check applies there. */
    private fun checkGuidingArrival(pose: Tracking.Pose) {
        val path = state.plannedPath
        if (path.isEmpty() || guidingArrivalAnnounced) return
        val (ex, ez) = path.last()
        val dist = kotlin.math.sqrt((ex - pose.x) * (ex - pose.x) + (ez - pose.z) * (ez - pose.z))
        if (dist < ARRIVAL_RADIUS_M) {
            guidingArrivalAnnounced = true
            sendSystemNote("[SYSTEM] Arrived at ${state.guidingDestinationLabel}.")
        }
    }

    private fun stopMappingStream() {
        mappingJob?.cancel(); mappingJob = null
        mappingChunkChannel?.close(); mappingChunkChannel = null
    }

    /** Same as stopMappingStream(), but actually waits for the stream to
     * finish instead of just requesting cancellation — closing the channel
     * (not cancelling the job) lets the gRPC call half-close cleanly, so
     * the server's UpdateMapping sees a normal end-of-stream and runs its
     * `finally` block (flush + finalize_landmarks + snapshot save) before
     * this returns. Needed by toolStopScan(): auto-transitioning straight
     * into walking afterward only makes sense once that finalize has had
     * the chance to persist what scanning just built. */
    private suspend fun stopMappingStreamAndAwait() {
        mappingChunkChannel?.close(); mappingChunkChannel = null
        mappingJob?.join(); mappingJob = null
    }

    /** Feeds one camera frame into the live mapping stream — called from
     * MainViewModel's frame loop while guiding/walking/scanning is active
     * (MainViewModel's mappingModeActive gate covers all three). Not every
     * mode needs this (reading/tracking/idle don't), so the caller gates
     * it. session_mode tells the server which pipeline to run (see
     * mapping_servicer.py's UpdateMapping / scan_session.py's
     * walking_lite/pure_walking) — only actually consulted server-side on
     * the stream's first chunk, but cheap to set on every one rather than
     * special-case the first.
     *
     * WALKING's cold-start warm-up gate (see the class-level walkingReady
     * doc comment): sends exactly ONE frame, then withholds every
     * subsequent one until the server's response to that first frame has
     * actually come back — there's no point burning camera/encode work or
     * server GPU time on more frames while the server is still busy on its
     * very first RTAB-Map/occupancy round trip for this session. */
    fun feedMappingFrame(jpeg: ByteArray) {
        val channel = mappingChunkChannel ?: return
        if (state.mode == "walking" && !walkingReady) {
            if (walkingFirstFrameSent) return
            walkingFirstFrameSent = true
        }
        val sessionMode = when (state.mode) {
            "walking" -> Tracking.SessionMode.WALKING
            "guiding" -> Tracking.SessionMode.GUIDING
            else -> Tracking.SessionMode.SCAN  // "scanning" and any other caller
        }
        channel.trySend(buildMappingChunk(jpeg, sessionMode))
    }

    /** Fires once, the first time a real MappingUpdate arrives for a
     * walking session (see the mapping-stream collector) — the actual
     * "walking mode is ready" transition per the user's explicit spec:
     * only NOW does it make sense to activate the local avoidance ticks/
     * Pixie/step estimator (previously all started immediately in
     * toolStartWalking(), before the server had processed anything at
     * all — the real cause of the reported cold-start lag) and only NOW
     * is Gemini told walking has actually started. */
    private fun activateWalkingOnceReady() {
        walkingReady = true
        pdrStepEstimator.resetAccumulator()
        pdrStepEstimator.start()
        pixieController.start()  // plays constantly but muted until the first path arrives
        startLocalAvoidanceTicks() // drives runUnifiedAvoidanceTick()
        onGuidanceUpdate("walking", emptyList())
        reportMode("walking")
        sendSystemNote(
            "[SYSTEM] Walking mode is now active and ready — ambient obstacle guidance has started. " +
                "Briefly acknowledge if relevant, otherwise no need to say anything further."
        )
    }

    /** Shared MappingChunk builder — used by feedMappingFrame(), driven by
     * MainViewModel's continuous camera collector for all of guiding/
     * walking/scanning. For GUIDING, once resolveGuidingGoalIfNeeded() has
     * set state.guidingGoalXz, it's attached to every subsequent chunk so
     * the server knows what to plan toward (see MappingChunk.has_goal's
     * proto comment) — WALKING sends no goal at all. Also overwrites
     * state.lastSentSnapshot with the local estimators' current
     * accumulator state (WALKING/GUIDING only; SCAN has no latency-
     * compensation need for this) — see LiveSessionState.lastSentSnapshot's
     * docstring for why this is a single overwritten value, not a history. */
    private fun buildMappingChunk(jpeg: ByteArray, sessionMode: Tracking.SessionMode): Tracking.MappingChunk {
        val builder = Tracking.MappingChunk.newBuilder()
            .setLocationId(locationId)
            .setImageData(com.google.protobuf.ByteString.copyFrom(jpeg))
            .setFrameTimestampNs(TimeUnit.MILLISECONDS.toNanos(System.currentTimeMillis()))
            .setPoseSource(Tracking.PoseSource.RTABMAP)
            .setSessionMode(sessionMode)
        val goal = state.guidingGoalXz
        if (sessionMode == Tracking.SessionMode.GUIDING && goal != null) {
            builder.setHasGoal(true).setGoalX(goal.first).setGoalZ(goal.second)
        }
        val chunk = builder.build()
        if (sessionMode != Tracking.SessionMode.SCAN) {
            state.lastSentSnapshot = chunk.frameTimestampNs to LiveSessionState.PoseSendSnapshot(
                angleTracker.accumulatedRotation(), pdrStepEstimator.distanceSinceReset(),
            )
        }
        return chunk
    }

    /** Feeds one camera luma frame into AngleTracker — called from
     * MainViewModel's continuous per-frame collector (same lumaFlow
     * hand-tracking/local-ORB-tracking already draw from, NOT the
     * interval-gated mapping-mode push) whenever WALKING/GUIDING is active.
     * Rotation tracking wants every frame for good frame-to-frame matching,
     * not a throttled one — see AngleTracker's own docstring. Takes raw
     * Y-plane luma directly (CameraManager.kt's lumaFlow) instead of a
     * decoded JPEG — see AngleTracker's docstring for why. */
    fun feedAngleLumaFrame(luma: ByteArray, width: Int, height: Int, rowStride: Int, rotationDegrees: Int) {
        if (state.mode != "walking" && state.mode != "guiding") return
        angleTracker.processLumaFrame(luma, width, height, rowStride, rotationDegrees)
    }

    // ── Main-path/sub-path joint navigation — both modes ──────────────────
    //
    // Path PLANNING moved entirely server-side (see CLAUDE.md's
    // "Server-planned walking path" note) — state.plannedPath (the MAIN
    // path) is server-computed, obstacle-aware, and re-derived fresh every
    // ~1Hz MappingUpdate. Redesigned per direct user feedback: rather than
    // locking the whole main path in place for a fixed duration (the
    // earlier, reverted 6s server-side heading-dwell hysteresis) or
    // continuously projecting/advancing along it (the earlier PathPursuit
    // design), the client now tracks progress by JOINT
    // (state.mainPathIdx) and, twice a second (avoidanceIntervalMs is
    // 500ms for this purpose), plans a short LOCAL sub-path from the
    // current pose to the current main-path joint — itself obstacle-
    // dodging, via a fresh AnalyzeFrame(TRAVERSABILITY) fan each tick, not
    // a stale/cached one. The beacon always steers toward the sub-path's
    // own first point (the nearest thing to walk toward right now, whether
    // that's a dodge point or the main joint itself) — its egocentric
    // bearing IS the "drift" the flapping-sound volume reflects.
    //
    // Arrival (main OR sub joint) is "close enough, with no other sub-path
    // joint still in between" — since the sub-path is recomputed fresh
    // every tick from the CURRENT pose, this reduces to: once the freshly-
    // computed sub-path is down to just the main joint itself (no dodge
    // point pending) AND the user has arrived at it, advance to the next
    // main joint. A sub-path dodge point never needs its own explicit
    // "arrived, remove it" step — the very next tick's fresh recompute
    // already reflects having passed it (or not) from the new pose.

    private fun startLocalAvoidanceTicks() {
        stopLocalAvoidanceTicks()
        avoidanceJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                runUnifiedAvoidanceTick()
                delay(avoidanceIntervalMs.toLong())
            }
        }
    }

    private fun stopLocalAvoidanceTicks() {
        avoidanceJob?.cancel(); avoidanceJob = null
    }

    /** One tick, both modes: the novelty-gated vision-check alert
     * (unrelated to steering) plus main-path/sub-path steering from the
     * CONTINUOUSLY extrapolated pose. */
    private suspend fun runUnifiedAvoidanceTick() {
        val frame = latestFrame() ?: return
        sendWalkingAmbientFrame(frame)
        // Novelty-gated, not a flat timer — see AngleTracker's own doc
        // comment on evaluateNovelty()/consumeNoveltyTrigger(). Still
        // subject to runPeriodicVisionCheck()'s own internal cooldown
        // (PERIODIC_ALERT_INTERVAL_MS, now 3s) as a safety net against a
        // rapidly-flickering novelty signal.
        if (angleTracker.consumeNoveltyTrigger()) {
            runPeriodicVisionCheck(frame)
        }

        // Fetched once per tick, for steerAlongMainPath()'s own local dodge
        // only now — the obstacle-ahead beep runs on its own separate,
        // fixed-rate poll (startObstacleAheadPolling()) instead of piggy-
        // backing on this tick, so it isn't held hostage by anything else
        // this tick does.
        val trav = fetchTraversability(frame)

        val authoritative = state.lastMappingPose ?: return
        val pose = HrtfBeacon.extrapolate(
            authoritative, angleTracker.accumulatedRotation(), pdrStepEstimator.distanceSinceReset(),
        )
        steerAlongMainPath(pose, trav)
    }

    /** Depth-map-based "obstacle directly ahead, close range, AND getting
     * closer" alert — requested directly by the user, refined per direct
     * follow-up feedback ("it alert too much... anyway to detect if the
     * obstacle within 1.5m is getting closer") — a static-but-close
     * obstacle (a wall the user is standing near but not approaching, a
     * table off to the side that just happens to sit in the middle
     * corridor) used to re-beep on every single poll once inside range,
     * which was the actual source of the "too much" complaint, not the
     * range itself. Deliberately NOT tied to Gemini Live at all (no
     * sendVideoFrame/sendSystemNote, no check_obstacle tool) — a plain fast
     * RPC (PerceptionService.AnalyzeFrame's DEPTH op, a single DA3 call + a
     * percentile check over the middle-width corridor server-side, no
     * RANSAC ground-plane fit). Runs on its OWN fixed-rate poll
     * (startObstacleAheadPolling()), decoupled from the much slower
     * MappingService stream and from avoidanceIntervalMs. No movement gate.
     *
     * lastObstacleDistanceM tracks the previous poll's reading so this poll
     * can compare against it: only counts as "approaching" when the new
     * reading is at least OBSTACLE_AHEAD_CLOSING_MARGIN_M closer than the
     * last one (a flat/oscillating reading near the noise floor shouldn't
     * count), OR there was no previous in-range reading at all (the very
     * first poll that finds something close still deserves a warning, since
     * there's nothing yet to compare a trend against). Reset to null the
     * instant the obstacle leaves range (obstacle.detected false, or beyond
     * OBSTACLE_THRESHOLD_M server-side) so a later, fresh approach starts
     * its own trend from scratch rather than comparing against a stale
     * reading from a completely different encounter. Still rate-limited
     * (OBSTACLE_AHEAD_COOLDOWN_MS) on top of the trend check. */
    private suspend fun pollObstacleAheadOnce() {
        val frame = latestFrame() ?: return
        val obstacle = fetchObstacleInfo(frame) ?: return
        if (!obstacle.detected) {
            lastObstacleDistanceM = null
            return
        }

        val distance = obstacle.distanceM
        val previous = lastObstacleDistanceM
        lastObstacleDistanceM = distance
        val approaching = previous == null || (previous - distance) >= OBSTACLE_AHEAD_CLOSING_MARGIN_M
        if (!approaching) return

        val now = System.currentTimeMillis()
        if (now - lastObstacleAheadWarnedAtMs < OBSTACLE_AHEAD_COOLDOWN_MS) return
        lastObstacleAheadWarnedAtMs = now

        playObstacleAheadBeep()
    }

    /** PerceptionService.AnalyzeFrame(DEPTH) — a fast, unary, mapping-
     * stream-independent call; see pollObstacleAheadOnce()'s doc comment. */
    private suspend fun fetchObstacleInfo(frame: ByteArray): Tracking.ObstacleInfo? {
        val stub = grpc.perceptionStub ?: return null
        return try {
            stub.analyzeFrame(
                Tracking.AnalyzeFrameRequest.newBuilder()
                    .setImageData(com.google.protobuf.ByteString.copyFrom(frame))
                    .addOps(Tracking.AnalysisOp.DEPTH)
                    .build()
            ).obstacle
        } catch (e: Exception) {
            Log.w(TAG, "fetchObstacleInfo failed: ${e.message}")
            null
        }
    }

    private fun startObstacleAheadPolling() {
        stopObstacleAheadPolling()
        // Fixed 2fps (500ms) per the user's explicit spec — independent of
        // avoidanceIntervalMs and, critically, started immediately in
        // toolStartWalking()/toolStartGuiding() rather than deferred behind
        // activateWalkingOnceReady() — this alert has nothing to do with
        // the mapper and shouldn't wait on its cold-start lag.
        obstacleAheadJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                pollObstacleAheadOnce()
                delay(OBSTACLE_AHEAD_POLL_INTERVAL_MS)
            }
        }
    }

    private fun stopObstacleAheadPolling() {
        obstacleAheadJob?.cancel(); obstacleAheadJob = null
        lastObstacleDistanceM = null
    }

    /** Plans a fresh local sub-path toward the current main-path joint,
     * steers Pixie toward its first point, and advances mainPathIdx on
     * arrival — see the section-level doc comment above for the full
     * design. [trav] is the SAME per-tick traversability fan
     * runUnifiedAvoidanceTick() already fetched for checkAndWarnObstacleAhead(). */
    private fun steerAlongMainPath(pose: Tracking.Pose, trav: Tracking.TraversabilityInfo?) {
        val mainPath = state.plannedPath
        if (mainPath.isEmpty() || state.mainPathIdx >= mainPath.size) {
            pixieController.mute()
            reportBeaconDirection(0f, muted = true)
            return
        }
        val target = mainPath[state.mainPathIdx]
        val subPath = computeSubPath(pose, target, trav)
        val steerTarget = subPath.first()

        val bearing = HrtfBeacon.directionTo(pose, steerTarget.first, steerTarget.second)
        pixieController.move(if (bearing.azimuthDeg < 0) PixiePoint.LEFT else PixiePoint.RIGHT)
        // Drift-to-volume mapping, per the user's explicit spec: silent
        // within +-NAV_DEADZONE_DEG (3deg), ramps to full volume by
        // +-NAV_RAMP_END_DEG (100deg), pinned at full beyond.
        pixieController.setVolume(gainForDeviation(bearing.azimuthDeg, NAV_DEADZONE_DEG, NAV_RAMP_END_DEG))
        reportBeaconDirection(bearing.azimuthDeg, muted = false)

        if (subPath.size == 1 && bearing.distanceM <= JOINT_ARRIVAL_RADIUS_M) {
            advanceMainPathJoint(pose)
        }
    }

    /** True if two main-path point lists describe (roughly) the same route
     * — used by the mapping-stream collector to decide whether a freshly
     * arrived path is genuinely new (reset mainPathIdx to 0) or just the
     * server re-serving the same stable route it served last update (keep
     * mainPathIdx exactly as-is, so local joint-arrival progress isn't
     * discarded). A small per-point tolerance guards against incidental
     * float noise rather than requiring bit-exact equality. */
    private fun pathsRoughlyEqual(
        a: List<Pair<Float, Float>>, b: List<Pair<Float, Float>>, epsM: Float = 0.05f,
    ): Boolean {
        if (a.size != b.size) return false
        for (i in a.indices) {
            val dx = a[i].first - b[i].first
            val dz = a[i].second - b[i].second
            if (kotlin.math.sqrt(dx * dx + dz * dz) > epsM) return false
        }
        return true
    }

    /** Advances past the current main-path joint (arrived, no sub-path
     * dodge point pending) and announces the new target's clock direction
     * — see announceClockDirection()'s own doc comment. */
    private fun advanceMainPathJoint(pose: Tracking.Pose) {
        state.mainPathIdx++
        val next = state.plannedPath.getOrNull(state.mainPathIdx) ?: return
        announceClockDirection(pose, next)
    }

    /** Builds a short LOCAL sub-path from [pose] toward [target] (the
     * current main-path joint), bending around anything the traversability
     * fan shows blocking the direct line — at most one dodge point,
     * recomputed fresh every avoidance tick (2Hz) from the CURRENT view
     * rather than searched/cached once. Returns [target] alone whenever
     * nothing's in the way (the common case), or when no [trav] fan is
     * available at all (fail open — direct steering, same as before this
     * redesign). A simplified, single-dodge-point VFH-style pick — much
     * smaller in scope than the old (removed) TraversabilityScorer, since
     * the MAIN route is already obstacle-aware; this only ever needs to
     * smooth out something that appeared since the main path was last
     * planned. */
    private fun computeSubPath(
        pose: Tracking.Pose, target: Pair<Float, Float>, trav: Tracking.TraversabilityInfo?,
    ): List<Pair<Float, Float>> {
        val direct = listOf(target)
        if (trav == null || trav.clearanceMCount == 0) return direct

        val bearing = HrtfBeacon.directionTo(pose, target.first, target.second)
        val directClearance = binClearance(trav, bearing.azimuthDeg) ?: return direct
        if (directClearance >= SUBPATH_SAFE_CLEARANCE_M || directClearance >= bearing.distanceM) return direct

        // Direct line is blocked closer than the target itself — search the
        // fan for the nearest bin (smallest deviation from the direct
        // bearing) that clears SUBPATH_SAFE_CLEARANCE_M.
        var bestAngle: Float? = null
        var bestClearance = 0f
        var bestDeviation = Float.MAX_VALUE
        var angle = trav.minAngleDeg
        while (angle <= trav.maxAngleDeg) {
            val clearance = binClearance(trav, angle)
            if (clearance != null && clearance >= SUBPATH_SAFE_CLEARANCE_M) {
                val deviation = abs(angle - bearing.azimuthDeg)
                if (deviation < bestDeviation) {
                    bestDeviation = deviation; bestAngle = angle; bestClearance = clearance
                }
            }
            angle += trav.angleStepDeg
        }
        val chosenAngle = bestAngle ?: return direct  // nothing open anywhere in the fan — steer direct anyway
        val dodgeDistance = (bestClearance - SUBPATH_DODGE_MARGIN_M)
            .coerceIn(SUBPATH_DODGE_MARGIN_M, SUBPATH_DODGE_MAX_M)
        val dodgePoint = HrtfBeacon.worldPointFrom(pose, chosenAngle, dodgeDistance)
        return listOf(dodgePoint, target)
    }

    /** Nearest fan bin's clearance_m at [azimuthDeg] — null if [azimuthDeg]
     * falls outside the fan's own angular range (nothing observed there
     * this frame). */
    private fun binClearance(trav: Tracking.TraversabilityInfo, azimuthDeg: Float): Float? {
        if (trav.angleStepDeg <= 0f) return null
        val idx = ((azimuthDeg - trav.minAngleDeg) / trav.angleStepDeg).let { Math.round(it) }
        return trav.clearanceMList.getOrNull(idx)
    }

    /** One AnalyzeFrame(TRAVERSABILITY) round trip per avoidance tick — the
     * local dodge layer's only sensing input. Returns null on any failure
     * (not connected, RPC error) — computeSubPath() fails OPEN in that
     * case (steers directly at the main joint, same as before this
     * sub-path redesign existed). */
    private suspend fun fetchTraversability(frame: ByteArray): Tracking.TraversabilityInfo? {
        val stub = grpc.perceptionStub ?: return null
        return try {
            val resp = stub.analyzeFrame(
                Tracking.AnalyzeFrameRequest.newBuilder()
                    .setImageData(com.google.protobuf.ByteString.copyFrom(frame))
                    .addOps(Tracking.AnalysisOp.TRAVERSABILITY)
                    .build()
            )
            resp.traversability
        } catch (e: Exception) {
            Log.w(TAG, "fetchTraversability failed: ${e.message}")
            null
        }
    }

    /** Converts an egocentric azimuth into a 1-12 clock position (12 =
     * straight ahead, 3 = directly right, 6 = behind, 9 = directly left) —
     * requested directly by the user: "at start and each time moving to
     * the next main path joint, give instruction in clock direction." */
    private fun clockPositionFor(azimuthDeg: Float): Int {
        var deg = azimuthDeg % 360f
        if (deg < 0f) deg += 360f
        var clock = Math.round(deg / 30f) % 12
        if (clock == 0) clock = 12
        return clock
    }

    /** Sends a one-off [SYSTEM] note giving the clock direction of [target]
     * relative to [pose]'s current facing — fired once at the start of a
     * WALKING/GUIDING session (first main path) and once per subsequent
     * main-joint advance (see advanceMainPathJoint()), NOT on every server
     * path update (which would re-announce roughly the same direction on
     * every ~1Hz update). Distinct from TRACKING mode's own "never clock
     * positions" rule — that rule is specific to hand-guidance's
     * plain-words convention; navigation's own turn cues are exactly what
     * clock positions are for. */
    private fun announceClockDirection(pose: Tracking.Pose, target: Pair<Float, Float>) {
        val bearing = HrtfBeacon.directionTo(pose, target.first, target.second)
        val clock = clockPositionFor(bearing.azimuthDeg)
        sendSystemNote(
            "[SYSTEM] Next turn: about $clock o'clock relative to your current facing direction. " +
                "Briefly say the clock direction, e.g. \"$clock o'clock\"."
        )
    }

    // ── Walking — server-planned "keep going forward" HRTF beacon ────────
    //
    // Reopens the SAME MappingService stream guiding uses (RTAB-Map pose +
    // live occupancy grid, via startMappingStream()/feedMappingFrame()) so
    // RTAB-Map keeps building a local map every frame, reset aggressively
    // (PURE_WALKING_LOST_RESET_S, server-side) whenever tracking is lost —
    // see CLAUDE.md's walking-mode local-map note. Path planning itself now
    // happens server-side (find_farthest_open_path()) — see the section
    // above for the unified client-side steering that consumes it.

    /** Only opens the mapping stream here — ticks/Pixie/PDR/the "walking
     * started" notification are all DEFERRED to activateWalkingOnceReady(),
     * fired once the server's response to the very first frame actually
     * lands (see the class-level walkingReady doc comment). Requested
     * directly by the user after real on-device cold-start lag: previously
     * everything activated immediately, before the server had processed
     * anything at all. Gemini gets a quick, honest status here — the real
     * "ready" announcement comes later as its own [SYSTEM] note. */
    private fun toolStartWalking(): JSONObject {
        stopActiveModes()
        state.mode = "walking"
        walkingReady = false
        walkingFirstFrameSent = false
        startMappingStream()       // RTAB-Map pose + live local grid, same stream guiding uses
        // Started immediately, NOT deferred to activateWalkingOnceReady() —
        // this alert has nothing to do with the mapper (see
        // pollObstacleAheadOnce()'s own doc comment) and shouldn't wait on
        // its cold-start lag.
        startObstacleAheadPolling()
        return JSONObject().put("status", "walking_starting")
            .put("note", "Warming up — wait for the ready [SYSTEM] note before describing walking as active.")
    }

    private fun toolStopWalking(): JSONObject {
        stopMappingStream()
        stopLocalAvoidanceTicks()
        stopObstacleAheadPolling()
        pixieController.stop()
        pdrStepEstimator.stop()
        walkingReady = false
        walkingFirstFrameSent = false
        state.mode = "idle"
        state.plannedPath = emptyList()
        onGuidanceUpdate("idle", emptyList())
        reportMode("idle")
        return JSONObject().put("status", "walking_stopped")
    }

    /** WALKING only — feeds Gemini one frame per WALKING_AMBIENT_FRAME_INTERVAL_MS
     * so it has continuous, current visual context of what's ahead, without
     * needing an explicit get_latest_frame/start_vision_stream tool call
     * first. GUIDING is unaffected (its own destination/route context comes
     * from the mapping stream, not raw frames). */
    private fun sendWalkingAmbientFrame(frame: ByteArray) {
        if (state.mode != "walking") return
        val now = System.currentTimeMillis()
        if (now - lastWalkingAmbientFrameSentAtMs < WALKING_AMBIENT_FRAME_INTERVAL_MS) return
        lastWalkingAmbientFrameSentAtMs = now
        sendVideoFrame(frame)
    }

    /** Both WALKING and GUIDING, once every PERIODIC_ALERT_INTERVAL_MS since
     * the LAST time this sent a request (not since a response finished
     * playing — a flat send-to-send cadence): sends the current frame and
     * asks Gemini's own vision to judge whether there's a genuinely
     * hazardous step-down/drop-off/obstacle right in front of the user, and
     * if not, to briefly narrate which way to keep going and what they're
     * approaching. Replaces the old local-depth-heuristic-gated hazard
     * check (clearance_m/dropoff_m thresholds from AnalyzeFrame's
     * TRAVERSABILITY op) — that heuristic's single-frame RANSAC ground-plane
     * fit proved unreliable on real stairs/drop-offs (see CLAUDE.md), so the
     * decision now runs on a flat cadence and leaves the actual judgment
     * entirely to Gemini's vision rather than a local pre-filter. This also
     * doubles as WALKING's "what are we heading to" ambient narration when
     * nothing hazardous is found, since it fires regardless of whether a
     * hazard is present. */
    private fun runPeriodicVisionCheck(frame: ByteArray) {
        val now = System.currentTimeMillis()
        if (now - lastPeriodicAlertAtMs < PERIODIC_ALERT_INTERVAL_MS) return
        lastPeriodicAlertAtMs = now

        val destinationHint = if (state.mode == "guiding" && state.guidingDestinationLabel.isNotBlank()) {
            " The user is currently being guided toward: ${state.guidingDestinationLabel}."
        } else ""
        val stylePrompt = if (state.mode == "walking") {
            "Reply in as few words as possible — short fragments, no need for full grammatical " +
                "sentences, this needs to be fast enough to dodge something in real time. E.g. " +
                "\"Wall ahead, Stop.\" \"A box 12 o'clock, can step over.\" \"Wall ahead, turn left.\" \"Basket 1 o'clock, dodge bit left.\" \"Clear, heading at the door.\""
        } else {
            "Keep it brief and to the point."
        }
        sendVideoFrame(frame)
        sendSystemNote(
            "[SYSTEM] Look at the attached camera frame — base your judgment and navigation on THIS " +
                "latest frame only; any earlier frames you've seen are for reference/context if needed, " +
                "not what you should be reacting to now. If a step down, drop-off, stairs, or an " +
                "obstacle is CLEARLY visible right in front of the user — only if you're genuinely " +
                "confident, don't guess — give a quick instruction on how to react: stop, step over it, " +
                "go around/dodge left or right, or proceed with caution, whichever fits what you see. " +
                "Otherwise, if nothing hazardous is visible, briefly say which direction to keep going " +
                "and what they appear to be approaching. $stylePrompt$destinationHint"
        )
        Log.i(TAG, "periodic vision check sent (mode=${state.mode})")
    }

    /** Dead-end alert — no walkable path anywhere (MappingUpdate.planned_path
     * came back empty — find_farthest_open_path() found nothing in any
     * direction, server-side), WALKING only. A synthesized ToneGenerator
     * tone, not a bundled audio asset, so it works without needing a sound
     * file supplied. Rate-limited
     * (ALERT_PERIOD_MS) so it doesn't fire every single grid update —
     * repeated, not one-shot, so the user keeps getting a cue while they
     * turn looking for a way through. */
    private fun playDeadEndAlert() {
        val now = System.currentTimeMillis()
        if (now - lastDeadEndAlertAtMs < ALERT_PERIOD_MS) return
        lastDeadEndAlertAtMs = now
        try {
            val tg = toneGenerator ?: ToneGenerator(AudioManager.STREAM_MUSIC, ToneGenerator.MAX_VOLUME).also { toneGenerator = it }
            tg.startTone(ToneGenerator.TONE_SUP_ERROR, ALERT_TONE_DURATION_MS)
        } catch (e: Exception) {
            Log.w(TAG, "dead-end alert tone failed: ${e.message}")
        }
    }

    /** Instant obstacle-ahead beep — a plain RPC signal (server-side depth
     * check), no Gemini involved. Rate-limiting already happened in the
     * caller via lastObstacleAheadWarnedAtMs, so this just triggers the
     * sound. Plays the bundled assets/beep.mp3 asset (via the Service-owned
     * playObstacleBeep callback) instead of a synthesized ToneGenerator
     * tone — changed per direct user feedback ("not that lightly beep
     * sound, but a like a warning beep"; a prior TONE_CDMA_ALERT_CALL_GUARD
     * attempt still wasn't audible in practice). */
    private fun playObstacleAheadBeep() {
        try {
            playObstacleBeep()
        } catch (e: Exception) {
            Log.w(TAG, "obstacle-ahead beep failed: ${e.message}")
        }
    }

    /** A short "pop" confirmation tone, played once a reading-mode scan's
     * FULL pipeline — OCR, correction (when configured), and storage — has
     * actually finished for one capture, not just when the frame was
     * grabbed. Requested directly by the user: since this app's users can't
     * see the screen, a distinct audio cue that a scan cycle is truly done
     * is real feedback, not decoration. Reuses the same shared
     * `toneGenerator` instance as playDeadEndAlert() (no bundled audio
     * asset needed), a different TONE_* constant so the two are never
     * confused. Unlike playDeadEndAlert(), not rate-limited — each call
     * site only calls this once per completed capture, never on a repeat
     * tick. */
    private fun playScanCompleteCue() {
        try {
            val tg = toneGenerator ?: ToneGenerator(AudioManager.STREAM_MUSIC, ToneGenerator.MAX_VOLUME).also { toneGenerator = it }
            tg.startTone(ToneGenerator.TONE_PROP_ACK, SCAN_COMPLETE_TONE_DURATION_MS)
        } catch (e: Exception) {
            Log.w(TAG, "scan-complete cue tone failed: ${e.message}")
        }
    }

    // ── Scanning (mapping pipeline only — no destination, no obstacle polling) ──

    private fun toolStartScan(): JSONObject {
        stopActiveModes()
        state.mode = "scanning"
        startMappingStream()
        onGuidanceUpdate("scanning", emptyList())
        reportMode("scanning")
        return JSONObject().put("status", "scan_started")
    }

    private suspend fun toolStopScan(): JSONObject {
        stopMappingStreamAndAwait()  // waits through the server's finalize (see docstring)
        // Auto-transition straight into walking mode — the user shouldn't
        // need to separately say "start walking" right after finishing a
        // scan of the space they're already standing in.
        val walkResult = toolStartWalking()
        return JSONObject().put("status", "scan_stopped").put("auto_walking", walkResult)
    }

    fun shutdown() {
        visionStreamJob?.cancel()
        // Only stop()s here — readingTts is owned/released by
        // LiveAssistantService (a Service-level field reused across
        // reconnects, like pixieController/streamingPlayer), not by this
        // per-connection ToolDispatcher instance.
        readingTts.stop()
        stopMappingStream()
        stopLiveReadingPipeline()
        stopLocalAvoidanceTicks()
        stopObstacleAheadPolling()
        pixieController.stop()
        pdrStepEstimator.stop()
        toneGenerator?.release(); toneGenerator = null
        correctionWorkerJob?.cancel()
        correctionChannel.close()
    }

    companion object {
        private const val TAG = "ToolDispatcher"

        // GUIDING's arrival radius — distance from the path's final point
        // at which checkGuidingArrival() announces arrival.
        private const val ARRIVAL_RADIUS_M = 1.0f
        // Main-path/sub-path joint arrival radius (steerAlongMainPath()) —
        // deliberately smaller than ARRIVAL_RADIUS_M above: joints along a
        // route sit much closer together than a whole destination's own
        // arrival tolerance.
        // Bumped 0.6m -> 1.5m per direct user request: advance to the next
        // joint (and announce its clock direction) with more lead time
        // before literal arrival, so the turn cue lands well before the
        // user actually needs to turn.
        private const val JOINT_ARRIVAL_RADIUS_M = 1.5f
        // Sub-path obstacle dodge (computeSubPath()) — the direct line to
        // the current main joint is only diverted once its own clearance
        // falls below this; the chosen dodge bearing must itself clear the
        // same bar. ~0.5m mirrors this codebase's existing "one person's
        // comfortable width" convention (see server/live_path_planner.py's
        // _WALKING_SAFE_CLEARANCE_M).
        private const val SUBPATH_SAFE_CLEARANCE_M = 0.5f
        // A dodge waypoint sits this far short of the chosen bearing's own
        // measured clearance (a safety margin, not a hard stop at the
        // obstacle itself), bounded so a very open bearing doesn't produce
        // a dodge point way out past where the user actually needs to
        // step.
        private const val SUBPATH_DODGE_MARGIN_M = 0.3f
        private const val SUBPATH_DODGE_MAX_M = 1.5f
        private const val ALERT_PERIOD_MS = 1500L
        private const val ALERT_TONE_DURATION_MS = 200
        // playScanCompleteCue() — short by design, a "pop", not an alert.
        private const val SCAN_COMPLETE_TONE_DURATION_MS = 80
        // read_aloud()'s speak-immediately-vs-store-to-session threshold —
        // requested directly by the user: a short capture (a label, a sign)
        // is read straight away; a longer one (a page) is saved to the
        // reading session instead of speaking the whole thing unprompted.
        private const val READ_ALOUD_IMMEDIATE_MAX_WORDS = 40

        // Vision-check cooldown (runPeriodicVisionCheck()) — GUIDING and
        // WALKING both. No longer a flat timer on its own: calls are now
        // gated by AngleTracker's ORB-novelty signal (see
        // runUnifiedAvoidanceTick()) — this is the MINIMUM gap enforced
        // between two sends regardless of how often novelty triggers, per
        // the user's explicit "still 3sec cooldown" spec. The hazard/
        // no-hazard judgment itself is still left entirely to Gemini's
        // vision, not a local distance heuristic.
        private const val PERIODIC_ALERT_INTERVAL_MS = 3000L
        // WALKING only (see sendWalkingAmbientFrame()) — how often a plain
        // context frame (no forced response) is fed to Gemini Live.
        private const val WALKING_AMBIENT_FRAME_INTERVAL_MS = 1000L

        // Depth-map obstacle-ahead beep (pollObstacleAheadOnce()) — a plain
        // RPC signal (PerceptionService.AnalyzeFrame's DEPTH op), NOT the
        // Gemini Live-invoked check_obstacle tool and NOT wired to Gemini at
        // all — the server thresholds range against
        // DA3DepthDetector.OBSTACLE_THRESHOLD_M (1.0m) over the middle-width
        // corridor (server/tools/depth.py's check_obstacle()); the client
        // has no range constant of its own to keep in sync — it just reads
        // ObstacleInfo.detected.
        //
        // Fixed 2fps poll — decoupled entirely from avoidanceIntervalMs and
        // from MappingService's own (much slower) update cadence, per the
        // user's explicit spec: the mapper's latency shouldn't gate how
        // quickly this alert can fire.
        private const val OBSTACLE_AHEAD_POLL_INTERVAL_MS = 500L
        private const val OBSTACLE_AHEAD_COOLDOWN_MS = 2000L
        // How much closer (metres) the current poll's reading must be than
        // the previous poll's for the obstacle to count as "approaching" —
        // see pollObstacleAheadOnce()'s own doc comment for why this exists
        // (a static-but-close obstacle used to re-beep every poll). Small
        // enough to catch a real, if slow, approach; large enough to not
        // fire on ordinary depth-estimate noise between two consecutive
        // 500ms-apart readings of the same still object.
        private const val OBSTACLE_AHEAD_CLOSING_MARGIN_M = 0.1f

        // Reading-mode blur skip/retry (see acquireSharpFrame()) — a blurry
        // frame is a wasted OCR call, so before spending one, re-sample by
        // waiting for the live camera to (hopefully) produce a sharper one,
        // up to BLUR_MAX_RETRIES times, before giving up on this cycle.
        private const val BLUR_RETRY_WAIT_MS = 500L
        private const val BLUR_MAX_RETRIES = 2

        // Full-resolution edge-device OCR capture (acquireSharpFrame()) —
        // a still capture + network round trip is inherently slower than
        // pulling an already-buffered live-stream frame, and main.py's own
        // StreamPauseGate auto-expires at 8s as a server-side safety net —
        // this client-side timeout stays comfortably above that so a
        // healthy-but-slow capture isn't cut off right as the server-side
        // safety net would otherwise resolve it on its own.
        private const val OCR_FULL_RES_TIMEOUT_MS = 10_000L

        // Live reading (startLiveReadingPipeline()) — real-time playback
        // pacing, same single-control simplification gt.py's own Live
        // Reading tab settled on ("play it out like realtime reading, only
        // need interval" — no separate advance/interval split). Not yet
        // Settings-configurable (unlike frameIntervalMs/scanIntervalMs/
        // avoidanceIntervalMs) — a reasonable follow-up if 5s turns out to
        // need tuning per-user.
        private const val LIVE_READING_INTERVAL_MS = 5000L
        // Real wall-clock wait once a frame is STILL blurry after
        // acquireSharpFrame()'s own retries — mirrors gt.py's
        // BLUR_GIVEUP_WAIT_S.
        private const val LIVE_READING_BLUR_GIVEUP_MS = 4000L

        // Visual object re-ID (get_object_from_memory()/is_this_object()) —
        // DINOv2 ViT-S/14 cosine similarity, see server/tools/embedder.py's
        // own docstring for the model's general "cosine >= 0.75 = same
        // target" guidance for continuous tracking re-ID; these two tools
        // use lower/task-specific thresholds instead:
        // - OBJECT_MATCH_MIN_SIM: get_object_from_memory() is picking the
        //   best of possibly-several candidate detections against ONE
        //   stored reference, so a slightly lower bar than a strict re-ID
        //   confirm is acceptable — ambiguity between near-tied candidates
        //   is handled separately (AMBIGUITY_MARGIN), not by raising this.
        private const val OBJECT_MATCH_MIN_SIM = 0.55f
        // Two candidates within this margin of each other are treated as
        // indistinguishable — return ambiguous=true instead of guessing.
        private const val AMBIGUITY_MARGIN = 0.075f
        // - IS_THIS_OBJECT_MIN_SIM: a direct "is this the same physical
        //   object" yes/no confirmation, not a best-of-several pick — held
        //   to a stricter bar than OBJECT_MATCH_MIN_SIM.
        private const val IS_THIS_OBJECT_MIN_SIM = 0.65f
        // - search_objects()'s own two-tier verdict, thresholds specified
        //   directly by the user: a confident match vs. a merely-plausible
        //   "resembles" match are both worth reporting (unlike the two
        //   thresholds above, which only ever expose one bar each) — below
        //   SEARCH_OBJECTS_RESEMBLE_SIM, a candidate is omitted entirely
        //   rather than reported as a firm miss.
        private const val SEARCH_OBJECTS_CONFIRM_SIM = 0.5f
        private const val SEARCH_OBJECTS_RESEMBLE_SIM = 0.3f

        // ── Pixie deviation-to-volume mapping — shared by tracking's
        // updateTrackingPixie() and guiding/walking's steerAlongMainPath().
        // Silent within [deadZoneDeg], ramps linearly to full volume by
        // [rampEndDeg], pinned at 1.0 beyond (never pinned at a LOWER
        // cutoff — full range down to 0, per the user's explicit spec) —
        // same shape validated in test_module/pixie_hrtf_app's own
        // gainForDeviation.
        fun gainForDeviation(deg: Float, deadZoneDeg: Float, rampEndDeg: Float): Float {
            val a = abs(deg)
            if (a <= deadZoneDeg) return 0f
            if (a >= rampEndDeg) return 1f
            return (a - deadZoneDeg) / (rampEndDeg - deadZoneDeg)
        }

        // Navigation (guiding/walking) flapping-sound cue — requested
        // directly by the user: silent for a drift of 0-3deg, ramps to
        // full volume across 3-100deg, pinned at full beyond 100deg
        // (tighter than the earlier +-180deg full-range mapping — a
        // deliberate narrowing so the cue reads as "off" much sooner).
        private const val NAV_DEADZONE_DEG = 3f
        private const val NAV_RAMP_END_DEG = 100f

        // Tracking mode's 2-phase hand-guidance cue (see updateTrackingPixie()).
        // Both phases read HrtfBeacon.directionFromBox()'s screen-space
        // azimuth/elevation, which — given that function's own pinhole
        // assumption (fx=fy=0.8*max(w,h)) — only ever ranges to roughly the
        // frame's own half-FOV (~25-32deg at the frame edge, depending on
        // aspect ratio), nothing close to navigation's full +-180deg swing —
        // so these get their own, much narrower ramp instead of reusing
        // NAV_RAMP_END_DEG (which would mean volume rarely nearing full even
        // at the frame edge).
        private const val TRACKING_H_DEADZONE_DEG = 5f
        private const val TRACKING_H_RAMP_END_DEG = 25f
        private const val TRACKING_V_DEADZONE_DEG = 5f
        private const val TRACKING_V_RAMP_END_DEG = 25f
    }
}
