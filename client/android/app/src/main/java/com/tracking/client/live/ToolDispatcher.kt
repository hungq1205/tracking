package com.tracking.client.live

import android.util.Log
import com.tracking.client.audio.HrtfBeaconPlayer
import com.tracking.client.device.DeviceToolHandler
import com.tracking.client.grpc.GrpcClientManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject
import tracking.Tracking
import java.util.concurrent.TimeUnit

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
    private val scope: CoroutineScope,
    private val latestFrame: () -> ByteArray?,
    private val sendVideoFrame: (ByteArray) -> Unit,
    private val sendSystemNote: (String) -> Unit,
    private val playPcm: (ByteArray) -> Unit,
    private val onTrackingStateChanged: (active: Boolean, target: String) -> Unit,
    private val onGuidanceUpdate: (mode: String, waypoints: List<Pair<Float, Float>>) -> Unit,
    private val hrtfBeacon: HrtfBeaconPlayer,
) {
    private var visionStreamJob: Job? = null
    private var mappingJob: Job? = null
    private var mappingChunkChannel: Channel<Tracking.MappingChunk>? = null

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
            onTrackingStateChanged(false, "")
        }
        if (state.mode == "guiding" || state.mode == "walking" || state.mode == "scanning") {
            stopMappingStream()
        }
        hrtfBeacon.stop()
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

    suspend fun dispatch(name: String, args: JSONObject): JSONObject = try {
        when (name) {
            "get_current_time" -> toolGetCurrentTime()
            "get_latest_frame" -> toolGetLatestFrame()
            "start_vision_stream" -> toolStartVisionStream()
            "stop_vision_stream" -> toolStopVisionStream()
            "run_detection" -> toolRunDetection(args.optString("object_description", ""))
            "check_obstacle" -> toolCheckObstacle()

            "enter_reading_mode" -> toolEnterReadingMode(args.optString("label", ""))
            "scan_current_view" -> toolScanCurrentView()
            "get_reading_section" -> toolGetReadingSection(args.optString("query", ""))
            "read_aloud" -> toolReadAloud(args.optString("scope", "new"))
            "flip_reading_direction" -> toolFlipReadingDirection()
            "exit_reading_mode" -> toolExitReadingMode()

            "start_tracking" -> toolStartTracking(args.optString("target", ""))
            "stop_tracking" -> toolStopTracking()
            "get_object_from_memory" -> toolGetObjectFromMemory(args.optString("query", ""))

            "query_memory" -> toolQueryMemory(args.optString("question", ""))
            "save_memory" -> toolSaveMemory(args.optString("label", ""), args.optString("note", ""))
            "remember_object" -> toolRememberObject(args.optString("label", ""), args.optString("description", ""))
            "list_memory_labels" -> toolListMemoryLabels()

            "start_guiding" -> toolStartGuiding(args.optString("destination", ""))
            "stop_guiding" -> toolStopGuiding()
            "get_current_location" -> toolGetCurrentLocation()

            "start_walking" -> toolStartWalking()
            "stop_walking" -> toolStopWalking()

            "start_scan" -> toolStartScan()
            "stop_scan" -> toolStopScan()

            "search_youtube", "get_video_info" ->
                JSONObject().put("error", "Music search is not yet available on-device (server-side yt-dlp utility was not ported in this pass).")

            "make_phone_call", "search_contacts", "set_alarm", "create_calendar_event", "play_video", "stop_music" ->
                dispatchDeviceTool(name, args)

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

    private fun toolEnterReadingMode(label: String): JSONObject {
        stopActiveModes()
        state.mode = "reading"
        state.readingBuffer = ""
        state.pageSummaries.clear()
        state.readingLabel = label.ifBlank { "reading" }
        state.readingDirection = "ltr"
        reportMode("reading", state.readingLabel)
        return JSONObject().put("status", "reading_mode_active").put("label", state.readingLabel)
    }

    private suspend fun toolScanCurrentView(): JSONObject {
        val frame = latestFrame() ?: return JSONObject().put("error", "No frame available. Point the camera at the text.")
        val text = ocrClient.readText(frame)
        if (text.isBlank()) return JSONObject().put("found", false).put("message", "No text detected in current view.")
        val newText = MemoryTextUtils.filterNewSentences(text, state.readingBuffer)
        if (newText.isBlank()) return JSONObject().put("found", false).put("message", "No new text (already scanned).")
        state.readingBuffer = if (state.readingBuffer.isEmpty()) newText else "${state.readingBuffer}\n$newText"
        val summary = newText.split(" ").take(30).joinToString(" ") + if (newText.split(" ").size > 30) "..." else ""
        state.pageSummaries.add(summary)
        return JSONObject()
            .put("found", true).put("new_text", newText)
            .put("page_count", state.pageSummaries.size).put("summary", summary)
    }

    private fun toolGetReadingSection(query: String): JSONObject {
        if (state.readingBuffer.isEmpty()) return JSONObject().put("error", "No text has been scanned yet. Use scan_current_view() first.")
        if (state.readingBuffer.length < 1200) return JSONObject().put("text", state.readingBuffer).put("source", "full_buffer")
        val chunks = splitChunks(state.readingBuffer)
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

    private suspend fun toolReadAloud(scope: String): JSONObject {
        val text = if (scope == "all") {
            if (state.readingBuffer.isEmpty()) return JSONObject().put("error", "No text has been scanned yet.")
            state.readingBuffer
        } else {
            val scanResult = toolScanCurrentView()
            if (scanResult.has("error") || !scanResult.optBoolean("found", false)) return scanResult
            scanResult.getString("new_text")
        }
        val stub = grpc.perceptionStub ?: return JSONObject().put("error", "Not connected.")
        try {
            stub.synthesize(Tracking.SynthesizeRequest.newBuilder().setText(text).build())
                .collect { chunk -> playPcm(chunk.pcmData.toByteArray()) }
        } catch (e: Exception) {
            return JSONObject().put("error", "TTS failed: ${e.message}")
        }
        return JSONObject().put("status", "read_aloud").put("scope", scope).put("chars", text.length)
    }

    private fun toolFlipReadingDirection(): JSONObject {
        state.readingDirection = if (state.readingDirection == "rtl") "ltr" else "rtl"
        return JSONObject().put("direction", state.readingDirection)
    }

    private fun toolExitReadingMode(): JSONObject {
        val label = state.readingLabel
        val chars = state.readingBuffer.length
        state.mode = "idle"; state.readingBuffer = ""; state.pageSummaries.clear(); state.readingLabel = ""
        reportMode("idle")
        return JSONObject().put("status", "exited").put("label", label).put("chars_discarded", chars)
    }

    // ── Tracking ─────────────────────────────────────────────────────────

    private fun toolStartTracking(target: String): JSONObject {
        if (target.isBlank()) return JSONObject().put("error", "target required")
        stopActiveModes()
        state.mode = "tracking"; state.trackingTarget = target
        onTrackingStateChanged(true, target)
        hrtfBeacon.start()  // plays constantly but muted until the target is visible — see updateTrackingBeacon()
        reportMode("tracking", target)
        return JSONObject().put("status", "tracking_started").put("target", target)
    }

    private fun toolStopTracking(): JSONObject {
        state.mode = "idle"; state.trackingTarget = ""
        onTrackingStateChanged(false, "")
        hrtfBeacon.stop()
        reportMode("idle")
        return JSONObject().put("status", "tracking_stopped")
    }

    /** Points the continuous HRTF beacon at the currently-tracked object's
     * on-screen position (see HrtfBeacon.directionFromBox) — called from
     * MainViewModel's frame loop on every local ORB tracking update while
     * mode == "tracking". Mutes when the target isn't currently visible. */
    fun updateTrackingBeacon(visible: Boolean, centerX: Float, centerY: Float, frameWidth: Int, frameHeight: Int) {
        if (!visible || frameWidth <= 0 || frameHeight <= 0) {
            hrtfBeacon.mute()
            return
        }
        val dir = HrtfBeacon.directionFromBox(centerX, centerY, frameWidth, frameHeight)
        hrtfBeacon.updateDirection(dir.azimuthDeg, dir.elevationDeg, dir.distanceM)
    }

    private suspend fun toolGetObjectFromMemory(query: String): JSONObject {
        val stub = grpc.perceptionStub ?: return JSONObject().put("error", "Not connected.")
        val queryVec = try {
            stub.embed(Tracking.EmbedRequest.newBuilder().setText(query).build()).vectorList.toFloatArray()
        } catch (e: Exception) { return JSONObject().put("error", "Embed failed: ${e.message}") }
        val matches = memoryStore.queryGlobal(queryVec, topK = 1).filter { it.score > 0.5f }
        if (matches.isEmpty()) return JSONObject().put("found", false)
        val m = matches.first()
        return JSONObject().put("found", true).put("label", m.label).put("description", m.text).put("score", m.score)
    }

    // ── Memory ───────────────────────────────────────────────────────────

    private suspend fun toolQueryMemory(question: String): JSONObject {
        val stub = grpc.perceptionStub ?: return JSONObject().put("error", "Not connected.")
        val queryVec = try {
            stub.embed(Tracking.EmbedRequest.newBuilder().setText(question).build()).vectorList.toFloatArray()
        } catch (e: Exception) { return JSONObject().put("error", "Embed failed: ${e.message}") }
        val matches = memoryStore.queryGlobal(queryVec, topK = 5).filter { it.score > 0.5f }
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

    private suspend fun toolRememberObject(label: String, description: String): JSONObject {
        if (label.isBlank() || description.isBlank()) return JSONObject().put("error", "label and description required")
        memoryStore.append(label, description, source = "object_description")
        embedAndStore(label, description)
        return JSONObject().put("status", "remembered").put("label", label)
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

    // ── Guiding (live MappingService-backed navigation) ─────────────────

    private suspend fun toolStartGuiding(destination: String): JSONObject {
        if (destination.isBlank()) return JSONObject().put("error", "destination required")
        stopActiveModes()
        state.mode = "guiding"
        state.guidingDestinationLabel = destination
        startMappingStream()
        hrtfBeacon.start()
        onGuidanceUpdate("guiding", state.navWaypoints)
        reportMode("guiding", destination)
        return JSONObject().put("status", "guiding_started").put("destination", destination)
            .put("note", "Route will be announced once the destination landmark has been located in the live map.")
    }

    private fun toolStopGuiding(): JSONObject {
        stopMappingStream()
        hrtfBeacon.stop()
        state.mode = "idle"; state.guidingDestinationLabel = ""; state.navWaypoints = emptyList()
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
        val channel = Channel<Tracking.MappingChunk>(Channel.UNLIMITED)
        mappingChunkChannel = channel
        mappingJob = scope.launch(Dispatchers.IO) {
            try {
                stub.updateMapping(channel.receiveAsFlow()).collect { update ->
                    state.lastMappingPose = update.pose
                    if (update.gridUpdated) {
                        // full_resync: replace wholesale (first update for this
                        // stream, or the map's explored bounds grew). Otherwise
                        // patch the existing grid in place from grid_delta — see
                        // MutableOccupancyGrid / mapping_servicer.py's UpdateMapping
                        // for why: re-shipping the whole grid every update doesn't
                        // scale as a session/map grows.
                        if (update.fullResync) {
                            state.mutableGrid = MutableOccupancyGrid.fromFull(update.grid)
                        } else {
                            val mg = state.mutableGrid
                            if (mg == null) {
                                Log.w(TAG, "Received a grid delta with no prior full_resync — dropping")
                            } else {
                                mg.applyDelta(update.gridDelta)
                            }
                        }
                        state.lastMappingGrid = state.mutableGrid?.toProto()
                        recomputeRoute()
                    }
                    checkWaypointProgress()
                    updateHrtfBeacon()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Mapping stream ended: ${e.message}")
            }
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
     * MainViewModel's frame loop while guiding/walking/scanning is active.
     * Not every mode needs this (reading/tracking/idle don't), so the
     * caller gates it. session_mode tells the server which pipeline to run
     * (see mapping_servicer.py's UpdateMapping / scan_session.py's
     * walking_lite) — only actually consulted server-side on the stream's
     * first chunk, but cheap to set on every one rather than special-case
     * the first. */
    fun feedMappingFrame(jpeg: ByteArray) {
        val channel = mappingChunkChannel ?: return
        val sessionMode = when (state.mode) {
            "walking" -> Tracking.SessionMode.WALKING
            "guiding" -> Tracking.SessionMode.GUIDING
            else -> Tracking.SessionMode.SCAN  // "scanning" and any other caller
        }
        channel.trySend(
            Tracking.MappingChunk.newBuilder()
                .setLocationId(locationId)
                .setImageData(com.google.protobuf.ByteString.copyFrom(jpeg))
                .setFrameTimestampNs(TimeUnit.MILLISECONDS.toNanos(System.currentTimeMillis()))
                .setPoseSource(Tracking.PoseSource.RTABMAP)
                .setSessionMode(sessionMode)
                .build()
        )
    }

    /**
     * Resolves state.guidingDestinationLabel to world (x, z) via the
     * FindLandmark RPC, then routes to it with the local A* planner.
     * MappingUpdate.landmarks stays empty for the duration of an active
     * stream now (GroundingDINO/backprojection is deferred server-side —
     * see CLAUDE.md's "Client-Orchestrated Live Session" section), so this
     * always resolves via a fresh FindLandmark call rather than checking a
     * live-streamed landmarks list first; FindLandmark itself already does
     * a cheap tag-match-first, GroundingDINO-fallback lookup server-side.
     */
    private suspend fun recomputeRoute() {
        val grid = state.lastMappingGrid ?: return
        val pose = state.lastMappingPose ?: return
        val destination = state.guidingDestinationLabel
        if (destination.isBlank()) return

        val stub = grpc.mappingStub ?: return
        val target = try {
            val resp = stub.findLandmark(
                Tracking.FindLandmarkRequest.newBuilder()
                    .setLocationId(locationId)
                    .setQuery(destination)
                    .build()
            )
            if (resp.found) resp.x to resp.z else null
        } catch (e: Exception) {
            Log.w(TAG, "FindLandmark failed for '$destination': ${e.message}")
            null
        } ?: return  // not resolved yet — try again on the next grid update

        val result = LocalPathPlanner(grid).findPath(pose.x to pose.z, target) ?: return
        state.navWaypoints = result.waypoints
        state.navWaypointIdx = 0
        state.navConfirmed = result.confirmed
        onGuidanceUpdate(state.mode, state.navWaypoints)
    }

    /** Points the continuous HRTF beacon (see HrtfBeaconPlayer) — at the
     * current waypoint for guiding, or at the most open direction ahead for
     * walking (no destination to route toward — see LocalPathPlanner.
     * findMostOpenDirection(), which replaced the old fixed-interval
     * AnalyzeFrame(DEPTH) polling + spoken obstacle alert entirely). Mutes
     * when there's nothing useful to point at yet. Called on every
     * MappingUpdate alongside checkWaypointProgress(). */
    private fun updateHrtfBeacon() {
        val pose = state.lastMappingPose ?: run { hrtfBeacon.mute(); return }

        if (state.mode == "walking") {
            val grid = state.lastMappingGrid
            val az = grid?.let { LocalPathPlanner(it).findMostOpenDirection(pose) }
            if (az == null) {
                hrtfBeacon.mute()
            } else {
                hrtfBeacon.updateDirection(az, 0f, OPEN_DIRECTION_DISTANCE_M)
            }
            return
        }

        if (state.navWaypoints.isEmpty() || state.navWaypointIdx >= state.navWaypoints.size) {
            hrtfBeacon.mute()
            return
        }
        val (wx, wz) = state.navWaypoints[state.navWaypointIdx]
        val dir = HrtfBeacon.directionTo(pose, wx, wz)
        hrtfBeacon.updateDirection(dir.azimuthDeg, dir.elevationDeg, dir.distanceM)
    }

    private fun checkWaypointProgress() {
        val pose = state.lastMappingPose ?: return
        if (state.navWaypoints.isEmpty() || state.navWaypointIdx >= state.navWaypoints.size) return
        val (wx, wz) = state.navWaypoints[state.navWaypointIdx]
        val dist = kotlin.math.sqrt((wx - pose.x) * (wx - pose.x) + (wz - pose.z) * (wz - pose.z))
        if (dist < 1.0f) {
            state.navWaypointIdx++
            if (state.navWaypointIdx >= state.navWaypoints.size) {
                sendSystemNote("[SYSTEM] Arrived at ${state.guidingDestinationLabel}.")
            } else {
                sendSystemNote("[SYSTEM] Waypoint reached, continuing toward ${state.guidingDestinationLabel}.")
            }
        }
    }

    // ── Walking (free-walk guiding — ambient HRTF only, no destination,
    //    no spoken obstacle alerts; see LocalPathPlanner.findMostOpenDirection
    //    and updateHrtfBeacon() above) ──────────────────────────────────────

    private fun toolStartWalking(): JSONObject {
        stopActiveModes()
        state.mode = "walking"
        startMappingStream()
        hrtfBeacon.start()  // plays constantly but muted until the grid has an open direction — see updateHrtfBeacon()
        onGuidanceUpdate("walking", emptyList())
        reportMode("walking")
        return JSONObject().put("status", "walking_started")
    }

    private fun toolStopWalking(): JSONObject {
        stopMappingStream()
        hrtfBeacon.stop()
        state.mode = "idle"
        onGuidanceUpdate("idle", emptyList())
        reportMode("idle")
        return JSONObject().put("status", "walking_stopped")
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
        stopMappingStream()
        hrtfBeacon.stop()
    }

    companion object {
        private const val TAG = "ToolDispatcher"

        // Nominal, not a real measurement — "most open direction" is a
        // direction, not a point target, so there's no real distance to
        // report. Chosen so HrtfBeaconPlayer's distance-based gain sits
        // mid-range (audible, not maxed), same reasoning as
        // HrtfBeacon.directionFromBox()'s own fixed distanceM.
        private const val OPEN_DIRECTION_DISTANCE_M = 5f
    }
}
