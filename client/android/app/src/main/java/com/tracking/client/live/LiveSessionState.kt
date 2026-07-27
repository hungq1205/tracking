package com.tracking.client.live

import tracking.Tracking

/**
 * Kotlin port of server/live_session.py's LiveSessionState — per-session
 * state now held on-device since orchestration runs here (see CLAUDE.md's
 * "Client-Orchestrated Live Session" section).
 */
class LiveSessionState {
    var mode: String = "idle" // idle | reading | tracking | guiding | walking | scanning

    // Reading mode — one entry per OCR pass that survived OcrBlockFilters'
    // raw-first fuzzy dedup (integrateRawBlock()), NOT one entry per
    // sentence/line — see ToolDispatcher.toolScanCurrentView(). Each block
    // holds RAW OCR text (used for all similarity/stitch matching) and
    // `corrected` (mirrors raw until an async Gemini correction patches it
    // in place — see ToolDispatcher's correction queue). Replaced a flat
    // `readingBuffer: String` + sentence-containment dedup, which rarely
    // recognized a noisy reread of the same page as "already read" and let
    // duplicates pile up; later replaced a plain `MutableList<String>`
    // (integrateBlock()) once OCR correction needed a stable per-block id
    // to patch results into asynchronously without racing new captures.
    var readingBlocks: MutableList<OcrBlockFilters.ReadingBlock> = mutableListOf()
    var readingLabel: String = ""
    var readingDirection: String = "ltr"
    var pageSummaries: MutableList<String> = mutableListOf()

    /** Index into `splitIntoSentences(readingBufferText())` — "how far
     * we've read." Replaces the old pendingReadingContinuation (a frozen
     * snapshot list captured only on interruption) with a single persistent
     * cursor that ToolDispatcher.speakSentencesFrom() advances after EVERY
     * sentence (not just on cancellation) — so it's always accurate whether
     * reading finishes normally, is interrupted, or errors out, and
     * continue_reading() can resume against the CURRENT (possibly grown by
     * live reading since) buffer instead of a stale snapshot. Known,
     * accepted imprecision: if an earlier block is later REPLACED with a
     * longer/different capture (OcrBlockFilters' "updated" case), sentence
     * boundaries before the cursor can shift, so the cursor is a best-effort
     * position, not a byte-exact guarantee. */
    var readingCursorSentenceIndex: Int = 0

    /** The reading buffer as one string (CORRECTED text — == raw for any
     * block whose correction hasn't landed yet), for callers that just
     * want the accumulated text (get_reading_section, read_aloud). */
    fun readingBufferText(): String = readingBlocks.joinToString("\n\n") { it.corrected }

    // Tracking mode
    var trackingTarget: String = ""

    // MappingService-backed live navigation — path PLANNING is now entirely
    // server-side (scan_server/live_path_planner.py, via
    // MappingUpdate.planned_path — see CLAUDE.md's "Server-planned walking
    // path" note); the client only follows it. GUIDING resolves
    // guidingDestinationLabel via FindLandmark as before, then sends the
    // resulting world point back to the server as guidingGoalXz (on every
    // subsequent MappingChunk, via has_goal/goal_x/goal_z) so the server
    // knows what to plan toward; WALKING sends no goal at all — the server
    // infers "keep walking forward" from its own RTAB-Map pose's heading.
    var guidingDestinationLabel: String = ""
    var guidingGoalXz: Pair<Float, Float>? = null
    // The server-planned MAIN path's joints (no prepended pose — see
    // ToolDispatcher.kt's "Main-path/sub-path joint navigation" note; this
    // used to be prepended with the pose for PathPursuit's whole-path
    // projection, which no longer runs). Replaced by a fresh list from
    // every MappingUpdate — the client tracks progress along it itself via
    // mainPathIdx, independent of how often the server actually replans.
    var plannedPath: List<Pair<Float, Float>> = emptyList()
    var pathConfirmed: Boolean = true
    // Index of the main-path joint currently being navigated toward — the
    // "retained joint" the user asked for, in place of the earlier (now
    // reverted) server-side 6s whole-path dwell lock. Advanced locally by
    // ToolDispatcher's 2Hz sub-path tick once the user arrives at the
    // current joint with no dodge sub-point still pending; reset to 0
    // whenever a fresh main path arrives from the server (a fresh path is
    // already computed relative to the CURRENT pose, so its own joint 0 is
    // the correct next target).
    var mainPathIdx: Int = 0

    // The last AUTHORITATIVE pose MappingService actually reported (RTAB-Map,
    // server-computed) — NOT the client's current best-estimate position,
    // which is derived on demand (see HrtfBeacon.extrapolate()) by folding
    // this pose forward through rotationTracker's accumulated rotation +
    // pdrStepEstimator's accumulated distance since this was received. Kept
    // separate (not overwritten by extrapolation) so get_current_location
    // and the next reconciliation always have the real last-known-good fix
    // to work from.
    var lastMappingPose: Tracking.Pose? = null

    // Latest-sent snapshot for latency compensation (see ToolDispatcher's
    // mapping-stream collector and CLAUDE.md's "Server-planned walking
    // path" note): the server's response describes an already-slightly-
    // stale pose (network + RTAB-Map/DA3 processing latency), so
    // ToolDispatcher fast-forwards it by whatever the rotation/PDR
    // accumulators have added since this snapshot was taken, rather than
    // accepting the response as "now" as-is. A SINGLE snapshot, not a
    // history buffer — the server now only ever processes the latest
    // available frame (see mapping_servicer.py's _latest_only_chunks(),
    // "Drop-to-latest mapping-chunk ingestion"), so a response won't
    // reliably correspond to a specific earlier send anyway; this is used
    // whenever present regardless of exact timestamp match, an accepted
    // approximation given how tight the loop runs.
    data class PoseSendSnapshot(val rotationAccum: FloatArray, val distanceAccum: Float)
    var lastSentSnapshot: Pair<Long, PoseSendSnapshot>? = null

    fun reset() {
        mode = "idle"
        readingBlocks.clear(); readingLabel = ""; readingDirection = "ltr"; pageSummaries.clear()
        readingCursorSentenceIndex = 0
        trackingTarget = ""
        guidingDestinationLabel = ""; guidingGoalXz = null
        plannedPath = emptyList(); pathConfirmed = true; mainPathIdx = 0
        lastSentSnapshot = null
    }
}
