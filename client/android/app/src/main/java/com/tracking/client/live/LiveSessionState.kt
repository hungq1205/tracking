package com.tracking.client.live

import tracking.Tracking

/**
 * Kotlin port of server/live_session.py's LiveSessionState — per-session
 * state now held on-device since orchestration runs here (see CLAUDE.md's
 * "Client-Orchestrated Live Session" section).
 */
class LiveSessionState {
    var mode: String = "idle" // idle | reading | tracking | guiding | walking | scanning

    // Reading mode
    var readingBuffer: String = ""
    var readingLabel: String = ""
    var readingDirection: String = "ltr"
    var pageSummaries: MutableList<String> = mutableListOf()

    // Tracking mode
    var trackingTarget: String = ""

    // Guiding mode (MappingService-backed live navigation) — also used by
    // walking mode for lastMappingPose/lastMappingGrid (walking has no
    // destination, so guidingDestinationLabel/navWaypoints stay unused for
    // it; see ToolDispatcher.updateHrtfBeacon()'s "walking" branch, which
    // reads the grid directly instead).
    var guidingDestinationLabel: String = ""
    var navWaypoints: List<Pair<Float, Float>> = emptyList()
    var navWaypointIdx: Int = 0
    var navConfirmed: Boolean = true
    var lastMappingPose: Tracking.Pose? = null
    var lastMappingGrid: Tracking.OccupancyGrid? = null
    // Persistent, delta-patched backing store for lastMappingGrid — see
    // MutableOccupancyGrid's docstring. Replaced wholesale on a
    // full_resync MappingUpdate, patched in place otherwise. null until
    // the first full_resync of a stream.
    var mutableGrid: MutableOccupancyGrid? = null

    fun reset() {
        mode = "idle"
        readingBuffer = ""; readingLabel = ""; readingDirection = "ltr"; pageSummaries.clear()
        trackingTarget = ""
        guidingDestinationLabel = ""; navWaypoints = emptyList(); navWaypointIdx = 0
        lastMappingGrid = null; mutableGrid = null
    }
}
