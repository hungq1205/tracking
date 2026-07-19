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

    // Guiding mode (MappingService-backed global route: RTAB-Map pose +
    // occupancy grid + LocalPathPlanner's A*). Walking mode uses neither —
    // it has no MappingService stream at all any more (see
    // ToolDispatcher.runAvoidanceTick()).
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

    // Local reactive HRTF obstacle-dodge (walking AND guiding — see
    // ToolDispatcher.runAvoidanceTick() / TraversabilityScorer). The
    // beacon's current EMA-smoothed azimuth, carried across ticks so
    // smoothing has something to smooth FROM; null at the start of a fresh
    // walking/guiding session (nothing to smooth from yet) and whenever a
    // tick mutes (kept as-is while muted, so a brief mute doesn't reset
    // continuity — see runAvoidanceTick()'s own comment).
    var smoothedBeaconAzimuthDeg: Float? = null

    fun reset() {
        mode = "idle"
        readingBuffer = ""; readingLabel = ""; readingDirection = "ltr"; pageSummaries.clear()
        trackingTarget = ""
        guidingDestinationLabel = ""; navWaypoints = emptyList(); navWaypointIdx = 0
        lastMappingGrid = null; mutableGrid = null
        smoothedBeaconAzimuthDeg = null
    }
}
