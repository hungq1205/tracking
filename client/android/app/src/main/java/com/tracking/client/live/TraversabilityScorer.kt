package com.tracking.client.live

import kotlin.math.abs

/**
 * Local reactive obstacle-dodge scoring — picks a steering azimuth from a
 * per-frame TraversabilityInfo fan (tracking.proto / server/tools/
 * traversability.py's estimate_traversability), optionally biased toward a
 * known goal bearing (guiding) or not (walking, no destination). Classic
 * Vector-Field-Histogram-style: score each candidate direction by its own
 * corridor-windowed clearance, penalized by how far it is from the goal
 * bearing and from the current (already-smoothed) heading, then pick the
 * peak. See CLAUDE.md's "Local reactive HRTF obstacle-dodge" note for the
 * full design discussion — this replaces LocalPathPlanner's old
 * occupancy-grid findMostOpenDirection()/castOpenRay() for walking, and
 * adds a goal-biased local dodge layer on top of guiding's existing global
 * A* route.
 */
object TraversabilityScorer {

    /**
     * [clearanceM]: per-bin nearest-obstacle clearance, azimuth
     * (minAngleDeg + i*stepDeg) per index i — straight from
     * TraversabilityInfo. [corridorHalfWidthDeg]: a candidate direction's
     * score uses the MINIMUM clearance over a small window of bins around
     * it (approximates the user's body needing to actually fit through a
     * gap, not just one ray missing an obstacle). [goalAzimuthDeg]:
     * egocentric bearing to pull toward — null for walking (no
     * destination), HrtfBeacon.directionTo(...).azimuthDeg for guiding.
     * [currentAzimuthDeg]: the beacon's current (already smoothed) azimuth,
     * penalizing large steering jumps — null on a fresh session's first
     * tick. Returns null when even the best candidate's own corridor
     * clearance is below [clearanceFloorM] — nothing safe to point at, mute.
     */
    fun pickSteeringAngle(
        clearanceM: List<Float>,
        minAngleDeg: Float,
        stepDeg: Float,
        corridorHalfWidthDeg: Float = 8f,
        goalAzimuthDeg: Float? = null,
        currentAzimuthDeg: Float? = null,
        weightGoal: Float = 0.6f,
        weightSteer: Float = 0.15f,
        clearanceFloorM: Float = 0.5f,
    ): Float? {
        val n = clearanceM.size
        if (n == 0 || stepDeg <= 0f) return null
        val windowBins = (corridorHalfWidthDeg / stepDeg).toInt().coerceAtLeast(0)

        var bestIdx = -1
        var bestScore = Float.NEGATIVE_INFINITY
        var bestCorridorClearance = 0f

        for (i in 0 until n) {
            var corridorClearance = clearanceM[i]
            for (offset in -windowBins..windowBins) {
                val j = i + offset
                if (j in 0 until n) corridorClearance = minOf(corridorClearance, clearanceM[j])
            }
            val azimuth = minAngleDeg + i * stepDeg
            var score = corridorClearance
            if (goalAzimuthDeg != null) score -= weightGoal * abs(azimuth - goalAzimuthDeg)
            if (currentAzimuthDeg != null) score -= weightSteer * abs(azimuth - currentAzimuthDeg)
            if (score > bestScore) {
                bestScore = score
                bestIdx = i
                bestCorridorClearance = corridorClearance
            }
        }

        if (bestIdx < 0 || bestCorridorClearance < clearanceFloorM) return null
        return minAngleDeg + bestIdx * stepDeg
    }

    /**
     * Exponential smoothing toward [target], shortest-path around the
     * ±180° wrap (not strictly needed given the fan's bounded FOV, but
     * correctness is cheap and this stays correct if the fan range ever
     * widens). Returns [target] directly when [current] is null — a fresh
     * session's first tick has nothing to smooth from yet.
     */
    fun smoothAzimuth(current: Float?, target: Float, alpha: Float = 0.35f): Float {
        if (current == null) return target
        var diff = (target - current) % 360f
        if (diff > 180f) diff -= 360f
        if (diff < -180f) diff += 360f
        return current + alpha * diff
    }
}
