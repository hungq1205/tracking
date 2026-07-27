package com.tracking.client.live

import kotlin.math.hypot

/**
 * Stateless polyline-following geometry — replaced LocalPathPlanner.kt's
 * role once path PLANNING moved server-side (see CLAUDE.md's
 * "Server-planned walking path" note).
 *
 * SUPERSEDED — unreferenced, left in place (same "kept in case revisited"
 * precedent this codebase uses elsewhere, e.g. HrtfBeaconPlayer.kt,
 * beacon_preview.py). ToolDispatcher.kt's steerAlongMainPath()/
 * computeSubPath() replaced the whole-path continuous-projection design
 * this class implemented with per-JOINT navigation (state.mainPathIdx) +
 * a 2Hz locally-recomputed obstacle-dodging sub-path — see CLAUDE.md's
 * "Main-path/sub-path joint navigation" note for the full redesign and why.
 */
object PathPursuit {

    /**
     * Projects [currentXz] onto the polyline [path] (world x/z, in order
     * from start to end): finds the closest point on any segment (clamped
     * to that segment, not the infinite line), and returns that point
     * together with its cumulative arc-length from path[0]. Null only for
     * an empty path.
     */
    fun nearestPointOnPath(
        path: List<Pair<Float, Float>>,
        currentXz: Pair<Float, Float>,
    ): Pair<Pair<Float, Float>, Float>? {
        if (path.isEmpty()) return null
        if (path.size == 1) return path[0] to 0f

        var bestPoint = path[0]
        var bestArcLen = 0f
        var bestDistSq = Float.MAX_VALUE
        var cumulative = 0f
        for (i in 0 until path.size - 1) {
            val a = path[i]; val b = path[i + 1]
            val segDx = b.first - a.first
            val segDz = b.second - a.second
            val segLenSq = segDx * segDx + segDz * segDz
            val segLen = kotlin.math.sqrt(segLenSq)
            val t = if (segLenSq > 1e-9f) {
                (((currentXz.first - a.first) * segDx + (currentXz.second - a.second) * segDz) / segLenSq)
                    .coerceIn(0f, 1f)
            } else 0f
            val projX = a.first + segDx * t
            val projZ = a.second + segDz * t
            val dx = currentXz.first - projX
            val dz = currentXz.second - projZ
            val distSq = dx * dx + dz * dz
            if (distSq < bestDistSq) {
                bestDistSq = distSq
                bestPoint = projX to projZ
                bestArcLen = cumulative + t * segLen
            }
            cumulative += segLen
        }
        return bestPoint to bestArcLen
    }

    /**
     * Walks forward along [path] by [lookaheadM] starting from
     * [arcLength] (as returned by [nearestPointOnPath]), clamping at the
     * path's final point if the look-ahead would run past the end.
     */
    fun advanceAlongPath(path: List<Pair<Float, Float>>, arcLength: Float, lookaheadM: Float): Pair<Float, Float> {
        if (path.isEmpty()) return 0f to 0f
        if (path.size == 1) return path[0]

        val target = arcLength + lookaheadM
        var cumulative = 0f
        for (i in 0 until path.size - 1) {
            val a = path[i]; val b = path[i + 1]
            val segLen = hypot((b.first - a.first).toDouble(), (b.second - a.second).toDouble()).toFloat()
            if (target <= cumulative + segLen) {
                val t = if (segLen > 1e-6f) ((target - cumulative) / segLen).coerceIn(0f, 1f) else 0f
                return (a.first + (b.first - a.first) * t) to (a.second + (b.second - a.second) * t)
            }
            cumulative += segLen
        }
        return path.last()
    }
}
