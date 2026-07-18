package com.tracking.client.live

import tracking.Tracking
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import java.util.PriorityQueue

/**
 * Kotlin port of scan_server/live_path_planner.py's LiveGridPathPlanner —
 * runs entirely on-device against the OccupancyGrid streamed back from
 * MappingService.UpdateMapping (server computes the grid; path-finding is
 * cheap graph search with no model dependency, so it runs locally to avoid
 * a network round trip per HRTF beacon update — see CLAUDE.md's
 * "Client-Orchestrated Live Session" section). Same cost model, same
 * 8-connected octile-heuristic A*, same closest-approach fallback and
 * clearance-aware string-pulling as the Python original.
 */
object GridClass {
    const val UNKNOWN = 0
    const val GROUND = 1
    const val LOW_STEP_OVER = 2
    const val OBSTACLE = 3
}

data class PathResult(
    val waypoints: List<Pair<Float, Float>>, // world (x, z), start not included
    val confirmed: Boolean,
    val reachedExactly: Boolean,
)

class LocalPathPlanner(
    private val grid: Tracking.OccupancyGrid,
    private val minPathClearanceM: Float = 0.0f,
) {
    private val resolution = grid.cellSize
    private val originX = grid.originX
    private val originZ = grid.originZ
    private val width = grid.width
    private val height = grid.height
    private val cls = grid.clsList
    private val clearance = grid.clearanceMList

    companion object {
        private const val CLEARANCE_PENALTY_SCALE = 4.0
        private const val CLEARANCE_DECAY_RATE = 8.0
        private const val NARROW_PENALTY_SCALE = 8.0
        private val COST_BY_CLASS = mapOf(
            GridClass.GROUND to 1.0,
            GridClass.UNKNOWN to 1.5,
            GridClass.LOW_STEP_OVER to 3.0,
        )
        private const val BLOCKED = Double.POSITIVE_INFINITY
    }

    private fun idx(row: Int, col: Int) = row * width + col

    fun worldToCell(x: Float, z: Float): Pair<Int, Int> {
        val col = ((x - originX) / resolution).toInt()
        val row = ((z - originZ) / resolution).toInt()
        return row to col
    }

    private fun cellToWorld(row: Int, col: Int): Pair<Float, Float> {
        val x = originX + (col + 0.5f) * resolution
        val z = originZ + (row + 0.5f) * resolution
        return x to z
    }

    private fun inBounds(row: Int, col: Int) = row in 0 until height && col in 0 until width

    private fun clearanceMultiplier(row: Int, col: Int): Double {
        if (clearance.isEmpty()) return 1.0
        val c = clearance[idx(row, col)].toDouble()
        var base = 1.0 + CLEARANCE_PENALTY_SCALE * exp(-CLEARANCE_DECAY_RATE * c)
        if (minPathClearanceM > 0.0 && c < minPathClearanceM) {
            val deficitFrac = (minPathClearanceM - c) / minPathClearanceM
            base *= 1.0 + NARROW_PENALTY_SCALE * deficitFrac
        }
        return base
    }

    private fun cost(row: Int, col: Int): Double {
        if (!inBounds(row, col)) return BLOCKED
        val c = cls[idx(row, col)]
        if (c == GridClass.OBSTACLE) return BLOCKED
        return (COST_BY_CLASS[c] ?: 1.0) * clearanceMultiplier(row, col)
    }

    private fun passable(row: Int, col: Int) =
        inBounds(row, col) && cls[idx(row, col)] != GridClass.OBSTACLE

    /**
     * "Steer toward the most open space" — walking mode's ambient HRTF
     * beacon has no destination/waypoint to route toward (unlike guiding,
     * which uses findPath() above), so it needs a different signal: which
     * direction, relative to the current heading, has the most clear space
     * ahead right now. Replaces the old fixed-interval AnalyzeFrame(DEPTH)
     * polling + spoken "[SYSTEM] Obstacle ~Xm ahead" alert (see CLAUDE.md's
     * "Mode exclusivity" / walking-mode notes) — this is continuous and
     * purely audio-directional, no spoken interruptions.
     *
     * Casts a short ray per candidate egocentric azimuth (via
     * HrtfBeacon.worldYawRad + straightforward 2D rotation — floor-
     * constrained navigation, pitch/roll don't matter) through this same
     * grid, stepping in `resolution`-sized increments up to [maxRangeM],
     * and returns whichever azimuth traveled farthest before hitting an
     * obstacle or leaving passable cells. Ties favor the smallest
     * |azimuth| (prefer continuing straight over an equally-open sharp
     * turn). Returns null only when even the very first step in every
     * candidate direction is already blocked/out of bounds.
     */
    fun findMostOpenDirection(
        pose: Tracking.Pose, maxRangeM: Float = 5f, coneDeg: Float = 90f, stepDeg: Float = 15f,
    ): Float? {
        val yawRad = HrtfBeacon.worldYawRad(pose)
        var bestAz: Float? = null
        var bestDist = 0f
        var az = -coneDeg
        while (az <= coneDeg) {
            val candidateRad = yawRad + Math.toRadians(az.toDouble())
            val dx = kotlin.math.sin(candidateRad).toFloat()
            val dz = kotlin.math.cos(candidateRad).toFloat()
            val dist = castOpenRay(pose.x, pose.z, dx, dz, maxRangeM)
            if (dist > bestDist || (dist == bestDist && bestAz != null && abs(az) < abs(bestAz))) {
                bestDist = dist
                bestAz = az
            }
            az += stepDeg
        }
        return bestAz
    }

    private fun castOpenRay(fromX: Float, fromZ: Float, dirX: Float, dirZ: Float, maxRangeM: Float): Float {
        val steps = (maxRangeM / resolution).toInt().coerceAtLeast(1)
        var traveled = 0f
        for (i in 1..steps) {
            val (row, col) = worldToCell(fromX + dirX * resolution * i, fromZ + dirZ * resolution * i)
            if (!passable(row, col)) break
            traveled = resolution * i
        }
        return traveled
    }

    /**
     * 8-connected A* with an octile heuristic, from startXz to goalXz (world
     * metres). Returns null only when nothing useful can be offered at all
     * (start itself not passable, or the search couldn't move anywhere).
     */
    fun findPath(startXz: Pair<Float, Float>, goalXz: Pair<Float, Float>): PathResult? {
        val start = worldToCell(startXz.first, startXz.second)
        val goal = worldToCell(goalXz.first, goalXz.second)
        if (!passable(start.first, start.second)) return null
        if (start == goal && passable(goal.first, goal.second)) {
            val confirmed = cls[idx(start.first, start.second)] != GridClass.UNKNOWN
            return PathResult(listOf(goalXz), confirmed, true)
        }

        val (raw, reachedExactly) = astar(start, goal) ?: return null
        val confirmed = raw.none { (r, c) -> cls[idx(r, c)] == GridClass.UNKNOWN }
        val simplified = simplify(raw)
        val endXz = if (reachedExactly) goalXz else cellToWorld(raw.last().first, raw.last().second)
        val waypoints = mutableListOf<Pair<Float, Float>>()
        for (i in 1 until simplified.size - 1) {
            waypoints.add(cellToWorld(simplified[i].first, simplified[i].second))
        }
        waypoints.add(endXz)
        return PathResult(waypoints, confirmed, reachedExactly)
    }

    private fun octile(a: Pair<Int, Int>, b: Pair<Int, Int>): Double {
        val dr = abs(a.first - b.first)
        val dc = abs(a.second - b.second)
        return (dr + dc) + (sqrt(2.0) - 2) * minOf(dr, dc)
    }

    private val neighborSteps = listOf(
        Triple(-1, 0, 1.0), Triple(1, 0, 1.0), Triple(0, -1, 1.0), Triple(0, 1, 1.0),
        Triple(-1, -1, sqrt(2.0)), Triple(-1, 1, sqrt(2.0)),
        Triple(1, -1, sqrt(2.0)), Triple(1, 1, sqrt(2.0)),
    )

    /** Returns (path, reachedExactly), or null if start couldn't be expanded at all. */
    private fun astar(start: Pair<Int, Int>, goal: Pair<Int, Int>): Pair<List<Pair<Int, Int>>, Boolean>? {
        fun reconstruct(node: Pair<Int, Int>, cameFrom: Map<Pair<Int, Int>, Pair<Int, Int>>): List<Pair<Int, Int>> {
            val path = mutableListOf(node)
            var cur = node
            while (cameFrom.containsKey(cur)) {
                cur = cameFrom.getValue(cur)
                path.add(cur)
            }
            path.reverse()
            return path
        }

        val openHeap = PriorityQueue<Pair<Double, Pair<Int, Int>>>(compareBy { it.first })
        openHeap.add(0.0 to start)
        val cameFrom = mutableMapOf<Pair<Int, Int>, Pair<Int, Int>>()
        val gScore = mutableMapOf(start to 0.0)
        val visited = mutableSetOf<Pair<Int, Int>>()
        var bestNode = start
        var bestDist = octile(start, goal)

        while (openHeap.isNotEmpty()) {
            val (_, current) = openHeap.poll()
            if (current in visited) continue
            visited.add(current)

            val d = octile(current, goal)
            if (d < bestDist) { bestDist = d; bestNode = current }
            if (current == goal) return reconstruct(current, cameFrom) to true

            val (r, c) = current
            for ((dr, dc, stepDist) in neighborSteps) {
                val nr = r + dr; val nc = c + dc
                if (!passable(nr, nc)) continue
                if (dr != 0 && dc != 0) {
                    if (!passable(r + dr, c) || !passable(r, c + dc)) continue
                }
                val stepCost = stepDist * cost(nr, nc)
                val tentativeG = gScore.getValue(current) + stepCost
                val neighbor = nr to nc
                if (tentativeG < (gScore[neighbor] ?: BLOCKED)) {
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentativeG
                    val fScore = tentativeG + octile(neighbor, goal)
                    openHeap.add(fScore to neighbor)
                }
            }
        }

        if (bestNode == start) return null
        return reconstruct(bestNode, cameFrom) to false
    }

    private fun lineCost(a: Pair<Int, Int>, b: Pair<Int, Int>): Double? {
        var (r, c) = a
        val (r1, c1) = b
        val dr = abs(r1 - r); val dc = abs(c1 - c)
        val sr = if (r1 > r) 1 else -1
        val sc = if (c1 > c) 1 else -1
        var err = dr - dc
        var total = 0.0
        var prevR = r; var prevC = c
        while (true) {
            if (!passable(r, c)) return null
            if (r != prevR || c != prevC) {
                val stepDist = if (r != prevR && c != prevC) sqrt(2.0) else 1.0
                total += stepDist * cost(r, c)
                prevR = r; prevC = c
            }
            if (r == r1 && c == c1) return total
            val e2 = 2 * err
            if (e2 > -dc) { err -= dc; r += sr }
            if (e2 < dr) { err += dr; c += sc }
        }
    }

    private fun simplify(path: List<Pair<Int, Int>>): List<Pair<Int, Int>> {
        if (path.size <= 2) return path
        val cumulative = DoubleArray(path.size)
        for (i in 1 until path.size) {
            val (r0, c0) = path[i - 1]; val (r1, c1) = path[i]
            val stepDist = if (r0 != r1 && c0 != c1) sqrt(2.0) else 1.0
            cumulative[i] = cumulative[i - 1] + stepDist * cost(r1, c1)
        }
        val simplified = mutableListOf(path[0])
        var anchorIdx = 0
        while (anchorIdx < path.size - 1) {
            var farthest = anchorIdx + 1
            for (j in anchorIdx + 1 until path.size) {
                val lc = lineCost(path[anchorIdx], path[j])
                val originalCost = cumulative[j] - cumulative[anchorIdx]
                if (lc != null && lc <= originalCost + 1e-6) {
                    farthest = j
                } else {
                    break
                }
            }
            simplified.add(path[farthest])
            anchorIdx = farthest
        }
        return simplified
    }
}
