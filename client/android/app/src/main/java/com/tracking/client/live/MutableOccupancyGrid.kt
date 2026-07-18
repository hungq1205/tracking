package com.tracking.client.live

import tracking.Tracking
import kotlin.math.roundToInt

/**
 * Client-side persistent occupancy grid, patched in place by
 * MappingUpdate.grid_delta instead of being replaced wholesale on every
 * update — the server now ships a full OccupancyGrid only on `full_resync`
 * (first update for a location, or whenever the map's explored bounding
 * box grew) and otherwise ships only the cells that actually changed since
 * the last update (see mapping_servicer.py's UpdateMapping / occupancy_map.
 * py's extract_dirty_delta() for the server-side half of this).
 *
 * Cell indices in a delta (ix, iz) are GLOBAL grid coordinates, stable
 * across deltas as long as the bounding box hasn't changed — exactly the
 * invariant a `full_resync` flag exists to protect (see ToolDispatcher's
 * collect loop: a delta is only ever applied on top of a grid that was
 * itself last fully (re)built from a `full_resync` message with the same
 * origin/dimensions).
 */
class MutableOccupancyGrid private constructor(
    private var width: Int,
    private var height: Int,
    private var originX: Float,
    private var originZ: Float,
    private var cellSize: Float,
    private var cls: IntArray,
    private var heightNorm: FloatArray,
    private var clearanceM: FloatArray,
) {
    companion object {
        fun fromFull(grid: Tracking.OccupancyGrid): MutableOccupancyGrid = MutableOccupancyGrid(
            width = grid.width, height = grid.height,
            originX = grid.originX, originZ = grid.originZ, cellSize = grid.cellSize,
            cls = grid.clsList.toIntArray(),
            heightNorm = grid.heightNormList.toFloatArray(),
            clearanceM = grid.clearanceMList.toFloatArray(),
        )
    }

    /** Applies a sparse cell patch in place. Cells outside the current
     * bounds are dropped (shouldn't happen — the server only sends deltas
     * when bounds haven't changed since the last full resync — but a
     * dropped out-of-bounds cell is a safe degrade, not a crash). */
    fun applyDelta(delta: Tracking.OccupancyGridDelta) {
        val originCol = (originX / cellSize).roundToInt()
        val originRow = (originZ / cellSize).roundToInt()
        for (cell in delta.cellsList) {
            val col = cell.ix - originCol
            val row = cell.iz - originRow
            if (row !in 0 until height || col !in 0 until width) continue
            val idx = row * width + col
            cls[idx] = cell.cls
            heightNorm[idx] = cell.heightNorm
            clearanceM[idx] = cell.clearanceM
        }
    }

    /** Rebuilds an immutable Tracking.OccupancyGrid snapshot for consumers
     * that expect the proto type directly (LocalPathPlanner, HRTF beacon
     * ray-casting) — cheap repackaging of already-known arrays, no
     * classification/distance-transform recompute (that stays server-side). */
    fun toProto(): Tracking.OccupancyGrid = Tracking.OccupancyGrid.newBuilder()
        .setWidth(width).setHeight(height)
        .setOriginX(originX).setOriginZ(originZ).setCellSize(cellSize)
        .addAllCls(cls.toList())
        .addAllHeightNorm(heightNorm.toList())
        .addAllClearanceM(clearanceM.toList())
        .build()
}
