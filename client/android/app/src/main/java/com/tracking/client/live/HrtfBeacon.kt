package com.tracking.client.live

import tracking.Tracking
import kotlin.math.atan2
import kotlin.math.sqrt

data class BeaconDirection(val azimuthDeg: Float, val elevationDeg: Float, val distanceM: Float)

/**
 * Egocentric direction (for HRTF audio-beacon spatialization) of a waypoint
 * relative to the current head pose. Runs entirely on-device — this is why
 * the camera is glasses-mounted (see the earlier design discussion in
 * CLAUDE.md's guiding-mode notes): camera pose IS head pose, so no separate
 * head tracker is needed, and computing this locally (rather than round-
 * tripping to the server per update) avoids latency/jitter in the beacon
 * direction while the user is actively walking toward it.
 */
object HrtfBeacon {
    /**
     * [targetX]/[targetZ]: world coordinates of the next waypoint, assumed
     * at floor/navigation height (this project's guiding/walking modes are
     * floor-constrained pedestrian navigation, not 3D flight — see
     * CLAUDE.md's "Continuous obstacle clearance" note on the same
     * assumption for path planning).
     *
     * Camera/world convention matches every pose this project produces
     * (scan_session.py / feature_tracker.py / RTAB-Map): OpenCV-style
     * camera-local axes (X-right, Y-down, Z-forward).
     *
     * azimuthDeg: 0 = straight ahead, positive = to the right.
     * elevationDeg: 0 = level with the camera, positive = above it.
     * Both are the raw geometric angle — a downstream HRTF engine may need
     * to negate/remap either axis to match its own sign convention; that
     * mapping isn't decided yet (see the beacon-placement design
     * discussion), so this deliberately returns engine-agnostic angles.
     */
    fun directionTo(pose: Tracking.Pose, targetX: Float, targetZ: Float): BeaconDirection {
        val wx = targetX - pose.x
        val wz = targetZ - pose.z
        // World-space displacement rotated into camera-local space via the
        // conjugate (inverse, for a unit quaternion) of the camera-to-world
        // rotation stored in Pose.
        val local = rotateByConjugate(pose.qx, pose.qy, pose.qz, pose.qw, wx, 0f, wz)
        val azimuthRad = atan2(local.first.toDouble(), local.third.toDouble())
        val horizDist = sqrt((local.first * local.first + local.third * local.third).toDouble())
        val elevationRad = atan2(-local.second.toDouble(), horizDist)
        return BeaconDirection(
            azimuthDeg = Math.toDegrees(azimuthRad).toFloat(),
            elevationDeg = Math.toDegrees(elevationRad).toFloat(),
            distanceM = sqrt(wx * wx + wz * wz),
        )
    }

    /**
     * Combines an authoritative server pose with the client's own local
     * motion estimate — [rotationDeltaQuat] (x, y, z, w), RotationTracker's
     * accumulatedRotation() — and [distanceM] (PdrStepEstimator's
     * distanceSinceReset()) — into a best-current-estimate Pose, bridging
     * the ~1Hz MappingService gap between server updates. See
     * ToolDispatcher's mapping-stream collector and CLAUDE.md's
     * "Server-planned walking path" note.
     *
     * Distance is walked along the NEW (rotation-updated) heading, not the
     * stale authoritative one — over the short (~1s) bridge window this
     * update spans, the difference is negligible, and using the current
     * heading is the more correct choice of the two in principle.
     */
    fun extrapolate(authoritative: Tracking.Pose, rotationDeltaQuat: FloatArray, distanceM: Float): Tracking.Pose {
        val newQuat = quatMultiply(
            floatArrayOf(authoritative.qx, authoritative.qy, authoritative.qz, authoritative.qw),
            rotationDeltaQuat,
        )
        val forward = rotate(newQuat[0], newQuat[1], newQuat[2], newQuat[3], 0f, 0f, 1f)
        return Tracking.Pose.newBuilder()
            .setX(authoritative.x + forward.first * distanceM)
            .setY(authoritative.y)
            .setZ(authoritative.z + forward.third * distanceM)
            .setQx(newQuat[0]).setQy(newQuat[1]).setQz(newQuat[2]).setQw(newQuat[3])
            .build()
    }

    /** Hamilton product a*b, both (x, y, z, w) unit quaternions — same
     * formula as RotationTracker.kt's own internal copy (kept separate
     * there since it's used for a different, self-contained composition;
     * this one is exposed for ToolDispatcher's latency-compensation math,
     * which needs the same op). */
    fun quatMultiply(a: FloatArray, b: FloatArray): FloatArray {
        val ax = a[0]; val ay = a[1]; val az = a[2]; val aw = a[3]
        val bx = b[0]; val by = b[1]; val bz = b[2]; val bw = b[3]
        return floatArrayOf(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    }

    /**
     * Egocentric direction of one screen-space point relative to ANOTHER
     * screen-space reference point (not necessarily frame center) — the
     * general form [directionFromBox] is a special case of. Real fix,
     * found from a direct user report: tracking mode's whole point is
     * guiding the user's HAND to the target object, but the original
     * [directionFromBox]-only design computed the object's position
     * relative to the FRAME CENTER — i.e. "which way to turn/look to
     * center the object," not "which way to move your hand toward it."
     * Those are only the same thing if the hand happens to already be at
     * the frame center, which isn't guaranteed at all. Callers guiding a
     * hand toward an object should pass the HAND's own screen position as
     * [refX]/[refY] instead of the frame center — see
     * ToolDispatcher.updateTrackingPixie().
     *
     * Approximates azimuth/elevation from the pixel offset via the same
     * pinhole FOV assumption server/tools/depth.py's `_estimate_K` uses
     * (fx=fy=0.8*max(w,h)) — not a real calibrated camera model, but
     * consistent with the only other place this project guesses
     * intrinsics.
     *
     * [distanceM] is a fixed nominal value, not a real measurement — 2D-only
     * tracking has no depth, so there's nothing to compute it from. Chosen
     * so HrtfBeaconPlayer's distance-based gain sits mid-range (audible, not
     * maxed) rather than implying a false precision.
     */
    fun directionBetweenPoints(targetX: Float, targetY: Float, refX: Float, refY: Float, frameWidth: Int, frameHeight: Int): BeaconDirection {
        val f = 0.8f * maxOf(frameWidth, frameHeight)
        val dx = targetX - refX
        val dy = targetY - refY
        return BeaconDirection(
            azimuthDeg = Math.toDegrees(atan2(dx.toDouble(), f.toDouble())).toFloat(),
            elevationDeg = Math.toDegrees(atan2(-dy.toDouble(), f.toDouble())).toFloat(),
            distanceM = 5f,
        )
    }

    /** Egocentric direction of a 2D-tracked object's box center relative to
     * the FRAME CENTER — a special case of [directionBetweenPoints] (ref =
     * frame center). NOT used by tracking mode's hand-guidance cue any
     * more (see that function's own doc comment for why) — kept for any
     * other caller that genuinely wants "where is this relative to where
     * the camera is pointed," e.g. a future look-at-target cue. */
    fun directionFromBox(centerX: Float, centerY: Float, frameWidth: Int, frameHeight: Int): BeaconDirection =
        directionBetweenPoints(centerX, centerY, frameWidth / 2f, frameHeight / 2f, frameWidth, frameHeight)

    /** World-space heading (degrees, 0 = camera-local +Z / no rotation,
     * positive = turned right) of [pose] — rotates the camera-local forward
     * vector (0, 0, 1) by its quaternion and reads atan2(x, z). Same
     * convention as AngleTracker.kt's own headingDegOf() and
     * mapping_servicer.py's _pose_heading_rad() (X-right/Y-down/Z-forward).
     * Used by ToolDispatcher's mapping-stream collector to feed each
     * accepted server fix into AngleTracker.setAuthoritativeHeadingDeg(). */
    fun worldHeadingDeg(pose: Tracking.Pose): Float {
        val forward = rotate(pose.qx, pose.qy, pose.qz, pose.qw, 0f, 0f, 1f)
        return Math.toDegrees(atan2(forward.first.toDouble(), forward.third.toDouble())).toFloat()
    }

    /** World (x, z) of a point [distanceM] ahead of [pose] at egocentric
     * [azimuthDeg] (0 = straight ahead, positive = right — same convention
     * as [directionTo]'s own return value) — the exact inverse of
     * [directionTo]: rotates a floor-plane direction vector by the pose's
     * quaternion DIRECTLY (not the conjugate [directionTo] uses) and offsets
     * by the pose's own position. Used by ToolDispatcher's sub-path dodge
     * computation to turn one egocentric "go this way for this far" reading
     * from the traversability fan into a real world waypoint. */
    fun worldPointFrom(pose: Tracking.Pose, azimuthDeg: Float, distanceM: Float): Pair<Float, Float> {
        val azRad = Math.toRadians(azimuthDeg.toDouble())
        val localX = (kotlin.math.sin(azRad) * distanceM).toFloat()
        val localZ = (kotlin.math.cos(azRad) * distanceM).toFloat()
        val world = rotate(pose.qx, pose.qy, pose.qz, pose.qw, localX, 0f, localZ)
        return (pose.x + world.first) to (pose.z + world.third)
    }

    private fun rotateByConjugate(
        qx: Float, qy: Float, qz: Float, qw: Float, vx: Float, vy: Float, vz: Float,
    ): Triple<Float, Float, Float> = rotate(-qx, -qy, -qz, qw, vx, vy, vz)

    /** v' = v + 2*qw*(q_xyz × v) + 2*(q_xyz × (q_xyz × v)) — standard quaternion-vector rotation. */
    private fun rotate(
        qx: Float, qy: Float, qz: Float, qw: Float, vx: Float, vy: Float, vz: Float,
    ): Triple<Float, Float, Float> {
        val uvx = qy * vz - qz * vy
        val uvy = qz * vx - qx * vz
        val uvz = qx * vy - qy * vx
        val uuvx = qy * uvz - qz * uvy
        val uuvy = qz * uvx - qx * uvz
        val uuvz = qx * uvy - qy * uvx
        return Triple(
            vx + 2 * (qw * uvx + uuvx),
            vy + 2 * (qw * uvy + uuvy),
            vz + 2 * (qw * uvz + uuvz),
        )
    }
}
