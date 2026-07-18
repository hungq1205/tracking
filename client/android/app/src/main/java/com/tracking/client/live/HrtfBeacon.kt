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
     * Egocentric direction of a 2D-tracked object's box center, for tracking
     * mode — where [HrtfBeacon.directionTo] needs a 3D pose + world target
     * (guiding/walking, backed by MappingService's RTAB-Map pose), local ORB
     * object tracking (TrackingBackend.kt) has no pose or depth at all, only
     * a pixel box in the current frame. Approximates azimuth/elevation from
     * the box center's pixel offset from frame center via the same pinhole
     * FOV assumption server/tools/depth.py's `_estimate_K` uses
     * (fx=fy=0.8*max(w,h)) — not a real calibrated camera model, but
     * consistent with the only other place this project guesses intrinsics.
     *
     * [distanceM] is a fixed nominal value, not a real measurement — 2D-only
     * tracking has no depth, so there's nothing to compute it from. Chosen
     * so HrtfBeaconPlayer's distance-based gain sits mid-range (audible, not
     * maxed) rather than implying a false precision.
     */
    fun directionFromBox(centerX: Float, centerY: Float, frameWidth: Int, frameHeight: Int): BeaconDirection {
        val f = 0.8f * maxOf(frameWidth, frameHeight)
        val dx = centerX - frameWidth / 2f
        val dy = centerY - frameHeight / 2f
        return BeaconDirection(
            azimuthDeg = Math.toDegrees(atan2(dx.toDouble(), f.toDouble())).toFloat(),
            elevationDeg = Math.toDegrees(atan2(-dy.toDouble(), f.toDouble())).toFloat(),
            distanceM = 5f,
        )
    }

    /**
     * World-space heading (yaw only, radians) of the camera's forward axis
     * — the world-space X-Z angle you'd add an egocentric azimuth to in
     * order to get an absolute world-space bearing. Used by
     * LocalPathPlanner.findMostOpenDirection() for walking mode's ambient
     * beacon, which has no waypoint to route toward and instead needs to
     * test candidate directions relative to current heading against the
     * occupancy grid.
     *
     * Pitch/roll are deliberately ignored — this project's navigation is
     * floor-constrained pedestrian movement (same assumption CLAUDE.md's
     * "Continuous obstacle clearance" note makes for path planning), so
     * only the yaw component of the pose's orientation is meaningful for
     * "which way is the user walking."
     */
    fun worldYawRad(pose: Tracking.Pose): Double {
        val forward = rotate(pose.qx, pose.qy, pose.qz, pose.qw, 0f, 0f, 1f)
        return atan2(forward.first.toDouble(), forward.third.toDouble())
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
