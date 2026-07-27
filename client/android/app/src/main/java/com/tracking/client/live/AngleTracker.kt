package com.tracking.client.live

import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Frame-to-frame ROTATION-ONLY visual tracking — a drop-in replacement for
 * the old `RotationTracker` (same `accumulatedRotation()`/`resetAccumulator()`/
 * `reset()` shapes, so the surrounding pose-bridging code in
 * ToolDispatcher.kt — the mapping-stream collector's latency compensation,
 * `buildMappingChunk()`, `stopActiveModes()` — barely changed), validated
 * first in test_module/pixie_hrtf_app/ before being ported here. See
 * CLAUDE.md's "Pixie + Angle modules" note.
 *
 * Two changes from the old `RotationTracker`, both ported from that test
 * harness after a real ~0.5s per-update lag investigation there:
 *
 * 1. [processLumaFrame] takes the camera's raw YUV Y-plane directly instead
 *    of round-tripping through a decoded Bitmap + JPEG encode + JPEG
 *    decode (the old `processFrame(frameJpeg: ByteArray)` did a full
 *    `BitmapFactory.decodeByteArray` → `ARGB_8888` copy → `cvtColor` every
 *    call) — the Y-plane already IS luma, so this skips a whole encode/
 *    decode/color-convert cycle. See CameraManager.kt's new `lumaFlow`.
 * 2. Configurable [maxFeatures]/[processResolution], defaulting to
 *    **1000 features / 480px** for this app — heavier than the test
 *    harness's own 320px/300-feature defaults, a deliberate separate
 *    choice for production. This tracker's own working image is
 *    intentionally independent of (and much smaller than) whatever
 *    CameraManager sends to MappingService/RTAB-Map over the network —
 *    that JPEG pipeline (frameIntervalMs/walkingIntervalMs, 640px cap) is
 *    completely unaffected by this file.
 *
 * On top of the unchanged rotation-only role, two new accessors —
 * [currentHeadingDeg]/[driftedAngleDeg] — expose what used to only be
 * computed ad hoc inline wherever a caller needed it. [setAuthoritativeHeadingDeg]
 * is the "update direction when direction info sent from RTAB-Map through
 * the server" half: called once per accepted MappingService pose (see
 * ToolDispatcher's mapping-stream collector), it both remembers the new
 * baseline heading AND resets the accumulator in one call — replacing the
 * bare `resetAccumulator()` call that used to sit at that point.
 *
 * Deliberately rotation-only: monocular vision has no scale for
 * translation (recoverPose()'s `t` output is a unit direction, not a real
 * distance — meaningless here), but rotation decomposed from the Essential
 * matrix has NO such ambiguity. Translation instead comes from
 * PdrStepEstimator (phone IMU step detection + a fixed stride length),
 * completely unaffected by this file — full-pose extrapolation
 * (HrtfBeacon.extrapolate(), combining THIS class's accumulatedRotation()
 * with PdrStepEstimator's distanceSinceReset()) is unchanged.
 */
class AngleTracker(private val maxFeatures: Int = 1000, private var processResolution: Int = 480) {

    private val openCvReady: Boolean by lazy {
        if (!OpenCVLoader.initDebug()) throw IllegalStateException("OpenCV failed to initialize")
        true
    }
    private var cachedOrb: ORB? = null
    private val orb: ORB
        get() = cachedOrb ?: ORB.create(maxFeatures).also { cachedOrb = it }
    private val matcher: BFMatcher by lazy { BFMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING, true) }

    private var prevKp: MatOfKeyPoint? = null
    private var prevDesc: Mat? = null

    // ── Vision-check novelty gate (WALKING/GUIDING's Gemini alert trigger)
    // Requested directly by the user: instead of asking Gemini's vision on
    // a flat timer regardless of whether anything actually changed, only
    // consider it once the scene has genuinely moved on. Reuses the SAME
    // ORB keypoints/descriptors this class already computes every luma
    // frame for rotation tracking — no second detection pass — compared
    // against the last "clear" (sharpness >= NOVELTY_BLUR_THRESHOLD, same
    // 40.0 value orb_novelty_gate.py's server-side blur gate historically
    // used) reference frame. A simplified, SINGLE-reference version of
    // orb_novelty_gate.py's server-side design (which matches against
    // every prior accepted frame, not just the latest) — good enough for
    // "did the view meaningfully change since we last looked," not a
    // scan-quality gate.
    private var noveltyRefDesc: Mat? = null
    @Volatile private var noveltyTriggered = false

    // Accumulated rotation since the last resetAccumulator() call — a unit
    // quaternion (x, y, z, w), composed by right-multiplying each new
    // per-frame delta on top of it (see processGray's own comment for the
    // composition convention). Identity until the first successful frame
    // pair.
    @Volatile private var accumQx = 0f
    @Volatile private var accumQy = 0f
    @Volatile private var accumQz = 0f
    @Volatile private var accumQw = 1f

    // The heading (degrees) as of the last setAuthoritativeHeadingDeg()
    // call — currentHeadingDeg() extrapolates forward from this using
    // whatever local rotation has accumulated since. Meaningless (stays 0)
    // until the first authoritative fix arrives, same as any other
    // just-connected state in this codebase.
    @Volatile private var baselineHeadingDeg = 0f

    /**
     * Feed the camera's raw Y-plane (luma) directly — see
     * CameraManager.kt's `lumaFlow`. [rowStride] may exceed [width]
     * (sensor row padding); [rotationDegrees] must be a multiple of 90 (as
     * CameraX always reports) and is applied as a cheap Mat rotation
     * instead of a Bitmap-matrix rotation.
     *
     * Matches against whatever frame was fed last, and — if there are
     * enough correspondences and the resulting Essential matrix
     * decomposition succeeds — folds the newly-estimated rotation into the
     * running accumulator. Returns true if the accumulator was actually
     * updated this call.
     */
    fun processLumaFrame(luma: ByteArray, width: Int, height: Int, rowStride: Int, rotationDegrees: Int): Boolean {
        return try {
            check(openCvReady)
            processGray(lumaToGray(luma, width, height, rowStride, rotationDegrees))
        } catch (e: Exception) {
            Log.w(TAG, "processLumaFrame failed: ${e.message}")
            false
        }
    }

    private fun lumaToGray(luma: ByteArray, width: Int, height: Int, rowStride: Int, rotationDegrees: Int): Mat {
        val padded = Mat(height, rowStride, CvType.CV_8UC1)
        padded.put(0, 0, luma)
        val gray = if (rowStride == width) padded else padded.submat(0, height, 0, width).clone().also { padded.release() }

        val rotated = when (rotationDegrees) {
            90 -> Mat().also { Core.rotate(gray, it, Core.ROTATE_90_CLOCKWISE); gray.release() }
            180 -> Mat().also { Core.rotate(gray, it, Core.ROTATE_180); gray.release() }
            270 -> Mat().also { Core.rotate(gray, it, Core.ROTATE_90_COUNTERCLOCKWISE); gray.release() }
            else -> gray
        }

        val target = processResolution
        val longEdge = max(rotated.width(), rotated.height())
        if (longEdge <= target) return rotated
        val scale = target.toDouble() / longEdge
        val resized = Mat()
        Imgproc.resize(rotated, resized, org.opencv.core.Size(rotated.width() * scale, rotated.height() * scale))
        rotated.release()
        return resized
    }

    private fun processGray(gray: Mat): Boolean {
        val kp = MatOfKeyPoint()
        val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kp, desc)
        if (desc.empty() || kp.empty()) {
            prevKp = null; prevDesc = null
            return false
        }

        // Reuses these SAME just-computed ORB features for the vision-check
        // novelty gate below — see that section's own doc comment for why
        // no second detection pass is needed.
        evaluateNovelty(gray, desc)

        val localKp = prevKp
        val localDesc = prevDesc
        prevKp = kp
        prevDesc = desc
        if (localKp == null || localDesc == null) return false  // first frame — nothing to compare yet

        val matches = MatOfDMatch()
        matcher.match(localDesc, desc, matches)
        val matchesArray = matches.toArray()
        if (matchesArray.size < MIN_MATCHES) return false

        val srcPoints = mutableListOf<Point>()
        val dstPoints = mutableListOf<Point>()
        val localKpArr = localKp.toArray()
        val kpArr = kp.toArray()
        for (m in matchesArray) {
            val s = localKpArr.getOrNull(m.queryIdx)?.pt ?: continue
            val d = kpArr.getOrNull(m.trainIdx)?.pt ?: continue
            srcPoints.add(s); dstPoints.add(d)
        }
        if (srcPoints.size < MIN_MATCHES) return false

        val srcPts = MatOfPoint2f().apply { fromList(srcPoints) }
        val dstPts = MatOfPoint2f().apply { fromList(dstPoints) }
        val cameraMatrix = assumedCameraMatrix(gray.width(), gray.height())

        val essential = try {
            Calib3d.findEssentialMat(srcPts, dstPts, cameraMatrix, Calib3d.RANSAC, 0.999, 1.5)
        } catch (e: Exception) {
            Log.w(TAG, "findEssentialMat failed: ${e.message}")
            return false
        }
        if (essential == null || essential.empty() || essential.rows() != 3 || essential.cols() != 3) return false

        val r = Mat()
        val t = Mat()  // discarded — see class docstring
        val poseMask = Mat()
        val inliers = try {
            Calib3d.recoverPose(essential, srcPts, dstPts, cameraMatrix, r, t, poseMask)
        } catch (e: Exception) {
            Log.w(TAG, "recoverPose failed: ${e.message}")
            return false
        }
        if (inliers < MIN_MATCHES || r.rows() != 3 || r.cols() != 3) return false

        // recoverPose's R maps a point expressed in the PREVIOUS frame's
        // camera coordinates into the CURRENT frame's camera coordinates
        // (v_curr = R * v_prev) — the opposite direction from what we want
        // to compose onto a running WORLD-frame orientation. See the
        // module docstring; NOT verified against a live device — if a
        // head turn ever sounds mirrored, this conjugate is the first
        // thing to check.
        val q = matToQuaternion(r)
        val qInv = floatArrayOf(-q[0], -q[1], -q[2], q[3])
        val composed = quatMultiply(floatArrayOf(accumQx, accumQy, accumQz, accumQw), qInv)
        accumQx = composed[0]; accumQy = composed[1]; accumQz = composed[2]; accumQw = composed[3]
        return true
    }

    /** Variance of the Laplacian — same formula CameraManager.kt's
     * computeSharpness() uses for the JPEG path, computed here directly on
     * the gray Mat this class already has (no extra decode). */
    private fun sharpnessScore(gray: Mat): Double {
        val lap = Mat()
        return try {
            Imgproc.Laplacian(gray, lap, CvType.CV_64F)
            val mean = MatOfDouble(); val stddev = MatOfDouble()
            Core.meanStdDev(lap, mean, stddev)
            val sigma = stddev.toArray().getOrElse(0) { 0.0 }
            sigma * sigma
        } finally {
            lap.release()
        }
    }

    /** Updates [noveltyTriggered] — see the class-level doc comment above
     * [noveltyRefDesc] for the full design. Only ever compares/updates the
     * reference on a "clear" (not blurry) frame; a blurry frame is simply
     * skipped for this purpose (its ORB features still feed the rotation
     * tracker above, unaffected). */
    private fun evaluateNovelty(gray: Mat, desc: Mat) {
        if (sharpnessScore(gray) < NOVELTY_BLUR_THRESHOLD) return

        val ref = noveltyRefDesc
        if (ref == null || ref.empty()) {
            noveltyRefDesc = desc
            return
        }

        val matches = MatOfDMatch()
        try {
            matcher.match(desc, ref, matches)
        } catch (e: Exception) {
            return
        }
        val matchArr = matches.toArray()
        val totalCount = desc.rows().coerceAtLeast(1)
        val newCount = matchArr.count { it.distance > NOVELTY_MATCH_DISTANCE_MAX }
        val newFraction = newCount.toDouble() / totalCount
        if (newFraction >= NOVELTY_NEW_FRACTION) {
            noveltyTriggered = true
            noveltyRefDesc = desc  // this frame becomes the new reference
        }
    }

    /** Pull-and-clear, called from ToolDispatcher's 2Hz avoidance tick —
     * true if the scene has changed enough (per [evaluateNovelty]) since
     * the reference frame to be worth considering a Gemini vision-check
     * call for (still subject to that call's own separate cooldown). */
    fun consumeNoveltyTrigger(): Boolean {
        val t = noveltyTriggered
        noveltyTriggered = false
        return t
    }

    /** Current accumulated rotation since the last [resetAccumulator] call, as (x, y, z, w). */
    fun accumulatedRotation(): FloatArray = floatArrayOf(accumQx, accumQy, accumQz, accumQw)

    /** Local-only drift (degrees) since the last [resetAccumulator]/
     * [setAuthoritativeHeadingDeg] call — a pure function of the rotation
     * accumulator, useful as a confidence/staleness signal independent of
     * whether an authoritative baseline is even being tracked. */
    fun driftedAngleDeg(): Float = headingDegOf(accumulatedRotation())

    /** Best current heading estimate (degrees), extrapolating the last
     * authoritative fix by whatever local rotation has accumulated since.
     * Meaningless (reads as just the raw drift) until the first
     * [setAuthoritativeHeadingDeg] call. */
    fun currentHeadingDeg(): Float = wrapDeg(baselineHeadingDeg + driftedAngleDeg())

    /** Called once per accepted authoritative fix (RTAB-Map heading, via
     * MappingService — see ToolDispatcher's mapping-stream collector).
     * Remembers the new baseline AND resets the accumulator in one call —
     * replaces a bare resetAccumulator() call at that same point. */
    fun setAuthoritativeHeadingDeg(headingDeg: Float) {
        baselineHeadingDeg = wrapDeg(headingDeg)
        resetAccumulator()
    }

    /** Starts a new rotation-since-last-fix window — called whenever a
     * fresh server heading fix is folded in. Does NOT clear the ORB
     * reference frame — frame-to-frame matching continuity is independent
     * of when the server last responded. */
    fun resetAccumulator() {
        accumQx = 0f; accumQy = 0f; accumQz = 0f; accumQw = 1f
    }

    /** Full reset (e.g. mode change) — also drops the ORB reference frame
     * and the authoritative baseline. */
    fun reset() {
        resetAccumulator()
        prevKp = null
        prevDesc = null
        baselineHeadingDeg = 0f
        noveltyRefDesc = null
        noveltyTriggered = false
    }

    private fun assumedCameraMatrix(width: Int, height: Int): Mat {
        val f = 0.8 * max(width, height)
        val cx = width / 2.0
        val cy = height / 2.0
        return Mat(3, 3, CvType.CV_64F).apply {
            put(0, 0, f); put(0, 1, 0.0); put(0, 2, cx)
            put(1, 0, 0.0); put(1, 1, f); put(1, 2, cy)
            put(2, 0, 0.0); put(2, 1, 0.0); put(2, 2, 1.0)
        }
    }

    companion object {
        private const val TAG = "AngleTracker"
        private const val MIN_MATCHES = 15  // well above the 5-point algorithm's bare minimum

        // Vision-check novelty gate (see evaluateNovelty()) — requested
        // directly by the user: "if we have a 70% new orb features compare
        // to the previous clear frame (40 blur threshold), then consider
        // calling the gemini live api for alert". NOVELTY_MATCH_DISTANCE_MAX
        // is a Hamming-distance cutoff on ORB's binary descriptors (32
        // bytes/256 bits per descriptor) — a match strictly worse than this
        // doesn't count as "the same feature," same order of magnitude as
        // typical ORB re-ID cutoffs elsewhere in this codebase's matching
        // (BFMatcher crossCheck already discards worse ties for the
        // rotation-tracking match above; this is a plain, non-crosschecked
        // one-way match, so an explicit distance cutoff is needed here).
        const val NOVELTY_BLUR_THRESHOLD = 40.0
        private const val NOVELTY_NEW_FRACTION = 0.70
        private const val NOVELTY_MATCH_DISTANCE_MAX = 64.0

        private fun wrapDeg(deg: Float): Float {
            var d = deg % 360f
            if (d > 180f) d -= 360f
            if (d < -180f) d += 360f
            return d
        }

        /** Standard rotation-matrix -> quaternion conversion (Shepperd's
         * method, numerically stable branch selection). Returns (x, y, z, w). */
        private fun matToQuaternion(r: Mat): FloatArray {
            val m00 = r.get(0, 0)[0]; val m01 = r.get(0, 1)[0]; val m02 = r.get(0, 2)[0]
            val m10 = r.get(1, 0)[0]; val m11 = r.get(1, 1)[0]; val m12 = r.get(1, 2)[0]
            val m20 = r.get(2, 0)[0]; val m21 = r.get(2, 1)[0]; val m22 = r.get(2, 2)[0]
            val trace = m00 + m11 + m22
            return when {
                trace > 0 -> {
                    val s = sqrt(trace + 1.0) * 2.0  // s = 4*qw
                    floatArrayOf(((m21 - m12) / s).toFloat(), ((m02 - m20) / s).toFloat(),
                        ((m10 - m01) / s).toFloat(), (0.25 * s).toFloat())
                }
                m00 > m11 && m00 > m22 -> {
                    val s = sqrt(1.0 + m00 - m11 - m22) * 2.0  // s = 4*qx
                    floatArrayOf((0.25 * s).toFloat(), ((m01 + m10) / s).toFloat(),
                        ((m02 + m20) / s).toFloat(), ((m21 - m12) / s).toFloat())
                }
                m11 > m22 -> {
                    val s = sqrt(1.0 + m11 - m00 - m22) * 2.0  // s = 4*qy
                    floatArrayOf(((m01 + m10) / s).toFloat(), (0.25 * s).toFloat(),
                        ((m12 + m21) / s).toFloat(), ((m02 - m20) / s).toFloat())
                }
                else -> {
                    val s = sqrt(1.0 + m22 - m00 - m11) * 2.0  // s = 4*qz
                    floatArrayOf(((m02 + m20) / s).toFloat(), ((m12 + m21) / s).toFloat(),
                        (0.25 * s).toFloat(), ((m21 - m12) / s).toFloat())
                }
            }
        }

        /** Hamilton product a*b, both (x, y, z, w) unit quaternions. */
        private fun quatMultiply(a: FloatArray, b: FloatArray): FloatArray {
            val (ax, ay, az, aw) = a
            val (bx, by, bz, bw) = b
            return floatArrayOf(
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            )
        }

        /** Rotates the camera-local forward vector (0, 0, 1) by quaternion
         * [q] and returns its floor-plane heading via atan2(x, z) — 0 = no
         * rotation, positive = turned right. Same convention this whole
         * project's HrtfBeacon.kt/mapping_servicer.py's _pose_heading_rad
         * already use (X-right/Y-down/Z-forward). */
        private fun headingDegOf(q: FloatArray): Float {
            val (qx, qy, qz, qw) = q
            val x = 2f * (qx * qz + qy * qw)
            val z = 1f - 2f * (qx * qx + qy * qy)
            return Math.toDegrees(Math.atan2(x.toDouble(), z.toDouble())).toFloat()
        }

        private operator fun FloatArray.component1() = this[0]
        private operator fun FloatArray.component2() = this[1]
        private operator fun FloatArray.component3() = this[2]
        private operator fun FloatArray.component4() = this[3]
    }
}
