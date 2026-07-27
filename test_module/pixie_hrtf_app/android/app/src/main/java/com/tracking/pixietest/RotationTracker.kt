package com.tracking.pixietest

import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
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
 * Frame-to-frame ROTATION-ONLY visual tracking — ORB detect + BFMatcher
 * match + RANSAC essential-matrix + recoverPose, every call. (An earlier
 * pass of this file also had a Lucas-Kanade optical-flow path; dropped
 * outright per direct request — ORB only now, no runtime method switch.)
 * Adapted from client/android/app/src/main/java/com/tracking/client/live/
 * RotationTracker.kt (same "separately deployed processes" duplication
 * precedent this project already uses for orb_novelty_gate.py/
 * live_path_planner.py — see this app's README). Bridges the gap between
 * pixie_hrtf_server.py's RTAB-Map-based heading fixes with a fast, local
 * head-orientation estimate.
 *
 * [processFrameLuma] takes the camera's raw YUV Y-plane directly (see
 * SimpleCameraSource) instead of round-tripping through a decoded/rotated
 * Bitmap + JPEG encode + JPEG decode — the Y-plane already IS luma, so this
 * skips a whole encode/decode/color-convert cycle. [processResolution] and
 * [maxFeatures] are both live-mutable (see MainActivity's dropdowns) —
 * changing either takes effect on the very next call, no reconnect needed.
 * See HeadingEstimator for how calls are decoupled from the camera thread
 * (drop-to-latest, never queues a backlog) — that fix and this file's ORB
 * work are complementary, not redundant.
 */
class RotationTracker {

    /** Long-edge pixel target the working grayscale image is downscaled to
     * before ORB ever sees it — three presets exposed via a dropdown (see
     * MainActivity). Smaller = faster but fewer usable features; larger =
     * more detail but slower. Mutable, read fresh every call. */
    @Volatile var processResolution: Int = 320

    /** ORB feature count — a dropdown in steps of 100 (see MainActivity).
     * Changing this invalidates the cached ORB detector so the new count
     * takes effect on the very next call. */
    var maxFeatures: Int = 300
        set(value) {
            if (field != value) {
                field = value
                cachedOrb = null
            }
        }

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

    // Accumulated rotation since the last resetAccumulator() call — a unit
    // quaternion (x, y, z, w), composed by right-multiplying each new
    // per-frame delta on top of it (see the composition-convention comment
    // below). Identity until the first successful frame pair.
    @Volatile private var accumQx = 0f
    @Volatile private var accumQy = 0f
    @Volatile private var accumQz = 0f
    @Volatile private var accumQw = 1f

    /** Last [processFrameLuma] call's total processing time, milliseconds
     * — ORB detect+match+RANSAC essential-matrix+recoverPose, i.e. the
     * exact cost the reported on-device lag traces back to. Logged every
     * call (see TAG) and also exposed here for on-screen display. */
    @Volatile var lastProcessingMs: Long = 0
        private set

    /** Achieved calls-per-second — 1000 / (time between the START of this
     * call and the START of the previous one), NOT 1000/lastProcessingMs.
     * With HeadingEstimator's drop-to-latest consumer this is the real
     * end-to-end throughput. */
    @Volatile var lastFps: Double = 0.0
        private set
    private var lastCallStartNs: Long = 0

    // Debug-only: why the last call did/didn't update the accumulator.
    @Volatile private var lastReason: String = "n/a"

    /**
     * Feed the camera's raw Y-plane (luma) directly — see
     * SimpleCameraSource.onLumaFrame. [rowStride] may exceed [width]
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
    fun processFrameLuma(luma: ByteArray, width: Int, height: Int, rowStride: Int, rotationDegrees: Int): Boolean {
        val startNs = System.nanoTime()
        if (lastCallStartNs != 0L) {
            val intervalMs = (startNs - lastCallStartNs) / 1_000_000.0
            if (intervalMs > 0) lastFps = 1000.0 / intervalMs
        }
        lastCallStartNs = startNs

        val ok = try {
            check(openCvReady)
            processGray(lumaToGray(luma, width, height, rowStride, rotationDegrees))
        } catch (e: Exception) {
            lastReason = "exception: ${e.message}"
            Log.w(TAG, "processFrameLuma failed: ${e.message}")
            false
        }
        lastProcessingMs = (System.nanoTime() - startNs) / 1_000_000
        Log.d(TAG, "processFrameLuma: ${lastProcessingMs}ms in=${width}x${height} res=$processResolution " +
            "features=$maxFeatures (~${"%.1f".format(lastFps)} fps) updated=$ok reason=$lastReason " +
            "headingDeg=${"%.1f".format(headingDegOf(accumulatedRotation()))}")
        return ok
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
            lastReason = "no features detected"
            return false
        }

        val localKp = prevKp
        val localDesc = prevDesc
        prevKp = kp
        prevDesc = desc
        if (localKp == null || localDesc == null) {
            lastReason = "first frame"
            return false  // first frame — nothing to compare yet
        }

        val matches = MatOfDMatch()
        matcher.match(localDesc, desc, matches)
        val matchesArray = matches.toArray()
        Log.d(TAG, "ORB: ${matchesArray.size} raw matches")
        if (matchesArray.size < MIN_MATCHES) {
            lastReason = "too few matches (${matchesArray.size} < $MIN_MATCHES)"
            return false
        }

        val srcPoints = mutableListOf<Point>()
        val dstPoints = mutableListOf<Point>()
        val localKpArr = localKp.toArray()
        val kpArr = kp.toArray()
        for (m in matchesArray) {
            val s = localKpArr.getOrNull(m.queryIdx)?.pt ?: continue
            val d = kpArr.getOrNull(m.trainIdx)?.pt ?: continue
            srcPoints.add(s); dstPoints.add(d)
        }
        if (srcPoints.size < MIN_MATCHES) {
            lastReason = "too few valid correspondences (${srcPoints.size} < $MIN_MATCHES)"
            return false
        }

        val srcPts = MatOfPoint2f().apply { fromList(srcPoints) }
        val dstPts = MatOfPoint2f().apply { fromList(dstPoints) }
        val cameraMatrix = assumedCameraMatrix(gray.width(), gray.height())

        val essential = try {
            Calib3d.findEssentialMat(srcPts, dstPts, cameraMatrix, Calib3d.RANSAC, 0.999, 1.5)
        } catch (e: Exception) {
            lastReason = "findEssentialMat threw: ${e.message}"
            Log.w(TAG, "findEssentialMat failed: ${e.message}")
            return false
        }
        if (essential == null || essential.empty() || essential.rows() != 3 || essential.cols() != 3) {
            lastReason = "findEssentialMat returned invalid matrix"
            return false
        }

        val r = Mat()
        val t = Mat()  // discarded — see class docstring
        val poseMask = Mat()
        val inliers = try {
            Calib3d.recoverPose(essential, srcPts, dstPts, cameraMatrix, r, t, poseMask)
        } catch (e: Exception) {
            lastReason = "recoverPose threw: ${e.message}"
            Log.w(TAG, "recoverPose failed: ${e.message}")
            return false
        }
        if (inliers < MIN_MATCHES || r.rows() != 3 || r.cols() != 3) {
            lastReason = "recoverPose too few inliers ($inliers < $MIN_MATCHES)"
            return false
        }
        lastReason = "ok"

        // recoverPose's R maps a point expressed in the PREVIOUS frame's
        // camera coordinates into the CURRENT frame's camera coordinates
        // (v_curr = R * v_prev) — the opposite direction from what we want
        // to compose onto a running WORLD-frame orientation. See the main
        // client app's RotationTracker.kt for the full derivation. NOT
        // verified against a live device — if a head turn ever sounds
        // mirrored, this conjugate is the first thing to check.
        val q = matToQuaternion(r)
        val qInv = floatArrayOf(-q[0], -q[1], -q[2], q[3])
        val composed = quatMultiply(floatArrayOf(accumQx, accumQy, accumQz, accumQw), qInv)
        accumQx = composed[0]; accumQy = composed[1]; accumQz = composed[2]; accumQw = composed[3]
        return true
    }

    /** Current accumulated rotation since the last [resetAccumulator] call, as (x, y, z, w). */
    fun accumulatedRotation(): FloatArray = floatArrayOf(accumQx, accumQy, accumQz, accumQw)

    /** Starts a new rotation-since-last-fix window — called whenever a
     * fresh server heading fix is folded in (see HeadingEstimator). Does
     * NOT clear the ORB reference frame — frame-to-frame matching
     * continuity is independent of when the server last responded. */
    fun resetAccumulator() {
        accumQx = 0f; accumQy = 0f; accumQz = 0f; accumQw = 1f
    }

    /** Full reset (e.g. "Reset anchor") — also drops the ORB reference frame. */
    fun reset() {
        resetAccumulator()
        prevKp = null
        prevDesc = null
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
        private const val TAG = "RotationTracker"
        private const val MIN_MATCHES = 15  // well above the 5-point algorithm's bare minimum

        // Dropdown presets for processResolution (long edge, px) — see
        // MainActivity's resolution Spinner.
        val RESOLUTION_PRESETS = intArrayOf(240, 320, 480)

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
        fun quatMultiply(a: FloatArray, b: FloatArray): FloatArray {
            val (ax, ay, az, aw) = a
            val (bx, by, bz, bw) = b
            return floatArrayOf(
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            )
        }

        /** Conjugate (== inverse for a unit quaternion) of (x, y, z, w). */
        fun quatConjugate(q: FloatArray): FloatArray = floatArrayOf(-q[0], -q[1], -q[2], q[3])

        /**
         * Rotates the camera-local forward vector (0, 0, 1) by quaternion
         * [q] and returns its floor-plane heading via atan2(x, z) — 0 = no
         * rotation, positive = turned right. Same convention this whole
         * project's HrtfBeacon.kt/mapping_servicer.py's _pose_heading_rad
         * already use (X-right/Y-down/Z-forward, "up" is -Y so heading is
         * measured purely in the X-Z floor plane).
         */
        fun headingDegOf(q: FloatArray): Float {
            val (qx, qy, qz, qw) = q
            // v' = q * (0,0,1) * q^-1, expanded directly (only the x/z
            // components are needed for a floor-plane heading).
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
