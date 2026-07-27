package com.tracking.client.live

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
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
 * Frame-to-frame ROTATION-ONLY visual tracking — bridges the gap between
 * MappingService's ~1Hz server updates with a fast, local head-orientation
 * estimate, so the HRTF beacon can keep swinging correctly when the user
 * turns their head even between server round trips. See CLAUDE.md's
 * "Server-planned walking path" note.
 *
 * Deliberately rotation-only: monocular vision has no scale for
 * translation (recoverPose()'s `t` output is a unit direction, not a real
 * distance — meaningless here), but rotation decomposed from the Essential
 * matrix has NO such ambiguity — it's exactly as metrically correct as
 * RTAB-Map's own rotation, just computed from two frames instead of a full
 * local map. Translation instead comes from PdrStepEstimator (phone IMU
 * step detection + a fixed stride length) — a deliberate split, not an
 * oversight; see that class's own docstring.
 *
 * Mirrors TrackingBackend.kt's existing on-device ORB detect/match
 * machinery (same ORB.create/BFMatcher calls, same JPEG->gray Mat decode)
 * but computes Calib3d.findEssentialMat + Calib3d.recoverPose between
 * consecutive frames instead of a Homography for a 2D box. No real camera
 * calibration exists in this app — the assumed pinhole camera matrix
 * (fx=fy=0.8*max(w,h)) is the same guess HrtfBeacon.kt's directionFromBox()
 * and server/tools/depth.py's _estimate_K already make; the ONE existing
 * precedent for "no calibration available, guess a K" in this codebase.
 */
class RotationTracker(private val nfeatures: Int = 500) {

    private val orb: ORB by lazy {
        if (!OpenCVLoader.initDebug()) throw IllegalStateException("OpenCV failed to initialize")
        ORB.create(nfeatures)
    }
    private val matcher: BFMatcher = BFMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING, true)

    private var prevKp: MatOfKeyPoint? = null
    private var prevDesc: Mat? = null

    // Accumulated rotation since the last resetAccumulator() call — a unit
    // quaternion (x, y, z, w), composed by right-multiplying each new
    // per-frame delta on top of it (see processFrame's own comment for the
    // composition convention). Identity until the first successful frame
    // pair.
    @Volatile private var accumQx = 0f
    @Volatile private var accumQy = 0f
    @Volatile private var accumQz = 0f
    @Volatile private var accumQw = 1f

    /**
     * Feed one new camera frame. Matches it against whatever frame was fed
     * last, and — if there are enough correspondences and the resulting
     * Essential matrix decomposition succeeds — folds the newly-estimated
     * rotation into the running accumulator. Always advances the internal
     * "previous frame" reference to this frame, whether or not a rotation
     * could be computed (so a single bad frame doesn't leave the tracker
     * comparing against an increasingly stale reference).
     *
     * Returns true if the accumulator was actually updated this call.
     */
    fun processFrame(frameJpeg: ByteArray): Boolean {
        val gray = decodeToGray(frameJpeg) ?: return false
        val kp = MatOfKeyPoint()
        val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kp, desc)
        if (desc.empty() || kp.empty()) {
            prevKp = null; prevDesc = null
            return false
        }

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
        // to compose onto a running WORLD-frame orientation. If
        // worldOrientation rotates a prev-camera-local vector into world
        // space (v_world = worldOrientation * v_prev), then substituting
        // v_prev = R^T * v_curr gives v_world = worldOrientation * R^T *
        // v_curr — i.e. the NEW world orientation is
        // (old world orientation) composed with R^T, not R. A unit
        // quaternion's inverse is its conjugate, so rather than
        // transposing the Mat we just compute R's quaternion and conjugate
        // it. NOT verified against a live device — see CLAUDE.md's
        // "Server-planned walking path" open-risk note; if a head turn
        // ever sounds mirrored, this is the first thing to check.
        val q = matToQuaternion(r)
        val qInv = floatArrayOf(-q[0], -q[1], -q[2], q[3])
        val composed = quatMultiply(floatArrayOf(accumQx, accumQy, accumQz, accumQw), qInv)
        accumQx = composed[0]; accumQy = composed[1]; accumQz = composed[2]; accumQw = composed[3]
        return true
    }

    /** Current accumulated rotation since the last [resetAccumulator] call, as (x, y, z, w). */
    fun accumulatedRotation(): FloatArray = floatArrayOf(accumQx, accumQy, accumQz, accumQw)

    /** Called whenever the client folds this accumulator into a fresh
     * authoritative server pose (see ToolDispatcher's mapping-stream
     * collector) — starts a new rotation-since-last-fix window. Does NOT
     * clear the ORB reference frame (prevKp/prevDesc) — frame-to-frame
     * matching continuity is independent of when the server last
     * responded. */
    fun resetAccumulator() {
        accumQx = 0f; accumQy = 0f; accumQz = 0f; accumQw = 1f
    }

    /** Full reset (e.g. mode change) — also drops the ORB reference frame. */
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

    private fun decodeToGray(frameJpeg: ByteArray): Mat? {
        val bitmap = BitmapFactory.decodeByteArray(frameJpeg, 0, frameJpeg.size) ?: return null
        val rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val mat = Mat()
        Utils.bitmapToMat(rgba, mat)
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY)
        rgba.recycle()
        return mat
    }

    companion object {
        private const val TAG = "RotationTracker"
        private const val MIN_MATCHES = 15  // well above the 5-point algorithm's bare minimum

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

        private operator fun FloatArray.component1() = this[0]
        private operator fun FloatArray.component2() = this[1]
        private operator fun FloatArray.component3() = this[2]
        private operator fun FloatArray.component4() = this[3]
    }
}
