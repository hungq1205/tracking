package com.tracking.client.tracking

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.tracking.client.grpc.GrpcClientManager
import com.tracking.client.model.ObjectTrack
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import tracking.Tracking

class TrackingBackend(
    private val grpcManager: GrpcClientManager,
    private val nfeatures: Int = 800,
    private val renewalIntervalMs: Long = 1000L,
) {

    private val orb: ORB by lazy {
        if (!OpenCVLoader.initDebug()) throw IllegalStateException("OpenCV failed to initialize")
        ORB.create(nfeatures)
    }
    private val matcher: BFMatcher = BFMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING, true)

    private var active = false
    private var prompt: String = ""
    private var lastRenewalMs = 0L

    // Protected by synchronized(this)
    private var refKeypoints: MatOfKeyPoint? = null
    private var refDescriptors: Mat? = null
    private var refCenterMat: Mat? = null
    private var initWidth: Float = 0f
    private var initHeight: Float = 0f

    // Last known box — kept even when LOST so the overlay doesn't snap to (0,0)
    @Volatile private var lastBox: FloatArray = floatArrayOf(0f, 0f, 0f, 0f)

    // EMA-smoothed center — reduces jitter from camera shake / ORB noise
    private var smoothCx: Float = 0f
    private var smoothCy: Float = 0f
    private val emaAlpha: Float = 0.4f  // lower = more smoothing, higher = more responsive

    private var refEmbedding: FloatArray? = null

    /**
     * One-shot init: detect object on server then extract ORB reference.
     * Must be called from a dedicated coroutine (not the frame loop) — it blocks on gRPC.
     */
    suspend fun initialize(frameJpeg: ByteArray, prompt: String): ObjectTrack? = withContext(Dispatchers.IO) {
        val stub = grpcManager.trackingStub ?: run {
            Log.e(TAG, "initialize: stub is null")
            return@withContext null
        }
        this@TrackingBackend.prompt = prompt
        Log.d(TAG, "initialize: detectObject prompt='$prompt'")

        val detection = try {
            stub.detectObject(Tracking.DetectionRequest.newBuilder().setPrompt(prompt).build())
        } catch (e: Exception) {
            Log.e(TAG, "detectObject failed", e)
            return@withContext null
        }
        Log.d(TAG, "detectObject: score=${detection.score} box=${detection.boxXyxyList}")
        if (detection.score <= 0f || detection.boxXyxyCount != 4) {
            Log.w(TAG, "initialize: no detection (score=${detection.score})")
            return@withContext null
        }

        val gray = decodeToGray(frameJpeg) ?: run {
            Log.e(TAG, "initialize: decodeToGray failed")
            return@withContext null
        }
        Log.d(TAG, "initialize: frame=${gray.width()}x${gray.height()}")

        val kp = MatOfKeyPoint()
        val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kp, desc)
        if (desc.empty()) {
            Log.w(TAG, "initialize: ORB descriptors empty")
            return@withContext null
        }
        Log.d(TAG, "initialize: ORB keypoints=${kp.toArray().size}")

        val box = clampBox(detection.boxXyxyList.toFloatArray(), gray.width(), gray.height())
        val cx = (box[0] + box[2]) / 2f
        val cy = (box[1] + box[3]) / 2f

        synchronized(this@TrackingBackend) {
            refDescriptors = desc.clone()
            refKeypoints = MatOfKeyPoint(*kp.toArray())
            refCenterMat = Mat(3, 1, CvType.CV_64F).apply {
                put(0, 0, cx.toDouble()); put(1, 0, cy.toDouble()); put(2, 0, 1.0)
            }
            initWidth = box[2] - box[0]
            initHeight = box[3] - box[1]
        }
        lastBox = box
        smoothCx = cx
        smoothCy = cy
        refEmbedding = getEmbedding(box)
        active = true
        lastRenewalMs = System.currentTimeMillis()
        Log.d(TAG, "initialize: success box=${box.toList()} cx=$cx cy=$cy")

        return@withContext ObjectTrack(
            boxXyxy = box.copyOf(),
            centerX = cx, centerY = cy,
            confidence = detection.score,
            visible = true, status = "INITIALIZED",
            frameWidth = gray.width(), frameHeight = gray.height(),
        )
    }

    /**
     * Per-frame update — pure local ORB + homography, no network calls, runs every frame.
     * Renewal is triggered asynchronously in the background when the interval elapses.
     */
    suspend fun update(frameJpeg: ByteArray): ObjectTrack? = withContext(Dispatchers.IO) {
        if (!active) return@withContext null

        // Take a snapshot of reference state under lock so renewal can update concurrently
        val (localDesc, localKp, localCenter, localW, localH) = synchronized(this@TrackingBackend) {
            if (refDescriptors == null || refKeypoints == null || refCenterMat == null)
                return@withContext null
            Snapshot(refDescriptors!!.clone(), MatOfKeyPoint(*refKeypoints!!.toArray()), refCenterMat!!.clone(), initWidth, initHeight)
        }

        val gray = decodeToGray(frameJpeg) ?: return@withContext buildLostTrack(0, 0)
        val kp = MatOfKeyPoint()
        val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kp, desc)
        if (desc.empty() || kp.empty()) return@withContext buildLostTrack(gray.width(), gray.height())

        val matches = MatOfDMatch()
        matcher.match(localDesc, desc, matches)
        val matchesArray = matches.toArray()
        Log.d(TAG, "update: matches=${matchesArray.size}")
        if (matchesArray.size < 10) {
            Log.w(TAG, "update: too few matches (${matchesArray.size})")
            return@withContext buildLostTrack(gray.width(), gray.height())
        }

        val srcPoints = mutableListOf<Point>()
        val dstPoints = mutableListOf<Point>()
        for (m in matchesArray) {
            val s = localKp.toArray().getOrNull(m.queryIdx)?.pt ?: continue
            val d = kp.toArray().getOrNull(m.trainIdx)?.pt ?: continue
            srcPoints.add(s); dstPoints.add(d)
        }
        val srcPts = MatOfPoint2f().apply { fromList(srcPoints) }
        val dstPts = MatOfPoint2f().apply { fromList(dstPoints) }

        val inlierMask = Mat()
        val homography = Calib3d.findHomography(srcPts, dstPts, Calib3d.RANSAC, 5.0, inlierMask)
        if (homography == null || homography.empty()) return@withContext buildLostTrack(gray.width(), gray.height())

        // Inlier anchor points only (mirrors desktop's anchor_pts from RANSAC inliers)
        val inlierDstPoints = dstPoints.filterIndexed { i, _ ->
            i < inlierMask.rows() && inlierMask.get(i, 0)?.firstOrNull()?.toInt() == 1
        }
        val matchedKpX = inlierDstPoints.map { it.x.toFloat() }
        val matchedKpY = inlierDstPoints.map { it.y.toFloat() }

        // Inlier ratio — mirrors desktop's float(np.mean(inliers))
        val inlierCount = Core.countNonZero(inlierMask)
        val conf = if (matchesArray.isNotEmpty()) inlierCount.toFloat() / matchesArray.size else 0f

        val transformed = Mat()
        Core.gemm(homography, localCenter, 1.0, Mat(), 0.0, transformed)
        val w = transformed[2, 0][0]
        if (w == 0.0) return@withContext buildLostTrack(gray.width(), gray.height())

        val rawCx = (transformed[0, 0][0] / w).toFloat()
        val rawCy = (transformed[1, 0][0] / w).toFloat()

        // EMA smoothing on center position — reduces jitter from camera shake / ORB noise
        smoothCx = emaAlpha * rawCx + (1f - emaAlpha) * smoothCx
        smoothCy = emaAlpha * rawCy + (1f - emaAlpha) * smoothCy
        val cx = smoothCx
        val cy = smoothCy

        val box = clampBox(
            floatArrayOf(cx - localW / 2f, cy - localH / 2f, cx + localW / 2f, cy + localH / 2f),
            gray.width(), gray.height()
        )
        lastBox = box

        if (System.currentTimeMillis() - lastRenewalMs > renewalIntervalMs) {
            lastRenewalMs = System.currentTimeMillis()
            GlobalScope.launch(Dispatchers.IO) { renewal(frameJpeg) }
        }

        Log.d(TAG, "update: cx=$cx cy=$cy conf=$conf inliers=$inlierCount/${matchesArray.size} box=${box.toList()}")
        return@withContext ObjectTrack(
            boxXyxy = box,
            centerX = cx, centerY = cy,
            confidence = conf, visible = true, status = "TRACKING",
            frameWidth = gray.width(), frameHeight = gray.height(),
            matchedKeypointsX = matchedKpX,
            matchedKeypointsY = matchedKpY,
        )
    }

    fun stop() {
        active = false
        synchronized(this) { refDescriptors = null; refKeypoints = null; refCenterMat = null }
    }

    private data class Snapshot(
        val desc: Mat, val kp: MatOfKeyPoint, val center: Mat, val initW: Float, val initH: Float
    )

    private suspend fun renewal(frameJpeg: ByteArray) {
        if (!active || prompt.isBlank()) return
        val stub = grpcManager.trackingStub ?: return
        val detection = try {
            stub.detectObject(Tracking.DetectionRequest.newBuilder().setPrompt(prompt).build())
        } catch (e: Throwable) {
            Log.w(TAG, "renewal detectObject failed", e); return
        }
        if (detection.score < 0.2f || detection.boxXyxyCount != 4) return

        val currentEmbedding = getEmbedding(detection.boxXyxyList.toFloatArray()) ?: return
        val previous = refEmbedding ?: return
        if (!isSimilar(previous, currentEmbedding)) return

        val gray = decodeToGray(frameJpeg) ?: return
        val kp = MatOfKeyPoint(); val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kp, desc)
        if (desc.empty()) return

        val box = clampBox(detection.boxXyxyList.toFloatArray(), gray.width(), gray.height())
        val cx = (box[0] + box[2]) / 2f
        val cy = (box[1] + box[3]) / 2f
        synchronized(this) {
            refKeypoints = MatOfKeyPoint(*kp.toArray())
            refDescriptors = desc.clone()
            refCenterMat = Mat(3, 1, CvType.CV_64F).apply {
                put(0, 0, cx.toDouble()); put(1, 0, cy.toDouble()); put(2, 0, 1.0)
            }
            initWidth = box[2] - box[0]
            initHeight = box[3] - box[1]
        }
        lastBox = box
        smoothCx = (box[0] + box[2]) / 2f
        smoothCy = (box[1] + box[3]) / 2f
        refEmbedding = currentEmbedding
        Log.d(TAG, "renewal: updated reference box=${box.toList()}")
    }

    private suspend fun getEmbedding(box: FloatArray): FloatArray? {
        val stub = grpcManager.trackingStub ?: return null
        return try {
            val resp = stub.getEmbedding(Tracking.EmbeddingRequest.newBuilder().addAllBoxXyxy(box.asList()).build())
            resp.embeddingList.toFloatArray()
        } catch (e: Exception) {
            Log.e(TAG, "getEmbedding failed", e); null
        }
    }

    private fun isSimilar(a: FloatArray, b: FloatArray): Boolean {
        if (a.size != b.size) return false
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        if (na == 0f || nb == 0f) return false
        return (dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb))) > 0.4f
    }

    private fun buildLostTrack(frameWidth: Int, frameHeight: Int): ObjectTrack {
        val box = lastBox.copyOf()
        return ObjectTrack(
            boxXyxy = box,
            centerX = (box[0] + box[2]) / 2f,
            centerY = (box[1] + box[3]) / 2f,
            confidence = 0f, visible = false, status = "LOST",
            frameWidth = frameWidth, frameHeight = frameHeight,
        )
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

    private fun clampBox(box: FloatArray, width: Int, height: Int): FloatArray = floatArrayOf(
        box[0].coerceAtLeast(0f), box[1].coerceAtLeast(0f),
        box[2].coerceAtMost(width.toFloat()), box[3].coerceAtMost(height.toFloat()),
    )

    companion object {
        private const val TAG = "TrackingBackend"
    }
}
