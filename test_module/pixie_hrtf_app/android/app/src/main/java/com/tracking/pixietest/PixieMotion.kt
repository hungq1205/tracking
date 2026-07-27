package com.tracking.pixietest

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.random.Random

/**
 * Local, Android-side mirror of pixie_hrtf_server.py's PixieMotion —
 * requested directly by the user so this app's pixie visualization/ping
 * tone work even with NO server connection at all. The server's own
 * PixieMotion still runs independently (drives its Gradio dashboard), but
 * this app no longer depends on receiving a pixie position over the wire
 * for anything — head-heading tracking is the only thing the server
 * connection is still for.
 *
 * Loops MOVING (travel ALONG the circumference of a circle to a random
 * point, never a straight chord) then WAITING then repeat, on its own
 * coroutine tick. **Sine-eased motion, per direct request for realism**:
 * each MOVING phase follows a raised-cosine (sine-derivative) angular
 * velocity profile — zero velocity at the start and end of the move,
 * peaking in the middle — instead of an instantly-on/instantly-off
 * constant angular velocity. [speedMps] still means what it says: it's
 * the PEAK velocity the ease profile reaches at its midpoint (chosen so a
 * given move's total duration works out consistently regardless of how
 * far it has to travel), not an average.
 */
class PixieMotion(
    @Volatile var radiusM: Float = 5.0f,
    @Volatile var speedMps: Float = 0.8f,  // moderate pace — the sine ease's average velocity runs ~64% of this peak
    @Volatile var waitS: Float = 10.0f,
    @Volatile var elevationDeg: Float = -30.0f,
) {
    @Volatile var thetaDeg: Float = 0f
        private set
    @Volatile var state: String = "MOVING"
        private set

    private var targetThetaDeg = Random.nextFloat() * 360f - 180f
    private var waitUntilMs = 0L

    // Sine-eased move parameters — fixed once at the START of each MOVING
    // phase (not recomputed mid-move even if radius/speed change live),
    // so a move stays smooth and self-consistent from start to finish.
    private var moveStartThetaDeg = 0f
    private var moveDeltaDeg = 0f
    private var moveDurationS = 0f
    private var moveElapsedS = 0f

    private val scope = CoroutineScope(Dispatchers.Default)
    private var job: Job? = null

    fun start() {
        if (job != null) return
        beginMove()
        job = scope.launch {
            var lastMs = System.currentTimeMillis()
            while (isActive) {
                delay(TICK_MS)
                val now = System.currentTimeMillis()
                val dtS = (now - lastMs) / 1000f
                lastMs = now

                if (state == "WAITING") {
                    if (now >= waitUntilMs) {
                        targetThetaDeg = Random.nextFloat() * 360f - 180f
                        beginMove()
                    }
                    continue
                }

                // MOVING — sine-eased (raised-cosine) angular position:
                // theta(t) = start + delta * (1 - cos(pi * t/T)) / 2.
                // Its derivative is a pure sine — zero velocity at t=0 and
                // t=T, peaking at t=T/2 — a smooth accelerate/decelerate
                // profile instead of an instant-start/instant-stop
                // constant angular velocity.
                moveElapsedS += dtS
                if (moveElapsedS >= moveDurationS) {
                    thetaDeg = targetThetaDeg
                    state = "WAITING"
                    waitUntilMs = now + (waitS * 1000).toLong()
                } else {
                    val frac = (moveElapsedS / moveDurationS).coerceIn(0f, 1f)
                    val eased = (1f - cos((PI * frac).toFloat())) / 2f
                    thetaDeg = wrapDeg(moveStartThetaDeg + moveDeltaDeg * eased)
                }
            }
        }
    }

    private fun beginMove() {
        state = "MOVING"
        moveStartThetaDeg = thetaDeg
        moveDeltaDeg = wrapDeg(targetThetaDeg - thetaDeg)
        // Duration chosen so the ease profile's PEAK velocity (at its
        // midpoint) equals the configured speedMps/radiusM angular rate —
        // integrating a sine velocity profile of peak V over duration T
        // covers V*T*(2/pi) of angular distance, so T = distance*pi/(2V).
        val peakAngularSpeedDegS = Math.toDegrees((speedMps / radiusM.coerceAtLeast(0.01f)).toDouble()).toFloat()
        moveDurationS = if (peakAngularSpeedDegS > 0.001f) {
            (abs(moveDeltaDeg) * PI.toFloat()) / (2f * peakAngularSpeedDegS)
        } else {
            0.001f
        }.coerceAtLeast(0.05f)
        moveElapsedS = 0f
    }

    fun stop() {
        job?.cancel()
        job = null
    }

    companion object {
        private const val TICK_MS = 50L

        private fun wrapDeg(deg: Float): Float {
            var d = deg % 360f
            if (d > 180f) d -= 360f
            if (d < -180f) d += 360f
            return d
        }
    }
}
