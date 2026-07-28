package com.tracking.client.live

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log

/**
 * Android's built-in step detector + a fixed stride length — WALKING/
 * GUIDING's translation-since-last-server-fix estimate, bridging the same
 * ~1Hz server-update gap RotationTracker bridges for rotation. See
 * CLAUDE.md's "Server-planned walking path" note for why translation is
 * deliberately NOT estimated from raw accelerometer peak-detection or from
 * vision: monocular vision has no scale for translation, and
 * `TYPE_STEP_DETECTOR` already does the hard part (gait-pattern
 * recognition) far more robustly than a bespoke peak detector would,
 * regardless of the phone's orientation in a pocket.
 *
 * Deliberately NOT `ImuSensor.kt`/`ImuRecorder.kt` (confirmed dead code —
 * orphaned, never instantiated anywhere in the live app, shaped for
 * offline accel/gyro CSV export, not step counting) — this is a small,
 * purpose-built listener instead.
 *
 * A single fixed stride length, not a per-user model — centimetre
 * accuracy is unnecessary here (this only ever bridges a ~1s gap before
 * the next authoritative server pose corrects it — see ToolDispatcher's
 * reconciliation logic — the error never gets the chance to accumulate
 * beyond that), so a stride-estimation model would be complexity without
 * real payoff.
 */
class PdrStepEstimator(context: Context, private val strideLengthM: Float = STRIDE_LENGTH_M) {

    private val sensorManager =
        context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val stepSensor: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)

    @Volatile private var stepsSinceReset = 0
    @Volatile private var lastStepAtMs = 0L
    private var registered = false

    private val listener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            stepsSinceReset++
            lastStepAtMs = System.currentTimeMillis()
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    /** Starts listening — a no-op (logged once) if this device has no
     * step-detector hardware, in which case translation bridging simply
     * stays at zero (the beacon still bridges rotation via RotationTracker,
     * just snaps position-wise on every server fix instead of extrapolating
     * between them). */
    fun start() {
        if (registered) return
        val sensor = stepSensor
        if (sensor == null) {
            Log.w(TAG, "no TYPE_STEP_DETECTOR sensor on this device — translation bridging disabled")
            return
        }
        sensorManager.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_NORMAL)
        registered = true
    }

    fun stop() {
        if (!registered) return
        sensorManager.unregisterListener(listener)
        registered = false
    }

    /** Distance traveled (metres) since the last [resetAccumulator] call —
     * along whatever heading the caller separately tracks (RotationTracker/
     * the extrapolated pose); this class has no notion of direction, only
     * "how far." */
    fun distanceSinceReset(): Float = stepsSinceReset * strideLengthM

    /** True if the step detector fired within the last [windowMs] — the
     * IMU-derived "is the user actively walking right now" signal
     * requested directly by the user for gating the depth-based obstacle-
     * ahead alert (see ToolDispatcher.checkAndWarnObstacleAhead()):
     * TYPE_STEP_DETECTOR fires per-step (roughly every 0.4-0.8s during
     * normal gait), so a couple of seconds of silence is a reasonable
     * "stopped walking" signal without needing a second, continuous
     * accelerometer-magnitude listener alongside this one. Reads false
     * before the first-ever step (lastStepAtMs starts at 0) — correct,
     * since nothing has confirmed movement yet at that point either. */
    fun isRecentlyMoving(windowMs: Long = MOVING_WINDOW_MS): Boolean =
        lastStepAtMs != 0L && (System.currentTimeMillis() - lastStepAtMs) <= windowMs

    /** Called whenever the client folds this into a fresh authoritative
     * server pose (see ToolDispatcher's mapping-stream collector) — starts
     * a new distance-since-last-fix window. */
    fun resetAccumulator() {
        stepsSinceReset = 0
    }

    companion object {
        private const val TAG = "PdrStepEstimator"
        const val STRIDE_LENGTH_M = 0.7f
        // Comfortably covers normal walking cadence (a step roughly every
        // 0.4-0.8s) while still reading "stopped" within ~2s of actually
        // stopping.
        private const val MOVING_WINDOW_MS = 2000L
    }
}
