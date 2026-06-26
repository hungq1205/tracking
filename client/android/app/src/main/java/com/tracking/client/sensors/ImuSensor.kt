package com.tracking.client.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow

data class ImuReading(
    val timestampNs: Long,
    val accelX: Float,
    val accelY: Float,
    val accelZ: Float,
    val gyroX: Float,
    val gyroY: Float,
    val gyroZ: Float,
)

/**
 * Streams fused IMU readings (accelerometer + gyroscope) from Android SensorManager.
 *
 * Both sensors run at 100 Hz (10 ms period) — sufficient for GTSAM VIO pre-integration
 * and avoids the HIGH_SAMPLING_RATE_SENSORS permission needed above 200 Hz on Android 12+.
 * Readings are paired by latching the most recent sample of the complementary sensor.
 * The hardware timestamp (SensorEvent.timestamp, nanoseconds) is preserved so the server
 * can time-synchronise IMU data with camera frames.
 */
class ImuSensor(context: Context) {

    private val sensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager

    private val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    val isAvailable: Boolean get() = accel != null && gyro != null

    fun readings(): Flow<ImuReading> = callbackFlow {
        var latestAccel: FloatArray? = null
        var latestGyro: FloatArray? = null
        var latestAccelTs: Long = 0L
        var latestGyroTs: Long = 0L

        val listener = object : SensorEventListener {
            override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit

            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        latestAccel = event.values.clone()
                        latestAccelTs = event.timestamp
                        val g = latestGyro ?: return
                        trySend(
                            ImuReading(
                                timestampNs = event.timestamp,
                                accelX = event.values[0],
                                accelY = event.values[1],
                                accelZ = event.values[2],
                                gyroX = g[0],
                                gyroY = g[1],
                                gyroZ = g[2],
                            )
                        )
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        latestGyro = event.values.clone()
                        latestGyroTs = event.timestamp
                        val a = latestAccel ?: return
                        trySend(
                            ImuReading(
                                timestampNs = event.timestamp,
                                accelX = a[0],
                                accelY = a[1],
                                accelZ = a[2],
                                gyroX = event.values[0],
                                gyroY = event.values[1],
                                gyroZ = event.values[2],
                            )
                        )
                    }
                }
            }
        }

        sensorManager.registerListener(listener, accel, 10_000)  // 100 Hz
        sensorManager.registerListener(listener, gyro, 10_000)

        awaitClose {
            sensorManager.unregisterListener(listener)
        }
    }
}
