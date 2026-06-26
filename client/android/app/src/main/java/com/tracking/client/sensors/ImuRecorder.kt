package com.tracking.client.sensors

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.launchIn
import kotlinx.coroutines.flow.onEach
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

/**
 * Records IMU readings to a CSV file for offline scan upload.
 *
 * CSV format (no header): timestamp_ns,ax,ay,az,gx,gy,gz
 * One row per reading. The server's scan pipeline expects this format.
 *
 * Usage:
 *   val recorder = ImuRecorder(context, outputDir)
 *   recorder.start(scope)       // begin recording alongside video capture
 *   ...
 *   val file = recorder.stop()  // returns the File for upload
 */
class ImuRecorder(
    context: Context,
    private val outputDir: File,
) {
    private val imuSensor = ImuSensor(context)
    private var writer: BufferedWriter? = null
    private var job: Job? = null
    private var outputFile: File? = null

    val isAvailable: Boolean get() = imuSensor.isAvailable

    fun start(scope: CoroutineScope) {
        if (job?.isActive == true) return
        outputFile = File(outputDir, "imu_data.csv").also { it.parentFile?.mkdirs() }
        writer = BufferedWriter(FileWriter(outputFile))
        job = imuSensor.readings()
            .onEach { r ->
                writer?.write(
                    "${r.timestampNs},${r.accelX},${r.accelY},${r.accelZ}," +
                    "${r.gyroX},${r.gyroY},${r.gyroZ}\n"
                )
            }
            .launchIn(scope)
    }

    fun stop(): File? {
        job?.cancel()
        job = null
        writer?.flush()
        writer?.close()
        writer = null
        return outputFile
    }
}
