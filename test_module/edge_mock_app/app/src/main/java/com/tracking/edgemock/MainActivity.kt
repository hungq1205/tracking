package com.tracking.edgemock

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.Collections

/**
 * Standalone mock-edge-device app — see EdgeMockZmqServer.kt for the socket
 * protocol. Deliberately a separate app/process from the tracking client
 * (com.tracking.client), so it can run on its own device as a real Pi
 * Zero 2 stand-in, not just another screen inside the phone app.
 *
 * Two independent modes, see activity_main.xml:
 * - Mode 1 (unchanged): this app ACTS AS a mock edge device (binds ports),
 *   for testing client/android's own EdgeZmqTestClient without real Pi
 *   hardware.
 * - Mode 2 (new): this app ACTS AS client/android would — connects OUT to
 *   a REAL edge device (already has mic/camera/speaker hardware, its own
 *   firmware) and runs a concurrent load test on all 3 channels. See
 *   EdgeDeviceClient.kt/LoadTestRunner.kt.
 */
class MainActivity : AppCompatActivity() {

    private val server = EdgeMockZmqServer()
    private val handler = Handler(Looper.getMainLooper())
    private var running = false

    private lateinit var startStopButton: Button
    private lateinit var statsText: TextView
    private lateinit var ipText: TextView

    private var audioTrack: AudioTrack? = null

    // ── Mode 2: real-edge-device load test ──────────────────────────────
    private val edgeClient = EdgeDeviceClient()
    private val loadTest = LoadTestRunner(edgeClient)
    private lateinit var edgeHostInput: EditText
    private lateinit var durationInput: EditText
    private lateinit var loadTestButton: Button
    private lateinit var loadTestStatusText: TextView
    private var loadTestStartAtMs = 0L

    private val statsPoller = object : Runnable {
        override fun run() {
            statsText.text = buildString {
                appendLine("Mic audio chunks sent: ${server.stats.micChunksSent}")
                appendLine("Frames sent: ${server.stats.framesSent}")
                appendLine("Audio chunks received from phone: ${server.stats.audioChunksReceived}")
                append("Audio bytes received: ${server.stats.audioBytesReceived}")
            }
            if (running) handler.postDelayed(this, 300)
        }
    }

    // Live progress while a Mode-2 load test is running — same 300ms
    // poll cadence as Mode 1's statsPoller, showing counts/rates as they
    // accumulate rather than only the final report.
    private val loadTestPoller = object : Runnable {
        override fun run() {
            val elapsedS = (System.currentTimeMillis() - loadTestStartAtMs) / 1000.0
            loadTestStatusText.text = "Running… (${"%.1f".format(elapsedS)}s)\n\n" + loadTest.liveReport(elapsedS).summary()
            if (loadTest.isRunning) handler.postDelayed(this, 300)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        startStopButton = findViewById(R.id.startStopButton)
        statsText = findViewById(R.id.statsText)
        ipText = findViewById(R.id.ipText)
        edgeHostInput = findViewById(R.id.edgeHostInput)
        durationInput = findViewById(R.id.durationInput)
        loadTestButton = findViewById(R.id.loadTestButton)
        loadTestStatusText = findViewById(R.id.loadTestStatusText)

        ipText.text = "This device's IP: ${localIpAddress() ?: "unknown (check Wi-Fi connection)"}"

        startStopButton.setOnClickListener {
            if (running) stopServer() else startServer()
        }
        loadTestButton.setOnClickListener {
            if (loadTest.isRunning) stopLoadTest() else startLoadTest()
        }
    }

    private fun startServer() {
        audioTrack = buildAudioTrack().also { it.play() }
        server.onAudioReceived = { pcm -> audioTrack?.write(pcm, 0, pcm.size) }
        server.start()
        running = true
        startStopButton.text = "Stop mock edge server"
        handler.post(statsPoller)
    }

    private fun stopServer() {
        running = false
        server.stop()
        audioTrack?.stop()
        audioTrack?.release()
        audioTrack = null
        startStopButton.text = "Start mock edge server"
    }

    private fun startLoadTest() {
        val host = edgeHostInput.text.toString().trim()
        if (host.isEmpty()) {
            loadTestStatusText.text = "Enter the real edge device's IP first."
            return
        }
        val durationS = durationInput.text.toString().toDoubleOrNull() ?: 4.0
        edgeClient.connect(host)
        loadTestStartAtMs = System.currentTimeMillis()
        loadTestButton.text = "Stop load test"
        loadTestStatusText.text = "Connecting to $host, running for ${durationS}s…"
        handler.post(loadTestPoller)
        loadTest.run(durationS) { report ->
            handler.post {
                edgeClient.disconnect()
                loadTestButton.text = "Start load test"
                loadTestStatusText.text = report.summary()
            }
        }
    }

    private fun stopLoadTest() {
        loadTest.stop()
        edgeClient.disconnect()
        loadTestButton.text = "Start load test"
    }

    override fun onDestroy() {
        super.onDestroy()
        if (running) stopServer()
        if (loadTest.isRunning) stopLoadTest()
    }

    private fun buildAudioTrack(): AudioTrack {
        val minBuf = AudioTrack.getMinBufferSize(16000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        return AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(16000)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(minBuf.coerceAtLeast(4096) * 4)
            .build()
    }

    private fun localIpAddress(): String? {
        return Collections.list(NetworkInterface.getNetworkInterfaces())
            .flatMap { Collections.list(it.inetAddresses) }
            .firstOrNull { !it.isLoopbackAddress && it is Inet4Address }
            ?.hostAddress
    }
}
