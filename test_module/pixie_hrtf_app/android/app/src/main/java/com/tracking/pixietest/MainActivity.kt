package com.tracking.pixietest

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

/**
 * Test harness for validating anchor-relative head tracking + a moving
 * pixie before building the real "Pixie" persistent-companion entity into
 * the main client app (client/android/) — see
 * test_module/pixie_hrtf_app/README.md.
 *
 * Every camera frame feeds [HeadingEstimator] locally (continuous ORB/
 * Essential-matrix drift tracking, see RotationTracker — Lucas-Kanade
 * optical flow was tried and dropped, ORB only now). [HeadingEstimator]'s
 * heading works entirely offline too (its accumulator's own baseline IS
 * the anchor — a server fix, when available, just corrects it more
 * precisely). A subset of frames is also forwarded to pixie_hrtf_server.py
 * over [WsHeadTrackClient] purely for that more-precise RTAB-Map
 * correction; connecting is now optional, not required — see [PixieMotion],
 * which runs the pixie's circular motion locally, independent of any
 * server connection (the server still runs its own independent copy for
 * its Gradio dashboard, but this app no longer depends on receiving it
 * over the wire — a real gap fixed per direct feedback that the app was
 * previously useless without a live server+RTAB-Map+DA3 stack running).
 *
 * A fixed UI tick recombines the local heading estimate with the pixie's
 * local position into TWO views (both drawn by [PixiePositionView]):
 * the pixie's anchor-relative bearing (left, doesn't rotate with your
 * head) and YOUR OWN drift from the anchor's original forward direction
 * (right, does rotate — literally headingDeg). The pixie's ego-relative
 * azimuth (not shown directly on either compass, but derivable as the
 * angular difference between them) drives [PingTonePlayer] — spatialized
 * via real HRTF convolution AND volume-scaled by deviation, quiet when
 * you're facing the pixie, louder the more you're turned away. There's no
 * camera preview in this UI — nothing useful to look at in the raw feed.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var camera: SimpleCameraSource
    private val ws = WsHeadTrackClient()
    private val estimator = HeadingEstimator()
    private lateinit var pingPlayer: PingTonePlayer
    // Runs locally regardless of server connection — see class doc and
    // PixieMotion's own doc for why this moved off the server dependency.
    private val pixieMotion = PixieMotion()

    private lateinit var hostInput: EditText
    private lateinit var portInput: EditText
    private lateinit var connectButton: Button
    private lateinit var resetButton: Button
    private lateinit var resolutionSpinner: Spinner
    private lateinit var pointsSpinner: Spinner
    private lateinit var targetFpsInput: EditText
    private lateinit var pixieRadiusInput: EditText
    private lateinit var pixieSpeedInput: EditText
    private lateinit var pixieWaitInput: EditText
    private lateinit var elevationSeekBar: SeekBar
    private lateinit var elevationLabel: TextView
    private lateinit var positionView: PixiePositionView
    private lateinit var statusText: TextView

    private val uiHandler = Handler(Looper.getMainLooper())
    private var connected = false
    private var lastStatusLine = ""

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCameraAndAudio() else {
            statusText.text = "Camera permission denied — cannot run this test."
        }
    }

    private val tickRunnable = object : Runnable {
        override fun run() {
            tick()
            uiHandler.postDelayed(this, TICK_MS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        hostInput = findViewById(R.id.hostInput)
        portInput = findViewById(R.id.portInput)
        connectButton = findViewById(R.id.connectButton)
        resetButton = findViewById(R.id.resetButton)
        resolutionSpinner = findViewById(R.id.resolutionSpinner)
        pointsSpinner = findViewById(R.id.pointsSpinner)
        targetFpsInput = findViewById(R.id.targetFpsInput)
        pixieRadiusInput = findViewById(R.id.pixieRadiusInput)
        pixieSpeedInput = findViewById(R.id.pixieSpeedInput)
        pixieWaitInput = findViewById(R.id.pixieWaitInput)
        elevationSeekBar = findViewById(R.id.elevationSeekBar)
        elevationLabel = findViewById(R.id.elevationLabel)
        positionView = findViewById(R.id.pixiePositionView)
        statusText = findViewById(R.id.statusText)

        camera = SimpleCameraSource(applicationContext).apply { jpegIntervalMs = SEND_INTERVAL_MS }
        pingPlayer = PingTonePlayer(applicationContext)

        setUpConfigSpinners()
        setUpElevationSlider()

        connectButton.setOnClickListener { if (connected) doDisconnect() else doConnect() }
        resetButton.setOnClickListener {
            estimator.resetLocal()
            ws.sendReset()
            statusText.text = "Anchor reset — look in the direction you want as \"straight ahead\"."
        }

        ws.onHeadingUpdate = { update -> estimator.onServerHeading(update) }
        ws.onStatus = { msg -> uiHandler.post { lastStatusLine = msg } }

        camera.onLumaFrame = { luma, w, h, rowStride, rotDeg ->
            // Continuous local drift tracking — every frame, decoupled
            // from the camera thread inside HeadingEstimator (drop-to-
            // latest), regardless of whether this frame is also sent to
            // the server.
            estimator.submitLumaFrame(luma, w, h, rowStride, rotDeg)
        }
        camera.onFrame = { jpeg, tsNs -> onCameraFrame(jpeg, tsNs) }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCameraAndAudio()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setUpConfigSpinners() {
        val resolutions = RotationTracker.RESOLUTION_PRESETS.toTypedArray()
        resolutionSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, resolutions.map { "${it}px" }
        )
        val defaultResIdx = resolutions.indexOf(estimator.rotationTracker.processResolution).coerceAtLeast(0)
        resolutionSpinner.setSelection(defaultResIdx)
        resolutionSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                estimator.rotationTracker.processResolution = resolutions[position]
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        val pointCounts = (1..10).map { it * 100 }  // 100..1000, step 100
        pointsSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item, pointCounts.map { it.toString() }
        )
        val defaultPtsIdx = pointCounts.indexOf(estimator.rotationTracker.maxFeatures).coerceAtLeast(0)
        pointsSpinner.setSelection(defaultPtsIdx)
        pointsSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                estimator.rotationTracker.maxFeatures = pointCounts[position]
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun setUpElevationSlider() {
        // Y-axis (elevation) coordinate — reported as sounding "too high
        // up" at the fixed 20 deg default with no way to change it; this
        // is that missing control. 0 = level with your head, + = above,
        // - = below, same convention HrtfBeacon's elevationDeg always used.
        elevationSeekBar.progress = pixieMotion.elevationDeg.toInt()
        elevationLabel.text = "Pixie elevation: ${pixieMotion.elevationDeg.toInt()} deg"
        elevationSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                pixieMotion.elevationDeg = progress.toFloat()
                elevationLabel.text = "Pixie elevation: $progress deg"
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun startCameraAndAudio() {
        camera.start(this)
        pingPlayer.start()
        pixieMotion.start()
        uiHandler.post(tickRunnable)
    }

    private fun doConnect() {
        val host = hostInput.text.toString().trim()
        val port = portInput.text.toString().trim().toIntOrNull() ?: DEFAULT_PORT
        if (host.isEmpty()) {
            statusText.text = "Enter the server's host/IP first."
            return
        }
        ws.connect(host, port)
        connected = true
        connectButton.text = "Disconnect"
    }

    private fun doDisconnect() {
        ws.disconnect()
        connected = false
        connectButton.text = "Connect"
        // Deliberately NOT muting pingPlayer here — the pixie/ping keep
        // working locally regardless of server connection (see class doc).
    }

    private fun onCameraFrame(jpeg: ByteArray, tsNs: Long) {
        // SimpleCameraSource already throttles how often this fires
        // (jpegIntervalMs) — only the "are we even connected" gate is left
        // to check here.
        if (!connected || !ws.isConnected) return
        estimator.onFrameSentToServer(tsNs)
        ws.sendFrame(jpeg, tsNs)
    }

    private fun tick() {
        // Applied live every tick rather than needing a button — cheap,
        // and lets you type a new value and see it take effect immediately.
        estimator.targetFps = targetFpsInput.text.toString().toIntOrNull() ?: 20
        pixieMotion.radiusM = pixieRadiusInput.text.toString().toFloatOrNull()?.coerceAtLeast(0.1f) ?: pixieMotion.radiusM
        pixieMotion.speedMps = pixieSpeedInput.text.toString().toFloatOrNull()?.coerceAtLeast(0.01f) ?: pixieMotion.speedMps
        pixieMotion.waitS = pixieWaitInput.text.toString().toFloatOrNull()?.coerceAtLeast(0f) ?: pixieMotion.waitS

        val headingDeg = estimator.currentHeadingDeg()
        // Pixie position always comes from the LOCAL motion sim now — see
        // PixieMotion's doc for why (works with zero server connection).
        val anchorAzimuth = pixieMotion.thetaDeg  // doesn't rotate with head
        val pixieEl = pixieMotion.elevationDeg
        var egoAzimuth = anchorAzimuth - headingDeg  // rotates with head — 0 = facing the pixie exactly
        egoAzimuth = ((egoAzimuth + 540f) % 360f) - 180f  // wrap to (-180, 180]

        // Always active — no server connection required for any of this.
        // pingPlayer spatializes AND scales volume from the same
        // ego-relative azimuth — see PingTonePlayer's own doc.
        pingPlayer.updateDirection(egoAzimuth, pixieEl)
        // Right compass now shows YOUR OWN drift from the anchor
        // (headingDeg) instead of the pixie's facing-relative bearing —
        // requested directly; egoAzimuth is still computed above since
        // it's what the ping tone/HRTF panning use.
        positionView.update(anchorAzimuth, pixieEl, headingDeg, muted = false)

        statusText.text = buildString {
            appendLine("Pixie: ${pixieMotion.state}  radius=${pixieMotion.radiusM}m  speed=${pixieMotion.speedMps}m/s")
            appendLine(if (connected) "Server: connected" else "Server: disconnected (pixie still moves locally)")
            if (lastStatusLine.isNotEmpty()) appendLine(lastStatusLine)
            appendLine("tracking_ok (server heading fix): ${estimator.lastTrackingOk}")
            appendLine("anchor-relative heading (your drift): %.1f deg".format(headingDeg))
            appendLine("pixie: anchor-az=%.1f deg  el=%.1f deg".format(anchorAzimuth, pixieEl))
            appendLine("facing deviation (ping azimuth+volume drive on this): %.1f deg".format(egoAzimuth))
            appendLine("resolution: ${estimator.rotationTracker.processResolution}px  " +
                "points: ${estimator.rotationTracker.maxFeatures}  target fps: ${estimator.targetFps}")
            appendLine("processing time: ${estimator.lastProcessingMs()}ms")
            append("achieved local tracking rate: %.1f fps".format(estimator.lastFps()))
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        uiHandler.removeCallbacks(tickRunnable)
        camera.stop()
        ws.disconnect()
        pingPlayer.stop()
        pixieMotion.stop()
        estimator.shutdown()
    }

    companion object {
        private const val DEFAULT_PORT = 8765
        private const val SEND_INTERVAL_MS = 300L
        private const val TICK_MS = 33L
    }
}
