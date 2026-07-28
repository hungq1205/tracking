package com.tracking.client.edge

import com.tracking.client.camera.CameraManager
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import org.zeromq.SocketType
import org.zeromq.ZMQ
import org.zeromq.ZMQException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/**
 * [EdgeDevice] implementation for a remote Raspberry-Pi-class edge device
 * (camera + mic + speaker), talking over plain ZeroMQ PUSH/PULL sockets —
 * same wire convention already proven in [EdgeZmqTestClient]/
 * `test_module/edge_mock_app`, productionized here as the real thing
 * [LiveAssistantService] swaps in via the "Use Remote Edge Device" Settings
 * toggle.
 *
 * The edge device always BINDS, this class always CONNECTS (matches every
 * existing PUSH/PULL precedent in this codebase — no REQ/REP handshake,
 * no polling either direction):
 *
 *   frame_out (edge PUSHes, this PULLs) -> [frameFlow]  — JPEG, 640px long-edge, q50
 *   luma_out  (edge PUSHes, this PULLs) -> [lumaFlow]   — raw Y-plane, 360px long-edge, 15fps
 *   mic_out   (edge PUSHes, this PULLs) -> [micFlow]    — 16kHz mono PCM16, 512-sample chunks
 *   audio_in  (edge PULLs,  this PUSHes) <- [emitAudio] — rendered PCM to play on the edge speaker
 *   control   (edge PULLs,  this PUSHes) <- [reportMode] — current mode string, gates luma_out only
 *
 * Each ZMQ message is a 2-frame multipart: [8-byte LE seq][8-byte LE double
 * unix-seconds timestamp], then the raw payload — identical framing to
 * [EdgeZmqTestClient]. [lumaFlow]'s payload additionally carries its own
 * small sub-header (width/height/rowStride/rotationDegrees as four
 * little-endian int32s) ahead of the luma bytes, since — unlike a JPEG or
 * raw PCM chunk — a luma frame needs that metadata to reconstruct a
 * [CameraManager.LumaFrame]; this sub-header is this class's own contract
 * with the Pi-side server, not an existing convention.
 *
 * [reportMode]'s control socket is a NARROW slice of the "dynamic control
 * channel" this class's docstring used to flag as a future follow-up — it
 * only tells the Pi which mode the app is in, purely so the Pi can skip
 * capturing/sending [lumaFlow] when nothing will consume it (only
 * walking/guiding actually process luma client-side — see
 * ToolDispatcher.feedAngleLumaFrame()). A fuller control channel (dynamic
 * frame_interval_ms/frame_quality/etc, matching frameIntervalMs/
 * scanIntervalMs/walkingIntervalMs's per-mode cadence) is still a real,
 * separate follow-up, not attempted here.
 */
class RemoteEdgeDevice(
    private val host: String,
    private val framePort: Int = 5602,
    private val lumaPort: Int = 5604,
    private val micPort: Int = 5601,
    private val audioInPort: Int = 5603,
    private val controlPort: Int = 5605,
) : EdgeDevice {

    private val _frameFlow = MutableSharedFlow<ByteArray>(
        extraBufferCapacity = 2, onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    override val frameFlow: SharedFlow<ByteArray> = _frameFlow

    private val _lumaFlow = MutableSharedFlow<CameraManager.LumaFrame>(
        extraBufferCapacity = 4, onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    override val lumaFlow: SharedFlow<CameraManager.LumaFrame> = _lumaFlow

    private val _micFlow = MutableSharedFlow<ByteArray>(
        extraBufferCapacity = 32, onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    override val micFlow: SharedFlow<ByteArray> = _micFlow

    private val _audioFlow = MutableSharedFlow<ByteArray>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    override val audioFlow: SharedFlow<ByteArray> = _audioFlow

    private var ctx: ZMQ.Context? = null
    private var framePull: ZMQ.Socket? = null
    private var lumaPull: ZMQ.Socket? = null
    private var micPull: ZMQ.Socket? = null
    private var audioPush: ZMQ.Socket? = null
    private var controlPush: ZMQ.Socket? = null
    @Volatile private var running = false
    private var audioSeq = 0L
    private var controlSeq = 0L

    override fun connect() {
        if (running) return
        running = true
        val context = ZMQ.context(1)
        ctx = context

        val frame = context.socket(SocketType.PULL).apply { rcvHWM = 4; connect("tcp://$host:$framePort") }
        framePull = frame
        val luma = context.socket(SocketType.PULL).apply { rcvHWM = 4; connect("tcp://$host:$lumaPort") }
        lumaPull = luma
        val mic = context.socket(SocketType.PULL).apply { rcvHWM = 32; connect("tcp://$host:$micPort") }
        micPull = mic
        val audioOut = context.socket(SocketType.PUSH).apply { sndHWM = 32; connect("tcp://$host:$audioInPort") }
        audioPush = audioOut
        val control = context.socket(SocketType.PUSH).apply { sndHWM = 8; connect("tcp://$host:$controlPort") }
        controlPush = control

        thread(name = "remote-edge-frame", isDaemon = true) {
            recvLoop(frame) { payload -> _frameFlow.tryEmit(payload) }
        }
        thread(name = "remote-edge-luma", isDaemon = true) {
            recvLoop(luma) { payload -> parseLumaPayload(payload)?.let { _lumaFlow.tryEmit(it) } }
        }
        thread(name = "remote-edge-mic", isDaemon = true) {
            recvLoop(mic) { payload -> _micFlow.tryEmit(payload) }
        }
    }

    override fun disconnect() {
        running = false
        framePull?.close(); lumaPull?.close(); micPull?.close(); audioPush?.close(); controlPush?.close()
        ctx?.term()
        framePull = null; lumaPull = null; micPull = null; audioPush = null; controlPush = null
        ctx = null
    }

    override fun emitAudio(pcm: ByteArray) {
        _audioFlow.tryEmit(pcm)
        val push = audioPush ?: return
        val header = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
            .putLong(audioSeq)
            .putDouble(System.currentTimeMillis() / 1000.0)
            .array()
        try {
            push.sendMore(header)
            push.send(pcm, 0)
            audioSeq++
        } catch (_: ZMQException) {
            // socket torn down mid-send during disconnect(); harmless
        }
    }

    override fun reportMode(mode: String) {
        val push = controlPush ?: return
        val header = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
            .putLong(controlSeq)
            .putDouble(System.currentTimeMillis() / 1000.0)
            .array()
        try {
            push.sendMore(header)
            push.send(mode.toByteArray(Charsets.UTF_8), 0)
            controlSeq++
        } catch (_: ZMQException) {
            // socket torn down mid-send during disconnect(); harmless
        }
    }

    private fun parseLumaPayload(payload: ByteArray): CameraManager.LumaFrame? {
        if (payload.size < 16) return null
        val buf = ByteBuffer.wrap(payload).order(ByteOrder.LITTLE_ENDIAN)
        val width = buf.int
        val height = buf.int
        val rowStride = buf.int
        val rotationDegrees = buf.int
        val luma = payload.copyOfRange(16, payload.size)
        return CameraManager.LumaFrame(luma, width, height, rowStride, rotationDegrees)
    }

    private inline fun recvLoop(sock: ZMQ.Socket, onPayload: (ByteArray) -> Unit) {
        sock.receiveTimeOut = 500
        while (running) {
            try {
                sock.recv(0) ?: continue // 16-byte seq+timestamp header, unused here
            } catch (_: ZMQException) {
                break
            }
            val payload = sock.recv(0) ?: continue
            onPayload(payload)
        }
    }
}
