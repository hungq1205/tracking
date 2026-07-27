package com.tracking.client.edge

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
 * Standalone ZeroMQ test client for the mock edge device
 * (test_module/edge_mock/mock_edge_server.py) — exists purely to verify the
 * PUSH/PULL wire protocol a real Pi-Zero-2 edge device would use, mirroring
 * [EdgeDevice]'s frame/audio directions:
 *
 *   mic_out   (edge PUSHes, this PULLs)  -> [micAudioFlow]  (play back locally)
 *   frame_out (edge PUSHes, this PULLs)  -> [frameFlow]     (display)
 *   audio_in  (edge PULLs, this PUSHes)  <- [sendAudioOut]  (mock rendered/HRTF audio)
 *
 * Each ZMQ message is [8-byte little-endian seq][8-byte double timestamp]
 * as one frame, then the raw payload as a second frame (multipart) — no
 * manual length-prefixing needed since ZMQ already frames messages.
 *
 * Not wired into [LocalEdgeDevice]/[EdgeDevice] production usage — this is a
 * connectivity/latency check only. A real RemoteEdgeDevice would reuse this
 * same socket setup but feed real edge hardware instead of the mock server.
 */
class EdgeZmqTestClient {

    data class Stats(
        var framesReceived: Int = 0,
        var audioChunksReceived: Int = 0,
        var audioChunksSent: Int = 0,
        var lastFrameSeq: Long = -1,
        var lastAudioSeq: Long = -1,
    )

    val stats = Stats()

    private val _frameFlow = MutableSharedFlow<ByteArray>(
        replay = 1, extraBufferCapacity = 2, onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val frameFlow: SharedFlow<ByteArray> = _frameFlow

    private val _micAudioFlow = MutableSharedFlow<ByteArray>(
        extraBufferCapacity = 32, onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )
    val micAudioFlow: SharedFlow<ByteArray> = _micAudioFlow

    private var ctx: ZMQ.Context? = null
    private var micPull: ZMQ.Socket? = null
    private var framePull: ZMQ.Socket? = null
    private var audioPush: ZMQ.Socket? = null
    @Volatile private var running = false

    fun connect(host: String, audioPort: Int = 5601, framePort: Int = 5602, audioInPort: Int = 5603) {
        if (running) return
        running = true
        val context = ZMQ.context(1)
        ctx = context

        val mic = context.socket(SocketType.PULL)
        mic.rcvHWM = 32
        mic.connect("tcp://$host:$audioPort")
        micPull = mic

        val frame = context.socket(SocketType.PULL)
        frame.rcvHWM = 4
        frame.connect("tcp://$host:$framePort")
        framePull = frame

        val audioOut = context.socket(SocketType.PUSH)
        audioOut.sndHWM = 32
        audioOut.connect("tcp://$host:$audioInPort")
        audioPush = audioOut

        thread(name = "edge-zmq-mic", isDaemon = true) { recvLoop(mic) { payload ->
            stats.audioChunksReceived++
            _micAudioFlow.tryEmit(payload)
        } }
        thread(name = "edge-zmq-frame", isDaemon = true) { recvLoop(frame) { payload ->
            stats.framesReceived++
            _frameFlow.tryEmit(payload)
        } }
    }

    /** Send a mock "rendered audio to play on the edge speaker" chunk, e.g. a test tone. */
    fun sendAudioOut(pcm: ByteArray) {
        val push = audioPush ?: return
        val header = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
            .putLong(stats.audioChunksSent.toLong())
            .putDouble(System.currentTimeMillis() / 1000.0)
            .array()
        try {
            push.sendMore(header)
            push.send(pcm, 0)
            stats.audioChunksSent++
        } catch (_: ZMQException) {
            // socket torn down mid-send during disconnect(); harmless for a test client
        }
    }

    fun disconnect() {
        running = false
        micPull?.close()
        framePull?.close()
        audioPush?.close()
        ctx?.term()
        micPull = null
        framePull = null
        audioPush = null
        ctx = null
    }

    private inline fun recvLoop(sock: ZMQ.Socket, onPayload: (ByteArray) -> Unit) {
        sock.receiveTimeOut = 500
        while (running) {
            val header = try {
                sock.recv(0) ?: continue
            } catch (_: ZMQException) {
                break
            }
            val payload = sock.recv(0) ?: continue
            if (header.size >= 8) {
                val seq = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN).long
                if (sock === micPull) stats.lastAudioSeq = seq else stats.lastFrameSeq = seq
            }
            onPayload(payload)
        }
    }
}
