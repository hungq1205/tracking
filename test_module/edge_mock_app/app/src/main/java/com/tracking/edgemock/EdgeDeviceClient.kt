package com.tracking.edgemock

import org.zeromq.SocketType
import org.zeromq.ZMQ
import org.zeromq.ZMQException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/**
 * Plays the CLIENT/phone side of the mic_out/frame_out/audio_in ZeroMQ link
 * — the opposite role from [EdgeMockZmqServer] (which pretends to BE the
 * edge device, for testing client/android's own `EdgeZmqTestClient`
 * without real Pi hardware). This class is for the reverse test: a REAL
 * edge device already exists (mic + camera + speaker hardware, its own
 * firmware), and this app stands in for `client/android` — a lightweight
 * harness to validate the real device's wire behavior BEFORE wiring it
 * into the full, heavy tracking app. Direct duplicate of client/android's
 * `EdgeZmqTestClient.kt` rather than a shared module — these are
 * deliberately separate standalone apps/processes (same "separately
 * deployed" precedent this codebase already uses elsewhere, e.g.
 * `orb_novelty_gate.py`/`live_path_planner.py`).
 *
 * Same wire protocol as [EdgeMockZmqServer]/`mock_edge_server.py` (the
 * already-established real spec, not invented here): the EDGE DEVICE BINDS
 * three PUSH/PULL sockets, this class CONNECTS to them as a client —
 *
 *   mic_out   (edge PUSHes, this PULLs)  -> [micAudioFlow]  real mic capture
 *   frame_out (edge PUSHes, this PULLs)  -> [frameFlow]     real camera frames
 *   audio_in  (edge PULLs, this PUSHes)  <- [sendAudioOut]  audio for the edge to render
 *
 * Each message is [8-byte little-endian seq][8-byte double timestamp] as
 * one ZMQ frame, then the raw payload as a second frame (multipart) — no
 * manual length-prefixing, ZMQ already frames messages.
 */
class EdgeDeviceClient {

    data class Received(val seq: Long, val timestamp: Double, val payload: ByteArray)

    private var ctx: ZMQ.Context? = null
    private var micPull: ZMQ.Socket? = null
    private var framePull: ZMQ.Socket? = null
    private var audioPush: ZMQ.Socket? = null
    @Volatile private var running = false

    private var micThread: Thread? = null
    private var frameThread: Thread? = null

    var onMicChunk: ((Received) -> Unit)? = null
    var onFrame: ((Received) -> Unit)? = null

    private var audioSeq = 0L

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

        micThread = thread(name = "edge-client-mic", isDaemon = true) { recvLoop(mic, onMicChunk) }
        frameThread = thread(name = "edge-client-frame", isDaemon = true) { recvLoop(frame, onFrame) }
    }

    /** Push one chunk of "audio the edge device should render" — see
     * MainActivity.kt's load-test generator for what this actually sends
     * and why (mono 24kHz PCM16, matching client/android's ONE currently-
     * real audio-out contract — see that file's own comment). */
    fun sendAudioOut(pcm: ByteArray) {
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

    fun disconnect() {
        running = false
        micThread?.join(1000); frameThread?.join(1000)
        micPull?.close(); framePull?.close(); audioPush?.close()
        ctx?.term()
        micPull = null; framePull = null; audioPush = null; ctx = null
        micThread = null; frameThread = null
    }

    private fun recvLoop(sock: ZMQ.Socket, onPayload: ((Received) -> Unit)?) {
        sock.receiveTimeOut = 500
        while (running) {
            val header = try {
                sock.recv(0) ?: continue
            } catch (_: ZMQException) {
                break
            }
            val payload = sock.recv(0) ?: continue
            var seq = -1L
            var ts = 0.0
            if (header.size >= 16) {
                val buf = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN)
                seq = buf.long
                ts = buf.double
            }
            onPayload?.invoke(Received(seq, ts, payload))
        }
    }
}
