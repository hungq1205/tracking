package com.tracking.edgemock

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.Date
import kotlin.concurrent.thread
import kotlin.math.sin
import org.zeromq.SocketType
import org.zeromq.ZMQ
import org.zeromq.ZMQException

/**
 * Mock edge device — stands in for a real Pi Zero 2 by BINDING the same
 * PUSH/PULL sockets the tracking app's `EdgeZmqTestClient` (Edge ZMQ Test
 * screen, client/android) connects to, so the whole link can be exercised
 * with a second physical Android device (or emulator) and zero non-Android
 * pieces. Deliberately its own standalone app, not a screen inside the main
 * client — this is a stand-in for separate Pi hardware, so it should run as
 * a separate process/device, not share the tracking app's process.
 *
 *   mic_out   PUSH bind :5601  -> mock mic audio (sweeping tone)
 *   frame_out PUSH bind :5602  -> mock camera frames (drawn bitmaps)
 *   audio_in  PULL bind :5603  <- whatever the phone sends to "play"
 *
 * Binding a plain TCP port from an Android app needs nothing beyond the
 * INTERNET permission -- no root, no special API -- as long as the port
 * isn't in the <1024 reserved range (ours are 5601-5603).
 */
class EdgeMockZmqServer {

    data class Stats(
        var micChunksSent: Int = 0,
        var framesSent: Int = 0,
        var audioChunksReceived: Int = 0,
        var audioBytesReceived: Long = 0,
    )

    val stats = Stats()

    private var ctx: ZMQ.Context? = null
    @Volatile private var running = false
    private var threads = listOf<Thread>()

    /** Called for every audio chunk received on audio_in, e.g. to play it back locally. */
    var onAudioReceived: ((ByteArray) -> Unit)? = null

    fun start(bindHost: String = "*", audioPort: Int = 5601, framePort: Int = 5602, audioInPort: Int = 5603) {
        if (running) return
        running = true
        val context = ZMQ.context(1)
        ctx = context

        threads = listOf(
            thread(name = "edge-mock-mic", isDaemon = true) { micOutLoop(context, "tcp://$bindHost:$audioPort") },
            thread(name = "edge-mock-frame", isDaemon = true) { frameOutLoop(context, "tcp://$bindHost:$framePort") },
            thread(name = "edge-mock-audio-in", isDaemon = true) { audioInLoop(context, "tcp://$bindHost:$audioInPort") },
        )
    }

    fun stop() {
        running = false
        threads.forEach { it.join(1000) }
        ctx?.term()
        ctx = null
    }

    private fun micOutLoop(context: ZMQ.Context, addr: String) {
        val sock = context.socket(SocketType.PUSH)
        sock.sndHWM = 32
        sock.bind(addr)
        val sampleRate = 16000
        val chunkSamples = 512
        var phase = 0.0
        var seq = 0L
        val intervalMs = (chunkSamples * 1000L) / sampleRate
        while (running) {
            val freq = 440.0 + 220.0 * sin(seq / 200.0)
            val bytes = ByteArray(chunkSamples * 2)
            for (i in 0 until chunkSamples) {
                val t = (phase + i) / sampleRate
                val sampleVal = (sin(2.0 * Math.PI * freq * t) * 0.3 * Short.MAX_VALUE).toInt().toShort()
                bytes[i * 2] = (sampleVal.toInt() and 0xFF).toByte()
                bytes[i * 2 + 1] = ((sampleVal.toInt() shr 8) and 0xFF).toByte()
            }
            phase += chunkSamples
            sendFramed(sock, seq, bytes)
            seq++
            stats.micChunksSent = seq.toInt()
            Thread.sleep(intervalMs)
        }
        sock.close()
    }

    private fun frameOutLoop(context: ZMQ.Context, addr: String) {
        val sock = context.socket(SocketType.PUSH)
        sock.sndHWM = 4
        sock.bind(addr)
        var seq = 0L
        val colors = intArrayOf(Color.rgb(220, 60, 60), Color.rgb(60, 160, 220), Color.rgb(60, 200, 100), Color.rgb(230, 190, 40))
        while (running) {
            val jpg = renderMockFrame(seq, colors[(seq % colors.size).toInt()])
            sendFramed(sock, seq, jpg)
            seq++
            stats.framesSent = seq.toInt()
            Thread.sleep(500) // 2fps
        }
        sock.close()
    }

    private fun renderMockFrame(seq: Long, color: Int): ByteArray {
        val bmp = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(color)
        val paint = Paint().apply {
            this.color = Color.WHITE
            strokeWidth = 4f
            style = Paint.Style.STROKE
        }
        canvas.drawRect(20f, 20f, 620f, 460f, paint)
        paint.style = Paint.Style.FILL
        paint.textSize = 32f
        canvas.drawText("MOCK EDGE FRAME #$seq", 40f, 60f, paint)
        canvas.drawText(SimpleDateFormat("HH:mm:ss").format(Date()), 40f, 100f, paint)
        val out = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 80, out)
        bmp.recycle()
        return out.toByteArray()
    }

    private fun audioInLoop(context: ZMQ.Context, addr: String) {
        val sock = context.socket(SocketType.PULL)
        sock.rcvHWM = 64
        sock.bind(addr)
        sock.receiveTimeOut = 500
        while (running) {
            try { sock.recv(0) } catch (_: ZMQException) { break } ?: continue
            val payload = sock.recv(0) ?: continue
            stats.audioChunksReceived++
            stats.audioBytesReceived += payload.size
            onAudioReceived?.invoke(payload)
        }
        sock.close()
    }

    private fun sendFramed(sock: ZMQ.Socket, seq: Long, payload: ByteArray) {
        val header = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN)
            .putLong(seq)
            .putDouble(System.currentTimeMillis() / 1000.0)
            .array()
        try {
            sock.sendMore(header)
            sock.send(payload, 0)
        } catch (_: ZMQException) {
            // torn down mid-send during stop(); harmless for a test server
        }
    }
}
