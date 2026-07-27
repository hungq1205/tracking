package com.tracking.client.audio

import android.content.Context
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

/**
 * On-device TTS for reading mode — replaces the old server-side
 * PerceptionService.Synthesize (KokoroTTS) round trip for THIS path only.
 * Gemini's own spoken voice is unrelated and unaffected — that comes
 * straight off the Gemini Live audio stream via StreamingAudioPlayer, never
 * through here. Requested directly by the user so reading-mode TTS keeps
 * working even when the gRPC server is unreachable — a real, repeated
 * issue confirmed via logcat this session: every read_aloud failure traced
 * back to ECONNREFUSED against the Synthesize RPC, even though scanning/
 * OCR (which never touch that RPC) worked fine the whole time.
 *
 * Android's own TextToSpeech already queues and plays utterances back to
 * back on its own (`speak(text, QUEUE_ADD, ...)`), so no manual
 * "wait for the previous one to finish, then start the next" logic is
 * needed for gapless playback itself. What this class adds on top, per
 * direct user request: a bounded lookahead of at most [MAX_QUEUED_AHEAD]
 * (3) sentences hand to the OS queue at any one time, refilled one at a
 * time as each finishes (via UtteranceProgressListener), rather than
 * dumping an entire — possibly still-growing, live-reading — buffer into
 * the OS queue up front. This keeps ToolDispatcher's own persistent cursor
 * (state.readingCursorSentenceIndex) the real source of truth for "how far
 * read," and keeps stop()/interruption bounded to a small, known number of
 * in-flight utterances instead of an arbitrarily long pre-committed
 * backlog that wouldn't reflect new text a live-reading session appends
 * mid-flight.
 */
class ReadingTtsPlayer(context: Context) {

    companion object {
        private const val TAG = "ReadingTtsPlayer"
        private const val MAX_QUEUED_AHEAD = 3
    }

    private var tts: TextToSpeech? = null
    @Volatile private var isReady = false

    // Settings "Other Sound Volume" slider — see SettingsScreen.kt. Applied
    // per-utterance via the KEY_PARAM_VOLUME bundle in fillQueue() below
    // (Android's TextToSpeech has no single persistent-track volume knob
    // the way AudioTrack/ExoPlayer do).
    @Volatile private var volume = 1f

    fun setVolume(gain: Float) {
        volume = gain.coerceIn(0f, 1f)
    }

    private class Session(
        val globalStartIndex: Int,
        val sentences: List<String>,
        val onSentenceDone: (globalIndex: Int) -> Unit,
        val onAllDone: () -> Unit,
    ) {
        var nextToQueue: Int = 0
        var queuedCount: Int = 0
    }

    // Single active session — starting a new one implicitly supersedes
    // whatever was playing before (same "newest wins" semantics the old
    // speakSentencesFrom() had via readingJob?.cancel()).
    @Volatile private var session: Session? = null

    init {
        val engine = TextToSpeech(context.applicationContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                isReady = true
                Log.d(TAG, "TextToSpeech initialized")
            } else {
                Log.e(TAG, "TextToSpeech init failed: status=$status")
            }
        }
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                Log.d(TAG, "onStart($utteranceId)")
            }
            override fun onDone(utteranceId: String?) {
                Log.d(TAG, "onDone($utteranceId)")
                handleSentenceFinished(utteranceId)
            }
            @Deprecated("Deprecated in API 21, still required to override")
            override fun onError(utteranceId: String?) {
                Log.e(TAG, "onError($utteranceId)")
                handleSentenceFinished(utteranceId)
            }
            override fun onError(utteranceId: String?, errorCode: Int) {
                Log.e(TAG, "onError($utteranceId, code=$errorCode)")
                handleSentenceFinished(utteranceId)
            }
        })
        try {
            engine.language = Locale.getDefault()
        } catch (e: Exception) {
            Log.w(TAG, "setLanguage failed, falling back to engine default: ${e.message}")
        }
        tts = engine
    }

    private fun handleSentenceFinished(utteranceId: String?) {
        val finishedIdx = utteranceId?.toIntOrNull() ?: return
        val s = session ?: return
        if (finishedIdx < s.globalStartIndex || finishedIdx >= s.globalStartIndex + s.sentences.size) return
        s.queuedCount--
        s.onSentenceDone(finishedIdx + 1) // cursor = one past the sentence that just finished
        fillQueue(s)
        val allQueued = s.nextToQueue >= s.sentences.size
        if (allQueued && s.queuedCount <= 0) {
            if (session === s) session = null
            s.onAllDone()
        }
    }

    private fun fillQueue(s: Session) {
        val engine = tts ?: return
        val params = Bundle().apply { putFloat(TextToSpeech.Engine.KEY_PARAM_VOLUME, volume) }
        while (s.queuedCount < MAX_QUEUED_AHEAD && s.nextToQueue < s.sentences.size) {
            val globalIdx = s.globalStartIndex + s.nextToQueue
            val text = s.sentences[s.nextToQueue]
            engine.speak(text, TextToSpeech.QUEUE_ADD, params, globalIdx.toString())
            s.nextToQueue++
            s.queuedCount++
        }
    }

    /** Speaks [sentences] (whose first element is at [globalStartIndex] in
     * the caller's own full sentence list), feeding the OS TTS queue at
     * most [MAX_QUEUED_AHEAD] at a time. [onSentenceDone] fires with the
     * NEW cursor position (one past whichever sentence just finished)
     * after each one completes — the caller should store this into its own
     * persistent position (e.g. state.readingCursorSentenceIndex)
     * immediately, same as the old PCM-streaming design did per chunk.
     * [onAllDone] fires once every sentence has finished. Supersedes any
     * currently-playing session outright. */
    fun speakFrom(
        globalStartIndex: Int,
        sentences: List<String>,
        onSentenceDone: (globalIndex: Int) -> Unit,
        onAllDone: () -> Unit,
    ) {
        if (!isReady) {
            Log.w(TAG, "speakFrom() called before TextToSpeech finished initializing — skipping")
            onAllDone()
            return
        }
        stop()
        if (sentences.isEmpty()) {
            onAllDone()
            return
        }
        val s = Session(globalStartIndex, sentences, onSentenceDone, onAllDone)
        session = s
        fillQueue(s)
    }

    /** Stops playback and clears the OS TTS queue immediately — the
     * reading-mode counterpart to StreamingAudioPlayer.stopAndFlush(),
     * which now only ever handles Gemini's own spoken voice. Safe to call
     * with nothing playing. */
    fun stop() {
        session = null
        tts?.stop()
    }

    fun release() {
        stop()
        tts?.shutdown()
        tts = null
    }
}
