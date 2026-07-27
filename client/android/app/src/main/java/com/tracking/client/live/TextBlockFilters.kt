package com.tracking.client.live

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/** One OCR-detected word, in pixel coordinates (OCR.space's overlay format). */
data class OcrWord(val text: String, val left: Double, val top: Double, val width: Double, val height: Double)

/** One OCR-detected line — the axis-aligned union of its words' boxes. */
data class OcrLine(
    val text: String,
    val left: Double,
    val top: Double,
    val width: Double,
    val height: Double,
    val words: List<OcrWord>,
)

/**
 * Kotlin port of gt.py's block-level fuzzy dedup + rotation/noise filters —
 * developed and tuned against real OCR output in that standalone harness
 * before being carried over here (see gt.py's own comments for the
 * synthetic/real-text testing behind each threshold below).
 */
object OcrBlockFilters {
    private const val TAG = "OcrBlockFilters"

    // ── Block-level fuzzy dedup ─────────────────────────────────────────
    // MemoryTextUtils.filterNewSentences() assumes near-identical OCR text
    // on repeat passes — a sentence is only recognized as "already read" via
    // exact substring containment. Real OCR noise varies pass to pass
    // (typos, garbled watermark text, a badly-truncated reread of the same
    // page), so that check rarely fires and duplicates pile up in the
    // reading buffer. Instead, each OCR pass is treated as one BLOCK and
    // matched against every block already stored via an ASYMMETRIC,
    // WORD-LEVEL containment score — "how much of the SHORTER text's
    // content shows up as runs of >=2 consecutive matching words somewhere
    // in the longer one". A plain symmetric ratio was tried first and
    // failed: a short, badly-garbled reread of an already-stored page
    // scores far too low purely from the length mismatch, letting it slip
    // past as a spurious "new" block. Character-level containment (no word
    // tokenization) fixed that but then scored unrelated paragraphs too
    // HIGH, since arbitrary short substrings coincidentally recur
    // throughout any text. Requiring >=2-consecutive-WORD matching runs
    // needs a genuine shared phrase, cleanly separating true rereads
    // (~0.75-0.9 in practice) from unrelated paragraphs (~0.05-0.1).
    const val DUP_CONTAINMENT = 0.45
    private val WORD_RE = Regex("[a-z0-9']+")

    private fun wordsOf(text: String): List<String> =
        WORD_RE.findAll(MemoryTextUtils.normalize(text)).map { it.value }.toList()

    /** Ratcliff/Obershelp matching blocks (same core algorithm as Python's
     * difflib.SequenceMatcher with autojunk=False) — returns (aStart,
     * bStart, size) triples for every maximal common contiguous run,
     * order-preserving (not a plain bag-of-words/bigram overlap, which
     * would let common short phrases match across unrelated positions). */
    private fun findLongestMatch(
        a: List<String>, b: List<String>, aLo: Int, aHi: Int, bLo: Int, bHi: Int,
    ): Triple<Int, Int, Int> {
        val b2j = HashMap<String, MutableList<Int>>()
        for (j in bLo until bHi) b2j.getOrPut(b[j]) { mutableListOf() }.add(j)
        var bestI = aLo
        var bestJ = bLo
        var bestSize = 0
        var j2len = HashMap<Int, Int>()
        for (i in aLo until aHi) {
            val newJ2len = HashMap<Int, Int>()
            for (j in b2j[a[i]] ?: emptyList()) {
                if (j < bLo || j >= bHi) continue
                val k = (j2len[j - 1] ?: 0) + 1
                newJ2len[j] = k
                if (k > bestSize) {
                    bestI = i - k + 1
                    bestJ = j - k + 1
                    bestSize = k
                }
            }
            j2len = newJ2len
        }
        return Triple(bestI, bestJ, bestSize)
    }

    private fun matchingBlocks(a: List<String>, b: List<String>): List<Triple<Int, Int, Int>> {
        val blocks = mutableListOf<Triple<Int, Int, Int>>()
        val queue = ArrayDeque<IntArray>()
        queue.addLast(intArrayOf(0, a.size, 0, b.size))
        while (queue.isNotEmpty()) {
            val (aLo, aHi, bLo, bHi) = queue.removeFirst()
            val (i, j, k) = findLongestMatch(a, b, aLo, aHi, bLo, bHi)
            if (k > 0) {
                blocks.add(Triple(i, j, k))
                if (aLo < i && bLo < j) queue.addLast(intArrayOf(aLo, i, bLo, j))
                if (i + k < aHi && j + k < bHi) queue.addLast(intArrayOf(i + k, aHi, j + k, bHi))
            }
        }
        return blocks
    }

    /** Fraction of the SHORTER text's words found as >=minRun-word runs in the longer one. */
    fun blockSimilarity(a: String, b: String, minRun: Int = 2): Double {
        val wa = wordsOf(a)
        val wb = wordsOf(b)
        if (wa.isEmpty() || wb.isEmpty()) return 0.0
        val (short, long) = if (wa.size <= wb.size) wa to wb else wb to wa
        val matched = matchingBlocks(long, short).filter { it.third >= minRun }.sumOf { it.third }
        return matched.toDouble() / short.size
    }

    // ── Raw-first block storage with async LLM correction ──────────────
    // Kotlin port of gt.py's integrate_raw_block()/apply_correction() — a
    // block's `raw` (OCR output) is stored IMMEDIATELY; `corrected`
    // (GeminiCorrectionClient output) starts as a mirror of `raw` and is
    // only later, asynchronously, patched in place once a correction
    // response lands for that specific block (see ToolDispatcher's
    // correction queue). Similarity matching below ALWAYS compares RAW
    // text — an LLM correction/translation pass can phrase the same
    // underlying page slightly differently between rereads, which would
    // corrupt the fuzzy-match signal if compared directly.
    data class ReadingBlock(val id: Int, var raw: String, var corrected: String)

    private val blockIdCounter = java.util.concurrent.atomic.AtomicInteger(1)

    // ── Fuzzy boundary-overlap stitching (sliding-scan reads) ───────────
    // blockSimilarity() above answers "is this basically a REREAD of an
    // existing block" (containment over the whole shorter text, position-
    // agnostic). That's a different question from "does this new capture
    // CONTINUE where the last one left off" — e.g. capture 1 reads "A B C",
    // capture 2 (camera panned slightly) reads "B C D E": these should
    // stitch into "A B C D E", not have one wholesale replace the other.
    // Reuses blockSimilarity()'s own typo-tolerant technique (matchingBlocks
    // over WORDS) but ANCHORS the match to the boundary — near the END of
    // the existing block's words AND near the START of the new capture's
    // words — rather than scoring containment anywhere.
    private val WORD_SPAN_RE = Regex("[A-Za-z0-9']+")

    /** (lowercased_word, startChar, endChar) — offsets into the ORIGINAL
     * (non-normalized) text, so a splice point can cut the real string
     * without losing the new capture's own casing/punctuation. */
    private fun wordSpans(text: String): List<Triple<String, Int, Int>> =
        WORD_SPAN_RE.findAll(text).map { Triple(it.value.lowercase(), it.range.first, it.range.last + 1) }.toList()

    /** Ratcliff/Obershelp ratio (2*matched/(len(a)+len(b))) — same formula
     * Python's difflib.SequenceMatcher.ratio() uses, reusing the same
     * matchingBlocks() machinery already used for word-level matching
     * above, just applied character-by-character here. */
    private fun charMatchRatio(a: String, b: String): Double {
        if (a.isEmpty() || b.isEmpty()) return 0.0
        val la = a.map { it.toString() }
        val lb = b.map { it.toString() }
        val matched = matchingBlocks(la, lb).sumOf { it.third }
        return 2.0 * matched / (la.size + lb.size)
    }

    private fun wordFuzzyEqual(a: String, b: String, minRatio: Double = 0.6): Boolean {
        if (a == b) return true
        if (a.isEmpty() || b.isEmpty()) return false
        return charMatchRatio(a, b) >= minRatio
    }

    /** Fallback for findFuzzyOverlapMerge() when the strict exact-word
     * anchor finds nothing — OCR quality is typically WORST right at a
     * frame's edge (motion blur, a partially-cut-off word), exactly where
     * the boundary anchor needs to be clean; requiring even one exact word
     * match there routinely misses a real continuation. Slides a window of
     * size k (largest first) comparing the LAST k words of `existing`
     * against the FIRST k words of `new`, with each word pair compared via
     * cheap character-ratio fuzzy equality instead of exact match —
     * tolerates typos IN the anchor words themselves, not just gaps around
     * them. Returns the accepted overlap size (word count) or null. */
    private fun findWindowedFuzzyOverlap(
        existingWords: List<String>, newWords: List<String>,
        maxWindow: Int = 12, minWindow: Int = 2, matchFrac: Double = 0.6,
    ): Int? {
        val hi = minOf(existingWords.size, newWords.size, maxWindow)
        for (k in hi downTo minWindow) {
            val tail = existingWords.takeLast(k)
            val head = newWords.take(k)
            val matches = tail.zip(head).count { (a, b) -> wordFuzzyEqual(a, b) }
            if (matches.toDouble() / k >= matchFrac) return k
        }
        return null
    }

    /** Returns the STITCHED text (existingRaw + only the genuinely-new tail
     * of newRaw) if newRaw's start fuzzily continues existingRaw's end,
     * else null. Two passes: (1) a strict exact-word matchingBlocks anchor
     * (cheap, precise), falling back to (2) findWindowedFuzzyOverlap when
     * pass 1 finds nothing — needed since the words directly at a frame's
     * boundary are typically the noisiest in the whole capture. */
    fun findFuzzyOverlapMerge(
        existingRaw: String, newRaw: String, minOverlapWords: Int = 2, boundarySlack: Int = 2,
    ): String? {
        val existingSpans = wordSpans(existingRaw)
        val newSpans = wordSpans(newRaw)
        if (existingSpans.size < minOverlapWords || newSpans.size < minOverlapWords) return null

        val existingWords = existingSpans.map { it.first }
        val newWords = newSpans.map { it.first }

        var best: Triple<Int, Int, Int>? = null
        for (blk in matchingBlocks(existingWords, newWords)) {
            if (blk.third < minOverlapWords) continue
            val endGap = existingWords.size - (blk.first + blk.third)
            val startGap = blk.second
            val current = best
            if (endGap <= boundarySlack && startGap <= boundarySlack && (current == null || blk.third > current.third)) {
                best = blk
            }
        }

        val chosen = best
        val tailWordIdx = if (chosen != null) {
            chosen.second + chosen.third
        } else {
            findWindowedFuzzyOverlap(existingWords, newWords) ?: return null
        }

        if (tailWordIdx >= newSpans.size) return existingRaw.trimEnd() // newRaw was fully covered — nothing new to append
        val newTail = newRaw.substring(newSpans[tailWordIdx].second).trim()
        if (newTail.isEmpty()) return existingRaw.trimEnd()
        return "${existingRaw.trimEnd()} $newTail"
    }

    /** Used when the whole-block containment check below (position-
     * agnostic — blockSimilarity()) judges a capture "the same block, but
     * longer/cleaner" and decides to REPLACE the stored block. Blindly
     * overwriting with newRaw is only safe if newRaw's own OCR pass
     * actually started from the same place existingRaw did — a real
     * observed failure mode is a capture whose overlap with the existing
     * block lands in the MIDDLE of existingRaw (not at its start), meaning
     * newRaw never captured existingRaw's own opening content at all; a
     * plain overwrite then silently discards that opening forever. Finds
     * where in existingRaw the best word-level match against newRaw
     * begins (unrestricted position, unlike findFuzzyOverlapMerge's
     * boundary-anchored search above) and returns existingRaw's own
     * PREFIX up to that point + newRaw in full, preserving whatever
     * existing content precedes the overlap instead of losing it. Returns
     * null if no real anchor is found at all (caller should fall back to
     * a plain overwrite). */
    fun findMidtextRealignMerge(existingRaw: String, newRaw: String, minOverlapWords: Int = 2): String? {
        val existingSpans = wordSpans(existingRaw)
        val newSpans = wordSpans(newRaw)
        if (existingSpans.size < minOverlapWords || newSpans.size < minOverlapWords) return null

        val existingWords = existingSpans.map { it.first }
        val newWords = newSpans.map { it.first }

        var best: Triple<Int, Int, Int>? = null
        for (blk in matchingBlocks(existingWords, newWords)) {
            if (blk.third < minOverlapWords) continue
            val current = best
            if (current == null || blk.third > current.third) best = blk
        }
        val chosen = best ?: return null

        val prefix = existingRaw.substring(0, existingSpans[chosen.first].second).trimEnd()
        return if (prefix.isEmpty()) newRaw.trim() else "$prefix ${newRaw.trim()}"
    }

    /** Folds one OCR pass into [blocks] (mutated in place). Returns
     * (kind, block):
     *   "empty"     - nothing to do, block is null
     *   "stitched"  - newRawText fuzzily CONTINUES the most recent block
     *                 (a sliding-scan/pan read) — that block's raw was
     *                 extended in place with only the genuinely-new tail.
     *                 Only ever checked against the LAST block: stitching
     *                 across an arbitrary earlier block would conflate
     *                 "this continues from where I just was" with "this is
     *                 a reread of something from a while ago" (the
     *                 duplicate/updated check below answers that one).
     *   "duplicate" - already-seen text, nothing changed, block is null
     *   "updated"   - matched an existing block but this capture is
     *                 longer/cleaner — raw (and corrected, reset to mirror
     *                 it) was replaced in place via findMidtextRealignMerge
     *                 (falling back to a plain overwrite if no anchor is
     *                 found at all)
     *   "new"       - a brand-new block was appended
     * "stitched"/"new"/"updated" all carry something a correction pass
     * hasn't already seen — callers should push exactly those three kinds
     * onto the correction queue; a plain "duplicate" changed nothing. */
    fun integrateRawBlock(blocks: MutableList<ReadingBlock>, newRawTextIn: String): Pair<String, ReadingBlock?> {
        val newRawText = newRawTextIn.trim()
        if (newRawText.isEmpty()) return "empty" to null

        if (blocks.isNotEmpty()) {
            val last = blocks.last()
            val stitched = findFuzzyOverlapMerge(last.raw, newRawText)
            if (stitched != null && stitched.length > last.raw.length) {
                Log.d(TAG, "[integrate] STITCHED (append) block #${last.id} — existing=${last.raw} + new=$newRawText -> ${stitched}")
                last.raw = stitched
                last.corrected = stitched
                return "stitched" to last
            }
        }

        var bestIdx = -1
        var bestScore = 0.0
        for (i in blocks.indices) {
            val score = blockSimilarity(blocks[i].raw, newRawText)
            if (score > bestScore) { bestScore = score; bestIdx = i }
        }

        if (bestIdx >= 0 && bestScore >= DUP_CONTAINMENT) {
            val existing = blocks[bestIdx]
            if (newRawText.length > existing.raw.length) {
                val merged = findMidtextRealignMerge(existing.raw, newRawText)
                if (merged != null && merged != newRawText) {
                    Log.d(TAG, "[integrate] REPLACED (updated, prefix preserved) block #${existing.id} — score=$bestScore old=${existing.raw} new_capture=$newRawText -> merged=$merged")
                    existing.raw = merged; existing.corrected = merged
                } else {
                    val fallback = merged ?: newRawText
                    Log.d(TAG, "[integrate] REPLACED (updated, no anchor found) block #${existing.id} — score=$bestScore old=${existing.raw} -> new=$fallback")
                    existing.raw = fallback; existing.corrected = fallback
                }
                return "updated" to existing
            }
            Log.d(TAG, "[integrate] DUPLICATE (ignored, block #${existing.id} already longer/equal) — score=$bestScore incoming=$newRawText")
            return "duplicate" to null
        }

        Log.d(TAG, "[integrate] NEW block (no match, bestScore=$bestScore) — text=$newRawText")
        val block = ReadingBlock(blockIdCounter.getAndIncrement(), newRawText, newRawText)
        blocks.add(block)
        return "new" to block
    }

    /** Finds the block a correction was requested for and inserts the
     * result — but ONLY if that block's raw text hasn't since been
     * replaced by an even newer/longer capture while the correction was
     * in flight (integrateRawBlock() can do that at any time, since OCR
     * keeps running concurrently with correction). A stale correction is
     * silently discarded rather than overwriting a newer capture's own
     * not-yet-arrived correction. Returns true if applied. */
    fun applyCorrection(blocks: List<ReadingBlock>, blockId: Int, rawTextAtRequest: String, correctedText: String): Boolean {
        val block = blocks.find { it.id == blockId } ?: return false
        if (block.raw != rawTextAtRequest) return false
        block.corrected = correctedText
        return true
    }

    // ── Rotation-consistency filter ─────────────────────────────────────
    // OCR.space's overlay gives axis-aligned per-line boxes, no rotation
    // angle directly — so each line's orientation is ESTIMATED from the
    // slope between its first and last word centers (needs >=2 words;
    // single-word lines have no evidence either way and are kept by
    // default). A line whose estimated angle deviates too far from the
    // page's dominant (median) angle is dropped — catches a stray,
    // differently-rotated box (e.g. a diagonal watermark) that the page's
    // real, consistently-oriented body text wouldn't share. This is
    // distinct from OCR.space's own detectOrientation, which only corrects
    // whole-PAGE rotation (0/90/180/270), not one box being crooked
    // relative to the rest of an already-correctly-oriented page.
    const val ROTATION_MAX_DEVIATION_DEG = 15.0

    private fun mod360(x: Double): Double {
        val m = x % 360.0
        return if (m < 0) m + 360.0 else m
    }

    private fun lineAngleDeg(line: OcrLine): Double? {
        if (line.words.size < 2) return null
        val centers = line.words
            .map { (it.left + it.width / 2) to (it.top + it.height / 2) }
            .sortedBy { it.first }
        val (x0, y0) = centers.first()
        val (x1, y1) = centers.last()
        if (abs(x1 - x0) < 1.0 && abs(y1 - y0) < 1.0) return null
        return Math.toDegrees(atan2(y1 - y0, x1 - x0))
    }

    private fun median(values: List<Double>): Double {
        val sorted = values.sorted()
        val n = sorted.size
        return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }

    /** Returns (kept, dropped). Needs >=2 lines with a computable angle to
     * have any evidence of a "dominant" orientation; otherwise keeps everything. */
    fun filterLinesByRotation(
        lines: List<OcrLine>, maxDeviationDeg: Double = ROTATION_MAX_DEVIATION_DEG,
    ): Pair<List<OcrLine>, List<OcrLine>> {
        val angles = lines.map { lineAngleDeg(it) }
        val known = angles.filterNotNull()
        if (known.size < 2) return lines to emptyList()
        val medianAngle = median(known)
        val kept = mutableListOf<OcrLine>()
        val dropped = mutableListOf<OcrLine>()
        for ((line, angle) in lines.zip(angles)) {
            if (angle == null) {
                kept.add(line)
                continue
            }
            val diff = abs(mod360(angle - medianAngle + 180) - 180)
            if (diff <= maxDeviationDeg) kept.add(line) else dropped.add(line)
        }
        return kept to dropped
    }

    // ── Short/small/isolated noise filter ───────────────────────────────
    // Catches page numbers, watermark stamps, and garbled OCR debris that
    // survive the rotation filter untouched (a single-word line has no
    // slope to measure at all). A line is dropped only when it's BOTH
    // ISOLATED (edge-to-edge gap to the nearest other line is large
    // relative to the page's own median line height) AND either short or
    // undersized. Isolation is required, not optional — it's what tells a
    // genuine short in-paragraph word/exclamation (sitting right next to
    // its paragraph, normal font size) apart from a page number or
    // watermark floating alone in the margin; length or font size alone
    // would risk dropping real short in-paragraph text.
    const val MIN_TEXT_CHARS = 8
    const val SMALL_HEIGHT_FRAC = 0.72
    const val ISOLATION_GAP_FACTOR = 1.8
    private val ALNUM_RE = Regex("[a-z0-9]")

    private fun lineGap(a: OcrLine, b: OcrLine): Double {
        val ax1 = a.left; val ay1 = a.top; val ax2 = a.left + a.width; val ay2 = a.top + a.height
        val bx1 = b.left; val by1 = b.top; val bx2 = b.left + b.width; val by2 = b.top + b.height
        val dx = maxOf(ax1 - bx2, bx1 - ax2, 0.0)
        val dy = maxOf(ay1 - by2, by1 - ay2, 0.0)
        return hypot(dx, dy)
    }

    /** Returns (kept, dropped). Needs >=2 lines to establish a dominant body
     * height to compare against; otherwise keeps everything. */
    fun filterLinesByNoise(
        lines: List<OcrLine>,
        minTextChars: Int = MIN_TEXT_CHARS,
        smallHeightFrac: Double = SMALL_HEIGHT_FRAC,
        isolationGapFactor: Double = ISOLATION_GAP_FACTOR,
    ): Pair<List<OcrLine>, List<OcrLine>> {
        if (lines.size < 2) return lines to emptyList()
        val heights = lines.map { it.height }.filter { it > 0 }
        if (heights.isEmpty()) return lines to emptyList()
        val medianHeight = median(heights)
        if (medianHeight <= 0) return lines to emptyList()

        val kept = mutableListOf<OcrLine>()
        val dropped = mutableListOf<OcrLine>()
        for (line in lines) {
            val textLen = ALNUM_RE.findAll(line.text.lowercase()).count()
            val isShort = textLen < minTextChars
            val isSmall = line.height < medianHeight * smallHeightFrac
            val nearestGap = lines.filter { it !== line }.minOfOrNull { lineGap(line, it) } ?: Double.POSITIVE_INFINITY
            val isIsolated = nearestGap > medianHeight * isolationGapFactor
            if (isIsolated && (isShort || isSmall)) dropped.add(line) else kept.add(line)
        }
        return kept to dropped
    }

    // ── Per-line blur filter ────────────────────────────────────────────
    // ToolDispatcher.acquireSharpFrame()'s blur skip/retry measures
    // sharpness for the WHOLE frame as one aggregate number — blind to a
    // frame that's only PARTLY blurry (e.g. a page mid-turn: one side
    // motion-blurred, the other still sharp), since the sharp region's
    // pixels pull the frame-wide average up enough to pass. This instead
    // measures sharpness on each line's OWN cropped patch, independent of
    // how sharp the rest of the frame is — catches exactly the "half the
    // page is legible, half isn't" case the whole-frame check can't. Same
    // variance-of-Laplacian metric as CameraManager.computeSharpness()
    // (meanStdDev, variance = stddev^2), just scoped to one line's crop.
    const val MIN_LINE_SHARPNESS = 60.0

    private fun regionSharpness(bitmap: Bitmap, left: Int, top: Int, width: Int, height: Int): Double {
        val x = left.coerceIn(0, (bitmap.width - 1).coerceAtLeast(0))
        val y = top.coerceIn(0, (bitmap.height - 1).coerceAtLeast(0))
        val w = width.coerceAtMost(bitmap.width - x)
        val h = height.coerceAtMost(bitmap.height - y)
        if (w <= 0 || h <= 0) return Double.MAX_VALUE // degenerate box — nothing to measure, don't penalize it
        val patch = Bitmap.createBitmap(bitmap, x, y, w, h)
        val rgba = Mat()
        Utils.bitmapToMat(patch, rgba)
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
        val laplacian = Mat()
        Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(laplacian, mean, stddev)
        val sd = stddev.toArray().getOrElse(0) { 0.0 }
        return sd * sd
    }

    /** Returns (kept, dropped). 0 disables the check entirely. */
    fun filterLinesByBlur(
        bitmap: Bitmap, lines: List<OcrLine>, minSharpness: Double = MIN_LINE_SHARPNESS,
    ): Pair<List<OcrLine>, List<OcrLine>> {
        if (minSharpness <= 0) return lines to emptyList()
        val kept = mutableListOf<OcrLine>()
        val dropped = mutableListOf<OcrLine>()
        for (line in lines) {
            val sharpness = regionSharpness(bitmap, line.left.toInt(), line.top.toInt(), line.width.toInt(), line.height.toInt())
            (if (sharpness >= minSharpness) kept else dropped).add(line)
        }
        return kept to dropped
    }
}
