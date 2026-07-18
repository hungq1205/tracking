package com.tracking.client.live

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.math.sqrt

/** Kotlin port of server/tools/memory_store.py's dedup helpers — sentence-level
 * "already scanned" filtering and suffix/prefix overlap trimming. */
object MemoryTextUtils {
    private val SENT_SPLIT_RE = Regex("(?<=[.!?])\\s+")

    fun normalize(text: String): String = text.trim().lowercase().replace(Regex("\\s+"), " ")

    fun filterNewSentences(blockText: String, existing: String, minChars: Int = 15): String {
        val block = blockText.trim()
        if (block.isEmpty()) return ""
        if (existing.isEmpty()) return block
        val existingNorm = normalize(existing)
        val newSentences = block.split(SENT_SPLIT_RE).mapNotNull { raw ->
            val sent = raw.trim()
            if (sent.isEmpty()) return@mapNotNull null
            val sentNorm = normalize(sent)
            if (sentNorm.length >= minChars && existingNorm.contains(sentNorm)) null else sent
        }
        return newSentences.joinToString(" ")
    }

    fun findOverlapSuffixPrefix(stored: String, new: String, minOverlap: Int = 20): Int {
        val storedNorm = normalize(stored)
        val newNorm = normalize(new)
        if (storedNorm.isEmpty() || newNorm.isEmpty()) return 0
        val maxLen = minOf(storedNorm.length, newNorm.length)
        for (size in maxLen downTo minOverlap) {
            if (storedNorm.takeLast(size) == newNorm.take(size)) return size
        }
        return 0
    }
}

data class MemoryMatch(val label: String, val text: String, val score: Float)

/**
 * On-device memory store — labels/notes + a small embedding index for
 * semantic search, replacing server/tools/memory_store.py + rag_store.py's
 * storage (moved to the phone per the migration plan; PerceptionService.Embed
 * remains the only remote dependency, for the actual vector computation —
 * see CLAUDE.md's "Client-Orchestrated Live Session" section for the
 * local/remote split rationale). Deliberately plain JSON files under the
 * app's private storage, mirroring JsonMemoryStore's own simplicity rather
 * than introducing a database dependency for what's a small amount of data.
 */
class LocalMemoryStore(context: Context) {
    private val dir = File(context.filesDir, "memory").apply { mkdirs() }

    private fun safeName(label: String) = label.trim().replace(Regex("[^\\w\\-:.]+"), "_").ifEmpty { "default" }
    private fun docPath(label: String) = File(dir, "${safeName(label)}.json")
    private fun vecPath(label: String) = File(dir, "${safeName(label)}.vec.json")

    @Synchronized
    fun append(label: String, text: String, source: String = "note"): Pair<String, String> {
        val doc = load(label)
        val fullText = doc.optString("full_text", "")
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return "" to fullText

        val overlap = MemoryTextUtils.findOverlapSuffixPrefix(fullText, trimmed)
        val appended = if (overlap > 0) {
            val newNorm = MemoryTextUtils.normalize(trimmed)
            val ratio = overlap.toFloat() / maxOf(newNorm.length, 1)
            val trimIdx = (trimmed.length * ratio).toInt().coerceIn(0, trimmed.length)
            trimmed.substring(trimIdx).trim()
        } else trimmed
        if (appended.isEmpty()) return "" to fullText

        val entries = doc.optJSONArray("entries") ?: JSONArray()
        entries.put(JSONObject().put("text", appended).put("source", source).put("created_at", System.currentTimeMillis()))
        doc.put("entries", entries)
        val newFullText = if (fullText.isEmpty()) appended else "$fullText\n$appended"
        doc.put("full_text", newFullText)
        doc.put("label", label)
        docPath(label).writeText(doc.toString())
        return appended to newFullText
    }

    @Synchronized
    fun load(label: String): JSONObject {
        val f = docPath(label)
        if (!f.exists()) return JSONObject().put("label", label).put("full_text", "").put("entries", JSONArray())
        return try { JSONObject(f.readText()) } catch (e: Exception) {
            JSONObject().put("label", label).put("full_text", "").put("entries", JSONArray())
        }
    }

    fun getFullText(label: String): String = load(label).optString("full_text", "")

    fun listLabels(): List<String> = dir.listFiles { f -> f.name.endsWith(".json") && !f.name.endsWith(".vec.json") }
        ?.map { it.name.removeSuffix(".json") } ?: emptyList()

    /** Appends one (text, vector) pair to this label's semantic index —
     * called alongside append()/save_memory whenever a new note/description
     * is stored, using the vector PerceptionService.Embed returned for it. */
    @Synchronized
    fun addEmbedding(label: String, text: String, vector: FloatArray) {
        val f = vecPath(label)
        val arr = if (f.exists()) try { JSONArray(f.readText()) } catch (e: Exception) { JSONArray() } else JSONArray()
        arr.put(JSONObject().put("text", text).put("vector", JSONArray(vector.map { it.toDouble() })))
        f.writeText(arr.toString())
    }

    /** Cosine-similarity search across every label's embedding index —
     * pure on-device math, no model call (the query itself must already be
     * embedded via PerceptionService.Embed by the caller). */
    @Synchronized
    fun queryGlobal(queryVector: FloatArray, topK: Int = 5): List<MemoryMatch> {
        val results = mutableListOf<MemoryMatch>()
        dir.listFiles { f -> f.name.endsWith(".vec.json") }?.forEach { f ->
            val label = f.name.removeSuffix(".vec.json")
            val arr = try { JSONArray(f.readText()) } catch (e: Exception) { return@forEach }
            for (i in 0 until arr.length()) {
                val entry = arr.getJSONObject(i)
                val vecArr = entry.getJSONArray("vector")
                val vec = FloatArray(vecArr.length()) { vecArr.getDouble(it).toFloat() }
                results.add(MemoryMatch(label, entry.optString("text", ""), cosineSim(queryVector, vec)))
            }
        }
        return results.sortedByDescending { it.score }.take(topK)
    }

    private fun cosineSim(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size || a.isEmpty()) return 0f
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        if (na == 0f || nb == 0f) return 0f
        return dot / (sqrt(na) * sqrt(nb))
    }
}
