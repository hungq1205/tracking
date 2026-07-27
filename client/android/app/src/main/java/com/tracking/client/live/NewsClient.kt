package com.tracking.client.live

import android.util.Log
import android.util.Xml
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import org.xmlpull.v1.XmlPullParser
import java.io.StringReader
import java.util.concurrent.TimeUnit

private const val TAG = "NewsClient"

/** One headline from [NewsClient.getTopHeadlines]/[NewsClient.search]. */
data class NewsArticle(val title: String, val source: String, val link: String, val pubDate: String)

/**
 * Direct 3rd-party call to Google News' public RSS feed — no API key, no
 * server proxy, same pattern as OcrClient.kt/YouTubeSearchClient.kt.
 * Hardcoded to Vietnam/Vietnamese (hl=vi&gl=VN&ceid=VN:vi) per this
 * feature's scope — no country/language params are exposed to Gemini at
 * all, so there's nothing for the model to get wrong. Uses Android's
 * built-in XmlPullParser (android.util.Xml) rather than adding a new XML
 * dependency for a handful of fields.
 */
class NewsClient {
    private val client = OkHttpClient.Builder().callTimeout(15, TimeUnit.SECONDS).build()

    suspend fun getTopHeadlines(maxResults: Int = 5): List<NewsArticle> = withContext(Dispatchers.IO) {
        val url = "https://news.google.com/rss".toHttpUrl().newBuilder()
            .addQueryParameter("hl", "vi")
            .addQueryParameter("gl", "VN")
            .addQueryParameter("ceid", "VN:vi")
            .build()
        fetchAndParse(url.toString(), maxResults)
    }

    suspend fun search(query: String, maxResults: Int = 5): List<NewsArticle> = withContext(Dispatchers.IO) {
        if (query.isBlank()) return@withContext emptyList()
        val url = "https://news.google.com/rss/search".toHttpUrl().newBuilder()
            .addQueryParameter("q", query)
            .addQueryParameter("hl", "vi")
            .addQueryParameter("gl", "VN")
            .addQueryParameter("ceid", "VN:vi")
            .build()
        fetchAndParse(url.toString(), maxResults)
    }

    private fun fetchAndParse(url: String, maxResults: Int): List<NewsArticle> = try {
        client.newCall(Request.Builder().url(url).build()).execute().use { resp ->
            val bodyStr = resp.body?.string() ?: ""
            if (!resp.isSuccessful) {
                Log.e(TAG, "fetch '$url' HTTP ${resp.code}")
                emptyList()
            } else {
                parseRss(bodyStr, maxResults).also { Log.d(TAG, "fetch '$url' -> ${it.size} article(s)") }
            }
        }
    } catch (e: Exception) {
        Log.e(TAG, "fetch '$url' failed", e)
        emptyList()
    }

    /** Minimal RSS <item> parser — only pulls title/link/source/pubDate,
     * the only fields get_top_news/search_news need. Standard XmlPullParser
     * usage: nextText() reads a START_TAG's text and leaves the parser at
     * its matching END_TAG, so the outer loop's next() call always advances
     * cleanly past it — no double-advance. */
    private fun parseRss(xml: String, maxResults: Int): List<NewsArticle> {
        if (xml.isBlank()) return emptyList()
        val parser = Xml.newPullParser()
        parser.setInput(StringReader(xml))
        val results = mutableListOf<NewsArticle>()
        var inItem = false
        var title = ""
        var link = ""
        var source = ""
        var pubDate = ""
        var eventType = parser.eventType
        while (eventType != XmlPullParser.END_DOCUMENT && results.size < maxResults) {
            when (eventType) {
                XmlPullParser.START_TAG -> when (parser.name) {
                    "item" -> {
                        inItem = true
                        title = ""; link = ""; source = ""; pubDate = ""
                    }
                    "title" -> if (inItem) title = parser.nextText()
                    "link" -> if (inItem) link = parser.nextText()
                    "pubDate" -> if (inItem) pubDate = parser.nextText()
                    "source" -> if (inItem) source = parser.nextText()
                }
                XmlPullParser.END_TAG -> if (parser.name == "item" && inItem) {
                    inItem = false
                    if (title.isNotBlank()) results.add(NewsArticle(title, source, link, pubDate))
                }
            }
            eventType = parser.next()
        }
        return results
    }
}
