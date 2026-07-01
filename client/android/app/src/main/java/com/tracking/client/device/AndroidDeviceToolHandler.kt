package com.tracking.client.device

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.AlarmClock
import android.provider.CalendarContract
import android.provider.ContactsContract
import android.util.Log
import androidx.core.content.ContextCompat
import org.json.JSONObject
import tracking.Tracking
import java.util.Calendar

class AndroidDeviceToolHandler(private val context: Context) : DeviceToolHandler {

    override val capabilities = listOf("make_phone_call", "set_alarm", "create_calendar_event", "search_contacts", "play_video", "stop_music")

    override suspend fun execute(toolCall: Tracking.DeviceToolCall): String {
        val args = try { JSONObject(toolCall.argsJson) } catch (e: Exception) { JSONObject() }
        Log.d(TAG, "Executing device tool: ${toolCall.name} args=$args")
        return try {
            when (toolCall.name) {
                "make_phone_call"       -> makePhoneCall(args)
                "set_alarm"             -> setAlarm(args)
                "create_calendar_event" -> createCalendarEvent(args)
                "search_contacts"       -> searchContacts(args)
                "play_video"            -> playVideo(args)
                "stop_music"            -> stopMusic()
                else -> """{"error":"Unknown device tool: ${toolCall.name}"}"""
            }
        } catch (e: Exception) {
            Log.e(TAG, "Device tool ${toolCall.name} failed", e)
            """{"error":"${e.message?.replace("\"", "'")}"}"""
        }
    }

    private fun makePhoneCall(args: JSONObject): String {
        val input = args.optString("contact_name_or_number", "")
        if (input.isBlank()) return """{"error":"No contact name or number provided"}"""
        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.CALL_PHONE)
                != PackageManager.PERMISSION_GRANTED) {
            return """{"error":"Phone call permission not granted. Ask the user to allow it in app settings."}"""
        }
        // Resolve name → number if the input doesn't look like a phone number
        val number = if (input.matches(Regex("[+\\d\\s\\-().]+"))) {
            input
        } else {
            lookupContactNumber(input)
                ?: return """{"error":"Contact '${input.replace("\"", "'")}' not found in contacts"}"""
        }
        val intent = Intent(Intent.ACTION_CALL, Uri.parse("tel:${Uri.encode(number)}"))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
        return """{"status":"dialing","number":"$number"}"""
    }

    private fun searchContacts(args: JSONObject): String {
        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.READ_CONTACTS)
                != PackageManager.PERMISSION_GRANTED) {
            return """{"error":"Contacts permission not granted"}"""
        }
        val query = args.optString("query", "")
        val selection = if (query.isBlank()) null else
            "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
        val selectionArgs = if (query.isBlank()) null else arrayOf("%$query%")

        val cursor = context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER,
            ),
            selection,
            selectionArgs,
            "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} ASC",
        )
        val results = mutableListOf<String>()
        cursor?.use {
            while (it.moveToNext() && results.size < 20) {
                val name = it.getString(0) ?: continue
                if (!results.contains(name)) results.add(name)
            }
        }
        val namesJson = results.joinToString(",") { "\"${it.replace("\"", "'")}\"" }
        return """{"contacts":[$namesJson],"count":${results.size}}"""
    }

    private fun lookupContactNumber(name: String): String? {
        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.READ_CONTACTS)
                != PackageManager.PERMISSION_GRANTED) return null
        val cursor = context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            arrayOf(ContactsContract.CommonDataKinds.Phone.NUMBER),
            "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?",
            arrayOf("%$name%"),
            null,
        )
        return cursor?.use { if (it.moveToFirst()) it.getString(0) else null }
    }

    private fun setAlarm(args: JSONObject): String {
        // Gemini sends time in HH:mm 24-hour format
        val timeStr = args.optString("time", "")
        val label   = args.optString("label", "Alarm")
        val parts   = timeStr.split(":")
        val hour    = parts.getOrNull(0)?.trim()?.toIntOrNull()
        val minute  = parts.getOrNull(1)?.trim()?.take(2)?.toIntOrNull() ?: 0
        val intent  = Intent(AlarmClock.ACTION_SET_ALARM).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            putExtra(AlarmClock.EXTRA_MESSAGE, label)
            putExtra(AlarmClock.EXTRA_SKIP_UI, true)
            if (hour != null) {
                putExtra(AlarmClock.EXTRA_HOUR, hour)
                putExtra(AlarmClock.EXTRA_MINUTES, minute)
            }
        }
        context.startActivity(intent)
        return """{"status":"alarm_set","time":"$timeStr","label":"$label"}"""
    }

    private fun createCalendarEvent(args: JSONObject): String {
        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.WRITE_CALENDAR)
                != PackageManager.PERMISSION_GRANTED) {
            return """{"error":"Calendar write permission not granted. Ask the user to allow it in app settings."}"""
        }
        val title       = args.optString("title", "Event")
        val startTime   = args.optString("start_time", "")
        val endTime     = args.optString("end_time", "")
        val description = args.optString("description", "")

        val startMs = parseTimeMillis(startTime) ?: System.currentTimeMillis()
        val endMs   = parseTimeMillis(endTime) ?: (startMs + 3600_000L)

        // Find primary calendar
        val calId = primaryCalendarId() ?: return """{"error":"No calendar account found on device"}"""

        val values = ContentValues().apply {
            put(CalendarContract.Events.CALENDAR_ID, calId)
            put(CalendarContract.Events.TITLE, title)
            put(CalendarContract.Events.DTSTART, startMs)
            put(CalendarContract.Events.DTEND, endMs)
            put(CalendarContract.Events.EVENT_TIMEZONE, "Asia/Ho_Chi_Minh")
            if (description.isNotBlank()) put(CalendarContract.Events.DESCRIPTION, description)
        }
        val uri = context.contentResolver.insert(CalendarContract.Events.CONTENT_URI, values)
        return if (uri != null) """{"status":"event_created","title":"$title"}"""
               else """{"error":"Failed to create calendar event"}"""
    }

    private fun primaryCalendarId(): Long? {
        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.READ_CALENDAR)
                != PackageManager.PERMISSION_GRANTED) return null
        val cursor = context.contentResolver.query(
            CalendarContract.Calendars.CONTENT_URI,
            arrayOf(CalendarContract.Calendars._ID),
            "${CalendarContract.Calendars.IS_PRIMARY} = 1",
            null,
            null,
        ) ?: return null
        return cursor.use { if (it.moveToFirst()) it.getLong(0) else null }
    }

    // Parses "HH:mm", "H:mm am/pm", or epoch millis string; returns null if unparseable.
    private fun parseTimeMillis(timeStr: String): Long? {
        if (timeStr.isBlank()) return null
        timeStr.toLongOrNull()?.let { return it }

        return try {
            val cal = Calendar.getInstance()
            // Try to extract hour and minute from strings like "14:30", "2:30 pm", "7am"
            val ampm  = timeStr.lowercase()
            val isPm  = ampm.contains("pm")
            val clean = ampm.replace(Regex("[^0-9:]"), " ").trim()
            val parts = clean.split(Regex("\\s+|:")).filter { it.isNotBlank() }
            val hour  = parts.getOrNull(0)?.toIntOrNull() ?: return null
            val min   = parts.getOrNull(1)?.toIntOrNull() ?: 0
            val h24   = when {
                isPm && hour < 12 -> hour + 12
                !isPm && hour == 12 -> 0
                else -> hour
            }
            cal.set(Calendar.HOUR_OF_DAY, h24)
            cal.set(Calendar.MINUTE, min)
            cal.set(Calendar.SECOND, 0)
            cal.set(Calendar.MILLISECOND, 0)
            // If the time is in the past today, assume tomorrow
            if (cal.timeInMillis < System.currentTimeMillis()) {
                cal.add(Calendar.DAY_OF_MONTH, 1)
            }
            cal.timeInMillis
        } catch (e: Exception) {
            null
        }
    }

    private fun stopMusic(): String {
        context.stopService(Intent(context, PlaybackService::class.java))
        return """{"status":"stopped"}"""
    }

    private fun playVideo(args: JSONObject): String {
        val videoId   = args.optString("video_id", "")
        val streamUrl = args.optString("stream_url", "")
        val title     = args.optString("title", "")
        val channel   = args.optString("channel", "")
        if (streamUrl.isBlank()) return """{"error":"No stream URL provided"}"""
        val intent = Intent(context, PlaybackService::class.java).apply {
            putExtra("stream_url", streamUrl)
            putExtra("video_id", videoId)
            putExtra("title", title)
            putExtra("channel", channel)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
            context.startForegroundService(intent)
        else
            context.startService(intent)
        return """{"status":"playing","video_id":"$videoId"}"""
    }

    companion object {
        private const val TAG = "AndroidDeviceToolHandler"
    }
}
