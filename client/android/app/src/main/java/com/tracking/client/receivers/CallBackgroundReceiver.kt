package com.tracking.client.receivers

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.ContactsContract
import android.telephony.TelephonyManager
import androidx.core.content.ContextCompat
import com.tracking.client.live.LiveAssistantService

/**
 * Notifies the running [LiveAssistantService] of an incoming call so Gemini
 * can tell the user who's calling (see ToolDeclarations.kt's DEVICE TOOLS
 * section) and, on request, answer it via the `answer_phone_call` tool
 * (AndroidDeviceToolHandler.answerPhoneCall — ANSWER_PHONE_CALLS +
 * TelecomManager.acceptRingingCall(), proceeding on the documented API for
 * a non-default-dialer app; see CLAUDE.md for the accepted risk that this
 * may turn out to require the dialer role on some devices).
 *
 * Transient — no reference to a running Service instance — so it always
 * starts (not binds) LiveAssistantService with the event as Intent extras;
 * the Service no-ops if no Gemini Live session is currently connected
 * (liveClient == null).
 */
class CallBackgroundReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) return
        val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE)
        if (state != TelephonyManager.EXTRA_STATE_RINGING) return

        @Suppress("DEPRECATION")
        val number = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)
        val label = resolveCallerLabel(context, number)

        val serviceIntent = Intent(context, LiveAssistantService::class.java).apply {
            action = LiveAssistantService.ACTION_INCOMING_CALL
            putExtra(LiveAssistantService.EXTRA_CALLER_LABEL, label)
        }
        ContextCompat.startForegroundService(context, serviceIntent)
    }

    private fun resolveCallerLabel(context: Context, number: String?): String {
        if (number.isNullOrBlank()) return "an unknown number"
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
                != PackageManager.PERMISSION_GRANTED) {
            return number
        }
        val cursor = context.contentResolver.query(
            Uri.withAppendedPath(ContactsContract.PhoneLookup.CONTENT_FILTER_URI, Uri.encode(number)),
            arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME),
            null, null, null,
        )
        val name = cursor?.use { if (it.moveToFirst()) it.getString(0) else null }
        return name ?: number
    }
}
