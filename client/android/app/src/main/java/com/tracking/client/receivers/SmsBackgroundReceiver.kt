package com.tracking.client.receivers

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import androidx.core.content.ContextCompat
import com.tracking.client.live.LiveAssistantService

/**
 * Notifies the running [LiveAssistantService] of an incoming SMS so Gemini
 * can read it out immediately (see ToolDeclarations.kt's DEVICE TOOLS
 * section). Multiple PDUs in one intent (a long concatenated SMS) are
 * joined into a single message body before forwarding. Same "transient
 * receiver, always starts (not binds) the Service, no-ops without a live
 * session" pattern as CallBackgroundReceiver.
 */
class SmsBackgroundReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent) ?: return
        if (messages.isEmpty()) return
        val sender = messages.first().displayOriginatingAddress ?: "unknown"
        val body = messages.joinToString("") { it.messageBody ?: "" }

        val serviceIntent = Intent(context, LiveAssistantService::class.java).apply {
            action = LiveAssistantService.ACTION_SMS_RECEIVED
            putExtra(LiveAssistantService.EXTRA_SMS_SENDER, sender)
            putExtra(LiveAssistantService.EXTRA_SMS_BODY, body)
        }
        ContextCompat.startForegroundService(context, serviceIntent)
    }
}
