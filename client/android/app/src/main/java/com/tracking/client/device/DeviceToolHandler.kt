package com.tracking.client.device

/** Plain, wire-independent call descriptor — device tools are dispatched
 * entirely in-process now (Gemini Live runs on this same device), so there's
 * no longer a proto message for this crossing the wire (see CLAUDE.md's
 * "Client-Orchestrated Live Session" section). */
data class DeviceToolCall(val callId: String, val name: String, val argsJson: String)

interface DeviceToolHandler {
    suspend fun execute(toolCall: DeviceToolCall): String  // JSON result
    val capabilities: List<String>
}
