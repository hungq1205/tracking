package com.tracking.client.device

import tracking.Tracking

interface DeviceToolHandler {
    suspend fun execute(toolCall: Tracking.DeviceToolCall): String  // JSON result
    val capabilities: List<String>
}
