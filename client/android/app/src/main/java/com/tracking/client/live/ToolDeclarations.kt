package com.tracking.client.live

import org.json.JSONArray
import org.json.JSONObject

/**
 * Kotlin port of server/live_tools/tool_declarations.py's SYSTEM_PROMPT and
 * TOOL_DECLARATIONS — now sent directly to Gemini Live from the client (see
 * CLAUDE.md's "Client-Orchestrated Live Session" section). GUIDING's
 * destination semantics changed: named zones are dropped, so `destination`
 * now resolves to a landmark name or an "x,z" coordinate pair, not a zone
 * label. search_youtube/get_video_info are declared but not yet
 * implemented on-device (ToolDispatcher returns a clear "not implemented"
 * error) — see CLAUDE.md for that known gap.
 */
object ToolDeclarations {

    const val SYSTEM_PROMPT = """You are a real-time voice assistant for a visually impaired person.
You receive audio input from the user and video frames from their phone camera.

CORE RULES:
- Keep all spoken responses SHORT. Audio UX demands brevity.
- [SYSTEM] messages are device events — respond to them IMMEDIATELY in audio.
- Call `get_current_time()` whenever the user asks what time or date it is.

VISION:
- Call `get_latest_frame()` BEFORE answering any question about what the user sees, what's nearby,
  what an object looks like, or any other visually-grounded question.
- Call `start_vision_stream()` when the user says "watch me", "look at this", "keep watching",
  "observe this", or implies they want sustained visual attention for 10-15 seconds. Do NOT use
  it for one-off visual questions — use `get_latest_frame()` instead.
- `stop_vision_stream()` when the user says they're done or after you finish commenting.

MEMORY:
- Call `query_memory(question)` proactively when the user mentions any named object, personal item.

READING MODE:
- `enter_reading_mode()` when the user wants to read a document, label, sign, or screen.
- `scan_current_view()` to silently capture OCR text without reading it aloud.
- `read_aloud(scope)`: scope="new" scans then speaks only newly captured text; scope="all" speaks
  the entire scanned buffer so far. The audio is played by a dedicated local TTS engine, NOT by
  you — do not narrate or repeat the text yourself after calling this.
- `get_reading_section(query)` to answer questions about already-scanned content.
- `exit_reading_mode()` when the user is done reading.

TRACKING:
- Only call `get_object_from_memory(query)` FIRST for a POSSESSIVE reference to something the user
  owns or previously saved ("my keys", "my wallet"). If found, call `start_tracking(target=label,
  description=description)`.
- For any other object, call `start_tracking(target=<what user said>)` directly, no memory lookup.
- While tracking, once both the target object and the user's hand become visible together, you
  will receive periodic [SYSTEM] guidance messages roughly every 5 seconds — respond immediately
  with brief directional guidance for moving the hand toward the object.

GUIDING (LIVE MAP-BASED NAVIGATION):
- `start_guiding(destination)` when the user wants directions to a place. `destination` is a
  landmark/functional-object name found in the environment (e.g. "the couch", "the fridge") — not
  a fixed zone label; there are no named rooms/zones any more, only landmarks the system has
  actually seen. Waypoint/arrival progress arrives as [SYSTEM] messages — respond immediately,
  concisely.
- Mapping runs live, automatically, for as long as guiding is active — there is no separate scan
  step.

WALKING (FREE-WALK — ambient audio-only obstacle avoidance, no destination):
- `start_walking()` plays a continuous directional audio beacon steering the user toward the most
  open space ahead, computed live from the occupancy map — this is a background audio cue, not
  something you narrate or react to; there are no [SYSTEM] messages for it.
- `stop_walking()` when the user is done. (`stop_guiding()` stops guiding mode the same way.)

SCANNING (BUILD UP THE MAP ON DEMAND):
- `start_scan()` when the user asks you to scan/map the room, or to "look around", with no
  navigation destination in mind. Immediately after calling it, ask the user (briefly) to pan the
  camera slowly around the space so the system can build up the occupancy map and tag landmarks.
- `stop_scan()` when the user says they're done scanning. This automatically starts walking mode's
  ambient audio beacon right after — do NOT also call `start_walking()` yourself.

DEVICE TOOLS:
- Before calling `set_alarm`, if the user did not give a label/name for the alarm, ask for one first.
- Before calling `create_calendar_event`, if the user did not give a title, ask for one first.
- When calling `set_alarm`, pass `time` in HH:mm 24-hour format (e.g. "07:00", "14:30").
- For `make_phone_call`, pass the contact's name exactly as the user said it.
- If `make_phone_call` returns "not found", call `search_contacts(query)` and ask which one they mean.
"""

    fun buildDeclarations(): JSONArray = JSONArray().apply {
        // ── Time ─────────────────────────────────────────────────────────
        put(decl("get_current_time", "Get the current local time and date for the user."))

        // ── Scene / Vision ───────────────────────────────────────────────
        put(decl("get_latest_frame", "Capture the current camera view and show it to you. Call before answering any visually-grounded question."))
        put(decl(
            "start_vision_stream",
            "Start sending live camera frames at 1 fps for up to 15 seconds. Use when the user wants sustained visual attention.",
            params { prop("reason", "string", "Brief reason for starting the stream") }
        ))
        put(decl("stop_vision_stream", "Stop the live camera stream immediately."))
        put(decl(
            "run_detection",
            "Run object detection on the current camera frame. Returns bounding box and confidence score.",
            params(required = listOf("object_description")) { prop("object_description", "string", "Natural language description of the object to find") }
        ))
        put(decl("check_obstacle", "Check whether an obstacle is directly ahead using depth estimation."))

        // ── Reading ──────────────────────────────────────────────────────
        put(decl(
            "enter_reading_mode",
            "Enter reading mode. Enables passive OCR accumulation from the camera.",
            params { prop("label", "string", "Optional memory label to associate scanned text with") }
        ))
        put(decl("scan_current_view", "Run OCR on the current camera frame and capture any visible text."))
        put(decl(
            "get_reading_section",
            "Retrieve the most relevant passage from previously scanned reading material.",
            params(required = listOf("query")) { prop("query", "string", "The question or topic to find in the scanned text") }
        ))
        put(decl(
            "read_aloud",
            "Read scanned document text aloud through a dedicated local TTS engine, not your own voice.",
            params { propEnum("scope", listOf("new", "all"), "'new' to scan and read newly captured text, 'all' to re-read everything scanned so far") }
        ))
        put(decl("flip_reading_direction", "Toggle reading direction between left-to-right and right-to-left."))
        put(decl("exit_reading_mode", "Exit reading mode and clear the scanned text buffer."))

        // ── Tracking ─────────────────────────────────────────────────────
        put(decl(
            "start_tracking",
            "Start tracking a specific object in the camera view. Pass a saved object's label/description when found via memory.",
            params(required = listOf("target")) {
                prop("target", "string", "Name or label of the object to track")
                prop("description", "string", "One-sentence appearance description, from get_object_from_memory result")
            }
        ))
        put(decl("stop_tracking", "Stop tracking the current object."))
        put(decl(
            "get_object_from_memory",
            "Search saved object memory for items matching the query. Only for the user's OWN previously-saved items.",
            params(required = listOf("query")) { prop("query", "string", "Description of the object to look up in memory") }
        ))

        // ── Memory ───────────────────────────────────────────────────────
        put(decl(
            "query_memory",
            "Semantically search all saved memories for information relevant to the question.",
            params(required = listOf("question")) { prop("question", "string", "The question or topic to search memory for") }
        ))
        put(decl(
            "save_memory",
            "Save a text note to a named memory label.",
            params(required = listOf("label", "note")) {
                prop("label", "string", "Memory label")
                prop("note", "string", "The text to save")
            }
        ))
        put(decl(
            "remember_object",
            "Save the current object in view to memory, generating a label and appearance description.",
            params(required = listOf("label", "description")) {
                prop("label", "string", "Short name for the object")
                prop("description", "string", "One-sentence appearance description, used to detect/retrieve the object")
            }
        ))
        put(decl("list_memory_labels", "List all named memories that have been saved."))

        // ── Guiding ──────────────────────────────────────────────────────
        put(decl(
            "start_guiding",
            "Start guiding the user to a destination landmark, using the live occupancy map.",
            params(required = listOf("destination")) { prop("destination", "string", "Name of the landmark/functional object to navigate to") }
        ))
        put(decl("stop_guiding", "Stop guiding/navigation and disable obstacle monitoring."))
        put(decl("get_current_location", "Determine the user's current location from the live map pose."))

        // ── Walking ──────────────────────────────────────────────────────
        put(decl("start_walking", "Start free-walk mode: continuous ambient audio beacon steering toward the most open direction ahead, using the live occupancy map. No destination required."))
        put(decl("stop_walking", "Stop walking mode and its audio beacon."))

        // ── Scanning ─────────────────────────────────────────────────────
        put(decl(
            "start_scan",
            "Start a mapping/scanning pass — asks the user to pan around so the system can build " +
                "up the occupancy map and tag landmarks. No destination required.",
        ))
        put(decl("stop_scan", "Stop the current scanning pass. Automatically starts walking mode's ambient audio beacon right after — no separate start_walking() call needed."))

        // ── Music / Video (declared; not yet implemented on-device — see CLAUDE.md) ──
        put(decl(
            "search_youtube",
            "Search YouTube for music or videos matching a query.",
            params(required = listOf("query")) { prop("query", "string", "Search terms") }
        ))
        put(decl(
            "get_video_info",
            "Fetch full metadata for specific YouTube video IDs.",
            params(required = listOf("video_ids")) { propArray("video_ids", "string", "List of YouTube video IDs") }
        ))

        // ── Device tools (executed locally, no server round trip) ────────
        put(decl(
            "make_phone_call",
            "Place a phone call to a contact by name or number.",
            params(required = listOf("contact_name_or_number")) { prop("contact_name_or_number", "string", "Contact name or phone number") }
        ))
        put(decl(
            "search_contacts",
            "Search the device's contacts.",
            params { prop("query", "string", "Partial name to search for") }
        ))
        put(decl(
            "set_alarm",
            "Set a device alarm.",
            params(required = listOf("time")) {
                prop("time", "string", "HH:mm 24-hour format")
                prop("label", "string", "Alarm label")
            }
        ))
        put(decl(
            "create_calendar_event",
            "Create a calendar event on the device.",
            params(required = listOf("title", "start_time")) {
                prop("title", "string", "Event title")
                prop("start_time", "string", "Start time")
                prop("end_time", "string", "End time")
                prop("description", "string", "Event description")
            }
        ))
        put(decl(
            "play_video",
            "Play a resolved video/audio stream URL on the device.",
            params(required = listOf("stream_url")) {
                prop("stream_url", "string", "Resolved stream URL")
                prop("video_id", "string", "Video ID")
                prop("title", "string", "Title")
                prop("channel", "string", "Channel name")
            }
        ))
        put(decl("stop_music", "Stop any currently-playing music/video."))
    }

    // ── declaration-builder helpers ─────────────────────────────────────

    private fun decl(name: String, description: String, parameters: JSONObject? = null): JSONObject =
        JSONObject().put("name", name).put("description", description).apply {
            if (parameters != null) put("parameters", parameters)
        }

    private class ParamsBuilder(private val required: List<String>) {
        val properties = JSONObject()
        fun prop(name: String, type: String, description: String) {
            properties.put(name, JSONObject().put("type", type).put("description", description))
        }
        fun propEnum(name: String, values: List<String>, description: String) {
            properties.put(name, JSONObject().put("type", "string").put("enum", JSONArray(values)).put("description", description))
        }
        fun propArray(name: String, itemType: String, description: String) {
            properties.put(
                name,
                JSONObject().put("type", "array").put("items", JSONObject().put("type", itemType)).put("description", description)
            )
        }
        fun build(): JSONObject = JSONObject().put("type", "object").put("properties", properties).apply {
            if (required.isNotEmpty()) put("required", JSONArray(required))
        }
    }

    private fun params(required: List<String> = emptyList(), block: ParamsBuilder.() -> Unit): JSONObject {
        val b = ParamsBuilder(required)
        b.block()
        return b.build()
    }
}
