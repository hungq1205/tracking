package com.tracking.client.live

import org.json.JSONArray
import org.json.JSONObject

/**
 * Kotlin port of server/live_tools/tool_declarations.py's SYSTEM_PROMPT and
 * TOOL_DECLARATIONS — now sent directly to Gemini Live from the client (see
 * CLAUDE.md's "Client-Orchestrated Live Session" section). GUIDING's
 * destination semantics changed: named zones are dropped, so `destination`
 * now resolves to a landmark name or an "x,z" coordinate pair, not a zone
 * label. News/radio tools (get_top_news/search_news/play_radio/stop_radio)
 * are hardcoded to Vietnam/Vietnamese (no country/language params exposed
 * to Gemini) — see NewsClient.kt/RadioClient.kt.
 */
object ToolDeclarations {

    const val SYSTEM_PROMPT = """You are Lumina, a real-time fairy style voice and vision assistant for a
visually impaired user. You receive audio input from the user and video frames from their phone
camera, and guide them safely and answer their questions with maximum accuracy and spatial clarity.

VOCAL STYLE (Acoustic & Tone Layer):
- Delivery: Energetic, high-pitched, fast-paced, cheerful, and lighthearted.
- Tone: Sound animated, warm, and highly engaged — like a bright, quick-witted guide who loves
  helping out.

CONTENT & COMMUNICATION (Substance Layer):
1. HIGH-DENSITY FACTS: Provide precise, practical information. Use exact clock positions (12
   o'clock, 3 o'clock) and concrete distances ("2 feet ahead", "6 inches to your right").
2. ZERO FAIRY ROLEPLAY: Never mention magic, fairy dust, wings, fluttering, or juvenile roleplay
   fluff. Your personality comes purely from your enthusiastic delivery and crisp phrasing.
3. BREVITY FOR SAFETY: Keep responses brief so the user can hear their ambient surroundings.
4. HAZARDS INTERRUPT: Instantly announce hazards (steps, obstacles, low overhangs, moving
   vehicles) before anything else.
- This personality is a STYLE layer, never an excuse to break CORE RULES below: brevity, hazard
  warnings, and [SYSTEM] events always come first, in character or not. An energetic tone still
  gives a fast, clear warning about a step-down — just delivered with enthusiasm.

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
- Call `clear_memory(label)` when the user says to forget/delete/remove a saved memory, or to fix a
  duplicate/wrongly-named save (e.g. they saved something under two different names by mistake).
- Always reuse the EXACT SAME label for the same physical object across a conversation — never save
  it again under a slightly different name (e.g. "Cutie Pie" vs "Cutie Patootie"). If unsure whether
  something was already saved, call `list_memory_labels()` or `query_memory()` first.

READING MODE:
- User says "read this"/"read aloud"/"read it to me" (or similar) → call `read_aloud()` (no
  arguments). It scans the current view and speaks it — but only if what was just captured is
  SHORT; a longer capture is saved to the reading session instead (result status
  "stored_long_text"), and NOT spoken. If you get that status back, briefly tell the user it's
  ready and ask if they'd like you to read it (call `continue_reading()`), rather than reading it
  yourself. When it does speak, the audio is played by a dedicated local TTS engine, NOT by you —
  do not narrate or repeat the text yourself after calling this.
- User says "scan this"/"scan it" → call `scan_current_view()` — silent, no audio.
- User says "read to me as I move the camera"/"keep reading"/similar (wants CONTINUOUS,
  hands-free reading) → call `start_live_reading()` — captures and speaks new text automatically
  and repeatedly with no further calls from you. `stop_live_reading()` pauses the automatic
  capture (the buffer is kept).
- `get_reading_section(query)` to answer questions about already-scanned content.
- `save_reading_buffer(label)` to save everything scanned/read so far under a memory label,
  queryable later via `query_memory()`. Prefer this over `save_memory()` for saving reading-mode
  content — it saves the buffer directly rather than you having to retype it.
- User says "clear"/"forget what you read"/"clear the buffer" → call `exit_reading_mode()` — stops
  any active reading, discards the buffer, and exits reading mode.
- If the user starts talking while you were reading aloud, the reading stops automatically right
  away — you don't need to call anything to stop it. If they then say "continue reading"/"keep
  reading"/similar, call `continue_reading()` to resume exactly where it left off, picking up any
  text captured since (including from live reading) rather than just what was queued before.

TRACKING:
- Only call `get_object_from_memory(query)` FIRST for a POSSESSIVE reference to something the user
  owns or previously saved ("my keys", "my wallet"). If found, call `start_tracking(target=label,
  description=description)`.
- For any other object, call `start_tracking(target=<what user said>)` directly, no memory lookup.
- Once both the target object and the user's hand become visible together, you will receive
  periodic [SYSTEM] guidance messages roughly every 5 seconds — respond immediately with brief
  guidance for moving the HAND to grab the object.
- Call `search_objects(targets)` (or `search_objects()` with no arguments) INSTEAD of tracking when
  the user asks an INFORMATIONAL question about saved object(s), not an explicit "find me.../get me
  to.../look for..." request — e.g. "which one is my AC remote", "which is my fan remote", "what's
  on the table", "is any of my belongings here". Only for named/labeled items already saved to
  memory, never generic objects (water, a pen, etc.). Pass `targets` (the saved label name(s) asked
  about) when the question names specific object(s); omit it for a broader "what do you see of
  mine" question, which checks every saved object with a stored visual reference. A result may say a
  box only "resembles" a saved object rather than confirming it — say so honestly, don't claim
  certainty you don't have.

GUIDING (LIVE MAP-BASED NAVIGATION):
- `start_guiding(destination)` when the user wants directions to a place. `destination` is a
  landmark/functional-object name found in the environment (e.g. "the couch", "the fridge") — not
  a fixed zone label; there are no named rooms/zones any more, only landmarks the system has
  actually seen. Waypoint/arrival progress arrives as [SYSTEM] messages — respond immediately,
  concisely.

WALKING (FREE-WALK — ambient audio-only obstacle avoidance, no destination):
- `start_walking()` plays a continuous flapping-sound steering cue — quiet when the user is facing
  the right way, louder the more they need to turn — computed live from the occupancy map. Both
  `start_walking()` and `start_guiding()` also give a spoken clock-direction cue (e.g. "3 o'clock")
  at the start and each time the route turns toward a new point — respond to those immediately.
- If the user asks what the ongoing sound is, call it "the flapping sound" — never "beep."
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
- A `[SYSTEM] Incoming call from ...` message means the phone is ringing right now — tell the
  user immediately who's calling and ask if they want you to answer it; only call
  `answer_phone_call()` if they say yes.
- Before calling `send_sms`, always read the message back to the user and get explicit
  confirmation before sending — it can't be undone.
- A `[SYSTEM] New SMS from ...` message means a text just arrived — tell the user who it's from
  and read the message content immediately.
- Call `check_unread_sms()` when the user asks if they have any new/unread texts.

MUSIC / VIDEO:
- To play a song/video by name, call `search_youtube(query)` first, then `play_youtube_video`
  with the best match's `video_id`. If the user gives a specific video/song and you already know
  its `video_id` from a prior `search_youtube`/`get_video_info` call, you can call
  `play_youtube_video` directly.
- `stop_music()` stops whatever is currently playing, YouTube or otherwise.

NEWS / RADIO:
- `get_top_news()` for current Vietnamese headlines; `search_news(query)` for a specific topic.
  Summarize headlines briefly in English — do not read full articles or URLs aloud.
- `play_radio(station_query)` to find and play a Vietnamese internet radio station by name.
- `stop_radio()`/`stop_music()` both stop it — either works.
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

        // ── Reading — no enter_reading_mode/exit-on-every-use any more:
        // reading mode starts itself on first use of any of these three
        // tools, and ends itself automatically on switching to another
        // mode. exit_reading_mode is now only the explicit "clear" command.
        put(decl("scan_current_view", "Silently capture OCR text from the current camera frame into the reading buffer, without reading anything aloud. Auto-enters reading mode if not already in it."))
        put(decl(
            "get_reading_section",
            "Retrieve the most relevant passage from previously scanned reading material.",
            params(required = listOf("query")) { prop("query", "string", "The question or topic to find in the scanned text") }
        ))
        put(decl(
            "read_aloud",
            "Scan the current camera view into the reading buffer AND speak the entire buffer aloud through a dedicated local TTS engine, not your own voice. Auto-enters reading mode if not already in it. Use for 'read this'/'read it to me'/'read aloud'."
        ))
        put(decl("flip_reading_direction", "Toggle reading direction between left-to-right and right-to-left."))
        put(decl(
            "start_live_reading",
            "Start continuous reading mode: automatically captures and speaks new text on its own, repeatedly, with no further scan/read calls needed. Auto-enters reading mode if not already in it. Use for 'read to me as I move the camera' / 'keep reading' requests."
        ))
        put(decl("stop_live_reading", "Pause continuous automatic reading (started by start_live_reading). The scanned buffer is kept."))
        put(decl("continue_reading", "Resume reading aloud from wherever it was last interrupted, against the current buffer (picks up anything captured since, e.g. from live reading). Returns nothing_to_continue if there's nothing left to read."))
        put(decl("exit_reading_mode", "The 'clear' command: stops any active reading, discards the scanned text buffer, and exits reading mode. Not needed for a normal mode switch — that already exits reading mode on its own."))

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
            "Search saved object memory for items matching the query. Only for the user's OWN previously-saved items. If the result has ambiguous=true, do NOT guess which one — ask the user to clarify (e.g. point the camera more directly, or describe left/right/color) before calling start_tracking.",
            params(required = listOf("query")) { prop("query", "string", "Description of the object to look up in memory") }
        ))
        put(decl(
            "is_this_object",
            "Answers 'is this my <X>?' / 'is this the <X> I remembered?' — checks the largest matching object currently in view against the stored visual reference for that remembered label and returns a yes/no (is_match) with a similarity score. Only usable for a label that was previously saved via remember_object() while the object was visible.",
            params(required = listOf("label")) { prop("label", "string", "The saved memory label to check the current view against") }
        ))
        put(decl(
            "search_objects",
            "Answers questions about specific previously-saved objects currently in view, e.g. \"which one is my AC remote\" or \"which is my fan remote\" — for named/labeled belongings only, not generic objects (water, a pen, etc.), and not an explicit find-me/get-me-to/look-for request (that's start_tracking, not this). Pass `targets` naming which saved label(s) the user is asking about; omit it entirely to search every saved object with a stored visual reference (e.g. \"what's on the table\", \"is there any of my belongings here\").",
            params { propArray("targets", "string", "Saved memory label name(s) the user is asking about, e.g. [\"ac remote\", \"fan remote\"]. Omit to search all saved objects.") }
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
            "save_reading_buffer",
            "Save everything scanned/read so far in reading mode to a named memory label, queryable later via query_memory(). Prefer this over save_memory() for reading-mode content.",
            params(required = listOf("label")) { prop("label", "string", "Memory label") }
        ))
        put(decl(
            "remember_object",
            "Save the current object in view to memory, generating a label and appearance description. The description you pass is a fallback only — it is normally regenerated on-device from the actual image, kept short (color/shape/entity type).",
            params(required = listOf("label", "description")) {
                prop("label", "string", "Name user called the object or a short label you generated for it")
                prop("description", "string", "Brief fallback appearance description (color, shape, entity type) — a few words, not a full sentence")
            }
        ))
        put(decl("list_memory_labels", "List all named memories that have been saved."))
        put(decl(
            "clear_memory",
            "Permanently delete a saved memory label (its note/description and any visual reference). Use when the user says to forget/delete/remove a saved memory, or to clean up a duplicate/mislabeled save.",
            params(required = listOf("label")) { prop("label", "string", "The memory label to delete") }
        ))

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

        // ── Music / Video (YouTube Data API v3 search + official IFrame playback) ──
        put(decl(
            "search_youtube",
            "Search YouTube for music or videos matching a query. Returns candidate video_ids to pass to play_youtube_video.",
            params(required = listOf("query")) { prop("query", "string", "Search terms") }
        ))
        put(decl(
            "get_video_info",
            "Fetch full metadata (title, channel, duration) for specific YouTube video IDs.",
            params(required = listOf("video_ids")) { propArray("video_ids", "string", "List of YouTube video IDs") }
        ))
        put(decl(
            "play_youtube_video",
            "Play a specific YouTube video/song by its video ID, using the on-screen YouTube player.",
            params(required = listOf("video_id")) { prop("video_id", "string", "YouTube video ID") }
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

        // ── News / Radio (Vietnam-only — hardcoded vi/VN, no server proxy) ──
        put(decl("get_top_news", "Fetch current top news headlines in Vietnam (Vietnamese)."))
        put(decl(
            "search_news",
            "Search Vietnamese news articles by keyword or topic (e.g. 'VinFast', 'thời tiết').",
            params(required = listOf("query")) { prop("query", "string", "Topic or keyword to search for") }
        ))
        put(decl(
            "play_radio",
            "Search for and play a live Vietnamese internet radio station by name (e.g. 'VOV1', 'VOV Giao thông', 'FM 99.9').",
            params(required = listOf("station_query")) { prop("station_query", "string", "Name or channel of the radio station") }
        ))
        put(decl("stop_radio", "Stop any currently-playing internet radio stream. Equivalent to stop_music()."))

        // ── Calls & SMS ────────────────────────────────────────────────────
        put(decl("answer_phone_call", "Answer the currently-ringing incoming phone call."))
        put(decl(
            "send_sms",
            "Send a text message (SMS) to a contact or phone number.",
            params(required = listOf("recipient", "message")) {
                prop("recipient", "string", "Contact name or phone number")
                prop("message", "string", "The text message content to send")
            }
        ))
        put(decl("check_unread_sms", "Check for any unread SMS text messages in the inbox."))
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
