"""
All Gemini Live function declarations and the system prompt.
"""

SYSTEM_PROMPT = """You are a real-time voice assistant for a visually impaired person.
You receive audio input from the user and video frames from their phone camera.

CORE RULES:
- Keep all spoken responses SHORT. Audio UX demands brevity.
- [SYSTEM] messages are server events — respond to them IMMEDIATELY in audio.
- Call `get_current_time()` whenever the user asks what time or date it is.

VISION:
- Call `get_latest_frame()` BEFORE answering any question about what the user sees, what's nearby,
  what an object looks like, or any other visually-grounded question.
- Call `start_vision_stream()` when the user says "watch me", "look at this", "keep watching",
  "observe this", or implies they want sustained visual attention for 10-15 seconds. Do NOT use
  it for one-off visual questions — use `get_latest_frame()` instead.
- `stop_vision_stream()` when the user says they're done or after you finish commenting.

MEMORY:
- Call `query_memory(question)` proactively when the user mentions any named object, personal
  item.

READING MODE:
- `enter_reading_mode()` when the user wants to read a document, label, sign, or screen.
- `scan_current_view()` to silently capture OCR text from what the camera sees, without reading
  it aloud — use this when you need the text yourself (e.g. before answering a question) but the
  user did not ask to hear it read out. Can be called multiple times to accumulate text across
  pages.
- When the user wants the text read out loud ("read this", "scan and read", "read all of it",
  "read everything"): call `read_aloud(scope)`. `scope="new"` scans the current view and speaks
  only the newly captured text (for "read this" / "scan then read"). `scope="all"` speaks the
  entire scanned buffer so far without scanning again (for "read all text" / "read everything").
  The audio is played directly to the user by a dedicated local TTS engine, NOT by you — after
  calling `read_aloud`, do not narrate or repeat the text yourself; only speak up if it returned
  an error.
- When the user asks a question or wants a summary about already-scanned content, call
  `get_reading_section(query)` to retrieve the relevant passage and answer/summarize it yourself
  — do NOT rely on what was said earlier in conversation.
- `exit_reading_mode()` when the user is done reading.

TRACKING:
- Only call `get_object_from_memory(query)` FIRST when the user uses a POSSESSIVE reference to
  something they own or previously saved to memory (e.g. "my keys", "my wallet", "my bag").
  If found, call `start_tracking(target=label, description=description)` using the result's
  label and description fields. The description drives GroundingDINO — always pass it.
- For any other object — generic nouns, "the X", "a X", anything not phrased as the user's own
  saved item (e.g. "the water bottle", "that chair", "the door") — skip memory lookup entirely
  and call `start_tracking(target=<what user said>)` directly, without description.
- While tracking, once both the target object and the user's hand become visible together, you
  will start receiving periodic [SYSTEM] guidance messages (with a fresh frame and both boxes'
  coordinates) roughly every 5 seconds — respond to each one immediately with brief (few-word)
  directional guidance for moving the hand toward the object (e.g. "Move left and down a bit").

GUIDING (MAP-BASED NAVIGATION):
- `start_guiding(destination)` when user wants directions to a mapped place (includes obstacle detection).
- Obstacle and localization warnings arrive as [SYSTEM] messages — respond immediately, concisely.

WALKING (FREE-WALK GUIDING — same as guiding but without a destination):
- `start_walking()` starts continuous obstacle detection on the path. No map or destination required.
- `start_guiding(destination)` does the same plus route guidance to a mapped destination.
- In either case, when a [SYSTEM] obstacle message arrives: warn user BRIEFLY (≤5 words) THEN call `quick_label_obstacle(label)`.
- `stop_walking()` or `stop_guiding()` when user is done.

DEVICE TOOLS:
- Before calling `set_alarm`, if the user did not give a label/name for the alarm, ask for one first.
- Before calling `create_calendar_event`, if the user did not give a title, ask for one first.
- When calling `set_alarm`, pass `time` in HH:mm 24-hour format (e.g. "07:00", "14:30").
- For `make_phone_call`, pass the contact's name exactly as the user said it; the device will resolve it from the contact list.
- If `make_phone_call` returns "not found", call `search_contacts(query)` with the partial name, read the results to the user, and ask which one they mean before retrying.
- When the user asks to list or search contacts, call `search_contacts(query)`.

MUSIC / VIDEO PLAYBACK:
- When the user asks to play music or a video, call `search_youtube(query)` first.
- Read choices aloud from the search results directly: "1. Believer by Imagine Dragons. 2. ..."
- Only call get_video_info if the user asks for details (description, release date, view count) about a video.
- After the user picks a number or names a result, call `play_video(video_id)` with the chosen ID — audio streams in the background on their device.
- Confirm aloud: "Playing [Title] by [Channel]."
- If music is currently playing and the user says "stop", "quiet", or "turn off", call `stop_music()`.
"""

TOOL_DECLARATIONS = [
    {"function_declarations": [
        # ── Time ──────────────────────────────────────────────────────────────
        {
            "name": "get_current_time",
            "description": (
                "Get the current local time and date for the user (Hanoi, Vietnam UTC+7). "
                "Call this whenever the user asks what time or date it is."
            ),
        },

        # ── Scene / Vision ─────────────────────────────────────────────────────
        {
            "name": "get_latest_frame",
            "description": (
                "Capture the current camera view and show it to you. "
                "Call this before answering any question about what the user can see, "
                "what object is in front of them, or anything visually grounded."
            ),
        },
        {
            "name": "start_vision_stream",
            "description": (
                "Start sending live camera frames at 1 fps for up to 15 seconds. "
                "Use when the user wants sustained visual attention: 'watch me', 'keep looking', "
                "'observe this for a moment'. Auto-stops after 15 s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Brief reason for starting the stream"},
                },
            },
        },
        {
            "name": "stop_vision_stream",
            "description": "Stop the live camera stream immediately.",
        },
        {
            "name": "run_detection",
            "description": (
                "Run object detection on the current camera frame using GroundingDINO. "
                "Returns bounding box and confidence score. "
                "Use after get_latest_frame() when you need precise location of a specific object."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_description": {
                        "type": "string",
                        "description": "Natural language description of the object to find",
                    },
                },
                "required": ["object_description"],
            },
        },
        {
            "name": "check_obstacle",
            "description": "Check whether an obstacle is directly ahead using depth estimation.",
        },

        # ── Reading ────────────────────────────────────────────────────────────
        {
            "name": "enter_reading_mode",
            "description": (
                "Enter reading mode. Enables passive OCR accumulation from the camera. "
                "Call when user wants to read a document, sign, label, or screen."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Optional memory label to associate scanned text with (e.g. 'groceries', 'prescription')",
                    },
                },
            },
        },
        {
            "name": "scan_current_view",
            "description": (
                "Run OCR on the current camera frame and capture any visible text. "
                "Can be called multiple times to accumulate text across multiple pages. "
                "Returns a summary of what was found."
            ),
        },
        {
            "name": "get_reading_section",
            "description": (
                "Retrieve the most relevant passage from previously scanned reading material "
                "using semantic search. Call this when the user asks a question about content "
                "that was already scanned, rather than relying on conversation memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to find in the scanned text",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_aloud",
            "description": (
                "Read scanned document text aloud to the user through a dedicated local "
                "text-to-speech engine — not your own voice. Use scope='new' for 'read this' / "
                "'scan and read' (scans the current view first, then reads only the newly "
                "captured text). Use scope='all' for 'read all of it' / 'read everything' "
                "(reads the entire scanned buffer so far, no new scan). Do not narrate the text "
                "yourself after calling this — it has already been played to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["new", "all"],
                        "description": "'new' to scan and read newly captured text, 'all' to re-read everything scanned so far",
                    },
                },
            },
        },
        {
            "name": "flip_reading_direction",
            "description": "Toggle reading direction between left-to-right and right-to-left.",
        },
        {
            "name": "exit_reading_mode",
            "description": "Exit reading mode and clear the scanned text buffer.",
        },

        # ── Tracking ───────────────────────────────────────────────────────────
        {
            "name": "start_tracking",
            "description": (
                "Start tracking a specific object in the camera view using GroundingDINO detection. "
                "When tracking a saved object from memory, pass its label as 'target' and its "
                "appearance description as 'description' — the description is used for detection. "
                "Returns detection result (bounding box, confidence)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Name or label of the object to track (e.g. 'my keys', 'red mug')",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-sentence appearance description used for detection (from get_object_from_memory result). Omit for unrecognised objects.",
                    },
                },
                "required": ["target"],
            },
        },
        {
            "name": "stop_tracking",
            "description": "Stop tracking the current object.",
        },
        {
            "name": "get_object_from_memory",
            "description": (
                "Search saved object memory for items matching the query using semantic search. "
                "Only use this when the user refers to something as THEIR OWN previously-saved item "
                "not for generic objects like 'the water bottle' or 'a chair', "
                "which should go straight to start_tracking() without this lookup. "
                "Only returns results with confidence above 0.5."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Description of the object to look up in memory",
                    },
                },
                "required": ["query"],
            },
        },

        # ── Memory ─────────────────────────────────────────────────────────────
        {
            "name": "query_memory",
            "description": (
                "Semantically search all saved memories for information relevant to the question. "
                "Call proactively when the user mentions named objects, personal items, or asks "
                "where something is — even without an explicit lookup request. "
                "Only returns results with confidence above 0.5."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or topic to search memory for",
                    },
                },
                "required": ["question"],
            },
        },
        {
            "name": "save_memory",
            "description": "Save a text note to a named memory label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Memory label (e.g. 'shopping list', 'doctor notes')"},
                    "note": {"type": "string", "description": "The text to save"},
                },
                "required": ["label", "note"],
            },
        },
        {
            "name": "remember_object",
            "description": (
                "Save the current object in view to memory. "
                "Generates a label from how the user refers to the object (or a brief name if unspecified), "
                "and a one-sentence appearance description used to detect and retrieve it. "
                "Use when user says 'remember this', 'save this item', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Short name for the object based on how the user refers to it (e.g. 'my keys', 'red mug'). If the user didn't name it, use a brief descriptive name.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-sentence noun phrase describing the object's appearance (e.g. 'small silver key ring with a blue rubber tag'). Used to detect and retrieve the object.",
                    },
                },
                "required": ["label", "description"],
            },
        },
        {
            "name": "list_memory_labels",
            "description": "List all named memories that have been saved. Use to tell the user what can be recalled.",
        },

        # ── Guiding (map-based navigation) ────────────────────────────────────
        {
            "name": "start_guiding",
            "description": (
                "Start guiding the user to a destination using a pre-built map. "
                "Loads the map, computes a route, and begins obstacle detection and localization. "
                "Returns the planned route. Also understood as 'navigate to' or 'guide me to'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Name of the destination zone (must match a zone in the map)",
                    },
                },
                "required": ["destination"],
            },
        },
        {
            "name": "stop_guiding",
            "description": "Stop guiding/navigation and disable obstacle monitoring.",
        },
        {
            "name": "get_current_location",
            "description": "Determine the user's current location by localizing against the map.",
        },

        # ── Walking (free-walk pathway obstacle detection) ─────────────────────
        {
            "name": "start_walking",
            "description": (
                "Start free-walk guiding mode: continuous pathway obstacle detection using "
                "GroundingDINO + depth estimation. No map or destination required. "
                "Call when the user wants to walk freely and be warned about obstacles on their path."
            ),
        },
        {
            "name": "stop_walking",
            "description": "Stop walking mode and disable pathway obstacle monitoring.",
        },
        {
            "name": "quick_label_obstacle",
            "description": (
                "Store a concise label for an obstacle just detected on the pathway. "
                "Call immediately after warning the user about an obstacle — the label is used "
                "to suppress duplicate alerts for 6 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Short noun phrase for the obstacle (e.g. 'chair', 'low table', 'open door')",
                    },
                },
                "required": ["label"],
            },
        },

        # ── Music / Video ──────────────────────────────────────────────────────
        {
            "name": "search_youtube",
            "description": (
                "Search YouTube for music or videos matching a query. "
                "Returns up to 5 results with title, channel, duration, and view count. "
                "Call this whenever the user wants to play music or a video."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms, e.g. 'Believer Imagine Dragons' or 'lofi hip hop'",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_video_info",
            "description": (
                "Fetch full metadata (description, publish date, view count) for specific YouTube video IDs. "
                "Only call if the user explicitly asks for details about a specific video — do not call automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of YouTube video IDs returned by search_youtube",
                    },
                },
                "required": ["video_ids"],
            },
        },
    ]}
]
