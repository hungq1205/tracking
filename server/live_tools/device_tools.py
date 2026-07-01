"""
Device-native tool declarations routed to the connected client (Android/iOS).
These are never executed server-side; the server relays them via AudioChunk.tool_call.
"""

DEVICE_TOOL_NAMES = {"make_phone_call", "set_alarm", "create_calendar_event", "search_contacts", "play_video", "stop_music"}

DEVICE_TOOL_DECLARATIONS = [
    {
        "name": "search_contacts",
        "description": (
            "Search the device contact list by name. "
            "Use when the user is unsure of a contact's exact name, asks to list contacts, "
            "or when make_phone_call fails with 'not found'. "
            "Returns a list of matching names so you can read them to the user and let them pick."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Partial name to search for, e.g. 'F', 'Bao', 'mom'. Empty string lists all contacts.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "make_phone_call",
        "description": (
            "Make a phone call. Pass a contact name (e.g. 'Mom', 'Bao') or a phone number. "
            "The device resolves names against the contact list automatically. "
            "Use when the user says 'call [name/number]', 'dial [number]', or similar."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contact_name_or_number": {
                    "type": "string",
                    "description": "Contact name exactly as the user said (e.g. 'Mom', 'Bao') or phone number (e.g. '+84912345678')",
                },
            },
            "required": ["contact_name_or_number"],
        },
    },
    {
        "name": "set_alarm",
        "description": (
            "Set an alarm on the device. "
            "Ask for a label if the user did not provide one. "
            "Use when the user says 'set an alarm for [time]', 'wake me up at [time]', etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "Alarm time in HH:mm 24-hour format, e.g. '07:00' for 7 AM, '14:30' for 2:30 PM.",
                },
                "label": {
                    "type": "string",
                    "description": "Label for the alarm, e.g. 'Morning meds', 'Meeting'. Ask the user if not specified.",
                },
            },
            "required": ["time", "label"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": (
            "Create a calendar event on the device. "
            "Use when the user wants to schedule an appointment, meeting, or reminder. "
            "The device will open the calendar app to confirm the event."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Title of the event, e.g. 'Doctor appointment'",
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in natural language, e.g. 'tomorrow 3pm', '2024-01-15 10:00'",
                },
                "end_time": {
                    "type": "string",
                    "description": "Optional end time in natural language",
                },
                "description": {
                    "type": "string",
                    "description": "Optional notes or description for the event",
                },
            },
            "required": ["title", "start_time"],
        },
    },
    {
        "name": "stop_music",
        "description": "Stop the currently playing music or video on the user's device.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "play_video",
        "description": (
            "Play a YouTube video in the background on the user's device via ExoPlayer. "
            "Call after the user selects a result from search_youtube. "
            "The server automatically extracts the audio stream URL before forwarding to the device."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "YouTube video ID from search_youtube results, e.g. 'ktvTqknDobU'",
                },
            },
            "required": ["video_id"],
        },
    },
]
