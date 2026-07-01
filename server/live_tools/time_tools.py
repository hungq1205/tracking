from datetime import datetime
from zoneinfo import ZoneInfo

_HANOI_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


async def tool_get_current_time(session, **_):
    now = datetime.now(_HANOI_TZ)
    return {
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%A, %d %B %Y"),
        "timezone": "Asia/Ho_Chi_Minh (UTC+7)",
    }


HANDLERS = {
    "get_current_time": tool_get_current_time,
}
