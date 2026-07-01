import asyncio
import json


async def search_youtube(session, query: str):
    cmd = [
        "yt-dlp", "--dump-json", "--flat-playlist",
        f"ytsearch5:{query}", "--no-warnings",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)

    results = []
    for i, line in enumerate(stdout.decode().strip().splitlines(), 1):
        if not line:
            continue
        try:
            info = json.loads(line)
        except json.JSONDecodeError:
            continue
        results.append({
            "index": i,
            "videoId": info.get("id", ""),
            "title": info.get("title", ""),
            "channel": info.get("uploader") or info.get("channel", ""),
            "duration": _fmt_dur(info.get("duration", 0)),
            "views": _fmt_views(info.get("view_count", 0)),
            "description": (info.get("description") or "")[:200],
        })

    session.state.music_search_results = results
    return {"results": results}


async def get_video_info(session, video_ids: list):
    async def _fetch(vid):
        cmd = [
            "yt-dlp", "--dump-json", "--no-warnings",
            f"https://www.youtube.com/watch?v={vid}",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
        info = json.loads(stdout.decode().strip())
        thumbnails = info.get("thumbnails") or []
        return {
            "videoId": info.get("id", ""),
            "title": info.get("title", ""),
            "channel": info.get("uploader", ""),
            "duration": _fmt_dur(info.get("duration", 0)),
            "views": _fmt_views(info.get("view_count", 0)),
            "description": (info.get("description") or "")[:500],
            "publishedAt": info.get("upload_date", ""),
            "thumbnail": thumbnails[-1].get("url", "") if thumbnails else "",
        }

    results = await asyncio.gather(*[_fetch(v) for v in video_ids], return_exceptions=True)
    return {"results": [r for r in results if not isinstance(r, Exception)]}


async def extract_stream_url(video_id: str) -> str:
    """Extract direct audio stream URL for a YouTube video. Not a Gemini tool — called internally by _dispatch_tool."""
    cmd = [
        "yt-dlp", "--get-url",
        "--format", "bestaudio[ext=m4a]/bestaudio/best",
        "--no-warnings",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=20.0)
    url = stdout.decode().strip().splitlines()[0]
    if not url:
        raise RuntimeError(f"yt-dlp returned no URL for video {video_id}")
    return url


def _fmt_dur(s):
    if not s:
        return "?:??"
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_views(n):
    if not n:
        return "unknown views"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M views"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K views"
    return f"{n} views"


HANDLERS = {
    "search_youtube": search_youtube,
    "get_video_info": get_video_info,
}
