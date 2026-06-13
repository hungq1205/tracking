#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import base64
import argparse
from typing import List, Dict, Any, Tuple

from openai import OpenAI

try:
    import cv2
except ImportError as e:
    raise ImportError("pip install opencv-python") from e


def _load_and_sample_frames(
    video_path: str,
    start_time: float,
    duration: int,
    fps: int = 2,
    max_frames: int = 60,
) -> Tuple[List[str], float]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # 计算总时长（毫秒）
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    total_ms = (frame_count / fps_video * 1000.0) if fps_video > 0 and frame_count > 0 else 0.0

    seg_start_ms = max(0.0, start_time * 1000.0)
    seg_end_ms = min(total_ms, (start_time + duration) * 1000.0) if total_ms > 0 else (start_time + duration) * 1000.0
    if seg_start_ms >= seg_end_ms:
        cap.release()
        return [], start_time  # empty interval

    # planned frame time points (milliseconds)
    step_ms = 1000.0 / max(fps, 1)
    times_ms = []
    t = seg_start_ms
    while t <= seg_end_ms + 1e-6:
        times_ms.append(t)
        t += step_ms

    if len(times_ms) > max_frames:
        stride = int(len(times_ms) / max_frames) + (0 if len(times_ms) % max_frames == 0 else 1)
        times_ms = times_ms[::stride]

    images_b64 = []
    for t in times_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        images_b64.append(b64)

    cap.release()
    return images_b64, seg_end_ms / 1000.0


def eval_gpt(
    model_name: str = "gpt-4o-mini",
    video_path: str = "/data/ruyi/dataset/streaming_vlm/Youtube_NBA/lKr5NocH5XM.mp4",
    query: str = "Please describe the video.",
    previous_text: str = "",
    start_time: float = 30,
    duration: int = 100,
    fps: int = 2,
) -> List[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_KEY not found")

    # 抽帧
    images_b64, actual_end_time = _load_and_sample_frames(
        video_path=video_path,
        start_time=start_time,
        duration=duration,
        fps=fps,
        max_frames=60,
    )
    if not images_b64:
        print("[TOKENS] No API request (no valid frames).")
        return [{"response": "Unable to extract valid frames for the specified time period.", "start_time": start_time, "end_time": start_time}]

    # improved commentary system prompt
    system_prompt = (
        "You are a live, precise sports commentator.\n"
        "Speak in vivid, present-tense play-by-play as if on air, immersing the listener.\n"
        "Return PLAIN TEXT ONLY — no Markdown, no lists, no headings, no code fences, no emojis.\n"
        "Describe only what is clearly visible in the provided frames. Do not invent or infer scores, clock, player names, teams, locations, or outcomes unless they are unmistakably shown.\n"
        "If a detail is unclear or off-screen, say it is unclear rather than guessing.\n"
        "Prefer concise, energetic sentences; avoid filler. 2–6 sentences are usually enough for a short clip.\n"
        "If previous commentary is provided, continue naturally from it without repeating it.\n"
        "Avoid meta-commentary about being an AI or about frames. Do not reference 'images' or 'prompts'; speak as if watching the live action.\n"
        "Read in-frame text (e.g., scoreboard) only if it is clearly legible, and quote it accurately.\n"
        "Do not include timestamps unless they are visible on screen.\n"
        "Respond in the same language as the user's query."
    )

    # user query + meta information
    header_text = (
        f"{query}\n"
        f"Time range: {start_time:.2f}s ~ {actual_end_time:.2f}s. "
        f"Frames sampled at {fps} fps (capped)."
    )
    if previous_text:
        header_text += f"\nPrevious commentary (context): {previous_text}"

    # organize multimodal messages (do not pass temperature)
    content = [{"type": "text", "text": header_text}]
    for b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=400,
        # do not pass temperature
    )

    
    try:
        text = resp.choices[0].message.content.strip()
    except Exception:
        text = str(resp)

    usage = getattr(resp, "usage", None)
    if usage is None:
        print("[TOKENS] This API did not return the usage field.")
    else:
        try:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        except Exception:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        print(f"[TOKENS] prompt={prompt_tokens} completion={completion_tokens} total={total_tokens} | model={getattr(resp, 'model', None)}")

    commentaries = [
        {
            "response": text,
            "start_time": float(start_time),
            "end_time": float(actual_end_time),
        }
    ]
    return commentaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini')
    parser.add_argument('--video_path', type=str, default='/data/ruyi/dataset/streaming_vlm/Youtube_NBA/lKr5NocH5XM.mp4')
    parser.add_argument('--query', type=str, default='Please describe the video.')
    parser.add_argument('--previous_text', type=str, default='')
    parser.add_argument('--start_time', type=float, default=30)
    parser.add_argument('--duration', type=int, default=100)
    args = parser.parse_args()

    out = eval_gpt(
        model_name=args.model_name,
        video_path=args.video_path,
        query=args.query,
        previous_text=args.previous_text,
        start_time=args.start_time,
        duration=args.duration,
    )
    print(out)
