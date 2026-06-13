#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import base64
import argparse
import json
import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

from openai import OpenAI

# Dependency: opencv-python
try:
    import cv2
except ImportError as e:
    raise ImportError("install opencv-python：pip install opencv-python") from e


def _load_and_sample_frames(
    video_path: str,
    start_time: float,
    duration: int,
    fps: float = 2.0,
    max_frames: int = 60,
) -> Tuple[List[str], float]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    total_ms = (frame_count / fps_video * 1000.0) if fps_video > 0 and frame_count > 0 else 0.0

    seg_start_ms = max(0.0, start_time * 1000.0)
    seg_end_ms = min(total_ms, (start_time + duration) * 1000.0) if total_ms > 0 else (start_time + duration) * 1000.0
    if seg_start_ms >= seg_end_ms:
        cap.release()
        return [], start_time

    # Allow fractional FPS; e.g., 0.5 fps -> take one frame every 2000ms
    fps_eff = float(fps) if float(fps) > 0 else 1e-6
    step_ms = 1000.0 / fps_eff

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

        h, w = frame.shape[:2]
        if h > 360:
            new_w = max(1, int(round(w * 360.0 / h)))
            frame = cv2.resize(frame, (new_w, 360), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        images_b64.append(b64)

    cap.release()
    return images_b64, seg_end_ms / 1000.0


def _openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAPI_KEY")
    if not api_key:
        raise EnvironmentError(f"Unfound OPENAI_KEY/OPENAI_API_KEY/OPENAPI_KEY")
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if base:
        return OpenAI(api_key=api_key, base_url=base)
    return OpenAI(api_key=api_key)


def eval_gpt_window(
    client: OpenAI,
    model_name: str,
    video_path: str,
    query: str,
    start_time: float,
    duration: int,
    fps: float = 2.0,
    preasr: str = "",
) -> Tuple[str, float, float, Dict[str, int]]:
    images_b64, actual_end_time = _load_and_sample_frames(
        video_path=video_path,
        start_time=start_time,
        duration=duration,
        fps=fps,
        max_frames=60,
    )
    if not images_b64:
        return "", start_time, start_time, {}

    system_prompt = (
        "You are a live, precise sports commentator.\n"
        "Speak in vivid, present-tense play-by-play as if on air.\n"
        "Return PLAIN TEXT ONLY — no Markdown, lists, headings, code fences, or emojis.\n"
        "Describe what is clearly visible; avoid guessing.\n"
        "Use concise, energetic sentences (2–6). Respond in the query language."
    )

    header = f"{query}\nTime range: {start_time:.2f}s ~ {actual_end_time:.2f}s. Frames sampled at {fps} fps (capped)."# Previous segment: {preasr}"
    content = [{"type": "text", "text": header}]
    for b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": content}],
            )
            txt = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            if usage is None:
                usage_dict = {}
            else:
                try:
                    usage_dict = {
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0)),
                        "total_tokens": int(getattr(usage, "total_tokens", 0)),
                    }
                except Exception:
                    usage_dict = {
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                    }
            return txt, start_time, actual_end_time, usage_dict
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"[score][ERROR] GPT call failed (skip window): {video_path} [{start_time},{start_time+duration}) | {last_err}")
    return "", start_time, start_time, {}



def read_score_metadata(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def seg_bounds_from_content(content: List[List[Any]]) -> Tuple[int, int]:
    if not content:
        return 0, 0
    return int(content[0][0]), int(content[-1][1])


def read_done_videos(out_path: str) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except Exception:
                continue
            v = obj.get("video")
            if isinstance(v, str):
                done.add(v)
    return done



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--score_metadata", type=str, default="")
    parser.add_argument("--video_root", type=str, default="/")
    parser.add_argument("--workers", type=int, default=64, help="")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--query", type=str, default="Please describe the video.")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    client = _openai_client()

    safe_model = args.model_name.replace("/", "_")
    out_dir = os.path.join(args.results_dir, safe_model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "merged_result.jsonl")

    rows = read_score_metadata(args.score_metadata)

    windows_by_video: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    tasks = []  # (video_rel, s, e, preasr)
    for row in rows:
        v_rel = row["video"]
        s, e = seg_bounds_from_content(row["content"])
        if e > s:
            windows_by_video[v_rel].append((int(s), int(e)))
            tasks.append((v_rel, int(s), int(e), row['preasr']))

    done = read_done_videos(out_path)
    videos_all = sorted(windows_by_video.keys())
    videos_plan = [v for v in videos_all if os.path.basename(v) not in done]
    vids_skip = [v for v in videos_all if os.path.basename(v) in done]
    if vids_skip:
        print(f"[score] RESUME skip={len(vids_skip)} already_done={', '.join(os.path.basename(v) for v in vids_skip[:6])}{'...' if len(vids_skip)>6 else ''}")

    tasks = [(v, s, e, preasr) for (v, s, e, preasr) in tasks if os.path.basename(v) not in done]

    state_lock = threading.Lock()
    remain: Dict[str, int] = {}
    agg_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    agg_start: Dict[str, int] = {}
    agg_end: Dict[str, int] = {}

    for v in videos_plan:
        win = windows_by_video[v]
        if not win:
            continue
        win.sort(key=lambda x: x[0])
        remain[v] = len(win)
        agg_start[v] = win[0][0]
        agg_end[v] = win[-1][1]
        print(f"[score] START {os.path.basename(v)} windows={len(win)} range=[{agg_start[v]},{agg_end[v]})")

    def _work_window(v_rel: str, s: int, e: int, preasr: str):
        v_abs = os.path.join(args.video_root, v_rel.lstrip("/"))
        dur = int(e - s)
        txt, s0, e0, usage = eval_gpt_window(
            client=client, model_name=args.model_name, video_path=v_abs,
            query=args.query, start_time=float(s), duration=dur, fps=args.fps,
            preasr=preasr,
        )
        tokens = int(usage.get("total_tokens", 0)) if usage else 0
        if txt and e0 > s0:
            with state_lock:
                agg_items[v_rel].append({"end_time": int(e0), "response": txt})
                remain[v_rel] -= 1
                left = remain[v_rel]
            print(f"[score] WINDOW {os.path.basename(v_rel)} [{s},{e}) len={len(txt)} tokens~{tokens} left={left}")
        else:
            with state_lock:
                remain[v_rel] -= 1
                left = remain[v_rel]
            print(f"[score][WARN] skip window: {os.path.basename(v_rel)} [{s},{e}) (empty or no frames) left={left}")
        # If all windows for this video are complete → write a line immediately
        if left == 0:
            items = agg_items[v_rel]
            items.sort(key=lambda x: int(x.get("end_time", 0)))
            obj = {
                "video": os.path.basename(v_rel),
                "chunk_start": int(agg_start[v_rel]),
                "chunk_end": int(agg_end[v_rel]),
                "items": items
            }
            line = json.dumps(obj, ensure_ascii=False) + "\n"
            with state_lock:  # Mutex with other video writes
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            print(f"[score] WRITE {os.path.basename(v_rel)} items={len(items)} "
                  f"range=[{agg_start[v_rel]},{agg_end[v_rel]}) out={out_path}")
        return tokens

    total_windows_plan = len(tasks)
    grand_tokens = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_work_window, v, s, e, preasr) for (v, s, e, preasr) in tasks]
        for fut in as_completed(futs):
            try:
                grand_tokens += int(fut.result() or 0)
            except Exception as e:
                print(f"[score][ERROR] worker failed: {e}")

    vids_done_now = sum(1 for v in videos_plan if os.path.basename(v) in read_done_videos(out_path))
    print(f"[score] DONE videos_written~{vids_done_now}/{len(videos_plan)} "
          f"windows={total_windows_plan} tokens~{grand_tokens} out={out_path}")


if __name__ == "__main__":
    main()
