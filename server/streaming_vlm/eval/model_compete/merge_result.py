#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
from collections import defaultdict

def load_shards(shard_dir: str):
    """
    Read all .jsonl files under shard_dir (compliant: one JSON object per line) and group:
    Returns dict: video -> {chunk_start: {"video", "chunk_start", "chunk_end", "items"}}
    For the same (video, chunk_start), only keep the one with maximum chunk_end.
    """
    by_video = defaultdict(dict)  # video -> {chunk_start -> seg}
    for fname in sorted(os.listdir(shard_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(shard_dir, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)  # Only consider compliant JSONL
                video = obj["video"]
                s = int(obj["chunk_start"])
                e = int(obj["chunk_end"])
                seg = {
                    "video": video,
                    "chunk_start": s,
                    "chunk_end": e,
                    "items": obj.get("items", []),
                }
                prev = by_video[video].get(s)
                if prev is None or e > prev["chunk_end"]:
                    by_video[video][s] = seg
    return by_video

def merge_to_single_line(by_video):
    """
    Merge multiple segments of the same video into a single line:
      chunk_start = minimum start
      chunk_end   = maximum end
      items       = concatenate in ascending order by chunk_start
    Returns list[dict], each dict is one output object per line.
    """
    merged_rows = []
    for video in sorted(by_video.keys()):
        segs = list(by_video[video].values())
        if not segs:
            continue
        segs.sort(key=lambda x: (x["chunk_start"], x["chunk_end"]))
        min_start = segs[0]["chunk_start"]
        max_end = max(s["chunk_end"] for s in segs)

        all_items = []
        for s in segs:
            items = s.get("items", [])
            if isinstance(items, list):
                all_items.extend(items)

        merged_rows.append({
            "video": video,
            "chunk_start": int(min_start),
            "chunk_end": int(max_end),
            "items": all_items
        })
    return merged_rows

def write_output(output_path: str, rows):
    with open(output_path, "w", encoding="utf-8") as out:
        for obj in rows:
            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")
    print(f"[merge] wrote {len(rows)} videos to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.shard_dir):
        raise FileNotFoundError(f"shard_dir not found: {args.shard_dir}")

    by_video = load_shards(args.shard_dir)
    rows = merge_to_single_line(by_video)
    write_output(os.path.join(args.shard_dir, "merged_result.jsonl"), rows)

if __name__ == "__main__":
    main()
