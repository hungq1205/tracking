#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
from collections import defaultdict

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def merge_shards(dir_path: str):
    segs = []
    for name in sorted(os.listdir(dir_path)):
        if not name.endswith(".jsonl"):
            continue
        if not name.startswith("score_segments_"):
            continue
        for obj in read_jsonl(os.path.join(dir_path, name)):
            segs.append(obj)
    # Deduplicate (for same video,start,end keep the last one)
    key2obj = {}
    for o in segs:
        k = (o["video"], o["start"], o["end"])
        key2obj[k] = o
    merged = list(key2obj.values())
    merged.sort(key=lambda x: (x["video"], x["start"], x["end"]))
    return merged

def summarize(segments):
    per_video = {}
    global_A = 0
    global_B = 0
    for o in segments:
        v = o["video"]
        a = int(o["votes"]["A"])
        b = int(o["votes"]["B"])
        if v not in per_video:
            per_video[v] = {"A": 0, "B": 0}
        per_video[v]["A"] += a
        per_video[v]["B"] += b
        global_A += a
        global_B += b

    summary = {"per_video": {}, "global": {}}
    for v in sorted(per_video.keys()):
        A = per_video[v]["A"]
        B = per_video[v]["B"]
        total = max(1, A + B)
        summary["per_video"][v] = {
            "A": A, "B": B,
            "win_rate_A": round(A / total, 4),
            "win_rate_B": round(B / total, 4),
        }
    gtotal = max(1, global_A + global_B)
    summary["global"] = {
        "A": global_A, "B": global_B,
        "win_rate_A": round(global_A / gtotal, 4),
        "win_rate_B": round(global_B / gtotal, 4),
    }
    return summary

def write_outputs(out_dir: str, segments, summary):
    os.makedirs(out_dir, exist_ok=True)
    seg_path = os.path.join(out_dir, "score_segments.jsonl")
    with open(seg_path, "w", encoding="utf-8") as f:
        for o in segments:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    sum_path = os.path.join(out_dir, "score_results.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rep_path = os.path.join(out_dir, "score_results.report.txt")
    lines = []
    g = summary["global"]
    lines.append(f"GLOBAL: A={g['A']} B={g['B']}  win_rate_A={g['win_rate_A']}  win_rate_B={g['win_rate_B']}")
    lines.append("")
    lines.append("Per-Video:")
    for v, s in summary["per_video"].items():
        lines.append(f"{v}: A={s['A']} B={s['B']}  win_rate_A={s['win_rate_A']}  win_rate_B={s['win_rate_B']}")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[merge] wrote:\n  {seg_path}\n  {sum_path}\n  {rep_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_dir", type=str, required=True,
                        help="result/{PAIR_TAG}/, containing various shard outputs score_segments_*.jsonl")
    args = parser.parse_args()

    segments = merge_shards(args.pair_dir)
    summary = summarize(segments)
    write_outputs(args.pair_dir, segments, summary)

if __name__ == "__main__":
    main()
