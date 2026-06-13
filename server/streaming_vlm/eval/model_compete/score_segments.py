#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# ===== Read/Write Tools =====
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_merged(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Read merged_result.jsonl (one video per line), return video->obj
    Expected fields: video, chunk_start, chunk_end, items
    items[i] at least contains { "end_time": <int/float>, "text": <str> }
    """
    m = {}
    for obj in read_jsonl(path):
        v = obj["video"]
        m[v] = obj
    return m

def seg_bounds_from_content(content: List[List[Any]]) -> Tuple[int, int, str]:
    """
    content: [[s,e,text], ...] covers a continuous window.
    Return (seg_start, seg_end, ref_text)
    """
    if not content:
        return 0, 0, ""
    s0 = int(content[0][0])
    e1 = int(content[-1][1])
    ref_text = "\n".join(str(x[2]) for x in content if len(x) >= 3)
    return s0, e1, ref_text

def extract_commentary(items: List[Dict[str, Any]], seg_start: int, seg_end: int) -> str:
    """
    Extract text from model items located in [seg_start, seg_end), concatenated in chronological order.
    Rule: select sentences where end_time ∈ (seg_start, seg_end].
    """
    out = []
    for it in items:
        et = it.get("end_time")
        if et is None:
            continue
        t = int(et)
        if seg_start < t <= seg_end:
            txt = it.get("response") if "response" in it else it.get("text")
            if isinstance(txt, str) and txt.strip():
                out.append(txt.strip())
    return " ".join(out)

# ===== Evaluation Interface (locally import methods from judge.py) =====
from judge import judge_pair  # Same directory
from judge import _mk_client, SYS_PROMPT, USER_TEMPLATE  # Optional: if you want to reuse

def score_pair(judge_model: str, meta_row: Dict[str, Any],
               cand_A: str, cand_B: str) -> Tuple[str, str]:
    """
    Return (vote_ab, vote_ba)
    """
    return judge_pair(judge_model, meta_row, cand_A, cand_B)

# ---- Single-line function for parallel processing (no change to evaluation logic) ----
def _process_one_line(line: str, 
                      A: Dict[str, Dict[str, Any]], 
                      B: Dict[str, Dict[str, Any]], 
                      judge_model: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Input: original single-line json string
    Output: (writable jsonl string or None, warning string or None)
    Only handle logic related to single samples within this function, evaluation flow consistent with original version.
    """
    try:
        row = json.loads(line)
        video_rel = row["video"]
        title = row.get("title", "")
        content = row["content"]

        seg_start, seg_end, ref_text = seg_bounds_from_content(content)
        # Check coverage
        a_obj = A.get(os.path.basename(video_rel))
        b_obj = B.get(os.path.basename(video_rel))
        if not a_obj or not b_obj:
            return None, f"[WARN] {video_rel}: missing merged line in A/B"

        a_cov = (int(a_obj["chunk_start"]), int(a_obj["chunk_end"]))
        b_cov = (int(b_obj["chunk_start"]), int(b_obj["chunk_end"]))
        cover_ok = (a_cov[0] <= seg_start and seg_end <= a_cov[1] and
                    b_cov[0] <= seg_start and seg_end <= b_cov[1])
        if not cover_ok:
            return None, f"[WARN] {video_rel}: window [{seg_start},{seg_end}) not covered by A{a_cov}/B{b_cov}, skip."

        cand_A = extract_commentary(a_obj.get("items", []), seg_start, seg_end)
        cand_B = extract_commentary(b_obj.get("items", []), seg_start, seg_end)
        # Pass metadata to judge, used for prompt
        meta = {
            "video": os.path.basename(video_rel),
            "seg_start": seg_start,
            "seg_end": seg_end,
            "title": title,
            "ref_text": ref_text
        }
        vote_ab, vote_ba = score_pair(judge_model, meta, cand_A, cand_B)
        votes_A = int(vote_ab == "A") + int(vote_ba == "A")
        votes_B = int(vote_ab == "B") + int(vote_ba == "B")
        if votes_A > votes_B:
            winner = "A"
        elif votes_B > votes_A:
            winner = "B"
        else:
            winner = "equal"

        out_obj = {
            "video": os.path.basename(video_rel),
            "title": title,
            "start": seg_start,
            "end": seg_end,
            "winner": winner,
            "votes": {"A": votes_A, "B": votes_B},
            "judge": {
                "model": judge_model,
                "vote_ab": vote_ab,
                "vote_ba": vote_ba
            }
        }
        return json.dumps(out_obj, ensure_ascii=False), None
    except Exception as e:
        return None, f"[WARN] exception: {e}"

# ===== Main Process =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_a", type=str, required=True, help="results/{model1}/merged_result.jsonl")
    parser.add_argument("--merged_b", type=str, required=True, help="results/{model2}/merged_result.jsonl")
    parser.add_argument("--score_metadata", type=str, default="")
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--start_line", type=int, required=True)
    parser.add_argument("--end_line", type=int, required=True)
    parser.add_argument("--out_path", type=str, required=True, help="Output shard: score_segments_shard.jsonl")
    args = parser.parse_args()

    A = load_merged(args.merged_a)
    B = load_merged(args.merged_b)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    skipped = 0
    written = 0

    with open(args.score_metadata, "r", encoding="utf-8") as f_in, \
         open(args.out_path, "w", encoding="utf-8") as f_out:

        # Only take lines in specified range, maintain original order
        lines: List[str] = [line for idx, line in enumerate(f_in) if args.start_line <= idx < args.end_line]

        # 32 threads parallel evaluation, map preserves order
        with ThreadPoolExecutor(max_workers=32) as ex:
            for out_line, warn in ex.map(lambda l: _process_one_line(l, A, B, args.judge_model), lines):
                if warn is not None:
                    print(warn)
                    skipped += 1
                elif out_line is not None:
                    f_out.write(out_line + "\n")
                    written += 1

    print(f"[score] written={written}, skipped={skipped}, out={args.out_path}")

if __name__ == "__main__":
    main()
