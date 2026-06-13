#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Tuple, Dict, Any
from openai import OpenAI

SYS_PROMPT = (
    """
    You are a strict A/B evaluator for sports video commentary.

    Rules:
    1) You MUST output exactly one uppercase letter: “A” or “B”.

    2) Evaluate in this order:
    
    Broadcast tone & pacing: Vary the emotion; deliver live play-by-play or brief, in-the-moment analysis. Not just scene description. Use a real-time broadcast voice that puts the audience courtside.

    Consistency & accuracy: Every detail must stay synchronized with the visuals and never contradict them.
    """
)

USER_TEMPLATE = (
    # "Video: {video}\n"
    # "Window: [{start}, {end}) seconds\n"
    # "{title_line}"
    "Reference transcript (Provide a reference for the tone and on-screen information):\n{reference}\n\n"
    "Candidate A:\n{A}\n\n"
    "Candidate B:\n{B}\n\n"
)


def _mk_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_KEY")
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if base:
        return OpenAI(api_key=api_key, base_url=base)
    return OpenAI(api_key=api_key)

def judge_once(model: str, video: str, start: int, end: int, ref_text: str,
               cand_A: str, cand_B: str, title: str = "") -> str:
    client = _mk_client()
    title_line = f"Title: {title}\n" if title else ""
    msg = USER_TEMPLATE.format(
        # video=video, start=start, end=end, title_line=title_line,
        reference=ref_text.strip()[:8000],  # Simple length limit
        A=cand_A.strip()[:8000], B=cand_B.strip()[:8000]
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": msg}],
            temperature=0.0,
        )
    except Exception as e:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": msg}],
        )
    out = (resp.choices[0].message.content or "").strip().upper()
    return "A" if out.startswith("A") else "B"

def judge_pair(model: str, meta: Dict[str, Any],
               text_A: str, text_B: str) -> Tuple[str, str]:
    """
    First ask AB, then ask BA; ties are not allowed.
    Returns (vote_ab, vote_ba), each ∈ {'A','B'}.
    """
    video = meta["video"]
    start = meta["seg_start"]
    end = meta["seg_end"]
    title = meta.get("title", "")
    ref_text = meta.get("ref_text", "")
    
    vote_ab = judge_once(model, video, start, end, ref_text, text_A, text_B, title)
    vote_ba_raw = judge_once(model, video, start, end, ref_text, text_B, text_A, title)
    # BA's return is from (B,A) perspective, need to flip back to (A,B)
    vote_ba = "A" if vote_ba_raw == "B" else "B"
    return vote_ab, vote_ba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--seg_start", type=int, required=True)
    parser.add_argument("--seg_end", type=int, required=True)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--ref_text", type=str, default="")
    parser.add_argument("--cand_a", type=str, required=True)
    parser.add_argument("--cand_b", type=str, required=True)
    args = parser.parse_args()

    meta = dict(video=args.video, seg_start=args.seg_start, seg_end=args.seg_end,
                title=args.title, ref_text=args.ref_text)
    ab, ba = judge_pair(args.judge_model, meta, args.cand_a, args.cand_b)
    print({"vote_ab": ab, "vote_ba": ba})

if __name__ == "__main__":
    main()
