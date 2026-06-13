#!/usr/bin/env bash
set -euo pipefail

# ===== Configuration =====
MODEL_NAME="gpt-4o-mini"
export EVAL_DATASET_PATH="/NFS/raid0/dataset/streaming_vlm_eval"
SCORE_METADATA="${EVAL_DATASET_PATH}/score_all_clips.jsonl"

WORKERS=64
FPS=0.25
QUERY="Continue the commentary seamlessly from the previous segment, matching its voice, vocabulary, and pacing. Call real-time play-by-play with brief, in-the-moment analysis in a courtside broadcast voice, varying emotion and tempo. Stay perfectly synchronized with the visuals—state only on-screen, confirmed details (score/clock/fouls only if visible; use neutral identifiers if unsure) and self-correct instantly if a read proves wrong."
QUERY="Describe the video."

RESULTS_DIR="results/model_compete/model_answers"

# Need to set in advance: export OPENAI_KEY=sk-xxxx
: "${OPENAI_KEY:?Please export OPENAI_KEY}";

python streaming_vlm/eval/model_compete/generate_gpt.py \
  --model_name "${MODEL_NAME}" \
  --score_metadata "${SCORE_METADATA}" \
  --video_root "${EVAL_DATASET_PATH}" \
  --workers ${WORKERS} \
  --fps ${FPS} \
  --query "${QUERY}" \
  --results_dir "${RESULTS_DIR}"
