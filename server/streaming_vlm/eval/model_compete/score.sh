#!/usr/bin/env bash
set -euo pipefail

# ===== Manual Configuration Section =====
MODEL1_NAME='mit-han-lab/StreamingVLM'   
MODEL2_NAME="gpt-4o-mini" 
JUDGE_MODEL="gpt-5"

export EVAL_DATASET_PATH="/NFS/raid0/dataset/streaming_vlm_eval"
SCORE_METADATA="${EVAL_DATASET_PATH}/temp_score_clips.jsonl"
SAFE_MODEL1_PATH=${MODEL1_NAME//\//_}
SAFE_MODEL2_PATH=${MODEL2_NAME//\//_}
RESULT_A="results/model_compete/model_answers/${SAFE_MODEL1_PATH}/merged_result.jsonl"
RESULT_B="results/model_compete/model_answers/${SAFE_MODEL2_PATH}/merged_result.jsonl"

DEVICE_NUM=8   

PAIR_TAG="${SAFE_MODEL1_PATH}_VS_${SAFE_MODEL2_PATH}"
OUT_DIR="results/model_compete/model_scores/${PAIR_TAG}"
mkdir -p "${OUT_DIR}"

TOTAL_LINE=$(wc -l < "${SCORE_METADATA}")

pids=()
for i in $(seq 0 $((DEVICE_NUM - 1))); do
  start=$(( i * TOTAL_LINE / DEVICE_NUM ))
  end=$(( (i + 1) * TOTAL_LINE / DEVICE_NUM ))

  python streaming_vlm/eval/model_compete/score_segments.py \
    --merged_a "${RESULT_A}" \
    --merged_b "${RESULT_B}" \
    --score_metadata "${SCORE_METADATA}" \
    --judge_model "${JUDGE_MODEL}" \
    --start_line "${start}" \
    --end_line "${end}" \
    --out_path "${OUT_DIR}/score_segments_${i}.jsonl" &

  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python streaming_vlm/eval/model_compete/merge_score.py --pair_dir $OUT_DIR


echo "Done. Outputs in ${OUT_DIR}"
