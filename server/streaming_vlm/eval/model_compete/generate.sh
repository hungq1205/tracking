#!/usr/bin/env bash
set -euo pipefail

# ========= Default Parameters (can be overridden by command line) =========
# MODEL_PATH_DEFAULT='chenjoya/LiveCC-7B-Instruct'
# MODEL_BASE_DEFAULT='LiveCC'
MODEL_PATH_DEFAULT='mit-han-lab/StreamingVLM'
MODEL_BASE_DEFAULT='Qwen'
DURATION_DEFAULT=10000
TEMPERATURE_DEFAULT=0.9

MODEL_PATH="$MODEL_PATH_DEFAULT"
MODEL_BASE="$MODEL_BASE_DEFAULT"
DURATION=$DURATION_DEFAULT
TEMPERATURE=$TEMPERATURE_DEFAULT


export EVAL_DATASET_PATH="/NFS/raid0/dataset/streaming_vlm_eval"
EVAL_METADATA="${EVAL_DATASET_PATH}/eval_all_metadata.jsonl"
START_TIME=1000
QUERY="Please describe the video."
QUERY="Act as a live sports broadcaster: in short present-tense lines, call the on-screen play first, then one brief courtside insight; vary your energy with the moment and include visible score/time.
Stay perfectly synced to the video—only describe what’s visible, never invent names or facts (use positions/numbers if unknown), and immediately correct yourself if the scoreboard contradicts you."
DEVICE_NUM=8

# ========= Command Line Parameter Parsing =========
# Usage: ./generate.sh --model-path PATH --model-base BASE
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path|-m)
      MODEL_PATH="${2}"; shift 2;;
    --model-base|-b)
      MODEL_BASE="${2}"; shift 2;;
    --duration|-d)
      DURATION="${2}"; shift 2;;
    --temperature|-t)
      TEMPERATURE="${2}"; shift 2;;
    -h|--help)
      echo "Usage: $0 [--model-path PATH] [--model-base BASE] [--duration DURATION] [--temperature TEMPERATURE]"; exit 0;;
    --)
      shift; break;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

SAFE_MODEL_PATH=${MODEL_PATH//\//_}
OUT_DIR="results/model_compete/model_answers/${SAFE_MODEL_PATH}"

# Line count (assuming one sample per line)
TOTAL_LINE=$(wc -l < "$EVAL_METADATA")

# Ensure output directory exists before parallel execution
mkdir -p "$OUT_DIR"

pids=()
for i in $(seq 0 $((DEVICE_NUM - 1))); do
  start=$(( i * TOTAL_LINE / DEVICE_NUM ))
  end=$(( (i + 1) * TOTAL_LINE / DEVICE_NUM ))   # Half-open interval: last line index is end-1

  CUDA_VISIBLE_DEVICES=$i \
  python streaming_vlm/eval/model_compete/generate_segments.py \
    --model_path "$MODEL_PATH" \
    --model_base "$MODEL_BASE" \
    --eval_metadata "$EVAL_METADATA" \
    --start_time "$START_TIME" \
    --duration "$DURATION" \
    --temperature "$TEMPERATURE" \
    --start_line "$start" \
    --end_line "$end" \
    --out_dir "$OUT_DIR/result_${i}.jsonl" \
    --query "$QUERY" &

  pids+=($!)
done

# Wait for all parallel subprocesses to finish
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "All shards finished."

# merge result
python streaming_vlm/eval/model_compete/merge_result.py \
  --shard_dir "results/model_compete/model_answers/${SAFE_MODEL_PATH}"
