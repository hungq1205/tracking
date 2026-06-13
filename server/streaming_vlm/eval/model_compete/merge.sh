#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_merge.sh chenjoya/LiveCC-7B-Instruct
#   or pass no parameters to use default model

MODEL_NAME="${1:-chenjoya/LiveCC-7B-Instruct}"

SAFE_MODEL_PATH=${MODEL_NAME//\//_}

python streaming_vlm/eval/model_compete/merge_result.py \
  --shard_dir "results/model_compete/model_answers/${SAFE_MODEL_PATH}"
