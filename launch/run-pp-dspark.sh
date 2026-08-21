#!/usr/bin/env bash
set -euo pipefail

# Source-tree launcher for DeepSeek-V4-Flash-0731 on SM80/A100-class GPUs.
# No Docker is required. Run from any directory after installing this checkout.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL="${DSV4_MODEL:-/models/DeepSeek-V4-Flash-0731}"
HOST="${DSV4_HOST:-0.0.0.0}"
PORT="${DSV4_PORT:-8000}"
MAXLEN="${DSV4_MAXLEN:-2048}"
MAX_NUM_SEQS="${DSV4_MAX_NUM_SEQS:-16}"
MAX_BATCHED_TOKENS="${DSV4_MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${DSV4_GPU_MEMORY_UTILIZATION:-0.98}"
SPEC_TOKENS="${DSV4_SPEC_TOKENS:-5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_SPARSE_DENSE_QUERY_BLOCK="${VLLM_SPARSE_DENSE_QUERY_BLOCK:-4}"
export VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE="${VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SPEC_ARGS=()
if [ "${DSV4_DISABLE_DSPARK:-0}" != "1" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
fi

cd "$REPO_ROOT"
exec python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --served-model-name "${DSV4_SERVED_MODEL_NAME:-deepseek-v4-flash-0731}" \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${SPEC_ARGS[@]}"
