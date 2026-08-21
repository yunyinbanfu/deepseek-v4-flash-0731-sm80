#!/bin/bash
# ★ BEST CONFIG: DeepSeek-V4-Flash-0731 on 4x CMP 170HX, PP4 *with* DSpark.
#   decode 98.1 tok/s aggregate | prefill ~5,300 t/s | large KV pool
#
# Usage: run-pp-dspark.sh [--plain]      (--plain = PP4 without DSpark, 50.8 t/s)
#        run-pp-dspark.sh --maxlen 131072
#
# DSpark+PP is NOT supported upstream -- it is enabled by seven local patches,
# applied here by BIND-MOUNT (the image installs vLLM with `pip install -e .`, so
# /vllm/vllm/... is live code and no rebuild is needed):
#
#   1 config/speculative.py                      draft_parallel_config.pipeline_parallel_size = 1
#                                                (the draft is not pipelined; it runs whole on the
#                                                 last rank, so it must not be asked for SupportsPP)
#   2 v1/worker/gpu/pp_utils.py                  + broadcast_draft() and the matching recv, i.e.
#                                                 vLLM PR #46994's spec-decode-over-PP relay, which
#                                                 is NOT in upstream main
#   3 v1/worker/gpu/model_runner.py              drop the dspark PP guard; call broadcast_draft()
#                                                 after propose(); scatter relayed draft tokens on
#                                                 non-last ranks
#   4 v1/worker/gpu/spec_decode/dspark/utils.py  drop NotImplementedError; load the draft's token
#                                                 embedding from the checkpoint (the target's lives
#                                                 on PP rank 0, the drafter on the last rank)
#   5 model_executor/layers/sparse_attn_indexer.py  persistent_topk capability gate (0001;
#                                                 precautionary, the report was retracted)
#   6 model_executor/layers/sparse_attn_indexer.py  prefill top-k torch fallback (0005a) --
#                                                 REQUIRED on sm_80: the CUDA top-k can emit
#                                                 uninitialised indices and kill a worker with
#                                                 Xid 31 above ~128k prefill
#   7 model_executor/layers/sparse_attn_indexer.py  row-chunk the [M,N] logits transient (0006)
#                                                 -- THE context-ceiling fix, and it is ENV-GATED;
#                                                 see DSV4_ROW_CHUNK below or it stays inert
#                                                 silent corruption at prompt len 2049-4096)
#
# Why it works at all: for DeepSeek-V4 the DSpark aux-hidden-state taps
# (dspark_target_layer_ids = [40,41,42] of 43 layers) AND lm_head both land on the
# last PP rank, which is where the model runner already builds the drafter.
# ---- configure these three for your machine -----------------------------------
IMG="${DSV4_IMAGE:-dsv4-a100:devel}"
# VLLM_SRC: the vllm/ package directory of your PATCHED checkout. The image installs
# vLLM with `pip install -e .`, so bind-mounting these five files applies the patches
# with no rebuild. If you applied the patches at build time instead, set
# DSV4_NO_MOUNT=1 and the mounts are skipped.
R="${DSV4_VLLM_SRC:-/opt/vllm-dsv4/vllm}"
MODEL="${DSV4_MODEL:-/models/DeepSeek-V4-Flash-0731}"
# -------------------------------------------------------------------------------
MAXLEN="${DSV4_MAXLEN:-32768}"    # with ROW_CHUNK set, 393216 and 1048576 both work
# Patch 0006 is ENV-GATED AND DEFAULTS TO OFF (0). Without this the context-ceiling fix is
# installed but INERT and you hit the old ~134k wall. Set 0 to reproduce the unpatched
# upstream path byte-for-byte.
#
# ONE-SHOT prefill:  256 reaches ~957k, 128 reaches the full 1,047,736.
# ACCUMULATING CHAT: 128 dies at ~718-733k with a CUDA illegal memory access (reproduced
#                    twice); 64 reached 1,002,852 over 405 turns. A one-shot prefill at the
#                    same depth is fine, so needle tests never catch this.
# Defaulting to 64: it is the safe value for both, and costs ~5% TTFT at 750k.
ROW_CHUNK="${DSV4_ROW_CHUNK:-64}"
SPEC='--speculative-config {"method":"dspark","num_speculative_tokens":5}'

while [ $# -gt 0 ]; do
  case "$1" in
    --plain)  SPEC=""; shift ;;
    --maxlen) MAXLEN="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

docker stop -t 60 dsv4-a100 >/dev/null 2>&1
docker rm dsv4-a100 >/dev/null 2>&1

# A prior illegal memory access can leave the cards unable to create CUDA
# contexts. rmmod of nvidia_uvm clears even an Xid-154 "OS Reboot" wedge without
# rebooting the host; the reset drops the power cap, so re-apply 180 W.
if ! docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
        --entrypoint python3 "$IMG" \
        -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(4)]' \
        >/dev/null 2>&1; then
  echo "GPUs wedged -> recovering"
  nvidia-smi -r -i 0,1,2,3 >/dev/null 2>&1
  rmmod nvidia_uvm; modprobe nvidia_uvm
  for g in 0 1 2 3; do nvidia-smi -i "$g" -pl 180 >/dev/null; done
fi

MOUNTS=""
if [ -z "${DSV4_NO_MOUNT:-}" ]; then
  for f in config/speculative.py \
           v1/worker/gpu/pp_utils.py \
           v1/worker/gpu/model_runner.py \
           v1/worker/gpu/spec_decode/dspark/utils.py \
           model_executor/layers/sparse_attn_indexer.py; do
    if [ ! -f "$R/$f" ]; then
      echo "ERROR: $R/$f not found. Set DSV4_VLLM_SRC to the vllm/ directory of your"
      echo "patched checkout, or set DSV4_NO_MOUNT=1 if the patches are baked into \$IMG."
      exit 1
    fi
    MOUNTS="$MOUNTS -v $R/$f:/vllm/vllm/$f:ro"
  done
fi

# shellcheck disable=SC2086
docker run -d --name dsv4-a100 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DSV4_LOGITS_ROW_CHUNK="$ROW_CHUNK" \
  -v "$MODEL":/model \
  $MOUNTS \
  --shm-size=16g -p 8098:8000 \
  "$IMG" vllm serve /model --served-model-name dsv4s \
  --pipeline-parallel-size 4 --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len "$MAXLEN" --max-num-batched-tokens 2048 --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-num-seqs 8 \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  $SPEC >/dev/null
echo "launched dsv4-a100 on :8098  (maxlen $MAXLEN, row_chunk $ROW_CHUNK, spec: ${SPEC:-none})"
