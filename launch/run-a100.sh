#!/bin/bash
# DeepSeek-V4-Flash-0731 on 4x CMP 170HX via haosdent/vllm @ dsv4-flash-a100.
# This is the FAST stack (2026-08-02): decode 74.7 tok/s aggregate, prefill ~900 t/s.
#
# Usage: run-a100.sh [--plain]     (--plain = no speculative decoding)
#
# Image: dsv4-a100:devel   (built from Dockerfile.devel in this directory)
# Model: dsv4-0731-orig    -- the ORIGINAL FP4+FP8 checkpoint, NOT our CT repack.
#        This branch handles the native checkpoint, so the repack is unnecessary.
#
# Two flags are load-bearing and were both learned the hard way:
#   --no-enable-flashinfer-autotune : flashinfer JIT is a HARD requirement at
#       engine init; it must be able to compile. See Dockerfile.devel for why the
#       base image has to carry a real CUDA toolkit.
#   --tokenizer-mode deepseek_v4    : from the working command in vllm#50576.
SPEC='--speculative-config {"method":"dspark","num_speculative_tokens":5}'
if [ "$1" = "--plain" ]; then SPEC=""; fi

docker stop -t 60 dsv4-a100 >/dev/null 2>&1
docker rm dsv4-a100 >/dev/null 2>&1

# The cards can be left unable to create CUDA contexts by a prior IMA. rmmod of
# nvidia_uvm clears even an Xid-154 "OS Reboot" wedge without rebooting the host.
if ! docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
        --entrypoint python3 dsv4-a100:devel \
        -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(4)]' \
        >/dev/null 2>&1; then
  echo "GPUs wedged -> recovering"
  nvidia-smi -r -i 0,1,2,3 >/dev/null
  rmmod nvidia_uvm; modprobe nvidia_uvm
  for g in 0 1 2 3; do nvidia-smi -i "$g" -pl 180 >/dev/null; done
fi

# shellcheck disable=SC2086
docker run -d --name dsv4-a100 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v ${DSV4_MODEL:-/models/DeepSeek-V4-Flash-0731}:/model --shm-size=16g -p 8098:8000 \
  dsv4-a100:devel vllm serve /model --served-model-name dsv4s \
  --tensor-parallel-size 4 --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 8192 --max-num-batched-tokens 2048 --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-num-seqs 8 \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  $SPEC >/dev/null
echo "launched dsv4-a100 (spec: ${SPEC:-none}) on :8098"
