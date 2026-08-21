# Framework Source Files

This directory contains the actual framework source files used in the local SM80 DeepSeek-V4-Flash experiments. Files are copied with their original vLLM-relative paths so they can be inspected directly on GitHub.

## `new-tmp-vllm/`

Source files copied from:

```text
/home/lxk/new-tmp/tmp/vllm
```

This tree contains the PP + DSpark enablement, sparse indexer long-context fixes, SM80 prefill top-k fallback, and optional FP8 Marlin -> BF16/cuBLAS shared-expert dense-linear route.

Key files:

```text
vllm/config/speculative.py
vllm/envs.py
vllm/model_executor/kernels/linear/scaled_mm/marlin.py
vllm/model_executor/layers/quantization/fp8.py
vllm/model_executor/layers/sparse_attn_indexer.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/pp_utils.py
vllm/v1/worker/gpu/spec_decode/dspark/utils.py
```

## `tmp-vllm-pr52532/`

Source files copied from:

```text
/home/lxk/tmp/vllm
```

This tree contains the later PR #52532 Marlin MoE token-order canonicalization fix that was validated separately.

Key files:

```text
vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
tests/kernels/moe/test_moe.py
```

## Why not upload the whole `/home/lxk/new-tmp` directory?

The raw directory contains git metadata, local build outputs, virtual environments, logs, nsys/ncu profiles and other large machine-local artifacts. This repository keeps the source files and reproducibility materials while excluding model weights and generated artifacts.
