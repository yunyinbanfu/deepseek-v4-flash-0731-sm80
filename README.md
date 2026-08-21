<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width="50%">
  </picture>
</p>

<h2 align="center">DeepSeek-V4-Flash-0731 SM80 vLLM Fork</h2>

<p align="center">
  A source-level vLLM fork for running DeepSeek-V4-Flash-0731 on SM80/A100-class GPUs with PP4 + DSpark, sparse MLA/indexer fallback, and SM80-oriented kernel fixes.
</p>

---

## What this repository is

This repository is a full vLLM source tree, not a Docker wrapper or a patch-only package. It is intended to be cloned, built, and launched like a normal vLLM checkout.

Model weights are not included. Put the model under a local path such as:

```bash
/srv/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

Upstream/reference context:

- Base project: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- Development base used for this fork: `haosdent/vllm` branch `dsv4-flash-a100`, around commit `12810046c`
- Target model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Target tested hardware class: NVIDIA SM80 / A100-class GPUs, including CMP 170HX-style environments

The original upstream README is kept as [README.vllm.md](README.vllm.md).

## Main changes in this fork

- DeepSeek-V4-Flash PP4 serving path: keeps the framework source directly in-tree and enables the PP + DSpark path used by the tested setup.
- DSpark + pipeline parallelism support: adapts draft model helpers and pipeline communication so speculative serving can work across PP ranks.
- Sparse MLA / sparse indexer SM80 fallback: adds row-chunked top-k/indexer fallback paths for long-context DeepSeek-V4-Flash workloads on SM80 where the newer kernel path is not available.
- Marlin MoE deterministic token ordering: applies the PR #52532 style fix by canonicalizing `sorted_token_ids` before the Marlin MoE kernel, avoiding output drift caused by semantically equivalent but physically different routed layouts.
- FP8 shared-expert dequant control: adds environment switches for selectively routing FP8 Marlin linear layers through BF16 dequant/cuBLAS paths when this is faster or easier to profile on the tested SM80 setup.
- Local launch and benchmark scripts: includes the scripts used for PP4 launch and offline throughput checks.

Key files to inspect:

```text
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/pp_utils.py
vllm/v1/worker/gpu/spec_decode/dspark/utils.py
vllm/model_executor/layers/sparse_attn_indexer.py
vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
vllm/model_executor/kernels/linear/scaled_mm/marlin.py
vllm/model_executor/layers/quantization/fp8.py
vllm/envs.py
launch/
bench/
patches.cmp170hx/
```

## Build from source

A typical editable install is:

```bash
git clone https://github.com/yunyinbanfu/deepseek-v4-flash-0731-sm80.git
cd deepseek-v4-flash-0731-sm80

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Use the CUDA/PyTorch stack that matches your machine. This fork was prepared for local source builds in a vLLM development environment; prebuilt binary artifacts are intentionally not committed.

## PP4 launch example

Example GPU selection used in local testing:

```bash
cd /path/to/deepseek-v4-flash-0731-sm80

CUDA_VISIBLE_DEVICES=0,2,3,4 \
PYTHONPATH=$PWD \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_SPARSE_DENSE_QUERY_BLOCK=4 \
VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model /srv/models/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.98
```

See [launch/run-pp-dspark.sh](launch/run-pp-dspark.sh) and [launch/run-a100.sh](launch/run-a100.sh) for local launch templates.

## Offline benchmark examples

Single-concurrency table-style test, fixed 1,024 input tokens and 128 output tokens:

```bash
cd /path/to/deepseek-v4-flash-0731-sm80

CUDA_VISIBLE_DEVICES=0,2,3,4 \
PYTHONPATH=$PWD \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_SPARSE_DENSE_QUERY_BLOCK=4 \
VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python bench/offline_1024x128_c1_r16_5rounds_table.py
```

Other local benchmark/profiling entry points:

```text
bench/offline_1024x128_5rounds.py
bench/offline_pp4_c1_r16_1024x256.py
bench/offline_pp4_1024x64_conc8_dequant_shared.py
```

Historical local results and configuration notes are kept in:

```text
RESULTS.cmp170hx.md
SETTINGS.cmp170hx.md
```

## Useful environment switches

```bash
# Sparse dense/indexer row blocking used in the tested PP4 path
export VLLM_SPARSE_DENSE_QUERY_BLOCK=4
export VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4

# Optional FP8 Marlin dequant route controls
export VLLM_MARLIN_FP8_DEQUANT_BF16=1
export VLLM_MARLIN_FP8_DEQUANT_INCLUDE=shared_experts
export VLLM_MARLIN_FP8_DEQUANT_EXCLUDE=
```

The FP8 dequant switches are intentionally opt-in. They are useful for testing whether a selected FP8 linear path should stay on Marlin or be dequantized to BF16 before GEMM on the local SM80 setup.

## Notes and caveats

- This repository does not include model weights, tokenizer files, generated build products, CUDA `.so` files, profiler databases, or Docker images.
- Build artifacts such as `*.so`, `*.abi3.so`, `build/`, `dist/`, `.venv/`, logs, and profiler outputs are ignored.
- The Marlin MoE canonicalization fix is applied at the `fused_marlin_moe()` wrapper level. It makes wrapper-level raw-vs-canonical routed layouts produce identical BF16 outputs; it does not change the low-level Marlin kernel reduction behavior itself.
- For general vLLM usage, documentation, and citation, see [README.vllm.md](README.vllm.md) and the upstream [vLLM documentation](https://docs.vllm.ai/).

## License

This fork keeps the upstream vLLM license. See [LICENSE](LICENSE).
