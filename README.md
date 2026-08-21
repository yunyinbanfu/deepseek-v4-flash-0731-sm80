# DeepSeek-V4-Flash-0731 on SM80 GPUs

This repository packages the modified framework source files, patches, launch scripts and benchmark harnesses used to run `deepseek-ai/DeepSeek-V4-Flash-0731` on SM80-class NVIDIA GPUs such as A100 / A800 / CMP 170HX.

The target model is DeepSeek-V4-Flash-0731, roughly 284B total parameters with about 13B active MoE parameters. The tested hardware was 4 x NVIDIA CMP 170HX, GA100 silicon, 64 GB HBM each, PCIe Gen2 x4, no P2P/NVLink. The same direction is intended for SM80 A100-like cards where native FP8 math is unavailable but BF16 tensor cores and Marlin paths are available.

## What This Adds

- Provides the actual modified vLLM framework source files under `framework/`, copied from the working server trees.
- Enables DeepSeek-V4-Flash serving on SM80 using the `haosdent/vllm:dsv4-flash-a100` branch as the base.
- Enables DSpark speculative decoding under pipeline parallelism, which vLLM originally rejected for this path.
- Fixes long-context sparse MLA indexer memory pressure by row-chunking the `[M, N]` FP32 logits temporary.
- Adds an SM80 prefill top-k torch fallback to avoid invalid top-k indices in long-context sparse indexer paths.
- Fixes Marlin MoE routed-layout determinism at the wrapper level by canonicalizing `sorted_token_ids` before entering the Marlin MoE kernel.
- Provides an optional FP8 block-weight to BF16 pre-dequant route for selected shared-expert dense linear layers on SM80.
- Ships launch scripts and benchmark harnesses used for decode, prefill, concurrency, long-context and correctness tests.

## Measured Highlights

These are end-to-end measurements from the local test environment. See `results/RESULTS.cmp170hx.md` for full caveats and tables.

| Item | Result |
|---|---:|
| PP4 + DSpark single-stream decode aggregate | 98.1 tok/s |
| PP4 plain single-stream decode aggregate | 50.8 tok/s |
| DSpark speedup under PP | 1.93x |
| PP4 prefill at ~77k context | ~5,300 tok/s |
| PP vs TP prefill at ~77k on PCIe Gen2 x4/no-P2P cards | ~6.6x |
| PP4 + DSpark aggregate decode at 64 concurrent requests | 712.8 tok/s |
| Verified one-shot context | 1,047,736 tokens |
| 1024 input / 128 output / 16 requests in one batch | median 136.65 tok/s |
| Optional shared-expert FP8->BF16 dequant, 1024x64 c=8 | ~45 tok/s -> 63-80 tok/s |

## Repository Layout

```text
framework/
  README.md
  new-tmp-vllm/        actual modified source files from /home/lxk/new-tmp/tmp/vllm
  tmp-vllm-pr52532/    later Marlin MoE PR #52532 source files from /home/lxk/tmp/vllm
patches/
  0001-pp-dspark-long-context-sm80.patch
  0002-marlin-moe-token-order-canonicalization.patch
  optional/0003-fp8-marlin-bf16-dequant-include-nvtx.patch
launch/
  run-pp-dspark.sh
  run-a100.sh
bench/
  benchmark harnesses and offline scripts
results/
  measured result notes and setting rationale
```

## Framework Source Files

The source files are copied with their original vLLM-relative paths so they can be inspected directly on GitHub or overlaid onto a vLLM checkout.

Main working tree:

```text
framework/new-tmp-vllm/
```

This was copied from:

```text
/home/lxk/new-tmp/tmp/vllm
```

It contains the PP + DSpark enablement, sparse indexer long-context fixes, SM80 prefill top-k fallback, and optional FP8 Marlin -> BF16/cuBLAS shared-expert dense-linear route.

Later Marlin MoE consistency fix:

```text
framework/tmp-vllm-pr52532/
```

This was copied from:

```text
/home/lxk/tmp/vllm
```

It contains the later PR #52532 token-order canonicalization source and regression test.

The whole `/home/lxk/new-tmp` directory is not uploaded raw because it also contains git metadata, local build outputs, virtual environments, logs, nsys/ncu profiles and other generated artifacts.

## Base vLLM

The source files and patches were prepared against:

```bash
git clone --branch dsv4-flash-a100 https://github.com/haosdent/vllm.git
cd vllm
git checkout 12810046c
```

Commit `12810046c` is titled:

```text
DSv4 SM80: DeepSeek-V4-Flash on A100 -- sparse MLA enablement, kernel tuning, and serving optimization
```

## Use The Source Files

To overlay the framework files directly onto a vLLM checkout:

```bash
cd /path/to/vllm
rsync -a /path/to/deepseek-v4-flash-0731-sm80/framework/new-tmp-vllm/ ./
```

To also overlay the later Marlin MoE PR #52532 consistency fix:

```bash
rsync -a /path/to/deepseek-v4-flash-0731-sm80/framework/tmp-vllm-pr52532/ ./
```

Alternatively, apply the patch files:

```bash
cd /path/to/vllm
git apply /path/to/deepseek-v4-flash-0731-sm80/patches/0001-pp-dspark-long-context-sm80.patch
git apply /path/to/deepseek-v4-flash-0731-sm80/patches/0002-marlin-moe-token-order-canonicalization.patch
```

Optional shared-expert FP8->BF16 dense-linear routing patch:

```bash
git apply /path/to/deepseek-v4-flash-0731-sm80/patches/optional/0003-fp8-marlin-bf16-dequant-include-nvtx.patch
```

Use the optional patch only after profiling. It is beneficial for selected shared expert dense layers, but should not be enabled globally for all FP8 linears.

## Launch

The main serving profile is pipeline-parallel DSpark:

```bash
/path/to/deepseek-v4-flash-0731-sm80/launch/run-pp-dspark.sh
```

Important defaults:

- `--pipeline-parallel-size 4`
- `--kv-cache-dtype fp8`
- `--tokenizer-mode deepseek_v4`
- `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`
- `DSV4_LOGITS_ROW_CHUNK=64` for long conversations, `128` or `256` for one-shot document prefill profiles

Model weights are not included in this repository. Point the launch script at a local copy of:

```text
deepseek-ai/DeepSeek-V4-Flash-0731
```

## Offline Benchmark Example

The offline script used for the 1024 input / 128 output / 16 requests batch test is included at:

```text
bench/offline_1024x128_c1_r16_5rounds_table.py
```

Run pattern:

```bash
cd /path/to/vllm
CUDA_VISIBLE_DEVICES=0,2,3,4 \
PYTHONPATH=/path/to/vllm \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_SPARSE_DENSE_QUERY_BLOCK=4 \
VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4 \
/path/to/python bench/offline_1024x128_c1_r16_5rounds_table.py
```

This benchmark intentionally uses `max_num_seqs=req_per_round` for the row labeled `conc=1, req_per_round=16`, meaning one submitted batch containing 16 schedulable requests.

## Correctness Notes

### Marlin MoE Token Order Canonicalization

The Marlin MoE fix does not change the low-level `moe_wna16_marlin_gemm` reduction behavior. Instead, it canonicalizes `sorted_token_ids` in the `fused_marlin_moe()` wrapper after `moe_align_block_size()` and before entering the Marlin kernel. This makes semantically equivalent routed layouts enter the kernel with the same physical token order.

Regression summary from the local verification:

```text
before wrapper canonicalization: differing_bf16=28, max_ulp=13
after wrapper canonicalization:  differing_bf16=0,  max_ulp=0
```

The single-token decode fast path is preserved.

### Sparse Indexer Long Context

The long-context fix row-chunks the sparse indexer's `[M, N]` FP32 logits temporary. Rows are independent for top-k, so this is exact, not an approximation. In the tested setup this removed the practical bug wall around ~134k tokens and reached the model's 1M context limit for one-shot prefill.

## Known Limits

- At 1M context, time to first token is minutes, not interactive.
- Retrieval accuracy degrades with depth even when the full window is reachable. Treat 1M as a large working set, not a reliable database.
- DSpark output is not bit-reproducible at temperature 0. This was also observed on upstream TP DSpark and is not specific to the PP enablement patch.
- The optional FP8->BF16 dequant path spends more VRAM and should be restricted with include patterns such as `shared_experts.gate_up_proj,shared_experts.down_proj`.

## License

This repository contains modified source files, patches and scripts around vLLM. vLLM is Apache-2.0 licensed; retain upstream notices when applying or redistributing patched source.
