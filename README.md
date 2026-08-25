# DeepSeek-V4-Flash-0731 SM80 vLLM Fork

This is a source-level vLLM fork for **DeepSeek-V4-Flash-0731**. The goal is to make DeepSeek-V4-Flash-0731 run reliably and as fast as possible on **SM80 / A100 / CMP 170HX** class GA100 GPUs, which do not have native FP8 Tensor Core support.

This is a full source tree. After cloning the repository, you can install, launch, and benchmark it in the same way as a normal vLLM source checkout.

```text
Target model: deepseek-ai/DeepSeek-V4-Flash-0731
Test hardware: 4 x NVIDIA CMP 170HX (GA100 sm_80, 65,536 MiB VRAM, PCIe Gen2 x4, no P2P), 4 x NVIDIA A800, and 8 x NVIDIA A100
Model path: /models/DeepSeek-V4-Flash-0731
Serving setup: TP1 + PP4, kv_cache_dtype=fp8, tokenizer_mode=deepseek_v4
```

Model weights are not included. You need to download the model weights separately and place them under your local model directory.

## Measured Benchmark Results

Unless otherwise noted, the results below are measured with the corresponding source code and scripts in this repository. Timing is end-to-end wall clock time. Throughput is counted by completion tokens, not by SSE chunks.

### 1,024 Input + 128 Output, Single Concurrency, 16 Requests, 5 Rounds

Script: `bench/offline_1024x128_c1_r16_5rounds_table.py`

Configuration:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
TP=1, PP=4
max_model_len=2048
max_num_seqs=16
max_num_batched_tokens=16384
kv_cache_dtype=fp8
enforce_eager=True
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_SPARSE_DENSE_QUERY_BLOCK=4
VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4
```

| Concurrency | Requests / round | 5-round output throughput tok/s | Median | Range | CV |
|---:|---:|---|---:|---:|---:|
| 1 | 16 | 121.18 / 134.54 / 136.65 / 138.05 / 139.07 | **136.65** | 121.18-139.07 | 5.46% |

Note: the first round includes more runtime noise and JIT after-effects. The last four rounds are stable at 134-139 tok/s. This test strictly fixes 1,024 input tokens and 128 output tokens, and uses `ignore_eos=True` to avoid misleading throughput caused by different output lengths across configurations.

### PP4 + DSpark End-to-End Serving Results

Hardware: 4 x CMP 170HX. Model: original DeepSeek-V4-Flash-0731 checkpoint.

| Metric | PP4 plain | PP4 + DSpark |
|---|---:|---:|
| decode, single stream, 3-content aggregate | 50.8 tok/s | **98.1 tok/s** |
| prefill @ 77k context | **5,272 tok/s** | 5,207 tok/s |
| aggregate decode @ 64 concurrent | 472.0 tok/s | **712.8 tok/s** |
| verified one-shot context | **1,047,736 tokens** | **1,047,736 tokens** |

### Decode by Content Type

400 output tokens, temperature=0.

| Prompt type | PP4 plain | PP4 + DSpark | Speedup |
|---|---:|---:|---:|
| technical exposition | 44.0 | 86.9 | 1.98x |
| open-ended prose | 55.1 | 93.9 | 1.70x |
| code generation | 55.1 | 118.8 | 2.16x |
| aggregate | **50.8** | **98.1** | **1.93x** |

### Concurrency Throughput

| concurrent requests | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| PP4 plain | 50.5 | 133.0 | 173.9 | 288.2 | 393.3 | 472.0 |
| PP4 + DSpark | **58.3** | **197.9** | **269.3** | 302.3 | **429.2** | **712.8** |

### Prefill: PP4 Is Better Suited for Long Context Than TP4

Single-request warm run, `--max-model-len 131072`.

| prompt tokens | 1,544 | 3,082 | 6,159 | 12,313 | 24,621 | 50,006 | 76,929 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PP4 | 1,966 | 2,706 | 3,660 | 4,524 | 5,113 | **5,321** | 5,272 |
| TP4 | 908 | 774 | 841 | 809 | 804 | 776 | 801 |
| PP/TP | 2.2x | 3.5x | 4.4x | 5.6x | 6.4x | **6.9x** | 6.6x |

### Long-Context Decode / Prefill

After the row-chunk context-ceiling fix, one-shot prefill was verified at million-token-scale context.

| real context | prefill tok/s | TTFT | decode tok/s |
|---|---:|---:|---:|
| ~7,700 | - | 2.1 s | **88.7** |
| ~100,000 | - | 22.1 s | 79.0 |
| ~200,000 | 4,486 | 50.5 s | 67.7 |
| ~385,000 | 3,425 | 120.4 s | 54.5 |
| ~769,000 | 2,525 | 334-336 s | 39.6 / 43.6 |
| **~1,040,000** | **1,904** | **544-550 s** | **35.6** |

Conclusion: as context grows from about 7.7k to about 1.04M, decode throughput drops from 88.7 tok/s to 35.6 tok/s, retaining about 40% of short-context decode performance. At million-token context, the main cost is prefill, not subsequent decode.

### Accumulated Conversation vs One-Shot Prefill

| mode | ROW_CHUNK | depth | outcome |
|---|---:|---:|---|
| accumulated chat | 128 | 733,120 | CUDA illegal memory access |
| accumulated chat repeat | 128 | 718,216 | same fault, same PP rank |
| one-shot prefill | 128 | 749,534 / 745,427 | pass |
| one-shot prefill | 128 | 1,047,736 | pass |
| accumulated chat | **64** | **1,002,852** | **405 turns, no crash** |

This result shows that long-context one-shot prefill and real multi-turn prefix-cache accumulated conversation stress different execution paths. `ROW_CHUNK=64` is more stable for real multi-turn scenarios.

### FP8 Shared-Expert Dequant Verification

This path was used to verify whether FP8 in dense/shared expert layers should be dequantized to BF16 before computation.

| Configuration | prefill+decode tok/s | decode tok/s | per-request decode tok/s |
|---|---:|---:|---:|
| shared dequant verify2 | 1079.03 | 63.47 | 7.93 |
| include-selected dequant | **1367.09** | **80.42** | **10.05** |

Corresponding environment variables:

```bash
export VLLM_MARLIN_FP8_DEQUANT_BF16=1
export VLLM_MARLIN_FP8_DEQUANT_INCLUDE=shared_experts
export VLLM_MARLIN_FP8_DEQUANT_EXCLUDE=
```

This switch is not a forced default path. It is used to selectively test which FP8 Marlin linear layers are better suited for FP8-to-BF16 dequantization on SM80.

## Quick Start

### 1. Clone the Source

```bash
git clone https://github.com/yunyinbanfu/deepseek-v4-flash-0731-sm80.git
cd deepseek-v4-flash-0731-sm80
```

### 2. Install

Install from source:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

You need a CUDA / PyTorch / compiler environment that matches your machine. This repository does not commit `.so` files, wheels, virtual environments, model weights, or profiling data.

### 3. Launch PP4 Serving

By default, the launcher uses GPU `0,1,2,3` and model path `/models/DeepSeek-V4-Flash-0731`:

```bash
bash launch/run-pp-dspark.sh
```

You can override them with environment variables:

```bash
DSV4_MODEL=/path/to/DeepSeek-V4-Flash-0731 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DSV4_PORT=8000 \
bash launch/run-pp-dspark.sh
```

Default script settings:

```bash
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_SPARSE_DENSE_QUERY_BLOCK=4
VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

To disable DSpark:

```bash
DSV4_DISABLE_DSPARK=1 bash launch/run-pp-dspark.sh
```

### 4. Run the 1,024 x 128 Five-Round Benchmark

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=$PWD \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_SPARSE_DENSE_QUERY_BLOCK=4 \
VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python bench/offline_1024x128_c1_r16_5rounds_table.py
```

Other scripts:

```text
bench/offline_1024x128_5rounds.py
bench/offline_pp4_c1_r16_1024x256.py
bench/offline_pp4_1024x64_conc8_dequant_shared.py
```

Full historical results and test notes:

```text
RESULTS.cmp170hx.md
SETTINGS.cmp170hx.md
patches.cmp170hx/
```

## Important Benchmarking Lessons

1. Do not calculate throughput by counting SSE chunks. Under speculative decoding, one chunk may contain multiple tokens. Count `completion_tokens` or actual token IDs instead.
2. Compared configurations must generate the same number of tokens. For benchmarks, using `ignore_eos=True` is recommended.
3. The first round after service startup may include Triton JIT/warmup noise. Inspect it separately or discard it.
4. Prefix caching can make repeated prompts skip prefill. Disable it or vary prompts when measuring prefill.
5. Do not estimate decode speed by subtracting `max_tokens=1` from `max_tokens=N`. Long-context prefill jitter can amplify the error.
6. One-shot prefill and accumulated conversation are not the same. Passing a million-token one-shot prefill does not guarantee that multi-turn prefix-cache serving will be stable.

## What Is Not Included

This repository does not include:

```text
Model weights
Docker images
Python virtual environments
Compiled .so / wheel artifacts
Nsight / sqlite profiling data
Runtime logs
```

`.gitignore` excludes these files to keep the repository cloneable.

## License

This fork follows the upstream vLLM license. See [LICENSE](LICENSE).
