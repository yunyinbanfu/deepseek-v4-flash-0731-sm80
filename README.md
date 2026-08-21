# DeepSeek-V4-Flash-0731 SM80 vLLM Fork

这是一个面向 **DeepSeek-V4-Flash-0731** 的 vLLM 源码级 fork，目标是在 **SM80 / A100 / CMP 170HX** 这类没有原生 FP8 Tensor Core 的 GA100 系 GPU 上，把 DeepSeek-V4-Flash-0731 跑通并尽量跑快。

完整源码树，clone 后可以按 vLLM 源码方式安装、启动和测试。

```text
目标模型: deepseek-ai/DeepSeek-V4-Flash-0731
测试硬件: 4 x NVIDIA CMP 170HX, GA100 sm_80, 65,536 MiB VRAM, PCIe Gen2 x4, no P2P
模型路径: /srv/models/deepseek-ai/DeepSeek-V4-Flash-0731
部署形态: TP1 + PP4, kv_cache_dtype=fp8, tokenizer_mode=deepseek_v4
```

模型权重需要自行下载到本地模型目录。

## 做了什么

这个 fork 主要解决的是：DeepSeek-V4-Flash-0731 在 SM80/A100 级别 GPU 上，使用 vLLM 跑 PP4 serving、DSpark speculative decoding、稀疏 MLA/indexer、FP8/Marlin/MoE 路径时遇到的可用性和性能问题。

核心改动包括：

- **PP4 + DSpark speculative decoding**：解除/适配 DSpark 在 pipeline parallel 下的限制，让 drafter 和目标模型在 PP4 下能够协同运行。
- **Sparse MLA / Indexer SM80 fallback**：为 SM80 增加 row-chunked top-k/indexer fallback，规避长上下文 prefill 中大 `[M, N]` logits 临时张量导致的显存墙和崩溃。
- **Marlin MoE token order canonicalization**：在 `fused_marlin_moe()` wrapper 层 canonicalize `sorted_token_ids`，消除语义等价 routed layout 因物理顺序不同带来的 BF16 输出不一致。
- **FP8 shared-expert dequant 控制**：增加 `VLLM_MARLIN_FP8_DEQUANT_*` 环境变量，让部分 FP8 Marlin linear 层可选择先 dequant 到 BF16 再走 GEMM，用于定位和优化 SM80 上 shared expert 的耗时。
- **DeepSeek-V4-Flash 启动/测试脚本**：提供 PP4 启动脚本和离线 benchmark 脚本。

关键源码入口：

```text
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu/pp_utils.py
vllm/v1/worker/gpu/spec_decode/dspark/utils.py
vllm/model_executor/layers/sparse_attn_indexer.py
vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
vllm/model_executor/kernels/linear/scaled_mm/marlin.py
vllm/model_executor/layers/quantization/fp8.py
vllm/envs.py
```

## 已测 Benchmark 结果

除特别说明外，结果均来自本仓库对应源码/脚本，端到端 wall clock 计时，completion tokens 计数，不统计 SSE chunk 数。

### 1,024 input + 128 output，单并发 16 请求，五轮

脚本：`bench/offline_1024x128_c1_r16_5rounds_table.py`

配置：

```text
CUDA_VISIBLE_DEVICES=0,2,3,4
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

| 并发 | 请求/轮 | 五轮输出吞吐 tok/s | 中位数 | 范围 | CV |
|---:|---:|---|---:|---:|---:|
| 1 | 16 | 121.18 / 134.54 / 136.65 / 138.05 / 139.07 | **136.65** | 121.18-139.07 | 5.46% |

说明：第一轮包含更多运行期抖动/JIT 后效应，后四轮稳定在 134-139 tok/s。这个测试严格固定 1,024 input tokens 和 128 output tokens，并开启 `ignore_eos=True`，避免不同配置输出长度不一致造成假吞吐。

### PP4 + DSpark 端到端 serving 结果

硬件：4 x CMP 170HX，模型：原始 DeepSeek-V4-Flash-0731 checkpoint。

| 指标 | PP4 plain | PP4 + DSpark |
|---|---:|---:|
| decode，single stream，3 类内容聚合 | 50.8 tok/s | **98.1 tok/s** |
| prefill @ 77k context | **5,272 tok/s** | 5,207 tok/s |
| aggregate decode @ 64 concurrent | 472.0 tok/s | **712.8 tok/s** |
| verified one-shot context | **1,047,736 tokens** | **1,047,736 tokens** |

### 不同内容类型下的 decode

400 output tokens，temperature=0。

| prompt 类型 | PP4 plain | PP4 + DSpark | 加速比 |
|---|---:|---:|---:|
| technical exposition | 44.0 | 86.9 | 1.98x |
| open-ended prose | 55.1 | 93.9 | 1.70x |
| code generation | 55.1 | 118.8 | 2.16x |
| aggregate | **50.8** | **98.1** | **1.93x** |

### 并发吞吐

| concurrent requests | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| PP4 plain | 50.5 | 133.0 | 173.9 | 288.2 | 393.3 | 472.0 |
| PP4 + DSpark | **58.3** | **197.9** | **269.3** | 302.3 | **429.2** | **712.8** |

### Prefill：PP4 相比 TP4 更适合长上下文

单请求 warm run，`--max-model-len 131072`。

| prompt tokens | 1,544 | 3,082 | 6,159 | 12,313 | 24,621 | 50,006 | 76,929 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PP4 | 1,966 | 2,706 | 3,660 | 4,524 | 5,113 | **5,321** | 5,272 |
| TP4 | 908 | 774 | 841 | 809 | 804 | 776 | 801 |
| PP/TP | 2.2x | 3.5x | 4.4x | 5.6x | 6.4x | **6.9x** | 6.6x |

### 长上下文 decode / prefill

经过 row-chunk context-ceiling 修复后，one-shot prefill 可测到百万 token 级上下文。

| real context | prefill tok/s | TTFT | decode tok/s |
|---|---:|---:|---:|
| ~7,700 | - | 2.1 s | **88.7** |
| ~100,000 | - | 22.1 s | 79.0 |
| ~200,000 | 4,486 | 50.5 s | 67.7 |
| ~385,000 | 3,425 | 120.4 s | 54.5 |
| ~769,000 | 2,525 | 334-336 s | 39.6 / 43.6 |
| **~1,040,000** | **1,904** | **544-550 s** | **35.6** |

结论：上下文从约 7.7k 增加到约 1.04M，decode 从 88.7 tok/s 降到 35.6 tok/s，仍保留约 40% 的短上下文 decode 能力。百万上下文主要成本是 prefill，而不是后续 decode。

### Accumulated conversation 与 one-shot prefill 的区别

| mode | ROW_CHUNK | depth | outcome |
|---|---:|---:|---|
| accumulated chat | 128 | 733,120 | CUDA illegal memory access |
| accumulated chat repeat | 128 | 718,216 | same fault, same PP rank |
| one-shot prefill | 128 | 749,534 / 745,427 | pass |
| one-shot prefill | 128 | 1,047,736 | pass |
| accumulated chat | **64** | **1,002,852** | **405 turns, no crash** |

这个结果说明：长上下文 one-shot prefill 和真实多轮 prefix-cache accumulated conversation 是两条不同压力路径。`ROW_CHUNK=64` 对真实多轮场景更稳。

### FP8 shared-expert dequant 验证

用于验证 dense/shared expert 中 FP8 是否需要先 dequant 到 BF16 再计算的路径。

| 配置 | prefill+decode tok/s | decode tok/s | per-request decode tok/s |
|---|---:|---:|---:|
| shared dequant verify2 | 1079.03 | 63.47 | 7.93 |
| include-selected dequant | **1367.09** | **80.42** | **10.05** |

对应环境变量：

```bash
export VLLM_MARLIN_FP8_DEQUANT_BF16=1
export VLLM_MARLIN_FP8_DEQUANT_INCLUDE=shared_experts
export VLLM_MARLIN_FP8_DEQUANT_EXCLUDE=
```

这个开关不是默认强制路径，而是用于选择性测试哪些 FP8 Marlin linear 层在 SM80 上更适合先 dequant 到 BF16。

## 快速开始

### 1. 克隆源码

```bash
git clone https://github.com/yunyinbanfu/deepseek-v4-flash-0731-sm80.git
cd deepseek-v4-flash-0731-sm80
```

### 2. 安装

按源码方式安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

需要使用与你机器匹配的 CUDA / PyTorch / 编译环境。本仓库不提交 `.so`、wheel、venv、模型权重或 profile 数据。

### 3. 启动 PP4 serving

默认使用 GPU `0,1,2,3` 和模型路径 `/models/DeepSeek-V4-Flash-0731`：

```bash
bash launch/run-pp-dspark.sh
```

可通过环境变量覆盖：

```bash
DSV4_MODEL=/path/to/DeepSeek-V4-Flash-0731 \
CUDA_VISIBLE_DEVICES=0,2,3,4 \
DSV4_PORT=8000 \
bash launch/run-pp-dspark.sh
```

脚本默认设置：

```bash
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_SPARSE_DENSE_QUERY_BLOCK=4
VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

关闭 DSpark 可用：

```bash
DSV4_DISABLE_DSPARK=1 bash launch/run-pp-dspark.sh
```

### 4. 跑 1,024 x 128 五轮 benchmark

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4 \
PYTHONPATH=$PWD \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_SPARSE_DENSE_QUERY_BLOCK=4 \
VLLM_SPARSE_DENSE_QUERY_BLOCK_DECODE=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python bench/offline_1024x128_c1_r16_5rounds_table.py
```

其他脚本：

```text
bench/offline_1024x128_5rounds.py
bench/offline_pp4_c1_r16_1024x256.py
bench/offline_pp4_1024x64_conc8_dequant_shared.py
```

完整历史结果和测试说明：

```text
RESULTS.cmp170hx.md
SETTINGS.cmp170hx.md
patches.cmp170hx/
```

## 重要测试经验

1. 不能按 SSE chunk 数算吞吐。spec decode 下一个 chunk 可能包含多个 token，要按 `completion_tokens` 或实际 token ids 计数。
2. 对比配置必须生成相同 token 数，建议 benchmark 使用 `ignore_eos=True`。
3. 服务启动后的第一轮可能包含 Triton JIT/warmup 抖动，应单独看或丢弃。
4. prefix caching 会让重复 prompt 跳过 prefill，测 prefill 时要关掉或换 prompt。
5. 不要用 `max_tokens=1` 和 `max_tokens=N` 相减来估 decode，长上下文 prefill 抖动会放大误差。
6. one-shot prefill 和 accumulated conversation 不是一回事，百万 token 能 one-shot pass 不代表多轮 prefix-cache 一定稳。

## 不包含什么

本仓库不包含：

```text
模型权重
Docker 镜像
Python 虚拟环境
编译产物 .so / wheel
Nsight / sqlite profile 数据
运行日志
```

`.gitignore` 已排除这些文件，避免仓库变成不可 clone 的大包。

## License

沿用 upstream vLLM 的许可证，见 [LICENSE](LICENSE)。
