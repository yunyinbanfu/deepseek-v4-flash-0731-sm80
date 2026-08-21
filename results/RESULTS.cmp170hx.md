# Measured results

Hardware: **4× NVIDIA CMP 170HX** (GA100, sm_80, VRAM-unlocked to 65,536 MiB, PCIe Gen2 x4,
no P2P, 180 W power cap). Model: `deepseek-ai/DeepSeek-V4-Flash-0731`, original checkpoint
(MXFP4 experts + FP8 e4m3 block-quantised attention, ~284B total / ~13B active).

All numbers are end-to-end wall clock measured from a client, prefix caching **off**, warm-up
request discarded. Harnesses are in [bench/](bench/).

---

## Headline

| | plain | **+ DSpark** |
|---|---|---|
| decode, single stream (3-content aggregate) | 50.8 | **98.1 tok/s** |
| prefill @ 77k context | 5,272 | 5,207 tok/s |
| aggregate decode @ 64 concurrent | 472.0 | **712.8 tok/s** |
| verified context | **1,047,736 tokens** | **1,047,736 tokens** |

---

## Rebase to c3046d1 (2026-08-13)

Upstream ran a 41-commit serving-optimization campaign on `dsv4-flash-a100`
(`f8ea5bb` → `c3046d1`, 2026-08-04). Adopting it on this rig (3× 170HX, PP3, the INT4
abliterated repack at 524k context) is worth a real but modest **+7% decode**:

| | `f8ea5bb` + 7 patches | `c3046d1` + 6 patches | Δ |
|---|---|---|---|
| decode, fixed prompt, token-counted | 86.1 ± 3.3 (n=15) | **92.0 ± 2.8 (n=20)** | **+6.8%, p<0.001** |
| needle @ 123k / 400k / 492k (94% of window) | verified @401k | **PASS / PASS / PASS** | ✅ |
| decode at 123k / 400k / 492k depth | — | 95 / 70 / 60 tok/s | known curve |
| gsm8k / tool-calling | 96.0% / 24 | 95.5% / 24/24 | noise |
| KV pool @524k maxlen | 2,013,978 | 2,013,136 | −0.04% |

**A "+30%" figure circulated for this range. It does not survive a paired A/B.** It came from
comparing two *different* benchmark contexts; the campaign's own in-tree record
(`benchmarks/kernels/dsv4_sm80_refutations.md` — worth reading in full) puts the decode step at
13.25 → 12.15 ms = **−8.3%**, which is exactly what the paired measurement reproduces. The
"+48% DSpark acceptance" part of that claim is flat on identical content (tok/chunk 2.49 → 2.50).

Two measurement traps this campaign re-confirmed:

- **A 3-content aggregate at n=1 is not a measurement.** An *unchanged* server swung
  **101.9 → 130.6 tok/s** across four back-to-back aggregate runs. Judge decode changes on
  ≥3 aggregate runs or on many fixed-prompt runs; never on one draw.
- **Sampling temperature (0 vs 0.6) makes no resolvable decode difference** on this stack.

What changed for the patch set and build:

- **`0001` is upstream now; `0002`–`0006` apply with zero rejects** — see
  [patches/README](patches/README.md#-2026-08-13-recommended-base-is-now-c3046d1--patch-0001-is-no-longer-needed),
  including how to recover `c3046d1` after upstream's second force-push (tarball + tree-SHA
  verification; no git method reaches it).
- **The range touches `csrc/`, so precompiled/bind-mount deployment cannot deliver it.**
  `libtorch_stable/topk.cu` routes small-batch decode top-k to FilteredTopK below the radix
  threshold (their measurement: 1.9–3.6× on that kernel) — compiled code. Full source build:
  [docker/Dockerfile.fullbuild](docker/Dockerfile.fullbuild), sm_80-only, ~115 min.
- Flags: `VLLM_MARLIN_FP8_DEQUANT_BF16=1` is an upstream-adopted **prefill** win (−35 ms
  TTFT@8k, block-fp8 dense only). `VLLM_MARLIN_DENSE_OCCUPANCY` is refuted (their own record).
  `VLLM_DSPARK_VOCAB_SHARD` has zero consumers at `c3046d1`; the adopted
  `use_local_argmax_reduction` spec-config switch is TP-only (O(2·tp) vs O(vocab) comms —
  pointless under PP/TP1).
- The row-chunk context-ceiling fix (`0006` + `DSV4_LOGITS_ROW_CHUNK`) was **not** taken
  upstream and is still load-bearing above ~134k context.

The branch tip has since been force-pushed again (2026-08-11) to a single squashed commit with
~11.7k insertions of newer work (DSpark vocab-shard wiring, hierarchical all-reduce, indexer
query-sharding). Nobody has published numbers for it; these patches will need re-checking there.

---

## Decode by content type

Speculative decoding's benefit is strongly content-dependent — a single prompt is not a
measurement. 400 tokens each, temperature 0.

| prompt type | PP4 plain | PP4 + DSpark | ratio |
|---|---|---|---|
| technical exposition | 44.0 | 86.9 | 1.98× |
| open-ended prose | 55.1 | 93.9 | 1.70× |
| code generation | 55.1 | 118.8 | 2.16× |
| **aggregate** | **50.8** | **98.1** | **1.93×** |

Mean acceptance length 3.03 of a possible 6; per-position acceptance
0.730 / 0.569 / 0.372 / 0.226 / 0.131.

## Concurrency — DSpark keeps winning under load on PP

| concurrent requests | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| PP4 plain | 50.5 | 133.0 | 173.9 | 288.2 | 393.3 | 472.0 |
| **PP4 + DSpark** | **58.3** | **197.9** | **269.3** | 302.3 | **429.2** | **712.8** |

This is the opposite of the tensor-parallel behaviour, where DSpark went *negative* above
about 8 concurrent (TP, c=16: DSpark 212 vs plain 289). Pipeline parallel leaves bubbles the
drafter can fill; tensor parallel does not.

## Prefill — PP scales with context, TP does not

Single request, warm, `--max-model-len 131072`.

| prompt tokens | 1,544 | 3,082 | 6,159 | 12,313 | 24,621 | 50,006 | 76,929 |
|---|---|---|---|---|---|---|---|
| **PP4** | 1,966 | 2,706 | 3,660 | 4,524 | 5,113 | **5,321** | 5,272 |
| **TP4** | 908 | 774 | 841 | 809 | 804 | 776 | 801 |
| PP/TP | 2.2× | 3.5× | 4.4× | 5.6× | 6.4× | **6.9×** | 6.6× |

TP is flat across a 50× range of context lengths. See [SETTINGS.md](SETTINGS.md) for why.

## Decode vs context — the speedup does not decay

| prompt tokens | 2k | 8k | 32k | 65k | 100k |
|---|---|---|---|---|---|
| PP4 plain | 48.5 | 45.8 | 43.7 | 41.2 | 38.8 |
| **PP4 + DSpark** | **117.8** | **129.3** | **99.7** | **99.5** | **90.0** |
| ratio | 2.4× | 2.8× | 2.3× | 2.4× | 2.3× |

Time to first token is identical between the two (0.78 s → 14.8 s), i.e. DSpark costs
nothing on prefill. Acceptance holds at 3.5–3.7 at long context.

## Speed vs context

Measured to the top of the model's range, after the [context-ceiling fix](#-context-ceiling--solved).
One harness, identical method at every point.

| real context | prefill tok/s | TTFT | decode tok/s |
|---|---|---|---|
| ~7,700 | — | 2.1 s | **88.7** |
| ~100,000 | — | 22.1 s | 79.0 |
| ~200,000 | 4,486 | 50.5 s | 67.7 |
| ~385,000 | 3,425 | 120.4 s | 54.5 |
| ~769,000 | 2,525 | 334–336 s | 39.6 / 43.6 |
| **~1,040,000** | **1,904** | **544–550 s** | **35.6** |

**Decode degrades gracefully: 88.7 → 35.6 tok/s, i.e. it retains 40% of its short-context rate
across a 135× context increase.** Generating at a full million tokens of context is still
faster than most people read.

The 769k row shows two independent runs (39.6 and 43.6) — DSpark is
[not deterministic](#-dspark-output-is-not-reproducible), so treat single decode numbers as
±10%.

**Prefill is the expensive half, not decode.** At 1M you wait ~9 minutes for the first token
and then generate at 35.6 tok/s. Budget accordingly: this is a batch/document tool at the top
of its range, not an interactive one.

⚠️ **These decode absolutes are a worst case.** Every prompt here is a random-word haystack,
and DSpark acceptance on random text is poor — prose and code reach 90–130 tok/s at short
context (see [Decode by content type](#decode-by-content-type)). The *shape* is trustworthy;
the absolute numbers are pessimistic.

### Why prefill DECAYS with context — and why the section above says it RISES

Both are true at different scales. The rising curve (1,966 → 5,321 tok/s) was measured
**1.5k → 77k** and stopped there, because the bug killed everything past ~150k.

- **Rising, ≤25k:** fixed per-chunk and pipeline fill/drain costs amortise. Plateau ~5,300.
- **Falling, ≳200k:** sparse attention keeps *attention* cheap — top-k selects a fixed 512
  blocks at any length — but **the indexer that chooses them scores every compressed key**,
  `M × N` with `N = seq_len / compress_ratio`. Confirmed directly: a 121,582-token prompt logs
  `N = 30,395`, exactly 121,582/4. Per-chunk cost therefore grows linearly with depth, total
  prefill is quadratic-ish, and throughput falls as 1/context.

Fitting *cost per token = fixed + proportional to position* matches within a few percent
(200,044 → 4,486 observed / 4,447 model; 538,505 → 3,017 / 3,062; 769,274 → 2,525 / 2,526)
and puts the crossover at **~550k**. That is a descriptive fit, not a profile.

**The same `M × N` buffer is why prefill decays out there *and* why it used to crash.**

## Time to first token

| context | PP4 | TP4 |
|---|---|---|
| 2k | 0.79 s | 1.67 s |
| 32k | 4.78 s | 26.93 s |
| 100k | **14.6 s** | **87.3 s** |

## num_speculative_tokens

| value | aggregate tok/s | mean acceptance |
|---|---|---|
| **5** (= `dspark_block_size`) | **98.1** | 3.03 |
| 7 | 60.3 | 1.43–2.51 |

4 and below are rejected by vLLM for this checkpoint.

---

## Correctness

- **Needle-in-a-haystack passes at 23k / 77k / 95k tokens with DSpark enabled** — a
  distinctive passphrase buried at 10% depth is retrieved verbatim. This tests that the
  sparse indexer actually selects the right blocks across the whole window, not merely that
  the run completes.
- Reasoning spot-checks correct ("17 sheep, all but 9 run away" → 9).

### ⚠️ DSpark output is not reproducible

At temperature 0, DSpark output differs from non-speculative output on all 6 probe prompts,
**and differs between two runs of the same server**. Each divergence begins at an obviously
low-confidence branch point and every substantive answer was correct.

Controls run to interpret this:
- plain PP4 **is** self-deterministic (two identical runs, byte-identical output);
- **TP4 + DSpark, on the stock upstream path with none of the patches in this repo, is
  also non-deterministic** — so this is a property of DSpark, not of the pipeline-parallel
  patches here.

If you need bit-reproducible output, run without `--speculative-config`.

---

## Limits

### ★ Context ceiling — SOLVED

**The full 1,047,736-token context runs.** That is `--max-model-len 1048576` minus the 24
generated tokens, i.e. the config cap, with no bug wall below it. Needle-in-haystack verified
(passphrase at 10% depth, so a PASS means the sparse indexer really selected the right blocks
across the whole window).

Settings: `DSV4_LOGITS_ROW_CHUNK=128`, `VLLM_PP_LAYER_PARTITION=12,12,12,7`,
`--gpu-memory-utilization 0.85`. KV pool 4,991,054 tokens (4.76× concurrency at 1M).

| real prompt tokens | TTFT | prefill tok/s | needle |
|---|---|---|---|
| 134,659 | 29.8 s | 4,524 | ✅ ← the old failure point |
| 292,351 | 77.3 s | 3,781 | ✅ |
| 538,505 | 178.5 s | 3,017 | ✅ |
| 769,274 | 304.6 s | 2,525 | ✅ |
| **1,047,736** | **550.2 s** | **1,904** | ✅ |

Also 4 concurrent 153,891-token prompts, 4/4 correct with no cross-request bleed.

**Everything the previous version of this document said about the ceiling scaling inversely
with `--max-model-len` was a symptom of this bug and is withdrawn.** Set `--max-model-len` to
what you need.

#### The cause

`fp8_mqa_logits_triton` allocates `logits = torch.empty((M, N), torch.float32)` — `M` = tokens
in the prefill chunk, `N = seq_len / compress_ratio` — and passes the whole buffer to the
top-k. It grows with context and is the largest allocation on the Triton fallback path (the
one sm_80 takes because DeepGEMM is unavailable). Above ~134k the worker dies with
`Xid 31 — MMU Fault ... ACCESS_TYPE_VIRT_WRITE`; the surviving ranks then emit misleading
`gloo ... Connection closed by peer` — **chase the Xid, not the gloo message.**

**`CUDA_LAUNCH_BLOCKING=1` is what located it**, in one run, after three sessions of code
reading produced three confident and wrong theories:

```
attention.py:496                 execute_in_parallel(lambda: indexer(...))
attention.py:893                 self.indexer_op(hidden_states, q_quant, k, weights)
sparse_attn_indexer.py:965/592   fp8_mqa_logits_triton(...)
mqa_logits_triton.py:389         _fp8_mqa_logits_kernel[grid](...)
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

#### The fix — [patch 0006](patches/0006-logits-row-chunk.patch)

Each row's top-k reads only its own `[ks, ke)` window, so **rows are independent** — computing
them in blocks is exact, not an approximation. `DSV4_LOGITS_ROW_CHUNK=256` reaches ~957,600;
**1M needs 128**. That the wall moves with the block size confirms the same allocation is still
the limiter — the chunk is a dial on it, not a cure, and a properly bounded buffer is the right
upstream fix.

**Cost: none measurable.** Prefill 1,456 vs 1,448 tok/s at 4k. Decode aggregate over 4 runs
each on the same live server: 82.7 / 83.1 / 98.9 / 110.2 with the fix, 97.9 / 100.7 / 102.3 /
102.7 without — overlapping, and the patch sits inside `if has_prefill:` so decode is untouched
by construction.

### Root-cause attempts that did NOT work — don't repeat these

| tried | result |
|---|---|
| PR **#49897** — torch fallback for `top_k_per_row_prefill` (Python) | no change to the ceiling; cost ~10% prefill |
| PR **#49139** — radix histogram ring fix (CUDA, needs full rebuild) | no change |
| PR **#50201** — harden `top_k_per_row` against NaN/under-fill (CUDA) | no change |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **unusable on these cards** — hard fail at model load: cannot map 20 MB with **28.6 GiB free**. CUDA VMM appears broken on GA100 CMP parts. |
| `VLLM_PP_LAYER_PARTITION=12,12,12,7` | **did not move the ceiling** — but worth setting anyway: **+85% KV pool** and it removes a real 8.7 GiB rank imbalance |

Two theories that fit the evidence beautifully and were both wrong:

**The radix threshold.** `RADIX_THRESHOLD = 32768` in `persistent_topk.cuh`; with
`compress_ratio 4` that is exactly 131,072 context tokens, matching the original boundary to
the token. Applying both radix PRs (full CUDA rebuild) changed nothing. It is **not** an
integer-width bug either — the largest offset the faulting kernel computes at these sizes is
~7.5e7, three orders of magnitude below 2³¹.

**The starved pipeline rank.** The last PP rank carries `lm_head` *and* the DSpark drafter
while vLLM sizes the KV pool uniformly, so it ran at **0.09 GiB free versus 8–9 GiB on its
peers** — and the ceiling correlated cleanly and monotonically with `--gpu-memory-utilization`
across three values. Rebalancing with `VLLM_PP_LAYER_PARTITION` brought every rank to 6–8 GiB
free and grew the KV pool 85%, and **the ceiling did not move by one token** — the fault simply
migrated to PP0, the rank with the *most* free memory, because under pipeline parallelism the
leading rank reaches the critical `N` first. Memory pressure only decided which rank noticed
first. **A clean monotonic correlation is not a cause.**

### ⚠️ Patch 0001 is precautionary, not a fix for an observed bug

Earlier versions of this repo stated as fact that without patch 0001, sm_80 emits
fluent-looking degenerate text at prompt lengths 2049–4096. **That is withdrawn.** The
original reporter has retracted it for `dsv4-flash-a100` after building and running the
branch, and **we could not reproduce it either.**

A/B on this hardware (4× 170HX, PP4 + DSpark, `dsv4-0731-orig`), toggling only the
`has_device_capability(90)` term, with a one-shot probe confirming which path was actually
selected each time:

```
gate ON  → [topk-probe] persistent=False cooperative=False topk_tokens=512
gate OFF → [topk-probe] persistent=True  cooperative=False topk_tokens=512
```

Needle retrieval with **`persistent_topk` active** (gate OFF — the allegedly broken path),
fresh KV cache, candidate counts spanning the claimed band:

| real prompt tokens | candidate count | in band (512–1024)? | needle |
|---|---|---|---|
| 1,967 | 491 | no | ✅ |
| 2,351 | 588 | **yes** | ✅ |
| 2,813 | 703 | **yes** | ✅ |
| 3,274 | 818 | **yes** | ✅ |
| 3,736 | 934 | **yes** | ✅ |
| 4,120 | 1,030 | no | ✅ |
| 4,505 | 1,126 | no | ✅ |

Clean throughout, and the answers are coherent — which is what a needle test detects, since
wrong indices would prevent retrieval.

**And the gate costs nothing either way.** Decode aggregate over 3 runs each: gate ON
107.1 / 99.3 / 90.8, gate OFF 104.7 / 92.5 / 85.5 — overlapping, inside DSpark's
[non-determinism](#-dspark-output-is-not-reproducible). Prefill in the band is identical
(2,399–2,992 tok/s either way).

**So we keep patch 0001 as a free guard** — the failure may well be real on older bases,
where the reporter first saw it — but nobody should apply it believing this branch is broken
without it. If you want to check on your own hardware, flip the `has_device_capability(90)`
term and log `use_persistent_topk` at the selection site; a code read is not sufficient, which
is the whole lesson here.

### Three or four cards

> ⚠️ **RETRACTED 2026-08-09.** This section previously read *"Four cards required"* and stated
> that 3 cards could not run this model. **That was wrong.** The native MXFP4+FP8 checkpoint
> runs on 3 cards. The original conclusion was reached without trying the one lever that
> matters — `VLLM_PP_LAYER_PARTITION` — which this repo already documents, but only ever with a
> four-entry value.

| configuration | result |
|---|---|
| 4 cards | works |
| **3 cards + `VLLM_PP_LAYER_PARTITION=15,15,13`** | **works** |
| 3 cards, default layer split | illegal memory access in Marlin MXFP4 expert repack |
| 2 cards | ~155 GiB of weights does not fit in ~127 GiB of VRAM |

**The 3-card run, in full:** `deepseek-ai/DeepSeek-V4-Flash-0731` native weights (shard SHA-256s
verified against the Hub), `--pipeline-parallel-size 3`, `--gpu-memory-utilization 0.95`,
`--max-model-len 131072`, `--kv-cache-dtype fp8`, DSpark **on** (`num_speculative_tokens 5`),
`DSV4_LOGITS_ROW_CHUNK=256`, `VLLM_PP_LAYER_PARTITION=15,15,13`. Loads 48/48 shards,
`GPU KV cache size: 222,408 tokens` (1.70× concurrency), decode ~78.8 tok/s median. Exercised
with 825 generations plus a 61-turn accumulating conversation to 118,411 tokens — zero errors.

**Why the default split fails.** 43 layers over 3 ranks defaults to `[15,14,14]`, and the last
pipeline rank *additionally* carries `lm_head` and the DSpark drafter. It runs out during the
target model's expert repack, which is why the traceback lands in
`marlin_utils_fp4.py:332 _repack_marlin_experts` and reads like a Marlin bug rather than a weight
**placement** problem. Giving that rank two fewer layers (`15,15,13`) fixes it. This is the same
reasoning behind the 4-card `12,12,12,7` split already documented in SETTINGS — it was simply
never applied to a 3-card layout.

The earlier claim that the failure is "independent of ... memory settings" was correct about
`--gpu-memory-utilization` and wrong in what it concluded: util sizes the **KV pool**, it does not
move **weights**, so it can never fix a load-time placement failure. Ruling out util does not rule
out memory — it points at the layer split. Pipeline parallel imposes no divisibility requirement
(43 splits fine over 3), unlike tensor parallel where 64 heads and 256 experts genuinely cannot.

Also corrected here: the checkpoint is **~155.4 GiB (166.9 GB)**, not ~140 GB — `du` under-reports
it against the true byte total of 166,886,535,336. The 2-card conclusion is unchanged.

#### 3 cards + the INT4 repack: 512k verified, and a very large KV pool

The old "untested lead" is confirmed too — the INT4 compressed-tensors repack (~150.4 GiB) also
runs on 3 cards, and it is the better 3-card option if you want context rather than the native
weights.

| | 3 cards, INT4 repack |
|---|---|
| `--max-model-len` | **524,288** (verified serving) |
| `DSV4_LOGITS_ROW_CHUNK` | **64** |
| `VLLM_PP_LAYER_PARTITION` | `15,15,13` |
| `--gpu-memory-utilization` | 0.95 |
| **GPU KV cache size** | **2,013,978 tokens** |
| max concurrency at full length | 3.84× |
| needle retrieval | ✅ correct at **401,532** prompt tokens (174 s cold prefill) |
| decode, short prompt | ~86 tok/s median |

★ **`DSV4_LOGITS_ROW_CHUNK` buys context, not just crash-safety.** Lowering it shrinks the sparse
indexer's transient `[M, N]` float32 buffer, and vLLM sizes the KV pool against *profiled peak*
memory — so that transient is charged against your context budget. Observed on 3 cards:

| row chunk | `--max-model-len` | `--max-num-batched-tokens` | KV pool |
|---|---|---|---|
| 256 | 131,072 | 2048 | 506,183 |
| 128 | 262,144 | 1024 | 1,487,285 |
| **64** | **524,288** | 1024 | **2,013,978** |

⚠️ These are **not** a clean single-variable sweep — `--max-model-len` and
`--max-num-batched-tokens` moved too, and the 256 row used the plain INT4 repack while the other
two used an overlay build of it (same size to within 0.03 GB). The direction is consistent and the
mechanism is understood, but do not read the exact ratios as attributable to the row chunk alone.

⚠️ **What is NOT claimed:** 1M has **not** been tested on 3 cards. The 2,013,978-token pool
exceeds 1,048,576 with ~1.9× headroom, so 1M *should* fit — but that is arithmetic, not a
measurement, and the 1,002,852-token accumulating conversation on this page was run on **4**
cards. Treat 3-card 1M as expected-but-unverified until someone posts the run.

---

## Accumulated conversation ≠ one-shot prefill

Everything above this section — including the 1,047,736 figure — is a **single one-shot prefill
scored by a single needle**. That tests the prefill path and nothing else. Driving the model the
way a user does (many turns, prefix cache reusing prior KV, decode at depth) behaves differently.

Harness: a 405-turn conversation over a real mixed corpus (engineering docs / source code / prose),
prefix caching on, facts planted at known turns and re-queried on a schedule.

### The ceiling is lower, and `DSV4_LOGITS_ROW_CHUNK` moves it

| mode | `ROW_CHUNK` | depth | outcome |
|---|---|---|---|
| accumulated chat | 128 | 733,120 | ⛔ CUDA illegal memory access |
| accumulated chat (repeat) | 128 | 718,216 | ⛔ same fault, same PP rank |
| one-shot prefill | 128 | 749,534 / 745,427 | ✅ needle hit, clean |
| one-shot prefill | 128 | 1,047,736 | ✅ |
| **accumulated chat** | **64** | **1,002,852** | ✅ **405 turns, no crash** |

A one-shot prefill at the *same* depth is fine — and at 1M, with a larger `N`. **So the wall is not
depth; it is the accumulated / prefix-cached path.** Halving the chunk moved it past 1M, exactly as
halving 256 → 128 moved the one-shot ceiling.

Fault site, reproduced twice, always the same rank (not the last one):

```
sparse_attn_indexer.py:618  sparse_attn_indexer
sparse_attn_indexer.py:164  _top_k_per_row_prefill_torch
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

⚠️ `topk` is the first synchronising op after `fp8_mqa_logits_triton`, so an async fault from that
kernel would be *reported* here regardless of origin. **`CUDA_LAUNCH_BLOCKING=1` has not yet been
run**, so treat the exact frame as unconfirmed. Leading hypothesis for why accumulation and not
one-shot: prefix-cache block reuse fragments the KV block table, and a bad index dereferences out
of range where a contiguous one-shot table survives. Untested.

**An OOM/IMA leaves the cards unable to create CUDA contexts.** Recover without rebooting:
`nvidia-smi -r -i <gpu>` then `rmmod nvidia_uvm; modprobe nvidia_uvm`, and re-apply the power cap.

### Retrieval accuracy vs depth — the window is reachable, not uniformly usable

Canary facts planted at known turns, 77 probes:

| context | recall | | context | recall |
|---|---|---|---|---|
| 150,000+ | **100%** | | 600,000+ | 86.7% |
| 300,000+ | 86.7% | | 750,000+ | **50.0%** |
| 450,000+ | 60.0% | | **900,000+** | **30.0%** |

**This is not caused by these patches.** Recall is statistically identical between `ROW_CHUNK` 128
and 64 at matched depth (100%/100% at 150k, differences of one probe elsewhere), which is what you
expect if row-chunking is exact. Chunk size changes only whether it **crashes**, never whether it
is **right**.

The likely cause is architectural: `index_topk = 512` is fixed while the candidate pool grows
linearly with context, so the share of context reachable falls from ~1.4% at 150k to ~0.23% at 900k
— a ~6× reduction against a ~3.3× drop in recall.

**Character of the failures matters more than the rate.** Of 23 misses: **16 returned a different
real fact from the conversation**, 6 abstained honestly, and **1** invented something. So this is
retrieval/binding failure, not fabrication — but a confident wrong answer sourced from elsewhere in
your own documents is arguably worse in practice than an obvious invention. Honest abstention only
appeared above ~918k, where retrieval returns *nothing*; in the 400–800k band it substitutes
confidently.

**A grounding/abstention system prompt does not help.** A/B at 118k with canary density doubled to
force retrieval failure: recall 77.1% vs 79.2% (one probe, noise) and **zero abstentions in either
arm** — not one substitution converted. The model has no signal that it is failing, so an
instruction about honesty has nothing to act on.

### Degenerate repetition: rare, and we cannot predict it

Across **1,172 turns** in six runs, **2** outputs degenerated into repetition (`max_rep_8gram` 7–8,
both stopping only on `max_tokens`) — **0.17%**, at 88,137 and 866,282 tokens.

Both happened to be engineering-doc summaries, which suggested repetitive source content as the
trigger. **Tested and rejected.** The rate of turns containing any repeated 8-gram: 16.4% vs 6.8%
(techdoc vs code) at 50–150k, but **12.4% vs 13.5% — reversed** at 300–750k. Depth-matched
permutation test, n=921: **p = 0.55**. And with 564 techdoc vs 566 code turns in the corpus,
2-of-2 landing on one type is p = 0.25 by chance — never evidence in the first place.

Mild repetition is normal and content-independent (~10–13% of all turns). Depth does not drive it
either: within one content type, mean `max_rep_8gram` is 1.20 / 1.09 / 1.20 / 1.15 / 1.17 across
0 → 750k+. **We have no identified trigger. Do not claim one.**

---

## Thinking recovers a lot of the lost long-context recall — the effort level does not

`thinking=False` is the default on 0731, so every number above this section — and, I suspect, most
numbers posted about this model — was measured with thinking **off**.

Matched pair: one container, thinking toggled **per request** via `chat_template_kwargs`, identical
`max_tokens`, canary density, corpus and depth. Nothing differs but the flag.

| ctx | thinking OFF | thinking ON |
|---|---|---|
| 0+ | 82% (14/17) | **100%** (17/17) |
| 150,000+ | 80% (12/15) | 93% (14/15) |
| 300,000+ | **27%** (4/15) | **60%** (12/20) |
| 450,000+ | 50% (5/10) | 60% (3/5) |
| **overall** | **61.4%** (35/57) | **80.7%** (46/57) |

**Fisher exact p = 0.038**, every bucket improves, and it replicates an earlier independent run
(81.1% → 96.9%, p = 0.060). It is the only intervention that has moved recall here: chunk size
changes it not at all at matched depth, and a grounding/abstention system prompt produced **zero**
abstentions in either arm.

Cost, with both arms on the same generous `max_tokens`: **~2×** generation (630 → 1,227 tokens per
turn) and decode essentially unchanged (69.1 → 65.3 tok/s). Output also came out **less** repetitive,
not more (repeated-8gram 29.5% → 18.8%; median distinct-4 0.981 → 0.989).

### The effort level buys nothing — use `high`, not `max`

Paired at **453,845 tokens**, same conversation, same 21 canaries, only `reasoning_effort` differs:

| effort | recall | reasoning chars | completion tokens |
|---|---|---|---|
| `high` | **15/21 = 71.4%** | 473 | 111 |
| `max` | **15/21 = 71.4%** | **1,284 (2.7×)** | **309 (2.8×)** |

Discordance is **2 each way** (McNemar exact p = 1.0). `max` genuinely thought 2.7× harder and
retrieved exactly as well. Independently consistent with the Terminal-Bench 2.1 result reported in
[vllm#50576](https://github.com/vllm-project/vllm/issues/50576) (+3/89 tasks for max thinking,
p = 0.629). **What matters is that the model thinks at all, not how hard you tell it to.**

### Only two effort states are actually reachable

Prompt tokens injected, measured deterministically against `/tokenize`:

| effort | injected |
|---|---|
| `low` / `minimal` / `medium` / **`high`** / `none` / no kwargs | **5 (nothing)** |
| **`max`** / **`xhigh`** | **84** |

The model defines exactly three (`low` is empty, plus `high` and `max` prompts) and asserts
membership — but on this path `high` injects nothing, i.e. renders identically to `low`. So "thinking
on at `high`" means *thinking on with no effort prefix*, which is what the table above measured.

### ⚠️ You need `--reasoning-parser deepseek_v4`, or thinking pollutes the answer

The `<think>` delimiters are **special tokens** and are stripped on decode. Without the parser the
reasoning text arrives inside `content` with nothing marking it — replies literally begin
*"We need answer classic. Need be careful. User asks…"*. With the parser it lands in its own field
and `content` stays a clean answer.

⚠️ On this build the returned field is **`reasoning`**, not `reasoning_content`. A client reading
only the latter silently sees zero thinking and cannot tell a thinking run from a non-thinking one.

---

## Measurement pitfalls found the hard way

Each of these produced a wrong number before it was caught:

1. **A streaming harness must not count SSE chunks.** Under speculative decoding one chunk
   carries several tokens (≈ the acceptance length). Counting chunks reported 24 tok/s where
   the true figure was 79.5. Rate against the server's `completion_tokens`.
2. **Compared configs must generate the same number of tokens** — pass `ignore_eos`, or
   speculative output diverges, hits EOS early, and you compare 50 tokens against 192.
3. **Discard the first request after boot** — Triton JIT makes it read ~4× low. A cold
   reading of 514 tok/s was really 1,966.
4. **Disable prefix caching for benchmarks**, or repeated prompts skip prefill entirely.
   The tell was a 100k-token prompt appearing to reach first token in 0.5 s.
5. **Do not derive decode rate by subtracting two calls** (`max_tokens=1` vs `max_tokens=N`).
   At long context prefill varies by seconds between runs and the subtraction produced
   135 tok/s sitting between neighbours of 43.8 and 38.6.
6. **Assert what is actually running** — `docker inspect NAME --format '{{join .Args " "}}'`.
   A stray launcher invocation once won the container name and served a whole sweep from a
   different configuration.
7. **A best steady-state window is not a benchmark.** Reporting the fastest 10-second
   logging window gave 3.6×; the honest end-to-end figure over mixed content was 1.9×.
8. ★ **Verify the access pattern users will actually drive, not just the headline size.** A
   one-shot prefill and an accumulating conversation are different code paths. This repo
   published "1M verified" on one-shot needles while a real chat on the same config crashed at
   ~725k.
9. ★ **One needle per length is not verification.** Single probes at each depth read 9/9 and
   looked reliable; repeat-probing found recall at 30% by 900k. Sample enough per depth to see a
   slope, not just a pass.
10. ★ **Match the answer budget across arms, or you measure the budget.** An A/B with
    `max_tokens` 400 on one arm and 3000 on the other (because thinking needs room) produced two
    wrong conclusions: "thinking makes output 4x more repetitive" and "thinking costs 4x tokens".
    Rerun with both at 6000 **reversed** the first. If one arm needs a bigger budget to function,
    raise it for **both**.
11. ★ **Toggle the treatment per-request against ONE server** where the API allows it, instead of
    running two servers. Container, weights and cache are then provably identical.
12. ★ **Check the base rate before calling a co-occurrence a pattern.** Two rare events sharing an
    attribute felt like a lead; the corpus was a 50/50 split, so p = 0.25 by chance. An effect
    that appears in one stratum and reverses in another is noise — look for replication before
    reporting.
