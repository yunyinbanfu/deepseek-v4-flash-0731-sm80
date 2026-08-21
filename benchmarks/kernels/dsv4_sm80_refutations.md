# DSv4-Flash on 8×A100 (SM80): measured refutations and measurement rules

Serving-optimization campaign record, 2026-08-04, branch `dsv4-flash-a100`.
Model `deepseek-ai/DeepSeek-V4-Flash-0731`, TP=8, DSpark spec decode
(`num_speculative_tokens=5`), `--kv-cache-dtype fp8_ds_mla`.

**Purpose: do not re-try anything in §1 without new evidence, and do not
trust a measurement that violates §2.** Every entry below was measured, not
argued. Full data: `/root/optim/*.md` on the campaign box; adopted changes are
in the commit history after `4adc46ce38`.

Baseline → end state: cold TTFT@8K 1089.9 → 475.7 ms (2.29×, config v10);
decode step 13.25 → 12.16 ms. The post-v9 model-level roofline
(`/root/optim/ROOFLINE.md` §7–10) puts the current architecture's TTFT floor
at ~296 ms on this hardware; §3 below carries the per-component basis.

## 1. Refuted optimizations

### Compilation / graphs
- **Enabling torch.compile/inductor** (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`):
  decode step +0.32 ms (5.1σ), TTFT null. The hot path is custom C/tilelang
  ops inductor cannot fuse across, and ~93% of decode kernels are already
  CUDA-graph replays. Also re-confirmed combined with the dequant flag (the
  "inductor fuses plain aten::mm better" interaction hypothesis is false).
  The auto-enabled default (compile mode NONE) is correct for this model.

### Parallelism / communication
- **Expert-parallel MoE** (`--enable-expert-parallel`): TTFT +50.8 ms (72σ),
  step +0.54 ms (27σ). The all2all dispatch+combine costs more than the
  better GEMM shapes save, on both phases. EP engaged correctly (32/256
  experts per rank) — this is a real measurement, not a misconfig.
- **Prefill context parallelism (PCP)**: hard `assert pcp_world_size == 1`
  for sliding-window MLA specs, and on 8 GPUs PCP=2 forces TP=4, which
  roughly doubles the weight-read-bound decode step. Wrong trade even if the
  assert were lifted.
- **Sequence parallelism / GEMM-comm overlap**: requires inductor (see
  above), the pass is hardcoded off upstream (`IS_DENSE=False`, vllm#25689),
  it forces `cudagraph_mode=FULL` (losing piecewise graphs decode relies
  on), and the spec-decode cudagraph shape rule only admits
  `num_speculative_tokens ∈ {1,3,7}` at TP=8 — a spec-config change. On top:
  RS+AG measured **+14% vs one all-reduce** (701 vs 613 µs @64 MiB), so
  overlap starts in the hole.
- **Custom all-reduce at the 64 MiB prefill payload**: 2.7% slower than
  NCCL (630 vs 613 µs); one-shot catastrophic (2177 µs). The in-tree 512 KB
  one/two-shot crossover is right. (Custom two-shot *does* win 1.5× in the
  1–32 MiB band — irrelevant at single-pass 8K prefill, noted for upstream.)
- **NCCL tuning at 64 MiB**: `NCCL_PROTO=Simple` null; LL128 and symm-mem
  clearly worse (no NVLS on A100). Prefill AR runs ~190 GB/s bus bandwidth,
  ~80–85% of the practical NCCL ceiling — there is no big win here.
- **"Prefill AR is slow due to rank skew/imbalance"**: refuted by aligning
  all 87 calls across all 8 rank traces (spread ≤6%; the 742 µs mean was
  inflated by one first-call warmup outlier).
- **Decode AR (86 × 14 µs/step)**: one-shot is correctly selected; the cost
  is 6.0 µs fixed barrier+launch vs 1.9 µs payload, so chunk-overlap is
  strictly counterproductive; the 33% "waiting" is ≤4% timing jitter, not
  load imbalance (identical per-rank kernel counts; straggler rotates
  uniformly). Only fewer barriers would help, and TP structurally requires
  the two per layer. Double-buffering the IPC buffers (~0.29 ms/step) was
  deliberately declined: silent cross-rank corruption failure mode.

### Marlin (weight-only dequant GEMM)
- **Giving dense Marlin more CTAs/SM — refuted from both directions, with the
  tile held bit-identical in one of them.** Both flags kept default-off as the
  record.
  (a) `VLLM_MARLIN_DENSE_OCCUPANCY=1`, narrow the tile to fit more CTAs:
  **1.2–2.4× SLOWER** at M=1/6/8.
  (b) `VLLM_MARLIN_RIGHTSIZE_SMEM=1`, keep the tile and request only the smem
  the pipeline uses — the launch asks for the whole 166,912 B carveout against
  a 49,664 B footprint (`marlin.cu:527`) and the value passed into the kernel
  is dead code (`marlin_template.h:270`): **0.56× on `fused_wqa_wkv`**. ncu
  confirms the flag did what it was designed to do *and* that this is what hurt:
  identical kernel template, grid 108→324, theoretical occupancy 6.25%→18.75%,
  achieved 6.09%→13.12% — while **DRAM throughput FELL 19.87%→11.97%** and
  duration rose 16.1→26.6 µs. More resident CTAs fragment the streaming rather
  than adding useful memory parallelism.
  **Unifying mechanism: block count and split-K depth are the same variable.**
  Work is handed out as `div_ceil(k_tiles * mn_tiles, gridDim.x)`, so raising
  CTAs/SM always deepens split-K (4.5-way → 13.5-way in (b)). Marlin's 1 CTA/SM
  wide tile with deep cp.async staging is doing real work, and the smem
  over-request — though real — is benign. Confirmed from the other direction on
  the MoE kernel: raising CTA/SM only ever recovers ground lost by forcing a
  narrow tile, and at prefill M the MoE's CTA/SM cap is not even binding (its
  selected `{64,256,256}` needs 71,680 B, so smem limits it to 2 regardless).
  *Note for instrument users:* ncu correctly named the binding constraint
  (`Block Limit Shared Mem = 1`) and relaxing it still lost 44%. A profiler
  names the constraint on a resource; it cannot tell you that relaxing it helps.
  **(c) `VLLM_MARLIN_SPIKE_WARPS=1` — more warps per CTA at fixed split-K:
  0.68×.** The decisive one, because it removes the confound (a) and (b) shared.
  Required hand-adding the `{128,64,256}` instantiation, since the ladder could
  not otherwise express "more warps at fixed `thread_n`". ncu: grid (108),
  tile, split-K and smem **all identical**, occupancy exactly 2× (6.25→12.50%),
  and **DRAM throughput FELL 19.96→14.03%** while duration rose 15.97→22.69 µs.
  Correctness passed (14/15 bit-identical). Repeated: 0.68× / 0.69×.
  **Durable form — state it empirically, because three mechanism claims were
  made here and all three were retracted:** *every perturbation that raises
  resident parallelism on this kernel costs it bandwidth.* Four data points
  (a), (b), (c) and the three-generation Triton GEMV of `1d69354858`. The
  retracted mechanisms were "1 CTA/SM starves memory-level parallelism"
  (refuted by measurement), "split-K reduction *traffic* dominates" (refuted by
  arithmetic — 7.0% of bytes), and "block count and split-K depth are the same
  variable, and that is the mechanism" (refuted by (c), which held split-K
  constant and still lost). A fourth mechanism was deliberately not offered.
- **Dense-GEMV from-scratch rewrite — DECLINED on actionability, not on cost.**
  Both outcomes lead nowhere while the rewrite is unfunded: "structure is fine,
  dequant is cheap" implies a kernel nobody is funded to write; "dequant is the
  wall" is a closure. The one branch needing no new code — routing each shape to
  whichever *already-built* kernel wins — prices at **0.077 ms/step, under the
  0.079 ms 3σ gate at n=20**.
- **"cuBLAS beats Marlin at M=6 on 4/5 shapes" was an artifact end to end.**
  Two compounding measurement faults: the harness normalized GB/s on the **fp8
  payload for both arms** (hiding cuBLAS's true bandwidth), and a single
  L2-resident weight set flattered whichever arm streams more bytes — which is
  cuBLAS, at 2×. Both fixed (`--gemv-rotate`, default 8). Honestly measured,
  **Marlin already wins the decode critical path in aggregate by ~1.4 µs/layer.**
  cuBLAS remains 1.5–2.2× more *bandwidth*-efficient and still loses on wall
  time where it counts.
- **cuBLAS's 1.5–2.2× bandwidth advantage: UNEXPLAINED, DELIBERATELY LEFT.** A
  closure category distinct from "refuted" — the fact is real and no outcome of
  explaining it leads to an actionable change. Recorded explicitly so nobody
  later reads silence as evidence it was closed.
- **Lifting the MoE's CTA/SM cap** (`ops.cu:325-328`, capped at 2 for
  `thread_m_blocks > 1`): the cap is *not* what binds at prefill M. The
  selected `{64,256,256}` needs 71,680 B and is smem-limited to 2 on its own
  arithmetic; the cap merely coincides. Raising it hands selection to
  `{64,128,128}` (53,248 B, 3 CTAs) — i.e. it buys occupancy by halving
  `thread_n`, which is exactly the refuted failure mode above. The MoE also
  over-requests smem, but by 15% (82,432 vs 71,680), not dense's 3.4×, and
  right-sizing changes nothing: 166912/73216 = 2 either way. The dense
  finding does **not** generalize to the MoE path.
- **cuBLAS/dequant-bf16 route for decode**: step wall −0.028 ± 0.060 ms =
  null, despite cuBLAS beating Marlin on **3 of 5** shapes at M=6 (measured
  with rotated weight sets; a single L2-resident set flatters cuBLAS, which
  streams 2× the weight bytes, and reverses `fused_wqa_wkv`) — the two
  Marlin wins are the two largest critical-path shapes, and 72% of cuBLAS's
  remaining advantage lands on the aux-stream shared-expert pair, which is
  free. (The flag is a real prefill win: −34.9 ms TTFT, adopted.) The
  1d69354858 hybrid/exclusion list has nothing left to buy at decode.
- **MoE W4A8-INT8 codegen**: u4b8/int8 proxy ceiling at matched
  group_blocks: 0.83× at M=8192 (21.4 ms TTFT ceiling — under the 40 ms
  stop rule), 1.18× *slower* at M=1. These shapes are not MMA-bound; int8
  buys 17%, not 2×. Not built.
- **MoE dequant-to-bf16 + grouped GEMM at prefill**: 4.5× worse even
  granting a roofline dequant and zero gather. All 256 experts are touched
  per pass (~192 tokens each), so dequant cannot amortize; a persistent
  bf16 expert cache is ~69 GB/rank — impossible.
- **MoE tile-config forcing**: the autotuner already picks per-GEMM optimal
  configs at both prefill M and decode M (best forced delta 2.4 µs on
  gate_up, 0 on down).
- **`VLLM_MARLIN_USE_ATOMIC_ADD`**: dead on this hardware at any M — SM80
  has no native bf16 atomicAdd and the MoE path also fails the n<2048 gate.
  This same hardware fact independently blocks folding `moe_sum` (already
  at 1.41 TB/s) into gemm2's epilogue.
- **`act_and_mul` epilogue fold into w13**: structurally impossible — gate
  and up operands sit 256 columns apart and no Marlin CTA tile spans them
  (widest thread_n=256 covers all-gate/no-up). A w2-prologue variant is
  possible but ~0.159 ms/step (≈2σ at n=20) for hours of
  `marlin_template.h` surgery — declined.

- **Mixed-input CUTLASS mainloop for the MoE at prefill M**: the ideal
  ceiling at these expert shapes is **49.6% MFU, measured by two independent
  implementations that converge to 0.4%** — cuBLAS batched bf16 (`torch.bmm`)
  reaches 1119.1 µs / 59.0% of peak on GEMM1 and a hand-configured CUTLASS
  `DefaultGemmGrouped` bf16 reaches 1123.8 µs / 58.8%. Rule 28 cuts both ways
  and was checked: the reference was not itself leaving performance on the
  table. A pure-bf16 batched GEMM with no quantization, no dequant tax, no
  gather and perfect expert balance runs 1999.4 µs/layer over both GEMMs
  against Marlin's 2946.0 (33.6%). The wide tile wins in *both* formats at
  192-row experts (bf16 1123.8 vs 1218.1; int8 731.9 vs 784.4), so that is a
  shape property, not a format one. No
  mixed-input kernel can beat a zero-tax bf16 mainloop, so the whole route
  caps at ~32 ms and was closed priced-and-declined without a prototype.
  Separately, CUTLASS's SM80 mixed-input path has **no 4-bit
  specialization**: `mma_mixed_input_tensor_op.h`'s `FragmentShuffler`
  covers only 2:1 (16-bit MMA / 8-bit load), there is no `uint4b_t`, and the
  mixed-dtype examples are Hopper/Blackwell only. W4A16 on SM80 would mean
  writing new CUTLASS internals; only W8A16 is buildable.
- **Grouped int8 (W8A8) for GEMM1 — the estimate-decay closure.** The kernel
  works: CUTLASS `DefaultGemmGrouped` int8 on SM80 runs 256x(192x512x4096) in
  732.5 us against Marlin's 1294.4 = **1.70x real**, and `tb128x128x64` beats
  `tb64x128x64` despite wasting 25% of its rows on 192-row experts (tile
  efficiency beats padding waste; this is *independent of*, not a counterpoint
  to, the split-K occupancy refutation above). It was closed anyway, on the
  decay of its own estimate: **33 -> 24.8 -> 18.1 -> 12-15 ms, every revision
  downward**, as each layer of surrounding machinery was found:
  (a) **there is no gather today** — `marlin_moe.py:229` passes the unexpanded
  `[8192, 4096]` with `size_m=8192` and expands via `sorted_token_ids` *inside*
  the kernel, so grouped GEMM's contiguous-A requirement is a **new** 201 MB
  write/layer (-6.7 ms), not a halved existing one;
  (b) `get_cutlass_moe_mm_data` is **not compiled on SM80** — `CMakeLists.txt`
  intersects `9.0a;10.x;12.x` against `8.0` -> empty (packaging gate, but real
  work);
  (c) **CUTLASS 2.x grouped epilogues are scalar-only** —
  `DefaultGemmGrouped` takes a thread-level `EpilogueOutputOp` whose
  `alpha_ptr` is documented "pointer to accumulator *scalar*", and it does
  **not** compose with `Sm80EVT`, which is what vLLM's 2D `cutlass_scaled_mm`
  uses for per-row x per-column dequant. So the GEMM must emit int32 and
  dequant in a separate pass (-4.3 ms) or inside `act_and_mul` (-1.4 ms, and a
  shared-kernel change).
  Trap recorded for anyone who revisits: `compute_problem_sizes` is templated
  on `SWAP_AB` with `SWAP_AB_THRESHOLD = 64` tuned to *SM90's* dispatch, and
  below it the emitted problem is **transposed** — inert at 192 uniform rows,
  live the moment routing is ragged. Pin `SWAP_AB=false` or re-derive it.
  Standalone bench kept at `/root/optim/grouped_int8_bench.cu` (plain nvcc,
  ~90 s, no vLLM rebuild); design memo `/root/optim/INT8_GEMM1.md`.
- **Register dequant is not the MoE's MFU problem.** e2m1→bf16 is
  mask/shift/or (`dequant.h:434`), no LUT or PRMT, and the e8m0 group scales
  are converted to bf16 at load (`marlin_utils_fp4.py:645`) so no exponent
  arithmetic reaches the mainloop. Cost is ~16/M_tile of MMA time — single
  digits at any sane tile. Design effort belongs in the mainloop.

### Small-kernel / launch-count work (decode)
- **Fused single-launch `moe_align_block_size`**: 2.2× slower at the real
  36-pair workload. The existing `num_experts<=64` fused CUDA path is
  itself 1.3–2.1× slower than the 2-launch fallback — the gate fences off a
  worse kernel, it does not withhold a win. One CTA is the wrong shape for
  this problem in any language.
- **`_dsv4_topk` num_warps>1**: monotonically worse; one warp keeps the
  256-wide reduction in register shuffles, more warps force smem+bar.sync.
- **Decode indexer top-k glue fusion**: realistic 0.07–0.09 ms/step = at
  the n=20 floor, and it requires collapsing a 24-CTA grid onto 1 CTA (the
  exact moe_align failure shape). Retired.
- **mHC pair fusion / launch-count reduction**: the inter-kernel gap under
  graph replay is **64 ns**; fusing every pair is worth 0.005 ms/step and
  needs a device-wide barrier (split-K reduction sits between the pair).
  Launch count is not the cost. Also refuted in the same family:
  token-blocking the fused kernel (1.2–2.4× slower), bf16 `fn` weights
  (0–3%, not the 35% byte count predicts), register-resident sinkhorn (2×
  slower than the cross-lane shuffle reduce).

### Sparse attention / indexer
- **fp8-direct KV read in the sparse prefill kernel**: 6.8× WORSE,
  bit-identical output. SM80 has no fp8 convert; the 256-entry LUT turns
  vector loads into 16k gathers, 188→255 regs with spill.
- **Transposed QK dot** (to halve MMA padding): +39% — softmax becomes
  cross-lane axis-0 reductions and eats the saving.
- **Byte/locality reduction in the sparse prefill gather generally**: the
  gather is **issue-bound, not byte-bound** (L1-resident vs 8 MB pool:
  3.5% delta). Only reducing the number of row-loads pays. The kernel is at
  a local optimum after the branch's prior tuning.
- **Indexer-compressor warp sweep**: `num_warps=1` already optimal
  (rows=8 tile; the ratio-4 layers own the indexer — see §2 populations).
- **kv_compress C128 CTA-splitting**: effect real (64 CTAs pay ~38%
  per-CTA underfill) but the warp fix already took the pair to 2.3 ms
  total; a two-pass split recovers ~0.6 ms — not worth a rewrite of a
  correctness-sensitive quantizing kernel.
- **num_warps=16 on the sparse-decode kernel**: worse at every measured
  point. And the BLOCK_H=8 win (adopted) is NOT occupancy: 255 regs + 86 KB
  smem pin 1 CTA/SM at either tile width; the gain is the halved tile's
  loads/accumulators.

### DSpark / sampling
- **`VLLM_DSPARK_VOCAB_SHARD` env flag**: dead — zero consumers; its module
  is absent from the tree. The real switch is
  `speculative_config.use_local_argmax_reduction` (adopted: −0.455 ms/step,
  640/640 greedy tokens bit-identical).
- **"Optimize the drafter" instinct**: the draft backbone is ~5–6% of the
  step, Markov sampling ~1%, draft logits ~0.1%; target verify + rest is
  ~93% (43:3 layer ratio, confirmed by eager-proportions profiling). The
  step is spec-decode-shaped; the time is not.
- **Rejection/resample sampling logic**: all seven kernels total 0.043
  ms/step. The "sampling tail" was two GEMMs and an argmax, one now sharded
  (adopted), the other (vocab-parallel LM head) at 75% of HBM peak with
  0.019 ms/step of theoretical headroom.

## 2. Measurement rules (violating these produced wrong numbers *in this campaign*)

1. **Never quote ITL from few trials.** Acceptance swings 2.2–5.7 within an
   arm; ITL = step/acceptance inherits sd ≈ 1.4 ms vs step-wall's ≈ 0.1.
   Judge decode on step wall (steady_wall / n_chunks) + acceptance
   separately; ITL only as a derived headline. Gates: n=20 (3σ ≈ 0.05–0.08
   ms step, ≈ 1.3–2.5 ms TTFT).
2. **Token-attributed ITL only.** Chunk-gap timing overstates ITL by
   ~the acceptance length under spec decode. Count tokens via
   `logprobs.tokens` per chunk.
3. **Prove coldness; never trust it.** Scrape prefix-cache hit counters
   before/after. A fixed prompt seed silently warmed the cache and produced
   a fake 69 ms TTFT once. Random-token prompts also flatter acceptance
   ~35% vs natural text — confirm ITL claims on natural text.
4. **Kernel-µs ≠ wall time on this model.** The decode step is a serial
   layer chain; the shared-expert MLP overlaps on an aux stream and is
   FREE. Weight every per-shape delta by its stream before predicting wall
   (a change won 4/5 shapes, won big in summed µs, and moved the wall by
   0.00). Conversely, exclusive time UNDER-counts a kernel overlapped by a
   *shorter* chain. Use the step-marker + stream-structure method in
   `/root/optim/FUSION.md`.
5. **Microbench decode kernels through CUDA-graph replay.** An eager loop
   with trailing sync measures the CPU enqueue path (3–8× off for <10 µs
   kernels) and will reject good candidates and accept bad ones.
6. **Never combine absolute numbers from two bench invocations.** Clock
   state drifts absolute µs ~17% run-to-run; within-run ratios hold to 3%.
   Measure both operating points in one invocation.
7. **Match every harness axis to the traced launch geometry — then sweep
   the pinned ones.** The sparse-decode bench pinned `block_h=16` and
   `batch=1` "to match serving"; production runs block_h(bug)=16 with
   num_queries=6. Pinning to what serving picks can only confirm serving;
   it hid a 2.2× larger win and the bug itself.
8. **"Per layer" is never simply 43 on this model.** Populations: 21
   ratio-4 layers (these own the indexer), 20 ratio-128, 2+3 neither. An
   *exact* 21 call-count identifies the indexer population; an average near
   21 does not (21 vs 20 is one apart). When the population matters, read
   the construction site (`attention.py: if self.compress_ratio == 4`).
   This caused four separate errors in one day.
9. **Launch-merging taxonomy.** Neighbour merge buys only the inter-kernel
   gap (64–95 ns under graphs — nothing). Absorbing work into an
   already-running kernel buys the absorbed kernel's duration — but check
   tile geometry first. Barrier'd collectives have a fixed per-call sync
   tax; only fewer calls helps, and if TP requires them you are done.
10. **When a change removes 7/8 of the work, the residual's efficiency is
    second-order.** Don't transfer GEMM-scaling intuitions to
    grid-per-row kernels (measured: ÷8 sharding yields 12.2% residual on
    the indexer logits kernel — better than linear).
11. **Numerics-preserving claims get token-identity gates.** Capture greedy
    token ids (n×128 tokens) on control and treatment; bit-identical or it
    isn't "numerics-preserving". Acceptance-indistinguishability alone
    resolves nothing below Δ≈1.2 at n=20.
12. **Ops infrastructure traps**: an editable install means the working
    tree IS production — unverified edits to shared files can ride into
    someone's A/B (announce venv use, not just GPU use; builds swap .so
    files under running processes). "The extension imports" does not mean
    "its op schemas match the source" (a stale .so served 10 of 126 ops).
    `pgrep -f` inside a watcher matches the watcher itself.
    `CCACHE_BASEDIR=/` + CMake/Ninja's per-build temp CWD = permanent 100%
    cache miss (fixed; unchanged rebuild is now ~35 s).
    **Never pipe a long benchmark through `tail`/`head`**: the filter buffers
    until EOF, so a backgrounded run yields an empty file, and the pipeline's
    exit status is the *filter's* — "exit 0" says nothing about whether the
    program succeeded. Redirect to a file and filter afterwards. Cost two
    runs in one session, one of them an 8-minute granted GPU window.
    Related: **a benchmark arm valid at one M is not valid at another** —
    `--kernel attn-input-gemm` builds 8 rotated weight sets *and* fp32
    references, which is seconds at M≤2048 and >1 h of single-core CPU at
    M=8192. When an arm only has to yield one number at a new operating
    point, write the three-line direct measurement instead of reusing it.
13. **Restating an inference more confidently is not confirming it.**
    Generate hypotheses from counts/traces; close them only at the
    construction site or with a discriminating measurement.
14. **Price a sharding change by the layer's output-bytes-to-FLOPs ratio,
    not by its redundant FLOPs.** Un-replicating a `disable_tp=True` GEMM
    does not recover the redundant FLOPs: all ranks compute *concurrently*,
    so the wall saving is 7/8 of the **per-rank** time, and sharding
    *creates* an all-gather that did not exist. Indexer query-sharding is
    worth ~39 ms because it emits tiny top-k indices against ~37 ms of
    eliminated compute (3.53 ms AG); `fused_wqa_wkv` is worth ~8 ms because
    it emits 24 MiB against a 440 µs GEMM. Same technique, opposite
    economics. Costing #27 by FLOP census produced 38 ms against a true
    ~8–18 ms.
15. **A roofline does not transfer across operating points.** "At the
    roofline" in this campaign has meant *local* per-kernel optima at one M.
    The model-level floors are elsewhere: prefill runs 15.8% MFU of a
    211 TFLOP workload, and 49% of the machine's FLOPs are redundant
    replication that no per-kernel view can see. Recompute the model roofline
    (`/root/optim/roofline.py`) when a structural item lands — it is a moving
    target, not a constant.
16. **`pgrep -f <pattern>` inside a shell wait-loop matches the loop's own
    command line**, so the condition never clears and the waiter spins
    forever. Cost real minutes twice here, once while watching for a
    benchmark that had already finished. Wait on a PID (`kill -0 $pid`) or
    use a pattern the waiter itself does not contain.

## 3. Where the remaining time is (measured floors, config v6)

- TTFT@8K ≈ 534 ms: MoE Marlin 126 ms (§1 — irreducible on SM80), dense
  cuBLAS ~76 ms (compute-bound, 100–178 TFLOP/s), sparse attn + indexer
  ~90 ms post-fixes (indexer query-sharding, ~−39 ms, is the one live
  lever — task #19), NCCL AR ~54 ms (at ceiling, overlap routes closed),
  mHC ~62 ms post-fix (mhc_post and row-sqrsum at 1.75–1.99 TB/s = HBM
  ceiling).
- Decode step ≈ 12.15 ms: ~93% is the 43-layer target-verify chain whose
  major kernels are at their measured roofline or refuted above; ~0.4
  ms/step of accepted-but-small items remain (see /root/optim/RESULTS.md).

## 4. Rules added in the structural round (v7–v10)

14. **Price a sharding change by the layer's output-bytes-to-FLOPs ratio, not
    its redundant FLOPs.** Removing 7/8 of replicated work only pays if the
    all-gather that replaces it is small relative to the GEMM it accelerates.
    Indexer top-k indices: tiny output, huge win (−39 ms). fused_wqa_wkv: 24
    MiB output vs a 440 µs GEMM, +7.8 ms net. A 171 µs GEMM cannot pay for a
    168 µs all-gather at any M. Corollary: the same technique has opposite
    economics at different operating points — decode partials that cost 98 KB
    cost 134 MB at prefill T.
15. **When a FLOP/byte count won't reconcile, check the label before
    re-deriving the arithmetic.** Twice in one day a "wrong number" was a
    right number attached to the wrong operator name (ROOFLINE's "mHC prenorm"
    was the compressor's fused_wkv_wgate; the real prenorm GEMM is 8× smaller
    by FLOPs and 24× bigger by achievable ms because it is bandwidth-bound).
16. **After two refuted hypotheses on one kernel, fund an instrument, not a
    third hypothesis — and treat the instrument's diagnosis as a hypothesis
    too.** ncu correctly identified a 3.4× smem over-request pinning 1 CTA/SM;
    right-sizing it made the kernel 44% SLOWER (block count IS split-K depth
    in Marlin's work distribution). A profiler names the binding constraint on
    a resource; it cannot tell you that relaxing it helps.
17. **A one-line source diff is not a one-mechanism diff.** A single changed
    address computation swung register allocation by 54 regs and flipped the
    sign of the measured effect between two otherwise-identical experiments.
    Attribute with a one-switch-per-rung ladder, each rung measured in the
    same invocation, before naming a mechanism.
18. **Issue count is a roofline term the roofline doesn't have.** The
    sparse-prefill gather is issue-bound: constant-true tile masks cost 9.4%
    of the kernel while appearing in no FLOP or byte account
    (VLLM_SPARSE_PREFILL_EXACT_TILE, adopted, bit-identical). When a kernel is
    at neither the compute nor the bandwidth ceiling, audit what it *issues*.
19. **Two sharding features over one tensor must not assume each other's
    partition.** VLLM_UNREPLICATE_ATTN_GEMMS uses an even split; the indexer
    query shard uses an uneven no-padding split. They agree only when
    n % tp == 0. The gather between them is what keeps them composable;
    "skip the gather and consume the shard directly" produces wrong top-k
    indices with no crash, on ragged batches only. Unify the partitions first
    or keep the gather.
20. **Report "condition met" by pointing at the line that implements it, not
    at the evidence artifact for it.** A fully-documented threshold table
    shipped with the threshold never wired; the reporting standard caught it
    only when a log line failed to resolve the constant's name.

## 5. Final round: v11.1 and the closing refutations

- **Batching two all-gathers by concatenating their buffers**: +6.43 ms TTFT
  at 9.9σ — REVERTED. `torch.cat` plus the two post-gather `.contiguous()`
  splits cost four full-size HBM passes (~400 MiB/layer) against one saved
  161 µs NCCL launch. Only a copy-free grouping (ncclGroupStart/End) avoids
  that, and it was then priced-and-declined: the per-call cost decomposes as
  75 µs fixed + 4.6 µs/MiB, so the gather worth attacking (82 MiB trio) is
  83% transfer — grouping can only remove fixed cost from calls that are
  already cheap (ceiling ~1.9–3.2 ms vs a 2.0 ms gate floor).
- **Fast-scan's serving "null" was contamination.** Measured ~1σ against a
  control that co-carried the (rejected) concat merge; re-gated on the clean
  tree it is a real **−3.40 ms at 4.9σ**. Rule: a null measured on a tree
  carrying any other change is not a null — re-gate suspicious nulls clean.
- **Prefill graph capture** is gated behind the per-layer sync removal
  (`cudaStreamSynchronize` is illegal inside a capture — CUDA rule, not vLLM
  policy) and then still blocked by the uniform-decode dispatch gate, the
  512 capture-size cap, and data-dependent ragged shapes. The host-latency
  pool it would target is ~10.8 ms; everything else host-side is the
  request-setup ramp (~4 ms of diffuse Python with no concentrated target
  and nothing to overlap on a cold request).

## 6. Rules earned in the final round

21. **Ask what the implementation adds that the idea doesn't mention.** The
    priced quantity was real three times (redundant FLOPs, payload bytes,
    saved launches) while the governing cost was elsewhere (the collective,
    the accumulator precision, the concat copies). Price the delta the code
    introduces, not the quantity the idea is about.
22. **A figure is true at one operating point of one implementation.** A
    6 µs custom-AR fixed cost was carried onto an NCCL all_gatherv (~40–64 µs
    measured); a prefill roofline was carried onto decode shapes; a decode
    partial-buffer cost was carried onto prefill T. Check the primary
    artifact at the actual operating point — and note the rendered summary
    table is exactly where a row goes missing (the 0.5 MiB AG point existed
    in the raw JSON but not the doc table).
23. **State findings absolutely or control both variables.** "Idle grew"
    compared mismatched profiler settings; the correction "idle is flat"
    compared mismatched configs. A corrected measurement does not license an
    uncontrolled comparison. Profile with the same profiler flags as the
    baseline (with_stack changes eager-gap numbers materially), and join GPU
    events to launches via BOTH cuda_runtime and cuda_driver or ~half the
    correlations silently land in the wrong bucket.
24. **An attribution is not a duration.** "9 memcpys at 213 µs" was 213 µs
    of idle *attributed to* 1–5 µs copies at the end of a host-busy window.
    Read the event's own duration before chasing it.
25. **A flag that touches distributed state must stay inert single-process**
    (guard on model_parallel_is_initialized() before get_tp_group()), and
    must be tested with the flag ON as well as off — the kernel suites run
    single-process and will pass with a flag that crashes serving.
26. **Enforce couplings in code, not in serve.sh comments.** The prenorm
    shard is gated on `sqrsum is None` — the fold's own signal — so the
    two-gather degenerate config is unrepresentable rather than documented.
27. **Monkeypatching `...kernels.mhc.triton.<name>` targets site-packages
    triton**: the package's `from .triton import *` rebinds the name. Use
    importlib.import_module on the full path.
28. **Never benchmark a hardware *capability* through an untuned reference
    implementation.** `torch._int_mm` measures 58 TOP/s on A100 — under 10% of
    the 624 TOP/s int8 peak, and 3.5x *slower* than bf16 at the MoE expert
    shapes. Taken at face value it says "int8 is slower here", which is the
    opposite of the truth: vLLM's own CUTLASS int8 path is 6x faster on the
    identical shape and reaches 58% of peak. The reference told us about
    itself, not about the machine.
29. **Always name the denominator.** A cost is only large or small relative to
    the alternative it buys. The int8 weight format costs +111 us inside
    Marlin's mainloop, which reads as a blocker until set against the 1035 us
    the mainloop itself is worth — 11%, not a blocker. Reporting the numerator
    alone nearly closed a live ~40 ms route (#33). Same failure family as
    rule 14: price what the change *adds*, against what it *buys*.
30. **The MoE's cost is a shape problem wearing a format problem's clothes.**
    mxfp4, int8, u8b128 and bf16 all behave well on the aggregate expert shape
    and all degrade on 192-row experts; only the amount differs. Every
    format-swap route in this document failed at the grouped structure, not
    at the numeric format — which is why the one route that survived (#38)
    changes the *kernel's handling of the grouping*, not the encoding.

31. **A cross-invocation delta under ~20% is not a result until repeated —
    and a plausible mechanism makes a cold-baseline artifact MORE convincing,
    not less.** This bench drifts ~17% run-to-run on idle GPUs (absolute
    numbers move; ratios within one invocation hold to 3%). The
    `VLLM_MARLIN_SPIKE_WARPS` sweep's first run showed three shapes *winning*
    1.16–1.19×, with a mechanism ready to explain it — those shapes take a
    different production tile, so the spike halved their split-K rather than
    doubling their warps. The repeat showed **1.00× on all three**: the run-1
    *baselines* had been cold (`indexer.wq_b` 14.1→11.9, `wq_b` 11.8→10.0,
    `shared_down` 7.2→6.1). A fake 1.2× on three shapes was one report away
    from being funded. The explanation arriving *with* the number is precisely
    what stops you re-running it — same failure shape as the three retracted
    mechanism claims in §1. **Put both arms in one invocation where possible,
    and repeat anything that crosses a decision boundary.**

32. **Pre-registered bands must not conclude a capability limit from an
    untuned reference implementation — that is rule 28 applied to your own
    gate.** The #41 audit registered "if the from-scratch spike measures
    ≤600 GB/s, the wall is not the dequant" — inside the very document whose
    premise is rule 28, with a first-cut hand-rolled kernel cast as the
    reference. Two faults: a day-one kernel underperforming a mature library
    refutes the *kernel*, not the design; and the inference did not follow at
    all, because **that experiment never varied dequant** — it varied
    implementation and held dequant at zero. The fix was free: template the
    weight dtype and gate on the **bf16÷fp8 ratio**, which is robust to how
    good the kernel is. *Check that your gate varies the term it claims to
    measure, and that the instrument is strong enough to carry the conclusion.*

28. **Validate numerics properties on captured real tensors; when that is
    impossible, construct inputs with the real spread.** A distribution that
    makes the property trivially true makes the test unable to fail. Three
    instances in two days, all the same shape: the int8/e4m3 format decision
    table (calibrated on synthetic outlier injections, and on real activations
    its own statistic spans the entire table and points at the format that is 4x
    worse); the two-shot's accumulation-order stability (bit-stable on
    random-normal data, and DIFFERENT on all 5 captured all-reduces, because
    real per-rank block scales diverge up to 120x where synthetic ones agree to
    ~2x); and the harness written to verify that an order flag was live, which
    could not fail on `torch.randn` for exactly the same reason. Real
    activations of this model carry outlier structure that no clean
    distribution reproduces. This is rule 3 ("random-token prompts flatter
    acceptance; confirm on natural text") extended from prompts to tensors.

Final standing config v11.1 (serve.sh): cold TTFT@8K **467.2 ms** (2.33× from
1089.9), decode step **12.16 ms**, ITL ≈ 3.1 ms. Eighteen adopted-or-record
commits after `4adc46ce38`. The remaining gap to 150 ms / 1.5 ms sits below
the measured per-component floors of this architecture on this hardware; the
levers that would move it (expert config, drafter config, native-FP8
hardware, vocabulary size) are outside this campaign's constraints.

## 7. The v12 round and the closure of the format class

- **int8-compressed prefill all-reduce — ADOPTED (v12, −18.75 ms at 31.7σ,
  gsm8k@16-shot 0.97 vs 0.98 control).** The one quality-gated numerics change
  that survived its battery. Facts that made it work: int8 blockwise beats
  e4m3 4× at equal wire bytes on real activations (the synthetic
  outlier-inversion crossover does not transfer); the quantize floors at
  1.04× traffic (a 455 µs figure was a tile-sweep artifact); the dequant
  fused into a bandwidth-bound consumer is net-positive; block width is
  throughput-free. Watch: a quality gate must EXERCISE the changed path —
  5-shot gsm8k prompts sit under the 2048-token threshold and would have
  gated nothing; 16-shot (~2.6k tokens) engages it.
- **Native int8 (W8A8) MoE — closed at every granularity.** Dense combined
  1.11× (GEMM2's k=256 loses 0.70×); per-expert 2D calls collapse to 3% of
  int8 peak at 192 rows (fixed-cost-flat from 192 to 1536 rows); sequential
  loop 18.8× dead. GEMM1-only via CUTLASS SM80 grouped int8: the kernel is
  real (1.70× vs Marlin at true grouped shapes, wide tb128 tile beats 25%
  padding) but the machinery ate it — no-gather-today means materializing
  contiguous per-expert A is NEW work (6.7 ms), moe_data isn't compiled on
  8.0, and CUTLASS 2.x grouped epilogues are scalar-only (no Sm80EVT), so
  production dequant costs 4.3 ms standalone or a shared-kernel change.
  Estimate decayed 33→24.8→18.1→12–15 ms, every revision downward.
- **Rule 28 — track your estimate's derivative.** Four strictly-downward
  revisions with a nameable cause ("pricing the mechanism, discovering the
  machinery afterwards") is evidence about the unexamined remainder, not
  four accidents. Apply the discount the trajectory implies, and let the
  author's own bias analysis outrank the author's latest number.
- **Rule 29 — never benchmark a hardware capability through an untuned
  reference implementation** (torch._int_mm at <10% of int8 peak would have
  inverted the W8A8 verdict; vLLM's CUTLASS path is 6× faster on the same
  shape).
- **The MoE's cost is a shape problem wearing a format problem's clothes.**
  mxfp4, int8 and bf16 all behave on the aggregate shape and all degrade on
  192-row experts; only the amount differs. This closes the format lever
  class permanently: any future proposal must change the SHAPES (expert
  count/size, routing structure) — which is model config, outside this
  campaign's constraints.

Final standing config v12: cold TTFT@8K **449.6 ms** (2.42× from 1089.9),
decode step **12.18 ms**, ITL ≈ 3.1 ms, gsm8k unchanged. Nineteen
adopted-or-record commits after `4adc46ce38`.

## 8. The rewrite round's final closures

- **maxnreg three-arm gate (sparse decode)**: 128 (2 CTAs/SM + 64 B spill)
  = +0.236 ms/step at 8.8σ REGRESSION; 168 (spill-only control) = null.
  Occupancy is conclusively the mechanism — the fifth independent
  confirmation across two kernel families that raising resident parallelism
  costs these kernels bandwidth. The smaller-footprint rewrite is dead with
  mechanism. The three-arm design (treatment + confound-control) is the
  template for reading a loss unambiguously.
- **Chunked AR/compute overlap (#39)**: closed by its own ideal-case
  arithmetic — 43 × 379 µs × (C−1)/C sits under the bar at every C before
  any mechanism cost. Measured anyway at C=2: overlap is actively NEGATIVE
  (−14.8% concurrent efficiency; the custom AR's 36 blocks contend with the
  GEMM for SMs — a third resource neither the NVLink nor the HBM roofline
  names). The MoE half died separately on rows-per-expert chunking
  (+34–38 ms at C=2 vs a 16.5 ms gross). Two-shot int8 AR accumulation
  order is owner-dependent AND OBSERVABLE ON REAL DATA (per-rank block
  scales diverge up to 120×) — synthetic-normal data hides it; order
  stability claims must be validated on captured tensors. The clean
  absolute-order implementation costs +12.22% (the ordering itself, not
  spill) and stays behind VLLM_AR_INT8_ABSOLUTE_ORDER, default off, as the
  only order-stable implementation should shard boundaries ever change.

31. **A cross-invocation delta under ~20% is not a result until repeated —
    and a plausible mechanism makes a cold-baseline artifact MORE
    convincing, not less.** A fake 1.2× on three shapes arrived with a
    ready explanation; the repeat showed 1.00× on all three.
32. **Pre-registered bands must not conclude a capability limit from an
    untuned reference, and must vary the term they claim to measure.**
    Caught inside the audit whose premise was the same rule.
33. **Compute the ideal-case prize before pricing the mechanism.** The
    cheapest possible test — one line of arithmetic on the pool times the
    fraction actually takeable — and the one most often skipped on the item
    its author expects to survive. Apply it symmetrically: the half you
    expect to die AND the half you expect to live.
34. **Validate numerics properties on captured real tensors; when
    impossible, construct inputs with the real spread.** A distribution
    that makes the property trivially true makes the test unable to fail
    (three instances: the format table, order stability, and a verification
    harness that could not fail on randn).

FINAL STANDING, end of campaign rounds: config v12 — cold TTFT@8K
**449.6 ms** (2.42× from 1089.9), decode step **12.14–12.18 ms**, ITL
≈ 3.0–3.1 ms, gsm8k unchanged. Both boards fully closed: every mechanism
class (kernel tuning, kernel rewrites, formats, communication incl.
compression, sharding, fusion, overlap, graph capture, scheduling,
front-end) is either adopted, or closed by measurement with its mechanism
named, or closed by convergent independent implementations. The remaining
distance to 150 ms / 1.5 ms requires changing the model's shapes, the
speculative config, or the silicon.

## 9. The gather's final closure (speed round)

The sparse-prefill gather's 266 µs/call gap over its issue floor is closed in
the profiler-named resources: summed over all 620 memory instructions the
L2 excessive-sector count is exactly ZERO (the "9.8 of 32 bytes/sector" rule
was predicated/partial-lane masking, not a stride — and the sector-alignment
mechanism offered for it belonged to a kernel this one never reads: the
gather consumes the dequantized bf16 workspace at 1024-B rows, not the
584-byte fp8 cache). The 42.5% execution-dependency stall has no hot spot
(largest single site 0.3% of samples) and its largest block is the flash
chain's true serial dependency; ncu's own remedy is "more active warps",
i.e. the ILP and occupancy framings are the same arithmetic, and occupancy
is five-times refuted. Bandwidth unsaturated throughout (48.7% mem-busy,
97.4% L2, DRAM active 4%): the kernel is what flash attention structurally
is at this shape. Config-knob battery same round: prefix-caching OFF is
+3.78 ms WORSE at 7.3σ (the caching-on allocator is cheaper for fresh
blocks even at 0%% hits), max-num-seqs 8 is +2.89 worse, block-size 256
null — v12's serving config is measured optimal.

35. **When a profiler rule quotes a ratio, find the counter that would have
    to be nonzero for the rule to be true.** Est. Speedup is a heuristic
    over aggregate ratios; the per-instruction `excessive` counters are the
    ground truth, and checking them removed a 33.8% headline plus an
    invasive KV-format change in ten minutes.

- **FP32 fusion on the gather — closed by counting.** Static SASS mix:
  FMUL 211 / FADD 10 / FFMA 2 / HMMA 64. FFMA needs a multiply feeding an
  add; this kernel's additions are the HMMA accumulate, into which an FMUL
  cannot fuse, so at most 10 of 211 multiplies could ever pair and 2
  already do. ncu's 14.6% "convert pairs" suggestion had no pairs.
  (Static counts, not dynamic — the structural point is trip-count-
  independent.) Rule 35 generalised: **check the premise the rule's REMEDY
  assumes, not just the ratio it reports.** The gather is now closed in
  every named resource: issue width, sectors, row-sharing, dependency
  structure, and instruction mix.

## 8. Post-campaign cleanup (behavior-preserving; /simplify pass)

The tree was reviewed by four independent cleanup agents (reuse /
simplification / efficiency / altitude) and the surviving findings applied.
Nothing here changes gated behavior; every touched suite re-run green.

- **Flag surface consolidated.** The five token-identical optimizations
  (VLLM_INDEXER_QUERY_SHARD, VLLM_UNREPLICATE_ATTN_GEMMS,
  VLLM_SPARSE_PREFILL_EXACT_TILE, VLLM_SPARSE_RAGGED_FAST_SCAN,
  VLLM_MHC_PRENORM_SHARD) are now **default-ON** with `=0` opt-outs — each
  self-gates on TP/shape and keeps its fallback. QUERY_SHARD_QPATH merged
  into QUERY_SHARD (they only ever shipped together); dead
  VLLM_DSPARK_VOCAB_SHARD deleted. serve.sh now exports only the three
  flags that trade something: DEQUANT_BF16 (VRAM), POST_FUSE_SQRSUM
  (numerics, gated), AR_INT8 (numerics, gated).
- **One partition rule.** The `base + (r < rem)` split lives once as
  `balanced_row_counts` / `balanced_row_bounds` (vllm/distributed/utils.py);
  indexer, attention-GEMM shard, and prenorm shard all derive from it.
  Verified bit-equal to all three old spellings over n∈[0,300),
  tp∈{1,2,4,8,16}.
- **Host-side waste removed** (measured by the efficiency agent):
  record_function scopes in the DSpark draft step now cost nothing unless
  profiling is configured (~18 µs/step); indexer Q-path row ranges derived
  once per step in the metadata builder instead of per layer (~60 µs/8K
  prefill); env reads in per-layer paths cached
  (construction-time attrs in DeepseekV4Attention, functools.cache
  elsewhere).
- **Reuse:** blockwise dequant now calls `block_dequant` (int8_utils),
  exclusion matching calls `is_layer_skipped(skip_with_substr=True)` — both
  verified bit-identical; `_decode_block_h` (byte-duplicate of the prefill
  tile rule) merged into `_sparse_block_h`; hand-rolled smem query replaced
  with `get_max_shared_memory_bytes()` (same attribute, verified equal).
- **Altitude:** input-GEMM fusion moved to the loader's AttentionLayerBase
  pass (`DeepseekV4Attention.process_weights_after_loading`) so it runs
  inside `device_loading_context`; dead `mhc_post_sqrsum_int8_tilelang`
  wrapper deleted and its scale-shape check inlined at the live call site;
  dspark layer call made keyword-explicit against arity churn.
- **Findings REJECTED with reasons:** per-call `std::getenv` of
  VLLM_AR_INT8_ABSOLUTE_ORDER is load-bearing (verify_default_rotated.py
  toggles it mid-process; comment now says so); routing the fused mhc_post
  through the allocating wrappers would add hot-path allocations; the cuh
  int8-AR template merge and the tilelang mhc_post 2x2 kernel dedup are
  measured-hot-kernel refactors whose no-op-ness cannot be proven without
  re-gating — recorded, not applied.
- **Trap for stubs:** the torchrun vocab-shard worker builds DSparkSpeculator
  via `__new__`, so new instance attributes read by `_sample_sequential`
  must have class-level defaults (caught by the test, fixed that way).

### Round 2 (/simplify, second pass)

- **Chunk sharding made rank-uniform by construction.** The old
  `shard_chunk_specs_by_query` dropped sub-chunks a rank owned no rows of;
  a trailing sub-chunk with fewer rows than ranks (possible via the logits
  budget's `chunk_m % max_q`) would give ranks different chunk lists —
  divergent per-chunk all-gather participant sets (hang), and the
  counts-zip paired sizes against a differently-filtered list. Now a
  `ShardedChunkSpec` carries `shard_row_counts` and `gather_start` WITH the
  slice (no zip, no inverse arithmetic in the consumer), and sub-chunks
  with fewer rows than ranks stay replicated on every rank. Pinned by
  test_chunk_list_is_rank_uniform + the reworked tiny-range test (18-test
  suite). Never observed in serving (8K single-request prefill produces no
  tiny sub-chunk); found by two independent review agents.
- **Q-path decline on mixed batches**: a batch with any replicated chunk
  declines Q-path sharding entirely (compute every row) — conservative and
  safe; documented in indexer_q_row_ranges.
- **NOTE: V3.2 shares this builder.** VLLM_INDEXER_QUERY_SHARD default-ON
  applies to DeepSeek-V3.2 configs too; the mechanism is model-agnostic
  (replicated indexer, rows partitioned, bit-identical by the same
  argument) but only DSv4 is serving-verified.
- **Fused Markov sampler**: step-invariants hoisted out of the 5-step loop
  (−53 µs/draft step measured, output-identical A/B) on the uncaptured
  path; SM count via num_compute_units.
- **d2t rides in MarkovFusionOperands** instead of being getattr'd off the
  model — the offset-form assumption is now part of the hook contract.
- **mhc_post_tilelang grew x_scales=None** and dispatches internally
  (same contract as mhc_fused_post_pre); model.py's _mhc_post_any deleted.
  The fused function's raw-kernel imports are all `_tk.`-qualified so they
  can never shadow this module's same-named wrappers again.
- Reuse: benchmark's Marlin block_m ladder now imports
  `_ladder_block_size_m` (its "not exposed" justification had gone stale —
  this branch exposed it); `_e2m1_decode` replaced by nvfp4_emulation's
  `_e2m1_inline` (verified bit-identical over all 16 code points);
  `_fp8_block_dequant` delegates to `block_dequant`; env save/restore in
  the torchrun worker uses `set_env_var`.
- validate-local-argmax dedup: the base speculator takes a
  `_local_argmax_hooks` class attribute; DSpark's 28-line override is gone.
- All four rotted `custom_all_reduce.py:NNN` line cites (three different
  wrong numbers for one site) replaced with the symbol name.
- **Skipped with reasons**: unconditional clamp in get_top_tokens (adds
  work to non-DSpark users; env now read once instead); fixing the warm-up
  at the source (zeros_like in CustomAllreduce — shared-infra change, would
  obsolete the whole workaround family; recorded); int8_all_reduce as a
  CustomAllreduce method (shared class + gated numerics); fusion
  eligibility moving from head to kernel module (single head exists);
  csrc ceil_div/remainder reconciliation (needs rebuild + re-gate);
  torchrun harness dedup via eplb_utils.distributed_run (working 8-GPU
  parity test, mechanism change); _encode_e4m3fn_u8 module move (churn).

## 9. Backport from wtdcode/vllm-backport@dsv4-a6000-opt, and the bug it flushed out

Reviewed all non-CI commits of the A6000 fork (built on this branch's own
backport). ADOPTED: the KV-zeroer corruption fix (cross-group block-id
mixing + stride-as-extent conflation; both defects confirmed present here;
kernel fix hand-ported onto our flattened-chunk zeroer with per-group
metadata and host-precomputed chunk base/len; 3 new regression guards),
the top-k determinism/correctness rewrite (exact (value desc, index asc)
selection; fixes tie-buffer overflow and multi-CTA stale-histogram bugs;
selection verified against a stable-sort reference at gsm8k shapes incl.
nonzero cu_seqlen_ks -- NOTE the kernel contract is ks-RELATIVE indices),
the DSv4 chat-template trailing-system fix, and the shm-sizing fix.
ADOPTED WITH FLIPPED DEFAULTS: deterministic moe_align (their default-on
costs +15 ms TTFT / +0.6 ms ITL HERE -- 61 MoE layers, decode included --
against an ulp-only reproducibility win; default OFF, knob kept) and the
fixed decode split-k (their default 8 vs the 16 our adaptive heuristic
picks at the gated 8K decode shape; default 0 = adaptive). SKIPPED:
island-aware allreduce (PCIe topology; we are NVSwitch), the parser
engine, CI/readme.

**The backport's gsm8k gate caught OUR bug, not theirs.** gsm8k-200@16shot
came back 0.825 vs the 0.97 control; bisection exonerated the backport --
the cause was round 2's `gather_token_start` carrying the chunk-spec's
REQUEST-GROUP-RELATIVE start where the top-k buffer write needs BATCH
tokens. Every single-request gate (TTFT bench, token capture, serving
sanity) has group base 0 and cannot see it; lm_eval's num_concurrent=8
put later requests' gathered indices at wrong buffer rows. Fixed in the
builder (token_start - rank offset within chunk = absolute chunk start;
identical to the arithmetic the round-2 refactor replaced) and pinned by
test_gather_start_is_batch_absolute_for_every_request_group.

Rules earned:
36. **A sharding/coordinate change is not gated until a CONCURRENT
    quality run has seen it.** Single-request measurements have every
    request-group base at 0, which hides exactly the offset class of bug.
    gsm8k at num_concurrent=8 is the cheapest standing probe.
37. **Never mutate the working tree while a server is loading.** Workers
    import at spawn; a mid-load edit gives scheduler and workers different
    module versions, and the resulting crash (or silence) tests a tree
    that never existed.

Final standing after backport: cold TTFT@8K 450.4 ms (n=5), ITL p50
~2.2 ms, gsm8k 0.97/0.97 -- the ~1.7 ms TTFT delta vs pre-backport is the
top-k correctness rewrite plus per-group zeroer launches, accepted as the
price of the corruption fixes.

## 10. The c64@256k campaign

Baselines re-measured honestly first: `vllm bench serve` defaults to the
model's generation_config (T=1.0) and random-token prompts collapse DSpark
acceptance in BOTH directions (1.0 at T=1.0, inflated ~4.5 at greedy);
serving baselines must pin the sampler and use natural text (rule 38). The
"2.94x resident" KV log line is a planning bound; sparse/SWA reality is
~27 resident 256k requests (~1050 blocks each), so c64@256k cold TTFT is
SERIALIZED PREFILL (FCFS-optimal), not capacity-bound. Median ITL on this
workload is a population artifact (decode-tail steps vs 1224 ms
prefill-carrying steps) — report mean+p99+populations (rule 39).

Calibrated model: step = F(C) + c(d)*chunk_tokens, with
F(C) = 13.1 + 4.51*C ms (per-resident decode work; two independent
instruments agreed 0.6%) and c(d) = 53.9 + 0.156*d_ktokens us/token.
40.5% of F was the decode indexer logits kernel: 4% of HBM, 8x replicated,
zero batching. Spec-K reduction is impossible (dspark_block_size=5 floor).

ADOPTED (256k config, serve_256k.sh):
- DEQUANT_BF16 at 256k (arm N1): uniform ~6% on TTFT/TPOT/ITL/tput at c64;
  its capacity cost is one resident of ~27. The original reason to disable
  it at long context was the planning-bound misread.
- **Decode-side indexer request-shard** (VLLM_INDEXER_DECODE_SHARD_MIN_REQS,
  default 4): partition decode query groups across TP ranks via the shared
  balanced partition; reassemble per ratio-4 layer with a capture-legal
  sum-of-zeros all-reduce (fp32-exact for indices < 2^24) + clamp barrier
  against the warm-up-garbage branch. Same-tree A/B at 200k ctx:
  F = 13.26 + 4.14*C (off) -> 18.22 + 1.90*C (on): slope -54%,
  F(22) 104 -> 60 ms (-42%), break-even C~2. gsm8k-200@16shot nc=8
  ENGAGEMENT-VERIFIED: 0.98/0.98. c64@256k composite: duration -7.9%,
  output tok/s 47.8 -> 51.9, TTFT 673 -> 622 s, TPOT 219 -> 200 ms,
  peak decode-tail throughput 150 -> 237 tok/s.

MEASUREMENT FINDING (rule 40): concurrent token-identity is NOT an
available gate on this stack — the CONTROL diverges from itself (4/8
requests) under per-request submission, under both determinism pins
(DETERMINISTIC_MOE_ALIGN=1, FIXED_DECODE_SPLITS=8), and under atomic
batched submission: arrival->co-batching varies run to run and flips
ulp-order at T=0. Concurrent correctness gates are construction argument +
CPU guards + statistical quality (gsm8k nc=8).

KNOWN ISSUE (open): engine deadlock at ctx~245k when two prefills are
admitted ~0.3 s apart during active decode ("No available shared memory
broadcast block found", rank-0-only collective signature). Not hit by c64
or serial admission. Logs: serve_F_HANG.log, ramp_HANG.log.

Rules 41-46 (256k campaign, rounds 2-3):
41. A quiet measurement window needs a GATE, not a wait: verify prompt-token
    delta 0 / preemptions 0 / Running stable across the window, or a leaked
    prefill tail ships as a finding.
42. Check the assert that bounds a knob before designing an arm around it
    (num_speculative_tokens >= dspark_block_size killed the K-reduction arm).
43. A profiling harness is a CONFIGURATION: diff its env against the served
    config and prove parity from a counter the flag moves (rounds 1-2 kernel
    tables were taken without the dequant flag; the KV-size line exposed it).
44. Never fit a line across a threshold the feature itself has (the decode
    shard's "+5 ms intercept" was a C=2 anchor below MIN_REQS fitted against
    sharded points).
45. Slopes come from differencing two operating points, not dividing one
    point by C.
46. To ask whether reads bind a kernel, vary the WORKING SET, not the
    kernel (sparse-attn: 62.9 MB vs 2.0 MB L2-resident pools moved
    throughput 5% — byte reduction buys nothing; the algorithm is the cost).

## 11. Round 3: the kernel round (post-MFU-accounting)

Honest ceiling first (prof R3): 46.32 GFLOP/token model-wide; 26.1% MFU at
the round's start; per-pool ideal floor ~19.9k tok/s ingest. Rounds 1-2
kernel figures were re-anchored after a config parity failure (rule 43).

ADOPTED (all default-on, =0 opt-outs):
- **K2-prefill, query-blocked dense-causal kernel for the 20 ratio-128
  layers** (VLLM_SPARSE_DENSE_QUERY_BLOCK, BLOCK_M=8, warps measured
  4/4/8/8): the layers have NO indexer and positional identity-prefix
  index lists, so the kernel derives rows in closed form and reads no
  list — killing the per-layer combine+pack as a second mechanism.
  Kernel 2.81x (36.2 -> 101.5 TF/s = 32.5% of peak); cold TTFT@245k
  -8.61%, three-armed within-tree A/B attributing 28% to index-removal /
  72% to blocking (matching the kernel-level 1.16x/2.81x split); gsm8k
  neutral; real-tensor numerics cos >= 0.9999995 at 32k AND 200k with
  max_abs = one bf16 ulp at both depths (13x rows, unchanged error =
  bounded by output rounding, not reassociation).
- **K1** q LUT-decode hoisted out of the decode paged indexer kernel:
  1.39-1.44x kernel, bit-identical (torch.equal), F slope -4.6%.
- **K4** KV-group gate re-keyed to grouped-CTA wave count (one wave = 324
  at 162 regs): -6.8% at the served shape; all four measured corners
  right where any M threshold gets three.
- **K7 + maxnreg lift**: k_scale factored out of the relu (legal by
  positive homogeneity; ACTIVE SET exact, selection can cross near-ties
  via signed-head-sum cancellation -> gsm8k-gated at 0.97, the int8-AR
  band) + the expired maxnreg=128 cap removed (set at 132 regs, kernel
  now 162; the cap spilled — rule 47).
- **K6** token-shard extended to the 22 ratio-128/SWA layers' fused_wqa_wkv:
  pre-registered A/B/A/B 6-load gate at 8K, executable verdict():
  ON 447.33 vs OFF 451.21 ms = -3.88 ms, ADOPT (predicted 4.47).

REFUTED: **K2-decode** by its own pre-registered stop rule — 1.2-1.9x
SLOWER. Mechanism: prefill is CTA-oversubscribed (15,360 CTAs/108 SMs) so
trading CTAs for shared row-loads is free; decode is CTA-starved (27) so
the same trade gives up exactly what it lacks. Also the SIXTH occupancy
confirmation (16 warps = exactly 2x occupancy at identical tile/smem/
split: 0.74x). Kernel + flag retained as executable record, default off.

FINAL c64@256k cold natural greedy (vs round-2 close):
duration 1105.5s (-10.4%), output 57.9 tok/s (+11.6%), ingest 15.11k
tok/s, TTFT med 557.3s (-10.3%), TPOT mean 177.3ms (-11.1%), peak decode
242 tok/s. Cumulative vs the original artifact baseline: output +40%,
TTFT -23%, TPOT -64%.

Round-3 measurement rules (47-52, drafted across reports):
47. A tuning's justification is a dated fact: re-measure the premise
    (register count, shape) before trusting the tuned value.
48. Pre-register bands and stop rules before first launch; a result
    outside a band is an attribution change to report, not a re-band.
49. Engagement logs carry REACHED COUNTS, not just on/off.
50. A too-good number is a reason to re-read, not to report (the perfect
    max_abs=0.0 that was 8 of 80 lines).
51. Preflights run per-operation, not per-session; and test the guard
    itself (the hand-rolled pgrep-matches-searcher reappearance).
52. Ops: setsid for servers (tool-call process-group reaping); killed
    servers leave VLLM:: workers holding memory and the next launch dies
    with a bare init failure; nvidia-smi compute-apps shows namespaced
    PIDs — ps is the real view; poll /health, not the launch PID; a
    window-open order must enumerate which prior duties it voids.

Round-4 findings (prof R4): post-round-3 cold TTFT c1 16.53s (-10.4%);
MFU 28.0% (useful 26.5%); genuinely open prefill remainder ~0.7s = 4% of
TTFT, owned by the ratio-4 ragged pool at 24.1% of its tile ceiling with
NO named mechanism (K2's lever needs positional index lists). Indexer
prefill kernel CLOSED at 97.6% of its own measured ceiling: the R3
"pipe-overlap failure" claim is REFUTED — epilogue-stripped the kernel
runs 162 TF/s = 51.9% of peak (l1tex 74-78% is the binder, operand
delivery starves the MMA; overlap is ~50% effective, exposed fp32 6.6%);
BN=256 and warps=8 both worse (7th occupancy confirmation). mHC "pool
halving" RETRACTED: _PRENORM_SMALL_T=32 route switch at C=5.33 migrates
the work to cuBLAS (kernel SET changed, not cost). Decode slope unchanged
by round 3 (2.12 ms/req; K1 is 68% of the small delta); owner is ONE
kernel, _sparse_attn_decode_partial_kernel, 1.083 ms/req = 53.7% of the
slope — mechanism search open. mnbt=32768 arm: NULL throughput/TTFT,
mean TPOT/ITL gains were residency redistribution (rule 39) — not
adopted, documented as a knob. Deadlock scope WIDENED: also hit at 200k
ctx / C=4 / serial admission after a profiled prefill in the same server
session — do not mix prefill profiling and decode ramps in one session.

53. Diff kernel SETS before pricing a component slope: a kernel whose
    call count changes across operating points has changed ROUTE, not
    cost (the mHC "anomaly").
54. Check which directions a tuning comment actually swept before
    trusting its optimum (BLOCK_N=128 had only ever been swept down).
55. Measure a kernel's ceiling with the rest of the kernel switched off
    before attributing a gap to interaction (the epilogue-stripped run
    refuted "overlap failure").

SPEC-DECODE + STRUCTURED OUTPUT (2026-08-07/08, c16 short-ctx schema
workload): scheduler-side jump-forward splice into truncated draft
windows REFUTED as implemented (tok/step index 78.5->32.0, ITL-arm
collapse 2.45x). Mechanism (corrected 08-08 after code audit, three
sites verified): the worker reads scheduled_spec_decode_tokens for
LENGTH ONLY (gpu/model_runner.py ~968); verified token content comes
from worker-resident req_states.draft_tokens. A scheduler-side splice
therefore desyncs the grammar bitmask rows (FSM advanced through
spliced tokens) from the tokens actually verified (drafter originals)
— the masked target is constrained by masks for a token path it is not
on. The original "drafter poisoning" mechanism is WRONG: the DSpark
drafter conditions only on accepted context (num_rejected accounting
input_batch.py ~455, valid_ctx_end speculator.py ~523, last_sampled
anchor ~508). Any draft-window repair MUST write through
req_states.draft_tokens (worker side), never scheduled lists alone.
Related measured facts: popcount==1 grammar states are 0.91% of
positions (forced-chain splice = +0.000 accept_len, dead); structural
oracle ceiling 1.5x accept_len; xgrammar 0.2.3 disable_any_whitespace
emits ": "/", " separators (NOT compact; '{"a":1}' is rejected —
"compact JSON" test streams must use the canonical spaced form);
validate_tokens costs 0.088 ms/step and cannot be skipped
(grammar_bitmask asserts on invalid scheduled tokens); grammar_bitmask
serial loop 0.40-0.57 ms/step at c16 (thread-pool path gated on
max_num_spec_tokens==0).

56. Name the denominator BEFORE publishing a derived metric, and run a
    ceiling preflight on every counter-derived number. RESOLVED
    (08-08 audit): the ceiling violations were NUMERATOR scrape bugs —
    unanchored substring matching absorbed sibling series
    (request_generation_tokens buckets into gen_tokens;
    accepted_tokens_per_pos doubled acc exactly 2x) and
    "time_per_output_token_seconds" silently matched the per-FINISHED-
    REQUEST metric (per-request mean, ramp-dominated for short
    requests) instead of the per-step vllm:inter_token_latency_seconds.
    Corollaries: match /metrics by EXACT metric name (strip labels);
    acceptance LENGTH = 1 + accepted/drafts (accepted excludes bonus,
    asserted <=5); spec_decode_num_draft_tokens has DIFFERENT semantics
    under structured output (invalid tokens subtracted) — never compare
    draft-acceptance RATE across schema arms, only acceptance length;
    equalize output length across arms before comparing per-request
    means. Corrected c16 board: accept_len 4.17 none / 3.32 schema,
    steady-state JSON ITL penalty ~1.26x (the 1.81x read was
    length-composition artifact).
57. A mechanism claim about a cross-process dataflow needs the READER
    side verified, not just the writer: the splice wrote valid tokens
    into a list the consumer never reads for content.
58. torch profiler on this build: FATAL AT START, full stop
    (amended 08-08 after the reduced-risk c4 attempt: engine died
    12 s after POST /start_profile with CUDA ULF + Xid 31 MMU
    fault, drain-then-stop discipline never reached, ZERO trace
    files). The earlier reading (crash at stop-under-load; careful
    capture possible) UNDERSTATED the hazard — the tracer running
    at all kills the engine. Worker-rank torch traces are
    UNOBTAINABLE on this build; the frontend-only trace is all
    that exists or will exist. Use nsys exclusively.

59. Post-forward interventions cannot raise accept_len (spec-json's
    theorem, 08-08): logits at window slot j depend only on drafts
    0..j-1; greedy rejection emits (longest matching prefix)+1, and
    the +1 is already the target's masked argmax (rejection_sampler
    stores target_argmax at first mismatch). After execute_model
    dispatches, you can only reject, never extend. Under async
    scheduling the validate/bitmask block runs AFTER dispatch
    (core.py ~650 vs ~720-736), so ALL scheduler-side draft repair of
    the in-flight window is void — the scheduler's token list affects
    the worker only via num_logits and the bitmask rows (verifier
    reads drafts from input_ids, rejection_sampler.py ~242). Splice
    lever (b) retracted at 0.00 ms on this theorem BEFORE any code was
    written — price the delta at the pipeline SITE where it must act,
    not the quantity the idea is about (rule 21 restated). SCOPE
    CAVEAT: the theorem binds interventions on a window whose forward
    has dispatched; whether a NEXT-window override (riding the next
    SchedulerOutput into the GPU draft buffer pre-_prepare_inputs,
    with a then-known FSM state) has a live pre-forward window is a
    separate, open question.

59-RESOLVED (08-08): the rule-59 scope caveat is CLOSED, no carve-out.
The "next-window" premise was off by one batch: drafting runs INSIDE
sample_tokens (speculator.propose gpu/model_runner.py:1566,
set_draft_tokens :1585, within sample_tokens :1451), so the drafts
validated in step k's deferred block are step k's OWN window, whose
forward dispatched at core.py:653 with originals already in input_ids
(input_batch.py:372-383). A pre-forward repair window would require
serializing host processing against dispatch — deleting the
batch-queue overlap dspark is exempted to keep (config/vllm.py:1107).
Also verified: FSM state at validate time is the true
post-(k-1)-acceptance state (update_from_output at core.py:709 runs
first) — the existing path has no staleness. Structural-DFA draft
masking priced (spec-json memo): runtime story SOUND (43 allowed-sets,
1.11 KiB CSR/schema, 253 KiB shared token-class table, <1 MiB total,
GPU-resident, ADVISORY — target still masked by the real FSM, so
desync costs acceptance never correctness; does NOT give back the
local-argmax win) but NO-GO to build: benefit bracket straddles the
1 ms bar on unknown q (drafter hit-rate under mask), and no sound
construction method demonstrated (allowed-set equality is not state
equality; xgrammar 0.2.3 exposes no state enumeration).

59-AMENDED (08-08, second correction): the 59-RESOLVED closure was
OVERBROAD. scheduler.py:2175 and :2204 are different call sites in
different scheduling MODES. In SYNC scheduling the pre-forward window
exists and is sound: update_from_output(N) advances the FSM to the
true post-N state (core.py:609) BEFORE post_step validates N+1's
drafts (:620 -> :2175), and schedule() builds N+1's window after
that — multi-slot repair with consistent bitmask rows is possible,
worth up to -1.30 ms/token (still needs override plumbing into
req_states.draft_tokens; scheduled lists carry length only). But
:2175 is DEAD CODE in production: post_step is gated on
`not async_scheduling` and dspark is exempted from the async-disable
list (config/vllm.py:1107-1112). The repair and async scheduling are
mutually exclusive BY CONSTRUCTION — async's speed comes from
dispatching before drafts are known, the same fact that denies the
repair a landing spot. Trade bound: repair <= -1.30 ms/token, so it
pays only if async-off costs less than that per token — UNMEASURED
(CAPACITY_256K.md prices async-off only in resident requests). The
one-toggle async-off A/B resolves the entire branch and precedes any
build decision. Rule: when two call sites share a function, trace the
MODE that production runs before generalizing from either.

STRUCTURAL-DFA DRAFT MASKING: CLOSED NEGATIVE (08-08, exhaustive
construction demo, spec_json_dfa_build.py). The allowed-set partition
is NOT a congruence: 7-8 aliased classes per schema (of 52-62
reachable allowed-sets; 1-step-refined state lower bound 63-78), with
witnesses showing up to 5 distinct transition functions behind one
allowed-set. The aliasing lands ON the maskable classes (4-6 of 7-8
per schema at the |A|<=8 gate, including |A|=1 classes — two states
permitting exactly one token, then diverging). Failure mode is worse
than no-benefit: an aliased DFA drifts silently, later masks exclude
the target's argmax, acceptance drops BELOW unmasked baseline, and
there is no self-resync (replay re-crosses the aliased transition);
drift detection requires the scheduler FSM = the host round-trip the
design existed to avoid. Advisory property protects correctness only,
not throughput. A sound DFA needs xgrammar matcher-state access
(0.2.3 exposes none) or reimplementing its compiler — upstream ask,
out of campaign scope. Runtime pricing kept as head start if state
access ever lands. ACCEPTANCE-LEVER FAMILY CLOSED: (a) zero measured,
(b) zero by theorem (rule 59), DFA-by-observation unsound,
DFA-by-grammar-internals out of scope. Surviving: (d) compact
separators -0.26..-0.37, (c') numpy fill ~0.13 host, async-off A/B
(one toggle) as the last open empirical branch.

DFA addenda (08-08): (1) conservative variant (mask only non-aliased
classes) is DOUBLE-LOCKED: soundness aside, it masks a strict subset
of positions whose full-masking bracket (-0.66..-1.85) already sat
midpoint-on-bar — a subset can only price lower. No self-resync
exists either way (replay re-crosses the aliased transition). (2) The
async-off decision threshold is MIX-WEIGHTED, not 1.30:
frac_schema x gain_schema > cost_all_arms (gain_schema <= 1.30);
1.30 applies only at 100% schema traffic. At 50/50 the bar is ~0.65
ms/token, at 20% schema ~0.26. Pre-register the functional rule and
obtain the deployment schema fraction before calling the verdict —
a "pass" at 1.20 could still be a net loss on the real mix. (3) A
negative async-off A/B ends the acceptance-lever family outright —
there is no DFA fallback (closed on soundness, not cost).

60. vLLM request-level histogram BUCKETS start at 0-0.3 s: interpolated
    percentiles for TTFT/queue/prefill are garbage at ms scale (a
    quantity with exact mean 0.0 interpolates to "p50 150 ms"). Use
    _sum/_count means (exact) or client-side percentiles. Also:
    prefill_time at concurrency is ~98% step-WAIT, not compute
    (47.7 us/token on a ~50 ms floor at c1); and async scheduling
    moves GPU wait between queue_time and prefill_time buckets — in
    any async on/off comparison, SUM the two before reading movement.
    Cold-init tax: first structured request pays ~390 ms synchronous
    (TokenizerInfo 313 + bitmask alloc 76 + first compile 17.6) on the
    engine input thread — prewarm with one throwaway schema request;
    VLLM_XGRAMMAR_CACHE_MB is NOT a lever (no eviction at 512 MB
    at realistic schema counts; tax is quantized 0-5 ms mean,
    ~26 ms = one missed schedule boundary worst case). Negative:
    stream-interval 1->8 nulls on TTFT (residual is not SSE
    contention); prefill/decode separation prices at +0.65 ms step
    during injection, ITL p99 +0.02 — not worth building.

61. Tree-fingerprint every capture/gate window (prof-c16's rule,
    08-08): record HEAD sha + sha256(git status --porcelain) +
    sha256(git diff) immediately before AND after each measurement
    window; if they differ, DISCARD the measurement — a profile or
    A/B of an unattributable tree state is worse than none. Born of a
    live collision: an editable-install tree IS production, so one
    agent's in-tree development rode into another agent's server
    launch mid-edit (AttributeError at import was the GOOD outcome; a
    clean-importing half-edit would have shipped silently into a
    .nsys-rep). Corollary: development happens in worktrees; the live
    tree is measurement-only. Corollary 2 (this session's recurring
    lesson, now thrice): serialize the box through explicit ownership
    sentinels and assume messages CROSS — a "reclaim" announced is
    not a reclaim acknowledged; the actor who currently holds a
    loading server wins ties.

61-COROLLARY (08-08): the window fingerprint must include INSTRUMENT
state, not just tree state — a system-wide nsys GPU-metrics session
left running (all 8 devices @10 kHz, ~05:16-05:26) silently taxed a
"clean" stock baseline captured inside it. Preflight every
measurement window with `nsys sessions list` (must be empty) and
record it alongside the tree fingerprint. The contaminated baseline
was discarded on mtime evidence; per-token numbers from it (7.5/8.4/
8.2/8.8) are INDICATIVE ONLY pending the redo.

62. R2a pre-registration (ttft-c16, from code reading core.py
    :653-694, config/vllm.py:544-549): with batch_queue_size=2, async
    scheduling structurally costs an arriving request ~1 extra step of
    TTFT (~26 ms at c16) versus sync, while buying GPU utilization —
    so the async-off arm should show TTFT DOWN ~26 ms and throughput
    DOWN. If TTFT does not move under async-off, the loop model is
    wrong and the §7 residual decomposition needs re-deriving. R2
    (re-drain input queue) is CLOSED at the site: the drain already
    runs last-before-schedule(); the ~13 ms is the blocking wait on
    the in-flight batch, unreachable by drain placement.
    SCORED (win2, 08-08): R2a FALSIFIED — async-off left the c16
    admission at exactly 3 steps (identical distribution), TTFT
    unmoved, and the counter-expectation (>1.30 ms/token cost)
    ALSO missed (measured ~0 at c16 short-ctx). Whatever quantizes
    admission to 3 steps is NOT the async batch queue — open
    question, loop model under re-derivation. Async-off adoption
    is NOT live (256k long-ctx async value unmeasured, rule 22);
    the sync-path repair trade re-opens ONLY if the long-ctx cost
    is also ~0, which nobody has measured.

63. Construction-site reads before funding sweeps (08-08, both board
    items revised): (a) sparse-decode-attn denominator was the WRONG
    POPULATION — compress_ratios has 46 entries (21 fours, 20
    one-twenty-eights, 5 zeros; rule-8 reproduction) and BOTH c4a and
    c128a run the same decode kernel: 41 layers, ONE call each (39.3
    target + 2.75 draft = 42.08/step). No pair to fuse. Zero-code
    surface EXHAUSTED at this shape: heads_blocks already 1 (BLOCK_H
    is per-CTA efficiency, cannot change CTA count), num_splits=1 is
    the docstring's own measured optimum (96 CTAs = 0.89 waves;
    blocked path "Default off: it loses"). Remaining lever is
    rewrite-shaped behind a maxnreg diagnostic sweep, and the pool is
    BIMODAL (c4a vs c128a = ~32x index density, cannot share a
    tile-scheduler plan) — sweep must report the two populations
    separately or it fits across a feature threshold. (b) decode-GEMM
    splitKreduce: gridY=6 is modal not structural (range 4-15); GEMMs
    are batched (2.8/layer), the per-sequence-M=6 worry is refuted.
    Algo frozen at capture → influence = process env only. Verified
    in torch 2.11.0+cu129: CUBLASLT_WORKSPACE_SIZE,
    CUBLAS_WORKSPACE_CONFIG, DISABLE_ADDMM_CUDA_LT present;
    TORCH_BLAS_PREFER_CUBLASLT ABSENT — don't plan around it. Priced
    test w/ kill condition: CUBLASLT_WORKSPACE_SIZE 1024->32768 KiB,
    re-capture, count splitKreduce/step — drop = lever confirmed
    (risk 1); unchanged = shape-driven, env route dead, Marlin
    re-check at M=96 is the only successor (rule 22). Two hypotheses
    died AT the site instead of in windows (num_splits pre-refuted by
    the docstring's C-sweep; gridY uniformity refuted by its tail) —
    the gating pattern pays.

64. A negative result requires POSITIVE evidence the treatment took
    effect (win5 near-miss, 08-08): CUBLASLT_WORKSPACE_SIZE=32768
    alone is silently CLAMPED to 8.125 MiB on this build (unified
    workspace mode caps the LT pool at the cuBLAS pool;
    CublasHandlePool.cpp, TORCH_WARN_ONCE only) — the arm would have
    looked clean, counts unchanged, and KILL would have closed a live
    route on a treatment that never applied. Defense pattern: (a)
    apply the full treatment (CUBLAS_WORKSPACE_CONFIG=:32768:2:16:8
    alongside the LT size), (b) grep the server log for the exact
    clamp warning post-warmup AND post-burst, (c) the verdict is VOID
    unless the runner prints CLAMP-CHECK: OK. Composite-treatment
    ruling: one-variable discipline applies at the HYPOTHESIS level
    ("adequate workspace end-to-end"), not the env-var count — two
    vars jointly implementing one treatment is one arm. Related
    craft: never reuse a capture driver whose instrument config has
    drifted (capture_window.py now hardcodes GPU metrics — using it
    would tax the treatment against an untaxed baseline); derive
    session names from the launch script, never guess; error exit
    codes must be disjoint from verdict exit codes.

65. Post-review corrections from re-measurement (prof-c16 §12,
    08-08) — two review-endorsed numbers overturned by better method:
    (a) the c16 in-graph all-reduce is 45-66% REAL PAYLOAD (aligned
    per-call minimum across all 8 workers ~32 µs → payload floor
    2.19-2.23 ms/step, stable 2% across windows) — the "97.5% barrier
    wait" read used a bad floor (p1=5.7 µs); board #6 still ~0
    recoverable, description corrected. (b) The 7.1% cross-worker
    kernel-time spread is NOT a desynchronization lead: remove the
    collectives and it collapses to 0.3% — it is barrier absorption,
    canon's already-refuted skew finding reproduced independently.
    (c) gridY on the decode GEMMs: BOTH prior readings wrong (not
    per-sequence M=6, not split-K factor) — it is ceil(rows/16)
    M-tiles and phase-tracks (5 inside draft graphs); the GEMMs are
    batched to M~96. Board #2 estimate rebuilt on the full grid
    distribution: 250-500 µs/step (was 400-700). Also: the 2.004
    calls/layer question dissolved via the same 41-layer population
    spec-json found — two independent readers converged. Capture
    attestation implementation: tree_fingerprint.sh + prof-c16.md
    §1.6.

66. State the measurement design and its noise floor IN THE SAME
    BREATH as a pre-registered band, and check the band clears the
    floor (spec-json post-mortem, 08-08). On this stack a
    restart-based serving A/B has a floor of ~0.33 ms/token
    (measured, patch-inert cell across restarts) — bands under the
    floor are numbers that cannot lose, the mirror image of gates
    that cannot fail (the impossible byte-identity gate shipped into
    the same window). Related corrections: "flag-not-fork" was false
    (compact_separators latches at XgrammarBackend.__post_init__;
    vllm.envs caches after service init — per-server, not
    per-request); grammar equivalence ≠ trajectory equivalence (a
    grammar-level json.loads-equality test verifies the GRAMMAR, not
    the SYSTEM — label untested claims untested). LIVE COROLLARY:
    disable_any_whitespace=true (adopted, in production) is the SAME
    quality-surface class — it changed the token path at every JSON
    separator and its quality direction is UNMEASURED (standing
    gsm8k probes never exercise the structured-output path — the
    quality gate must exercise the changed path). Not claimed
    harmful (it killed a real runaway pathology); claimed unmeasured.
    Any eval arm built for the parked compact-separator patch must
    include a disable_any_whitespace on/off arm at the same cost.

67. TTFT CONSOLIDATION (ttft-c16 §10 re-derivation after R2a's
    falsification): prefill_time is a fixed MULTIPLE of the step wall
    (2.3-3.3x across c1->c16, across batch-queue depth 1 vs 2), not a
    fixed step count — L2 is RETIRED into L3. At this operating point
    TTFT and ITL share ONE lever: the decode step wall (~3 ms TTFT
    per 1 ms step wall). No pipeline-depth lever exists separate from
    step wall. Ratio law rests on two points; c4/c8 confirmation with
    in-run step wall queued (1 min). Probe epistemology: at
    concurrency, steps_elapsed counts BACKGROUND completions, so it
    tracks TTFT/step_wall by construction and cannot localize time
    within prefill_time — it is informative only through invariance
    tests (as it was for R2a); the c1 row is the load-bearing one.
    Per-hop stamps NOT funded (knowing why 3x only pays if the 3 has
    a knob; no evidence it does). Miss taxonomy worth keeping:
    mechanism error (predicted benefit from removing a non-cost) vs
    magnitude error (predicted cost that measured ~0) — the former is
    the deeper miss.

68. Two benchmarking traps (ttft-c16 §11, 08-08): (a) benchmarking
    any vLLM component with a bare AutoTokenizer inflates everything
    touching get_vocab() by ~50 ms/call — production uses
    cached_tokenizer_from_config which precomputes the vocab
    (get_vocab 62.6 ms raw vs 0.0 cached; the "54 ms per-request
    reasoning parser" find was this artifact — 0.23 ms in
    production). Cross-check any per-request cost claim against the
    serving cells it would have to show up in BEFORE running a
    micro-bench. (b) Same-process A/B of allocation-heavy setup code
    is contaminated by the first arm (cached arm read 2x its
    isolated figure; in-process result disagreed with isolated
    processes about the SIGN) — one process per arm, always.
    Rule-29 sharpening: a component with 230-400 ms run-to-run
    spread quoted as a point estimate (313) makes any derived band
    meaningless — quote distributions as ranges. The prewarm 40 ms
    remainder stays LABELED (first-structured-request only, eager
    init long complete; leading suspect = worker-side first bitmask
    ship/apply ~1.48 MiB + possible new captured shape); the
    discriminator (also record first NON-schema request in the cold
    gate) rides the next natural gate run — 40 ms once per server
    start does not fund a window.

WINDOW QUEUE FINAL VERDICTS (08-08): (win4) R1 GIL switch-interval
REJECTED by its own gates — residual 5.3 -> 29.1 ms (5.5x WORSE),
client TTFT +25 ms, step-wall co-gate blown (+2.6 vs ±0.9): the
unmeasured context-switch cost side dominates on 192 cores; the 5 ms
convoy microbench did not model the engine. (win5) CUBLASLT workspace
KILL on matched-cell data: stock 270.05 vs treatment 265.05
splitKreduce/step = 1.85% (anchors matched 1.7%); split-K is
SHAPE-DRIVEN, env route closed; the first AMBIGUOUS verdicts were
archival-baseline cell mismatch (292.81 came from the max+schema
window) — the runner's refusal to over-read 9.5% was correct, and the
ratio-normalized prediction (1.8%) was confirmed by the matched
rerun. Successor (Marlin routing at M=96, risk 3+) priced separately,
not auto-funded. Also scored: canon-61 fingerprint discipline fired
against its own operator — win5 attempt 1 DISCARDED because the team
lead edited the canon doc mid-window; the edit freeze is behavioral,
not just tooling. Queue totals: 5 windows + 1 rerun, 2 adopted
(disable_any_whitespace w/ quality caveat, prewarm -218 ms cold
first request), 2 rejected by gates (R1, CUBLASLT), 1 parked
(compact separators, quality-surface), 2 mechanisms killed (async-off
R2a both directions, splice family), ratio law confirmed to 4 points
pending final c1/c4/c8 within-run check.

67-SCORED (08-08 final check, four within-run points): the RATIO LAW
AS REGISTERED IS REFUTED — prefill_in_step_walls = 3.2 (c1, self
wall 12.05 ms), 4.44 (c4), 3.75 (c8), 2.99-3.04 (c16): the 2.3-3.3
band breaks at c4/c8 and the variation is non-monotonic (hump).
QUALITATIVE consolidation survives (TTFT tracks the step wall, no
separate pipeline lever found anywhere); the QUANTITATIVE claim
softens to "TTFT = 3-4.5x step wall, multiplier varies with
concurrency, mechanism of the variation unknown". §10 owes the §7
treatment per its own pre-registration. c1 self-measured step wall
12.05 ms is consistent with the campaign's v12 12.14.

69. Enumerate comparability axes as a CHECKLIST, never a sentence
    about one axis (spec-json's win5 post-mortem): prominently
    documenting ONE axis (config parity) created false confidence
    that comparability was handled in general, while a second axis
    (cell identity) was hardcoded inconsistently between two files
    by the same author in the same hour. The anchors caught it; the
    rule exists so they don't have to.

67-SETTLED (08-08, five-point single-instrument run, c2-c16): the
busy-regime relationship is AFFINE: prefill = 52.1 + 1.37 x
step_wall (residuals <= 0.9 ms across a 2.5x step-wall range; ratio
falls monotonically 5.54 -> 3.03 as the form requires; the earlier
"hump" was instrument-mixing, the "3-4.5x" band was an average over
this decreasing curve). THE NUMBER FOR THE KERNEL BOARD: a kernel
win cashes ~1.37x into TTFT (marginal slope), NOT 3x — e.g. the
sparse-decode 4.8 ms/step pool is ~6.6 ms of TTFT, not 14.4. NEW
NAMED OBJECT: the 52 ms busy-regime prefill intercept —
concurrency-independent, unattributed (c1 idle regime sits off this
line at ~37 ms; the intercept is a busy-engine property). It is now
the single largest unattributed TTFT term and the natural successor
investigation to the retired L2/L3 framing.

70. REVIEWER-ROUND RESOLUTIONS (08-08 PM, tail-hunt 4-capture
    replication + W-B/census): (a) NEW TOP BOARD ITEM — the eager
    launch path of mixed prefill/decode steps: the 3.42 ms reduce
    tail is 99.6-99.9% inside the 2.8% of steps carrying a prefill
    chunk; fixed launch shape (36x512, 442k calls) = all excess is
    wait; the straggler rank is IDLE (5-15% busy, host inside a
    single 3+ ms cudaGraphLaunch) — per-rank host launch skew, not
    load, not the collective (canon 65 mean intact). A full
    chunked-prefill eager forward = ~170 ms wall for 40-45 ms
    median-rank non-collective work (3.5-4x stretch). Cost ceiling
    3.75/2.14% of wall (max arm) to 11-12/6-7% (none arm) wait/dead
    — UPPER BOUND: CUPTI taxes each of ~3700 eager launches per
    forward per rank (the exact blamed path); first funded step is a
    launch-tracing-free measurement (CUDA events / engine-counter
    A/B), NOT an implementation. Subsumes board #3's eager-idle;
    plausibly feeds the 52 ms TTFT intercept. (b) The reviewer's
    621 ms gap = nsys ATTACH (head of every capture, all 8 devices,
    zero mid-capture >20 ms gaps in 4 exports; same event family as
    the tail's largest instances, different costume per capture) —
    CLOSED; RULE: drop the first ~0.7 s of any nsys capture from
    sqlite-derived denominators. (c) 1-stage force at 768 KB decode
    ARs: NULL with canon-64-complete treatment evidence (census:
    422,338 1stage / 0 2stage kernels; serving deltas +0.04..+0.24
    ms/token = noise floor) — the 512 KB compile-time crossover is
    VINDICATED in-graph at the real operating point; the standalone
    table transfers. (d) H2D census: 9.41 small copies/step/rank =
    29.1 us/step/rank total device time; dominant contributor is an
    UNIDENTIFIED 384 B x4.18/step (not in the predicted site list);
    grammar bitmask (76 us/step/rank) is 2.6x the entire small
    population. WP4 pricing gated on identifying the 384 B source.

71. BIMODAL DECODE STEP (W-A attribution + code confirmation): 32.4%
    of pure-decode steps ran the piecewise/breakable path at 34.3 ms
    vs 2.4 ms fullgraph — 14x at IDENTICAL work (matched pair: 78
    rows = 34.9 ms/1802 kernels, 84 rows = 2.7 ms/226 kernels; batch
    size is NOT the driver). PREDICATE: the capture loop round_up()s
    configured sizes to multiples of decode_query_len (6), leaving
    HOLES at 30/54/78/102.. tokens (= 5,9,13,17.. requests — nothing
    rounds into them), and dispatch is exact-key with NO pad-up
    fallback (cudagraph_utils.py:375 key=(num_tokens,loras)) — those
    request counts pay 14x on every ramp/drain/fluctuation. Prize
    ceiling 39.9% of drain-window wall (existence confound-robust:
    GPU-busy 35.5 vs 3.1 ms; kernel count is a dispatch property).
    Fixes: config-only hole-filling capture sizes (A/B live); durable
    = dispatch pad-up fallback. Related W-A facts: engine is
    worker-bound (54% block_on_oldest, ~2% own work — no scheduler
    intercept exists); straggler = HOST PYTHON COST (58.6%
    interpreter top-of-stack, on-CPU density 0.92 — not descheduled,
    not driver-blocked); instrument tax measured directly: 13.8% of
    launch-thread samples inside CUPTI/injection; sync tails benign
    (dedicated waiter thread; main thread 0.46 ms/step); NVTX
    attribution queries MUST join globalTid (ranges are thread-local
    — globalPid joins produce impossible numbers); H2D small-copy
    census is CONFIG-DEPENDENT (384 B population absent here; this
    config: attn_metadata 5/step + preprocess 3/step); layerwise
    NVTX is structurally unavailable (hooks live in the compile
    pipeline that VLLM_USE_BREAKABLE_CUDAGRAPH disables); the 52 ms
    intercept needs an ADMISSION-TRIGGERED capture window.

72. W-C ncu verdicts: board #1 CLOSED — sparse-decode kernel at its
    shape's floor (85% of occupancy-permitted issue rate; top stall
    = flash chain's own serial dependency 32.9/34.9% c4a/c128a; both
    ncu-recommended levers measured NEGATIVE: maxnreg=128 +11.6%
    slower [occupancy refutation #8], num_splits=2 +1.6..+20%
    slower; the "31.87% coalescing" counter is absorbed by L1 at
    95.2% hit / 1.34% DRAM). Do not fund a rewrite. FOUND INSTEAD:
    block_k=32 at rocm_aiter_mla_sparse.py:3099 is "Tuned on gfx950"
    (an AMD part) — BLOCK_K=64 measures +6.4-7.9% on the c4a
    population and -8% on c128a short-ctx → ~170 us/step if applied
    PER-POPULATION; NOT numerics-preserving (1e-3 rel) so it needs a
    quality gate, not a flip.

73. HOLE-FILL ADOPTED (win7 + c5 kill-shot A/B, 08-08): adding
    capture sizes {30,54,78,102,126} (multiples of decode_query_len
    that nothing round_up()s into) to cudagraph_capture_sizes —
    config-only. At the pure-hole operating point (steady c5 = 30
    tokens): step wall 25.6 -> 18.4 ms (-28%), per-token ITL 8.68 ->
    6.11 ms (-30%), throughput +22%, acceptance unchanged; treatment
    evidence = config-line census (30/54/78/102/126 present) + this
    A/B. Drain-heavy c16 burst: +3-8% agg (paired +11.7%, population
    +3-8% honest). At steady c16: within noise (no harm). MAGNITUDE
    DEFLATION recorded: the untraced hole cost is ~7 ms/step, not
    the 32 ms the CUPTI-taxed W-A capture showed — the tracer's
    measured 13.8% launch-path tax compounds over the slow path's
    372 launches/forward; capture-derived magnitudes on
    launch-heavy paths overstate by ~4x here. Existence claims from
    traces, magnitudes from untraced A/Bs. Durable fix (upstream
    item, not yet built): dispatch pad-up fallback in
    cudagraph_utils.dispatch — round num_tokens up to the nearest
    captured uniform descriptor instead of exact-key falling to
    piecewise; would retire the hole CLASS for every
    decode_query_len and capture list, not just this config's.

74. Correlation-key postmortem (winB→v3): the zero cross-process
    overlap was NOT a seeded-hash bug (blake2b was already
    deterministic) — assign_request_id() REPLACES request_id with an
    internal id after input processing, so the two processes hashed
    DIFFERENT STRINGS correctly. Two proposed fixes (md5 swap) would
    have changed every value and preserved the failure. Lessons:
    (a) "test the property the feature needs, not the unit you just
    wrote" — the v2 test validated the helper in isolation and was
    true-but-irrelevant; the v3 test reproduces the defect from
    first principles + AST-guards every call site (which caught a
    missing site immediately); (b) a STALE label is worse than a
    missing one — the eager-forced path skipped dispatch() and
    inherited the previous step's fallback reason; v3 invariant
    (verified 576/576): reason is None iff a FULL graph ran; (c)
    key_miss n=682/1536/1792 are mixed chunked-prefill steps above
    max_cudagraph_capture_size=512 — correctly NONE, now labeled.

75. winC verdicts — the TTFT budget NAMED and the output chain CLOSED
    (tail-hunt, 08-09): (a) Output-copy chain = PIPELINE SLACK on four
    grounds: waits are 99.1% GPU-busy (0.9% idle = 73.5 ms = 0.47% of
    window, the hard ceiling on ANY fix); 97.1% of in-wait GPU work is
    the waiting step's OWN, 0.0% is step N+1's (batch-queue
    inheritance hypothesis REFUTED); the copy event already fires
    13 us after sample-done and 1.93 ms BEFORE draft-done (the
    "narrow wait_stream to a sample event" candidate = 0.013 ms/step,
    killed before writing); engine dispatches N+1 ~21 ms before
    blocking on N. METHOD RULE: host overlap is not GPU overlap — a
    thread being inside forward(N+1) says nothing about N+1 kernels
    executing; only kernel-timestamp overlap discriminates. (b) The
    "71 ms arrival->scheduled segment" NEVER EXISTED — it was
    p50(TTFT) - p50(scheduled->first_token): PERCENTILES DO NOT
    SUBTRACT (means do, and reconcile exactly: 44.47+134.26=178.73).
    Real arrival->scheduled p50 27.37 ms; owner = engine step-loop
    cadence (request admitted by the very NEXT schedule() in p50-0
    steps; waits ~0.8 period due to arrival-phase correlation). The
    burst-start cohort's 52.6 ms is a transient COINCIDENTALLY equal
    to the 52 ms intercept — do not conflate. (c) THE TTFT BUDGET
    (mean 178.73 ms, n=60): 67.6% = the ADMITTING STEP's worker
    execution — by construction never FULL (mixed prefill+decode →
    uniform_token_count=None → no FULL descriptor): ~70.8 ms real
    prefill GPU work (~3x a decode step, largely irreducible) + ~22
    ms/step host-bound stall on the piecewise/eager path (UPPER BOUND,
    CUPTI-taxed); 24.2% = next-schedule() cadence wait; 6.7% = RPC/
    preprocess hop (execute_model.end -> forward.start p50 8.1 ms);
    all else <1.5%. TTFT levers in order: the mixed-step launch path,
    cadence, the hop. scheduled->first_token is ONE engine step for
    60/60 requests (step-id delta 0).

76. TRACE-TAX CALIBRATION + PINNED-PREALLOC ADOPTED (08-09): (a)
    3-arm identical-burst calibration: untraced TRUTH = TTFT 162.5 /
    emission ITL 28.6 / step 23.6 ms; graph-level trace +25% TTFT,
    +4% ITL, ±0 step; node-level +21% TTFT, +11% ITL, +19% step —
    node-mode's cudaGraphLaunch-fills-the-forward reading confirmed
    as tracer artifact; winB->winC "2-3 ms regression" resolved as
    trace-condition variance. RULE: adoption gates on untraced arms
    (already the practice — no past gate contaminated); node trace
    only for in-graph attribution; graph-level for NVTX timing work.
    (b) Hot-path cudaHostAlloc defect (reviewer-found, s=48: 27-29 ms
    x8 ranks + 28.9 ms all-GPU gap): mechanism = torch caching host
    allocator pays cudaHostAlloc only at NEW HIGH-WATER marks, so the
    batch-scaling grammar-bitmask staging (1.55 MB/step, 99.7% of
    hot-path pinned bytes) was the culprit at admission ramps. FIX
    ADOPTED (pinned_prealloc, commit d7ba6441f9): per-site persistent
    pinned pools (5 distinct, AST-guarded), superseded generations
    retained against in-flight copies, warmup high-water touch;
    33.7 MB/rank reserved. Gate: sample-range max 30.2 -> 3.38 ms,
    p99 1.60, no in-window alloc >0.14 ms; labeled remainder ~21 tiny
    allocs/rank (output-path pinned targets, out of scope, ~0.03% of
    window). Design rules worth keeping: per-site pools not shared
    (shared = reuse distance in CALLS not steps = correctness
    regression dressed as optimization); a pool that grows must
    RETAIN superseded buffers until quiesce.

77. REVIEWER-LOOP CLOSE (08-10, commit b9c3db3a81): pad-up dispatch
    ADOPTED (41-line standalone; mechanism corrected twice en route —
    not exact-key but mode-SHADOWING: decode keys at round_up(cs,dql)
    shadow mixed keys at cs in narrow bands under each round_up;
    correctness rested on padding being the existing 77%-majority
    path with slot_idx<0 guards cited to the fp8_ds_mla writer).
    Final composed profile (winJ2): ZERO NONE forwards — every mixed
    step in a captured bucket via pad-up (132->136 etc.); per-request
    TTFT p50 106.3 / p95 180.8 ms vs 163.6 / 290.7 before the fix
    chain (-35%/-38% under identical instrumentation). Five
    reviewer-driven adoptions total. Ops lessons: explicit || exit
    guards, never set -e + ERR-trap cleverness (a failed guard ran
    an entire uninstrumented capture that LOOKED complete); `git
    apply --3way` prints per-file "Applied cleanly" then rolls back
    everything (success-shaped failure output); archive superseded
    patches immediately (file ambiguity = the same failure shape);
    orphaned workers accumulate across restart generations — census
    GPUs by nvidia-smi compute-apps before every launch, not pgrep
    alone.
