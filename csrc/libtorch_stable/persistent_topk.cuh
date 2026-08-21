/*
 * Persistent TopK Scheduler for DSA Indexer
 */

#ifndef PERSISTENT_TOPK_CUH_
#define PERSISTENT_TOPK_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>

#include "topk_histogram_4096.cuh"

namespace vllm {
namespace persistent {

// ============================================================================
// Constants
// ============================================================================

constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;

// Medium path: all shared state in dynamic smem (no static __shared__,
// which would inflate the kernel's smem footprint and kill occupancy
// for the decode/trivial paths).
constexpr size_t kMediumHistBytes = 2 * (RADIX + 128) * sizeof(int);  // 3072
constexpr size_t kMediumScalarsBytes = 5 * sizeof(int);               // 20
constexpr size_t kMediumHeaderSize =
    (kMediumHistBytes + kMediumScalarsBytes + 127) & ~size_t(127);  // 3200
constexpr int MAX_BUFFERED_ITEMS = 4096;
constexpr size_t kSmemMedium =
    kMediumHeaderSize + 2 * MAX_BUFFERED_ITEMS * sizeof(int);  // 35968
constexpr uint32_t RADIX_THRESHOLD = 32768;

// Decode path constants
constexpr int kDecodeBins = 2048;
constexpr uint32_t HIST2048_THRESHOLD = 8192;

// Large path: fixed shared memory for histograms + scalars
constexpr size_t kFixedSmemLarge =
    ((RADIX + RADIX + 5) * sizeof(uint32_t) + 15) & ~size_t(15);

// ============================================================================
// Common helpers
// ============================================================================

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

// ============================================================================
// Vectorized load helpers
// ============================================================================

// Unconditional float4 load with cache hint (.cg = cache at global level only).
__device__ __forceinline__ void load_float4(const float* ptr, float& v0,
                                            float& v1, float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// Per-element predicated scalar loads with -inf default.
__device__ __forceinline__ void load_float4_predicated(const float* ptr,
                                                       int base, int seq_len,
                                                       float& v0, float& v1,
                                                       float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  int p0 = (base < seq_len);
  int p1 = (base + 1 < seq_len);
  int p2 = (base + 2 < seq_len);
  int p3 = (base + 3 < seq_len);
  asm volatile(
      "{\n"
      "  .reg .pred pr0, pr1, pr2, pr3;\n"
      "  setp.ne.u32 pr0, %4, 0;\n"
      "  setp.ne.u32 pr1, %5, 0;\n"
      "  setp.ne.u32 pr2, %6, 0;\n"
      "  setp.ne.u32 pr3, %7, 0;\n"
      "  mov.u32 %0, 0xFF800000;\n"
      "  mov.u32 %1, 0xFF800000;\n"
      "  mov.u32 %2, 0xFF800000;\n"
      "  mov.u32 %3, 0xFF800000;\n"
      "  @pr0 ld.global.cg.u32 %0, [%8];\n"
      "  @pr1 ld.global.cg.u32 %1, [%8+4];\n"
      "  @pr2 ld.global.cg.u32 %2, [%8+8];\n"
      "  @pr3 ld.global.cg.u32 %3, [%8+12];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// ============================================================================
// Large path: inter-CTA coordination state (one per group)
// ============================================================================

struct RadixRowState {
  uint32_t histogram[3][256];  // Triple-buffered histograms
  uint32_t remaining_k;
  uint32_t prefix;
  int arrival_counter;
  int output_counter;
};

// ============================================================================
// Kernel parameters
// ============================================================================

struct PersistentTopKParams {
  const float* __restrict__ input;      // [num_rows, stride]
  int32_t* __restrict__ output;         // [num_rows, top_k]
  const int32_t* __restrict__ lengths;  // [num_rows]
  RadixRowState* row_states;            // large path: per-group state
  uint32_t num_rows;
  uint32_t stride;
  uint32_t top_k;           // actual k value for output stride
  uint32_t chunk_size;      // large path: elements per CTA
  uint32_t ctas_per_group;  // 1=medium, >1=large
  uint32_t max_seq_len;     // max seq_len across all rows (for early CTA exit)
};

// ============================================================================
// Decode path: 2048-bin histogram for short sequences (seq_len <= 8192)
// Uses 11-bit half-precision bins for fine granularity.
// One histogram pass typically suffices since 8192/2048 = 4 elements/bin avg.
// ============================================================================

// 11-bit bin from half-precision representation (ascending: high values -> high
// bins)
__device__ __forceinline__ uint32_t decode_bin(float x) {
  __half hx = __float2half(x);
  uint16_t bits = __half_as_ushort(hx);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return key >> 5;
}

template <int TopK>
__device__ __noinline__ void histogram_2048_topk(
    const float* __restrict__ logits, int32_t* __restrict__ output_indices,
    int32_t seq_len) {
  extern __shared__ int decode_smem[];
  const int tx = threadIdx.x;
  const int lane = tx & 31;

  // ---- Layout constants ----
  constexpr int SBASE = 8192 - 8;           // 8184
  constexpr int RHIST = RADIX + 128;        // 384
  constexpr int BOFF = 2 * RHIST;           // 768
  constexpr int DBUF = (SBASE - BOFF) / 2;  // 3708
  constexpr int MAX_ITEMS_PER_THREAD =
      (HIST2048_THRESHOLD + kThreadsPerBlock - 1) / kThreadsPerBlock;

  enum : int { sTHR = 0, sOUT = 1, sREF = 2, sFIN = 3, sBUF0 = 4, sBUF1 = 5 };

  // ---- Initialize scalars (prevents stale data from prior rows) ----
  if (tx < 8) {
    decode_smem[SBASE + tx] = 0;
  }

  // ---- Phase 1: Build 2048-bin histogram with float4 vectorized loads ----
  int* histo = decode_smem;
  uint16_t reg_bins[MAX_ITEMS_PER_THREAD];
  int nitems = 0;

  for (int i = tx; i < kDecodeBins; i += kThreadsPerBlock) {
    histo[i] = 0;
  }
  __syncthreads();

  const int n_vec = (seq_len + 3) >> 2;
  const bool row_aligned = ((reinterpret_cast<uintptr_t>(logits) & 15) == 0);

  for (int i = tx; i < n_vec; i += kThreadsPerBlock) {
    const int base = i << 2;
    float v0, v1, v2, v3;

    if (row_aligned && base + 3 < seq_len) {
      load_float4(logits + base, v0, v1, v2, v3);
    } else {
      load_float4_predicated(logits + base, base, seq_len, v0, v1, v2, v3);
    }

    const uint16_t b0 = static_cast<uint16_t>(decode_bin(v0));
    const uint16_t b1 = static_cast<uint16_t>(decode_bin(v1));
    const uint16_t b2 = static_cast<uint16_t>(decode_bin(v2));
    const uint16_t b3 = static_cast<uint16_t>(decode_bin(v3));
    reg_bins[nitems++] = b0;
    reg_bins[nitems++] = b1;
    reg_bins[nitems++] = b2;
    reg_bins[nitems++] = b3;
    atomicAdd(&histo[b0], 1);
    atomicAdd(&histo[b1], 1);
    atomicAdd(&histo[b2], 1);
    atomicAdd(&histo[b3], 1);
  }
  __syncthreads();

  // ---- CUB suffix sum ----
  using BlockScanT = cub::BlockScan<int, kThreadsPerBlock>;
  const int h0 = histo[2 * tx];
  const int pair_sum = h0 + histo[2 * tx + 1];

  auto& scan_storage = *reinterpret_cast<typename BlockScanT::TempStorage*>(
      decode_smem + kDecodeBins);

  int pair_prefix, total;
  BlockScanT(scan_storage).ExclusiveSum(pair_sum, pair_prefix, total);

  // Find threshold bin purely from registers
  const int pair_suffix = total - pair_prefix;

  if (pair_suffix >= TopK && (pair_suffix - h0) < TopK) {
    decode_smem[SBASE + sTHR] = 2 * tx;
  }
  {
    const int right_suf = pair_suffix - h0;
    const int next_suf = pair_suffix - pair_sum;
    if (right_suf >= TopK && next_suf < TopK) {
      decode_smem[SBASE + sTHR] = 2 * tx + 1;
    }
  }
  __syncthreads();

  const int threshold = decode_smem[SBASE + sTHR];

  // ---- Phase 2: exact-threshold refinement + index-ordered selection ----
  //
  // The historical collection claimed output slots with first-come-first-
  // served atomicAdds (order non-deterministic) and resolved boundary ties
  // FCFS from a capacity-limited buffer (set non-deterministic, and silently
  // dropped candidates past DBUF). The indexer feeds these indices into an
  // order-sensitive online-softmax, making temp=0 decoding non-deterministic
  // (#50576). Replaced by: (a) counting the strictly-above mass, (b) radix-
  // refining the threshold coarse bin to the exact 32-bit ordered key and
  // the slot count r among exactly-equal keys, (c) one index-ordered sweep
  // assigning stable positions via per-iteration block prefix scans. Output
  // is the selected indices in ascending order.
  const uint32_t uthr = static_cast<uint32_t>(threshold);
  const int n_vec_iters = (n_vec + kThreadsPerBlock - 1) / kThreadsPerBlock;
  int* refine[2] = {decode_smem, decode_smem + RHIST};
  int* warp_sums = decode_smem + BOFF;  // 32 ints; buffers are otherwise free

  // num_above = count of elements in bins strictly above the threshold bin.
  // histo[] (2048 bins) is still intact.
  {
    if (tx == 0) decode_smem[SBASE + sOUT] = 0;
    __syncthreads();
    int my_above = 0;
    for (int b = threshold + 1 + tx; b < kDecodeBins; b += kThreadsPerBlock) {
      my_above += histo[b];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      my_above += __shfl_down_sync(0xffffffff, my_above, offset);
    }
    if (lane == 0 && my_above > 0) {
      atomicAdd(&decode_smem[SBASE + sOUT], my_above);
    }
    __syncthreads();
  }
  const int num_above = decode_smem[SBASE + sOUT];
  int remaining_k = TopK - num_above;

  // Per-element candidate mask (coarse bin == threshold bin), kept in
  // registers; MAX_ITEMS_PER_THREAD <= 8 elements per thread here.
  uint32_t active = 0;
  {
    int item = 0;
    for (int iter = 0; iter < n_vec_iters; iter++) {
      const int i = tx + iter * kThreadsPerBlock;
      const bool vec_valid = (i < n_vec);
#pragma unroll 4
      for (int sub = 0; sub < 4; sub++) {
        const int elem_idx = (i << 2) + sub;
        uint32_t bin = 0;
        if (vec_valid) bin = reg_bins[item++];
        if (vec_valid && elem_idx < seq_len && bin == uthr) {
          active |= 1u << (iter * 4 + sub);
        }
      }
    }
  }

  // Radix-refine to the exact 32-bit ordered key. Values are re-read from
  // global (row is <= 32 KB, L1/L2 resident). Counting atomics commute, so
  // every round is deterministic.
  uint32_t exact_thr = 0;
#pragma unroll 4
  for (int pass = 0; pass < 4; ++pass) {
    const int bit_offset = 24 - pass * 8;
    __syncthreads();
    if (tx < RADIX + 1) refine[0][tx] = 0;
    __syncthreads();

    for (int iter = 0; iter < n_vec_iters; iter++) {
      const int i = tx + iter * kThreadsPerBlock;
#pragma unroll 4
      for (int sub = 0; sub < 4; sub++) {
        if (active & (1u << (iter * 4 + sub))) {
          const uint32_t fp32 = convert_to_uint32_v2(logits[(i << 2) + sub]);
          atomicAdd(&refine[0][(fp32 >> bit_offset) & 0xFF], 1);
        }
      }
    }
    __syncthreads();

#pragma unroll 8
    for (int s = 0; s < 8; ++s) {
      if (tx < RADIX) {
        const int stride = 1 << s;
        const int sb = s & 1;
        const int db = sb ^ 1;
        int value = refine[sb][tx];
        if (tx < RADIX - stride) value += refine[sb][tx + stride];
        refine[db][tx] = value;
      }
      __syncthreads();
    }

    if (tx < RADIX && refine[0][tx] >= remaining_k &&
        refine[0][tx + 1] < remaining_k) {
      decode_smem[SBASE + sREF] = tx;
      decode_smem[SBASE + sFIN] = refine[0][tx + 1];
    }
    __syncthreads();

    const int ref_thr = decode_smem[SBASE + sREF];
    remaining_k -= decode_smem[SBASE + sFIN];
    exact_thr |= static_cast<uint32_t>(ref_thr) << bit_offset;

    // Narrow the candidate mask to the chosen sub-bin.
    uint32_t next_active = 0;
    for (int iter = 0; iter < n_vec_iters; iter++) {
      const int i = tx + iter * kThreadsPerBlock;
#pragma unroll 4
      for (int sub = 0; sub < 4; sub++) {
        if (active & (1u << (iter * 4 + sub))) {
          const uint32_t fp32 = convert_to_uint32_v2(logits[(i << 2) + sub]);
          if (((fp32 >> bit_offset) & 0xFF) == static_cast<uint32_t>(ref_thr)) {
            next_active |= 1u << (iter * 4 + sub);
          }
        }
      }
    }
    active = next_active;
  }
  const int tie_slots = remaining_k;

  // ---- Phase 3: index-ordered selection sweep ----
  // Iteration `iter` covers the contiguous index span
  // [iter*4*kThreadsPerBlock, ...); within it, order is (thread, sub).
  if (tx == 0) {
    decode_smem[SBASE + sBUF0] = 0;  // running definite count
    decode_smem[SBASE + sBUF1] = 0;  // running tie count
  }
  __syncthreads();

  const int warp_id = tx >> 5;
  {
    int item = 0;
    for (int iter = 0; iter < n_vec_iters; iter++) {
      const int i = tx + iter * kThreadsPerBlock;
      const bool vec_valid = (i < n_vec);
      uint32_t def_bits = 0, tie_bits = 0;
      float vals[4];
#pragma unroll 4
      for (int sub = 0; sub < 4; sub++) {
        const int elem_idx = (i << 2) + sub;
        uint32_t bin = 0;
        if (vec_valid) bin = reg_bins[item++];
        if (!vec_valid || elem_idx >= seq_len) continue;
        if (bin > uthr) {
          def_bits |= 1u << sub;
        } else if (bin == uthr) {
          vals[sub] = logits[elem_idx];
          const uint32_t key = convert_to_uint32_v2(vals[sub]);
          if (key > exact_thr) {
            def_bits |= 1u << sub;
          } else if (key == exact_thr) {
            tie_bits |= 1u << sub;
          }
        }
      }

      const uint32_t packed =
          (static_cast<uint32_t>(__popc(def_bits)) << 16) |
          static_cast<uint32_t>(__popc(tie_bits));
      uint32_t winc = packed;
#pragma unroll
      for (uint32_t o = 1; o < 32; o *= 2) {
        const uint32_t n = __shfl_up_sync(0xffffffff, winc, o);
        if (lane >= static_cast<int>(o)) winc += n;
      }
      if (lane == 31) warp_sums[warp_id] = static_cast<int>(winc);
      __syncthreads();

      uint32_t inter_prefix = 0, iter_total = 0;
#pragma unroll
      for (int w = 0; w < kThreadsPerBlock / 32; ++w) {
        const uint32_t ws = static_cast<uint32_t>(warp_sums[w]);
        if (w < warp_id) inter_prefix += ws;
        iter_total += ws;
      }
      const uint32_t thread_excl = inter_prefix + (winc - packed);

      int def_prefix = decode_smem[SBASE + sBUF0] +
                       static_cast<int>(thread_excl >> 16);
      int tie_prefix = decode_smem[SBASE + sBUF1] +
                       static_cast<int>(thread_excl & 0xFFFF);

#pragma unroll 4
      for (int sub = 0; sub < 4; sub++) {
        const int elem_idx = (i << 2) + sub;
        if (def_bits & (1u << sub)) {
          const int tie_used = (tie_prefix < tie_slots) ? tie_prefix : tie_slots;
          output_indices[def_prefix + tie_used] = elem_idx;
          def_prefix++;
        } else if (tie_bits & (1u << sub)) {
          if (tie_prefix < tie_slots) {
            output_indices[def_prefix + tie_prefix] = elem_idx;
          }
          tie_prefix++;
        }
      }
      __syncthreads();
      if (tx == 0) {
        decode_smem[SBASE + sBUF0] += static_cast<int>(iter_total >> 16);
        decode_smem[SBASE + sBUF1] += static_cast<int>(iter_total & 0xFFFF);
      }
      __syncthreads();
    }
  }
}

// ============================================================================
// Medium path: coarse FP16 histogram + 4-pass FP32 radix refinement
// For sequences 8K < seq_len <= 64K.
// ============================================================================

// Adapted from:
// https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/elementwise/topk.cu#L87
// by: DarkSharpness
// which at the same time is an optimized topk kernel copied from tilelang
// kernel
template <int TopK>
__device__ __noinline__ void histogram_256_topk(
    const float* __restrict__ logits, int* __restrict__ output_indices,
    int logits_offset, int seq_len) {
  // All shared state lives in dynamic shared memory to avoid static
  extern __shared__ char medium_smem[];

  int (*shared_histogram)[RADIX + 128] =
      reinterpret_cast<int (*)[RADIX + 128]>(medium_smem);
  int* medium_scalars = reinterpret_cast<int*>(medium_smem + kMediumHistBytes);
  int& shared_output_count = medium_scalars[0];
  int& shared_threshold_bin = medium_scalars[1];
  int* shared_buffered_count = &medium_scalars[2];
  int& shared_final_k = medium_scalars[4];
  int (*buffered_indices)[MAX_BUFFERED_ITEMS] =
      reinterpret_cast<int (*)[MAX_BUFFERED_ITEMS]>(medium_smem +
                                                    kMediumHeaderSize);

  const int thread_id = threadIdx.x;
  int remaining_k = TopK;

  if (thread_id < RADIX + 1) {
    shared_histogram[0][thread_id] = 0;
  }
  __syncthreads();

  for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
    const auto bin = convert_to_uint8(logits[idx + logits_offset]);
    atomicAdd(&shared_histogram[0][bin], 1);
  }
  __syncthreads();

  auto compute_cumulative_sum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (__builtin_expect(thread_id < RADIX, 1)) {
        const int stride = 1 << i;
        const int src_buffer = i & 1;
        const int dst_buffer = src_buffer ^ 1;
        int value = shared_histogram[src_buffer][thread_id];
        if (thread_id < RADIX - stride) {
          value += shared_histogram[src_buffer][thread_id + stride];
        }
        shared_histogram[dst_buffer][thread_id] = value;
      }
      __syncthreads();
    }
  };

  compute_cumulative_sum();

  if (thread_id < RADIX && shared_histogram[0][thread_id] > remaining_k &&
      shared_histogram[0][thread_id + 1] <= remaining_k) {
    shared_threshold_bin = thread_id;
    shared_buffered_count[0] = 0;
    shared_output_count = 0;
  }
  __syncthreads();

  const int threshold_bin = shared_threshold_bin;
  remaining_k -= shared_histogram[0][threshold_bin + 1];

  // ---- Deterministic refinement + selection (see #50576) ----
  //
  // The historical version claimed output slots FCFS (order non-
  // deterministic), refined from a MAX_BUFFERED_ITEMS-capped buffer
  // (silently dropping candidates on overflow — with large tie masses,
  // e.g. relu'd logits, overflow is the common case) and resolved final
  // ties FCFS (set non-deterministic). Replaced with buffer-free radix
  // refinement over full-row rescans (a prefix pattern narrows candidates,
  // like sampler.cu's topKPerRowJob) and one index-ordered selection sweep
  // with per-iteration block prefix scans.
  const uint32_t uthr8 = static_cast<uint32_t>(threshold_bin);
  uint32_t exact_thr = 0;
  uint32_t refine_mask = 0;  // high bits of the ordered key fixed so far

  if (remaining_k > 0) {
#pragma unroll 4
    for (int pass = 0; pass < 4; ++pass) {
      const int bit_offset = 24 - pass * 8;
      __syncthreads();
      if (thread_id < RADIX + 1) {
        shared_histogram[0][thread_id] = 0;
      }
      __syncthreads();

      for (int idx = thread_id; idx < seq_len; idx += kThreadsPerBlock) {
        const float logit_value = logits[idx + logits_offset];
        if (convert_to_uint8(logit_value) != static_cast<int>(uthr8)) continue;
        const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
        if ((fp32_bits & refine_mask) != exact_thr) continue;
        atomicAdd(&shared_histogram[0][(fp32_bits >> bit_offset) & 0xFF], 1);
      }
      __syncthreads();

      compute_cumulative_sum();

      if (thread_id < RADIX &&
          shared_histogram[0][thread_id] >= remaining_k &&
          shared_histogram[0][thread_id + 1] < remaining_k) {
        shared_threshold_bin = thread_id;
        shared_final_k = shared_histogram[0][thread_id + 1];
      }
      __syncthreads();

      const int ref_thr = shared_threshold_bin;
      remaining_k -= shared_final_k;
      exact_thr |= static_cast<uint32_t>(ref_thr) << bit_offset;
      refine_mask |= 0xFFu << bit_offset;
      __syncthreads();
    }
  }
  const int tie_slots = remaining_k;

  // ---- Index-ordered selection sweep ----
  int* warp_sums = &buffered_indices[0][0];  // scratch; buffer is unused now
  if (thread_id == 0) {
    shared_buffered_count[0] = 0;  // running definite count
    shared_buffered_count[1] = 0;  // running tie count
  }
  __syncthreads();

  const int lane = thread_id & 31;
  const int warp_id = thread_id >> 5;
  for (int chunk = 0; chunk < seq_len; chunk += kThreadsPerBlock) {
    const int idx = chunk + thread_id;
    bool is_def = false, is_tie = false;
    if (idx < seq_len) {
      const float logit_value = logits[idx + logits_offset];
      const int bin = convert_to_uint8(logit_value);
      if (bin > threshold_bin) {
        is_def = true;
      } else if (bin == threshold_bin) {
        if (tie_slots == 0) {
          // remaining_k was 0 before refinement: nothing from this bin.
        } else {
          const uint32_t fp32_bits = convert_to_uint32_v2(logit_value);
          if (fp32_bits > exact_thr) {
            is_def = true;
          } else if (fp32_bits == exact_thr) {
            is_tie = true;
          }
        }
      }
    }

    const uint32_t packed = (is_def ? 0x10000u : 0u) | (is_tie ? 1u : 0u);
    uint32_t winc = packed;
#pragma unroll
    for (uint32_t o = 1; o < 32; o *= 2) {
      const uint32_t n = __shfl_up_sync(0xffffffff, winc, o);
      if (lane >= static_cast<int>(o)) winc += n;
    }
    if (lane == 31) warp_sums[warp_id] = static_cast<int>(winc);
    __syncthreads();

    uint32_t inter_prefix = 0, iter_total = 0;
#pragma unroll
    for (int w = 0; w < kThreadsPerBlock / 32; ++w) {
      const uint32_t ws = static_cast<uint32_t>(warp_sums[w]);
      if (w < warp_id) inter_prefix += ws;
      iter_total += ws;
    }
    const uint32_t thread_excl = inter_prefix + (winc - packed);

    const int def_prefix =
        shared_buffered_count[0] + static_cast<int>(thread_excl >> 16);
    const int tie_prefix =
        shared_buffered_count[1] + static_cast<int>(thread_excl & 0xFFFF);

    if (is_def) {
      const int tie_used = (tie_prefix < tie_slots) ? tie_prefix : tie_slots;
      output_indices[def_prefix + tie_used] = idx;
    } else if (is_tie && tie_prefix < tie_slots) {
      output_indices[def_prefix + tie_prefix] = idx;
    }
    __syncthreads();
    if (thread_id == 0) {
      shared_buffered_count[0] += static_cast<int>(iter_total >> 16);
      shared_buffered_count[1] += static_cast<int>(iter_total & 0xFFFF);
    }
    __syncthreads();
  }
}

// ============================================================================
// Inter-CTA sync primitives
// ============================================================================

__device__ __forceinline__ int ld_acquire(int* ptr) {
  int state = 0;
#if (__CUDA_ARCH__ >= 700)
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(state)
               : "l"(ptr));
#else
  asm volatile("ld.cg.global.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
#endif
  return state;
}

__device__ __forceinline__ void red_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
               :
               : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicAdd(ptr, val);
#endif
}

__device__ __forceinline__ void st_release(int* ptr, int val) {
#if (__CUDA_ARCH__ >= 700)
  asm volatile("fence.acq_rel.gpu;\n");
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" : : "l"(ptr), "r"(val));
#else
  __threadfence();
  atomicExch(ptr, val);
#endif
}

__device__ __forceinline__ void wait_ge(int* ptr, int target_val,
                                        int thread_idx) {
  if (thread_idx == 0) {
#pragma unroll 1
    while (ld_acquire(ptr) < target_val) {
    }
  }
  __syncthreads();
}

// ============================================================================
// Large path: multi-CTA radix select for sequences > 64K
//
// Each row is processed by a group of CTAs. Each CTA loads its chunk into
// shared memory as ordered uint32, then participates in 4 rounds of
// coordinated radix select via global-memory histograms and barriers.
// ============================================================================

// ============================================================================
// Multi-CTA cooperative RadixTopK for a single large row.
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

template <int TopK, uint32_t VEC_SIZE>
__device__ void radix_topk(const float* __restrict__ row_input,
                           int32_t* __restrict__ row_output, uint32_t seq_len,
                           uint32_t my_chunk_start, uint32_t chunk_size,
                           uint32_t* local_histogram, uint32_t* suffix_sum,
                           uint32_t* shared_scalars, uint32_t* shared_ordered,
                           RadixRowState* state, uint32_t cta_in_group,
                           uint32_t ctas_per_group, int& barrier_phase,
                           uint32_t iter, uint32_t tx) {
  const uint32_t my_chunk_end = (my_chunk_start + chunk_size < seq_len)
                                    ? my_chunk_start + chunk_size
                                    : seq_len;
  const uint32_t actual_chunk_size =
      (my_chunk_start < seq_len) ? (my_chunk_end - my_chunk_start) : 0;

  // -- Stage 1: Load chunk to shared memory as ordered uint32 --
  {
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

    for (uint32_t i = tx * VEC_SIZE; i < aligned_size;
         i += kThreadsPerBlock * VEC_SIZE) {
      const float* src = row_input + my_chunk_start + i;
      if constexpr (VEC_SIZE == 4) {
        float4 v = *reinterpret_cast<const float4*>(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
        shared_ordered[i + 2] = convert_to_uint32_v2(v.z);
        shared_ordered[i + 3] = convert_to_uint32_v2(v.w);
      } else if constexpr (VEC_SIZE == 2) {
        float2 v = *reinterpret_cast<const float2*>(src);
        shared_ordered[i] = convert_to_uint32_v2(v.x);
        shared_ordered[i + 1] = convert_to_uint32_v2(v.y);
      } else {
        shared_ordered[i] = convert_to_uint32_v2(*src);
      }
    }
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size;
         i += kThreadsPerBlock) {
      shared_ordered[i] = convert_to_uint32_v2(row_input[my_chunk_start + i]);
    }
  }
  __syncthreads();

  // -- Init radix select state --
  if (tx == 0) {
    shared_scalars[0] = 0;     // prefix
    shared_scalars[1] = TopK;  // remaining_k
  }
  __syncthreads();

  // Round 0's working histogram must be zero. The triple-buffer chain only
  // guarantees that when the previous group iteration was also a large row
  // (its round 2 zeroed this buffer); a short-row iteration in between runs
  // no rounds, leaving stale counts that corrupt the round-0 threshold (and
  // with it the pivot, the selected set, and the output bounds). Zero it
  // explicitly behind the initial barrier — no extra synchronization needed.
  if (cta_in_group == 0) {
    uint32_t* round0_hist = state->histogram[(iter * 4) % 3];
    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      round0_hist[i] = 0;
    }
    __syncthreads();
  }

  // -- Initial barrier --
  if (tx == 0) {
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  if (cta_in_group == 0 && tx == 0) {
    st_release(&state->output_counter, 0);
  }

  // -- Stage 2: 4 rounds of radix select --
  for (uint32_t round = 0; round < 4; round++) {
    const uint32_t global_round = iter * 4 + round;
    const uint32_t shift = 24 - round * 8;
    const uint32_t prefix = shared_scalars[0];
    const uint32_t remaining_k = shared_scalars[1];

    uint32_t* current_hist = state->histogram[global_round % 3];
    uint32_t* next_hist = state->histogram[(global_round + 1) % 3];

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      local_histogram[i] = 0;
    }
    __syncthreads();

    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      uint32_t ordered = shared_ordered[i];
      uint32_t mask = (round == 0) ? 0u : (~0u << (32 - round * 8));
      if ((ordered & mask) == prefix) {
        uint32_t bucket = (ordered >> shift) & 0xFF;
        atomicAdd(&local_histogram[bucket], 1);
      }
    }
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      if (local_histogram[i] > 0) {
        atomicAdd(&current_hist[i], local_histogram[i]);
      }
    }

    if (cta_in_group == 0) {
      for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
        next_hist[i] = 0;
      }
    }

    if (tx == 0) {
      red_release(&state->arrival_counter, 1);
    }
    wait_ge(&state->arrival_counter,
            (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
    barrier_phase++;
    __syncthreads();

    for (uint32_t i = tx; i < RADIX; i += kThreadsPerBlock) {
      suffix_sum[i] = current_hist[i];
    }
    __syncthreads();

    for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
      uint32_t val = 0;
      if (tx < RADIX) {
        val = suffix_sum[tx];
        if (tx + stride < RADIX) val += suffix_sum[tx + stride];
      }
      __syncthreads();
      if (tx < RADIX) suffix_sum[tx] = val;
      __syncthreads();
    }

    if (tx == 0) {
      shared_scalars[2] = 0;
      shared_scalars[3] = remaining_k;
    }
    __syncthreads();

    if (tx < RADIX) {
      uint32_t count_ge = suffix_sum[tx];
      uint32_t count_gt = (tx + 1 < RADIX) ? suffix_sum[tx + 1] : 0;
      if (count_ge >= remaining_k && count_gt < remaining_k) {
        shared_scalars[2] = tx;
        shared_scalars[3] = remaining_k - count_gt;
      }
    }
    __syncthreads();

    if (tx == 0) {
      shared_scalars[0] = prefix | (shared_scalars[2] << shift);
      shared_scalars[1] = shared_scalars[3];
    }
    __syncthreads();
  }  // end 4 radix rounds

  // -- Stage 3: deterministic collection (see #50576) --
  //
  // The historical collection claimed per-CTA output ranges FCFS on a global
  // counter and resolved == pivot ties FCFS across CTAs, so both output
  // order and the tie set depended on CTA/warp scheduling. Deterministic
  // scheme: each CTA publishes its (gt, eq) counts, a barrier makes them
  // visible, then every CTA derives its exclusive bases from lower-numbered
  // CTAs (chunks are contiguous index ranges, so CTA order == index order)
  // and writes via index-ordered per-iteration prefix scans. Output: all
  // > pivot indices in ascending order, then the first (TopK - total_gt)
  // == pivot indices in ascending order.
  const uint32_t ordered_pivot = shared_scalars[0];
  const uint32_t tie_slots = shared_scalars[1];

  // Local counts (deterministic block reduction).
  if (tx == 0) {
    suffix_sum[0] = 0;
    suffix_sum[1] = 0;
  }
  __syncthreads();
  {
    uint32_t my_gt = 0, my_eq = 0;
    for (uint32_t i = tx; i < actual_chunk_size; i += kThreadsPerBlock) {
      const uint32_t v = shared_ordered[i];
      my_gt += (v > ordered_pivot);
      my_eq += (v == ordered_pivot);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      my_gt += __shfl_down_sync(0xffffffff, my_gt, offset);
      my_eq += __shfl_down_sync(0xffffffff, my_eq, offset);
    }
    if (tx % 32 == 0) {
      if (my_gt > 0) atomicAdd(&suffix_sum[0], my_gt);
      if (my_eq > 0) atomicAdd(&suffix_sum[1], my_eq);
    }
  }
  __syncthreads();

  // Publish packed counts. chunk_size is bounded by shared memory
  // (< 64K elements), so 16 bits per component suffice. The counts buffer
  // reuses the histogram slot that is not referenced again until it is
  // zeroed (behind a barrier) in the next iteration's round 0/1.
  uint32_t* counts_buf = state->histogram[(iter * 4 + 5) % 3];
  if (tx == 0) {
    counts_buf[cta_in_group] = (suffix_sum[0] << 16) | suffix_sum[1];
    red_release(&state->arrival_counter, 1);
  }
  wait_ge(&state->arrival_counter,
          (barrier_phase + 1) * static_cast<int>(ctas_per_group), tx);
  barrier_phase++;
  __syncthreads();

  uint32_t my_gt_base = 0, my_eq_base = 0, total_gt = 0;
  for (uint32_t c = 0; c < ctas_per_group; ++c) {
    const uint32_t packed = counts_buf[c];
    const uint32_t gt_c = packed >> 16;
    if (c < cta_in_group) {
      my_gt_base += gt_c;
      my_eq_base += packed & 0xFFFF;
    }
    total_gt += gt_c;
  }

  // Index-ordered selection sweep over this CTA's chunk.
  uint32_t* warp_sums = local_histogram;        // 32 entries scratch
  uint32_t* running = local_histogram + 32;     // [def, tie]
  if (tx == 0) {
    running[0] = 0;
    running[1] = 0;
  }
  __syncthreads();

  const uint32_t lane = tx & 31;
  const uint32_t warp_id = tx >> 5;
  for (uint32_t chunk = 0; chunk < actual_chunk_size;
       chunk += kThreadsPerBlock) {
    const uint32_t i = chunk + tx;
    bool is_def = false, is_tie = false;
    if (i < actual_chunk_size) {
      const uint32_t v = shared_ordered[i];
      is_def = (v > ordered_pivot);
      is_tie = (v == ordered_pivot);
    }

    const uint32_t packed = (is_def ? 0x10000u : 0u) | (is_tie ? 1u : 0u);
    uint32_t winc = packed;
#pragma unroll
    for (uint32_t o = 1; o < 32; o *= 2) {
      const uint32_t n = __shfl_up_sync(0xffffffff, winc, o);
      if (lane >= o) winc += n;
    }
    if (lane == 31) warp_sums[warp_id] = winc;
    __syncthreads();

    uint32_t inter_prefix = 0, iter_total = 0;
#pragma unroll
    for (uint32_t w = 0; w < kThreadsPerBlock / 32; ++w) {
      const uint32_t ws = warp_sums[w];
      if (w < warp_id) inter_prefix += ws;
      iter_total += ws;
    }
    const uint32_t thread_excl = inter_prefix + (winc - packed);

    if (is_def) {
      const uint32_t pos = my_gt_base + running[0] + (thread_excl >> 16);
      row_output[pos] = static_cast<int32_t>(my_chunk_start + i);
    } else if (is_tie) {
      const uint32_t rank = my_eq_base + running[1] + (thread_excl & 0xFFFF);
      if (rank < tie_slots) {
        row_output[total_gt + rank] = static_cast<int32_t>(my_chunk_start + i);
      }
    }
    __syncthreads();
    if (tx == 0) {
      running[0] += iter_total >> 16;
      running[1] += iter_total & 0xFFFF;
    }
    __syncthreads();
  }
}

// ============================================================================
// Persistent kernel — BS≤32, decode/medium/large paths with RadixTopK
// BS>32 uses standalone histogram_256_buffered_topk (separate kernel,
// see filtered_topk.cuh)
// ============================================================================

template <int TopK = 2048, uint32_t VEC_SIZE = 1>
__global__ void __launch_bounds__(kThreadsPerBlock, 2)
    persistent_topk_kernel(PersistentTopKParams params) {
  const uint32_t tx = threadIdx.x;
  extern __shared__ uint8_t smem_raw[];

  // ========================================================================
  // Group mode: multi-CTA groups with static round-robin row assignment.
  // Non-large rows: CTA-0 handles trivial/decode/medium.
  // Large rows: all CTAs in the group cooperate via RadixTopK.
  // ========================================================================
  const uint32_t ctas_per_group = params.ctas_per_group;
  const uint32_t group_id = blockIdx.x / ctas_per_group;
  const uint32_t cta_in_group = blockIdx.x % ctas_per_group;
  const uint32_t num_groups = gridDim.x / ctas_per_group;
  const uint32_t chunk_size = params.chunk_size;

  if (blockIdx.x >= num_groups * ctas_per_group) return;

  // Early exit: non-CTA-0 threads are never needed if no large rows exist
  if (cta_in_group != 0 && params.max_seq_len <= RADIX_THRESHOLD) return;

  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem_raw);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  uint32_t* shared_ordered =
      reinterpret_cast<uint32_t*>(smem_raw + kFixedSmemLarge);

  // RadixRowState for multi-CTA cooperative radix.
  // Zero-initialization is done host-side via cudaMemsetAsync in topk.cu
  // before launch — that gives a stream-ordered happens-before edge for all
  // CTAs, which the previous in-kernel init (CTA-0 only + intra-CTA
  // __syncthreads) did not provide and which manifested as a race against
  // CTA-1+'s first red_release on arrival_counter.
  RadixRowState* state = &params.row_states[group_id];

  int barrier_phase = 0;
  const uint32_t total_iters = (params.num_rows + num_groups - 1) / num_groups;

  for (uint32_t iter = 0; iter < total_iters; iter++) {
    // Static round-robin: all CTAs in the group implicitly agree on the row
    uint32_t row_idx = group_id + iter * num_groups;
    if (row_idx >= params.num_rows) break;

    const uint32_t seq_len = params.lengths[row_idx];
    int32_t* row_output = params.output + row_idx * params.top_k;
    const float* row_input = params.input + row_idx * params.stride;

    if (seq_len <= RADIX_THRESHOLD) {
      if (cta_in_group == 0) {
        if (seq_len <= static_cast<uint32_t>(TopK)) {
          // Trivial case: seq_len <= TopK
          for (uint32_t i = tx; i < static_cast<uint32_t>(TopK);
               i += kThreadsPerBlock) {
            row_output[i] = (i < seq_len) ? static_cast<int32_t>(i) : -1;
          }
        } else if (seq_len <= static_cast<uint32_t>(HIST2048_THRESHOLD)) {
          histogram_2048_topk<TopK>(row_input, row_output, seq_len);
        } else {
          histogram_256_topk<TopK>(row_input, row_output, 0, seq_len);
        }
      }
      continue;
    }

    const uint32_t my_chunk_start = cta_in_group * chunk_size;
    radix_topk<TopK, VEC_SIZE>(
        row_input, row_output, seq_len, my_chunk_start, chunk_size,
        local_histogram, suffix_sum, shared_scalars, shared_ordered, state,
        cta_in_group, ctas_per_group, barrier_phase, iter, tx);
  }
}

}  // namespace persistent

// ============================================================================
// ============================================================================
// Optimized FilteredTopK — single CTA per row for bs > 32.
// Kept with persistent_topk so the portable fallback owns the non-cluster path.
// ============================================================================
namespace filtered_topk {

namespace hist4096 = topk_histogram_4096;

// ============================================================================
// FilteredTopK — single CTA per row for bs > 32
// Adapted from https://github.com/flashinfer-ai/flashinfer/pull/2215
// ============================================================================

#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }

#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T, size_t N>
struct vec_t {
  T data[N];

  FLASHINFER_INLINE T& operator[](size_t i) { return data[i]; }
  FLASHINFER_INLINE const T& operator[](size_t i) const { return data[i]; }

  FLASHINFER_INLINE void cast_load(const T* ptr) {
#pragma unroll
    for (size_t i = 0; i < N; ++i) {
      data[i] = ptr[i];
    }
  }
};
#undef FLASHINFER_INLINE

// FilteredTopK traits for different data types
template <typename DType>
struct FilteredTopKTraits;

// Specialization for float (32-bit): coarse histogram uses FP16 high 8 bits, 4
// refinement rounds
template <>
struct FilteredTopKTraits<float> {
  using OrderedType = uint32_t;
  static constexpr int NUM_REFINE_ROUNDS = 4;
  static constexpr int FIRST_REFINE_SHIFT = 24;

  __device__ __forceinline__ static uint8_t ToCoarseKey(float x) {
    // Convert to FP16 representation and extract high 8 bits
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
  }

  __device__ __forceinline__ static OrderedType ToOrdered(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  }
};

constexpr uint32_t FILTERED_TOPK_BLOCK_THREADS = 1024;
constexpr uint32_t FILTERED_TOPK_SMEM_INPUT_SIZE =
    16 * 1024;  // 16K indices per buffer
constexpr size_t FILTERED_TOPK_SMEM_DYNAMIC =
    sizeof(int) * 2 * FILTERED_TOPK_SMEM_INPUT_SIZE;  // 128KB

/*!
 * \brief Filtered Top-K kernel for ragged sequences.
 *
 * \tparam DType Data type (float, half, nv_bfloat16)
 * \tparam IdType Index type (int32_t)
 * \tparam VEC_SIZE Vector size for input loads (1, 2, 4, or 8)
 */
template <typename DType, typename IdType, int VEC_SIZE, uint32_t MAX_K = 2048,
          bool UsePredicatedShortLoads = false>
__global__ void __launch_bounds__(FILTERED_TOPK_BLOCK_THREADS)
    FilteredTopKUnifiedKernel(const DType* __restrict__ input,
                              IdType* __restrict__ output,
                              const IdType* __restrict__ lengths,
                              uint32_t num_rows, uint32_t top_k,
                              uint32_t max_len) {
  constexpr uint32_t BLOCK_SIZE = FILTERED_TOPK_BLOCK_THREADS;
  constexpr int RADIX = 256;
  constexpr int SMEM_INPUT_SIZE = FILTERED_TOPK_SMEM_INPUT_SIZE;

  const uint32_t bid = blockIdx.x;
  const int tx = threadIdx.x;

  if (bid >= num_rows) return;

  const int length =
      (lengths != nullptr) ? lengths[bid] : static_cast<int>(max_len);
  const DType* score = input + bid * max_len;
  IdType* dst = output + bid * top_k;

  // Trivial case: length <= top_k
  if (length <= static_cast<int>(top_k)) {
    for (int i = tx; i < static_cast<int>(top_k); i += BLOCK_SIZE) {
      dst[i] = (i < length) ? static_cast<IdType>(i) : static_cast<IdType>(-1);
    }
    return;
  }

  // Short path
  if (length <= 32768) {
    extern __shared__ uint8_t _smem_reg[];
    if constexpr (UsePredicatedShortLoads) {
      hist4096::histogram_4096_topk_predicated<MAX_K, 12, 8>(score, dst, length,
                                                             _smem_reg);
    } else {
      hist4096::histogram_4096_topk<MAX_K, 12, 8>(score, dst, length,
                                                  _smem_reg);
    }
    return;
  }

  // Static shared memory
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];
  alignas(128) __shared__ int s_indices[MAX_K];

  auto& s_histogram = s_histogram_buf[0];

  // Dynamic shared memory for input double buffer
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  using Traits = FilteredTopKTraits<DType>;
  int topk = top_k;

  // Stage 1: 8-bit coarse histogram with vectorized loads
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  vec_t<DType, VEC_SIZE> score_vec;

  const int aligned_length = (length / VEC_SIZE) * VEC_SIZE;
#pragma unroll 2
  for (int base = tx * VEC_SIZE; base < aligned_length;
       base += BLOCK_SIZE * VEC_SIZE) {
    score_vec.cast_load(&score[base]);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      const auto bin = Traits::ToCoarseKey(score_vec[j]);
      atomicAdd(&s_histogram[bin], 1);
    }
  }
  // Handle tail
  for (int i = aligned_length + tx; i < length; i += BLOCK_SIZE) {
    const auto bin = Traits::ToCoarseKey(score[i]);
    atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // Suffix sum
  const auto run_cumsum = [&]() {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  constexpr int NUM_ROUNDS = Traits::NUM_REFINE_ROUNDS;
  constexpr int FIRST_SHIFT = Traits::FIRST_REFINE_SHIFT;

  // ---- Deterministic refinement + selection (see #50576) ----
  //
  // The historical version claimed slots FCFS (order non-deterministic),
  // refined from an SMEM_INPUT_SIZE-capped buffer (silently dropping
  // candidates on overflow) and resolved final ties FCFS (set non-
  // deterministic). Replaced with buffer-free radix refinement over
  // full-row rescans narrowed by a prefix pattern, then one index-ordered
  // selection sweep writing `dst` directly in ascending index order.
  using Ordered = typename Traits::OrderedType;
  Ordered exact_thr = 0;
  Ordered refine_mask = 0;
  int tie_slots = topk;

  if (topk > 0) {
#pragma unroll
    for (int round = 0; round < NUM_ROUNDS; ++round) {
      const int offset = FIRST_SHIFT - round * 8;
      __syncthreads();
      if (tx < RADIX + 1) s_histogram[tx] = 0;
      __syncthreads();

      for (int i = tx; i < length; i += BLOCK_SIZE) {
        const auto raw = score[i];
        if (static_cast<int>(Traits::ToCoarseKey(raw)) != threshold_bin) {
          continue;
        }
        const Ordered ordered = Traits::ToOrdered(raw);
        if ((ordered & refine_mask) != exact_thr) continue;
        atomicAdd(&s_histogram[(ordered >> offset) & 0xFF], 1);
      }
      __syncthreads();

      run_cumsum();
      if (tx < RADIX && s_histogram[tx] >= tie_slots &&
          s_histogram[tx + 1] < tie_slots) {
        s_threshold_bin_id = tx;
        s_counter = s_histogram[tx + 1];
      }
      __syncthreads();

      const int sub_bin = s_threshold_bin_id;
      tie_slots -= s_counter;
      exact_thr |= static_cast<Ordered>(sub_bin) << offset;
      refine_mask |= static_cast<Ordered>(0xFF) << offset;
      __syncthreads();
    }
  }

  // Index-ordered selection sweep; positions from per-iteration block
  // prefix scans, so dst holds the selected indices in ascending order.
  int* warp_sums = s_indices;  // scratch; staging buffer is unused now
  if (tx == 0) {
    s_num_input[0] = 0;  // running definite count
    s_num_input[1] = 0;  // running tie count
  }
  __syncthreads();

  const uint32_t sweep_lane = tx & 31;
  const uint32_t sweep_warp = tx >> 5;
  for (int chunk = 0; chunk < length; chunk += BLOCK_SIZE) {
    const int i = chunk + tx;
    bool is_def = false, is_tie = false;
    if (i < length) {
      const auto raw = score[i];
      const int bin = static_cast<int>(Traits::ToCoarseKey(raw));
      if (bin > threshold_bin) {
        is_def = true;
      } else if (bin == threshold_bin && tie_slots >= 0 && topk > 0) {
        const Ordered ordered = Traits::ToOrdered(raw);
        if (ordered > exact_thr) {
          is_def = true;
        } else if (ordered == exact_thr) {
          is_tie = true;
        }
      }
    }

    const uint32_t packed = (is_def ? 0x10000u : 0u) | (is_tie ? 1u : 0u);
    uint32_t winc = packed;
#pragma unroll
    for (uint32_t o = 1; o < 32; o *= 2) {
      const uint32_t n = __shfl_up_sync(0xffffffff, winc, o);
      if (sweep_lane >= o) winc += n;
    }
    if (sweep_lane == 31) warp_sums[sweep_warp] = static_cast<int>(winc);
    __syncthreads();

    uint32_t inter_prefix = 0, iter_total = 0;
#pragma unroll
    for (uint32_t w = 0; w < BLOCK_SIZE / 32; ++w) {
      const uint32_t ws = static_cast<uint32_t>(warp_sums[w]);
      if (w < sweep_warp) inter_prefix += ws;
      iter_total += ws;
    }
    const uint32_t thread_excl = inter_prefix + (winc - packed);

    const int def_prefix =
        s_num_input[0] + static_cast<int>(thread_excl >> 16);
    const int tie_prefix =
        s_num_input[1] + static_cast<int>(thread_excl & 0xFFFF);

    if (is_def) {
      const int tie_used = (tie_prefix < tie_slots) ? tie_prefix : tie_slots;
      dst[def_prefix + tie_used] = static_cast<IdType>(i);
    } else if (is_tie && tie_prefix < tie_slots) {
      dst[def_prefix + tie_prefix] = static_cast<IdType>(i);
    }
    __syncthreads();
    if (tx == 0) {
      s_num_input[0] += static_cast<int>(iter_total >> 16);
      s_num_input[1] += static_cast<int>(iter_total & 0xFFFF);
    }
    __syncthreads();
  }
}

// Helper to compute GCD for VEC_SIZE selection
constexpr uint32_t gcd(uint32_t a, uint32_t b) {
  while (b != 0) {
    uint32_t t = b;
    b = a % b;
    a = t;
  }
  return a;
}

// Compute optimal VEC_SIZE based on max_len and dtype
// Returns 1, 2, 4, or 8
template <typename DType>
constexpr int ComputeFilteredTopKVecSize(uint32_t max_len) {
  constexpr int MAX_VEC = 16 / sizeof(DType);  // 4 for float32, 8 for fp16/bf16
  // Use GCD to find largest power-of-2 divisor
  const uint32_t g = gcd(max_len, static_cast<uint32_t>(MAX_VEC));
  return static_cast<int>(g);
}

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  constexpr size_t smem_size = FILTERED_TOPK_SMEM_DYNAMIC;
  constexpr int MAX_VEC = 16 / sizeof(DType);

  dim3 grid(num_rows);
  dim3 block(FILTERED_TOPK_BLOCK_THREADS);
  void* args[] = {&input,    &output_indices, &lengths,
                  &num_rows, &top_k_val,      &max_len};

  const int vec_size = ComputeFilteredTopKVecSize<DType>(max_len);

#define DISPATCH_VEC_SIZE(VS)                                                 \
  if (vec_size == VS) {                                                       \
    auto kernel =                                                             \
        FilteredTopKUnifiedKernel<DType, IdType, VS, MAX_K, (VS != MAX_VEC)>; \
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(                                \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));     \
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args,   \
                                          smem_size, stream));                \
    return cudaSuccess;                                                       \
  }

  DISPATCH_VEC_SIZE(1)
  DISPATCH_VEC_SIZE(2)
  DISPATCH_VEC_SIZE(4)
  if constexpr (MAX_VEC >= 8) {
    DISPATCH_VEC_SIZE(8)
  }
#undef DISPATCH_VEC_SIZE

  return cudaSuccess;
}

}  // namespace filtered_topk

template <typename DType, typename IdType, uint32_t MAX_K = 2048>
cudaError_t FilteredTopKRaggedTransform(const DType* input,
                                        IdType* output_indices,
                                        const IdType* lengths,
                                        uint32_t num_rows, uint32_t top_k_val,
                                        uint32_t max_len,
                                        cudaStream_t stream = 0) {
  return filtered_topk::FilteredTopKRaggedTransform<DType, IdType, MAX_K>(
      input, output_indices, lengths, num_rows, top_k_val, max_len, stream);
}

}  // namespace vllm

#endif  // PERSISTENT_TOPK_CUH_
