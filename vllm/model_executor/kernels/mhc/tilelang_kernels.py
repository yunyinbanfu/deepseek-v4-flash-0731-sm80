# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from functools import cache
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang
from vllm.utils.math_utils import cdiv

# TileLang is used for MHC on CUDA and ROCm. Keep non-GPU imports cheap so
# registering the Python wrapper modules does not require TileLang everywhere.
if TYPE_CHECKING or current_platform.is_cuda_alike():
    if not has_tilelang():
        raise ImportError(
            "tilelang is required for mhc but is not installed. Install it with "
            "`pip install tilelang`."
        )
    import tilelang
    import tilelang.language as T
else:
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]

ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()


@cache
def compute_num_split(block_k: int, k: int | None, grid_size: int) -> int:
    fixed = envs.VLLM_MHC_FIXED_NUM_SPLIT
    if fixed > 0:
        # Batch-invariant mode: grid_size tracks the batch token count, so the
        # occupancy-derived split factor (and with it the reduction order of
        # every token's mHC GEMM) changes with batch composition. Pin it,
        # keeping only the k-based clamp, which is constant per call site.
        split_k = fixed
        if k is not None:
            split_k = min(split_k, max(1, cdiv(k, block_k) // 4))
        return max(split_k, 1)
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // grid_size
    if k is not None:
        # avoid split_k for small k
        num_block_k = cdiv(k, block_k)
        split_k = min(split_k, num_block_k // 4)
    split_k = max(split_k, 1)
    return split_k


pass_configs: dict[tilelang.PassConfigKey, Any] = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

if current_platform.is_cuda():
    pass_configs[tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL] = 10


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
):
    """Deeply fused kernels, everything other than gemm & sqrsum in mHC pre block."""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    # outputs
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        if ENABLE_PDL:
            T.pdl_sync()
        ##################################################################
        # _pre_norm_fn_fwd_norm
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            ##################################################################
            # _pre_split_mixes_fwd (post & comb)
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            ##################################################################
            # _sinkhorn_fwd
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # save comb_mix to global memory
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            ##################################################################
            # _pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )
            ###################################################################
            # _pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


# Copied from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/mhc.py#L478


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_pre_big_fuse_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            # Pass 1: stash unnormalized weighted-sum output in shared memory
            # as bf16 (matches the rounding that RMSNorm would see) while
            # accumulating the per-position squared sum.
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            # Pass 2: scale by rsqrt * norm_weight and write the result to HBM.
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_pre_big_fuse_broadcast_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    residual_out,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 1,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        # rms scale
        rms[0] = T.rsqrt(rms[0] / hidden_size + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared(hidden_block, T.bfloat16)
                xl = T.alloc_fragment(hidden_block, T.float32)
                T.copy(residual[i, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                for i_hc, i1_h in T.Parallel(hc_mult, hidden_block):
                    residual_out[i, i_hc, i0_h * hidden_block + i1_h] = xs[i1_h]

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_fused_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    h_blk: int = 256,
    tile_n: int = 1,
    split_k: int = 1,
) -> tilelang.JITKernel:
    """Fused mhc post-mapping + pre-norm GEMM FMA"""
    m = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)
    h_per_split = h // split_k
    n_tiles = n_out // tile_n

    comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    weight_t: T.Tensor((n_out, hc, h), T.float32)  # type: ignore[no-redef, valid-type]
    yp_out: T.Tensor((split_k, m, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    rp_out: T.Tensor((split_k, m), T.float32)  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]

    h_iters = h_per_split // n_thr
    num_warps = n_thr // 32

    with T.Kernel(m, n_tiles, split_k, threads=n_thr) as (i_n, i_nt, i_ks):
        tid = T.get_thread_binding()
        warp_id = tid // 32
        lane = tid % 32

        s_warp = T.alloc_shared((num_warps, tile_n + 1), T.float32)
        s_post = T.alloc_shared((hc,), T.float32)
        s_comb = T.alloc_shared((hc, hc), T.float32)

        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        new_r = T.alloc_local((hc,), T.float32)

        T.clear(acc)
        T.clear(sqr)
        h_split_start = i_ks * h_per_split

        if ENABLE_PDL:
            T.pdl_sync()

        T.copy(post_mix[i_n, 0], s_post)
        T.copy(comb_mix[i_n, 0, 0], s_comb)

        for j in T.unroll(hc):
            pm[j] = s_post[j]
        for j in T.unroll(hc):
            for k in T.unroll(hc):
                cm[k, j] = s_comb[k, j]

        # Each thread owns h_iters elements of the k-split's h slice.
        for it in T.serial(h_iters):
            h_idx = h_split_start + it * n_thr + tid

            # Compute new residual from layer output and past residual
            for j in T.unroll(hc):
                new_r[j] = pm[j] * x_in[i_n, h_idx]
                for k in T.unroll(hc):
                    new_r[j] += cm[k, j] * residual_in[i_n, k, h_idx]

            # populate residual_out and compute sqr sum
            if i_nt == 0:
                for j in T.unroll(hc):
                    residual_out[i_n, j, h_idx] = new_r[j]
                    sqr[0] += new_r[j] * new_r[j]

            # Per-thread FMA into acc[n]
            for n in T.unroll(tile_n):
                for j in T.unroll(hc):
                    acc[n] += weight_t[i_nt * tile_n + n, j, h_idx] * new_r[j]

        for n in T.unroll(tile_n):
            acc[n] = T.warp_reduce_sum(acc[n])
        if i_nt == 0:
            sqr[0] = T.warp_reduce_sum(sqr[0])

        # Cross-warp reduce via shared mem
        if lane == 0:
            for n in T.unroll(tile_n):
                s_warp[warp_id, n] = acc[n]
            if i_nt == 0:
                s_warp[warp_id, tile_n] = sqr[0]
        T.sync_threads()

        # Warp 0 does the final cross-warp sum and writes outputs
        if warp_id == 0:
            if lane < tile_n:
                v = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v += s_warp[w, lane]
                yp_out[i_ks, i_n, i_nt * tile_n + lane] = v

            if i_nt == 0 and lane == 0:
                v2 = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v2 += s_warp[w, tile_n]
                rp_out[i_ks, i_n] = v2

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_post_int8_tilelang(
    a,
    b,
    c,
    d_q,
    d_s,
    x,
    hc: int,
    hidden: int,
    blk: int = 32,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    """mhc_post reading the all-reduce output in the compressed int8 form.

    Identical algebra to mhc_post_tilelang; only `d` changes representation,
    from bf16 to int8 codes plus one bf16 scale per `blk` elements. Fusing the
    dequantize here is what makes the compressed all-reduce pay: a bf16 phase 2
    would hand back half the wire saving. It also makes this kernel cheaper --
    `d` falls from 8 KB/token to 4.25 KB against ~72 KB of total traffic, and
    the dequantize is one multiply on a kernel that is bandwidth-bound.
    """
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d_q: T.Tensor((n, h), T.int8)  # type: ignore[no-redef, valid-type]
    d_s: T.Tensor((n, h // blk), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        dq_shared = T.alloc_shared(h_blk, T.int8)
        ds_shared = T.alloc_shared(h_blk // blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        if ENABLE_PDL:
            T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Serial(T.ceildiv(h, h_blk)):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d_q[i_n, i0_h * h_blk], dq_shared)
            T.copy(d_s[i_n, i0_h * h_blk // blk], ds_shared)

            T.copy(b_shared, b_local)
            # Dequantize: code i takes the scale of the block it belongs to.
            # The integer divide is what the scale-indexing unit test pins.
            for i1_h in T.Parallel(h_blk):
                d_local[i1_h] = dq_shared[i1_h] * ds_shared[i1_h // blk]

            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            T.copy(x_local, x[i_n, 0, i0_h * h_blk])
        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_post_tilelang(
    a,
    b,
    c,
    d,
    x,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    # rename for shorter code
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        if ENABLE_PDL:
            T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Serial(T.ceildiv(h, h_blk)):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            T.copy(x_local, x[i_n, 0, i0_h * h_blk])
        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def hc_prenorm_gemm_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hidden_size: int,
    hc_mult: int = 4,
    n_out: int = 24,
    n_thr: int = 512,
    tile_n: int = 12,
    n_splits: int = 1,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    hc_hidden_size = hc_mult * hidden_size
    k_per_split = hc_hidden_size // n_splits
    k_iters = k_per_split // n_thr
    n_tiles = T.ceildiv(n_out, tile_n)

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)  # type: ignore[no-redef, valid-type]
    fn: T.Tensor((n_out, hc_hidden_size), T.float32)  # type: ignore[no-redef, valid-type]
    out: T.Tensor((n_splits, num_tokens, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    sqrsum: T.Tensor((n_splits, num_tokens), T.float32)  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, n_tiles, n_splits, threads=n_thr) as (
        i_n,
        i_t,
        i_s,
    ):
        tid = T.get_thread_binding()
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        T.clear(acc)
        T.clear(sqr)

        if ENABLE_PDL:
            T.pdl_sync()

        for it in T.serial(k_iters):
            i_k = i_s * k_per_split + it * n_thr + tid
            x_val = x[i_n, i_k]
            for i_o in T.unroll(tile_n):
                out_idx = i_t * tile_n + i_o
                if out_idx < n_out:
                    acc[i_o] += x_val * fn[out_idx, i_k]
            if i_t == 0:
                sqr[0] += x_val * x_val

        for i_o in T.unroll(tile_n):
            acc[i_o] = T.warp_reduce_sum(acc[i_o])
        if i_t == 0:
            sqr[0] = T.warp_reduce_sum(sqr[0])

        lane = tid % 32
        warp_id = tid // 32
        num_warps = n_thr // 32
        warp_acc = T.alloc_shared((num_warps, tile_n), T.float32)
        warp_sqr = T.alloc_shared(num_warps, T.float32)

        if lane == 0:
            for i_o in T.unroll(tile_n):
                warp_acc[warp_id, i_o] = acc[i_o]
            if i_t == 0:
                warp_sqr[warp_id] = sqr[0]
        T.sync_threads()

        if warp_id == 0:
            if lane < tile_n:
                reduced_acc = T.alloc_var(T.float32, init=0.0)
                for i_w in T.unroll(num_warps):
                    reduced_acc += warp_acc[i_w, lane]
                out_idx = i_t * tile_n + lane
                if out_idx < n_out:
                    out[i_s, i_n, out_idx] = reduced_acc
            if lane == 0 and i_t == 0:
                reduced_sqr = T.alloc_var(T.float32, init=0.0)
                for i_w in T.unroll(num_warps):
                    reduced_sqr += warp_sqr[i_w]
                sqrsum[i_s, i_n] = reduced_sqr

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def hc_prenorm_gemm_block_m_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hidden_size: int,
    hc_mult: int = 4,
    n_out: int = 24,
    n_thr: int = 512,
    tile_n: int = 12,
    block_m: int = 2,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    hc_hidden_size = hc_mult * hidden_size
    k_iters = hc_hidden_size // n_thr
    n_tiles = T.ceildiv(n_out, tile_n)
    m_tiles = T.ceildiv(num_tokens, block_m)

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)  # type: ignore[no-redef, valid-type]
    fn: T.Tensor((n_out, hc_hidden_size), T.float32)  # type: ignore[no-redef, valid-type]
    out: T.Tensor((1, num_tokens, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    sqrsum: T.Tensor((1, num_tokens), T.float32)  # type: ignore[no-redef, valid-type]

    with T.Kernel(m_tiles, n_tiles, threads=n_thr) as (i_mt, i_t):
        tid = T.get_thread_binding()
        acc = T.alloc_local((block_m, tile_n), T.float32)
        sqr = T.alloc_local((block_m,), T.float32)
        T.clear(acc)
        T.clear(sqr)

        if ENABLE_PDL:
            T.pdl_sync()

        for it in T.serial(k_iters):
            i_k = it * n_thr + tid
            fn_val = T.alloc_local((tile_n,), T.float32)
            for i_o in T.unroll(tile_n):
                out_idx = i_t * tile_n + i_o
                if out_idx < n_out:
                    fn_val[i_o] = fn[out_idx, i_k]
                else:
                    fn_val[i_o] = 0.0
            for i_m in T.unroll(block_m):
                token_idx = i_mt * block_m + i_m
                if token_idx < num_tokens:
                    x_val = x[token_idx, i_k]
                    for i_o in T.unroll(tile_n):
                        acc[i_m, i_o] += x_val * fn_val[i_o]
                    if i_t == 0:
                        sqr[i_m] += x_val * x_val

        for i_m in T.unroll(block_m):
            for i_o in T.unroll(tile_n):
                acc[i_m, i_o] = T.warp_reduce_sum(acc[i_m, i_o])
            if i_t == 0:
                sqr[i_m] = T.warp_reduce_sum(sqr[i_m])

        lane = tid % 32
        warp_id = tid // 32
        num_warps = n_thr // 32
        warp_acc = T.alloc_shared((num_warps, block_m, tile_n), T.float32)
        warp_sqr = T.alloc_shared((num_warps, block_m), T.float32)

        if lane == 0:
            for i_m in T.unroll(block_m):
                for i_o in T.unroll(tile_n):
                    warp_acc[warp_id, i_m, i_o] = acc[i_m, i_o]
                if i_t == 0:
                    warp_sqr[warp_id, i_m] = sqr[i_m]
        T.sync_threads()

        if warp_id == 0:
            reduced_acc = T.alloc_local((block_m,), T.float32)
            reduced_sqr = T.alloc_local((block_m,), T.float32)
            T.clear(reduced_acc)
            T.clear(reduced_sqr)
            for i_m in T.unroll(block_m):
                token_idx = i_mt * block_m + i_m
                if token_idx < num_tokens:
                    if lane < tile_n:
                        for i_w in T.unroll(num_warps):
                            reduced_acc[i_m] += warp_acc[i_w, i_m, lane]
                        out_idx = i_t * tile_n + lane
                        if out_idx < n_out:
                            out[0, token_idx, out_idx] = reduced_acc[i_m]
                    if lane == 0 and i_t == 0:
                        for i_w in T.unroll(num_warps):
                            reduced_sqr[i_m] += warp_sqr[i_w, i_m]
                        sqrsum[0, token_idx] = reduced_sqr[i_m]

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def hc_head_fuse_tilelang(
    residual,
    fn,
    hc_scale,
    hc_base,
    out,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int = 4,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Two-pass fused kernel for hc_head.

    Pass 1: accumulate per-token squared sum and hc_mult dot-products
            (projections onto fn rows) using cross-thread reducers.
    Pass 2: apply sigmoid-gated weighted sum of residual channels to output.

    Avoids materialising mixes / rsqrt / pre tensors to global memory.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_dim = hc_mult * hidden_size
    h_block = math.gcd(h_blk, hidden_size)
    n_h = hidden_size // h_block

    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]
    fn: T.Tensor[[hc_mult, hc_dim], T.float32]  # type: ignore[no-redef,valid-type]
    hc_scale: T.Tensor[[1], T.float32]  # type: ignore[no-redef,valid-type]
    hc_base: T.Tensor[[hc_mult], T.float32]  # type: ignore[no-redef,valid-type]
    out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i:
        if ENABLE_PDL:
            T.pdl_sync()

        # ------------------------------------------------------------------
        # Pass 1 – for each residual channel m_c and h_block:
        #   • accumulate squared sum (for RMS norm denominator)
        #   • accumulate hc_mult dot-products with fn rows
        # ------------------------------------------------------------------
        sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        mixes_r = T.alloc_reducer((hc_mult,), T.float32, replication="all")
        T.fill(sqrsum_r, 0.0)
        T.fill(mixes_r, 0.0)

        for m_c in T.serial(hc_mult):
            for i_h in T.serial(n_h):
                x_local = T.alloc_fragment(h_block, T.float32)
                T.copy(residual[i, m_c, i_h * h_block], x_local)

                for k in T.Parallel(h_block):
                    sqrsum_r[0] += x_local[k] * x_local[k]

                for m_m in T.unroll(hc_mult):
                    fn_local = T.alloc_fragment(h_block, T.float32)
                    T.copy(fn[m_m, m_c * hidden_size + i_h * h_block], fn_local)
                    for k in T.Parallel(h_block):
                        mixes_r[m_m] += x_local[k] * fn_local[k]

        T.finalize_reducer(sqrsum_r)
        T.finalize_reducer(mixes_r)

        # ------------------------------------------------------------------
        # Compute pre_mix = sigmoid(mix * rsqrt * scale + base) + eps
        # ------------------------------------------------------------------
        pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
        rsqrt_val = T.alloc_fragment(1, T.float32)
        rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)
        for m in T.Parallel(hc_mult):
            pre_mix_shared[m] = (
                T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
            )

        # ------------------------------------------------------------------
        # Pass 2 – apply_mix: pipelined weighted sum over residual channels
        # ------------------------------------------------------------------
        for i0_h in T.Pipelined(n_h, num_stages=2):
            xs = T.alloc_shared((hc_mult, h_block), T.bfloat16)
            xl = T.alloc_fragment((hc_mult, h_block), T.float32)
            T.copy(residual[i, 0, i0_h * h_block], xs, disable_tma=True)
            T.copy(xs, xl)

            ol = T.alloc_fragment(h_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc_mult):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(h_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]

            T.copy(ol, out[i, i0_h * h_block], disable_tma=True)

        if ENABLE_PDL:
            T.pdl_trigger()


# ---------------------------------------------------------------------------
# Decode-shape experiments for the mHC boundary pair. At decode the boundary
# costs two launches -- mhc_fused_tilelang then mhc_pre_big_fuse_with_norm --
# and they are separated by the split-k reduction over `yp_out`, so the pair
# cannot merge without a device-wide barrier. What is addressable is the cost
# of each launch; these variants target the two terms the trace points at.
# ---------------------------------------------------------------------------


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_fused_block_m_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    tile_n: int = 4,
    split_k: int = 8,
    block_m: int = 8,
    w_bf16: bool = False,
) -> tilelang.JITKernel:
    """Token-blocked mhc post-mapping + pre-norm GEMM FMA.

    ``mhc_fused_tilelang`` puts the token index on the grid, so every token's
    CTA re-reads the whole of ``weight_t``: at the decode shape that is
    ``num_tokens * 1.57 MB`` of fp32 weight against a ~2 MB unique working set.
    Here the token loop lives inside the CTA, so each ``(n, k)`` weight element
    is fetched once and reused across ``block_m`` tokens -- the same lever
    ``hc_prenorm_gemm_block_m_tilelang`` applies on the prefill side.

    Two other differences from the grid-over-tokens kernel:
    ``residual_in``/``x_in`` are hoisted into registers before the unrolled
    ``hc`` loop rather than re-indexed inside it, and ``weight_t`` may be
    bf16 (``w_bf16``), halving the dominant traffic term.
    """
    m = T.dynamic("num_tokens")
    h = hidden
    h_per_split = h // split_k
    n_tiles = n_out // tile_n
    h_iters = h_per_split // n_thr
    num_warps = n_thr // 32
    w_dtype = T.bfloat16 if w_bf16 else T.float32
    split_k = T.dynamic("split_k")

    comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    weight_t: T.Tensor((n_out, hc, h), w_dtype)  # type: ignore[no-redef, valid-type]
    yp_out: T.Tensor((split_k, m, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    rp_out: T.Tensor((split_k, m), T.float32)  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]

    m_tiles = T.ceildiv(m, block_m)
    with T.Kernel(m_tiles, n_tiles, split_k, threads=n_thr) as (i_mt, i_nt, i_ks):
        tid = T.get_thread_binding()
        warp_id = tid // 32
        lane = tid % 32

        s_warp = T.alloc_shared((num_warps, block_m, tile_n + 1), T.float32)
        s_post = T.alloc_shared((block_m, hc), T.float32)
        s_comb = T.alloc_shared((block_m, hc, hc), T.float32)

        acc = T.alloc_local((block_m, tile_n), T.float32)
        sqr = T.alloc_local((block_m,), T.float32)
        wl = T.alloc_local((tile_n, hc), T.float32)
        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        new_r = T.alloc_local((hc,), T.float32)
        rin = T.alloc_local((hc,), T.float32)
        xv = T.alloc_local((1,), T.float32)

        T.clear(acc)
        T.clear(sqr)
        h_split_start = i_ks * h_per_split
        m_start = i_mt * block_m

        if ENABLE_PDL:
            T.pdl_sync()

        for mm, j in T.Parallel(block_m, hc):
            if m_start + mm < m:
                s_post[mm, j] = post_mix[m_start + mm, j]
        for mm, j, k in T.Parallel(block_m, hc, hc):
            if m_start + mm < m:
                s_comb[mm, j, k] = comb_mix[m_start + mm, j, k]
        T.sync_threads()

        for it in T.serial(h_iters):
            h_idx = h_split_start + it * n_thr + tid

            for n in T.unroll(tile_n):
                for j in T.unroll(hc):
                    wl[n, j] = weight_t[i_nt * tile_n + n, j, h_idx]

            for mm in T.unroll(block_m):
                if m_start + mm < m:
                    for j in T.unroll(hc):
                        pm[j] = s_post[mm, j]
                    for j in T.unroll(hc):
                        for k in T.unroll(hc):
                            cm[k, j] = s_comb[mm, k, j]

                    xv[0] = x_in[m_start + mm, h_idx]
                    for k in T.unroll(hc):
                        rin[k] = residual_in[m_start + mm, k, h_idx]

                    for j in T.unroll(hc):
                        new_r[j] = pm[j] * xv[0]
                        for k in T.unroll(hc):
                            new_r[j] += cm[k, j] * rin[k]

                    if i_nt == 0:
                        for j in T.unroll(hc):
                            residual_out[m_start + mm, j, h_idx] = new_r[j]
                            sqr[mm] += new_r[j] * new_r[j]

                    for n in T.unroll(tile_n):
                        for j in T.unroll(hc):
                            acc[mm, n] += wl[n, j] * new_r[j]

        for mm in T.unroll(block_m):
            for n in T.unroll(tile_n):
                acc[mm, n] = T.warp_reduce_sum(acc[mm, n])
            if i_nt == 0:
                sqr[mm] = T.warp_reduce_sum(sqr[mm])

        if lane == 0:
            for mm in T.unroll(block_m):
                for n in T.unroll(tile_n):
                    s_warp[warp_id, mm, n] = acc[mm, n]
                if i_nt == 0:
                    s_warp[warp_id, mm, tile_n] = sqr[mm]
        T.sync_threads()

        # A T.alloc_var is declared once, so its initializer does not re-run per
        # unrolled token; a local buffer written before each accumulation does.
        red = T.alloc_local((2,), T.float32)
        if warp_id == 0:
            for mm in T.unroll(block_m):
                if m_start + mm < m:
                    if lane < tile_n:
                        red[0] = 0.0
                        for w in T.unroll(num_warps):
                            red[0] += s_warp[w, mm, lane]
                        yp_out[i_ks, m_start + mm, i_nt * tile_n + lane] = red[0]
                    if i_nt == 0 and lane == 0:
                        red[1] = 0.0
                        for w in T.unroll(num_warps):
                            red[1] += s_warp[w, mm, tile_n]
                        rp_out[i_ks, m_start + mm] = red[1]

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_pre_big_fuse_with_norm_reg_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    n_thr: int = 96,
):
    """mhc_pre_big_fuse_with_norm with the sinkhorn held in registers.

    The production kernel is flat in worker width (8.33 us at both 64 and 256
    workers at batch 1), which places warp 0's ``sinkhorn_repeat`` dependent
    4x4 normalizations on the critical path. Warp 0 there keeps the matrix in
    a fragment, so each half-iteration costs a cross-lane ``T.reduce_sum`` plus
    16 fp32 divisions.

    Here every lane of warp 0 holds the whole 4x4 in registers and repeats the
    same arithmetic, which removes all 2 * sinkhorn_repeat cross-lane
    reductions, and each half-iteration divides ``hc_mult`` times instead of
    ``hc_mult**2`` by folding the normalizer into a reciprocal. Numerics move:
    ``x / y`` becomes ``x * (1 / y)``, so results agree to a few ulp per
    iteration rather than bit-exactly.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )

            cm = T.alloc_local((hc_mult, hc_mult), T.float32)
            rs = T.alloc_local((hc_mult,), T.float32)
            cs = T.alloc_local((hc_mult,), T.float32)

            for j in T.unroll(hc_mult):
                for k in T.unroll(hc_mult):
                    cm[j, k] = (
                        mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                        + hc_base[j * hc_mult + k + hc_mult * 2]
                    )

            for j in T.unroll(hc_mult):
                rs[j] = cm[j, 0]
                for k in T.unroll(hc_mult):
                    rs[j] = T.max(rs[j], cm[j, k])
            for j in T.unroll(hc_mult):
                for k in T.unroll(hc_mult):
                    cm[j, k] = T.exp(cm[j, k] - rs[j])

            for j in T.unroll(hc_mult):
                rs[j] = 0.0
                for k in T.unroll(hc_mult):
                    rs[j] += cm[j, k]
                rs[j] = 1.0 / rs[j]
            for j in T.unroll(hc_mult):
                for k in T.unroll(hc_mult):
                    cm[j, k] = cm[j, k] * rs[j] + hc_sinkhorn_eps

            for k in T.unroll(hc_mult):
                cs[k] = 0.0
                for j in T.unroll(hc_mult):
                    cs[k] += cm[j, k]
                cs[k] = 1.0 / (cs[k] + hc_sinkhorn_eps)
            for j in T.unroll(hc_mult):
                for k in T.unroll(hc_mult):
                    cm[j, k] *= cs[k]

            for _ in T.serial(sinkhorn_repeat - 1):
                for j in T.unroll(hc_mult):
                    rs[j] = 0.0
                    for k in T.unroll(hc_mult):
                        rs[j] += cm[j, k]
                    rs[j] = 1.0 / (rs[j] + hc_sinkhorn_eps)
                for j in T.unroll(hc_mult):
                    for k in T.unroll(hc_mult):
                        cm[j, k] *= rs[j]

                for k in T.unroll(hc_mult):
                    cs[k] = 0.0
                    for j in T.unroll(hc_mult):
                        cs[k] += cm[j, k]
                    cs[k] = 1.0 / (cs[k] + hc_sinkhorn_eps)
                for j in T.unroll(hc_mult):
                    for k in T.unroll(hc_mult):
                        cm[j, k] *= cs[k]

            if T.get_thread_binding() == 0:
                for j in T.unroll(hc_mult):
                    for k in T.unroll(hc_mult):
                        comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_post_sqrsum_int8_tilelang(
    a,
    b,
    c,
    d_q,
    d_s,
    x,
    sqrsum,
    hc: int,
    hidden: int,
    blk: int = 32,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    """mhc_post_sqrsum reading the all-reduce output in the compressed form.

    The variant VLLM_MHC_POST_FUSE_SQRSUM=1 + VLLM_MHC_AR_INT8=1 selects.
    Only `d`'s representation changes; the fold is untouched, because it reduces
    over `x_local` -- this kernel's own output, which stays bf16 -- and never
    reads `d`. So the disjoint ownership the prenorm shard relies on
    (fold owns gemm_out_sqrsum, shard owns gemm_out_mul) is unaffected, and
    VLLM_MHC_PRENORM_SHARD's `sqrsum is None` guard still means what it meant.
    """
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d_q: T.Tensor((n, h), T.int8)  # type: ignore[no-redef, valid-type]
    d_s: T.Tensor((n, h // blk), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    sqrsum: T.Tensor((1, n), T.float32)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        dq_shared = T.alloc_shared(h_blk, T.int8)
        ds_shared = T.alloc_shared(h_blk // blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)
        sq_acc = T.alloc_fragment((hc, h_blk), T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        if ENABLE_PDL:
            T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)
        T.clear(sq_acc)

        for i0_h in T.Serial(T.ceildiv(h, h_blk)):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d_q[i_n, i0_h * h_blk], dq_shared)
            T.copy(d_s[i_n, i0_h * h_blk // blk], ds_shared)

            T.copy(b_shared, b_local)
            for i1_h in T.Parallel(h_blk):
                d_local[i1_h] = dq_shared[i1_h] * ds_shared[i1_h // blk]

            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            for i_hco, i1_h in T.Parallel(hc, h_blk):
                xb = T.float32(T.bfloat16(x_local[i_hco, i1_h]))
                sq_acc[i_hco, i1_h] += xb * xb

            T.copy(x_local, x[i_n, 0, i0_h * h_blk])

        sq_row = T.alloc_fragment(hc, T.float32)
        sq_tot = T.alloc_fragment(1, T.float32)
        T.reduce_sum(sq_acc, sq_row, dim=1)
        T.reduce_sum(sq_row, sq_tot, dim=0)
        if T.get_thread_binding() == 0:
            sqrsum[0, i_n] = sq_tot[0]
        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit(
    pass_configs=pass_configs,
)
def mhc_post_sqrsum_tilelang(
    a,
    b,
    c,
    d,
    x,
    sqrsum,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    """mhc_post plus the prenorm GEMM's companion row sqrsum.

    Above the cuBLAS crossover the prefill boundary writes the post-mapped
    residual and then reads it back twice: once for the GEMM and once for
    ``_row_sqrsum_kernel``. The sqrsum is a pure reduction over data this
    kernel already holds in registers, so folding it in removes a full
    ``num_tokens * hc * hidden`` bf16 read (268 MB per boundary at T=8192)
    and one launch, at no extra traffic here.

    The square is taken on the bf16-rounded value so the result matches the
    standalone reduction over the stored tensor.
    """
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    sqrsum: T.Tensor((1, n), T.float32)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)
        sq_acc = T.alloc_fragment((hc, h_blk), T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        if ENABLE_PDL:
            T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)
        T.clear(sq_acc)

        for i0_h in T.Serial(T.ceildiv(h, h_blk)):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            for i_hco, i1_h in T.Parallel(hc, h_blk):
                xb = T.float32(T.bfloat16(x_local[i_hco, i1_h]))
                sq_acc[i_hco, i1_h] += xb * xb

            T.copy(x_local, x[i_n, 0, i0_h * h_blk])

        sq_row = T.alloc_fragment(hc, T.float32)
        sq_tot = T.alloc_fragment(1, T.float32)
        T.reduce_sum(sq_acc, sq_row, dim=1)
        T.reduce_sum(sq_row, sq_tot, dim=0)
        if T.get_thread_binding() == 0:
            sqrsum[0, i_n] = sq_tot[0]
        if ENABLE_PDL:
            T.pdl_trigger()
