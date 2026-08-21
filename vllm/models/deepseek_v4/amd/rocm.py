# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import cast

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
    DeepseekV4FlashMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    _LAYER_TYPE_C4A,
    _LAYER_TYPE_C128A,
    _LAYER_TYPE_SWAONLY,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
    FlashMLASchedMeta,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_query_blocks,
    build_ragged_indices_from_dense,
    lengths_to_indptr,
    prefill_query_block_size,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
    rocm_sparse_attn_prefill_blocked,
)
from vllm.v1.worker.workspace import current_workspace_manager


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    return lengths_to_indptr(lengths)


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
            mask=swa_offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


def _build_ragged_into_graph_buffers(
    dense_indices: torch.Tensor,
    lengths: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ragged metadata straight into the persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses, so the returned
    tensors have to be backed by stable storage; indptr continues to bound
    reads. Building in place gets that for free -- the previous form allocated
    fresh tensors and copied them over, which cost two device-to-device
    memcpy nodes per builder per step.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    ragged_out = ragged_indices_buffer[: max(num_rows * max_entries_per_row, 1)]
    build_ragged_indices_from_dense(
        dense_indices,
        lengths,
        indices_out=ragged_out,
        indptr_out=indptr_out,
    )
    return ragged_out, indptr_out


def uniform_decode_group_size(
    causal: bool,
    num_decodes: int,
    num_decode_tokens: int,
    query_start_loc_cpu: torch.Tensor | None,
) -> int:
    """Query tokens per decode request, or 0 if the step is not blockable.

    The query-blocked decode kernel gives one CTA one request's whole query
    group, which needs two things this function checks and nothing else can:

    * **Every decode request has the same query count**, so blocks land on
      request boundaries. Captured decode batches are uniform by construction
      (``AttentionCGSupport.UNIFORM_BATCH``); an eager mixed batch need not be.
    * **The step is causal.** The DSpark *draft* step builds a non-causal SWA
      index list (every query also attends to future query tokens,
      ``sparse_swa.py::_compute_dspark_noncausal_swa_indices_kernel``), so
      those lists are not slices of one sliding run and the block's union
      derivation would not hold. Target-verify -- where the 43 layers and all
      of the decode time are -- is causal.
    """
    if not causal or num_decodes <= 0 or query_start_loc_cpu is None:
        return 0
    if num_decode_tokens % num_decodes:
        return 0
    group = num_decode_tokens // num_decodes
    if group < 2:
        return 0
    lens = query_start_loc_cpu[1 : num_decodes + 1] - query_start_loc_cpu[:num_decodes]
    return group if bool((lens == group).all()) else 0


@dataclass
class _PrefillChunkSlices:
    """Prefill metadata slices for one request chunk.

    Every DSv4 layer of a given type re-derives identical slices from the step's
    metadata, so they are built once per step and cached on the SWA metadata.
    """

    chunk_size: int
    query_start: int
    query_end: int
    seq_lens: torch.Tensor
    gather_lens: torch.Tensor
    swa_block_table: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    compressed_seq_lens: torch.Tensor | None
    compressed_block_table: torch.Tensor | None
    # (block_req, block_qstart) for the query-blocked ratio-128 path, keyed by
    # tile width. Derived from this chunk's query layout alone, so one build
    # serves all 20 ratio-128 layers.
    query_blocks: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )


@dataclass
class DeepseekV4ROCMAiterMLASparseMetadata(DeepseekV4FlashMLAMetadata):
    """ROCm-specific DeepSeek V4 metadata carrying ragged decode topk."""

    c128a_decode_topk_ragged_indices: torch.Tensor | None = None
    c128a_decode_topk_ragged_indptr: torch.Tensor | None = None


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None
    # Query tokens per decode request when every request has the same count and
    # the step is causal; 0 otherwise. See `uniform_decode_group_size`.
    decode_query_group_size: int = 0
    # Per-step prefill metadata slices, keyed by DSv4 layer type. Fresh instance per
    # build(), so the cache never outlives the metadata it was derived from.
    prefill_chunk_slices: dict[str, list[_PrefillChunkSlices]] = field(
        default_factory=dict
    )


class DeepseekV4ROCMAiterMLASparseMetadataBuilder(DeepseekV4FlashMLAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c128a_decode_topk_ragged_indices_buffer: torch.Tensor | None = None
        self.c128a_decode_topk_ragged_indptr_buffer: torch.Tensor | None = None
        if self.compress_ratio == 128:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        dense_decode = base.c128a_global_decode_topk_indices
        decode_lens = base.c128a_decode_topk_lens
        if dense_decode is not None and decode_lens is not None:
            assert self.c128a_decode_topk_ragged_indices_buffer is not None
            assert self.c128a_decode_topk_ragged_indptr_buffer is not None
            ragged_indices, ragged_indptr = _build_ragged_into_graph_buffers(
                dense_decode.reshape(dense_decode.shape[0], -1),
                decode_lens,
                self.c128a_decode_topk_ragged_indices_buffer,
                self.c128a_decode_topk_ragged_indptr_buffer,
                dense_decode.shape[0],
                self.c128a_max_compressed,
            )

        return DeepseekV4ROCMAiterMLASparseMetadata(
            **vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    def build_tile_scheduler(
        self, num_decode_tokens: int
    ) -> dict[str, FlashMLASchedMeta | None]:
        # The ragged Triton decode never calls FlashMLA, so skip planning
        # its scheduler metadata entirely.
        return dict.fromkeys((_LAYER_TYPE_SWAONLY, _LAYER_TYPE_C4A, _LAYER_TYPE_C128A))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # The non-causal (DSpark draft) path widens each token's SWA index list
        # to ``noncausal_index_width`` (>= window_size), so size the persistent
        # ragged buffer to the wider bound to cover both causal and non-causal.
        swa_index_width = max(self.window_size, self.noncausal_index_width)
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        if (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            ragged_indices, ragged_indptr = _build_ragged_into_graph_buffers(
                base.decode_swa_indices.reshape(base.num_decode_tokens, -1),
                base.decode_swa_lens,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                base.num_decode_tokens,
                # Actual dense width for this build: window_size (causal) or
                # noncausal_index_width (DSpark non-causal draft).
                base.decode_swa_indices.shape[-1],
            )

        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
            decode_query_group_size=uniform_decode_group_size(
                common_attn_metadata.causal,
                base.num_decodes,
                base.num_decode_tokens,
                base.query_start_loc_cpu,
            ),
        )


class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterMLASparseMetadataBuilder"]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder


class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Block scale for the preshuffled weight; None = not preshuffled.
        self._wqa_wkv_scale: torch.Tensor | None = None
        self._wo_b_scale: torch.Tensor | None = None

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def prepare_attn_preshuffle(self) -> None:
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )
        from vllm.model_executor.utils import replace_parameter

        def _prep(linear) -> torch.Tensor | None:
            w = getattr(linear, "weight", None)
            if w is None or w.dim() != 2:
                return None
            # K % 128 (group-128 quant) and N % 16 (shuffle_weight) must hold.
            if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
                return None
            ws = getattr(linear, "weight_scale_inv", None)  # per-block scale
            if ws is None:
                return None
            if ws.dtype == torch.float8_e8m0fnu:
                ws = _upcast_e8m0_to_fp32(ws).contiguous()
            # Shuffle the weight in place (single weight, no unshuffled copy).
            replace_parameter(
                linear,
                "weight",
                rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)),
            )
            return ws

        self._wqa_wkv_scale = _prep(self.fused_wqa_wkv)
        self._wo_b_scale = _prep(self.wo_b)

    def _bpre_attn_gemm(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        x: torch.Tensor,
        reduce_tp: bool,
    ) -> torch.Tensor:
        from vllm._aiter_ops import rocm_aiter_ops

        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
        out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_fp8, weight, x_scale, scale, output_dtype=x.dtype
        )
        if reduce_tp and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._wqa_wkv_scale is not None and hidden_states.dim() == 2:
            return self._bpre_attn_gemm(
                self.fused_wqa_wkv.weight, self._wqa_wkv_scale, hidden_states, False
            )
        return super()._fused_wqa_wkv_gemm(hidden_states)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        zf = z.flatten(1)
        if self._wo_b_scale is not None and zf.dim() == 2:
            return self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, True)
        return self.wo_b(zf)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                (
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    topk_lens,
                ) = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
            # Only the ratio-128 layers: their compressed lists are positional
            # prefixes that nest across a request's query group, so the group
            # shares one pass over the rows. The ratio-4 layers select per
            # query and cannot. (SWA-only layers qualify structurally too, but
            # carry 128 rows a query -- left on the per-query kernel so the
            # arm's attribution stays on the population it is priced against.)
            group_size=(
                swa_metadata.decode_query_group_size
                if self.compress_ratio == 128
                else 0
            ),
        )

    def _prefill_chunk_slices(
        self,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> list[_PrefillChunkSlices]:
        cache_key = (
            _LAYER_TYPE_SWAONLY
            if attn_metadata is None
            else (_LAYER_TYPE_C128A if self.compress_ratio == 128 else _LAYER_TYPE_C4A)
        )
        cached = swa_metadata.prefill_chunk_slices.get(cache_key)
        if cached is not None:
            return cached

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None
        assert gather_lens is not None
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None

        num_decodes = swa_metadata.num_decodes
        prefill_token_base = int(query_start_loc_cpu[num_decodes])
        swa_block_table = swa_metadata.block_table[num_decodes:]
        compressed_block_table = (
            None if attn_metadata is None else attn_metadata.block_table[num_decodes:]
        )

        chunks: list[_PrefillChunkSlices] = []
        for chunk_start in range(0, swa_metadata.num_prefills, self.PREFILL_CHUNK_SIZE):
            chunk_end = min(
                chunk_start + self.PREFILL_CHUNK_SIZE, swa_metadata.num_prefills
            )
            chunk_seq_lens = seq_lens[chunk_start:chunk_end]
            chunks.append(
                _PrefillChunkSlices(
                    chunk_size=chunk_end - chunk_start,
                    query_start=int(query_start_loc_cpu[num_decodes + chunk_start])
                    - prefill_token_base,
                    query_end=int(query_start_loc_cpu[num_decodes + chunk_end])
                    - prefill_token_base,
                    seq_lens=chunk_seq_lens,
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    swa_block_table=swa_block_table[chunk_start:chunk_end],
                    query_start_loc=query_start_loc[
                        num_decodes + chunk_start : num_decodes + chunk_end + 1
                    ],
                    query_start_loc_cpu=query_start_loc_cpu[
                        num_decodes + chunk_start : num_decodes + chunk_end + 1
                    ],
                    compressed_seq_lens=(
                        None
                        if compressed_block_table is None
                        else chunk_seq_lens // self.compress_ratio
                    ),
                    compressed_block_table=(
                        None
                        if compressed_block_table is None
                        else compressed_block_table[chunk_start:chunk_end]
                    ),
                )
            )

        swa_metadata.prefill_chunk_slices[cache_key] = chunks
        return chunks

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decode_tokens = swa_metadata.num_decode_tokens

        if not swa_only:
            assert attn_metadata is not None
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
            compressed_block_size = attn_metadata.block_size // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0
            compressed_block_size = 0

        M = N + self.window_size + self.max_num_batched_tokens

        # The ratio-128 layers have no indexer, so their index list is the
        # positional identity prefix plus the SWA window and consecutive
        # queries read nested rows -- the one population where a query block
        # can share a KV tile. The ratio-4 layers' top-512 sets are genuine
        # per-query selections, so a block of 8 could want 8x the rows.
        block_m = (
            prefill_query_block_size(q.shape[1], q.shape[2])
            if not swa_only and self.compress_ratio == 128
            else 0
        )

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk in self._prefill_chunk_slices(attn_metadata, swa_metadata):
            query_start = chunk.query_start
            query_end = chunk.query_end
            if not swa_only:
                assert compressed_k_cache is not None
                assert chunk.compressed_seq_lens is not None
                assert chunk.compressed_block_table is not None
                # compressed_k_cache is OCP on every platform (Triton encoder).
                dequantize_and_gather_k_cache(
                    kv[: chunk.chunk_size],
                    compressed_k_cache,
                    seq_lens=chunk.compressed_seq_lens,
                    gather_lens=None,
                    block_table=chunk.compressed_block_table,
                    block_size=compressed_block_size,
                    offset=0,
                    use_fnuz=False,
                )

            dequantize_and_gather_k_cache(
                kv[: chunk.chunk_size],
                swa_k_cache,
                seq_lens=chunk.seq_lens,
                gather_lens=chunk.gather_lens,
                block_table=chunk.swa_block_table,
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            if block_m:
                blocks = chunk.query_blocks.get(block_m)
                if blocks is None:
                    blocks = build_query_blocks(
                        chunk.query_start_loc_cpu, block_m, q.device
                    )
                    chunk.query_blocks[block_m] = blocks
                rocm_sparse_attn_prefill_blocked(
                    q=q[query_start:query_end],
                    kv=kv.view(-1, 1, q.shape[-1]),
                    block_req=blocks[0],
                    block_qstart=blocks[1],
                    query_start_loc=chunk.query_start_loc,
                    seq_lens=chunk.seq_lens,
                    gather_lens=chunk.gather_lens,
                    scale=self.scale,
                    head_dim=self.head_dim,
                    nope_head_dim=self.nope_head_dim,
                    rope_head_dim=self.rope_head_dim,
                    attn_sink=self.attn_sink,
                    top_k=top_k,
                    row_stride=M,
                    swa_offset=N,
                    compress_ratio=self.compress_ratio,
                    window_size=self.window_size,
                    block_m=block_m,
                    output=output[query_start:query_end],
                )
                continue

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                chunk.query_start_loc,
                chunk.seq_lens,
                chunk.gather_lens,
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            rocm_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )
