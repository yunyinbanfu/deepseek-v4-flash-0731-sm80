# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.kernels.linear.gemv_triton import (
    bf16_gemv,
    should_use_triton_gemv,
)
from vllm.model_executor.kernels.mhc.ar_int8 import ar_hoisted
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.common.ops import (
    fused_indexer_q_rope_quant,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.utils import balanced_row_bounds, balanced_row_counts
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)

# Below this many tokens, token-sharding the replicated input GEMMs cannot pay
# for the all-gather it adds: at TP=8 the merged trio's collective costs ~270 us
# against a GEMM that only reaches that size well into prefill. Decode widths
# (M<=8 under DSpark) are orders of magnitude below it.
_UNREPLICATE_MIN_TOKENS = 1024


@triton.jit
def _fill_short_context_topk_indices(
    output,
    positions,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    # small triton kernel that selects every candidate, -1 otherwise
    row = tl.program_id(0)
    offsets = tl.arange(0, PADDED_TOP_K)
    num_compressed = (tl.load(positions + row) + 1) // COMPRESS_RATIO
    tl.store(
        output + row * TOP_K + offsets,
        tl.where(offsets < num_compressed, offsets, -1),
        mask=offsets < TOP_K,
    )


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> tuple[str, torch.dtype]:
    """Map ``(layout, --kv-cache-dtype)`` to ``(cache_dtype_str, torch_dtype)``.

    Both layouts are paged; they differ in the per-token block format. The
    ``fp8_ds_mla`` format is UE8M0 block-scaled fp8 packed as ``uint8`` (the
    canonical ``fp8_ds_mla`` string is written back onto ``cache_config`` so the
    page-size specs pick the 576B per-token slot). Plain-row backends store each
    token's KV row in its element dtype: bf16 or per-tensor FP8 E4M3.
    """
    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8

    # Plain bf16 / per-tensor fp8 KV row (FlashInfer).
    if kv_cache_dtype.startswith("fp8"):
        return kv_cache_dtype, torch.float8_e4m3fn
    # auto / bfloat16 -> plain bf16 KV row.
    return kv_cache_dtype, torch.bfloat16


class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    """DeepseekV4 MLA attention layer.

    The platform-specific sparse-MLA forward (``forward_mqa`` /
    ``get_padded_num_q_heads`` / ``_o_proj`` / ``backend_cls``) is provided by a
    subclass — ``DeepseekV4FlashMLAAttention`` /
    ``DeepseekV4FlashInferSM120Attention`` /
    ``DeepseekV4FlashInferMLAAttention`` (CUDA) or
    ``DeepseekV4ROCMAiterMLAAttention`` (ROCm) — selected by the platform-specific
    deepseek_v4 model module. The base is never instantiated directly.
    """

    # Provided by the platform subclass.
    backend_cls: ClassVar[type[AttentionBackend]]
    # KV-cache per-token block format (both layouts are paged). True (default)
    # = fp8_ds_mla (UE8M0 block-scaled fp8 packed as uint8); False = plain
    # bf16 / per-tensor fp8 KV row. Backends can override the instance hook when
    # a single attention class dispatches across arch-specific layouts.
    use_fp8_ds_mla_layout: ClassVar[bool] = True
    # Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
    # workspace allocated in _forward_prefill and is also read by the dummy-run
    # path to pre-reserve that workspace.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    @classmethod
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q/output buffers are allocated at.

        The layer allocates the q/output buffers at
        ``[N, get_padded_num_q_heads(n_local_heads), head_dim]``. Must satisfy
        ``result >= num_heads``. Backends with no padding constraint return
        ``num_heads``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Platform-specific sparse MLA forward; writes attention into ``output``."""
        raise NotImplementedError

    @abstractmethod
    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + wo_a + wo_b output projection (platform-specific)."""
        raise NotImplementedError

    def _uses_fp8_ds_mla_layout(self) -> bool:
        """Return whether this instance stores fp8 KV in fp8_ds_mla layout."""
        return self.use_fp8_ds_mla_layout

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)

        self.prefix = prefix  # Alias for compatibility with compressor
        # Read once: these gate per-forward branches in every layer.
        self._unreplicate_gemms = envs.VLLM_UNREPLICATE_ATTN_GEMMS and tp_size > 1
        self._multi_stream_threshold = envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
        self._unreplicate_all_layers = envs.VLLM_UNREPLICATE_ATTN_GEMMS_ALL_LAYERS
        if self._unreplicate_gemms:
            # Rule 49: report the population the shard actually reaches, not
            # just that the flag is on. Only the ratio-4 layers carry the
            # merged input trio; every layer carries fused_wqa_wkv.
            n_layers = config.num_hidden_layers
            n_trio = sum(
                1 for r in config.compress_ratios[:n_layers] if max(1, r) == 4
            )
            reached = n_layers if self._unreplicate_all_layers else n_trio
            logger.info_once(
                "VLLM_UNREPLICATE_ATTN_GEMMS: token-sharding fused_wqa_wkv on "
                "%d/%d attention layers at >=%d tokens, TP=%d "
                "(%d carry the merged input trio and shard it too, %d do not).",
                reached,
                n_layers,
                _UNREPLICATE_MIN_TOKENS,
                tp_size,
                n_trio,
                n_layers - n_trio,
            )
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # NOTE(zyongye) Compress ratio can't be 0
        # we do this for because MTP layer is not included
        # in the compress ratio list
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5

        # Padded Q head count is dictated by the platform subclass.
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        # Sink padded to the same head count, initialized to -inf (no sink
        # effect). Weight loading fills the first n_local_heads slots.
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        # When the all-reduce is hoisted (task #35) the decoder layer performs it,
        # so this layer must not. Same predicate as the MoE half -- the suppressed
        # set and the re-added set are derived from one source, never two.
        self._ar_hoisted = ar_hoisted(vllm_config)
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            reduce_results=not self._ar_hoisted,
            prefix=f"{prefix}.wo_b",
        )

        # Initialize rotary embedding before the indexer/compressor consume it.
        self.rotary_emb = build_deepseek_v4_rope(
            config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.topk_indices_buffer = topk_indices_buffer
        self.eager_scratch_pool = eager_scratch_pool

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            # aux_stream_list[2] is free here (outer GEMMs joined) for the inner
            # overlap of wq_b+fused_indexer_q_rope_quant vs compressor. None on
            # ROCm, where aux_stream_list is None.
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
                eager_scratch_pool=eager_scratch_pool,
            )

        # Will be None on ROCm for now.
        self.aux_stream_list = aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        # ---- Attention / KV-cache setup ----
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len

        # Resolve the kv-cache dtype from this backend's block format. The same
        # resolution drives the SWA cache tensor dtype below.
        self.kv_cache_dtype, self.kv_cache_torch_dtype = _resolve_dsv4_kv_cache_dtype(
            self._uses_fp8_ds_mla_layout(), cache_config.cache_dtype, cache_config
        )

        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        # Register with compilation context for metadata lookup.
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self
        self.kv_cache = torch.tensor([])

        # Create the compressor for layers with compress_ratio > 1; after the
        # attention setup above so its KV-cache prefix (self.prefix) is set.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
                eager_scratch_pool=eager_scratch_pool,
            )

        # Built after weight loading by `fuse_input_gemm_weights`; see there.
        self.fused_input_weight: torch.Tensor | None = None
        self.fused_input_splits: list[int] = []

    def process_weights_after_loading(self, dtype: torch.dtype) -> None:
        # Loader hook (the AttentionLayerBase pass in model_loader/utils.py):
        # runs for every loader including DummyModelLoader, and inside
        # device_loading_context, so the merged buffer is built on the target
        # device even under CPU offloading.
        self.fuse_input_gemm_weights()

    def fuse_input_gemm_weights(self) -> None:
        """Concatenate the three bf16 input projections into one weight.

        `attn_gemm_parallel_execute` runs four GEMMs off the same
        hidden_states. Three of them are unquantized and read the same x, so
        one GEMM over the concatenated weight does the same work with a third
        of the launches and a third of the x traffic. Measured per layer on
        A100 with rotated (L2-cold) weights, us:

            M          1      6      8     64    2048
            separate  31.6   38.5   38.9   41.1  345.9
            merged    20.2   21.0   21.0   22.9  213.6

        i.e. -14 us/layer at batch-1 decode (x21 ratio-4 layers = -0.30 ms of
        a ~10.9 ms step) and -132 us/layer on a 2048-token prefill chunk.

        The three weights become views into the concatenated buffer, so the
        only lasting allocation is the copy that replaces them; the originals
        drop with their last reference. Nothing else has to change, because a
        row-slice of a contiguous [N, K] tensor is itself contiguous.
        """
        if self.compressor is None or self.indexer is None:
            # Only the ratio-4 layers carry all three; the others have one
            # input GEMM and nothing to merge.
            return
        if not current_platform.is_cuda():
            # The merged GEMM hands `weights_proj` to the indexer as fp32
            # instead of bf16. The Triton indexer-q kernel casts to fp32 on
            # entry either way (fused_indexer_q.py:169), but the XPU op
            # documents a bf16 input (_xpu_ops.py:396), so platforms other
            # than CUDA keep the three-GEMM path until measured there.
            return
        parts = [
            self.compressor.fused_wkv_wgate.weight,
            self.indexer.compressor.fused_wkv_wgate.weight,
            self.indexer.weights_proj.weight,
        ]
        if any(
            w is None or w.dtype != torch.bfloat16 or w.shape[1] != self.hidden_size
            for w in parts
        ):
            return
        merged = torch.cat([w.detach() for w in parts], dim=0).contiguous()
        splits = [w.shape[0] for w in parts]
        offset = 0
        for w, n in zip(parts, splits):
            w.data = merged[offset : offset + n]
            offset += n
        self.fused_input_weight = merged
        self.fused_input_splits = splits

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-allocate attention output with FlashMLA-padded head count.
        # The op writes into `o_padded`; we slice to n_local_heads after.
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Metadata-independent input GEMMs + RMSNorm stay in the captured
        # graph; the metadata-dependent rest (q up-proj + kv-insert, indexer,
        # compressor, MLA attention) runs in the eager break.
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )

        # attention_impl is wrapped with @eager_break_during_capture: this is
        # where the breakable cudagraph capture breaks (the attention op runs
        # eagerly between captured graph segments).
        self.attention_impl(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # Inverse-RoPE + wo_a + wo_b output projection (platform-specific).
        return self._o_proj(o, positions)

    @staticmethod
    def _shard_tokens(x: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """This rank's slice of the token dim, plus the split it came from.

        `balanced_row_counts` is the split the indexer query-shard and the mHC
        prenorm shard also use; sharing one partition is what lets these
        features compose (refutations rule 19).
        """
        tp = get_tensor_model_parallel_world_size()
        rows = balanced_row_counts(x.shape[0], tp)
        lo, hi = balanced_row_bounds(
            0, x.shape[0], get_tensor_model_parallel_rank(), tp
        )
        return x[lo:hi], rows

    @staticmethod
    def _gather_tokens(y: torch.Tensor, rows: list[int]) -> torch.Tensor:
        """Undo `_shard_tokens`: reassemble the exact rows each rank owned."""
        return get_tp_group().all_gatherv(y, dim=0, sizes=rows)

    def _unreplicate_tokens(self, n_tokens: int) -> bool:
        """Whether to token-shard the replicated input GEMMs for this batch.

        Both GEMMs are replicated because their consumers need full-width
        output on every rank (see the flag's note in envs.py), so sharding
        costs an all-gather. That only pays at prefill widths: at decode the
        GEMM is far smaller than the collective's fixed cost, and below one
        row per rank there is nothing to split.
        """
        return self._unreplicate_gemms and n_tokens >= _UNREPLICATE_MIN_TOKENS

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # MergedColumnParallelLinear returns (output, bias); bias is None.
        # Token-sharded callers pass their own slice and gather themselves.
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        return qr_kv

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        # fused_wqa_wkv (heaviest) on default; the three lighter input GEMMs
        # on aux streams 0..2 when their owning module exists. ln_events[0]
        # is the fan-out start event; ln_events[1..3] are per-aux done events.
        # On ROCm, aux_streams is None and execute_in_parallel runs serially.
        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        # fused_wqa_wkv is replicated on every rank in BOTH branches below, so
        # the token-shard applies to both; the merged input trio only decides
        # what else rides on the same sliced x. Gating the shard on the trio
        # left the 22 layers without an indexer running it at full M on all 8
        # ranks (rule 49).
        sharded = self._unreplicate_tokens(hidden_states.shape[0]) and (
            self.fused_input_weight is not None or self._unreplicate_all_layers
        )
        # Each GEMM gathers its own output; do NOT batch the gathers by
        # concatenating (measured regression -- see the refutations doc).
        # A copy-free variant (one ncclGroupStart/End around both
        # all_gatherv calls) is the only merge worth retrying.
        gemm_in, rows = (
            self._shard_tokens(hidden_states) if sharded else (hidden_states, [])
        )

        if self.fused_input_weight is not None:
            # One GEMM for all three (see fuse_input_gemm_weights). It occupies
            # a single aux slot, so the other two stay None and nothing waits on
            # events that were never recorded.
            merged_w = self.fused_input_weight
            splits = self.fused_input_splits

            def merged_input_gemm() -> torch.Tensor:
                # cuBLAS at every M; the CTA-per-row Triton GEMV wins only at
                # M=1 here, not worth a second shape-specific gate.
                return torch.mm(gemm_in, merged_w.T, out_dtype=torch.float32)

            aux_fns[0] = merged_input_gemm
            qr_kv, (merged_out, _, _) = execute_in_parallel(
                lambda: self._fused_wqa_wkv_gemm(gemm_in),
                aux_fns,
                self.ln_events[0],
                self.ln_events[1:4],
                aux_streams,
                enable=hidden_states.shape[0]
                <= self._multi_stream_threshold,
            )
            if sharded:
                qr_kv = self._gather_tokens(qr_kv, rows)
                merged_out = self._gather_tokens(merged_out, rows)
            kv_score, indexer_kv_score, indexer_weights = merged_out.split(
                splits, dim=-1
            )
            # weights_proj used to round through bf16 here; the merged GEMM
            # accumulates in fp32 and the consumer casts to fp32 anyway
            # (fused_indexer_q.py:169), so this is the more precise of the two.
            return qr_kv, kv_score, indexer_kv_score, indexer_weights

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # N=64 is narrow enough that cuBLAS splits K and reduces --
                # two launches for a 512 KB weight. One CTA per output row
                # does it in one; measured 5.4 -> 2.8 us at M=1.
                w = indexer.weights_proj.weight
                if should_use_triton_gemv(hidden_states, w):
                    return bf16_gemv(hidden_states, w)
                # ReplicatedLinear returns (output, bias); bias is None.
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        # Only fused_wqa_wkv rides the shard here. The aux GEMMs above stay
        # replicated: each emits a much smaller output against a much smaller
        # GEMM, so rule 14 has to price them separately before they move.
        def fused_wqa_wkv() -> torch.Tensor:
            return self._fused_wqa_wkv_gemm(gemm_in)

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= self._multi_stream_threshold,
        )
        if sharded:
            qr_kv = self._gather_tokens(qr_kv, rows)

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    @eager_break_during_capture
    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # wq_b + kv_insert (+ MLA compressor when an indexer is present) ride
        # on the default stream so q stays on its consumer stream (forward_mqa
        # downstream reads q on default). Indexer/compressor go on aux for
        # overlap with default's GEMM + cache write.
        if self.indexer is not None:
            aux_streams = self.aux_stream_list
            indexer = self.indexer
            # Local ref so the closure keeps a non-None type for mypy.
            assert self.compressor is not None
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                return q

            # 3-way overlap (matches TRT-LLM PR #14142 Level 1): default runs
            # wq_b+kv_insert; slot [0] runs the full indexer; slot [1] runs the
            # MLA compressor. Slot [2] is reserved for the indexer's inner
            # overlap. ROCm (aux_streams is None) falls back to sequential.
            q, _ = execute_in_parallel(
                wq_b_kv_insert,
                [
                    lambda: indexer(
                        hidden_states,
                        qr,
                        indexer_kv_score,
                        indexer_weights,
                        positions,
                        self.indexer_rotary_emb,
                    ),
                    lambda: compressor(kv_score, positions, self.rotary_emb),
                ],
                self.ln_events[0],
                [self.ln_events[1], self.ln_events[2]],
                [aux_streams[0], aux_streams[1]] if aux_streams is not None else None,
                enable=aux_streams is not None,
            )
        elif self.compressor is not None:
            # wq_b + kv_insert on default, compressor on aux.
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor

            def wq_b_kv_insert() -> torch.Tensor:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
                return q

            q, _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # MLA attention writes into the pre-allocated `out` buffer
        # ([num_tokens, padded_heads, head_dim]).
        self.forward_mqa(q, kv, positions, out)

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            # Profile run: kernel doesn't fire; produce a padded tensor so
            # downstream FlashMLA gets the right shape.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        # The fused insert ops require int64 position_ids; the runner's positions
        # buffer is already int64, so no cast is needed.
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cache_dtype = swa_kv_cache.dtype

        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if cache_dtype == torch.uint8:
            # fp8_ds_mla UE8M0 paged path. Horizontally fused:
            #   Q side:  per-head RMSNorm (no weight) + GPT-J RoPE, zero-filling
            #            the padding head slots.
            #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert.
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            if self.eager_scratch_pool is not None:
                q_out = self.eager_scratch_pool.q_out(q.shape[0])
                torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(
                    q,
                    kv,
                    q_out,
                    swa_kv_cache_2d,
                    swa_metadata.slot_mapping,
                    positions,
                    cos_sin_cache,
                    self.padded_heads,
                    self.eps,
                    swa_metadata.block_size,
                )
                return q_out
            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.padded_heads,
                self.eps,
                swa_metadata.block_size,
            )

        # Plain-row path: the [num_blocks, block_size, 512] cache stores the KV
        # row in its element dtype (no Q padding). bf16 rewrites q in place;
        # per-tensor fp8 writes a separately-allocated fp8 q and quantizes the
        # KV row.
        block_size = swa_metadata.block_size
        swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
        if cache_dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
            return q

        # per-tensor fp8 (torch.float8_e4m3fn)
        q_fp8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            swa_kv_cache_3d,
            swa_metadata.slot_mapping,
            positions,
            cos_sin_cache,
            self._flashinfer_fp8_kv_scale,
            self._flashinfer_fp8_q_scale_inv,
            self.eps,
            block_size,
        )
        return q_fp8

    def _global_topk_output_buffers(
        self, topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.compress_ratio != 4 or self.eager_scratch_pool is None:
            return None
        return self.eager_scratch_pool.global_topk_outputs(topk_indices)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
        aux_stream: torch.cuda.Stream | None = None,
        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1024
        self.compress_ratio = compress_ratio
        self.eager_scratch_pool = eager_scratch_pool
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        if self.use_fp4_kv:
            # MXFP4 stores two values per byte plus one UE8M0 byte per 32 values.
            # head_dim bytes = 64 packed values + 4 UE8M0 scales = 68.
            k_cache_head_dim = self.head_dim // 2 + self.head_dim // MXFP4_BLOCK_SIZE
        else:
            # NOTE(yifan): FP8 indexer cache uses the same layout as V3.2:
            # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
            k_cache_head_dim = (
                self.head_dim + self.head_dim // self.quant_block_size * 4
            )
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
            eager_scratch_pool=eager_scratch_pool,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
            num_heads=self.n_head,
        )

        # Q-path half of VLLM_INDEXER_QUERY_SHARD. Read once, as the builder
        # does. The FP4 path is excluded because its fused_indexer_q_rope_quant
        # contract returns a re-viewed scale tensor, which full-size output
        # buffers would have to reproduce outside the op; it also requires
        # SM100 datacenter hardware.
        self.shard_q_path = envs.VLLM_INDEXER_QUERY_SHARD and not self.use_fp4_kv

        # None on ROCm — maybe_execute_in_parallel falls back to sequential.
        self.aux_stream = aux_stream
        self.ln_events: list[torch.cuda.Event] = [
            torch.cuda.Event(),
            torch.cuda.Event(),
        ]

    def _sharded_q_row_ranges(
        self,
        attn_metadata: Any,
        num_tokens: int,
    ) -> list[tuple[int, int]] | None:
        """Query rows this rank's Q path must compute, or None for all of them.

        Completes VLLM_INDEXER_QUERY_SHARD: that flag already gives each rank a
        contiguous slice of the query rows to score, but every rank still runs
        the replicated `wq_b` GEMM and the fused RoPE/quant kernel over *all*
        rows and reads back one eighth. The ranges come from the chunk metadata
        the same flag produced, so they are exactly the rows the indexer will
        read -- there is no second partition that could disagree with it.
        """
        if not self.shard_q_path or not isinstance(attn_metadata, dict):
            return None
        indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
        prefill = indexer_metadata.prefill
        if prefill is None:
            return None
        # Derived once per step in the metadata builder; every indexer layer
        # reads the same answer. Guard against a caller whose row count
        # disagrees with the builder's rather than silently truncating.
        ranges = prefill.q_row_ranges
        if ranges is not None and ranges[-1][1] > num_tokens:
            ranges = None
        # Log both outcomes: a null A/B arm is otherwise indistinguishable from
        # a flag that never engaged. Both messages take a bounded set of
        # arguments, since `info_once` dedupes on them.
        if ranges is None:
            logger.info_once(
                "Indexer Q-path sharding INACTIVE (%s): running the replicated "
                "wq_b over every query row.",
                "batch contains decode requests"
                if indexer_metadata.num_decodes > 0
                else "VLLM_INDEXER_QUERY_SHARD did not shard this batch",
            )
        else:
            logger.info_once(
                "Indexer Q-path sharding ENGAGED: wq_b and the fused "
                "RoPE/quant kernel run over this rank's query rows only."
            )
        return ranges

    def _wq_b_and_q_quant_rows(
        self,
        row_ranges: list[tuple[int, int]],
        qr: torch.Tensor,
        positions: torch.Tensor,
        indexer_weights: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`wq_b` + fused RoPE/quant over `row_ranges` only.

        The outputs keep their full `[num_tokens, ...]` shape and the rows
        outside `row_ranges` are never written: `sparse_attn_indexer` indexes
        them globally (`q_quant[chunk.token_start : chunk.token_end]`), so
        compacting the tensors would mean re-deriving those offsets in a second
        place. Rows this rank does not own are never read -- `row_ranges` is
        built from the very chunk bounds that do the reading.
        """
        q_quant = torch.empty(
            (qr.shape[0], self.n_head, self.head_dim),
            dtype=current_platform.fp8_dtype(),
            device=qr.device,
        )
        weights = torch.empty(
            (qr.shape[0], self.n_head), dtype=torch.float32, device=qr.device
        )
        for lo, hi in row_ranges:
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr[lo:hi])
            q = q.view(-1, self.n_head, self.head_dim)
            fused_indexer_q_rope_quant(
                positions[lo:hi],
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights[lo:hi],
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=False,
                output_buffers=(q_quant[lo:hi], weights[lo:hi]),
            )
        return q_quant, weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        compressor = self.compressor

        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
            if indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:
                # candidates num smaller than topk, every candidate is selected
                # but we still need to build k cache
                compressor(compressed_kv_score, positions, rotary_emb)
                assert self.topk_indices_buffer is not None
                num_tokens = (
                    indexer_metadata.num_decode_tokens
                    + indexer_metadata.num_prefill_tokens
                )
                if num_tokens > 0:
                    _fill_short_context_topk_indices[(num_tokens,)](
                        self.topk_indices_buffer,
                        positions,
                        TOP_K=self.topk_tokens,
                        COMPRESS_RATIO=self.compress_ratio,
                        PADDED_TOP_K=triton.next_power_of_2(self.topk_tokens),
                        num_warps=8,
                    )
                return self.topk_indices_buffer

        q_row_ranges = self._sharded_q_row_ranges(attn_metadata, qr.shape[0])

        def wq_b_and_q_quant():
            if q_row_ranges is not None:
                return self._wq_b_and_q_quant_rows(
                    q_row_ranges, qr, positions, indexer_weights, rotary_emb
                )
            # ReplicatedLinear returns (output, bias); bias is None.
            q, _ = self.wq_b(qr)
            q = q.view(-1, self.n_head, self.head_dim)
            outputs = None
            if self.eager_scratch_pool is not None and self.use_fp4_kv:
                outputs = self.eager_scratch_pool.indexer_q_outputs(q.shape[0])
            return fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                indexer_weights,
                self.softmax_scale,
                self.n_head**-0.5,
                use_fp4=self.use_fp4_kv,
                output_buffers=outputs,
            )

        # compressor returns None and writes K to the indexer KV cache; the
        # join orders that write before indexer_op (skip_k_cache_insert=True).
        (q_quant, weights), k = maybe_execute_in_parallel(
            wq_b_and_q_quant,
            lambda: compressor(compressed_kv_score, positions, rotary_emb),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
