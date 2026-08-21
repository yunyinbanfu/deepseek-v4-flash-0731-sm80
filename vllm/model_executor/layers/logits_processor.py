# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

import torch
import torch.nn.functional as F

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_gather,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform


def reduce_global_argmax(
    values: torch.Tensor,
    global_ids: torch.Tensor,
) -> torch.Tensor:
    """Reduce per-shard maxima that already carry global vocab ids.

    ``values`` and ``global_ids`` are ``[..., num_shards]``; the result is
    ``[...]``. Ties resolve to the lowest global id, taken as the minimum id
    among the shards that reach the maximum, so the result depends neither on
    shard order nor on ``torch.argmax``'s tie-break. That rule is the one that
    matches a replicated argmax over the concatenated vocab, which returns the
    first -- i.e. lowest -- maximal index.

    The masked-out lanes carry the row's largest id rather than a sentinel, so
    the result is always an id some shard offered. That matters for a NaN row,
    where ``values == best`` is false everywhere: an out-of-vocab id would be
    returned into an embedding lookup and fail as an async device-side assert,
    while a replicated ``argmax`` on the same row returns a valid (if
    meaningless) token.
    """
    best = values.max(dim=-1, keepdim=True).values
    fallback = global_ids.amax(dim=-1, keepdim=True).expand_as(global_ids)
    candidates = torch.where(values == best, global_ids, fallback)
    return candidates.min(dim=-1).values


# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()
        # Dtype of the lm_head projection. Defaults to the model dtype; an
        # fp32 head (via `--hf-overrides '{"head_dtype": "float32"}'`) is
        # required for RL training-inference consistency.
        model_config = get_current_vllm_config().model_config
        self.head_dtype = model_config.head_dtype if model_config is not None else None

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            logits = self._apply_cap_and_scale(logits)
        return logits

    def _apply_cap_and_scale(self, logits: torch.Tensor) -> torch.Tensor:
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project hidden states through the lm_head, honoring head_dtype."""
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            return lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias
            )

        if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError(
                "A head_dtype different from the model dtype is only "
                "supported for an unquantized lm_head."
            )
        if (
            self.head_dtype == torch.float32
            and (current_platform.is_cuda() or current_platform.is_rocm())
            and hidden_states.is_cuda
        ):
            # Accumulate the projection directly into fp32. This avoids
            # materializing an fp32 copy of the lm_head weight on every step,
            # unlike casting both operands. `torch.mm(out_dtype=...)` only
            # supports fp32 output for fp16/bf16 inputs, and is only
            # implemented for CUDA and ROCm (the latter via the non-Lt GEMM
            # path); other platforms fall back to the cast path below.
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.mm(flat, lm_head.weight.t(), out_dtype=self.head_dtype)
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(self.head_dtype)
            return logits.reshape(*hidden_states.shape[:-1], -1)
        return F.linear(
            hidden_states.to(self.head_dtype),
            lm_head.weight.to(self.head_dtype),
            embedding_bias.to(self.head_dtype) if embedding_bias is not None else None,
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)

        # Gather logits for TP
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_shard_logits(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        extra_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This rank's slice of the logits, stopping before the vocab gather.

        Everything :meth:`forward` applies elementwise (soft cap, scale) but
        without the all-gather, plus the padding mask that makes an argmax over
        the slice safe. The mask is not cosmetic: a shard whose vocab range
        runs past ``org_vocab_size`` holds rows the weight loader zero-fills
        (``vocab_parallel_embedding.py:485``), so a padded column scores 0 and
        wins outright whenever every real logit is negative. :meth:`forward`
        avoids this by truncating *after* the gather, which a sharded consumer
        cannot do.

        ``extra_logits`` is an optional ``[*, shard_width]`` term added before
        the mask, for a selection that spans the sum of two vocab-parallel
        projections over the same shard layout -- DSpark's Markov transition
        bias on top of the base draft logits.
        """
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        logits = self._apply_cap_and_scale(logits)
        if extra_logits is not None:
            logits = logits + extra_logits

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        extra_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = lm_head.tp_size

        logits = self.get_shard_logits(
            lm_head, hidden_states, embedding_bias, extra_logits
        )

        local_max_vals, local_max_indices = logits.max(dim=-1)

        # Convert shard-local indices to global vocab indices.
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        # Exchange (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        num_rows = hidden_states.shape[0]
        # envs re-reads os.environ per attribute access; bind once per call.
        use_allreduce = envs.VLLM_LOCAL_ARGMAX_ALLREDUCE
        if use_allreduce:
            # A sum-all_reduce over a buffer each rank zeroes outside its own
            # two lanes carries exactly what the all_gather carried: lane
            # 2r holds rank r's value and 2r+1 its global id, because every
            # other rank adds exact +0.0 there and x + 0.0 == x for every
            # float, inf and NaN included. Worth the wider payload: [B, 2]
            # takes the pynccl branch plus an all_gather temporary and a
            # movedim copy, while [B, 2 * tp] is still small enough for the
            # one-shot custom all-reduce.
            scattered = torch.zeros(
                num_rows, tp_size, 2, dtype=torch.float32, device=local_pair.device
            )
            scattered[:, lm_head.tp_rank] = local_pair
            gathered = tensor_model_parallel_all_reduce(scattered.view(num_rows, -1))
        else:
            # [batch, 2] -> [batch, 2 * tp_size]
            gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
        gathered = gathered.view(num_rows, tp_size, 2)
        selected = reduce_global_argmax(
            gathered[:, :, 0], gathered[:, :, 1].to(torch.int64)
        )
        if use_allreduce:
            # A no-op on real values -- every candidate is some rank's
            # `local_argmax + vocab_start`. It bounds the one case that is not
            # real: CustomAllreduce.custom_all_reduce's non-capturing warm-up
            # branch returns `torch.empty_like` (uninitialised). Every caller
            # turns this result into a token id and looks it up in an
            # embedding, so unclamped that warm-up dies on a device-side
            # assert. `all_gather` has no such path, hence the condition.
            selected = selected.clamp_(0, self.org_vocab_size - 1)
        return selected

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
