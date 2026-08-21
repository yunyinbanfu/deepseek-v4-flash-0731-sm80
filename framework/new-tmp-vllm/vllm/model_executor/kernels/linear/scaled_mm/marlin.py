# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import block_dequant
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    is_fp8_marlin_supported,
    prepare_fp8_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kFp8Static128BlockSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)


def _dequant_excluded(layer: torch.nn.Module) -> bool:
    prefix = getattr(layer, "prefix", "")
    include_patterns = [
        p for p in envs.VLLM_MARLIN_FP8_DEQUANT_INCLUDE.split(",") if p
    ]
    if include_patterns and not is_layer_skipped(
        prefix, include_patterns, skip_with_substr=True
    ):
        logger.debug("keeping %s on Marlin (dequant include miss)", prefix)
        return True
    patterns = [p for p in envs.VLLM_MARLIN_FP8_DEQUANT_EXCLUDE.split(",") if p]
    if not patterns:
        return False
    excluded = is_layer_skipped(prefix, patterns, skip_with_substr=True)
    if excluded:
        logger.debug("keeping %s on Marlin (dequant exclusion)", prefix)
    return excluded


class MarlinFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """
    FP8 Marlin kernel for GPUs that lack FP8 hardware support.
    Leverages the Marlin kernel for fast weight-only FP8 quantization.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."
        # Check if platform supports FP8 Marlin
        if not is_fp8_marlin_supported():
            return False, "FP8 Marlin requires compute capability 7.5 or higher"
        if envs.VLLM_BATCH_INVARIANT:
            return False, "FP8 Marlin not supported for batch invariant execution."
        if (
            compute_capability is not None
            and compute_capability >= 89
            and not envs.VLLM_TEST_FORCE_FP8_MARLIN
        ):
            return (
                False,
                "To apply FP8 Marlin on high-capability GPUs, please set "
                "VLLM_TEST_FORCE_FP8_MARLIN=1",
            )
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        super().__init__(c, layer_param_names)
        self.marlin_input_dtype = None
        self.block_quant = self.config.weight_quant_key in {kFp8Static128BlockSym}
        self.size_k_first = not self.block_quant

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.block_quant:
            weight, weight_scale_inv = process_fp8_weight_block_strategy(
                layer.weight, layer.weight_scale_inv
            )
            # Update layer with new values
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale_inv", weight_scale_inv.data)
        # Non-block: callers must pass weight in (K, N) layout.

        if getattr(layer, "is_bmm", False):
            # BMM layers (DeepSeek V4 `wo_a`) are consumed as raw block-fp8
            # weight + weight_scale_inv by the attention einsum, never
            # through apply_weights(); the Marlin repack would destroy them.
            # Same exemption the deep_gemm and xpu kernels make.
            return

        if envs.VLLM_MARLIN_FP8_DEQUANT_BF16 and not _dequant_excluded(layer):
            if self.block_quant:
                self._dequantize_layer_for_cublas(layer)
                return
            logger.warning_once(
                "VLLM_MARLIN_FP8_DEQUANT_BF16 applies only to block-quantized "
                "FP8 layers; falling back to Marlin for this layer."
            )

        layer.input_scale = None
        prepare_fp8_layer_for_marlin(
            layer, self.size_k_first, input_dtype=self.marlin_input_dtype
        )
        del layer.input_scale

    def _dequantize_layer_for_cublas(self, layer: torch.nn.Module) -> None:
        """Dequantize the block-fp8 weight to the model dtype once at load
        and drop the fp8 copy, so apply_weights can run plain cuBLAS.

        On A100 (DSv4-Flash TP=8 shapes) cuBLAS on the bf16 weight beats
        Marlin at every M measured, 1 through 2048 — Marlin's 145 KB
        smem-staging structure buys nothing when weights stream once at
        M=1, and its in-kernel dequant loses at prefill M too. Costs the
        fp8-vs-bf16 byte difference in VRAM; the freed/allocated sizes are
        logged so the trade is visible in the load logs rather than assumed.
        """
        weight = layer.weight
        scale_inv = layer.weight_scale_inv
        n, k = weight.shape
        weight_dq = block_dequant(
            weight, scale_inv.to(torch.float32), layer.weight_block_size
        ).to(layer.orig_dtype)
        freed = weight.nbytes + scale_inv.nbytes
        replace_parameter(layer, "weight", weight_dq)
        del layer.weight_scale_inv
        layer.marlin_fp8_dequant = True
        logger.info(
            "fp8->%s dequant for cuBLAS: prefix=%s shape=(%d, %d) freed %d B, allocated %d B",
            layer.orig_dtype,
            getattr(layer, "prefix", ""),
            n,
            k,
            freed,
            weight_dq.nbytes,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "marlin_fp8_dequant", False):
            if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
                torch.cuda.nvtx.range_push(
                    f"linear:marlin_fp8_dequant_bf16:{getattr(layer, 'prefix', '')}:"
                    f"x={tuple(x.shape)}:w={tuple(layer.weight.shape)}"
                )
                try:
                    return torch.nn.functional.linear(x, layer.weight, bias)
                finally:
                    torch.cuda.nvtx.range_pop()
            return torch.nn.functional.linear(x, layer.weight, bias)
        if self.block_quant:
            weight_scale = layer.weight_scale_inv
        else:
            weight_scale = layer.weight_scale
        if envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
            torch.cuda.nvtx.range_push(
                f"linear:marlin_fp8:{getattr(layer, 'prefix', '')}:"
                f"x={tuple(x.shape)}:w={tuple(layer.weight.shape)}"
            )
            try:
                return apply_fp8_marlin_linear(
                    input=x,
                    weight=layer.weight,
                    weight_scale=weight_scale,
                    workspace=layer.workspace,
                    size_n=layer.output_size_per_partition,
                    size_k=layer.input_size_per_partition,
                    input_dtype=self.marlin_input_dtype,
                    bias=bias,
                )
            finally:
                torch.cuda.nvtx.range_pop()
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            input_dtype=self.marlin_input_dtype,
            bias=bias,
        )

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        pass
