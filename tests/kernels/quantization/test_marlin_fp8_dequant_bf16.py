# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MARLIN_FP8_DEQUANT_BF16: load-time dequant of block-fp8 Marlin
layers to the model dtype (cuBLAS route).

CPU-only on purpose: the dequant path is plain torch, and the end-to-end
flag-on server A/B is what validates it on GPU.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)

QB = 128


def _make_layer(n: int, k: int, scale_dtype: torch.dtype) -> torch.nn.Module:
    torch.manual_seed(0)
    w = torch.randn(n, k, dtype=torch.float32) * 0.02
    wv = w.view(n // QB, QB, k // QB, QB)
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    if scale_dtype == torch.float8_e8m0fnu:
        # ue8m0 scales are power-of-two; quantize amax/448 up to one.
        scales = (amax / 448.0).log2().ceil().exp2()
    else:
        scales = amax / 448.0
    w_fp8 = (wv / scales).clamp(-448.0, 448.0).view(n, k).to(torch.float8_e4m3fn)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_fp8, requires_grad=False)
    layer.weight_scale_inv = torch.nn.Parameter(
        scales.view(n // QB, k // QB).to(scale_dtype), requires_grad=False
    )
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = [QB, QB]
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    return layer


@pytest.mark.parametrize("n,k", [(1536, 4096), (4096, 256)])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e8m0fnu])
def test_dequant_matches_reference_and_frees_fp8(n, k, scale_dtype):
    layer = _make_layer(n, k, scale_dtype)
    scale_full = (
        layer.weight_scale_inv.to(torch.float32)
        .repeat_interleave(QB, 0)
        .repeat_interleave(QB, 1)
    )
    ref = (layer.weight.to(torch.float32) * scale_full).to(torch.bfloat16)

    kernel = object.__new__(MarlinFP8ScaledMMLinearKernel)  # dequant needs no init
    kernel._dequantize_layer_for_cublas(layer)

    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.shape == (n, k)
    torch.testing.assert_close(layer.weight, ref, rtol=0, atol=0)
    # The fp8 copy and its scales must actually be gone -- the VRAM trade
    # (+1x fp8 bytes, not +2x) depends on it.
    assert not hasattr(layer, "weight_scale_inv")
    assert layer.marlin_fp8_dequant

    # apply_weights must route to plain cuBLAS/F.linear.
    kernel.block_quant = True
    x = torch.randn(4, k, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x)
    torch.testing.assert_close(out, torch.nn.functional.linear(x, layer.weight))
