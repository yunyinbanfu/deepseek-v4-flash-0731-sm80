# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.linear.gemv_triton import (
    MAX_GEMV_TOKENS,
    bf16_gemv,
    should_use_triton_gemv,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# DeepSeek-V4-Flash shapes from config.json: (hidden_size, index_n_heads) for
# the indexer's weights_proj and (hidden_size, n_routed_experts) for the MoE
# gate. 4096x1 covers a degenerate N.
SHAPES = [(4096, 64), (4096, 256), (4096, 1), (2048, 128)]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 5, 8])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_bf16_gemv_matches_reference(k: int, n: int, m: int, out_dtype) -> None:
    set_random_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02

    got = bf16_gemv(x, w, out_dtype)
    assert got.shape == (m, n)
    assert got.dtype == out_dtype

    # fp32 reference: the kernel accumulates in fp32, so an fp32 output should
    # track it far more tightly than a bf16 one, which is the point of the
    # tier for the MoE gate.
    exact = x.float() @ w.float().T
    tol = 5e-3 if out_dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(got.float(), exact, atol=tol, rtol=tol)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@torch.inference_mode()
def test_should_use_triton_gemv_gate() -> None:
    w = torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda")

    ok = torch.randn(MAX_GEMV_TOKENS, 4096, dtype=torch.bfloat16, device="cuda")
    assert should_use_triton_gemv(ok, w)

    # Past the measured crossover cuBLAS wins, so the gate must decline.
    too_many = torch.randn(
        MAX_GEMV_TOKENS + 1, 4096, dtype=torch.bfloat16, device="cuda"
    )
    assert not should_use_triton_gemv(too_many, w)

    # fp16/fp32 activations and non-contiguous inputs are not handled.
    assert not should_use_triton_gemv(ok.float(), w.float())
    assert not should_use_triton_gemv(ok.T.contiguous().T[:, :2048], w)
