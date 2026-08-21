# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""block_size_m selection for the Marlin expert GEMM.

The selection used to be four lines inline in `fused_marlin_moe`, so nothing
could test it. It now has to hold two properties at once:

  * without an expert histogram it must reproduce the historical ladder for
    every shape, bit for bit -- that is what keeps every model other than the
    one measured, and every cudagraph-captured decode shape, unchanged;
  * with a histogram it must pick the rung that runs fewer padded MMA rows,
    which is the quantity the measurements showed the kernel's time tracks
    between the 48 and 64 rungs.

The reference ladder below is a copy of the pre-change code, deliberately not
importing the implementation, so a drift in either shows up here.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    ADAPTIVE_BLOCK_SIZE_M_CANDIDATES,
    ADAPTIVE_BLOCK_SIZE_M_MIN_TOKENS,
    moe_padded_rows,
    select_block_size_m,
)
from vllm.platforms import current_platform


def reference_ladder(m: int, topk: int, num_experts: int) -> int:
    """Verbatim copy of the selection as it stood before the refinement."""
    for block_size_m in [8, 16, 32, 48, 64]:
        if m * topk / num_experts / block_size_m < 0.9:
            break
    return block_size_m


def reference_padded_rows(counts: list[int], b: int) -> int:
    return sum(-(-c // b) for c in counts) * b


# (m, topk, num_experts) -> rung, walking each ladder boundary from below and
# above. E=256/topk=6 is DeepSeek-V4-Flash at TP=8; E=8/topk=2 is Mixtral-like.
LADDER_CASES = [
    (1, 6, 256, 8),
    (6, 6, 256, 8),
    (307, 6, 256, 8),  # avg 7.19, just under the 8 rung's 0.9
    (308, 6, 256, 16),  # avg 7.22, first M over the 8 rung
    (315, 6, 256, 16),  # avg 7.38, over the 8 rung
    (614, 6, 256, 16),  # avg 14.39
    (615, 6, 256, 32),  # avg 14.41
    (1228, 6, 256, 32),  # avg 28.78
    (1229, 6, 256, 48),  # avg 28.80
    (1843, 6, 256, 48),  # avg 43.19
    (1844, 6, 256, 64),  # avg 43.22
    (2048, 6, 256, 64),  # avg 48.0 exactly -- the measured shape
    (8192, 6, 256, 64),
    (1, 2, 8, 8),
    (32, 2, 8, 16),
    (64, 2, 8, 32),
    (128, 2, 8, 48),
    (256, 2, 8, 64),
    (4096, 2, 8, 64),
]


@pytest.mark.parametrize("m,topk,num_experts,expected", LADDER_CASES)
def test_ladder_unchanged_without_histogram(m, topk, num_experts, expected):
    assert select_block_size_m(m, topk, num_experts) == expected
    assert reference_ladder(m, topk, num_experts) == expected


@pytest.mark.parametrize("num_experts", [8, 64, 128, 256])
@pytest.mark.parametrize("topk", [1, 2, 4, 6, 8])
def test_ladder_unchanged_over_a_dense_m_grid(num_experts, topk):
    """No histogram => historical behaviour for every shape, not just the ones
    that happen to be parametrized above."""
    for m in list(range(1, 600)) + [1024, 2048, 4096, 8192, 16384, 65536]:
        assert select_block_size_m(m, topk, num_experts) == reference_ladder(
            m, topk, num_experts
        )


def test_dsv4_prefill_shape_takes_48_when_rows_favour_it():
    """M=2048, E=256, topk=6: the ladder returns 64 while the average expert
    holds exactly 48 tokens. A perfectly balanced load runs 49152 rows at 48
    against 65536 at 64, so 48 must win."""
    counts = [48] * 256
    rows = [reference_padded_rows(counts, b) for b in ADAPTIVE_BLOCK_SIZE_M_CANDIDATES]
    assert rows == [12288, 16384]
    assert select_block_size_m(2048, 6, 256) == 64
    assert select_block_size_m(2048, 6, 256, rows) == 48


def test_dsv4_prefill_shape_keeps_64_when_rows_do_not():
    """The distribution that made a blind swap unsafe: a uniformly random
    routing at the same shape spreads counts either side of 48, so nearly every
    expert needs a second 48-block and 64 stays cheaper."""
    counts = [40, 56] * 128  # straddles 48, as a random draw does
    rows = [reference_padded_rows(counts, b) for b in ADAPTIVE_BLOCK_SIZE_M_CANDIDATES]
    assert rows[0] > rows[1]
    assert select_block_size_m(2048, 6, 256, rows) == 64


@pytest.mark.parametrize("m", [1, 2, 6, 8, 16, 64, 256, 511])
def test_decode_shapes_are_never_refined(m):
    """Decode runs inside a cudagraph, where block_size_m has to be a
    capture-time constant. Even handed a histogram that would favour 48, the
    selection must return the ladder's rung."""
    rows = [1, 10**9]  # would select 48 if the refinement were reachable
    assert select_block_size_m(m, 6, 256, rows) == reference_ladder(m, 6, 256)
    assert select_block_size_m(m, 6, 256, rows) < 48


def test_min_token_gate_is_the_only_reason_a_prefill_shape_is_skipped():
    m = ADAPTIVE_BLOCK_SIZE_M_MIN_TOKENS
    assert select_block_size_m(m - 1, 6, 8, [1, 10**9]) == reference_ladder(m - 1, 6, 8)
    assert select_block_size_m(m, 6, 8, [1, 10**9]) == 48


@pytest.mark.parametrize("m,topk,num_experts", [(1229, 6, 256), (615, 6, 256)])
def test_lower_rungs_are_not_refined(m, topk, num_experts):
    """Only the top rung is refined; the measurements do not cover swaps below
    48, where per-block cost stops tracking the block size."""
    assert reference_ladder(m, topk, num_experts) != 64
    rows = [1, 10**9]
    assert select_block_size_m(m, topk, num_experts, rows) == reference_ladder(
        m, topk, num_experts
    )


def test_ties_keep_the_larger_rung():
    """Equal row counts must not churn the rung: 64 runs fewer, larger blocks
    for the same rows, so it is the conservative side of a tie. 192 tokens per
    expert is 4 exact blocks of 48 and 3 exact blocks of 64."""
    counts = [192] * 256
    rows = [reference_padded_rows(counts, b) for b in ADAPTIVE_BLOCK_SIZE_M_CANDIDATES]
    assert rows[0] == rows[1] == 49152
    assert select_block_size_m(8192, 6, 256, rows) == 64


@pytest.mark.parametrize("num_experts", [8, 256])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_moe_padded_rows_matches_reference(num_experts, seed):
    g = torch.Generator().manual_seed(seed)
    topk_ids = torch.randint(0, num_experts, (2048, 6), generator=g)
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts).tolist()
    rows = moe_padded_rows(topk_ids, num_experts, (8, 16, 32, 48, 64)).tolist()
    assert rows == [reference_padded_rows(counts, b) for b in (8, 16, 32, 48, 64)]


def test_moe_padded_rows_ignores_unrouted_slots():
    """-1 marks a slot no expert will run; the alignment skips it and so must
    the row count."""
    topk_ids = torch.tensor([[0, 1, -1], [0, -1, -1], [2, 2, 1]])
    rows = moe_padded_rows(topk_ids, 4, (8,)).tolist()
    # experts 0,1,2 are touched with 2/2/2 tokens => 3 blocks of 8.
    assert rows == [24]


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(),
    reason="needs CUDA",
)
@pytest.mark.parametrize("block_size", [16, 32, 48, 64])
def test_moe_padded_rows_matches_moe_align_block_size(block_size):
    """The strong check: the row count must equal what the alignment kernel
    itself reports as num_tokens_post_padded, since that is the number the GEMM
    grid is sized from."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    num_experts = 256
    g = torch.Generator(device="cuda").manual_seed(0)
    topk_ids = torch.randint(
        0, num_experts, (2048, 6), generator=g, device="cuda", dtype=torch.int32
    )
    _, _, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, num_experts, None, ignore_invalid_experts=True
    )
    rows = moe_padded_rows(topk_ids, num_experts, (block_size,))
    assert rows.item() == num_tokens_post_padded.item()
