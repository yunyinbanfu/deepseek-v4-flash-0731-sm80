# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocab-parallel greedy selection must pick the token replication picks.

Everything here runs on CPU with no process group: the collective in
``LogitsProcessor.get_top_tokens`` is simulated by splitting a full-vocab row
into shards and feeding the per-shard results to ``reduce_global_argmax``,
which is where the correctness lives. ``get_shard_logits`` is exercised
against a real ``ParallelLMHead`` built with a monkeypatched TP rank/size, so
the shard geometry is the production one rather than one the test invented.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor,
    reduce_global_argmax,
)

# DeepSeek-V4-Flash / Qwen3-DSpark draft vocab is 129,280; the small sizes
# exercise uneven splits, which is where an offset bug would hide.
VOCAB_SIZES = [129280, 4096, 1000, 17]
TP_SIZES = [2, 4, 8]


def _shard(logits: torch.Tensor, tp: int):
    """Split [B, V] into tp shards, returning [B, tp] maxima and global ids."""
    v = logits.shape[-1]
    per = (v + tp - 1) // tp
    values, ids = [], []
    for r in range(tp):
        lo, hi = r * per, min((r + 1) * per, v)
        if lo >= hi:
            # An empty trailing shard must never win; -inf is what a real rank
            # with no columns would contribute.
            values.append(torch.full((logits.shape[0],), float("-inf")))
            ids.append(torch.zeros(logits.shape[0], dtype=torch.long))
            continue
        val, idx = logits[:, lo:hi].max(dim=-1)
        values.append(val)
        ids.append(idx + lo)
    return torch.stack(values, dim=-1), torch.stack(ids, dim=-1)


@pytest.mark.cpu_test
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
@pytest.mark.parametrize("tp", TP_SIZES)
@pytest.mark.parametrize("batch", [1, 5])
def test_sharded_matches_replicated_argmax(vocab_size: int, tp: int, batch: int):
    torch.manual_seed(vocab_size * tp + batch)
    logits = torch.randn(batch, vocab_size, dtype=torch.float32)

    expected = logits.argmax(dim=-1)
    got = reduce_global_argmax(*_shard(logits, tp))

    # Token-for-token equality, not closeness: this selects a token id.
    assert torch.equal(got, expected), (
        f"sharded selection diverged at V={vocab_size} tp={tp}: "
        f"{got.tolist()} vs {expected.tolist()}"
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("tp", TP_SIZES)
def test_ties_resolve_to_lowest_global_id(tp: int):
    """Exact ties are the one case torch.argmax leaves unspecified, so the
    reduction defines the rule and this pins it."""
    vocab_size = 4096
    logits = torch.zeros(1, vocab_size, dtype=torch.float32)
    # Same maximum in several shards at once.
    per = vocab_size // tp
    tied = [r * per + 3 for r in range(tp)]
    for i in tied:
        logits[0, i] = 9.0

    got = reduce_global_argmax(*_shard(logits, tp))
    assert got.item() == min(tied)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tp", TP_SIZES)
def test_shard_order_does_not_matter(tp: int):
    """Ranks may deliver shards in any order; ids travel with their values."""
    torch.manual_seed(0)
    logits = torch.randn(3, 4096, dtype=torch.float32)
    values, ids = _shard(logits, tp)

    perm = torch.randperm(tp)
    shuffled = reduce_global_argmax(values[:, perm], ids[:, perm])
    assert torch.equal(shuffled, logits.argmax(dim=-1))


@pytest.mark.cpu_test
@pytest.mark.parametrize("poison", [float("nan"), float("-inf")])
def test_degenerate_rows_still_return_an_in_range_id(poison: float):
    """No row may select an id no shard owns.

    A NaN row makes ``values == best`` false in every lane; a sentinel there
    would travel into the caller's embedding lookup and surface as an async
    device-side assert (observed, on 8 ranks, before the reduction was made
    total). An all -inf row is the same shape of degeneracy from the other
    side.
    """
    tp, batch = 8, 3
    values = torch.full((batch, tp), poison)
    ids = torch.arange(tp).expand(batch, tp) * 1000

    got = reduce_global_argmax(values, ids)
    assert ((got >= 0) & (got <= ids.max())).all(), got.tolist()

    # A single live lane wins outright -- except against NaN, which propagates
    # through max() and poisons the row for a replicated argmax just the same.
    values[1, 5] = 1.0
    winner = reduce_global_argmax(values, ids)[1].item()
    assert winner == (5000 if poison == float("-inf") else ids.max().item())


@pytest.mark.cpu_test
def test_empty_trailing_shard_never_wins():
    """A rank holding no columns contributes -inf and must be ignored."""
    logits = torch.randn(2, 17, dtype=torch.float32)
    got = reduce_global_argmax(*_shard(logits, 8))
    assert torch.equal(got, logits.argmax(dim=-1))


def _make_head(monkeypatch, vocab_size: int, dim: int, rank: int, tp: int):
    """A real ParallelLMHead for `rank` of `tp`, without a process group."""
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: tp
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )
    return ParallelLMHead(vocab_size, dim, bias=False, params_dtype=torch.float32)


@pytest.mark.cpu_test
def test_shard_logits_masks_zero_filled_padding(monkeypatch: pytest.MonkeyPatch):
    """The padded tail of a shard is zero-filled by the weight loader, so it
    scores exactly 0 and wins any row whose real logits are all negative.

    V=1000 over tp=8 pads to 1024, i.e. 128 columns per rank and 24 padded
    columns on rank 7 -- the only rank that has any.
    """
    vocab_size, dim, tp, rank = 1000, 4, 8, 7
    head = _make_head(monkeypatch, vocab_size, dim, rank, tp)
    assert head.shard_indices.num_org_vocab_padding == 24

    # Real rows: strictly negative logits for a positive input.
    torch.manual_seed(0)
    head.weight.data.fill_(0.0)
    head.weight.data[: head.num_org_embeddings_per_partition] = -torch.rand(
        head.num_org_embeddings_per_partition, dim
    )
    hidden = torch.ones(2, dim)

    lp = LogitsProcessor(vocab_size)
    unmasked = lp._apply_head(head, hidden, None)
    assert unmasked.argmax(dim=-1).min().item() >= head.num_org_embeddings_per_partition

    masked = lp.get_shard_logits(head, hidden)
    assert torch.isneginf(masked[:, -24:]).all()
    # With the mask the shard-local argmax lands on a real vocab row, so the
    # id it contributes to the reduction is in range.
    local = masked.argmax(dim=-1) + head.shard_indices.org_vocab_start_index
    assert (local < vocab_size).all()


@pytest.mark.cpu_test
def test_shard_logits_extra_logits_is_summed_before_selection(
    monkeypatch: pytest.MonkeyPatch,
):
    """`extra_logits` must join the argmax, not be applied afterwards."""
    vocab_size, dim = 64, 4
    head = _make_head(monkeypatch, vocab_size, dim, rank=0, tp=1)
    torch.manual_seed(0)
    head.weight.data.normal_()
    hidden = torch.randn(3, dim)

    lp = LogitsProcessor(vocab_size)
    base = lp.get_shard_logits(head, hidden)
    extra = torch.randn(3, vocab_size)

    torch.testing.assert_close(
        lp.get_shard_logits(head, hidden, None, extra), base + extra
    )
    assert torch.equal(
        lp.get_top_tokens(head, hidden, None, extra), (base + extra).argmax(dim=-1)
    )
