# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.utils import KVBlockZeroer, _zero_kv_blocks_kernel


def _zeroer_for(
    storages: list[torch.Tensor],
    *,
    strides: list[int] | None = None,
    extents: list[int] | None = None,
    ratios: list[int] | None = None,
    group: int = 0,
) -> KVBlockZeroer:
    """Minimal zeroer state for contiguous [num_blocks, page] test storages.

    Built directly so tests can focus on kernel behavior without constructing
    model attention groups. Defaults describe the dense case: addressing
    stride == logical extent, no virtual block splitting.
    """
    device = storages[0].device
    pages = [s.shape[-1] for s in storages]
    meta = KVBlockZeroer.build_meta(
        [s.data_ptr() for s in storages],
        strides or pages,
        extents or pages,
        ratios or [1] * len(storages),
        device,
    )
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._group_meta = {} if meta is None else {group: meta}
    return zeroer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)
    zeroer = _zeroer_for([storage])

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([[1]])
        zeroer.zero_block_ids([[2]])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    storage_a = torch.ones((num_blocks, 10496), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, 2112), dtype=torch.int32, device=device)
    zeroer = _zeroer_for([storage_a, storage_b])

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([[1, 2]])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_interleaved_layer_views_zero_only_their_own_bytes():
    """THE STRIPE REGRESSION GUARD (#50576).

    DSv4's per-layer caches can be interleaved views over one pool: the block
    stride spans every layer, so a kernel that derives the zeroed EXTENT from
    the stride wipes a whole pool stripe -- including the head of neighboring
    live blocks. Model the pool as [num_blocks, num_layers, page]: each layer
    view has stride num_layers * page but owns only page elements per block.
    """
    device = torch.device("cuda")
    num_blocks, num_layers, page = 4, 3, 64
    pool = torch.ones(
        (num_blocks, num_layers, page), dtype=torch.int32, device=device
    )
    stride = num_layers * page
    # One segment per layer view, addressed from the layer's first block.
    meta = KVBlockZeroer.build_meta(
        [pool.data_ptr() + layer * page * 4 for layer in (0, 2)],
        [stride, stride],
        [page, page],
        [1, 1],
        device,
    )
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._group_meta = {0: meta}

    zeroer.zero_block_ids([[1]])
    torch.cuda.synchronize()

    assert torch.all(pool[0] == 1) and torch.all(pool[2:] == 1)
    assert torch.all(pool[1, 0] == 0), "layer 0's block 1 must be zeroed"
    assert torch.all(pool[1, 2] == 0), "layer 2's block 1 must be zeroed"
    # The stripe bug would have wiped this live neighboring layer too.
    assert torch.all(pool[1, 1] == 1), "layer 1 is NOT registered and must survive"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_group_scoped():
    """THE CROSS-GROUP REGRESSION GUARD (#50576).

    Block ids are only meaningful within their own kv-cache group: with
    virtual block splitting the same id maps to different pages in groups
    with different geometry. Zeroing group 0's new block must not touch
    group 1's identically-numbered live block.
    """
    device = torch.device("cuda")
    storage_a = torch.ones((4, 128), dtype=torch.int32, device=device)
    storage_b = torch.ones((4, 96), dtype=torch.int32, device=device)

    meta_a = KVBlockZeroer.build_meta(
        [storage_a.data_ptr()], [128], [128], [1], device
    )
    meta_b = KVBlockZeroer.build_meta([storage_b.data_ptr()], [96], [96], [1], device)
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._group_meta = {0: meta_a, 1: meta_b}

    zeroer.zero_block_ids([[1], [3]])
    torch.cuda.synchronize()

    assert torch.all(storage_a[1] == 0) and torch.all(storage_b[3] == 0)
    # The flat-list bug applied every id to every group.
    assert torch.all(storage_a[3] == 1), "group 0 must not zero group 1's id"
    assert torch.all(storage_b[1] == 1), "group 1 must not zero group 0's id"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_virtual_block_split_zeroes_every_sub_block():
    """ratio > 1: one logical block spans ratio kernel blocks, each at its own
    stride offset, and each zeroed only over its logical extent."""
    device = torch.device("cuda")
    num_kernel_blocks, page = 8, 48
    stride = 2 * page  # interleaved with a neighbor view that must survive
    pool = torch.ones((num_kernel_blocks, 2, page), dtype=torch.int32, device=device)
    meta = KVBlockZeroer.build_meta(
        [pool.data_ptr()], [stride], [page], [2], device
    )
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._group_meta = {0: meta}

    # Logical block 1 = kernel blocks 2 and 3.
    zeroer.zero_block_ids([[1]])
    torch.cuda.synchronize()

    assert torch.all(pool[:2, 0] == 1) and torch.all(pool[4:, 0] == 1)
    assert torch.all(pool[2, 0] == 0) and torch.all(pool[3, 0] == 0)
    assert torch.all(pool[:, 1] == 1), "the interleaved neighbor must survive"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_every_n_blocks_specialization():
    """After warmup, no launch should trigger a first-request JIT compile.

    ``n_blocks`` is ``do_not_specialize``, so a single warmup launch must
    cover every block count.
    """
    device = torch.device("cuda")
    num_blocks = 64
    storage = torch.ones((num_blocks, 4), dtype=torch.int32, device=device)
    zeroer = _zeroer_for([storage])

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids([list(range(n_blocks))])
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    storage = torch.ones((1, 4), dtype=torch.int32, device=device)
    zeroer = _zeroer_for([storage])

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pages_with_no_large_common_divisor_are_fully_zeroed():
    """Page sizes whose only common divisor is small must still zero fully.

    DeepSeek-V4 mixes a 9344-element MLA page, a 2112-element indexer page and
    an 8192-element compressor page. Neither 9344 nor 2112 is a multiple of
    the 1024-element chunk this kernel uses -- both leave a tail that only the
    store mask covers.
    """
    device = torch.device("cuda")
    num_blocks = 3
    page_sizes = [9344, 2112, 8192]
    storages = [
        torch.ones((num_blocks, ps), dtype=torch.int32, device=device)
        for ps in page_sizes
    ]
    zeroer = _zeroer_for(storages)

    zeroer.zero_block_ids([[1]])
    torch.accelerator.synchronize()

    for storage, ps in zip(storages, page_sizes):
        assert torch.all(storage[0] == 1), "block 0 must be untouched"
        assert torch.all(storage[2] == 1), "block 2 must be untouched"
        # The whole page, tail included -- a truncating chunk map would leave
        # the last ps % 1024 elements set.
        assert torch.all(storage[1] == 0), f"page {ps} not fully zeroed"


def test_every_launched_program_has_work():
    """No program may be launched only to exit empty.

    The chunk list is flattened per (segment, sub-block) rather than sized to
    the largest page, so the program count is the sum over segments of
    ``ratio * cdiv(extent, CHUNK_ELEMS)`` -- and no chunk crosses a sub-block
    boundary.
    """
    page_sizes = [9344, 2112, 8192, 4]
    ratios = [1, 2, 1, 4]
    meta = KVBlockZeroer.build_meta(
        [0] * len(page_sizes),
        [ps * 2 for ps in page_sizes],  # interleaved: stride != extent
        page_sizes,
        ratios,
        torch.device("cpu"),
    )
    assert meta is not None
    _, seg_periods, chunk_seg, chunk_base, chunk_len, n_chunks = meta
    chunk_elems = KVBlockZeroer.CHUNK_ELEMS

    expected = sum(
        r * ((ps + chunk_elems - 1) // chunk_elems)
        for ps, r in zip(page_sizes, ratios)
    )
    assert n_chunks == expected
    assert chunk_seg.numel() == n_chunks
    assert seg_periods.tolist() == [ps * 2 * r for ps, r in zip(page_sizes, ratios)]

    # Chunks tile each (segment, sub-block) extent exactly once, never
    # crossing into the interleaved gap between sub-blocks.
    covered: dict[tuple[int, int], set[int]] = {}
    for seg, base, length in zip(
        chunk_seg.tolist(), chunk_base.tolist(), chunk_len.tolist()
    ):
        stride = page_sizes[seg] * 2
        sub_block, within = divmod(base, stride)
        assert sub_block < ratios[seg]
        assert within + length <= page_sizes[seg], "chunk leaks past its extent"
        rows = covered.setdefault((seg, sub_block), set())
        assert not rows & set(range(within, within + length)), "overlap"
        rows.update(range(within, within + length))
    for (seg, _), rows in covered.items():
        assert rows == set(range(page_sizes[seg]))
    assert len(covered) == sum(ratios)
