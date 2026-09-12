# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-CTA Triton ``moe_align_block_size`` for decode-sized batches.

The CUDA op it replaces issues three fills, two cub scans and the alignment
kernel per MoE layer; this path must produce the same ``sorted_token_ids``,
``expert_ids`` and ``num_tokens_post_pad`` (golden torch reference from
test_moe_align_block_size, and the CUDA op itself when available), with and
without an expert-parallel ``expert_map``.
"""

import os

import pytest
import torch

from vllm.model_executor.layers.fused_moe import moe_align_block_size as mab
from vllm.platforms import current_platform

from .test_moe_align_block_size import torch_moe_align_block_size


@pytest.fixture(autouse=True)
def small_align_on():
    """The path is off by default (a wash on A40); these tests exercise it."""
    prev = os.environ.get("VLLM_MOE_ALIGN_SMALL")
    os.environ["VLLM_MOE_ALIGN_SMALL"] = "1"
    try:
        yield
    finally:
        if prev is None:
            del os.environ["VLLM_MOE_ALIGN_SMALL"]
        else:
            os.environ["VLLM_MOE_ALIGN_SMALL"] = prev

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton moe_align is CUDA-only"
)


def _same_alignment(got, ref, block_size: int) -> bool:
    """Equal outputs, allowing any order inside one expert's block group."""
    g_sorted, g_expert, g_post = got
    r_sorted, r_expert, r_post = ref
    if g_post.item() != r_post.item() or not torch.equal(g_expert, r_expert):
        return False
    n = int(g_post.item())
    if not torch.equal(g_sorted[n:], r_sorted[n:]):
        return False
    return all(
        torch.equal(
            torch.sort(g_sorted[i : i + block_size])[0],
            torch.sort(r_sorted[i : i + block_size])[0],
        )
        for i in range(0, n, block_size)
    )


@pytest.mark.parametrize("num_tokens", [1, 3, 6, 16])
@pytest.mark.parametrize("topk,num_experts", [(6, 384), (3, 128), (8, 256)])
@pytest.mark.parametrize("block_size", [8, 16, 64])
@pytest.mark.parametrize("use_expert_map", [False, True])
@torch.inference_mode()
def test_small_align_matches_reference(
    num_tokens, topk, num_experts, block_size, use_expert_map
):
    torch.manual_seed(num_tokens * 7 + topk)
    topk_ids = (
        torch.stack([torch.randperm(num_experts)[:topk] for _ in range(num_tokens)])
        .to(torch.int32)
        .cuda()
    )
    expert_map = None
    if use_expert_map:
        n_local = num_experts // 8
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[n_local : 2 * n_local] = torch.arange(
            n_local, dtype=torch.int32, device="cuda"
        )
    assert mab._use_small_align(topk_ids, num_experts, expert_map, True)

    got = mab.moe_align_block_size(
        topk_ids, block_size, num_experts, expert_map, ignore_invalid_experts=True
    )
    ref = torch_moe_align_block_size(topk_ids, block_size, num_experts, expert_map)
    assert _same_alignment(got, ref, block_size)

    # The CUDA op on the same inputs (the path used above this batch size).
    sorted_ids = torch.empty_like(got[0])
    expert_ids = torch.empty_like(got[1])
    post = torch.empty_like(got[2])
    torch.ops._moe_C.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, post, expert_map
    )
    assert got[2].item() == post.item()
    n = int(post.item())
    assert torch.equal(got[1][: n // block_size], expert_ids[: n // block_size])
    assert _same_alignment(
        (got[0][:n], got[1][: n // block_size], got[2]),
        (sorted_ids[:n], expert_ids[: n // block_size], post),
        block_size,
    )


@torch.inference_mode()
def test_small_align_graph_replay_and_large_fallback():
    torch.manual_seed(0)
    topk_ids = torch.randint(0, 384, (6, 6), dtype=torch.int32, device="cuda")
    ref = torch_moe_align_block_size(topk_ids, 8, 384, None)
    static = topk_ids.clone()
    out = mab.moe_align_block_size(static, 8, 384, None, ignore_invalid_experts=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mab.moe_align_block_size(
            static, 8, 384, None, ignore_invalid_experts=True
        )
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert _same_alignment(out, ref, 8)
    # Above the small-path bound the CUDA op is used unchanged.
    big = torch.randint(0, 384, (64, 6), dtype=torch.int32, device="cuda")
    assert not mab._use_small_align(big, 384, None, True)
    got = mab.moe_align_block_size(big, 8, 384, None, ignore_invalid_experts=True)
    assert got[2].item() == torch_moe_align_block_size(big, 8, 384, None)[2].item()
