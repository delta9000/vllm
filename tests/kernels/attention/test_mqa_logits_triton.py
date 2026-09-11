# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton FP8 MQA logits (the sparse indexer's DeepGEMM stand-in)."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)

D = 128


def _random_e4m3(*shape: int, gen: torch.Generator) -> torch.Tensor:
    """Finite e4m3 values spanning the whole range (no NaN codes)."""
    x = torch.randn(*shape, device="cuda", generator=gen) * 4
    return x.clamp(-448, 448).to(torch.float8_e4m3fn)


def _logits_reference(q, k, k_scale, weights):
    """[M, H, D] x [N, D] -> [M, N] in fp32."""
    s = torch.einsum("mhd,nd->mhn", q.float(), k.float()).clamp_min(0)
    return (s * weights[:, :, None]).sum(1) * k_scale[None, :]


def _assert_logits_close(out, ref):
    finite = torch.isfinite(ref)
    assert torch.equal(finite, torch.isfinite(out)), "-inf mask differs"
    scale = ref[finite].abs().max().clamp_min(1.0)
    torch.testing.assert_close(
        out[finite] / scale, ref[finite] / scale, rtol=0, atol=2e-5
    )


class _PagedCase:
    """A packed uint8 indexer K cache with a padded block stride.

    Each block holds ``block_size`` rows of D e4m3 value bytes followed by
    ``block_size`` fp32 scales, which is what the CUDA writer and
    ``cp_gather_indexer_k_quant_cache`` agree on.
    """

    def __init__(self, ctx_lens, block_size, pad_bytes=192, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.block_size = block_size
        self.ctx_lens = ctx_lens
        blocks_per_seq = [-(-n // block_size) for n in ctx_lens]
        max_blocks = max(blocks_per_seq)
        num_blocks = sum(blocks_per_seq) + 3
        page = block_size * (D + 4)
        storage = torch.randint(
            0,
            256,
            (num_blocks, page + pad_bytes),
            dtype=torch.uint8,
            device="cuda",
            generator=gen,
        )
        self.cache = storage[:, :page].view(num_blocks, block_size, D + 4)
        assert self.cache.stride(0) == page + pad_bytes
        # Shuffled physical blocks so the block table is actually exercised.
        perm = torch.randperm(num_blocks, device="cuda", generator=gen)
        self.block_table = torch.zeros(
            (len(ctx_lens), max_blocks), dtype=torch.int32, device="cuda"
        )
        self.keys, self.scales = [], []
        used = 0
        for b, n in enumerate(ctx_lens):
            ids = perm[used : used + blocks_per_seq[b]]
            used += blocks_per_seq[b]
            self.block_table[b, : len(ids)] = ids.int()
            k = _random_e4m3(n, D, gen=gen)
            s = torch.rand(n, device="cuda", generator=gen) + 0.5
            pos = torch.arange(n, device="cuda")
            blk, tok = ids[pos // block_size], pos % block_size
            flat = storage.view(-1)
            base = blk.long() * storage.stride(0)
            val_idx = (
                base[:, None] + tok[:, None].long() * D + torch.arange(D, device="cuda")
            )
            flat[val_idx.reshape(-1)] = k.view(torch.uint8).reshape(-1)
            sc_idx = (
                base[:, None]
                + block_size * D
                + tok[:, None].long() * 4
                + (torch.arange(4, device="cuda"))
            )
            flat[sc_idx.reshape(-1)] = s.view(torch.uint8).reshape(-1)
            self.keys.append(k)
            self.scales.append(s)


@pytest.mark.parametrize("block_size", [64, 1])
@pytest.mark.parametrize("next_n,per_token_ctx", [(1, False), (2, False), (3, True)])
@pytest.mark.parametrize("num_heads", [4, 64])
def test_paged_matches_reference(block_size, next_n, per_token_ctx, num_heads):
    ctx_lens = [1, 63, 64, 65, 700] if block_size > 1 else [1, 5, 40]
    case = _PagedCase(ctx_lens, block_size)
    gen = torch.Generator(device="cuda").manual_seed(1)
    batch = len(ctx_lens)
    max_model_len = max(ctx_lens) + 37
    q = _random_e4m3(batch, next_n, num_heads, D, gen=gen)
    weights = torch.randn(batch * next_n, num_heads, device="cuda", generator=gen)
    lens = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    if per_token_ctx:
        # Row j of request b sees L_b - next_n + 1 + j keys (clamped at 0).
        j = torch.arange(next_n, device="cuda", dtype=torch.int32)
        context_lens = (lens[:, None] - next_n + 1 + j).clamp_min(0)
    else:
        context_lens = lens[:, None]

    out = fp8_paged_mqa_logits_triton(
        q,
        case.cache.unsqueeze(-2),
        weights,
        context_lens,
        case.block_table,
        max_model_len,
    )

    ref = torch.full_like(out, float("-inf"))
    for b, n in enumerate(ctx_lens):
        for j in range(next_n):
            row = b * next_n + j
            visible = max(n - next_n + 1 + j, 0)
            if visible == 0:
                continue
            ref[row, :visible] = _logits_reference(
                q[b, j][None],
                case.keys[b][:visible],
                case.scales[b][:visible],
                weights[row][None],
            )[0]
    _assert_logits_close(out, ref)


@pytest.mark.parametrize("num_heads", [4, 64])
def test_nonpaged_matches_reference(num_heads):
    gen = torch.Generator(device="cuda").manual_seed(2)
    M, N = 77, 1000
    q = _random_e4m3(M, num_heads, D, gen=gen)
    k = _random_e4m3(N, D, gen=gen)
    k_scale = torch.rand(N, device="cuda", generator=gen) + 0.5
    weights = torch.randn(M, num_heads, device="cuda", generator=gen)
    ks = torch.randint(0, N // 2, (M,), device="cuda", generator=gen, dtype=torch.int32)
    ke = (ks + torch.randint(1, N // 2, (M,), device="cuda", generator=gen)).int()
    ke[0] = ks[0]  # empty window

    out = fp8_mqa_logits_triton(q, (k, k_scale[:, None]), weights, ks, ke)

    ref = _logits_reference(q, k, k_scale, weights)
    cols = torch.arange(N, device="cuda")
    ref[(cols[None, :] < ks[:, None]) | (cols[None, :] >= ke[:, None])] = float("-inf")
    _assert_logits_close(out, ref)


def test_paged_layout_matches_cuda_gather():
    """The decode reader must agree with the CUDA prefill gather's layout.

    Reads the same cache through ``cp_gather_indexer_k_quant_cache`` and
    checks the decode kernel's logits against keys taken from the gather,
    so a layout drift on either side fails here.
    """
    case = _PagedCase([130, 64, 9], block_size=64)
    total = sum(case.ctx_lens)
    dst_k = torch.empty((total, D), dtype=torch.float8_e4m3fn, device="cuda")
    dst_scale = torch.empty((total, 4), dtype=torch.uint8, device="cuda")
    cu = torch.tensor(
        [0, *torch.tensor(case.ctx_lens).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    ops.cp_gather_indexer_k_quant_cache(
        case.cache, dst_k, dst_scale, case.block_table, cu
    )
    assert torch.equal(dst_k.view(torch.uint8), torch.cat(case.keys).view(torch.uint8))
    assert torch.equal(dst_scale.view(torch.float32)[:, 0], torch.cat(case.scales))

    gen = torch.Generator(device="cuda").manual_seed(3)
    q = _random_e4m3(len(case.ctx_lens), 1, 8, D, gen=gen)
    weights = torch.randn(len(case.ctx_lens), 8, device="cuda", generator=gen)
    lens = torch.tensor(case.ctx_lens, dtype=torch.int32, device="cuda")[:, None]
    out = fp8_paged_mqa_logits_triton(
        q, case.cache.unsqueeze(-2), weights, lens, case.block_table, 256
    )
    for b in range(len(case.ctx_lens)):
        s, e = cu[b].item(), cu[b + 1].item()
        ref = _logits_reference(
            q[b, 0][None],
            dst_k[s:e],
            dst_scale.view(torch.float32)[s:e, 0],
            weights[b][None],
        )[0]
        _assert_logits_close(out[b, : e - s], ref)


def test_paged_is_cuda_graph_capturable():
    case = _PagedCase([300, 20], block_size=64)
    gen = torch.Generator(device="cuda").manual_seed(4)
    q = _random_e4m3(2, 1, 8, D, gen=gen)
    weights = torch.randn(2, 8, device="cuda", generator=gen)
    lens = torch.tensor([[300], [20]], dtype=torch.int32, device="cuda")
    args = (q, case.cache.unsqueeze(-2), weights, lens, case.block_table, 512)
    eager = fp8_paged_mqa_logits_triton(*args)  # also compiles

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp8_paged_mqa_logits_triton(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)

    # Replay must follow new lengths, not the ones seen at capture.
    lens.copy_(torch.tensor([[5], [20]], dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isinf(captured[0, 5:]).all() and torch.isfinite(captured[0, :5]).all()
    assert torch.equal(captured[1], eager[1])
