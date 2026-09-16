# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate-only MQA logits for the V4.1 two-level indexer, for SM < 9.0.

The dense path scores every KV position, masks all but the candidate blocks,
then runs top-k over the full ``[rows, max_model_len]`` row. For V4.1-Flash
the candidates are ``candidate_topk_blocks * candidate_block_size`` = 2048 * 8
= 16384 positions, so at a 512k context 97% of that work is thrown away.

This is the SM 8.x counterpart to upstream's DeepGEMM sparse path (#56254):
same restructuring -- score the candidates, top-k in the compressed column
space, remap back to request-local positions -- but with a portable Triton
kernel over the FP8 indexer cache instead of ``fp8_fp4_sparse_mqa_logits``
(SM100 + DeepGEMM >= 2.8) and DeepSelect.

Gathers dominate this kernel, so it wants far more warps than the dense one
(which streams contiguous KV): measured on A40 at a 512k context, 8 warps over
128 columns beats the dense kernel's 2/64 by ~5x at 8 rows. It also wants the
candidate block ids sorted ascending -- see ``sort_candidate_blocks``.
"""

import torch

from vllm.model_executor.kernels.attention.dsa.mqa_logits_triton import (
    _paged_cache_block_stride,
)
from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8_e4m3 import e4m3fn_bits_to_fp32, needs_e4m3fn_emulation

# Gather-bound: more warps hide the scattered KV loads. See module docstring.
_SPARSE_BLOCK_N = 128
_SPARSE_NUM_WARPS = 8


@triton.jit
def _load_e4m3_as_f16(ptr, mask, EMULATE_E4M3: tl.constexpr):
    if EMULATE_E4M3:
        return e4m3fn_bits_to_fp32(tl.load(ptr, mask=mask, other=0)).to(tl.float16)
    return tl.load(ptr, mask=mask, other=0.0).to(tl.float16)


@triton.jit
def _sparse_paged_mqa_logits_kernel(
    q_ptr,  # [B, NEXT_N, H, D] e4m3 (uint8 bits when emulating)
    kv_ptr,  # uint8 base of the paged cache
    w_ptr,  # [rows, H] fp32, Q scale folded in
    ctx_ptr,  # [B, 1] or [B, NEXT_N] int32
    bt_ptr,  # [B, max_blocks] int32
    cand_ptr,  # [rows, K] int32 request-local candidate block ids (-1 pad)
    out_ptr,  # [rows, K * CBS] fp32
    ctx_stride_b,
    ctx_stride_j,
    bt_stride,
    cand_stride,
    kv_block_stride,
    out_stride,
    num_sparse_cols,
    NEXT_N: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # paged cache block size (tokens)
    CBS: tl.constexpr,  # candidate block size (tokens)
    BLOCK_N: tl.constexpr,  # sparse columns per program
    CTX_PER_TOKEN: tl.constexpr,
    EMULATE_E4M3: tl.constexpr,
):
    """Score only the candidate blocks.

    Sparse column ``c`` holds the logit for request-local position
    ``cand[row, c // CBS] * CBS + c % CBS``. Columns whose candidate slot is
    -1, or whose position is past the row's visible context, store -inf so a
    plain top-k over the compressed row skips them.
    """
    row = tl.program_id(0)
    n0 = tl.program_id(1) * BLOCK_N
    b = row // NEXT_N
    j = row % NEXT_N

    offs_c = n0 + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + row.to(tl.int64) * out_stride + offs_c
    out_mask = offs_c < num_sparse_cols

    # Keys visible to query row (b, j) -- same convention as the dense kernel.
    if CTX_PER_TOKEN:
        lim = tl.load(ctx_ptr + b * ctx_stride_b + j * ctx_stride_j)
    else:
        lim = tl.load(ctx_ptr + b * ctx_stride_b) - NEXT_N + 1 + j
    lim = tl.maximum(lim, 0)

    slot = offs_c // CBS
    cand = tl.load(
        cand_ptr + row.to(tl.int64) * cand_stride + slot, mask=out_mask, other=-1
    )
    pos = cand * CBS + (offs_c % CBS)
    valid = out_mask & (cand >= 0) & (pos < lim)

    offs_h = tl.arange(0, HP)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H
    # Q tile [HP, D]; padded heads are zero and carry zero weight.
    q = _load_e4m3_as_f16(
        q_ptr + (row.to(tl.int64) * H + offs_h[:, None]) * D + offs_d[None, :],
        h_mask[:, None],
        EMULATE_E4M3,
    )
    w = tl.load(w_ptr + row.to(tl.int64) * H + offs_h, mask=h_mask, other=0.0)

    bt_row = bt_ptr + b.to(tl.int64) * bt_stride
    safe_pos = tl.where(valid, pos, 0)
    pages = tl.load(bt_row + safe_pos // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
    tok = safe_pos % BLOCK_SIZE

    # A block is BLOCK_SIZE rows of D value bytes, then BLOCK_SIZE fp32 scales.
    block_base = kv_ptr + pages * kv_block_stride
    k = _load_e4m3_as_f16(
        block_base[:, None] + tok[:, None] * D + offs_d[None, :],
        valid[:, None],
        EMULATE_E4M3,
    )
    scale_ptrs = (block_base + BLOCK_SIZE * D + tok * 4).to(tl.pointer_type(tl.float32))
    scale = tl.load(scale_ptrs, mask=valid, other=0.0)

    s = tl.dot(k, tl.trans(q))  # [BLOCK_N, HP], fp32 accumulate
    s = tl.sum(tl.maximum(s, 0.0) * w[None, :], axis=1) * scale
    tl.store(out_ptrs, tl.where(valid, s, float("-inf")), out_mask)


@triton.jit
def _remap_kernel(
    col_ptr,  # [rows, topk] int32 sparse-column indices
    col_stride,
    cand_ptr,  # [rows, K] int32
    cand_stride,
    out_ptr,  # [rows, topk] int32 request-local positions
    out_stride,
    k,
    CBS: tl.constexpr,
    K_POW2: tl.constexpr,
):
    """Map compressed columns back to request-local positions; -1 stays -1."""
    row = tl.program_id(0)
    cols = tl.arange(0, K_POW2)
    c = tl.load(col_ptr + row * col_stride + cols, mask=cols < k, other=-1)
    valid = c >= 0
    cc = tl.where(valid, c, 0)
    cand = tl.load(cand_ptr + row * cand_stride + cc // CBS)
    pos = cand * CBS + cc % CBS
    # A -1 candidate slot can only be picked if the row had fewer than topk
    # finite columns, in which case the position is meaningless -- drop it.
    valid = valid & (cand >= 0)
    tl.store(
        out_ptr + row * out_stride + cols, tl.where(valid, pos, -1), mask=cols < k
    )


def sort_candidate_blocks(candidate_blocks: torch.Tensor) -> None:
    """Sort each row's candidate block ids ascending, in place.

    ``select_candidate_blocks`` emits them in score order, which makes the
    per-column KV gathers hop randomly over the cache. Sorting them costs one
    [rows, K] sort in the candidate-write layer (layer 20) and is then reused
    by all eight indexer layers in the step; on A40 at 8 rows it halves the
    sparse logits kernel. ``-1`` pads sort to the front, which the kernels
    handle. Nothing downstream depends on candidate order.
    """
    candidate_blocks.copy_(candidate_blocks.sort(dim=-1).values)


def sparse_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
) -> torch.Tensor:
    """Decode MQA logits over the candidate blocks only.

    Args and cache layout match ``fp8_paged_mqa_logits_triton``, except that
    the result is ``[rows, num_candidates * candidate_block_size]`` in
    compressed column space rather than ``[rows, max_model_len]``. Use
    ``sparse_topk_remap`` to get request-local positions back.
    """
    batch_size, next_n, num_heads, dim = q.shape
    rows = batch_size * next_n
    block_size = kv_cache.shape[1]
    assert dim % 16 == 0
    assert block_size % candidate_block_size == 0, (
        "a candidate block must not straddle two pages",
        block_size,
        candidate_block_size,
    )
    kv_block_stride = _paged_cache_block_stride(kv_cache, dim)
    emulate = needs_e4m3fn_emulation()
    q = q.contiguous().view(torch.uint8 if emulate else torch.float8_e4m3fn)

    assert weights.shape[0] >= rows and weights.shape[1] == num_heads
    assert weights.stride(1) == 1 and weights.stride(0) == num_heads
    assert block_table.dim() == 2 and block_table.stride(1) == 1
    if context_lens.dim() == 1:
        context_lens = context_lens.unsqueeze(-1)
    ctx_per_token = next_n > 1 and context_lens.shape[1] == next_n
    assert ctx_per_token or context_lens.shape[1] == 1, context_lens.shape
    assert candidate_blocks.shape[0] >= rows and candidate_blocks.stride(1) == 1

    num_sparse_cols = candidate_blocks.shape[1] * candidate_block_size
    out = torch.empty((rows, num_sparse_cols), dtype=torch.float32, device=q.device)
    grid = (rows, triton.cdiv(num_sparse_cols, _SPARSE_BLOCK_N))
    _sparse_paged_mqa_logits_kernel[grid](
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        out,
        context_lens.stride(0),
        context_lens.stride(1),
        block_table.stride(0),
        candidate_blocks.stride(0),
        kv_block_stride,
        out.stride(0),
        num_sparse_cols,
        NEXT_N=next_n,
        H=num_heads,
        HP=max(16, triton.next_power_of_2(num_heads)),
        D=dim,
        BLOCK_SIZE=block_size,
        CBS=candidate_block_size,
        BLOCK_N=_SPARSE_BLOCK_N,
        CTX_PER_TOKEN=ctx_per_token,
        EMULATE_E4M3=emulate,
        num_warps=_SPARSE_NUM_WARPS,
    )
    return out


def sparse_topk_remap(
    logits: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    topk_tokens: int,
    topk_indices: torch.Tensor,
) -> None:
    """Top-k the compressed logits and write request-local positions out.

    ``logits`` is [rows, num_sparse_cols] with -inf in unusable columns, so a
    plain row top-k needs no per-row bound. Writes into ``topk_indices[:rows]``.

    A row with fewer than ``topk_tokens`` finite columns (a prompt shorter than
    topk, or a mostly -1 candidate row) makes top-k return -inf columns whose
    positions are meaningless; those are dropped to -1 here rather than being
    remapped into reads past the end of the context.
    """
    rows = logits.shape[0]
    vals, cols = torch.topk(logits, topk_tokens, dim=-1)
    cols = cols.to(torch.int32)
    cols = torch.where(torch.isneginf(vals), torch.full_like(cols, -1), cols)
    _remap_kernel[(rows,)](
        cols,
        cols.stride(0),
        candidate_blocks,
        candidate_blocks.stride(0),
        topk_indices,
        topk_indices.stride(0),
        topk_tokens,
        CBS=candidate_block_size,
        K_POW2=triton.next_power_of_2(topk_tokens),
    )
