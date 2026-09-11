# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton FP8 MQA logits for the sparse-attention indexer.

Drop-in stand-ins for DeepGEMM's ``fp8_fp4_mqa_logits`` (prefill) and
``fp8_fp4_paged_mqa_logits`` (decode) on GPUs DeepGEMM does not support
(below SM 9.0). FP8 only: the MXFP4 indexer cache is SM 10.x-only anyway.

Both compute, per query row ``m`` and key ``n``::

    logits[m, n] = k_scale[n] * sum_h weights[m, h] * relu(q[m, h] . k[n])

reducing heads in-register, so the output is 2-D and nothing of size
``[H, M, N]`` is ever materialised. Neither kernel synchronises with the
host, so both are CUDA-graph capturable. Positions outside a row's context
are written as ``-inf`` (DeepGEMM's ``clean_logits=True``); the downstream
top-k kernels only read inside the context either way.

On GPUs whose Triton has no ``fp8e4nv`` (Ampere) the e4m3 operands are read
as ``uint8`` and decoded in-kernel; every e4m3 value is exact in fp16, so the
dot products run on fp16 tensor cores with an fp32 accumulator.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8_e4m3 import e4m3fn_bits_to_fp32, needs_e4m3fn_emulation


def use_triton_mqa_logits() -> bool:
    """Whether the CUDA sparse indexer should use these kernels.

    DeepGEMM's MQA-logits kernels (and their scheduling metadata) need
    SM 9.0+. ``has_deep_gemm()`` cannot be the test: vLLM vendors DeepGEMM,
    so it imports fine on every GPU and only fails at launch.
    """
    return current_platform.is_cuda() and not current_platform.has_device_capability(90)


# Tile sizes, swept on A40 (SM 8.6) at H=64, D=128.
_PAGED_BLOCK_N = 64
_PAGED_NUM_WARPS = 2
_BLOCK_M = 32
_BLOCK_N = 64
_NUM_WARPS = 4


@triton.jit
def _load_e4m3_as_f16(ptr, mask, EMULATE_E4M3: tl.constexpr):
    if EMULATE_E4M3:
        return e4m3fn_bits_to_fp32(tl.load(ptr, mask=mask, other=0)).to(tl.float16)
    else:
        return tl.load(ptr, mask=mask, other=0.0).to(tl.float16)


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,  # [B, NEXT_N, H, D] e4m3 (uint8 bits when EMULATE_E4M3)
    kv_ptr,  # uint8 base of the paged cache; block b starts at b * kv_block_stride
    w_ptr,  # [B * NEXT_N, H] fp32
    ctx_ptr,  # [B, 1] or [B, NEXT_N] int32
    bt_ptr,  # [B, max_blocks] int32
    out_ptr,  # [B * NEXT_N, max_model_len] fp32
    ctx_stride_b,
    ctx_stride_j,
    bt_stride,
    kv_block_stride,
    out_stride,
    max_model_len,
    NEXT_N: tl.constexpr,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_IN_PAGE: tl.constexpr,
    CTX_PER_TOKEN: tl.constexpr,
    EMULATE_E4M3: tl.constexpr,
):
    row = tl.program_id(0)
    n0 = tl.program_id(1) * BLOCK_N
    b = row // NEXT_N
    j = row % NEXT_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + row.to(tl.int64) * out_stride + offs_n
    out_mask = offs_n < max_model_len

    # Keys visible to query row (b, j): given per token, or derived from the
    # request length as L_b - NEXT_N + 1 + j.
    if CTX_PER_TOKEN:
        lim = tl.load(ctx_ptr + b * ctx_stride_b + j * ctx_stride_j)
    else:
        lim = tl.load(ctx_ptr + b * ctx_stride_b) - NEXT_N + 1 + j
    lim = tl.maximum(lim, 0)
    if n0 >= lim:
        tl.store(out_ptrs, tl.full([BLOCK_N], float("-inf"), tl.float32), out_mask)
        return

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

    valid = offs_n < lim
    bt_row = bt_ptr + b.to(tl.int64) * bt_stride
    # A tile inside one page has a uniform page index, which lets Triton
    # vectorise the byte loads; otherwise gather pages per token. Tokens past
    # the context read a valid page and are masked to -inf below.
    if TILE_IN_PAGE:
        page = tl.load(bt_row + n0 // BLOCK_SIZE).to(tl.int64)
        pages = tl.zeros([BLOCK_N], dtype=tl.int64) + page
        tok = (n0 % BLOCK_SIZE) + tl.arange(0, BLOCK_N)
    else:
        pages = tl.load(bt_row + offs_n // BLOCK_SIZE, mask=valid, other=0)
        pages = pages.to(tl.int64)
        tok = offs_n % BLOCK_SIZE
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


def _paged_cache_block_stride(kv_cache: torch.Tensor, dim: int) -> int:
    """Byte stride between blocks of the packed ``uint8`` indexer K cache.

    Blocks are not packed back to back: the page is padded for alignment and
    block-outermost layouts interleave other layers' pages, so ``stride(0)``
    is the only valid block stride. Never ``.view()`` or ``.contiguous()`` the
    cache here -- the latter would copy all of it on every call.
    """
    row_bytes = dim + 4
    assert kv_cache.dtype == torch.uint8, kv_cache.dtype
    assert kv_cache.shape[-1] == row_bytes, (kv_cache.shape, dim)
    assert kv_cache.dim() == 3 or (kv_cache.dim() == 4 and kv_cache.shape[2] == 1)
    assert kv_cache.stride(-1) == 1 and kv_cache.stride(1) == row_bytes, (
        "indexer K cache rows must be dense within a block",
        kv_cache.stride(),
    )
    block_stride = kv_cache.stride(0)
    assert block_stride >= kv_cache.shape[1] * row_bytes and block_stride % 4 == 0
    return block_stride


def _as_kernel_fp8(t: torch.Tensor, emulate: bool) -> torch.Tensor:
    assert t.dtype in (torch.float8_e4m3fn, torch.uint8), t.dtype
    return t.view(torch.uint8) if emulate else t.view(torch.float8_e4m3fn)


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Decode MQA logits over the paged indexer K cache.

    Args:
        q: ``[B, next_n, H, D]`` e4m3.
        kv_cache: ``[num_blocks, block_size, (1,) D + 4]`` uint8; each block is
            ``block_size`` row-major e4m3 keys followed by their fp32 scales.
            ``stride(0)`` may be padded.
        weights: ``[B * next_n, H]`` fp32, with the Q scale folded in.
        context_lens: ``[B, next_n]`` per-row lengths, or ``[B]``/``[B, 1]``
            request lengths (row ``j`` then sees ``L - next_n + 1 + j`` keys).
        block_table: ``[B, max_blocks]`` int32.

    Returns:
        ``[B * next_n, max_model_len]`` fp32, ``-inf`` outside each context.
    """
    batch_size, next_n, num_heads, dim = q.shape
    block_size = kv_cache.shape[1]
    assert dim % 16 == 0
    kv_block_stride = _paged_cache_block_stride(kv_cache, dim)
    emulate = needs_e4m3fn_emulation()
    q = _as_kernel_fp8(q.contiguous(), emulate)

    rows = batch_size * next_n
    assert weights.shape[0] >= rows and weights.shape[1] == num_heads
    assert weights.stride(1) == 1 and weights.stride(0) == num_heads
    assert block_table.dim() == 2 and block_table.stride(1) == 1
    if context_lens.dim() == 1:
        context_lens = context_lens.unsqueeze(-1)
    assert context_lens.dim() == 2 and context_lens.shape[0] >= batch_size
    ctx_per_token = next_n > 1 and context_lens.shape[1] == next_n
    assert ctx_per_token or context_lens.shape[1] == 1, context_lens.shape

    out = torch.empty((rows, max_model_len), dtype=torch.float32, device=q.device)
    block_n = max(16, min(_PAGED_BLOCK_N, block_size))
    grid = (rows, triton.cdiv(max_model_len, block_n))
    _fp8_paged_mqa_logits_kernel[grid](
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        out,
        context_lens.stride(0),
        context_lens.stride(1),
        block_table.stride(0),
        kv_block_stride,
        out.stride(0),
        max_model_len,
        NEXT_N=next_n,
        H=num_heads,
        HP=max(16, triton.next_power_of_2(num_heads)),
        D=dim,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        TILE_IN_PAGE=block_size % block_n == 0,
        CTX_PER_TOKEN=ctx_per_token,
        EMULATE_E4M3=emulate,
        num_warps=_PAGED_NUM_WARPS,
    )
    return out


@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,  # [M, H, D] fp16 (pre-decoded)
    k_ptr,  # [N, D] e4m3 (uint8 bits when EMULATE_E4M3)
    k_scale_ptr,  # [N] fp32
    w_ptr,  # [M, H] fp32
    ks_ptr,  # [M] int32
    ke_ptr,  # [M] int32
    out_ptr,  # [M, N] fp32
    M,
    N,
    k_scale_stride,
    out_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EMULATE_E4M3: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n0 = tl.program_id(1) * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < M
    mask_n = offs_n < N

    ks = tl.load(ks_ptr + offs_m, mask=mask_m, other=0)
    ke = tl.load(ke_ptr + offs_m, mask=mask_m, other=0)
    out_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * out_stride + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    # Whole tile outside every row's [ks, ke): skip the math.
    if (n0 >= tl.max(ke)) | (n0 + BLOCK_N <= tl.min(ks)):
        tl.store(
            out_ptrs,
            tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32),
            mask=out_mask,
        )
        return

    k = _load_e4m3_as_f16(
        k_ptr + offs_n[:, None].to(tl.int64) * D + offs_d[None, :],
        mask_n[:, None],
        EMULATE_E4M3,
    )
    kt = tl.trans(k)  # [D, BLOCK_N]
    scale = tl.load(k_scale_ptr + offs_n * k_scale_stride, mask=mask_n, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for h in tl.static_range(H):
        q = tl.load(
            q_ptr + (offs_m[:, None].to(tl.int64) * H + h) * D + offs_d[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(w_ptr + offs_m * H + h, mask=mask_m, other=0.0)
        acc += tl.maximum(tl.dot(q, kt), 0.0) * w[:, None]
    acc = acc * scale[None, :]
    valid = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
    tl.store(out_ptrs, tl.where(valid, acc, float("-inf")), mask=out_mask)


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Prefill MQA logits over gathered (non-paged) keys.

    Args:
        q: ``[M, H, D]`` e4m3.
        kv: ``(k [N, D] e4m3, k_scale [N] or [N, 1] fp32)``.
        weights: ``[M, H]`` fp32, with the Q scale folded in.
        cu_seqlen_ks, cu_seqlen_ke: ``[M]`` int32 key window ``[ks, ke)``.

    Returns:
        ``[M, N]`` fp32, ``-inf`` outside each row's window.
    """
    k_fp8, k_scale = kv
    M, H, D = q.shape
    N = k_fp8.shape[0]
    assert k_fp8.shape[1] == D and D % 16 == 0 and k_fp8.is_contiguous()
    if k_scale.dim() == 2:
        assert k_scale.shape[1] == 1
        k_scale = k_scale[:, 0]
    assert k_scale.shape[0] == N and k_scale.dtype == torch.float32
    assert weights.shape == (M, H)
    emulate = needs_e4m3fn_emulation()
    # Decoding Q once up front beats re-decoding it for every key tile.
    q16 = q.view(torch.float8_e4m3fn).to(torch.float16)
    weights = weights.contiguous()

    out = torch.empty((M, N), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    _fp8_mqa_logits_kernel[grid](
        q16,
        _as_kernel_fp8(k_fp8, emulate),
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        M,
        N,
        k_scale.stride(0),
        out.stride(0),
        H=H,
        D=D,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        EMULATE_E4M3=emulate,
        num_warps=_NUM_WARPS,
    )
    return out
