# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton mHC pre-norm projection for GPUs without DeepGEMM (SM < 9.0).

Computes ``out[s, m, :] = x[m, k_s] @ fn[:, k_s]^T`` over ``SPLIT_K`` K-chunks
and ``sqrsum[s, m] = sum(x[m, k_s]^2)``, the ``(mixes, sqrsum)`` pair the
``mhc_pre_big_fuse*`` kernels reduce over their ``n_splits`` axis.

The TileLang ``hc_prenorm_gemm_block_m`` kernel that served prefill here
handles two tokens per CTA and re-reads the whole 2 MB fp32 ``fn`` for every
pair, i.e. ~4 GB of L2 traffic per 4096-token chunk. This kernel tiles 32
tokens per CTA on ``tl.dot`` (fp32 FMA by default, so numerics stay in the
same class as the fp32 TileLang kernel; ``tf32`` is opt-in and is what the
DeepGEMM path on Hopper uses) and splits K so that even a few hundred
tokens fill the SMs.
"""

import torch

from vllm.triton_utils import tl, triton

# fp32 FMA ("ieee") keeps the exact-fp32 numerics of the kernel it replaces;
# "tf32" is ~4x faster on tensor cores and is the precision DeepGEMM uses for
# this projection on SM 9.0+.
PRECISION = "ieee"
BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 64
SPLIT_K = 4


@triton.jit
def _hc_prenorm_gemm_kernel(
    x_ptr,  # [M, K] bf16
    fn_ptr,  # [N, K] fp32
    out_ptr,  # [SPLIT_K, M, N] fp32
    sqr_ptr,  # [SPLIT_K, M] fp32
    M,
    N,
    K,
    k_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    K = tl.multiple_of(K, 8)
    k_per_split = tl.multiple_of(k_per_split, BLOCK_K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    k_start = tl.multiple_of(pid_s * k_per_split, BLOCK_K)
    k_end = tl.minimum(k_start + k_per_split, K)

    x_base = x_ptr + offs_m[:, None] * K
    fn_base = fn_ptr + offs_n[:, None] * K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        kk = tl.multiple_of(k0, BLOCK_K) + offs_k
        kmask = kk < k_end
        x = tl.load(
            x_base + kk[None, :], mask=mask_m[:, None] & kmask[None, :], other=0.0
        ).to(tl.float32)
        w = tl.load(
            fn_base + kk[None, :], mask=mask_n[:, None] & kmask[None, :], other=0.0
        )
        acc += tl.dot(x, tl.trans(w), input_precision=PRECISION)
        sq += tl.sum(x * x, axis=1)

    out_ptrs = out_ptr + (pid_s * M + offs_m)[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(sqr_ptr + pid_s * M + offs_m, sq, mask=mask_m)


def hc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
) -> None:
    """Fill ``out`` [S, M, N] and ``sqrsum`` [S, M] (S = ``out.shape[0]``)."""
    M, K = x.shape
    N = fn.shape[0]
    S = out.shape[0]
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert fn.dtype == torch.float32 and fn.is_contiguous() and fn.shape[1] == K
    assert out.shape == (S, M, N) and sqrsum.shape == (S, M)
    assert N <= BLOCK_N and K % 8 == 0
    if M == 0:
        return
    chunks = triton.cdiv(K, BLOCK_K)
    k_per_split = triton.cdiv(chunks, S) * BLOCK_K
    assert S * k_per_split >= K
    _hc_prenorm_gemm_kernel[(triton.cdiv(M, BLOCK_M), S)](
        x,
        fn,
        out,
        sqrsum,
        M,
        N,
        K,
        k_per_split,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        PRECISION=PRECISION,
        num_warps=4,
        num_stages=3,
    )
