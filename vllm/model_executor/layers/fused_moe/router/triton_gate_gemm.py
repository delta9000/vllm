# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M MoE router gate GEMM (bf16 x bf16 -> fp32) in Triton.

On GPUs without a specialised router kernel the gate runs through cuBLAS,
which at decode sizes (M <= 16, K = 5120, N = 384) picks a split-K
algorithm and a separate ``splitKreduce`` kernel: two launches per MoE
layer. This kernel is one launch: split-K partials are combined by the last
CTA to finish each N tile through a self-resetting counter (deterministic,
CUDA-graph safe, no host-side zeroing).
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

MAX_M = 16
BLOCK_N = 32
BLOCK_K = 64
SPLIT_K = 8


@triton.jit
def _gate_gemm_kernel(
    x_ptr,  # [M, K] bf16
    w_ptr,  # [N, K] bf16
    out_ptr,  # [M, N] fp32
    part_ptr,  # [SPLIT_K, M, N] fp32 (SPLIT_K > 1)
    cnt_ptr,  # [n_tiles] int32, zero at rest (SPLIT_K > 1)
    M,
    N,
    K,
    k_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    K = tl.multiple_of(K, 8)
    k_per_split = tl.multiple_of(k_per_split, BLOCK_K)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    k_start = tl.multiple_of(pid_s * k_per_split, BLOCK_K)
    k_end = tl.minimum(k_start + k_per_split, K)
    x_base = x_ptr + offs_m[:, None] * K
    w_base = w_ptr + offs_n[:, None] * K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        kk = tl.multiple_of(k0, BLOCK_K) + offs_k
        kmask = kk < k_end
        x = tl.load(
            x_base + kk[None, :], mask=mask_m[:, None] & kmask[None, :], other=0.0
        )
        w = tl.load(
            w_base + kk[None, :], mask=mask_n[:, None] & kmask[None, :], other=0.0
        )
        acc += tl.dot(x, tl.trans(w))

    out_mask = mask_m[:, None] & mask_n[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        part_row = offs_m[:, None] * N + offs_n[None, :]
        tl.store(part_ptr + pid_s * M * N + part_row, acc, mask=out_mask)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tl.debug_barrier()
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                total += tl.load(
                    part_ptr + s * M * N + part_row,
                    mask=out_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.store(out_ptrs, total, mask=out_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="release", scope="gpu")


_WORKSPACE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_workspace(device: torch.device, part_elems: int, tiles: int):
    if device.type == "cuda":
        key = (device.index or 0, torch.cuda.current_stream(device).stream_id)
    else:  # TRITON_INTERPRET=1 on CPU tensors
        key = (-1, 0)
    ws = _WORKSPACE.get(key)
    if ws is None or ws[0].numel() < part_elems or ws[1].numel() < tiles:
        part_elems = max(part_elems, ws[0].numel() if ws else 0)
        tiles = max(tiles, ws[1].numel() if ws else 0)
        ws = (
            torch.empty(part_elems, dtype=torch.float32, device=device),
            torch.zeros(tiles, dtype=torch.int32, device=device),
        )
        _WORKSPACE[key] = ws
    return ws


def triton_gate_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x`` [M <= 16, K] bf16 times ``weight`` [N, K] bf16 -> [M, N] fp32."""
    M, K = x.shape
    N = weight.shape[0]
    assert M <= MAX_M and weight.shape[1] == K and K % 8 == 0
    assert x.dtype == weight.dtype and x.dtype in (torch.bfloat16, torch.float16)
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    if M == 0:
        return out
    n_tiles = triton.cdiv(N, BLOCK_N)
    chunks = triton.cdiv(K, BLOCK_K)
    split = max(1, min(SPLIT_K, chunks // 2))
    k_per_split = triton.cdiv(chunks, split) * BLOCK_K
    split = triton.cdiv(K, k_per_split)
    if split > 1:
        part, cnt = _get_workspace(x.device, split * M * N, n_tiles)
    else:
        part, cnt = out, out
    _gate_gemm_kernel[(n_tiles, split)](
        x,
        weight,
        out,
        part,
        cnt,
        M,
        N,
        K,
        k_per_split,
        BLOCK_M=MAX_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_K=split,
        num_warps=4,
        num_stages=3,
    )
    return out


def _triton_gate_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device
    )


direct_register_custom_op(
    "triton_gate_gemm", triton_gate_gemm, fake_impl=_triton_gate_gemm_fake
)
