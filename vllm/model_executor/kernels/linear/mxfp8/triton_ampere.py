# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 W8A16 linear in Triton for NVIDIA GPUs below SM 9.0.

Keeps the checkpoint layout -- ``weight`` [N, K] e4m3 and ``weight_scale``
[N, K // 32] e8m0 bytes -- and runs one of two kernels chosen by M:

* ``_mxfp8_gemv_fma_kernel`` (M <= 4, decode): a streaming GEMV with no
  tensor cores and no weight traffic through shared memory. Every thread decodes
  its 16 B of weight bytes in registers, multiplies by the block scale and
  the broadcast activation rows, and keeps fp32 partial sums in a 4-D
  register tile that is reduced once after the loop, with the next weight
  tile prefetched into registers so DRAM stays busy across iterations
  (Triton only pipelines loops that contain a ``tl.dot``).
* ``_mxfp8_linear_kernel`` (M > 4): tiled tensor-core GEMM, decoding e4m3 to
  bf16 and applying each 32-wide block scale to the fp32 block result, so it
  matches dequantize-then-matmul. Decode-sized M (<= 16) uses a [16, N] tile
  and split-K; prefill a [64, 128] tile.

Split-K in both kernels combines partials through a self-resetting per-tile
counter: the last CTA to finish a tile reduces the partials in a fixed order
and writes the bf16 output, so a launch is one kernel, deterministic and
CUDA-graph capturable (no host syncs, fixed workspace addresses). The split
rule favours long-lived CTAs (>= ``min_iters`` loop iterations each) and
splits only when the N tiles alone do not fill the SMs, so each CTA streams
in steady state rather than in one lock-step burst.

Grouped layers (``bmm_batch_size`` G > 1): ``x`` is [M, G, K] and the weight
[G * N, K]; group ``g`` multiplies its own [N, K] slab.

Below SM 8.9 Triton has no ``float8_e4m3fn`` dtype, so the weight is passed
as a ``uint8`` view and decoded with :mod:`vllm.triton_utils.fp8_e4m3`.
"""

from dataclasses import dataclass, replace

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8_e4m3 import (
    e4m3fn_bits_to_bf16,
    e4m3fn_bits_to_fp32,
    needs_e4m3fn_emulation,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# Largest M on the register-FMA GEMV, and on the decode-sized mma tile.
FMA_MAX_M = 4
DECODE_MAX_M = 16
# CTAs per SM the split rule aims to keep resident (A40: 84 SMs).
_NUM_SMS = 84


@dataclass(frozen=True)
class TileConfig:
    """Launch configuration for one kernel variant.

    ``fill_target`` is the CTA count split-K aims for (0: never split);
    ``min_iters`` the fewest loop iterations a CTA may be left with.
    """

    variant: str  # "fma" or "mma"
    block_n: int
    k_blocks: int  # 32-wide K blocks per loop iteration
    num_warps: int
    num_stages: int  # mma only: Triton's pipeliner needs a tl.dot to engage
    fill_target: int = 2 * _NUM_SMS
    min_iters: int = 8
    block_m: int = 0  # mma only; fma derives it from M
    prefetch: bool = True  # fma only: register double-buffering of the weights

    def regs_estimate(self, block_m: int) -> int:
        """Rough fp32 registers per thread held by the fma accumulators."""
        return block_m * self.block_n * self.k_blocks * 32 // (32 * self.num_warps)


# Defaults by M bucket (1, 4, 16): paper values; the cold sweep in
# dev/tools/sweep_cold.py writes per-shape winners into DECODE_CONFIGS.
DECODE_DEFAULTS: dict[int, TileConfig] = {
    1: TileConfig("fma", block_n=32, k_blocks=4, num_warps=4, num_stages=1),
    4: TileConfig("fma", block_n=16, k_blocks=4, num_warps=4, num_stages=1),
    16: TileConfig(
        "mma", block_n=64, k_blocks=4, num_warps=4, num_stages=3, block_m=16
    ),
}
# (m_bucket, N, K) -> config, for the per-GPU shapes measured on the target.
DECODE_CONFIGS: dict[tuple[int, int, int], TileConfig] = {}
PREFILL_CONFIG = TileConfig(
    "mma", block_n=128, k_blocks=4, num_warps=8, num_stages=3, fill_target=0, block_m=64
)


def _m_bucket(M: int) -> int:
    return 1 if M <= 1 else FMA_MAX_M if M <= FMA_MAX_M else DECODE_MAX_M


def decode_config(M: int, N: int, K: int) -> TileConfig:
    b = _m_bucket(M)
    return DECODE_CONFIGS.get((b, N, K), DECODE_DEFAULTS[b])


@triton.jit
def _e8m0_to_fp32(s):
    """e8m0 byte -> 2^(s - 127) as fp32, including the subnormal 2^-127 for
    s == 0 (which ``exp2(s - 127.0)`` also yields)."""
    bits = s.to(tl.int32) << 23
    bits = tl.where(s == 0, 0x00400000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _load_w_bf16(ptrs, mask, EMULATE_E4M3: tl.constexpr):
    if EMULATE_E4M3:
        return e4m3fn_bits_to_bf16(tl.load(ptrs, mask=mask, other=0))
    else:
        return tl.load(ptrs, mask=mask, other=0.0).to(tl.bfloat16)


@triton.jit
def _load_w_raw(ptrs, mask, EMULATE_E4M3: tl.constexpr):
    """Undecoded weight tile: uint8 bits when emulating, else native e4m3
    (whose masked fill must be a float)."""
    if EMULATE_E4M3:
        return tl.load(ptrs, mask=mask, other=0)
    else:
        return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def _epilogue(
    acc,
    out_ptrs,
    out_mask,
    bias_ptr,
    rows,
    mask_n,
    part_ptr,
    part_row,
    cnt_ptr,
    tile,
    pid_k,
    stride_part,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Store ``acc`` [BLOCK_M, BLOCK_N] fp32, reducing over K-splits first."""
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(bias_ptr + rows, mask=mask_n, other=0.0).to(tl.float32)[
                None, :
            ]
        tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=out_mask)
    else:
        # Publish this split's partial, then let the last CTA to arrive at
        # this output tile reduce all of them (fixed order, so deterministic).
        tl.store(part_ptr + pid_k * stride_part + part_row, acc, mask=out_mask)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tl.debug_barrier()
            total = tl.zeros_like(acc)
            for s in tl.static_range(SPLIT_K):
                total += tl.load(
                    part_ptr + s * stride_part + part_row,
                    mask=out_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            if HAS_BIAS:
                total += tl.load(bias_ptr + rows, mask=mask_n, other=0.0).to(
                    tl.float32
                )[None, :]
            tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=out_mask)
            # Leave the counter at zero for the next launch (graph replay).
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="release", scope="gpu")


@triton.jit
def _mxfp8_gemv_fma_kernel(
    x_ptr,  # [M, G, K] bf16, strides (stride_xm, stride_xg, 1)
    w_ptr,  # [G * N, K] e4m3 (uint8 bits when EMULATE_E4M3)
    s_ptr,  # [G * N, K // 32] e8m0 bytes
    bias_ptr,  # [G * N] or unused
    out_ptr,  # [M, G, N] bf16, strides (stride_om, stride_og, 1)
    part_ptr,  # [SPLIT_K, G, M, N] fp32 workspace (split-K only)
    cnt_ptr,  # [tiles] int32, zero at rest (split-K only)
    M,
    N,
    K,
    G,
    stride_xm,
    stride_xg,
    stride_om,
    stride_og,
    k_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EMULATE_E4M3: tl.constexpr,
    PREFETCH: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_mg = tl.program_id(2)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    pid_m = pid_mg % num_m_tiles
    g = pid_mg // num_m_tiles

    STEP: tl.constexpr = K_BLOCKS * 32
    K = tl.multiple_of(K, 32)
    k_per_split = tl.multiple_of(k_per_split, STEP)
    stride_xm = tl.multiple_of(stride_xm, 8)
    x_goff = tl.multiple_of(g * stride_xg, 8)
    o_goff = tl.multiple_of(g * stride_og, 8)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, K_BLOCKS)
    offs_kk = tl.arange(0, 32)
    mask_m = offs_m < M
    mask_n = offs_n < N
    rows = g * N + offs_n

    k_start = tl.multiple_of(pid_k * k_per_split, STEP)
    k_end = tl.minimum(k_start + k_per_split, K)

    # Weight and activation are [BLOCK_N, K_BLOCKS, 32] tiles in one layout:
    # the inner 32 is one scale block and each thread owns 16 contiguous
    # bytes of it (one 16 B load). The activation row is read through a
    # pointer broadcast over BLOCK_N (L1 hits), which keeps it in the same
    # layout, so the weight never goes through shared memory. Only the tiny
    # [BLOCK_N, K_BLOCKS, 1] scale tile is rearranged per iteration (a few
    # bytes per thread, two barriers) -- reading it pre-broadcast instead
    # costs 32 predicated byte loads per thread, which is worse.
    kin3 = (offs_kb[:, None] * 32 + offs_kk[None, :])[None, :, :]
    w_ptrs = w_ptr + rows[:, None, None] * K + kin3
    s_ptrs = s_ptr + rows[:, None, None] * (K // 32) + offs_kb[None, :, None]
    x_ptrs = tl.broadcast_to(x_ptr + x_goff + kin3, (BLOCK_N, K_BLOCKS, 32))
    x_ptrs = x_ptrs + (pid_m * BLOCK_M) * stride_xm
    wmask = tl.broadcast_to(mask_n[:, None, None], (BLOCK_N, K_BLOCKS, 32))
    smask = mask_n[:, None, None]
    xmask = tl.zeros((BLOCK_N, K_BLOCKS, 32), dtype=tl.int1) | True

    acc0 = tl.zeros((BLOCK_N, K_BLOCKS, 32), dtype=tl.float32)
    if BLOCK_M >= 2:
        acc1 = tl.zeros((BLOCK_N, K_BLOCKS, 32), dtype=tl.float32)
    if BLOCK_M >= 4:
        acc2 = tl.zeros((BLOCK_N, K_BLOCKS, 32), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_N, K_BLOCKS, 32), dtype=tl.float32)

    # Register double-buffering: the weight tile of iteration i+1 is in
    # flight while iteration i decodes and accumulates.
    w_nxt = _load_w_raw(
        w_ptrs + k_start, wmask & ((k_start + kin3) < k_end), EMULATE_E4M3
    )
    s_nxt = tl.load(
        s_ptrs + k_start // 32,
        mask=smask & ((k_start + offs_kb * 32) < k_end)[None, :, None],
        other=0,
    )
    for k0 in range(k_start, k_end, STEP):
        k0 = tl.multiple_of(k0, STEP)
        if PREFETCH:
            w_raw = w_nxt
            s_raw = s_nxt
            k1 = k0 + STEP
            w_nxt = _load_w_raw(
                w_ptrs + k1, wmask & ((k1 + kin3) < k_end), EMULATE_E4M3
            )
            s_nxt = tl.load(
                s_ptrs + k1 // 32,
                mask=smask & ((k1 + offs_kb * 32) < k_end)[None, :, None],
                other=0,
            )
        else:
            w_raw = _load_w_raw(
                w_ptrs + k0, wmask & ((k0 + kin3) < k_end), EMULATE_E4M3
            )
            s_raw = tl.load(
                s_ptrs + k0 // 32,
                mask=smask & ((k0 + offs_kb * 32) < k_end)[None, :, None],
                other=0,
            )
        if EMULATE_E4M3:  # noqa: SIM108 -- constexpr branch, not a ternary
            w = e4m3fn_bits_to_fp32(w_raw)
        else:
            w = w_raw.to(tl.float32)
        w = w * _e8m0_to_fp32(s_raw)  # [BLOCK_N, K_BLOCKS, 32] fp32
        xk = xmask & ((k0 + kin3) < k_end)
        x0 = tl.load(x_ptrs + k0, mask=xk & (pid_m * BLOCK_M < M), other=0.0)
        acc0 += x0.to(tl.float32) * w
        if BLOCK_M >= 2:
            x1 = tl.load(
                x_ptrs + (k0 + stride_xm),
                mask=xk & (pid_m * BLOCK_M + 1 < M),
                other=0.0,
            )
            acc1 += x1.to(tl.float32) * w
        if BLOCK_M >= 4:
            x2 = tl.load(
                x_ptrs + (k0 + 2 * stride_xm),
                mask=xk & (pid_m * BLOCK_M + 2 < M),
                other=0.0,
            )
            acc2 += x2.to(tl.float32) * w
            x3 = tl.load(
                x_ptrs + (k0 + 3 * stride_xm),
                mask=xk & (pid_m * BLOCK_M + 3 < M),
                other=0.0,
            )
            acc3 += x3.to(tl.float32) * w

    r0 = tl.sum(tl.sum(acc0, axis=2), axis=1)  # [BLOCK_N]
    if BLOCK_M == 1:
        res = r0[None, :]
    elif BLOCK_M == 2:
        r1 = tl.sum(tl.sum(acc1, axis=2), axis=1)
        res = tl.trans(tl.join(r0, r1))  # [2, BLOCK_N]
    else:
        r1 = tl.sum(tl.sum(acc1, axis=2), axis=1)
        r2 = tl.sum(tl.sum(acc2, axis=2), axis=1)
        r3 = tl.sum(tl.sum(acc3, axis=2), axis=1)
        # tl.join stacks along a NEW last axis: join(join(a, b), join(c, d))
        # is [BLOCK_N, 2, 2] with element [n, j, i] = (a, b, c, d)[2 * i + j],
        # so flattening it yields a, c, b, d. Pair (r0, r2) and (r1, r3) so
        # the flattened row order is r0, r1, r2, r3.
        res = tl.trans(
            tl.reshape(tl.join(tl.join(r0, r2), tl.join(r1, r3)), (BLOCK_N, 4))
        )  # [4, BLOCK_N]

    out_mask = mask_m[:, None] & mask_n[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + o_goff + offs_n[None, :]
    part_row = (g * M + offs_m)[:, None] * N + offs_n[None, :]
    tile = pid_mg * tl.num_programs(0) + pid_n
    _epilogue(
        res,
        out_ptrs,
        out_mask,
        bias_ptr,
        rows,
        mask_n,
        part_ptr,
        part_row,
        cnt_ptr,
        tile,
        pid_k,
        G * M * N,
        HAS_BIAS,
        SPLIT_K,
    )


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    bias_ptr,
    out_ptr,
    part_ptr,
    cnt_ptr,
    M,
    N,
    K,
    G,
    stride_xm,
    stride_xg,
    stride_om,
    stride_og,
    k_per_split,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EMULATE_E4M3: tl.constexpr,
):
    """Tensor-core path; same argument layout as the fma kernel."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_mg = tl.program_id(2)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    pid_m = pid_mg % num_m_tiles
    g = pid_mg // num_m_tiles

    # Alignment facts the wrapper guarantees, so weight/activation loads
    # vectorize to 16 B regardless of runtime integer specialization. The
    # group offsets must be hinted as products: a hint on the stride alone
    # does not survive the multiply by the runtime group index.
    K = tl.multiple_of(K, 32)
    k_per_split = tl.multiple_of(k_per_split, K_BLOCKS * 32)
    stride_xm = tl.multiple_of(stride_xm, 8)
    x_goff = tl.multiple_of(g * stride_xg, 8)
    o_goff = tl.multiple_of(g * stride_og, 8)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 32)
    mask_m = offs_m < M
    mask_n = offs_n < N
    rows = g * N + offs_n  # weight / scale / bias rows of this group

    k_start = tl.multiple_of(pid_k * k_per_split, K_BLOCKS * 32)
    k_end = tl.minimum(k_start + k_per_split, K)

    x_base = x_ptr + offs_m[:, None] * stride_xm + x_goff
    w_base = w_ptr + rows[:, None] * K
    s_base = s_ptr + rows * (K // 32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, K_BLOCKS * 32):
        for kb in tl.static_range(K_BLOCKS):
            k = tl.multiple_of(k0 + kb * 32, 32)
            # K % 32 == 0, so a block is either wholly inside [k_start, k_end)
            # or wholly outside it.
            blk_ok = k < k_end
            w = _load_w_bf16(
                w_base + (k + offs_k)[None, :],
                mask_n[:, None] & blk_ok,
                EMULATE_E4M3,
            )  # [BLOCK_N, 32]
            x = tl.load(
                x_base + (k + offs_k)[None, :],
                mask=mask_m[:, None] & blk_ok,
                other=0.0,
            )  # [BLOCK_M, 32]
            scale = tl.load(s_base + k // 32, mask=mask_n & blk_ok, other=0)
            blk = tl.dot(x, tl.trans(w))  # [BLOCK_M, BLOCK_N] fp32
            acc += blk * _e8m0_to_fp32(scale)[None, :]

    out_mask = mask_m[:, None] & mask_n[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + o_goff + offs_n[None, :]
    part_row = (g * M + offs_m)[:, None] * N + offs_n[None, :]
    tile = pid_mg * tl.num_programs(0) + pid_n
    _epilogue(
        acc,
        out_ptrs,
        out_mask,
        bias_ptr,
        rows,
        mask_n,
        part_ptr,
        part_row,
        cnt_ptr,
        tile,
        pid_k,
        G * M * N,
        HAS_BIAS,
        SPLIT_K,
    )


# Split-K workspace, keyed by (device, stream) so concurrent streams never
# share partials or counters. Grown on demand; vLLM's warmup runs the largest
# decode shapes before CUDA-graph capture, so the address is stable by then.
_WORKSPACE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_workspace(
    device: torch.device, part_elems: int, tiles: int
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _pick_split(cfg: TileConfig, tiles: int, K: int) -> tuple[int, int]:
    """Return (split_k, k_per_split), k_per_split a multiple of the K step.

    Split only when the output tiles alone leave SMs idle, and never below
    ``min_iters`` iterations per CTA, so CTAs stream in steady state.
    """
    step = cfg.k_blocks * 32
    chunks = triton.cdiv(K, step)
    split = 1
    if cfg.fill_target > 0 and tiles < cfg.fill_target:
        split = min(
            max(1, chunks // max(1, cfg.min_iters)),
            triton.cdiv(cfg.fill_target, tiles),
        )
    k_per_split = triton.cdiv(chunks, split) * step
    split = triton.cdiv(K, k_per_split)
    return split, k_per_split


def triton_mxfp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    groups: int,
    emulate_e4m3: bool,
) -> torch.Tensor:
    """``x`` [M, G, K] (any strides with unit K stride) times ``weight``
    [G * N, K] -> [M, G, N] in ``x.dtype``."""
    M, G, K = x.shape
    GN = weight.shape[0]
    assert groups == G and GN % G == 0 and K % MXFP8_BLOCK_SIZE == 0
    if x.stride(2) != 1 or x.stride(0) % 8 or x.stride(1) % 8:
        # The kernel promises 16 B-aligned activation rows.
        x = x.contiguous()
    N = GN // G
    # The kernel also promises 16 B-aligned output rows (stride_og = N).
    assert N % 8 == 0, f"output width {N} must be a multiple of 8"
    out = torch.empty((M, G, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out

    cfg = decode_config(M, N, K) if M <= DECODE_MAX_M else PREFILL_CONFIG
    variant_kw: dict[str, bool] = {}
    if cfg.variant == "fma":
        kernel = _mxfp8_gemv_fma_kernel
        block_m = triton.next_power_of_2(min(M, FMA_MAX_M))
        variant_kw["PREFETCH"] = cfg.prefetch
    else:
        kernel = _mxfp8_linear_kernel
        block_m = cfg.block_m
    n_tiles = triton.cdiv(N, cfg.block_n)
    m_tiles = triton.cdiv(M, block_m)
    tiles = n_tiles * m_tiles * G
    split_k, k_per_split = _pick_split(cfg, tiles, K)

    if split_k > 1:
        part, cnt = _get_workspace(x.device, split_k * G * M * N, tiles)
    else:
        part, cnt = out, out  # unused; keeps the signature uniform

    w_arg = weight.view(torch.uint8) if emulate_e4m3 else weight
    kernel[(n_tiles, split_k, m_tiles * G)](
        x,
        w_arg,
        weight_scale,
        bias if bias is not None else out,
        out,
        part,
        cnt,
        M,
        N,
        K,
        G,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        k_per_split,
        BLOCK_M=block_m,
        BLOCK_N=cfg.block_n,
        K_BLOCKS=cfg.k_blocks,
        SPLIT_K=split_k,
        HAS_BIAS=bias is not None,
        EMULATE_E4M3=emulate_e4m3,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        **variant_kw,
    )
    return out


def _triton_mxfp8_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    groups: int,
    emulate_e4m3: bool,
) -> torch.Tensor:
    M, G, _ = x.shape
    return torch.empty((M, G, weight.shape[0] // G), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    "triton_mxfp8_linear",
    triton_mxfp8_linear,
    fake_impl=_triton_mxfp8_linear_fake,
)


def set_decode_config(m_bucket: int, N: int, K: int, **fields) -> None:
    """Override the decode config for one shape (used by the cold sweep)."""
    base = DECODE_CONFIGS.get((m_bucket, N, K), DECODE_DEFAULTS[m_bucket])
    DECODE_CONFIGS[(m_bucket, N, K)] = replace(base, **fields)


class TritonMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A16 (and grouped BMM) via the Triton kernels above, for CUDA
    devices below SM 9.0. Weights stay in checkpoint layout."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "Triton MXFP8 linear is CUDA-only"
        if compute_capability is None:
            cap = current_platform.get_device_capability()
            if cap is None:
                return False, "no CUDA device"
            compute_capability = cap.to_int()
        if compute_capability >= 90:
            return False, "SM 9.0+ has native MXFP8 GEMMs"
        if compute_capability < 80:
            return False, "needs bf16 tensor cores (SM 8.0+)"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is not None and c.bmm_batch_size <= 0:
            return False, "bmm_batch_size must be positive"
        return True, None

    @property
    def groups(self) -> int:
        return self.config.bmm_batch_size or 1

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype != MXFP8_VALUE_DTYPE or weight.ndim != 2:
            raise ValueError(
                f"Triton MXFP8 expects a 2-D {MXFP8_VALUE_DTYPE} weight, got "
                f"{weight.dtype} {tuple(weight.shape)}"
            )
        N, K = weight.shape
        if N % self.groups != 0 or K % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"weight {tuple(weight.shape)} is not divisible by "
                f"groups={self.groups} / block={MXFP8_BLOCK_SIZE}"
            )
        scale = layer.weight_scale.data[:N, : K // MXFP8_BLOCK_SIZE].contiguous()
        if scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(f"expected {MXFP8_SCALE_DTYPE} scales, got {scale.dtype}")
        # Idempotent: rerunning on a reload re-derives the same tensors.
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)
        layer.weight_block_size = [1, MXFP8_BLOCK_SIZE]

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        G = self.groups
        K = layer.weight.shape[1]
        # Only a grouped layer takes the grouped path. A 3-D input to an
        # ungrouped one is an ordinary [..., K] activation, flattened below;
        # reading it as [M, G, K] would mistake seq_len for the group.
        if G > 1:
            # Grouped: x is [M, G, K] (a transposed [G, M, K] buffer from the
            # inverse-RoPE kernel is fine), output [M, G, N].
            assert x.ndim == 3 and x.shape[1] == G and x.shape[2] == K
            return torch.ops.vllm.triton_mxfp8_linear(
                x,
                layer.weight,
                layer.weight_scale,
                bias,
                G,
                needs_e4m3fn_emulation(),
            )
        lead = x.shape[:-1]
        out = torch.ops.vllm.triton_mxfp8_linear(
            x.reshape(-1, K).unsqueeze(1),
            layer.weight,
            layer.weight_scale,
            bias,
            1,
            needs_e4m3fn_emulation(),
        )
        return out.view(*lead, -1)
