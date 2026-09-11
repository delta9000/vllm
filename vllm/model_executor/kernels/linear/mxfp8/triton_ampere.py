# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 W8A16 linear in Triton for NVIDIA GPUs below SM 9.0.

The Marlin MXFP8 kernel that otherwise serves these GPUs reaches ~35% of
DRAM bandwidth at decode (M <= 16) on an A40, and its repacked weight cannot
be reused for the grouped ``wo_a`` projection of DeepSeek V4, which today is
dequantized to bf16 at load. This module keeps the checkpoint layout --
``weight`` [N, K] e4m3 and ``weight_scale`` [N, K // 32] e8m0 bytes -- and
runs one Triton kernel for three cases:

* decode (M <= 16): a bandwidth-oriented split-K GEMV. Each CTA streams a
  [BLOCK_N, k-chunk] slab of weight bytes, decodes them in registers, and
  contracts them against the (L2-resident) activation rows on the tensor
  cores; the K-splits combine through a self-resetting per-tile counter so a
  single launch produces the final bf16 output.
* prefill (M > 16): the same kernel with a [64, 128] output tile, no split-K
  and a pipelined K loop.
* grouped (``bmm_batch_size``): ``x`` is [M, G, K] and the weight is
  [G * N, K]; group ``g`` multiplies its own [N, K] slab. Only the grid and
  the pointer arithmetic change.

Scales are applied per 32-wide K block to the fp32 result of a bf16 tensor
core dot over that block, so the product is numerically the same as a
dequantize-then-matmul reference (every e4m3 value is exact in bf16).
Below SM 8.9 Triton has no ``float8_e4m3fn`` dtype, so the weight is passed
as a ``uint8`` view and decoded with the shared helpers in
:mod:`vllm.triton_utils.fp8_e4m3`.

No host synchronisation, no data-dependent host control flow and a stable
workspace address make every launch CUDA-graph capturable.
"""

from dataclasses import dataclass

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8_e4m3 import e4m3fn_bits_to_bf16, needs_e4m3fn_emulation
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# Largest M served by the decode (split-K) configuration.
DECODE_MAX_M = 16


@dataclass
class TileConfig:
    """Launch configuration; ``split_target`` is the CTA count split-K aims
    for (0 disables split-K). Module-level instances are mutable so a
    benchmark can sweep them in place."""

    block_m: int
    block_n: int
    k_blocks: int  # 32-wide K blocks per loop iteration
    num_warps: int
    num_stages: int
    split_target: int
    max_split: int


# Swept on paper only; the defaults target an A40 (84 SMs, 696 GB/s): ~512
# CTAs of 4 warps, each reading 8 KB of weight per loop iteration.
DECODE_CONFIG = TileConfig(
    block_m=DECODE_MAX_M,
    block_n=64,
    k_blocks=4,
    num_warps=4,
    num_stages=1,
    split_target=512,
    max_split=32,
)
PREFILL_CONFIG = TileConfig(
    block_m=64,
    block_n=128,
    k_blocks=4,
    num_warps=4,
    num_stages=3,
    split_target=0,
    max_split=1,
)


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
def _mxfp8_linear_kernel(
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
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_mg = tl.program_id(2)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    pid_m = pid_mg % num_m_tiles
    g = pid_mg // num_m_tiles

    # Alignment facts the wrapper guarantees, so weight/activation loads
    # vectorize to 16 B regardless of runtime integer specialization.
    K = tl.multiple_of(K, 32)
    k_per_split = tl.multiple_of(k_per_split, K_BLOCKS * 32)
    stride_xm = tl.multiple_of(stride_xm, 8)
    # The group offsets must be hinted as products: a hint on the stride
    # alone does not survive the multiply by the runtime group index.
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
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(bias_ptr + rows, mask=mask_n, other=0.0).to(tl.float32)[
                None, :
            ]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)
    else:
        # Publish this split's partial, then let the last CTA to arrive at
        # this output tile reduce all of them (fixed order, so deterministic).
        part_row = (g * M + offs_m)[:, None] * N + offs_n[None, :]
        tl.store(part_ptr + (pid_k * G * M * N) + part_row, acc, mask=out_mask)
        tl.debug_barrier()
        tile = (pid_mg * tl.num_programs(0)) + pid_n
        prev = tl.atomic_add(cnt_ptr + tile, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            tl.debug_barrier()
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                total += tl.load(
                    part_ptr + (s * G * M * N) + part_row,
                    mask=out_mask,
                    other=0.0,
                    cache_modifier=".cg",
                )
            if HAS_BIAS:
                total += tl.load(bias_ptr + rows, mask=mask_n, other=0.0).to(
                    tl.float32
                )[None, :]
            tl.store(out_ptrs, total.to(out_ptr.dtype.element_ty), mask=out_mask)
            # Leave the counter at zero for the next launch (graph replay).
            tl.atomic_xchg(cnt_ptr + tile, 0, sem="release", scope="gpu")


# Split-K workspace, keyed by (device, stream) so concurrent streams never
# share partials or counters. Grown on demand; vLLM's warmup runs the largest
# decode shapes before CUDA-graph capture, so the address is stable by then.
_WORKSPACE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_workspace(
    device: torch.device, part_elems: int, tiles: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.index or 0, torch.cuda.current_stream(device).stream_id)
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
    """Return (split_k, k_per_split) with k_per_split a multiple of the
    per-iteration K step and split_k * k_per_split >= K."""
    step = cfg.k_blocks * 32
    chunks = triton.cdiv(K, step)
    split = 1
    if cfg.split_target > 0 and tiles < cfg.split_target:
        split = min(cfg.max_split, chunks, triton.cdiv(cfg.split_target, tiles))
        split = max(split, 1)
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

    cfg = DECODE_CONFIG if M <= DECODE_MAX_M else PREFILL_CONFIG
    n_tiles = triton.cdiv(N, cfg.block_n)
    m_tiles = triton.cdiv(M, cfg.block_m)
    tiles = n_tiles * m_tiles * G
    split_k, k_per_split = _pick_split(cfg, tiles, K)

    if split_k > 1:
        part, cnt = _get_workspace(x.device, split_k * G * M * N, tiles)
    else:
        part, cnt = out, out  # unused; keeps the signature uniform

    w_arg = weight.view(torch.uint8) if emulate_e4m3 else weight
    _mxfp8_linear_kernel[(n_tiles, split_k, m_tiles * G)](
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
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        K_BLOCKS=cfg.k_blocks,
        SPLIT_K=split_k,
        HAS_BIAS=bias is not None,
        EMULATE_E4M3=emulate_e4m3,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
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


class TritonMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A16 (and grouped BMM) via the Triton kernel above, for CUDA
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
        if G > 1 or x.ndim == 3:
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
