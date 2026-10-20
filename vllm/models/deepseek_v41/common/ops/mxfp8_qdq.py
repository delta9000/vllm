# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-place MXFP8 quantize-dequantize of bf16 activations.

The 1017 reference rounds the attention output to e4m3 with one ue8m0 scale
per 32 elements before wo_a (``act_quant(o, 32, "ue8m0", ..., inplace=True)``,
inference/model.py:787 and :1073). On SM 8.x wo_a runs W8A16 on bf16 input, so
this reproduces only the rounding, not an FP8 GEMM. The e4m3 conversion goes
through integer bit patterns, so it compiles below SM 8.9.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8_e4m3 import (
    ceil_log2,
    e4m3fn_bits_to_fp32,
    fp32_to_e4m3fn_bits,
)

MXFP8_GROUP_SIZE = 32


@triton.jit
def _mxfp8_qdq_kernel(
    x_ptr,
    num_groups,
    GROUP: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
    cols = tl.arange(0, GROUP)
    offs = rows[:, None] * GROUP + cols[None, :]
    mask = rows[:, None] < num_groups
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Reference: amax floored at 1e-4, scale rounded up to a power of two.
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.math.exp2(ceil_log2(amax * (1.0 / 448.0)))
    q = fp32_to_e4m3fn_bits(x / scale[:, None])
    y = e4m3fn_bits_to_fp32(q) * scale[:, None]
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


def mxfp8_qdq_(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to MXFP8 (e4m3, ue8m0 per 32 along its storage) in place.

    ``x`` must be dense in memory (any permutation of a contiguous tensor)
    with its last dimension contiguous and a multiple of 32, so consecutive
    storage groups of 32 are groups of 32 along the last dimension.
    """
    assert x.dtype == torch.bfloat16, x.dtype
    assert x.stride(-1) == 1 and x.shape[-1] % MXFP8_GROUP_SIZE == 0, x.shape
    base = x
    if not base.is_contiguous():
        base = x.transpose(0, 1)
    assert base.is_contiguous(), "mxfp8_qdq_ needs dense storage"
    num_groups = x.numel() // MXFP8_GROUP_SIZE
    if num_groups == 0:
        return x
    groups_per_program = 16
    grid = (triton.cdiv(num_groups, groups_per_program),)
    _mxfp8_qdq_kernel[grid](
        base,
        num_groups,
        GROUP=MXFP8_GROUP_SIZE,
        GROUPS_PER_PROGRAM=groups_per_program,
    )
    return x
