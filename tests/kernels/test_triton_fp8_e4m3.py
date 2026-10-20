# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise checks for the integer e4m3fn helpers in ``vllm.triton_utils``.

The helpers only use integer and fp16/fp32 arithmetic, so they run (and are
checked) on every CUDA GPU, not just the ones that need them.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8_e4m3 import (
    ceil_log2,
    e4m3fn_bits_to_bf16,
    e4m3fn_bits_to_fp32,
    fp32_to_e4m3fn_bits,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="OCP e4m3fn emulation is CUDA-only"
)


_BLOCK = 1024


@triton.jit
def _decode_fp32_kernel(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask, other=0)
    tl.store(dst + offs, e4m3fn_bits_to_fp32(x), mask=mask)


@triton.jit
def _decode_bf16_kernel(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask, other=0)
    tl.store(dst + offs, e4m3fn_bits_to_bf16(x), mask=mask)


@triton.jit
def _encode_kernel(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask, other=0.0)
    tl.store(dst + offs, fp32_to_e4m3fn_bits(x), mask=mask)


@triton.jit
def _ceil_log2_kernel(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(src + offs, mask=mask, other=1.0)
    tl.store(dst + offs, ceil_log2(x), mask=mask)


def _launch(kernel, src: torch.Tensor, dst: torch.Tensor) -> None:
    n = src.numel()
    kernel[(triton.cdiv(n, _BLOCK),)](src, dst, n, BLOCK=_BLOCK)


def _all_codes() -> torch.Tensor:
    return torch.arange(256, dtype=torch.uint8, device="cuda")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality, NaN included; ``torch.equal`` misses -0.0 vs +0.0."""
    int_dtype = {4: torch.int32, 2: torch.int16}[a.element_size()]
    nan = torch.isnan(a) & torch.isnan(b)
    return bool(((a.view(int_dtype) == b.view(int_dtype)) | nan).all())


@pytest.mark.parametrize(
    "kernel,dtype",
    [(_decode_fp32_kernel, torch.float32), (_decode_bf16_kernel, torch.bfloat16)],
)
def test_decode_matches_torch_on_every_code(kernel, dtype):
    codes = _all_codes()
    out = torch.empty(256, dtype=dtype, device="cuda")
    _launch(kernel, codes, out)
    ref = codes.view(torch.float8_e4m3fn).to(dtype)
    assert _same_bits(out, ref)
    # Signed zero is the case a float negation would get wrong.
    assert torch.signbit(out[0x80]) and not torch.signbit(out[0x00])


def test_encode_round_trips_every_code():
    codes = _all_codes()
    values = codes.view(torch.float8_e4m3fn).to(torch.float32)
    out = torch.empty(256, dtype=torch.uint8, device="cuda")
    _launch(_encode_kernel, values, out)
    finite = ~torch.isnan(values)
    assert torch.equal(out[finite], codes[finite])
    assert bool((out[~finite] & 0x7F == 0x7F).all())


def test_encode_matches_torch_rounding_and_saturation():
    gen = torch.Generator(device="cuda").manual_seed(0)
    n = 1 << 16
    values = torch.cat(
        [
            torch.randn(n, device="cuda", generator=gen) * 30,
            torch.randn(n, device="cuda", generator=gen) * 1e-2,  # subnormals
            torch.randn(n, device="cuda", generator=gen) * 1e-3,
            # Ties and the edges of both ranges.
            torch.tensor(
                [
                    0.0,
                    -0.0,
                    448.0,
                    -448.0,
                    464.0,
                    1e6,
                    -1e6,
                    2.0**-6,
                    2.0**-9,
                    2.0**-10,
                    3 * 2.0**-10,
                    1.0625,
                    1.1875,
                    240.0,
                    248.0,
                ],
                device="cuda",
            ),
        ]
    )
    out = torch.empty(values.numel(), dtype=torch.uint8, device="cuda")
    _launch(_encode_kernel, values, out)
    # torch saturates out-of-range values only after clamping; do the same.
    ref = values.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(out, ref)


def test_ceil_log2_exact_near_powers_of_two():
    """Every float32 from 0 to 2000 ulps above 2^k, k in [-126, 127]: the band
    where tl.ceil(tl.log2(x)) returns k instead of k + 1."""
    k = torch.arange(-126, 128, dtype=torch.int64)
    ulps = torch.arange(0, 2000, dtype=torch.int64)
    bits = ((k[:, None] + 127) << 23) + ulps[None, :]
    src = bits.to(torch.int32).flatten().view(torch.float32).cuda()
    dst = torch.empty_like(src)
    _launch(_ceil_log2_kernel, src, dst)
    # ulp 0 is exactly 2^k; anything above rounds up to k + 1.
    want = (k[:, None] + (ulps[None, :] > 0)).flatten().to(torch.float32).cuda()
    assert torch.equal(dst, want)
