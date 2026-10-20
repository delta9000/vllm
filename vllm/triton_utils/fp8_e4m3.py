# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OCP e4m3fn in Triton on GPUs whose Triton backend has no fp8 dtype.

Triton's NVIDIA backend enables ``fp8e4nv`` only at compute capability >= 8.9,
so on Ampere any kernel that loads, stores or bitcasts ``float8_e4m3fn`` fails
at JIT time. The helpers here move the conversion into integer arithmetic on
``uint8`` bit patterns: callers pass the fp8 tensor as ``.view(torch.uint8)``
and select the emulated path with a ``tl.constexpr`` flag set from
:func:`needs_e4m3fn_emulation`, so native platforms compile exactly the code
they did before.

All three conversions are bitwise-identical to ``Tensor.to()`` over every
e4m3fn code, including signed zero and NaN (see
``tests/kernels/test_triton_fp8_e4m3.py``). Signs are applied by OR-ing the
integer bit pattern, never by float negation: under fast-math ``-(+0.0)``
folds back to ``+0.0``, which loses the sign of byte ``0x80`` in a way
``torch.equal`` cannot detect.
"""

import functools

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@functools.cache
def needs_e4m3fn_emulation() -> bool:
    """Whether Triton kernels must emulate ``float8_e4m3fn`` on this device.

    True on NVIDIA GPUs below SM 8.9 (``supports_fp8()`` is exactly the
    capability at which Triton enables ``fp8e4nv``). ROCm and SM 8.9+ keep the
    native dtype.
    """
    return current_platform.is_cuda() and not current_platform.supports_fp8()


@triton.jit
def e4m3fn_bits_to_fp32(u):
    """Decode e4m3fn bit patterns (``uint8``) to ``float32``.

    Normal codes are ``(-1)^s * 2^(e-7) * (1 + m/8)``: rebias the exponent to
    fp32 (``e + 120``) and place the mantissa at bit 20. Subnormals
    (``e == 0``) are ``m * 2^-9``. ``0x7F``/``0xFF`` are NaN; e4m3fn has no
    infinities.
    """
    bits = u.to(tl.int32)
    e = (bits >> 3) & 0xF
    m = bits & 0x7
    normal = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    mag = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)
    mag = tl.where((bits & 0x7F) == 0x7F, float("nan"), mag)
    signed = mag.to(tl.int32, bitcast=True) | ((bits & 0x80) << 24)
    return signed.to(tl.float32, bitcast=True)


@triton.jit
def e4m3fn_bits_to_bf16(u):
    """Decode e4m3fn bit patterns (``uint8``) to ``bfloat16``, cheaply.

    Cheaper than :func:`e4m3fn_bits_to_fp32` for inner loops: the 7 magnitude
    bits are placed where fp16 keeps its low exponent and top mantissa bits,
    so ``2^(e-7) * (1 + m/8)`` becomes the fp16 value ``2^(e-15) * (1 + m/8)``
    and one exact ``x256`` restores it. fp16 subnormals line up with e4m3
    subnormals under the same shift. Every e4m3fn value is exact in bf16.
    """
    u16 = u.to(tl.int16)
    h = (((u16 & 0x7F) << 7) | ((u16 & 0x80) << 8)).to(tl.float16, bitcast=True)
    f = (h * 256.0).to(tl.bfloat16)
    return tl.where((u16 & 0x7F) == 0x7F, float("nan"), f)


@triton.jit
def fp32_to_e4m3fn_bits(x):
    """Encode ``float32`` to e4m3fn bit patterns (``uint8``).

    Round-to-nearest-even, saturating to +-448 like
    ``cvt.rn.satfinite.e4m3x2.f32`` and ``Tensor.to(float8_e4m3fn)``; NaN
    encodes as ``0x7F``.
    """
    sign = (x.to(tl.int32, bitcast=True) >> 31) & 1
    ax = tl.minimum(tl.abs(x), 448.0)
    # Normal range: round the fp32 mantissa to 3 bits with an integer add (the
    # carry propagates into the exponent), then rebias 127 -> 7.
    ab = ax.to(tl.int32, bitcast=True)
    norm_bits = ((ab + 0x7FFFF + ((ab >> 20) & 1)) >> 20) - (120 << 3)
    # Subnormal range (ax < 2^-6): q = RNE(ax * 2^9) in 0..8, where q == 8 is
    # exactly the smallest normal (bits 0x08). Adding 2^23 makes the fp32
    # adder do the rounding.
    sub_bits = ((ax * 512.0 + 8388608.0) - 8388608.0).to(tl.int32)
    bits = tl.where(ax < 0.015625, sub_bits, norm_bits)
    bits = tl.where(x != x, 0x7F, bits)
    return (bits | (sign << 7)).to(tl.uint8)


@triton.jit
def ceil_log2(x):
    """Exact ``ceil(log2(x))`` as float32, for positive normal float32 ``x``.

    Block-scale exponents (UE8M0, power-of-two scales) need it exact.
    ``tl.ceil(tl.log2(x))`` lowers to libdevice's ``log2f``, which returns k
    instead of k + 1 for inputs a few ulps above ``2^k``: the scale comes out 2x
    too small and the block's largest values saturate. Reading the exponent
    field is exact, matches the reference's ``fast_round_scale``, and is
    cheaper. Callers floor ``x`` above the subnormal range.
    """
    bits = x.to(tl.int32, bitcast=True)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return e.to(tl.float32)
