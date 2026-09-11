# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton MXFP8 W8A16 linear for CUDA below SM 9.0 (``TritonMxfp8LinearKernel``).

Every case compares against dequantize-then-matmul in fp32. The Triton
kernels accumulate exact products of e4m3 values and power-of-two scales in
fp32, so the only differences are summation order and the final bf16
rounding of the output.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton MXFP8 linear is CUDA-only"
)

# DeepSeek V4.1-Flash dense linears per GPU at TP=8 (N, K).
V41_TP8_SHAPES = [
    ("fused_wqa_wkv", 1280 + 512, 5120),
    ("wq_b", 32768 // 8, 1280),
    ("wo_b", 5120, 8192 // 8),
    ("shared_w13", 2 * 2304 // 8, 5120),
    ("shared_w2", 5120, 2304 // 8),
]
# wo_a per GPU: [n_local_groups * o_lora_rank, heads_per_group * head_dim].
V41_WO_A = [("wo_a_tp8", 1, 1024, 4096), ("wo_a_tp4", 2, 1024, 4096)]
# M=1..4 run the register-FMA GEMV, 5..16 the decode mma tile, >16 prefill.
DECODE_MS = [1, 2, 3, 4, 5, 8, 16, 17]


def _kernel_cls():
    from vllm.model_executor.kernels.linear.mxfp8.triton_ampere import (
        TritonMxfp8LinearKernel,
    )

    return TritonMxfp8LinearKernel


def _make_layer(n: int, k: int, device: str, bias: bool = False):
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
    weight, weight_scale = _mxfp8_e4m3_quantize_torch(weight_bf16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.bias = (
        torch.nn.Parameter(
            torch.randn(n, device=device, dtype=torch.bfloat16), requires_grad=False
        )
        if bias
        else None
    )
    return layer


def _make_kernel(groups: int | None, layer, cls=None):
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    kernel = (cls or _kernel_cls())(Mxfp8LinearLayerConfig(bmm_batch_size=groups))
    kernel.process_weights_after_loading(layer)
    return kernel


def _reference(x: torch.Tensor, weight, weight_scale, bias=None) -> torch.Tensor:
    w = dequant_mxfp8_to_bf16(weight, weight_scale).float()
    out = x.float() @ w.T
    if bias is not None:
        out = out + bias.float()
    return out


def _check(out: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all(), f"{name}: non-finite output"
    # float64 norms: in the extreme-magnitude case the ~1e-30 values square
    # to ~1e-60, which underflows fp32, so the reference norm comes out as
    # exactly 0 and the ratio is NaN even when the kernel is correct.
    rel = (out.double() - expected.double()).norm() / expected.double().norm()
    assert rel < 4e-3, f"{name}: relative error {rel:.2e}"
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("name,n,k", V41_TP8_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("num_tokens", DECODE_MS + [64, 4096])
@torch.inference_mode()
def test_matches_dequant_reference(name, n, k, num_tokens):
    torch.manual_seed(num_tokens)
    layer = _make_layer(n, k, "cuda")
    ref_w, ref_s = layer.weight.data.clone(), layer.weight_scale.data.clone()
    kernel = _make_kernel(None, layer)
    assert layer.weight.dtype == torch.float8_e4m3fn  # checkpoint layout kept
    x = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x)
    assert out.shape == (num_tokens, n)
    _check(out, _reference(x, ref_w, ref_s), name)


@pytest.mark.parametrize("num_tokens", [1, 3, 8, 64])
@torch.inference_mode()
def test_bias_and_leading_dims(num_tokens):
    torch.manual_seed(0)
    n, k = 576, 288  # shared_w2-like: K not a multiple of the 128-wide K step
    layer = _make_layer(n, k, "cuda", bias=True)
    ref_w, ref_s = layer.weight.data.clone(), layer.weight_scale.data.clone()
    kernel = _make_kernel(None, layer)
    x = torch.randn(2, num_tokens, k, device="cuda", dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x, layer.bias)
    assert out.shape == (2, num_tokens, n)
    _check(out.view(-1, n), _reference(x.view(-1, k), ref_w, ref_s, layer.bias), "bias")


@pytest.mark.parametrize("name,groups,r,k", V41_WO_A, ids=lambda v: str(v))
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 64])
@torch.inference_mode()
def test_grouped_matches_bmm_reference(name, groups, r, k, num_tokens):
    """wo_a on the Triton kernel: ``x`` [T, G, K] against weight [G * R, K]
    -> [T, G, R], with the transposed [G, T, K] input layout the
    inverse-RoPE op produces. G == 1 takes the flattened path."""
    torch.manual_seed(num_tokens)
    layer = _make_layer(groups * r, k, "cuda")
    ref_w, ref_s = layer.weight.data.clone(), layer.weight_scale.data.clone()
    kernel = _make_kernel(groups, layer)
    x_gtk = torch.randn(groups, num_tokens, k, device="cuda", dtype=torch.bfloat16)
    x = x_gtk.transpose(0, 1)  # [T, G, K], non-contiguous like the real input
    out = kernel.apply_weights(layer, x)
    assert out.shape == (num_tokens, groups, r)
    w = dequant_mxfp8_to_bf16(ref_w, ref_s).float()
    expected = torch.bmm(x_gtk.float(), w.view(groups, r, k).transpose(1, 2))
    _check(out.transpose(0, 1).reshape(-1, r), expected.reshape(-1, r), name)


@torch.inference_mode()
def test_split_k_counters_reset_and_graph_replay():
    """The split-K fixup counters must return to zero after every launch so
    repeated launches -- and CUDA-graph replays -- give identical results,
    on both the fma (M=1) and the mma (M=8) kernels."""
    from vllm.model_executor.kernels.linear.mxfp8 import triton_ampere as ta

    torch.manual_seed(1)
    n, k = 576, 5120  # shared_w13: few N tiles, so K is split
    layer = _make_layer(n, k, "cuda")
    ref_w, ref_s = layer.weight.data.clone(), layer.weight_scale.data.clone()
    kernel = _make_kernel(None, layer)
    for m in (1, 8):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        first = kernel.apply_weights(layer, x).clone()
        for _ in range(3):
            assert torch.equal(kernel.apply_weights(layer, x), first)
        key = (x.device.index or 0, torch.cuda.current_stream(x.device).stream_id)
        _, counters = ta._WORKSPACE[key]
        torch.cuda.synchronize()
        assert int(counters.abs().sum()) == 0

        static_x = x.clone()
        static_out = kernel.apply_weights(layer, static_x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = kernel.apply_weights(layer, static_x)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(static_out, first)
        static_x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        _check(static_out, _reference(static_x, ref_w, ref_s), f"graph replay M={m}")


@torch.inference_mode()
def test_config_overrides_all_variants():
    """Every sweep-able configuration knob must give the same numbers."""
    from vllm.model_executor.kernels.linear.mxfp8 import triton_ampere as ta

    torch.manual_seed(2)
    n, k = 4096, 1280
    layer = _make_layer(n, k, "cuda")
    ref_w, ref_s = layer.weight.data.clone(), layer.weight_scale.data.clone()
    kernel = _make_kernel(None, layer)
    saved = dict(ta.DECODE_CONFIGS)
    try:
        for m in (1, 4, 8):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = _reference(x, ref_w, ref_s)
            bucket = ta._m_bucket(m)
            variant = "fma" if m <= ta.FMA_MAX_M else "mma"
            for block_n, k_blocks, prefetch, fill in (
                (16, 2, True, 0),
                (32, 4, False, 84),
                (64, 2, True, 336),
                (32, 8, True, 168),
            ):
                if variant == "mma" and block_n < 32:
                    continue
                ta.set_decode_config(
                    bucket,
                    n,
                    k,
                    variant=variant,
                    block_n=block_n,
                    k_blocks=k_blocks,
                    prefetch=prefetch,
                    fill_target=fill,
                    min_iters=4,
                    block_m=16,
                )
                _check(kernel.apply_weights(layer, x), ref, f"M={m} cfg={block_n}")
    finally:
        ta.DECODE_CONFIGS.clear()
        ta.DECODE_CONFIGS.update(saved)


@torch.inference_mode()
def test_extreme_magnitudes_stay_finite():
    torch.manual_seed(0)
    n, k = 5120, 1024
    layer = _make_layer(n, k, "cuda")
    ref_w, ref_s = layer.weight.data.clone(), layer.weight_scale.data.clone()
    kernel = _make_kernel(None, layer)
    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    x[0] = 0.0
    x[1] = 1e-30
    x[2] = 400.0
    x[3, ::2] = 0.0
    out = kernel.apply_weights(layer, x)
    assert torch.isfinite(out).all()
    assert (out[0] == 0).all()
    _check(out, _reference(x, ref_w, ref_s), "extreme")
    out = kernel.apply_weights(layer, x[:2])  # fma path
    assert (out[0] == 0).all()
    _check(out, _reference(x[:2], ref_w, ref_s), "extreme fma")


@torch.inference_mode()
def test_kernel_selected_on_ampere():
    """On SM 8.x the selector must pick the Triton kernel ahead of Marlin
    for dense layers, Marlin for a single-group BMM and Triton for real
    groups."""
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )

    if current_platform.get_device_capability().major != 8:
        pytest.skip("selection test is for SM 8.x")
    assert type(init_mxfp8_linear_kernel()) is _kernel_cls()
    assert type(init_mxfp8_linear_kernel(bmm_batch_size=1)) is MarlinMxfp8LinearKernel
    assert type(init_mxfp8_linear_kernel(bmm_batch_size=2)) is _kernel_cls()
