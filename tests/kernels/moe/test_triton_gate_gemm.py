# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M Triton MoE gate GEMM (bf16 x bf16 -> fp32) vs cuBLAS."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.triton_gate_gemm import (
    MAX_M,
    triton_gate_gemm,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton gate GEMM is CUDA-only"
)


@pytest.mark.parametrize("num_tokens", [1, 2, 6, 8, 16])
@pytest.mark.parametrize("hidden,num_experts", [(5120, 384), (1024, 128), (4096, 256)])
@torch.inference_mode()
def test_gate_gemm_matches_cublas(num_tokens, hidden, num_experts):
    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(num_experts, hidden, device="cuda", dtype=torch.bfloat16) * 0.05
    out = triton_gate_gemm(x, w)
    ref = torch.mm(x, w.T, out_dtype=torch.float32)
    exact = x.float() @ w.float().T
    assert out.dtype == torch.float32 and out.shape == (num_tokens, num_experts)
    torch.testing.assert_close(out, exact, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-2)
    # Split-K counters reset: repeated calls are bit-identical.
    assert torch.equal(triton_gate_gemm(x, w), out)


@torch.inference_mode()
def test_gate_linear_uses_triton_tier_at_decode_sizes():
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    if current_platform.get_device_capability().major >= 9:
        pytest.skip("SM 9.0+ has specialised router kernels")
    gate = GateLinear(5120, 384, bias=False, out_dtype=torch.float32).cuda()
    gate.weight.data.normal_(0, 0.05)
    assert gate.allow_triton_small_m_gemm
    x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
    out, _ = gate(x)
    torch.testing.assert_close(
        out, x.float() @ gate.weight.float().T, rtol=1e-4, atol=1e-3
    )
    # Above MAX_M the cuBLAS tier still serves.
    x_big = torch.randn(MAX_M + 1, 5120, device="cuda", dtype=torch.bfloat16)
    out_big, _ = gate(x_big)
    torch.testing.assert_close(
        out_big, x_big.float() @ gate.weight.float().T, rtol=1e-3, atol=1e-2
    )


@torch.inference_mode()
def test_gate_gemm_graph_replay():
    torch.manual_seed(0)
    x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(384, 5120, device="cuda", dtype=torch.bfloat16) * 0.05
    out = triton_gate_gemm(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = triton_gate_gemm(x, w)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x.float() @ w.float().T, rtol=1e-4, atol=1e-3)
