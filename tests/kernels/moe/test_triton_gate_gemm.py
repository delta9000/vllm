# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M Triton MoE gate GEMM (bf16 x bf16 -> fp32) vs cuBLAS."""

import tempfile

import pytest
import torch

from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.fused_moe.router.triton_gate_gemm import (
    MAX_M,
    triton_gate_gemm,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton gate GEMM is CUDA-only"
)


@pytest.fixture
def tp1():
    """GateLinear reads the TP group and the vLLM config at construction."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import model_parallel_is_initialized

    with set_current_vllm_config(VllmConfig()):
        if not model_parallel_is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
            )
            initialize_model_parallel(tensor_model_parallel_size=1)
        yield


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
def test_gate_linear_uses_triton_tier_at_decode_sizes(tp1):
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    if current_platform.get_device_capability().major >= 9:
        pytest.skip("SM 9.0+ has specialised router kernels")
    # params_dtype must be set explicitly: the tier is bf16-weight only, and
    # GateLinear otherwise takes the default dtype (fp32), not the model's.
    gate = GateLinear(
        5120,
        384,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
    ).cuda()
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
