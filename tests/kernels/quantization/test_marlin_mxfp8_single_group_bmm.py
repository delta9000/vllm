# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marlin for a single-group MXFP8 "BMM" layer.

DeepSeek V4 ``wo_a`` is a ColumnParallel linear with ``bmm_batch_size ==
n_local_groups``; at TP == o_groups that is 1 and the layer is a plain
[T, K] x [N, K] linear. Below Blackwell the selector used to fall through to
the emulation kernel, which dequantizes the weight to bf16 at load (2x the
bytes per decode step). Marlin must now be selected for that case, accept
the [T, 1, K] input the inverse-RoPE op produces, and match the dequantized
reference.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Marlin MXFP8 is CUDA-only"
)

R, K = 1024, 4096  # wo_a per GPU at TP=8: [o_lora_rank, heads_per_group * head_dim]


def _make_layer(n: int, k: int):
    weight_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    weight, weight_scale = _mxfp8_e4m3_quantize_torch(weight_bf16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    return layer


@torch.inference_mode()
def test_marlin_selected_for_single_group_bmm():
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )

    if current_platform.get_device_capability().major >= 10:
        pytest.skip("Blackwell has DeepGEMM for grouped MXFP8")
    assert type(init_mxfp8_linear_kernel(bmm_batch_size=1)) is MarlinMxfp8LinearKernel
    # Real groups are not Marlin's.
    assert type(init_mxfp8_linear_kernel(bmm_batch_size=2)) is not (
        MarlinMxfp8LinearKernel
    )


def _make_layer_uniform_scales(n: int, k: int):
    """The fixture of test_marlin_tile_padding's MXFP8 round trip: e8m0
    exponents in [121, 128), i.e. scales in [2^-6, 2^0]."""
    weight = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 4).to(
        torch.float8_e4m3fn
    )
    scales = torch.randint(121, 128, (n, k // 32), dtype=torch.uint8, device="cuda")
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    return layer


@pytest.mark.parametrize("fixture", ["quantizer", "uniform_scales"])
@pytest.mark.parametrize("num_tokens", [1, 4, 8, 64, 4096])
@torch.inference_mode()
def test_marlin_single_group_matches_dequant_reference(num_tokens, fixture):
    """The [T, 1, K] input through the bmm kernel must equal both the same
    Marlin kernel on the flat [T, K] input (bit for bit) and the dequantized
    reference. Two fixtures: the MXFP8 quantizer (per-block scales) and the
    uniform-scale one Marlin's own padding test uses -- if only ``quantizer``
    fails, the fixture, not the wrapper, is the suspect."""
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    torch.manual_seed(num_tokens)
    make = _make_layer if fixture == "quantizer" else _make_layer_uniform_scales
    layer = make(R, K)
    ref_w = dequant_mxfp8_to_bf16(layer.weight.data, layer.weight_scale.data).float()
    kernel = MarlinMxfp8LinearKernel(Mxfp8LinearLayerConfig(bmm_batch_size=1))
    kernel.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.int32  # Marlin's repacked fp8, not bf16

    x_gtk = torch.randn(1, num_tokens, K, device="cuda", dtype=torch.bfloat16)
    x = x_gtk.transpose(0, 1)  # [T, 1, K] as the inverse-RoPE op returns it
    out = kernel.apply_weights(layer, x)
    assert out.shape == (num_tokens, 1, R)
    assert out.dtype == torch.bfloat16
    # Same kernel, plain 2-D input: the bmm view must change nothing.
    dense = kernel.apply_weights(layer, x_gtk[0].contiguous())
    assert torch.equal(out.view(num_tokens, R), dense), "[T, 1, K] view differs"
    expected = x_gtk[0].float() @ ref_w.T
    rel = (dense.float() - expected).norm() / expected.norm()
    assert rel < 4e-3, f"{fixture}: relative error {rel:.2e}"
