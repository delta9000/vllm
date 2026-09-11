# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse-MLA attention for DeepSeek V4.1 on NVIDIA SM 8.x.

Ampere has neither FlashMLA-sparse (SM 9.x/10.x) nor FlashInfer sparse MLA
(SM 10.x/12.x). This backend keeps everything the FlashMLA layer does --
metadata, KV gathering, top-k + SWA index merging, the prefill chunk plan,
image-token visibility -- and swaps the two attention kernels for the
portable Triton sparse-MLA kernels. Its o_proj keeps ``wo_a`` in MXFP8 --
through Marlin when the rank holds a single o-group (TP == o_groups) or the
grouped Triton kernel otherwise -- and only falls back to the FlashMLA
layer's bf16 ``torch.bmm`` over a load-time dequantized weight when neither
kernel was selected (``VLLM_DISABLED_KERNELS``).

Those kernels live in ``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py``.
Despite the module name they are plain Triton; its AITER fast paths are
gated on gfx942/gfx950 and never engage on CUDA, where the generic
single-pass decode kernel runs with e4m3 emulated below SM 8.9.
"""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.linear.mxfp8.marlin import (
    MarlinMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.triton_ampere import (
    TritonMxfp8LinearKernel,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4_1.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v4_1.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4_1.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_decode as triton_sparse_attn_decode,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_prefill as triton_sparse_attn_prefill,
)

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


class DeepseekV41TritonMLASparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekSparseSWATritonBackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV41SparseSWAMetadataBuilder]:
        return DeepseekV41SparseSWAMetadataBuilder


class DeepseekV41TritonMLAAttention(DeepseekV4FlashMLAAttention):
    """DeepSeek V4.1 sparse MLA on SM 8.x via Triton kernels."""

    backend_cls = DeepseekV41TritonMLASparseBackend
    swa_backend_cls = DeepseekSparseSWATritonBackend

    # The prefill gather workspace is PREFILL_CHUNK_SIZE full-context rows of
    # bf16 KV, reserved for the whole run: 2.09 GB per rank at 512k with the
    # default 4. The chunk plan packs requests by area, so short concurrent
    # prompts still share a chunk at 1; only several maximum-length prompts at
    # once are split. On 48 GB parts that memory is worth more as KV cache.
    PREFILL_CHUNK_SIZE = 1

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # The Triton kernels mask heads themselves; FlashMLA's 64/128
        # padding would only add work.
        return num_heads

    def _wo_a_keeps_fp8(self) -> bool:
        # Marlin (single group) and Triton (any groups) keep wo_a in MXFP8
        # and take the [T, G, D] input through the layer itself. Emulation
        # dequantizes to bf16 and needs the inherited torch.bmm path. The
        # kernel is (re)selected in process_weights_after_loading, so resolve
        # on first forward rather than in __init__.
        cached = getattr(self, "_wo_a_fp8", None)
        if cached is None:
            kernel = getattr(self.wo_a.quant_method, "kernel", None)
            cached = isinstance(
                kernel, (MarlinMxfp8LinearKernel, TritonMxfp8LinearKernel)
            )
            self._wo_a_fp8 = cached
        return cached

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if not self._wo_a_keeps_fp8():
            return super()._o_proj(o, positions)
        # Inverse RoPE only (no activation quant: W8A16), then wo_a on the
        # [T, G, D] view the op returns (Marlin flattens G == 1 itself; the
        # Triton kernel takes the groups directly).
        o_proj_input, _ = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            quant_group_size=self._o_proj_block_size,
            tma_aligned_scales=False,
            quantize=False,
        )
        z = self.wo_a(o_proj_input)  # [T, G, o_lora_rank]
        return self.wo_b(z.flatten(1))

    def _sparse_prefill(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        triton_sparse_attn_prefill(
            q=q,
            kv=kv,
            indices=indices,
            topk_length=topk_length,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            attn_sink=self.attn_sink,
            output=out,
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # None for SWA-only layers
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            topk_indices, topk_lens = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                attn_metadata.block_size // self.compress_ratio,
                swa_metadata.is_valid_token[:num_decode_tokens],
            )

        triton_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=None,
            swa_ragged_indptr=None,
            topk_ragged_indices=None,
            topk_ragged_indptr=None,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
        )
