# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

# Largest topk_ids.numel() the single-CTA Triton path handles (decode).
SMALL_ALIGN_MAX_NUMEL = 128
SMALL_ALIGN_MAX_EXPERTS = 1024


@triton.jit
def _moe_align_small_kernel(
    topk_ids_ptr,  # [numel] int32/int64 global expert ids
    expert_map_ptr,  # [num_global_experts] int32, or unused
    sorted_ids_ptr,  # [max_padded] int32 out
    expert_ids_ptr,  # [num_blocks] int32 out
    num_post_pad_ptr,  # [1] int32 out
    numel,
    num_experts,
    max_padded,
    num_blocks,
    block_size,
    BLOCK_T: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr,
    MAX_EXPERT_BLOCKS: tl.constexpr,
    HAS_MAP: tl.constexpr,
):
    """One CTA does the whole alignment for a decode-sized batch.

    Same contract as ``_moe_C.moe_align_block_size`` with
    ``ignore_invalid_experts=True``: entries whose expert is not local
    (``expert_map == -1``) are dropped, tokens are grouped by (local) expert
    in stable order, each group is padded to ``block_size`` with the
    sentinel ``numel``, ``expert_ids`` names each block's expert (-1 past the
    end) and ``num_post_pad`` is the padded total. All outputs are
    initialised here, so no host-side fills or scans are needed.
    """
    t = tl.arange(0, BLOCK_T)
    tmask = t < numel
    e = tl.load(topk_ids_ptr + t, mask=tmask, other=-1).to(tl.int32)
    if HAS_MAP:
        e = tl.load(expert_map_ptr + e, mask=tmask & (e >= 0), other=-1).to(tl.int32)
    valid = tmask & (e >= 0) & (e < num_experts)
    e_safe = tl.where(valid, e, 0)

    # Per-expert counts: histogram of the valid entries; the invalid ones were
    # sent to bin 0 and are subtracted back out.
    ex = tl.arange(0, BLOCK_E)
    cnt = tl.histogram(e_safe, BLOCK_E)
    n_invalid = tl.sum(tl.where(valid, 0, 1).to(tl.int32), axis=0)
    cnt = cnt - tl.where(ex == 0, n_invalid, 0)
    padded = ((cnt + block_size - 1) // block_size) * block_size
    offs = tl.cumsum(padded, axis=0) - padded  # exclusive
    total = tl.sum(padded, axis=0)

    # Stable rank of each entry within its expert group.
    same = valid[:, None] & valid[None, :] & (e_safe[:, None] == e_safe[None, :])
    before = t[None, :] < t[:, None]
    rank = tl.sum((same & before).to(tl.int32), axis=1)
    pos = tl.gather(offs, e_safe, 0) + rank

    # Initialise the outputs, then scatter the real entries over them.
    for s0 in range(0, max_padded, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        tl.store(
            sorted_ids_ptr + s,
            tl.full((BLOCK_S,), 0, tl.int32) + numel,
            mask=s < max_padded,
        )
    b = tl.arange(0, BLOCK_S)
    for b0 in range(0, num_blocks, BLOCK_S):
        bb = b0 + b
        tl.store(
            expert_ids_ptr + bb, tl.full((BLOCK_S,), -1, tl.int32), mask=bb < num_blocks
        )
    tl.debug_barrier()
    tl.store(sorted_ids_ptr + pos, t, mask=valid)
    first_block = offs // block_size
    n_expert_blocks = padded // block_size
    for j in tl.static_range(MAX_EXPERT_BLOCKS):
        tl.store(
            expert_ids_ptr + first_block + j,
            ex,
            mask=(j < n_expert_blocks) & (ex < num_experts),
        )
    tl.store(num_post_pad_ptr + tl.arange(0, 1), tl.full((1,), 0, tl.int32) + total)


def _use_small_align(
    topk_ids: torch.Tensor,
    num_experts: int,
    expert_map: torch.Tensor | None,
    ignore_invalid_experts: bool,
) -> bool:
    return (
        envs.VLLM_MOE_ALIGN_SMALL
        and (current_platform.is_cuda() or topk_ids.device.type == "cpu")
        and topk_ids.numel() <= SMALL_ALIGN_MAX_NUMEL
        and 0 < num_experts <= SMALL_ALIGN_MAX_EXPERTS
        and topk_ids.dtype in (torch.int32, torch.int64)
        and (expert_map is None or ignore_invalid_experts)
    )


def moe_align_block_size_small(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    topk_flat = topk_ids.reshape(-1)
    if not topk_flat.is_contiguous():
        topk_flat = topk_flat.contiguous()
    _moe_align_small_kernel[(1,)](
        topk_flat,
        expert_map if expert_map is not None else topk_flat,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        numel,
        num_experts,
        sorted_ids.numel(),
        expert_ids.numel(),
        block_size,
        BLOCK_T=max(16, triton.next_power_of_2(numel)),
        BLOCK_E=max(16, triton.next_power_of_2(num_experts)),
        BLOCK_S=128,
        MAX_EXPERT_BLOCKS=triton.cdiv(numel, block_size),
        HAS_MAP=expert_map is not None,
        num_warps=4,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
        should be padded to a multiple of block_size,
    - ignore_invalid_experts: A flag indicating whether to ignore invalid
        experts. When False, all expert_ids in topk_ids will participate in
        counting and ranking, but invalid experts in expert_ids will be marked
        as -1. When True, all invalid expert_ids in topk_ids will be ignored
        and will not participate in counting or ranking, and there will be no
        -1 in expert_ids.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    if _use_small_align(topk_ids, num_experts, expert_map, ignore_invalid_experts):
        # Decode: one Triton CTA replaces the CUDA op's fills, cub scans and
        # alignment kernel (6 launches per MoE layer) with a single launch.
        moe_align_block_size_small(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Parameters:
    - max_tokens_per_batch (int): Number of tokens in each batch (both
        valid and invalid).
    - block_size (int): block_size to align the data to.
    - expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
        the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.
    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.
    """

    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad
