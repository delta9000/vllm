# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 mHC: fused post + delayed pre op, and the Triton pre-norm GEMM.

``mhc_fused_post_pre_delayed_tilelang`` must equal the two-op path
(``mhc_post_tilelang`` then ``mhc_pre_delayed_tilelang``) it replaces, at
decode sizes (fused FMA kernel) and above (post kernel + Triton GEMM), and
both must match the torch references. The CPU test checks the op's
plumbing (argument order, split-K reduction, pre-mix carry) with torch
stand-ins for the TileLang kernels, so it runs without a GPU.
"""

import contextlib
import os

import pytest
import torch

from vllm.model_executor.kernels.mhc.torch import (
    mhc_post_torch,
    mhc_pre_delayed_torch,
)
from vllm.platforms import current_platform


@contextlib.contextmanager
def monkeypatch_env(name, value):
    """vllm.envs re-reads os.environ on every attribute access."""
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            del os.environ[name]
        else:
            os.environ[name] = prev

HC, HIDDEN = 4, 5120
MIX = HC * (HC + 2)
EPS = 1e-6
POST_ALPHA = 2.0
SINKHORN = 20


def _inputs(num_tokens: int, device: str, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(dtype).to(device)

    residual = rn(num_tokens, HC, HIDDEN, dtype=torch.bfloat16)
    x = rn(num_tokens, HIDDEN, dtype=torch.bfloat16)
    post_mix = torch.sigmoid(rn(num_tokens, HC)).to(device) * POST_ALPHA
    comb = torch.softmax(rn(num_tokens, HC, HC), dim=-1).to(device)
    fn = rn(MIX, HC * HIDDEN, scale=0.01)
    hc_scale = torch.tensor([1.0, 1.0, 1.0], device=device)
    hc_base = rn(MIX, scale=0.1)
    pre_mix = torch.sigmoid(rn(num_tokens, HC)).to(device) + EPS
    norm_w = (1.0 + rn(HIDDEN, scale=0.1)).to(torch.bfloat16)
    return (
        residual,
        x,
        post_mix.unsqueeze(-1),
        comb,
        fn,
        hc_scale,
        hc_base,
        pre_mix,
        norm_w,
    )


def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * w.float()).to(
        torch.bfloat16
    )


def _reference(residual, x, post_mix, comb, fn, hc_scale, hc_base, pre_mix, norm_w):
    res_cur = mhc_post_torch(x, residual, post_mix, comb)
    post, comb2, layer_in, nxt = mhc_pre_delayed_torch(
        res_cur, fn, hc_scale, hc_base, EPS, EPS, EPS, POST_ALPHA, SINKHORN, pre_mix
    )
    return res_cur, post, comb2, _rmsnorm(layer_in, norm_w), nxt


def _check(got, ref, tag: str):
    names = ("residual", "post", "comb", "layer_input", "next_pre_mix")
    for name, g, r in zip(names, got, ref):
        assert g.shape == r.shape, f"{tag} {name}: {g.shape} vs {r.shape}"
        assert g.dtype == r.dtype, f"{tag} {name}: {g.dtype} vs {r.dtype}"
        torch.testing.assert_close(
            g.float(), r.float(), rtol=2e-2, atol=2e-2, msg=f"{tag} {name}"
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="TileLang mHC is CUDA-only")
@pytest.mark.parametrize("num_tokens", [1, 6, 8, 16, 17, 64, 200, 4096])
@torch.inference_mode()
def test_fused_matches_split_path_and_reference(num_tokens):
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_delayed_tilelang,
        mhc_post_tilelang,
        mhc_pre_delayed_tilelang,
    )

    inp = _inputs(num_tokens, "cuda", seed=num_tokens)
    residual, x, post_mix, comb, fn, hc_scale, hc_base, pre_mix, norm_w = inp
    fused = mhc_fused_post_pre_delayed_tilelang(
        x,
        residual,
        post_mix,
        comb,
        fn,
        hc_scale,
        hc_base,
        EPS,
        EPS,
        EPS,
        POST_ALPHA,
        SINKHORN,
        pre_mix,
        norm_w,
        EPS,
    )
    res_cur = mhc_post_tilelang(x, residual, post_mix, comb)
    split = (res_cur,) + tuple(
        mhc_pre_delayed_tilelang(
            res_cur,
            fn,
            hc_scale,
            hc_base,
            EPS,
            EPS,
            EPS,
            POST_ALPHA,
            SINKHORN,
            pre_mix=pre_mix,
            norm_weight=norm_w,
            norm_eps=EPS,
        )
    )
    _check(fused, split, "fused vs split")
    _check(fused, _reference(*inp), "fused vs torch")
    # The fused path must be tighter to the split path than to the reference
    # (same kernels, only summation order differs).
    for g, s in zip(fused, split):
        assert (g.float() - s.float()).abs().max() <= 1e-2 * (1 + s.float().abs().max())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Triton GEMM is CUDA-only")
@pytest.mark.parametrize("num_tokens", [17, 64, 100, 1024, 4096])
@torch.inference_mode()
def test_triton_prenorm_gemm(num_tokens):
    from vllm.model_executor.kernels.mhc.tilelang import _tilelang_hc_prenorm_gemm
    from vllm.model_executor.kernels.mhc.triton_prenorm_gemm import (
        SPLIT_K,
        hc_prenorm_gemm_triton,
    )

    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, HC * HIDDEN, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(MIX, HC * HIDDEN, device="cuda") * 0.01
    out = torch.empty(SPLIT_K, num_tokens, MIX, device="cuda")
    sq = torch.empty(SPLIT_K, num_tokens, device="cuda")
    hc_prenorm_gemm_triton(x, fn, out, sq)
    ref = x.float() @ fn.T
    ref_sq = x.float().square().sum(-1)
    torch.testing.assert_close(out.sum(0), ref, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(sq.sum(0), ref_sq, rtol=1e-5, atol=1e-2)
    # The dispatcher reaches this kernel only under VLLM_MHC_TRITON_PRENORM,
    # which is off by default (it loses to TileLang at every measured token
    # count on A40). With it on, tokens > 16 route here bit-for-bit.
    with monkeypatch_env("VLLM_MHC_TRITON_PRENORM", "1"):
        out2 = torch.empty(SPLIT_K, num_tokens, MIX, device="cuda")
        sq2 = torch.empty(SPLIT_K, num_tokens, device="cuda")
        _tilelang_hc_prenorm_gemm(x, fn, out2, sq2, HC * HIDDEN, 1, n_splits=SPLIT_K)
        assert torch.equal(out, out2) and torch.equal(sq, sq2)
    # Off (the default) it must not: the TileLang kernels fill one split.
    out3 = torch.empty(1, num_tokens, MIX, device="cuda")
    sq3 = torch.empty(1, num_tokens, device="cuda")
    _tilelang_hc_prenorm_gemm(x, fn, out3, sq3, HC * HIDDEN, 1, n_splits=1)
    torch.testing.assert_close(out3[0], ref, rtol=1e-4, atol=1e-3)


@torch.inference_mode()
def test_fused_op_plumbing_cpu(monkeypatch):
    """Runs the op with torch stand-ins for the three TileLang kernels: the
    fused FMA kernel (post + projection + sqrsum over split_k), the post
    kernel, and the coefficient/collapse kernel with the delayed pre-mix."""
    import vllm.model_executor.kernels.mhc.tilelang as tl_mod
    import vllm.model_executor.kernels.mhc.tilelang_kernels as tk
    import vllm.model_executor.kernels.mhc.warmup as wu

    def fake_fused(
        comb_mix,
        residual_in,
        post_mix,
        x_in,
        weight_t,
        yp_out,
        rp_out,
        residual_out,
        hc,
        hidden,
        n_out,
        tile_n=1,
        split_k=1,
    ):
        assert yp_out.shape == (split_k, residual_in.shape[0], n_out)
        new_r = mhc_post_torch(x_in, residual_in, post_mix.unsqueeze(-1), comb_mix)
        residual_out.copy_(new_r)
        flat = new_r.float().view(new_r.shape[0], -1)
        w = weight_t.reshape(n_out, -1)
        chunk = flat.shape[1] // split_k
        for s in range(split_k):
            sl = slice(s * chunk, (s + 1) * chunk)
            yp_out[s].copy_(flat[:, sl] @ w[:, sl].T)
            rp_out[s].copy_(flat[:, sl].square().sum(-1))

    def fake_post(a, b, c, d, out, hc, hidden):
        out.copy_(mhc_post_torch(d, b, c.unsqueeze(-1), a))

    def fake_prenorm(
        mixes,
        sqrsum,
        hc_scale,
        hc_base,
        residual,
        post,
        comb,
        layer_input,
        norm_weight,
        pre_mix_in,
        pre_mix_out,
        *,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        hc_mult,
        use_pre_mix_in,
        save_pre_mix,
        rms_numel,
    ):
        assert use_pre_mix_in and save_pre_mix and rms_numel == hc_mult * hidden_size
        mx = mixes.sum(0) * torch.rsqrt(sqrsum.sum(0) / rms_numel + rms_eps)[:, None]
        pre_mix_out.copy_(
            torch.sigmoid(mx[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
            + hc_pre_eps
        )
        post.copy_(
            torch.sigmoid(
                mx[:, hc_mult : 2 * hc_mult] * hc_scale[1]
                + hc_base[hc_mult : 2 * hc_mult]
            )
            * hc_post_mult_value
        )
        c = mx[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult) * hc_scale[2] + hc_base[
            2 * hc_mult :
        ].view(1, hc_mult, hc_mult)
        c = torch.softmax(c, -1) + hc_sinkhorn_eps
        c = c / (c.sum(-2, keepdim=True) + hc_sinkhorn_eps)
        for _ in range(sinkhorn_repeat - 1):
            c = c / (c.sum(-1, keepdim=True) + hc_sinkhorn_eps)
            c = c / (c.sum(-2, keepdim=True) + hc_sinkhorn_eps)
        comb.copy_(c.reshape(-1, hc_mult * hc_mult))
        collapsed = (
            (pre_mix_in.unsqueeze(-1) * residual.float()).sum(1).to(torch.bfloat16)
        )
        layer_input.copy_(_rmsnorm(collapsed, norm_weight))

    monkeypatch.setattr(tk, "mhc_fused_tilelang", fake_fused)
    monkeypatch.setattr(tk, "mhc_post_tilelang", fake_post)
    monkeypatch.setattr(wu.MHC_PRE_NORM_KERNEL, "__call__", fake_prenorm, raising=False)
    monkeypatch.setattr(
        type(wu.MHC_PRE_NORM_KERNEL),
        "__call__",
        lambda self, *a, **k: fake_prenorm(*a, **k),
    )

    def fake_triton_gemm(x, fn, out, sq):
        chunk = -(-x.shape[1] // out.shape[0])
        for s in range(out.shape[0]):
            sl = slice(s * chunk, (s + 1) * chunk)
            out[s].copy_(x[:, sl].float() @ fn[:, sl].T)
            sq[s].copy_(x[:, sl].float().square().sum(-1))

    import vllm.model_executor.kernels.mhc.triton_prenorm_gemm as pg

    monkeypatch.setattr(pg, "hc_prenorm_gemm_triton", fake_triton_gemm)
    monkeypatch.setattr(tl_mod, "TRITON_PRENORM_MIN_TOKENS", 16)
    for num_tokens in (1, 6, 16, 17, 40):
        inp = _inputs(num_tokens, "cpu", seed=num_tokens)
        residual, x, post_mix, comb, fn, hc_scale, hc_base, pre_mix, norm_w = inp
        # x.device.type == "cuda" gates the Triton GEMM inside
        # _tilelang_hc_prenorm_gemm; on CPU call the stand-in directly.
        monkeypatch.setattr(
            tl_mod,
            "_tilelang_hc_prenorm_gemm",
            lambda x_, fn_, out, sq, *a, n_splits=1, **k: fake_triton_gemm(
                x_, fn_, out, sq
            ),
        )
        got = tl_mod.mhc_fused_post_pre_delayed_tilelang(
            x,
            residual,
            post_mix,
            comb,
            fn,
            hc_scale,
            hc_base,
            EPS,
            EPS,
            EPS,
            POST_ALPHA,
            SINKHORN,
            pre_mix,
            norm_w,
            EPS,
        )
        _check(got, _reference(*inp), f"cpu T={num_tokens}")
