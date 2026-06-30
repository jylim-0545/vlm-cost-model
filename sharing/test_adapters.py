"""GPU-free unit checks for the model-agnostic pieces of sharing.

Two groups, both runnable without a model or GPU:
  - sharing.adapters : ridge closed-form recovers a known linear map; z-score stats;
                       identity raw-inject passthrough; MLP shapes; save/load round-trip.
  - sharing.cost     : encode-share FLOP saving is monotone in N and -> 1-1/N; storage
                       sharing saves for N>=2; break-even is finite & non-negative.

Run:  python -m sharing.test_adapters
Needs torch (sharing.adapters) + PyYAML (sharing.cost) — both in the repo env.
"""
from __future__ import annotations

import math
import tempfile

import torch

from sharing.adapters import (RidgeAffine, ZScoreMLP, build_adapter, fit_ridge,
                              load_adapter, save_adapter, zscore_stats)
from sharing import cost


# ----------------------------- adapters -------------------------------------
def test_zscore_stats_shapes_and_values():
    X = torch.randn(500, 7) * 3 + 2
    m, s = zscore_stats(X)
    assert m.shape == (1, 7) and s.shape == (1, 7)
    Z = (X - m) / s
    assert Z.mean(0).abs().max() < 1e-4          # centered
    assert (Z.std(0) - 1).abs().max() < 1e-2     # unit variance


def test_ridge_recovers_known_linear_map():
    """If Y is an exact affine function of z-scored X, ridge (small lam) reconstructs it."""
    torch.manual_seed(0)
    X = torch.randn(4000, 16)
    m, s = zscore_stats(X)
    Z = (X - m) / s
    Wtrue = torch.randn(16, 12); btrue = torch.randn(12)
    Y = Z @ Wtrue + btrue
    adapter = RidgeAffine.fit(X, Y, lam=1e-3)
    pred = adapter(X)
    rel = (pred - Y).norm() / Y.norm()
    assert rel < 0.05, f"ridge reconstruction rel-error too high: {rel:.4f}"


def test_raw_inject_identity_is_zscore_passthrough():
    X = torch.randn(64, 1152) * 2 + 1
    m, s = zscore_stats(X)
    a = RidgeAffine.identity(1152, m, s)
    out = a(X)
    assert torch.allclose(out, (X - m) / s, atol=1e-5)
    assert a.dim_in == 1152 and a.dim_out == 1152


def test_build_adapter_kinds_and_dim_guard():
    assert isinstance(build_adapter("raw", 1152, 1152), RidgeAffine)
    assert isinstance(build_adapter("ridge", 1152, 1152), RidgeAffine)
    mlp = build_adapter("mlp_recon", 1152, 1024, hidden=256)
    assert isinstance(mlp, ZScoreMLP) and mlp.dim_in == 1152 and mlp.dim_out == 1024
    # raw/ridge/affine require equal dims
    for k in ("raw", "ridge", "affine"):
        try:
            build_adapter(k, 1152, 1024)
            assert False, f"{k} should reject differing dims"
        except ValueError:
            pass


def test_mlp_forward_shape_and_zscore():
    a = ZScoreMLP(1152, 128, 1152)
    m, s = zscore_stats(torch.randn(200, 1152))
    a.set_stats(m, s)
    out = a(torch.randn(5, 1152))
    assert out.shape == (5, 1152)


def test_adapter_save_load_roundtrip():
    X = torch.randn(300, 8); Y = torch.randn(300, 8)
    a = RidgeAffine.fit(X, Y)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        save_adapter(a, f.name, kind="ridge", dim_in=8, dim_out=8)
        b, meta = load_adapter(f.name)
    assert meta["kind"] == "ridge"
    assert torch.allclose(a(X), b(X), atol=1e-5)

    mlp = ZScoreMLP(8, 16, 8)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        save_adapter(mlp, f.name, kind="mlp_recon")
        c, _ = load_adapter(f.name)
    x = torch.randn(4, 8)
    assert torch.allclose(mlp(x), c(x), atol=1e-5)


# ------------------------------- cost ---------------------------------------
def test_adapter_flops_matches_report():
    # 1152->2048->1152, 729 tokens should be ~6.88 GFLOPs (REPORT L375)
    g = cost.adapter_gflops(1152, 2048, 1152, 729)
    assert abs(g - 6.88) < 0.05, f"adapter GFLOPs {g:.3f} != report 6.88"
    # affine is much cheaper than the MLP
    assert cost.affine_gflops(1152, 1152, 729) < g


def test_encode_share_monotone_and_limit():
    fracs = [cost.encode_share(n)["saving_frac"] for n in (1, 2, 4, 8, 16)]
    assert fracs[0] <= 1e-9                                   # N=1: no sharing, no saving
    assert all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1))  # monotone up
    # adapters ~free => saving_frac -> 1 - 1/N ; at N=4 the report quotes ~74%
    s4 = cost.encode_share(4)
    assert 0.70 < s4["saving_frac"] < 0.78, s4["saving_frac"]
    assert cost.encode_share(64)["saving_frac"] > 0.95


def test_store_share_saves_for_multiple_backbones():
    keys = ["internvl3.5-8b", "llava-ov-7b"]
    one = cost.store_share(["internvl3.5-8b"], n_frames=16)
    two = cost.store_share(keys, n_frames=16)
    # sharing one hub across 2 backbones must save vs storing both natively
    assert two["saving_frac"] > 0
    assert two["native_total_bytes"] > one["native_total_bytes"]
    assert two["hub_bytes"] == one["hub_bytes"]              # hub store is N-independent


def test_break_even_shared_finite_and_amortizes():
    keys = ["internvl3.5-8b", "llava-ov-7b", "internvl3.5-4b"]
    r1 = cost.break_even_shared(["internvl3.5-8b"], n_frames=64, tier_name="s3_same_region",
                                retention_days=30)
    r3 = cost.break_even_shared(keys, n_frames=64, tier_name="s3_same_region", retention_days=30)
    assert r1["nstar"] >= 0 and not math.isinf(r1["nstar"])
    # same hub store, but its fixed cost is shared across more query streams -> store_saving up
    assert r3["store_saving_frac"] > r1["store_saving_frac"]


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
