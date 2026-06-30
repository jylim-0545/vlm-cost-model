"""GPU-free unit checks for sharing.adapters (the model-agnostic adapter math).

  ridge closed-form recovers a known linear map; z-score stats; identity raw-inject
  passthrough; MLP shapes; build_adapter dim guard; save/load round-trip.

Run:  python -m sharing.test_adapters       (needs only torch)
"""
from __future__ import annotations

import tempfile

import torch

from sharing.adapters import (RidgeAffine, ZScoreMLP, build_adapter, load_adapter,
                              save_adapter, zscore_stats)


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
    rel = (adapter(X) - Y).norm() / Y.norm()
    assert rel < 0.05, f"ridge reconstruction rel-error too high: {rel:.4f}"


def test_raw_inject_identity_is_zscore_passthrough():
    X = torch.randn(64, 1152) * 2 + 1
    m, s = zscore_stats(X)
    a = RidgeAffine.identity(1152, m, s)
    assert torch.allclose(a(X), (X - m) / s, atol=1e-5)
    assert a.dim_in == 1152 and a.dim_out == 1152


def test_build_adapter_kinds_and_dim_guard():
    assert isinstance(build_adapter("raw", 1152, 1152), RidgeAffine)
    assert isinstance(build_adapter("ridge", 1152, 1152), RidgeAffine)
    mlp = build_adapter("mlp_recon", 1152, 1024, hidden=256)
    assert isinstance(mlp, ZScoreMLP) and mlp.dim_in == 1152 and mlp.dim_out == 1024
    for k in ("raw", "ridge", "affine"):          # these require equal dims
        try:
            build_adapter(k, 1152, 1024)
            assert False, f"{k} should reject differing dims"
        except ValueError:
            pass


def test_mlp_forward_shape_and_zscore():
    a = ZScoreMLP(1152, 128, 1152)
    m, s = zscore_stats(torch.randn(200, 1152))
    a.set_stats(m, s)
    assert a(torch.randn(5, 1152)).shape == (5, 1152)


def test_adapter_save_load_roundtrip():
    X = torch.randn(300, 8); Y = torch.randn(300, 8)
    a = RidgeAffine.fit(X, Y)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        save_adapter(a, f.name, kind="ridge")
        b, meta = load_adapter(f.name)
    assert meta["kind"] == "ridge"
    assert torch.allclose(a(X), b(X), atol=1e-5)

    mlp = ZScoreMLP(8, 16, 8)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        save_adapter(mlp, f.name, kind="mlp_recon")
        c, _ = load_adapter(f.name)
    x = torch.randn(4, 8)
    assert torch.allclose(mlp(x), c(x), atol=1e-5)


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
