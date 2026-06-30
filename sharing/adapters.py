"""Vision-token sharing adapters — the per-backbone "spoke" maps, ported from our
token-sharing study (`EfficientVLM/scripts/{e1_stagea,d6_adapteronly,d12_holistic}.py`).

A single shared vision **hub** (stock `google/siglip-so400m-patch14-384`, 729 tokens ×
1152-d per 384px image) is encoded ONCE; a small per-backbone **adapter** then maps the
hub tokens into that backbone's vision-token space so the (frozen) backbone reads them in
place of its own encoder output. This module holds ONLY the adapters + their fitting math
— pure `torch`, no model load, no GPU required — so it is unit-testable on its own. The
model-touching side (hub encoder, backbone injection, training loop) lives in
`sharing.methods` / `sharing.train`.

Three adapter variants, matching the study's "raw → recon → E2E" ladder
(see sharing/FINDINGS.md for the recovery numbers):

  RidgeAffine  f(x) = z(x) @ W + b ,  z(x) = (x - mean) / std
               - identity init  (W=I, b=0)  -> "raw inject": z-scored hub tokens fed
                 directly, NO training. Works only when d_in == d_out (same-family / OV).
               - ridge init     (closed-form least-squares z(x)->y) -> the label-free
                 "Stage-A" operational adapter. Fit in seconds, generalizes best (linear
                 ≥ MLP for fine-grained reconstruction; see FINDINGS "R² ≠ accuracy").
  ZScoreMLP    f(x) = net(z(x)) ,  net = Linear(d_in,H)->GELU->Linear(H,d_out)
               - the "mlp_recon" (MSE-pretrained to mimic native tokens) and "mlp_e2e"
                 (recon-pretrained, then VQA-CE fine-tuned) capacity adapter.

z-score standardization is mandatory (raw SigLIP activations have heavy outliers that
blow up a bare linear map). `mean`/`std` are stored as buffers so the adapter is
self-contained once fit.

Importing this module pulls in `torch` but NOTHING else (no transformers, no model).
"""
from __future__ import annotations

import torch
import torch.nn as nn

HUB_DIM = 1152              # SigLIP-so400m token width
HUB_TOKENS_PER_IMAGE = 729  # 27x27 patches @384, no CLS


def zscore_stats(X: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean/std over rows of X [n, d] -> (mean [1,d], std [1,d]). std is
    floored by `eps` so a dead feature does not divide by zero (matches the study's
    `xs = X.std(0) + 1e-6`)."""
    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True) + eps
    return mean, std


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float = 1.0,
              eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form ridge token-matching on z-scored X. Returns (mean, std, W, b) such that
    ``((X - mean)/std) @ W + b`` reconstructs Y in least squares with L2 penalty `lam`.
    This is the Stage-A adapter (`d6_adapteronly.py` "ridgeaffine", `e1_stagea.py`):
    label-free (Y = native post-encoder tokens), no SGD. b is the target mean so the map
    is exact in expectation. Operates in float for numerical stability."""
    X = X.float(); Y = Y.float()
    mean, std = zscore_stats(X, eps)
    Z = (X - mean) / std
    d = Z.shape[1]
    ym = Y.mean(0, keepdim=True)
    W = torch.linalg.solve(Z.T @ Z + lam * torch.eye(d, device=Z.device, dtype=Z.dtype),
                           Z.T @ (Y - ym))
    return mean, std, W, ym.squeeze(0)


class RidgeAffine(nn.Module):
    """z-score + affine: ``f(h) = ((h - mean)/std) @ W + b``. Covers BOTH the
    no-train "raw inject" (identity init) and the closed-form "ridge" Stage-A adapter.
    `W`/`b` are Parameters so the affine can ALSO be SGD-fine-tuned (the study's trainable
    "affine" variant); `mean`/`std` are frozen buffers from the fitting corpus."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())
        self.W = nn.Parameter(W.float())
        self.b = nn.Parameter(b.float())

    @property
    def dim_in(self) -> int:
        return self.W.shape[0]

    @property
    def dim_out(self) -> int:
        return self.W.shape[1]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return ((h - self.mean) / self.std) @ self.W + self.b

    @classmethod
    def identity(cls, dim: int, mean: torch.Tensor | None = None,
                 std: torch.Tensor | None = None) -> "RidgeAffine":
        """Raw-inject adapter: W=I, b=0 -> output is just the z-scored hub tokens. Requires
        d_in == d_out (same-family, e.g. SigLIP hub -> LLaVA-OV's SigLIP VT slot). If
        mean/std are None they default to 0/1 (caller should pass corpus stats for a real
        raw-inject; identity stats = passthrough)."""
        mean = torch.zeros(1, dim) if mean is None else mean
        std = torch.ones(1, dim) if std is None else std
        return cls(mean, std, torch.eye(dim), torch.zeros(dim))

    @classmethod
    def fit(cls, X: torch.Tensor, Y: torch.Tensor, lam: float = 1.0) -> "RidgeAffine":
        """Closed-form Stage-A fit on a corpus of (hub_token, native_token) pairs."""
        mean, std, W, b = fit_ridge(X, Y, lam)
        return cls(mean, std, W, b)


class ZScoreMLP(nn.Module):
    """z-score + 2-layer MLP: ``f(h) = net((h - mean)/std)``, net = Linear(d_in,H) -> GELU
    -> Linear(H, d_out). The capacity adapter for "mlp_recon" (MSE-pretrained) and
    "mlp_e2e" (then VQA-CE fine-tuned). `mean`/`std` frozen buffers; the two Linears train.
    Default H=2048 matches the study (D6_MLP_H); the report's 6.88 GFLOPs / 4.72M-param
    figure is this net at d_in=d_out=1152, H=2048."""

    def __init__(self, dim_in: int = HUB_DIM, hidden: int = 2048, dim_out: int | None = None,
                 mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        m = torch.zeros(1, dim_in) if mean is None else mean.float()
        s = torch.ones(1, dim_in) if std is None else std.float()
        self.register_buffer("mean", m)
        self.register_buffer("std", s)
        self.net = nn.Sequential(nn.Linear(dim_in, hidden), nn.GELU(), nn.Linear(hidden, dim_out))

    @property
    def dim_in(self) -> int:
        return self.net[0].in_features

    @property
    def dim_out(self) -> int:
        return self.net[-1].out_features

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Replace the z-score buffers with corpus stats (call before pretrain)."""
        self.mean.copy_(mean.float().to(self.mean.device))
        self.std.copy_(std.float().to(self.std.device))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net((h - self.mean) / self.std)


def build_adapter(kind: str, dim_in: int = HUB_DIM, dim_out: int | None = None,
                  hidden: int = 2048, mean: torch.Tensor | None = None,
                  std: torch.Tensor | None = None) -> nn.Module:
    """Factory for an UNFIT adapter shell (training fills in the weights). `kind` in
    {raw, affine, ridge, mlp_recon, mlp_e2e}. ridge is fit closed-form by the trainer
    (here it returns an identity-init affine to be overwritten); mlp_* return a ZScoreMLP.
    raw/affine return a RidgeAffine (identity init)."""
    dim_out = dim_in if dim_out is None else dim_out
    if kind in ("raw", "affine", "ridge"):
        if dim_in != dim_out:
            raise ValueError(f"{kind} adapter needs dim_in==dim_out (got {dim_in}!={dim_out}); "
                             "use an mlp_* adapter to change dimension")
        return RidgeAffine.identity(dim_in, mean, std)
    if kind in ("mlp_recon", "mlp_e2e", "mlp"):
        return ZScoreMLP(dim_in, hidden, dim_out, mean, std)
    raise ValueError(f"unknown adapter kind '{kind}'")


def save_adapter(adapter: nn.Module, path: str, **meta) -> None:
    """Serialize an adapter + its z-score stats + arbitrary metadata (kind, dims, ...)."""
    blob = {"class": type(adapter).__name__, "state_dict": adapter.state_dict()}
    blob.update(meta)
    torch.save(blob, path)


def load_adapter(path: str, map_location: str = "cpu") -> tuple[nn.Module, dict]:
    """Inverse of `save_adapter`. Rebuilds the module from saved shapes, loads weights,
    returns (adapter, meta)."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    sd = blob["state_dict"]
    if blob["class"] == "RidgeAffine":
        adapter = RidgeAffine(sd["mean"], sd["std"], sd["W"], sd["b"])
    elif blob["class"] == "ZScoreMLP":
        din = sd["net.0.weight"].shape[1]
        hidden = sd["net.0.weight"].shape[0]
        dout = sd["net.2.weight"].shape[0]
        adapter = ZScoreMLP(din, hidden, dout)
        adapter.load_state_dict(sd)
    else:
        raise ValueError(f"unknown adapter class '{blob['class']}'")
    meta = {k: v for k, v in blob.items() if k not in ("class", "state_dict")}
    return adapter, meta
