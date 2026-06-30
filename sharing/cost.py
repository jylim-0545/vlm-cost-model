"""Encode-once-share-many + canonical-TokenStore accounting for vision-token SHARING.

PURE arithmetic: no torch, no transformers, no GPU. Reuses the cost model's own byte math
(`config.ModelSpec.vision_token_bytes`) and tier costs (`StorageTier`), so it stays
consistent with `analyze/breakeven_reuse.py` and `pruning/cost.py`. Safe to import/run in
any env (only PyYAML via `config`).

This is the cost-model realization of the token-sharing study's proposed-but-unmeasured
"E4 — storage/serving accounting" (REPORT_VTOKEN_UNIFY.md §6). The study showed that ONE
shared SigLIP **hub** encoding + a light per-backbone **adapter** can drive N independent
VLM backbones (holistic ≈native; same-family fine ≈native with a learned adapter —
see sharing/FINDINGS.md for where it works). Here we price that:

  TWO savings axes when serving / ingesting the SAME image for N backbones:

  (A) ENCODE compute — "encode once, serve N".
      baseline (no sharing): each backbone runs its OWN vision tower  -> N x ViT
      hub-and-spoke:          one hub ViT + N cheap adapters           -> ViT + N x adapter
      The adapter is ~1% of a ViT (REPORT L375: 2-layer MLP 6.88 GFLOPs vs SigLIP ViT
      665 GFLOPs), so at N=4 the vision-encode compute drops ~74%. Reported in FLOPs
      (hardware-free, the study's unit) and optionally in GPU-$ via an encode-ms estimate.

  (B) STORAGE — canonical TokenStore (the vt_reuse / §7 angle, generalized to N backbones).
      baseline: store EACH backbone's reusable vision tokens separately -> sum_i bytes_i
      shared:   store ONE hub token set, every backbone adapts it at read -> bytes_hub
      Bytes from config (exact), so this plugs straight into the tier/break-even machinery.

  Break-even: the one-time hub encode F and the hub storage rent are AMORTIZED across the
  N backbones' query streams, so the per-backbone break-even rate drops vs an unshared store.

CAVEAT (carried from the study, must travel with the cost number): sharing is near-lossless
only for HOLISTIC tasks and SAME-encoder-family fine-grained backbones; cross-encoder
fine-grained loses ~15-21 accuracy points (FINDINGS §"when shareable"). The compute/storage
win trades against accuracy OUTSIDE that sweet spot — these functions price the win; they do
not assert it is free everywhere.
"""
from __future__ import annotations

import math

from config import load_models, load_prices, load_storage_tiers

# --- hub + adapter constants, sourced from REPORT_VTOKEN_UNIFY.md (L31, L375, §1.1) ------
HUB_DIM = 1152                 # SigLIP-so400m token width
HUB_TOKENS_PER_IMAGE = 729     # 27x27 patches @384 (no CLS)
HUB_VIT_GFLOPS = 665.0         # SigLIP-so400m vision tower, per 384px image (REPORT L375)
HUB_VIT_PARAMS_M = 428.0       # SigLIP-so400m ViT params (M)
# 2-layer MLP adapter 1152->2048->1152: REPORT L375 = 6.88 GFLOPs/img, 4.72M params (~1% of ViT)
ADAPTER_MLP_GFLOPS = 6.88
ADAPTER_MLP_PARAMS_M = 4.72


def adapter_gflops(d_in: int = HUB_DIM, hidden: int = 2048, d_out: int = HUB_DIM,
                   n_tokens: int = HUB_TOKENS_PER_IMAGE) -> float:
    """FLOPs (G) for a 2-layer MLP adapter applied to `n_tokens` tokens. Per token =
    2*(d_in*hidden + hidden*d_out) MACs*2. Defaults reproduce the report's 6.88 GFLOPs
    (1152->2048->1152, 729 tokens)."""
    per_tok = 2.0 * (d_in * hidden + hidden * d_out)   # 2 flops per MAC
    return per_tok * n_tokens / 1e9


def affine_gflops(d_in: int = HUB_DIM, d_out: int = HUB_DIM,
                  n_tokens: int = HUB_TOKENS_PER_IMAGE) -> float:
    """FLOPs (G) for the ridge/affine adapter (single matmul d_in->d_out) over n_tokens.
    Cheaper than the MLP — the Stage-A operational adapter."""
    return 2.0 * d_in * d_out * n_tokens / 1e9


# Representative encode latency for the GPU-$ view, when no measured CSV is supplied.
# The repo measures encode on H100; the token-sharing study ran on RTX 4090. We keep the
# encode-ms as REPRESENTATIVE and let the adapter cost be a FLOP-fraction of it (adapters
# share the ViT's arithmetic profile). Pass --hub-encode-ms / a CSV for measured numbers.
REPRESENTATIVE = {
    "hub_encode_ms_per_image": 18.0,   # SigLIP-so400m @384, 1 image, bf16 (representative)
    "gpu_tflops_bf16": 130.0,          # realistic sustained bf16 throughput (MFU-discounted)
}


def encode_share(n_backbones: int, vit_gflops: float = HUB_VIT_GFLOPS,
                 adapter_gflops_each: float = ADAPTER_MLP_GFLOPS) -> dict:
    """Vision-ENCODE compute for serving the same image to `n_backbones`, in GFLOPs.
    baseline = N x ViT (each backbone encodes itself); shared = 1 x hub ViT + N x adapter.
    Homogeneous-N assumption (every backbone ~one ViT), matching REPORT L375's 74%@N=4.
    Returns GFLOPs + saving fraction."""
    n = int(n_backbones)
    baseline = n * vit_gflops
    shared = vit_gflops + n * adapter_gflops_each
    saving = baseline - shared
    return {
        "n_backbones": n,
        "baseline_gflops": baseline,
        "shared_gflops": shared,
        "saving_gflops": saving,
        "saving_frac": (saving / baseline) if baseline > 0 else 0.0,
        "adapter_gflops_each": adapter_gflops_each,
        "vit_gflops": vit_gflops,
    }


def encode_share_usd(n_backbones: int, hub_encode_ms: float | None = None,
                     adapter_gflops_each: float = ADAPTER_MLP_GFLOPS,
                     vit_gflops: float = HUB_VIT_GFLOPS, gpu_rate_usd_per_s: float | None = None,
                     n_images: int = 1) -> dict:
    """GPU-$ version of `encode_share` for ingesting `n_images` images for N backbones.
    Adapter ms is the ViT ms scaled by the FLOP ratio (adapters share the ViT arithmetic
    profile). gpu_rate from prices.yaml (H100) unless overridden."""
    hub_ms = REPRESENTATIVE["hub_encode_ms_per_image"] if hub_encode_ms is None else hub_encode_ms
    if gpu_rate_usd_per_s is None:
        gpu_rate_usd_per_s = load_prices()["compute"]["gpu_h100_usd_per_hour"] / 3600.0
    adapter_ms = hub_ms * (adapter_gflops_each / vit_gflops)
    g = encode_share(n_backbones, vit_gflops, adapter_gflops_each)
    baseline_s = g["n_backbones"] * hub_ms / 1e3 * n_images
    shared_s = (hub_ms + g["n_backbones"] * adapter_ms) / 1e3 * n_images
    return {
        **g, "n_images": n_images, "hub_encode_ms": hub_ms, "adapter_ms": adapter_ms,
        "baseline_usd": baseline_s * gpu_rate_usd_per_s,
        "shared_usd": shared_s * gpu_rate_usd_per_s,
        "saving_usd": (baseline_s - shared_s) * gpu_rate_usd_per_s,
    }


def hub_bytes(n_frames: int = 1, hub_tokens_per_frame: int = HUB_TOKENS_PER_IMAGE,
              hub_dim: int = HUB_DIM, dtype_bytes: int = 2) -> int:
    """Bytes of the canonical hub token store: n_frames x hub_tokens x hub_dim x dtype.
    One copy serves every backbone."""
    return n_frames * hub_tokens_per_frame * hub_dim * dtype_bytes


def store_share(backbone_keys: list[str], n_frames: int = 1,
                hub_tokens_per_frame: int = HUB_TOKENS_PER_IMAGE, hub_dim: int = HUB_DIM) -> dict:
    """STORAGE bytes for the reusable vision tokens of `backbone_keys` over `n_frames`.
    baseline = sum_i config.vision_bytes(key, per_frame_tokens_i * n_frames)  (each backbone
    stores its own); shared = ONE hub token set. Uses the repo's exact byte math; dynamic-
    token models are skipped (need an explicit n_vis). Returns bytes + saving fraction."""
    cfg = load_models()
    per = {}
    native_total = 0
    for key in backbone_keys:
        spec = cfg.models[key]
        pft = spec.per_frame_tokens
        if not isinstance(pft, int):
            raise ValueError(f"{key} has dynamic per_frame_tokens; sharing storage needs a "
                             "fixed per-frame token count (use InternVL/LLaVA-OV)")
        b = cfg.vision_bytes(key, pft * n_frames)
        per[key] = b
        native_total += b
    shared = hub_bytes(n_frames, hub_tokens_per_frame, hub_dim, cfg.dtype_bytes)
    return {
        "backbones": list(backbone_keys),
        "n_frames": n_frames,
        "per_backbone_bytes": per,
        "native_total_bytes": native_total,
        "hub_bytes": shared,
        "saving_bytes": native_total - shared,
        "saving_frac": (native_total - shared) / native_total if native_total > 0 else 0.0,
    }


def break_even_shared(backbone_keys: list[str], n_frames: int, tier_name: str,
                      retention_days: float, hub_encode_ms: float | None = None,
                      resource_price: float | None = None) -> dict:
    """Amortized break-even for the canonical TokenStore serving N=len(backbone_keys)
    backbones. The one-time hub encode F and the hub storage rent are paid ONCE and shared
    across all N backbones' query streams; each backbone's per-query saving is the native
    vision encode it now SKIPS (it adapts the stored hub tokens instead). We report the
    AGGREGATE break-even total-query-rate N* (queries/month across all N backbones) at which
    the shared store beats per-backbone recompute. Mirrors analyze.breakeven_reuse / pruning.

      F        = hub_encode (one ViT) [+ adapters, negligible per ingest]
      saving/q = native encode skipped per query (representative ViT ms) - retrieval(hub bytes)
      storage  = hub store rent over retention
      N*       = (F + storage) / (R * saving_per_query)
    """
    prices = load_prices()
    tier = load_storage_tiers()[tier_name]
    gpu_rate = prices["compute"]["gpu_h100_usd_per_hour"] / 3600.0
    rp = gpu_rate if resource_price is None else resource_price
    R = retention_days / 30.0
    hub_ms = REPRESENTATIVE["hub_encode_ms_per_image"] if hub_encode_ms is None else hub_encode_ms

    s = store_share(backbone_keys, n_frames)
    hub_b = s["hub_bytes"]
    # encode skipped per query = the native ViT over the WHOLE video the backbone would
    # otherwise run (n_frames frames; ~hub ViT/frame as a homogeneous proxy). The retrieval
    # of the (n_frames-frame) hub store is weighed against it — same structure as vt_reuse.
    encode_skip_s = hub_ms * n_frames / 1e3
    retrieval_per_q = tier.network_cost_usd(hub_b, rp, include_egress=False)
    saving_per_q = encode_skip_s * gpu_rate - retrieval_per_q
    F_usd = hub_ms * n_frames / 1e3 * gpu_rate            # encode the hub video once at ingest
    storage_total = tier.storage_cost_usd(hub_b, retention_days)
    nstar = math.inf if saving_per_q <= 0 else (F_usd + storage_total) / (R * saving_per_q)
    return {
        "backbones": s["backbones"], "n_backbones": len(backbone_keys), "n_frames": n_frames,
        "tier": tier_name, "hub_MB": hub_b / 1e6, "native_total_MB": s["native_total_bytes"] / 1e6,
        "store_saving_frac": s["saving_frac"], "F_usd": F_usd, "storage_total_usd": storage_total,
        "saving_per_q_usd": saving_per_q, "retrieval_per_q_usd": retrieval_per_q, "nstar": nstar,
    }


def sweep_n(backbone_key: str, ns, vit_gflops: float = HUB_VIT_GFLOPS,
            adapter_gflops_each: float = ADAPTER_MLP_GFLOPS) -> list[dict]:
    """Encode-share saving as N (number of shared backbones) grows, for a homogeneous fleet
    of `backbone_key`. Shows the FLOP saving climbing toward 1 - 1/N as adapters are ~free."""
    return [encode_share(n, vit_gflops, adapter_gflops_each) for n in ns]
