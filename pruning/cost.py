"""Keep-ratio -> storage / break-even projection for vision-token pruning.

PURE arithmetic: no torch, no transformers, no GPU. Reuses the cost model's own byte
math (`config.ModelSpec.vision_token_bytes`) and tier costs
(`StorageTier.network_cost_usd / storage_cost_usd`), so it stays consistent with
`analyze/breakeven_reuse.py`. Safe to import/run in the `vlmcost` env or anywhere.

What pruning does to the economics (see pruning/README.md for the framing):
  The STORED vision-token footprint shrinks to k = round(keep * n_vis). The reuse
  path then stores fewer bytes (storage + retrieval down) and prefills fewer vision
  tokens (reuse front-end down). The recompute BASELINE stays at full n_vis (it
  reconstructs the full-quality result every query); the accuracy demo is what
  justifies that k tokens ≈ full quality, making the comparison meaningful.

Why not call `analyze.breakeven_reuse.break_even()` directly:
  that function derives the one-time store cost as F = cold_ttft - tok_inject, which
  equals `encode` ONLY when cold and reuse prefill the SAME tokens. Under pruning the
  reuse path prefills FEWER tokens, so that identity breaks. We therefore compute the
  economics explicitly here with the TRUE one-time cost F = encode (you still encode
  the full frames once, then score+prune+store), mirroring CLAUDE.md Section 7.

Latency model (per request, GPU-ms; decode cancels in break-even, CLAUDE.md Section 7):
  cold front  b = encode_full + prefill_full           (full recompute, unchanged by keep)
  reuse front r = prefill(k) + retrieval(bytes(k))      (encode skipped; prefill shrinks)
  prefill(k)   = prefill_full * keep**alpha             (vision tokens dominate prefill;
                                                          alpha~1.1, mildly super-linear)
  one-time   F = encode_full                            (encode full frames once at ingest)
A base record supplies (encode_full, prefill_full, h2d_full) either from a real
reuse_real.csv row (--base-csv, recommended) or from a documented representative model
(REPRESENTATIVE below). Bytes are ALWAYS exact (from config), never representative.
"""
from __future__ import annotations

import math
from pathlib import Path

from config import load_models, load_prices, load_storage_tiers

# Representative per-token latency coefficients, for when no measured CSV is given.
# Sourced from CLAUDE.md Section 13 (InternVL-8B @128f/b1, n_vis=32768): encoder is
# ~linear (~19 us/token); prefill is mildly super-linear (~n^1.2) and ~3160 ms at
# n_vis=32768 -> coefficient 3160 / 32768**1.2. dec+prep is small on the GPU pipeline.
# These are REPRESENTATIVE medians (NOT a fresh measurement) — pass --base-csv for real.
REPRESENTATIVE = {
    "encode_us_per_tok": 19.0,        # ViT encode, us/token (linear)
    "prefill_ms_at_32768": 3160.0,    # reuse prefill at n_vis=32768
    "prefill_alpha": 1.2,             # prefill ~ n_vis**alpha
    "decprep_ms": 38.0,               # GPU NVDEC decode+preprocess (360p, ~flat)
    "h2d_gbps": 50.0,                 # measured DRAM->GPU bandwidth (~42-52 GB/s)
}
PRUNE_ALPHA = 1.1   # how reuse prefill scales with keep-ratio (vision-token share of prefill)


def representative_base_rec(model_key: str, frames: int = 128, n_vis: int | None = None) -> dict:
    """Build a base record (latencies from REPRESENTATIVE coefficients, bytes exact)
    for one (model, frames). For models with fixed tokens/frame (InternVL 256,
    LLaVA-OV 196) n_vis is derived; for dynamic-token models pass --n-vis."""
    cfg = load_models()
    spec = cfg.models[model_key]
    if n_vis is None:
        pft = spec.per_frame_tokens
        if not isinstance(pft, int):
            raise ValueError(f"{model_key} has dynamic per_frame_tokens; pass n_vis explicitly")
        n_vis = pft * frames
    R = REPRESENTATIVE
    encode_full = R["encode_us_per_tok"] * n_vis / 1e3 + R["decprep_ms"]      # ms (encode + dec/prep)
    prefill_full = R["prefill_ms_at_32768"] * (n_vis / 32768.0) ** R["prefill_alpha"]   # ms
    token_bytes = cfg.vision_bytes(model_key, n_vis)
    h2d_full = token_bytes / (R["h2d_gbps"] * 1e9) * 1e3                       # ms
    return {
        "model": model_key, "frames": frames, "batch": 1, "n_vis": n_vis,
        "cold_ttft": encode_full + prefill_full,   # front = encode(+decprep) + prefill
        "tok_inject": prefill_full,                # reuse front = prefill only
        "h2d_tok": h2d_full,
        "token_bytes": token_bytes,
        "kv_bytes": cfg.kv_bytes(model_key, n_vis),
        "source": "representative",
    }


def base_rec_from_csv(csv_path: str, model_key: str, frames: int | None = None,
                      batch: int = 1) -> dict:
    """Pull a real base record from a reuse_real.csv via the existing loader. Picks the
    (model, batch[, frames]) row with the largest n_vis (the headline operating point)."""
    from analyze.breakeven_reuse import load_reuse
    recs = [r for r in load_reuse(csv_path)
            if r["model"] == model_key and r["batch"] == batch
            and (frames is None or r["frames"] == frames)
            and r.get("cold_ttft") and r.get("tok_inject")]
    if not recs:
        raise ValueError(f"no usable rows for model={model_key} batch={batch} "
                         f"frames={frames} in {csv_path}")
    rec = max(recs, key=lambda r: r["n_vis"])
    rec = dict(rec); rec["source"] = f"csv:{Path(csv_path).name}"
    rec.setdefault("h2d_tok", 0.0)
    return rec


def project_pruned(base_rec: dict, model_key: str, keep: float, alpha: float = PRUNE_ALPHA) -> dict:
    """Return a record describing the STORED+SERVED state at keep-ratio `keep`:
    bytes exact from config, reuse-side latencies scaled. The cold baseline is left at
    full n_vis (full-quality recompute)."""
    cfg = load_models()
    n_k = max(1, round(base_rec["n_vis"] * keep))
    prefill_full = base_rec["tok_inject"]
    encode_full = base_rec["cold_ttft"] - base_rec["tok_inject"]   # encode(+decprep), pruning-invariant
    scale = (n_k / base_rec["n_vis"]) ** alpha
    return {
        "keep": keep, "n_vis": n_k, "model": model_key,
        "encode_full_ms": encode_full,
        "prefill_full_ms": prefill_full,
        "prefill_k_ms": prefill_full * scale,
        "token_bytes": cfg.vision_bytes(model_key, n_k),
        "kv_bytes": cfg.kv_bytes(model_key, n_k),
        "h2d_tok_ms": base_rec.get("h2d_tok", 0.0) * keep,
    }


def break_even_pruned(base_rec: dict, model_key: str, keep: float, tier, gpu_rate: float,
                      retention_days: float, alpha: float = PRUNE_ALPHA,
                      include_egress: bool = False,
                      resource_price: float | None = None) -> tuple[float, dict]:
    """Break-even query RATE (queries/month) for vt_reuse of the pruned tokens.
    Returns (N*, components). inf if reuse never beats recompute. Mirrors
    analyze.breakeven_reuse.break_even semantics, with F = true encode."""
    rp = gpu_rate if resource_price is None else resource_price
    R = retention_days / 30.0
    p = project_pruned(base_rec, model_key, keep, alpha)
    b_front_s = (p["encode_full_ms"] + p["prefill_full_ms"]) / 1e3       # full recompute front
    r_front_s = p["prefill_k_ms"] / 1e3                                  # reuse front (encode skipped)
    retrieval = (tier.network_cost_usd(p["token_bytes"], rp, include_egress)
                 + p["h2d_tok_ms"] / 1e3 * rp)
    saving = (b_front_s - r_front_s) * gpu_rate - retrieval
    F_usd = p["encode_full_ms"] / 1e3 * gpu_rate                         # encode once at ingest
    storage_total = tier.storage_cost_usd(p["token_bytes"], retention_days)
    comp = {"n_vis": p["n_vis"], "token_bytes": p["token_bytes"], "F_usd": F_usd,
            "saving_per_q": saving, "storage_total": storage_total, "retrieval_per_q": retrieval}
    if saving <= 0:
        return math.inf, comp
    return (F_usd + storage_total) / (R * saving), comp


def sweep(base_rec: dict, model_key: str, keeps, tier_name: str, retention_days: float,
          alpha: float = PRUNE_ALPHA, resource_price: float | None = None) -> list[dict]:
    """Run break_even_pruned across keep-ratios on one tier -> list of rows."""
    prices = load_prices()
    tier = load_storage_tiers()[tier_name]
    gpu_rate = prices["compute"]["gpu_h100_usd_per_hour"] / 3600.0
    rows = []
    for keep in keeps:
        nstar, c = break_even_pruned(base_rec, model_key, keep, tier, gpu_rate,
                                     retention_days, alpha, resource_price=resource_price)
        rows.append({"keep": keep, "tier": tier_name, "n_vis": c["n_vis"],
                     "token_MB": c["token_bytes"] / 1e6, "nstar": nstar,
                     "storage_usd": c["storage_total"], "saving_per_q_usd": c["saving_per_q"]})
    return rows
