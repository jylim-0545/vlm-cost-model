"""Cost-side accounting for vision-token SHARING (GPU-FREE).

Two views of the hub-and-spoke economics (see sharing/cost.py), both from config bytes +
representative encode latency (pass --hub-encode-ms for measured):

  (A) ENCODE "once, serve N" — vision-encode GFLOPs and GPU-$ vs N backbones, showing the
      saving climb to ~1-1/N (N=4 -> ~74%, REPORT L375).
  (B) STORAGE canonical TokenStore — one hub token set vs each backbone storing its own,
      and the amortized break-even rate N* (queries/month, all backbones) at which the
      shared store beats per-backbone recompute.

  python -m sharing.demo_cost
  python -m sharing.demo_cost --backbones internvl3.5-8b,llava-ov-7b,internvl3.5-4b --frames 64
  python -m sharing.demo_cost --hub-encode-ms 17.8 --retention-days 1500

Reads nothing from the GPU; runs anywhere the repo imports (no torch needed).
"""
from __future__ import annotations

import argparse
import csv as csvmod
import math


def _fmt(x: float) -> str:
    return "never" if math.isinf(x) else f"{x:.1f}"


def main() -> None:
    from sharing import cost
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", default="internvl3.5-8b,llava-ov-7b,internvl3.5-4b",
                    help="comma-separated model keys sharing ONE hub (fixed-token models)")
    ap.add_argument("--frames", type=int, default=64, help="frames per video (storage axis)")
    ap.add_argument("--ns", default="1,2,4,8,16", help="N backbones for the encode-share sweep")
    ap.add_argument("--hub-encode-ms", type=float, default=None,
                    help="measured SigLIP hub encode ms/image (default: representative)")
    ap.add_argument("--retention-days", type=float, default=None)
    ap.add_argument("--no-gpu-stall", action="store_true",
                    help="retrieval overlapped with compute (resource_price=0)")
    ap.add_argument("--tiers", default="local_nvme,s3_same_region")
    ap.add_argument("--out-csv", help="also write the break-even rows here")
    a = ap.parse_args()

    backbones = [x.strip() for x in a.backbones.split(",") if x.strip()]
    ns = [int(x) for x in a.ns.split(",")]
    tiers = a.tiers.split(",")
    prices = cost.load_prices()
    retention = a.retention_days or prices["defaults"]["retention_time_days"]
    rp = 0.0 if a.no_gpu_stall else None
    hub_ms = a.hub_encode_ms

    # ---- (A) encode once, serve N ----
    print("\nvision-token sharing — cost accounting")
    print(f"  hub = SigLIP-so400m ({cost.HUB_VIT_GFLOPS:.0f} GFLOPs, {cost.HUB_DIM}-d × "
          f"{cost.HUB_TOKENS_PER_IMAGE} tok/img)  adapter ≈ {cost.ADAPTER_MLP_GFLOPS:.2f} GFLOPs "
          f"(~{100*cost.ADAPTER_MLP_GFLOPS/cost.HUB_VIT_GFLOPS:.1f}% of ViT)")
    if hub_ms is None:
        print(f"  NOTE: encode-$ uses a REPRESENTATIVE hub encode "
              f"({cost.REPRESENTATIVE['hub_encode_ms_per_image']:.1f} ms/img); pass "
              f"--hub-encode-ms for measured. GFLOPs are exact; bytes are exact.")

    print("\n  (A) ENCODE — one hub + N adapters vs N native encoders")
    print(f"  {'N':>4} {'baseline_GF':>12} {'shared_GF':>11} {'saving%':>9} {'$/1k-img saved':>15}")
    for n in ns:
        u = cost.encode_share_usd(n, hub_encode_ms=hub_ms, n_images=1000)
        print(f"  {n:>4d} {u['baseline_gflops']:>12.0f} {u['shared_gflops']:>11.0f} "
              f"{100*u['saving_frac']:>8.1f}% {u['saving_usd']:>15.4f}")

    # ---- (B) storage + break-even ----
    s = cost.store_share(backbones, n_frames=a.frames)
    print(f"\n  (B) STORAGE — canonical TokenStore for {len(backbones)} backbones, "
          f"{a.frames} frames")
    for k, b in s["per_backbone_bytes"].items():
        print(f"      native  {k:<18} {b/1e6:>8.2f} MB")
    print(f"      native total      {s['native_total_bytes']/1e6:>8.2f} MB")
    print(f"      shared hub        {s['hub_bytes']/1e6:>8.2f} MB  "
          f"(saving {100*s['saving_frac']:.1f}%)")

    print(f"\n  break-even N* (queries/month, ALL backbones) — hub store beats recompute")
    print(f"  retention={retention:g}d  gpu=${prices['compute']['gpu_h100_usd_per_hour']}/h  "
          f"gpu_stall={'OFF' if a.no_gpu_stall else 'ON'}")
    rows = []
    for tier in tiers:
        r = cost.break_even_shared(backbones, a.frames, tier, retention,
                                   hub_encode_ms=hub_ms, resource_price=rp)
        rows.append({"tier": tier, **{k: r[k] for k in
                     ("n_backbones", "hub_MB", "native_total_MB", "store_saving_frac",
                      "F_usd", "storage_total_usd", "saving_per_q_usd", "nstar")}})
        print(f"      {tier:<16} hub={r['hub_MB']:.2f}MB  store_saving={100*r['store_saving_frac']:.1f}%"
              f"  save/q=${r['saving_per_q_usd']*1e6:.3f}µ  N*={_fmt(r['nstar'])}/mo")

    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n  wrote {a.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
