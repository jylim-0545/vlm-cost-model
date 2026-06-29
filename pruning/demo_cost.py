"""Cost-side validation of vision-token pruning (GPU-FREE).

Sweeps keep-ratios and shows how the stored footprint and the vt_reuse break-even
rate N* shrink as we prune. Bytes are exact (config); reuse latencies come from a
real reuse_real.csv (--base-csv, recommended) or a documented representative model.

  python -m pruning.demo_cost                                  # internvl3.5-8b, representative
  python -m pruning.demo_cost --model internvl3.5-8b --base-csv results/nextqa/reuse_real.csv
  python -m pruning.demo_cost --no-gpu-stall --retention-days 1500

Reads nothing from the GPU; runs in the `vlmcost` env or anywhere the repo imports.
"""
from __future__ import annotations

import argparse
import csv as csvmod
import math


def _fmt(x: float) -> str:
    return "never" if math.isinf(x) else f"{x:.1f}"


def main() -> None:
    from pruning import cost
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="internvl3.5-8b")
    ap.add_argument("--base-csv", help="reuse_real.csv to pull real base latencies from")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--n-vis", type=int, default=None, help="override n_vis (dynamic-token models)")
    ap.add_argument("--batch", type=int, default=1, help="batch row to pick from --base-csv")
    ap.add_argument("--keeps", default="1.0,0.5,0.333,0.222,0.111",
                    help="comma-separated keep-ratios (1.0 = no pruning)")
    ap.add_argument("--alpha", type=float, default=cost.PRUNE_ALPHA,
                    help="reuse prefill ~ keep**alpha")
    ap.add_argument("--retention-days", type=float, default=None)
    ap.add_argument("--no-gpu-stall", action="store_true",
                    help="retrieval overlapped with compute (resource_price=0)")
    ap.add_argument("--tiers", default="local_nvme,s3_same_region")
    ap.add_argument("--out-csv", help="also write the swept rows here")
    a = ap.parse_args()

    keeps = [float(x) for x in a.keeps.split(",")]
    tiers = a.tiers.split(",")
    prices = cost.load_prices()
    retention = a.retention_days or prices["defaults"]["retention_time_days"]
    rp = 0.0 if a.no_gpu_stall else None

    if a.base_csv:
        base = cost.base_rec_from_csv(a.base_csv, a.model, frames=a.frames, batch=a.batch)
    else:
        base = cost.representative_base_rec(a.model, frames=a.frames, n_vis=a.n_vis)

    encode_ms = base["cold_ttft"] - base["tok_inject"]
    print(f"\nvision-token pruning — cost projection (vt_reuse)")
    print(f"  model={a.model}  n_vis(full)={base['n_vis']}  frames={a.frames}  batch={a.batch}")
    print(f"  base latencies [{base['source']}]: encode(+dec/prep)={encode_ms:.0f}ms  "
          f"reuse-prefill={base['tok_inject']:.0f}ms  h2d_tok={base.get('h2d_tok',0):.1f}ms")
    print(f"  retention={retention:g}d  gpu=${prices['compute']['gpu_h100_usd_per_hour']}/h  "
          f"gpu_stall={'OFF' if a.no_gpu_stall else 'ON'}  alpha={a.alpha}")
    if base["source"] == "representative":
        print("  NOTE: latencies are REPRESENTATIVE (CLAUDE.md §13 medians), not a fresh "
              "measurement; bytes are exact. Use --base-csv for measured numbers.")

    all_rows = []
    for tier in tiers:
        rows = cost.sweep(base, a.model, keeps, tier, retention, alpha=a.alpha, resource_price=rp)
        print(f"\n  ── {tier} ──")
        print(f"  {'keep':>6} {'n_vis':>7} {'token_MB':>9} {'storage_$µ':>11} "
              f"{'save/q_$µ':>10} {'N* /mo':>9}")
        for r in rows:
            print(f"  {r['keep']:>6.3f} {r['n_vis']:>7d} {r['token_MB']:>9.2f} "
                  f"{r['storage_usd']*1e6:>11.3f} {r['saving_per_q_usd']*1e6:>10.4f} "
                  f"{_fmt(r['nstar']):>9}")
        all_rows.extend(rows)

    # headline check: N* and bytes are monotone non-increasing as we prune harder
    for tier in tiers:
        seq = [r for r in all_rows if r["tier"] == tier]
        seq.sort(key=lambda r: -r["keep"])           # 1.0 -> small
        nstars = [r["nstar"] for r in seq]
        finite = [x for x in nstars if not math.isinf(x)]
        mono = all(finite[i] >= finite[i + 1] - 1e-9 for i in range(len(finite) - 1))
        print(f"\n  [{tier}] N* monotone-decreasing with pruning: "
              f"{'OK' if mono else 'NO'}  ({' -> '.join(_fmt(x) for x in nstars)})")

    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f"\n  wrote {a.out_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
