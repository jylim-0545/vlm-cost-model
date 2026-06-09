"""Workload-level TCO saving from a REAL view/query-rate (VPM) distribution.

Given total_vpm.csv (one column: views_per_month, one row per video) and the measured
cost primitives, decide PER VIDEO whether to cache (kv_reuse / vt_reuse) or just recompute
(baseline) — cache only if it saves money (N >= break-even N*) — then report how much of the
TOTAL inference TCO is saved across the whole workload.

Per video, over retention R months, with rate N (= vpm):
  baseline_TCO = N*R * cold_full_s * gpu_rate                      (front + DECODE, no storage)
  cache_saving(variant) = N*R * saving_per_q - F_usd - storage_total   (>0 only above N*)
  per-video optimal = baseline - max(0, best achievable cache_saving)
Total saving% = sum_v max(0, best cache_saving_v) / sum_v baseline_TCO_v.

ASSUMPTION: total_vpm.csv has only the rate, not each video's n_vis. So all videos are
evaluated at ONE operating point (--model, --frame -> n_vis, --batch); VPM supplies only the
popularity N. (Give per-video duration/resolution to map n_vis per video instead.)
"""
from __future__ import annotations
import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_prices, load_storage_tiers           # noqa: E402
from analyze.breakeven_reuse import load_reuse, break_even, WANT   # noqa: E402

VIDEO = "5396384503"   # the pinned single video (same as figures); cost primitives at the op point


def load_vpm(path: str) -> list[float]:
    out = []
    with open(path) as f:
        rd = csv.DictReader(f)
        col = rd.fieldnames[0]
        for r in rd:
            try:
                out.append(float(r[col]))
            except (TypeError, ValueError):
                pass
    return out


def cold_full_ms(reuse_csv: str, model: str, frame: int, batch: int) -> float | None:
    vals = []
    with open(reuse_csv) as f:
        for d in csv.DictReader(f):
            if (d["model"] == model and int(d["frames"]) == frame and int(d.get("batch") or 1) == batch
                    and d["video_id"].split("_b")[0] == VIDEO
                    and d["variant"] == "cold" and d["metric"] == "full"):
                vals.append(float(d["value_ms"]))
    return statistics.median(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vpm", default="total_vpm.csv")
    ap.add_argument("--reuse-csv", default="results/nextqa/reuse_real.csv")
    ap.add_argument("--models", help="comma list (default: all in csv)")
    ap.add_argument("--frame", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--retention-days", type=float, default=30.0)
    ap.add_argument("--tier", help="single tier; omit to sweep all")
    ap.add_argument("--no-gpu-stall", action="store_true")
    ap.add_argument("--no-egress", dest="egress", action="store_false")
    ap.set_defaults(egress=True)
    a = ap.parse_args()

    vpm = load_vpm(a.vpm)
    prices = load_prices(); gpu_hr = prices["compute"]["gpu_h100_usd_per_hour"]; gpu_rate = gpu_hr / 3600.0
    tiers = load_storage_tiers()
    tier_items = [(a.tier, tiers[a.tier])] if a.tier else list(tiers.items())
    R = a.retention_days / 30.0
    recs = load_reuse(a.reuse_csv)
    # index records by (model) for VIDEO at (frame,batch)
    by_model = {}
    for rec in recs:
        if (rec["video_id"].split("_b")[0] == VIDEO and rec["frames"] == a.frame
                and rec["batch"] == a.batch):
            by_model[rec["model"]] = rec
    models = a.models.split(",") if a.models else sorted(by_model)

    n_vid = len(vpm); total_N = sum(vpm)
    print(f"\nWorkload TCO saving — {n_vid:,} videos, vpm: mean={total_N/n_vid:.1f} "
          f"median={statistics.median(vpm):.2f} max={max(vpm):.0f}")
    print(f"op point: video={VIDEO} frame={a.frame} batch={a.batch}  retention={a.retention_days:g}d "
          f"gpu=${gpu_hr}/h  gpu_stall={'OFF' if a.no_gpu_stall else 'ON'} egress={'on' if a.egress else 'off'}")
    print(f"(ASSUMPTION: every video evaluated at this one n_vis; VPM gives only popularity N)\n")

    for tname, tier in tier_items:
        print(f"── {tname} ──")
        print(f"  {'model':16}{'n_vis':>7}{'kv N*':>9}{'vt N*':>9} | "
              f"{'kv save%':>9}{'kv used%':>9} | {'vt save%':>9}{'vt used%':>9} | {'best%':>8}{'best used%':>11}")
        for m in models:
            rec = by_model.get(m)
            if not rec:
                continue
            rp = 0.0 if a.no_gpu_stall else None
            cf_ms = cold_full_ms(a.reuse_csv, m, a.frame, a.batch)
            if cf_ms is None:
                continue
            base_q = cf_ms / 1e3 * gpu_rate                         # per-query baseline $ (front+decode)
            res = {}
            for variant in ("kv_reuse", "vt_reuse"):
                nstar, c = break_even(variant, rec, tier, gpu_rate, a.retention_days, a.egress, rp)
                s, F, St = c["saving_per_q"], c["F_usd"], c["storage_total"]
                # per-video cache saving (>0 only above N*); total baseline
                tot_base = sum(N * R * base_q for N in vpm)
                save = used = 0.0; n_used = 0
                for N in vpm:
                    g = N * R * s - F - St
                    if g > 0:
                        save += g; n_used += 1
                res[variant] = (nstar, save, tot_base, n_used)
            (kvN, kvsave, tb, kvn) = res["kv_reuse"]
            (vtN, vtsave, _, vtn) = res["vt_reuse"]
            # best: per video pick the larger positive of {kv,vt}
            best = bestn = 0.0; bn = 0
            kvc = break_even("kv_reuse", rec, tier, gpu_rate, a.retention_days, a.egress, rp)[1]
            vtc = break_even("vt_reuse", rec, tier, gpu_rate, a.retention_days, a.egress, rp)[1]
            for N in vpm:
                gkv = N * R * kvc["saving_per_q"] - kvc["F_usd"] - kvc["storage_total"]
                gvt = N * R * vtc["saving_per_q"] - vtc["F_usd"] - vtc["storage_total"]
                g = max(gkv, gvt)
                if g > 0:
                    best += g; bn += 1
            nv = rec.get("n_vis", 0)
            f = lambda x: f"{x:.2f}" if x != float('inf') else "never"
            print(f"  {m:16}{nv:>7}{f(kvN):>9}{f(vtN):>9} | "
                  f"{100*kvsave/tb:>8.2f}%{100*kvn/n_vid:>8.1f}% | "
                  f"{100*vtsave/tb:>8.2f}%{100*vtn/n_vid:>8.1f}% | "
                  f"{100*best/tb:>7.2f}%{100*bn/n_vid:>10.1f}%")
        print()


if __name__ == "__main__":
    main()
