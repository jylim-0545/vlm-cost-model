"""Break-even from REAL reuse measurements (measure/reuse_real.py), not the
encode/prefill/decode-split analytical model.

Unlike analyze/price_model.py (which assumes the reuse front-end is free — kv_reuse
pays only decode), this uses the MEASURED front-ends, so the real vLLM cache-hit
overhead (kv_warm ~50-120ms) and the real encode-skip prefill (tok_inject) are kept:

  per-query front-end (GPU-time):
    baseline    b = cold_ttft                       (encode + prefill, measured)
    vt_reuse r = tok_inject                       (encode skipped, real vision prefill)
    kv_reuse    r = kv_warm                           (encode+prefill skipped, real warm path)
  decode is common to all three -> CANCELS in b-r (we only need front-ends here).

  one-time store cost F (paid once):
    vt_reuse F = encode      = cold_ttft - tok_inject   (the saved vision-tower work)
    kv_reuse    F = encode+pref = cold_ttft                  (build the full KV once)

  per-access retrieval (2 hops, CLAUDE.md Section 5):
    storage->DRAM = tier.network_cost_usd(bytes)            (COMPUTED from bytes + tier)
    DRAM->GPU     = h2d_s * gpu_rate                         (MEASURED H2D, GPU stalls)

  break-even rate  N* (queries/month) = (F + storage_total) / (R * per_query_saving)
    per_query_saving = (b - r)*gpu_rate - retrieval_per_access
    R = retention_days/30 ;  inf if saving <= 0 (reuse never beats recompute).

Usage:
  python -m analyze.breakeven_reuse                       # results/reuse_real.csv, all tiers
  python -m analyze.breakeven_reuse --csv <path> --retention-days 30
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_prices, load_storage_tiers  # noqa: E402

# (variant, metric) we need, and a short name for each
WANT = {
    ("cold", "ttft"): "cold_ttft",
    ("vt_reuse", "ttft_inject"): "tok_inject",
    ("kv_reuse", "ttft_warm"): "kv_warm",
    ("vt_reuse", "h2d_tok"): "h2d_tok",
    ("kv_reuse", "h2d_kv"): "h2d_kv",
}


def load_reuse(path: str) -> list[dict]:
    """One record per (model, video_id, frames, batch) with median ms + bytes/n_vis."""
    vals: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[tuple, dict] = {}
    with open(path) as f:
        for d in csv.DictReader(f):
            b = int(d.get("batch") or 1)
            key = (d["model"], d["video_id"], int(d["frames"]), b)
            short = WANT.get((d["variant"], d["metric"]))
            if short:
                vals[key][short].append(float(d["value_ms"]))
            meta[key] = d
    recs = []
    for key, mv in vals.items():
        m = meta[key]
        rec = {"model": key[0], "video_id": key[1], "frames": key[2], "batch": key[3],
               "n_vis": int(m["n_vis"]), "token_bytes": int(m["token_bytes"]),
               "kv_bytes": int(m["kv_bytes"]),
               "duration_s": float(m["duration_s"]) if m.get("duration_s") else None}
        for short, lst in mv.items():
            rec[short] = statistics.median(lst)
        recs.append(rec)
    recs.sort(key=lambda r: (r["model"], r["batch"], r["n_vis"]))
    return recs


def break_even(variant: str, rec: dict, tier, gpu_rate: float, retention_days: float,
               include_egress: bool, resource_price: float | None = None) -> tuple[float, dict]:
    """Return (N* queries/month, components dict). inf if reuse never pays off.

    resource_price = $/s of the resource that STALLS during retrieval (storage->DRAM
    and DRAM->GPU H2D). Defaults to gpu_rate (assume the H100 idles fully during the
    fetch). Pass 0 to model retrieval fully overlapped with compute (no GPU stall) —
    then the only retrieval cost left is egress."""
    rp = gpu_rate if resource_price is None else resource_price
    R = retention_days / 30.0
    b_front_s = rec["cold_ttft"] / 1e3
    if variant == "vt_reuse":
        r_front_s = rec["tok_inject"] / 1e3
        bytes_ = rec["token_bytes"]
        h2d_s = rec["h2d_tok"] / 1e3
        F_usd = (b_front_s - r_front_s) * gpu_rate          # encode only, once
    else:  # kv_reuse
        r_front_s = rec["kv_warm"] / 1e3
        bytes_ = rec["kv_bytes"]
        h2d_s = rec["h2d_kv"] / 1e3
        F_usd = b_front_s * gpu_rate                        # encode+prefill, once
    retrieval = tier.network_cost_usd(bytes_, rp, include_egress) + h2d_s * rp
    saving = (b_front_s - r_front_s) * gpu_rate - retrieval
    storage_total = tier.storage_cost_usd(bytes_, retention_days)
    comp = {"F_usd": F_usd, "saving_per_q": saving, "storage_total": storage_total,
            "retrieval_per_q": retrieval, "front_saving_usd": (b_front_s - r_front_s) * gpu_rate}
    if saving <= 0:
        return math.inf, comp
    return (F_usd + storage_total) / (R * saving), comp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/reuse_real.csv")
    ap.add_argument("--retention-days", type=float, default=None)
    ap.add_argument("--tier", help="single tier; omit to sweep all")
    ap.add_argument("--models", help="comma-separated model keys to keep (default: all)")
    ap.add_argument("--no-gpu-stall", action="store_true",
                    help="model retrieval as fully overlapped with compute (resource_price=0); "
                         "only egress remains in the retrieval cost")
    ap.add_argument("--by-video", action="store_true",
                    help="one row per (model,video,frames); default aggregates videos by (model,n_vis)")
    a = ap.parse_args()

    prices = load_prices()
    tiers = load_storage_tiers()
    gpu_rate = prices["compute"]["gpu_h100_usd_per_hour"] / 3600.0
    rp = 0.0 if a.no_gpu_stall else None       # None -> break_even uses gpu_rate
    retention = a.retention_days or prices["defaults"]["retention_time_days"]
    sweep = [tiers[a.tier]] if a.tier else list(tiers.values())
    recs = load_reuse(a.csv)
    if a.models:
        keep = set(a.models.split(","))
        recs = [r for r in recs if r["model"] in keep]
    # drop OOM-contaminated configs: a batch that overflowed KV cache → preemption, not
    # batch efficiency (detected as kv_warm ≈ cold, i.e. cold−kv ≈ 0). e.g. InternVL-8B 128f×b16.
    recs = [r for r in recs if not (r.get("kv_warm") and r.get("cold_ttft")
                                    and (r["cold_ttft"] - r["kv_warm"]) < 0.3 * r["cold_ttft"])]
    # aggregate per-video repeats into one row per (model, frames, BATCH): median over videos.
    # (InternVL = 256 tok/frame fixed → a given frame count gives fixed n_vis across videos.)
    if not a.by_video:
        grp: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for r in recs:
            k = (r["model"], r["frames"], r["batch"])
            for m in ("cold_ttft", "tok_inject", "kv_warm", "h2d_tok", "h2d_kv",
                      "n_vis", "token_bytes", "kv_bytes"):
                if r.get(m) is not None:
                    grp[k][m].append(r[m])
        agg = []
        for (model, frames, batch), mv in grp.items():
            rec = {"model": model, "frames": frames, "batch": batch, "video_id": f"agg_n{len(mv['n_vis'])}"}
            for m, lst in mv.items():
                rec[m] = int(statistics.median(lst)) if m in ("n_vis", "token_bytes", "kv_bytes") \
                    else statistics.median(lst)
            agg.append(rec)
        recs = sorted(agg, key=lambda r: (r["model"], r["batch"], r["n_vis"]))
    if not recs:
        print(f"no records in {a.csv}"); return

    print(f"\nBreak-even query RATE (queries/MONTH) from REAL reuse measurements")
    print(f"retention={retention:g}d  gpu=${prices['compute']['gpu_h100_usd_per_hour']}/h  "
          f"gpu_stall={'OFF (retrieval overlapped)' if a.no_gpu_stall else 'ON (H100 idles during fetch)'}  "
          f"(N* = queries/month above which reuse beats recompute; 'never' = saving<=0)")
    for tier in sweep:
        print(f"\n  ── {tier.name}  (bw={tier.bandwidth_gbps}GB/s  egress=${tier.egress_price_usd_per_gb}/GB  "
              f"store=${tier.usd_per_gb_month}/GB-mo) ──")
        print(f"  {'model':<16} {'n_vis':>6} {'frm':>4} {'bat':>4} | {'kv N*':>8} {'kv(noEg)':>9} | {'vt N*':>8} {'vt(noEg)':>10}")
        last = None
        for rec in sorted(recs, key=lambda r: (r["model"], r["batch"], r["n_vis"])):
            tag = (rec["model"], rec["batch"])
            if last is not None and tag != last:
                print("  " + "-" * 64)
            last = tag
            def be(v, eg):
                x, _ = break_even(v, rec, tier, gpu_rate, retention, eg, resource_price=rp)
                return "never" if math.isinf(x) else f"{x:.1f}"
            print(f"  {rec['model']:<16} {rec['n_vis']:>6} {rec['frames']:>4} {rec['batch']:>4} | "
                  f"{be('kv_reuse', True):>8} {be('kv_reuse', False):>9} | "
                  f"{be('vt_reuse', True):>8} {be('vt_reuse', False):>10}")

    # detail for the largest n_vis on the canonical tier (object_same_region)
    tier = tiers.get("object_same_region", sweep[0])
    rec = recs[-1]
    print(f"\n  detail @ n_vis={rec['n_vis']} ({rec['frames']}f), tier={tier.name}, retention={retention:g}d:")
    for v in ("vt_reuse", "kv_reuse"):
        x, c = break_even(v, rec, tier, gpu_rate, retention, include_egress=False, resource_price=rp)
        nstar = "never" if math.isinf(x) else f"{x:.1f}/mo"
        print(f"    {v:11s}: F=${c['F_usd']*1e6:.2f}µ  store_total=${c['storage_total']*1e6:.2f}µ  "
              f"retrieval/q=${c['retrieval_per_q']*1e6:.3f}µ  saving/q=${c['saving_per_q']*1e6:.3f}µ  -> N*={nstar}")


if __name__ == "__main__":
    main()
