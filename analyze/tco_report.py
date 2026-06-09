"""Workload-TCO report figures: from total_vpm.csv + measured cost primitives, compute the
per-video-optimal (cache iff it saves) TCO saving across the workload, and plot it.
Outputs results/report/*.png and prints the summary table used in Report.md."""
from __future__ import annotations
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_prices, load_storage_tiers          # noqa: E402
from analyze.breakeven_reuse import load_reuse, break_even   # noqa: E402

VIDEO = "5396384503"
MODELS = ["internvl3.5-8b", "llava-ov-7b", "qwen2.5-vl-7b", "qwen3-vl-8b"]   # IVL-4B/14B dropped from report
SHORT = {"internvl3.5-4b": "IVL-4B", "internvl3.5-8b": "IVL-8B", "internvl3.5-14b": "IVL-14B",
         "llava-ov-7b": "LLaVA-7B", "qwen2.5-vl-7b": "Qwen2.5-7B", "qwen3-vl-8b": "Qwen3-8B"}
REUSE_CSV = "results/nextqa/reuse_real.csv"
OUT = Path("results/report"); OUT.mkdir(parents=True, exist_ok=True)
VCOL = {"kv_reuse": "#d62728", "vt_reuse": "#2ca02c", "baseline": "#888888"}   # unified palette
TCOL = {"local_nvme": "#1f77b4", "s3_same_region": "#ff7f0e"}                  # tier palette


def load_workload(path="total_vpm.csv"):
    """Return [(N=views/month, R=age in months), ...]; R per-video amortizes F & storage."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                N = float(r["views_per_month"]); R = float(r.get("age_months") or 0)
                if R > 0:
                    out.append((N, R))
            except (TypeError, ValueError):
                pass
    return out


def cold_full_ms(model, frame, batch=1):
    vals = [float(d["value_ms"]) for d in csv.DictReader(open(REUSE_CSV))
            if d["model"] == model and int(d["frames"]) == frame and int(d.get("batch") or 1) == batch
            and d["video_id"].split("_b")[0] == VIDEO and d["variant"] == "cold" and d["metric"] == "full"]
    return statistics.median(vals) if vals else None


def _comp(variant, rec, tier, gpu_rate, egress, rp):
    """R-independent components: per-query saving, one-time F, storage PER MONTH (from a 30d call)."""
    c = break_even(variant, rec, tier, gpu_rate, 30.0, egress, rp)[1]
    return c["saving_per_q"], c["F_usd"], c["storage_total"]   # storage_total at 30d = 1-month rent


def saving_pct(pairs, rec, cf_ms, tier, gpu_rate, egress, rp, variant):
    """pairs = [(N views/mo, R age in months)]. Per-video R amortizes F & storage.
    variant in {kv_reuse, vt_reuse, best}.
    Returns (save_pct, video_pct, view_pct):
      save_pct  = Σ_cached max(0,gain) / Σ_all baseline_TCO   (% of total TCO saved)
      video_pct = #cached / #videos                          (% of videos worth caching)
      view_pct  = Σ_cached N·R / Σ_all N·R                    (% of total views those cover)."""
    base_q = cf_ms / 1e3 * gpu_rate
    tot_base = sum(N * R * base_q for N, R in pairs)
    tot_view = sum(N * R for N, R in pairs)
    comps = [_comp("kv_reuse", rec, tier, gpu_rate, egress, rp),
             _comp("vt_reuse", rec, tier, gpu_rate, egress, rp)] if variant == "best" \
        else [_comp(variant, rec, tier, gpu_rate, egress, rp)]
    save = 0.0; used = 0; view = 0.0
    for N, R in pairs:
        g = max(N * R * s - F - St * R for (s, F, St) in comps)
        if g > 0:
            save += g; used += 1; view += N * R
    return 100 * save / tot_base, 100 * used / len(pairs), 100 * view / tot_view


def nstar_at(variant, rec, tier, gpu_rate, egress, rp, R_months):
    """Break-even N at a given retention R (months) — for a representative table value."""
    s, F, Stmo = _comp(variant, rec, tier, gpu_rate, egress, rp)
    if s <= 0:
        return float("inf")
    return (F + Stmo * R_months) / (R_months * s)


def main():
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--frame", type=int, default=128)   # operating point for fig1/fig4/table
    _ap.add_argument("--no-retrieval", action="store_true",
                     help="exclude retrieval cost entirely (rp=0, egress off) -> compute-only ceiling; "
                          "storage rent still counts. Figures get a _noretr suffix.")
    _ap.add_argument("--retention-months", type=float, default=None,
                     help="FIXED retention R (months) for all videos. Default = median age_months "
                          "(realistic retention from the data). Override with a number if needed.")
    _ap.add_argument("--batch", type=int, default=8,
                     help="serving batch size (per-request cost primitives at this batch). Default 8 "
                          "(batch=16 unavailable for IVL-4B/14B/LLaVA). decode collapses as batch grows.")
    _args = _ap.parse_args()
    FR = _args.frame; BATCH = _args.batch
    EG = not _args.no_retrieval          # egress on unless retrieval excluded
    RP = 0.0 if _args.no_retrieval else None   # resource_price=0 -> no GPU-stall / H2D cost
    SUF = "_noretr" if _args.no_retrieval else ""
    RTAG = "retrieval EXCLUDED" if _args.no_retrieval else "retrieval incl."
    _wl = load_workload()                        # [(N views/mo, age in months)]
    Ns = [N for N, _ in _wl]; ages = [a for _, a in _wl]
    R_FIX = _args.retention_months if _args.retention_months else statistics.median(ages)  # default = median age
    pairs = [(N, R_FIX) for N in Ns]             # every video amortized over the same R
    R_med = R_FIX
    prices = load_prices(); gpu_rate = prices["compute"]["gpu_h100_usd_per_hour"] / 3600.0
    tiers = load_storage_tiers()
    recs = load_reuse(REUSE_CSV)
    idx = {}  # (model, frame, batch) -> rec
    for r in recs:
        if r["video_id"].split("_b")[0] == VIDEO:
            idx[(r["model"], r["frames"], r["batch"])] = r

    def rec_of(m, fr): return idx.get((m, fr, BATCH))

    # ---- Fig 1: headline — best TCO saving% per model, per tier ----
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    tnames = list(tiers); x = range(len(MODELS)); w = 0.38
    for i, tn in enumerate(tnames):
        ys = []
        for m in MODELS:
            rec = rec_of(m, FR); cf = cold_full_ms(m, FR, BATCH)
            ys.append(saving_pct(pairs, rec, cf, tiers[tn], gpu_rate, EG, RP, "vt_reuse")[0] if rec and cf else 0)
        bars = ax.bar([xx + (i - 0.5) * w for xx in x], ys, w, label=tn, color=TCOL[tn])
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.3, f"{y:.0f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=15)
    ax.set_ylabel("TCO saved (%)")
    ax.set_title(f"Workload TCO saving — vision-token reuse (per-video cache, frame={FR}, b={BATCH}, R={R_FIX:.0f}mo)")
    ax.legend(title="storage tier"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"fig1_tco_saving_by_model{SUF}.png", dpi=150); plt.close(fig)

    # ---- Fig 2: workload distributions — VPM CCDF + age histogram ----
    sv = sorted(Ns, reverse=True); n = len(sv)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
    a1.loglog(sv, [(i + 1) / n for i in range(n)], lw=1.5)
    a1.set_xlabel("views per month (N)"); a1.set_ylabel("fraction of videos ≥ N")
    a1.set_title(f"VPM (median={statistics.median(Ns):.1f})"); a1.grid(alpha=0.3, which="both")
    a2.hist(ages, bins=40, color="#888"); a2.axvline(statistics.median(ages), ls="--", c="r")
    a2.set_xlabel("age (months) = retention R"); a2.set_ylabel("# videos")
    a2.set_title(f"age (median={statistics.median(ages):.0f}mo; cost uses R={R_FIX:.0f}mo)")
    fig.suptitle(f"Workload — {n:,} videos", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"fig2_vpm_ccdf{SUF}.png", dpi=150); plt.close(fig)

    # ---- Fig 3: saving% vs frame (n_vis sensitivity), local_nvme, best ----
    frames = [16, 32, 64, 128]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for m in MODELS:
        xs, ys = [], []
        for fr in frames:
            rec = rec_of(m, fr); cf = cold_full_ms(m, fr, BATCH)
            if rec and cf:
                xs.append(rec["n_vis"]); ys.append(saving_pct(pairs, rec, cf, tiers["local_nvme"], gpu_rate, EG, RP, "vt_reuse")[0])
        if xs: ax.plot(xs, ys, "o-", label=SHORT[m], lw=1.5)
    ax.set_xlabel("n_vis (vision tokens)"); ax.set_ylabel("TCO saved (%)")
    ax.set_title("vision-token reuse: TCO saving vs n_vis (local_nvme)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"fig3_saving_vs_nvis{SUF}.png", dpi=150); plt.close(fig)

    # ---- Fig 4/5: kv vs vt saving% per model, one per tier (retrieval INCLUDED) ----
    for tn, fname, note in [("s3_same_region", "fig4_kv_vs_vt_s3.png", " — KV dies on the slow tier"),
                            ("local_nvme", "fig5_kv_vs_vt_local.png", " — fast tier, KV reuse wins")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for i, (variant, lab) in enumerate([("kv_reuse", "KV reuse"), ("vt_reuse", "vision-token reuse")]):
            ys = []
            for m in MODELS:
                rec = rec_of(m, FR); cf = cold_full_ms(m, FR, BATCH)
                ys.append(saving_pct(pairs, rec, cf, tiers[tn], gpu_rate, EG, RP, variant)[0] if rec and cf else 0)
            bars = ax.bar([xx + (i - 0.5) * w for xx in x], ys, w, label=lab, color=VCOL[variant])
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y + 0.2, f"{y:.0f}", ha="center", fontsize=8)
        ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=15)
        ax.set_ylabel("TCO saved (%)"); ax.set_title(f"KV vs vision-token reuse on {tn} (frame={FR}, b={BATCH}, retrieval incl.){note}")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / fname.replace(".png", SUF + ".png"), dpi=150); plt.close(fig)

    # ---- Fig 6/7 (§3.4): per TIER, KV vs vt with retrieval EXCLUDED (compare vs §3.2 incl) ----
    series = [("KV reuse", "kv_reuse", "#d62728"), ("vision-token reuse", "vt_reuse", "#2ca02c")]
    for tn, fname in [("local_nvme", "fig6_retr_local.png"), ("s3_same_region", "fig7_retr_s3.png")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for i, (lab, var, col) in enumerate(series):
            ys = []
            for m in MODELS:
                rec = rec_of(m, FR); cf = cold_full_ms(m, FR, BATCH)
                ys.append(saving_pct(pairs, rec, cf, tiers[tn], gpu_rate, False, 0.0, var)[0] if rec and cf else 0)
            bars = ax.bar([xx + (i - 0.5) * w for xx in x], ys, w, label=lab, color=col)
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y + 0.3, f"{y:.0f}", ha="center", fontsize=8)
        ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=15)
        ax.set_ylabel("TCO saved (%)")
        ax.set_title(f"{tn}: KV vs vision-token reuse — retrieval EXCLUDED (frame={FR}, b={BATCH}, R={R_FIX:.0f}mo)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)

    # ---- summary table (markdown): vt_reuse = MAIN, kv_reuse = comparison ----
    print(f"\n### Summary — vt_reuse MAIN (frame={FR}, b={BATCH}, R={R_FIX:.0f}mo, {RTAG})\n")
    print(f"| model | n_vis | local vt% | s3 vt% | [cmp] local kv% | [cmp] s3 kv% |")
    print("|---|---|---|---|---|---|")
    for m in MODELS:
        rec = rec_of(m, FR); cf = cold_full_ms(m, FR, BATCH)
        if not rec or not cf: continue
        lv = saving_pct(pairs, rec, cf, tiers["local_nvme"], gpu_rate, EG, RP, "vt_reuse")[0]
        sv = saving_pct(pairs, rec, cf, tiers["s3_same_region"], gpu_rate, EG, RP, "vt_reuse")[0]
        lk = saving_pct(pairs, rec, cf, tiers["local_nvme"], gpu_rate, EG, RP, "kv_reuse")[0]
        sk = saving_pct(pairs, rec, cf, tiers["s3_same_region"], gpu_rate, EG, RP, "kv_reuse")[0]
        print(f"| {SHORT[m]} | {rec['n_vis']} | {lv:.1f}% | {sv:.1f}% | {lk:.1f}% | {sk:.1f}% |")

    # ---- Break-even table: s3 + retrieval INCLUDED, vt vs kv side-by-side (one table to compare) ----
    tot_v = sum(Ns)
    def be_cov(variant, tier):
        ne = nstar_at(variant, rec_of_m, tier, gpu_rate, True, None, R_FIX)
        if ne == float("inf"):
            return "never", 0.0, 0.0
        cached = [N for N in Ns if N >= ne]
        return f"{ne:.2f}", 100 * len(cached) / len(Ns), 100 * sum(cached) / tot_v
    tier_s3 = tiers["s3_same_region"]
    print(f"\n### Break-even — s3_same_region, retrieval INCLUDED (vt vs kv, frame={FR}, b={BATCH}, R={R_FIX:.0f}mo)")
    print(f"(N_even = views/month to beat recompute; %vid = share of {len(Ns):,} videos with vpm ≥ N_even; "
          "%view = their share of total views)")
    print("| model | n_vis | vt N_even | vt %vid | vt %view | kv N_even | kv %vid | kv %view |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        rec_of_m = rec_of(m, FR)
        if not rec_of_m: continue
        vne, vvid, vview = be_cov("vt_reuse", tier_s3)
        kne, kvid, kview = be_cov("kv_reuse", tier_s3)
        print(f"| {SHORT[m]} | {rec_of_m['n_vis']} | {vne} | {vvid:.0f}% | {vview:.1f}% | "
              f"{kne} | {kvid:.0f}% | {kview:.1f}% |")

    print(f"\nfigures -> {OUT}/")


if __name__ == "__main__":
    main()
