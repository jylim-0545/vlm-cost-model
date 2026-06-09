"""{TITLE} result figures (frame=64) from results/reuse_real.csv.

Fig 1  TTFT (batch=1): baseline / kv_reuse(+H2D) / vt_reuse  bar
Fig 2  Throughput vs batch[1,4,8,16]: 3 cases (H2D hidden, retrieval excluded)
Fig 3  Break-even N* (batch=1): vt_reuse & kv_reuse x 3 tiers x H2D on/off

  TTFT:  baseline = cold_ttft ; kv_reuse = kv_warm + h2d_kv (retrieval reflected) ;
         vt_reuse = vt_inject (embeds H2D marginal -> excluded)
  Throughput (tok/s) = decode_tokens*1000 / full_per_request  (full = ttft+decode, retrieval hidden)
  Break-even uses analyze.breakeven_reuse.break_even; H2D-on = GPU stalls during fetch,
         H2D-off = retrieval pipelined/hidden (resource_price=0).
"""
from __future__ import annotations
import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_prices, load_storage_tiers  # noqa: E402
from analyze.breakeven_reuse import break_even        # noqa: E402

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--model", default="internvl3.5-8b")
_ap.add_argument("--frame", type=int, default=128)
_ap.add_argument("--dataset", default="nextqa")
_a = _ap.parse_args()
MODEL = _a.model; FRAME = _a.frame; DATASET = _a.dataset
TAG = {"internvl3.5-8b": "internvl8b", "internvl3.5-4b": "internvl4b", "internvl3.5-14b": "internvl14b",
       "qwen2.5-vl-7b": "qwen25", "qwen3-vl-8b": "qwen3vl8b", "llava-ov-7b": "llavaov"}.get(MODEL, MODEL.replace(".", "").replace("-", ""))
TITLE = {"internvl3.5-8b": "InternVL-8B", "internvl3.5-4b": "InternVL-4B", "internvl3.5-14b": "InternVL-14B",
         "qwen2.5-vl-7b": "Qwen2.5-VL", "qwen3-vl-8b": "Qwen3-VL-8B", "llava-ov-7b": "LLaVA-OV-7B"}.get(MODEL, MODEL)
DECODE = 256
CSV = os.path.expanduser(f"~/VLM/results/{DATASET}/reuse_real.csv")
OUT = Path(os.path.expanduser(f"~/VLM/results/{DATASET}/{TAG}")); OUT.mkdir(parents=True, exist_ok=True)

# ---- pin to ONE video (stem) so n_vis is consistent per frame. Multi-video models (Qwen,
# dynamic tok/frame) otherwise mix videos with DIFFERENT n_vis per frame -> the frame sweep
# folds back on the n_vis x-axis. Fixed-tok/frame models (InternVL/LLaVA) are unaffected (same
# n_vis across videos) but the filter is harmless. Pick the stem with the most rows = the
# batch-sweep video (the only one with batch>1 coverage). ----
import re as _re
def _stem(vid):  # strip the _b{N} batch suffix
    return _re.sub(r"_b\d+$", "", vid or "")
_cnt = defaultdict(int)
with open(CSV) as _f:
    for _r in csv.DictReader(_f):
        if _r["model"] == MODEL:
            _cnt[_stem(_r["video_id"])] += 1
VIDEO = max(_cnt, key=_cnt.get) if _cnt else None
print(f"[fig] {MODEL}: pinned video stem = {VIDEO} ({_cnt.get(VIDEO,0)} rows)")

# ---- load: (batch) -> metric medians for {TITLE} at FRAME ----
WANT = {("cold", "ttft"): "cold_ttft", ("cold", "full"): "cold_full",
        ("kv_reuse", "ttft_warm"): "kv_warm", ("kv_reuse", "full_warm"): "kv_full",
        ("vt_reuse", "ttft_inject"): "vt_ttft", ("vt_reuse", "full_inject"): "vt_full",
        ("kv_reuse", "h2d_kv"): "h2d_kv", ("vt_reuse", "h2d_tok"): "h2d_tok"}
vals = defaultdict(lambda: defaultdict(list)); meta = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["model"] != MODEL or int(r["frames"]) != FRAME or _stem(r["video_id"]) != VIDEO:
            continue
        b = int(r["batch"] or 1)
        s = WANT.get((r["variant"], r["metric"]))
        if s:
            vals[b][s].append(float(r["value_ms"]))
        meta[b] = r
B = {b: {k: statistics.median(v) for k, v in mv.items()} for b, mv in vals.items()}
for b in B:
    B[b]["n_vis"] = int(meta[b]["n_vis"]); B[b]["kv_bytes"] = int(meta[b]["kv_bytes"])
    B[b]["token_bytes"] = int(meta[b]["token_bytes"])
batches = sorted(B)
if 1 not in B:                       # no data for this (model, FRAME, pinned video) -> skip, don't crash
    print(f"[fig] {TITLE} frame={FRAME}: NO batch-1 data for video {VIDEO} (skipping figures)")
    sys.exit(0)
b1 = B[1]
print(f"[fig] {TITLE} frame={FRAME} batches={batches} n_vis={b1['n_vis']}")

# ===== Fig 1: TTFT vs n_vis (batch=1, frame sweep), H2D excluded =====
FB = defaultdict(lambda: defaultdict(list)); FBn = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["model"] != MODEL or int(r["batch"] or 1) != 1 or _stem(r["video_id"]) != VIDEO:
            continue
        nf = int(r["frames"]); s = WANT.get((r["variant"], r["metric"]))
        if s:
            FB[nf][s].append(float(r["value_ms"]))
        FBn[nf] = int(r["n_vis"])
fs = [f for f in sorted(FB) if f <= FRAME]; xs = [FBn[nf] for nf in fs]   # frame sweep <= FRAME (drop overflow frames)
def sval(nf, key, h2d=None):                          # TTFT + (optional) DRAM->GPU H2D
    v = statistics.median(FB[nf][key])
    return v + (statistics.median(FB[nf][h2d]) if h2d else 0.0)
fig, ax = plt.subplots(figsize=(5.0, 3.6))
for case, key, h2d, c in [("baseline", "cold_ttft", None, "#888"),
                          ("kv_reuse (+H2D)", "kv_warm", "h2d_kv", "#d62728"),
                          ("vt_reuse (+H2D)", "vt_ttft", "h2d_tok", "#2ca02c")]:
    ax.plot(xs, [sval(nf, key, h2d) for nf in fs], "o-", label=case, color=c, lw=1.8)
ax.set_xlabel("n_vis (vision tokens)"); ax.set_ylabel("TTFT incl. H2D (ms)")
ax.set_title(f"{TITLE} TTFT vs n_vis (batch=1, H2D incl.)")
ax.legend(); ax.grid(alpha=0.3)
ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(xs)
ax2.set_xticklabels(fs); ax2.set_xlabel("n_frame")
fig.tight_layout(); fig.savefig(OUT / f"fig1_ttft.png", dpi=150); plt.close(fig)

# ===== Fig 2: Throughput vs batch (3 cases, retrieval hidden) =====
def tput(full_ms):  # decode tokens per second, end-to-end (ttft+decode), retrieval excluded
    return DECODE * 1000.0 / full_ms
bf = [b for b in batches if b <= 8]                    # exclude batch=16
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for case, key, c in [("baseline", "cold_full", "#888"),
                     ("kv_reuse", "kv_full", "#d62728"),
                     ("vt_reuse", "vt_full", "#2ca02c")]:
    ys = [tput(B[b][key]) for b in bf]
    ax.plot(bf, ys, "o-", label=case, color=c)
ax.set_xlabel("batch size"); ax.set_ylabel("throughput (tok/s)")
ax.set_xticks(bf); ax.set_title(f"{TITLE} throughput ({FRAME}f, retrieval hidden)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / f"fig2_throughput.png", dpi=150); plt.close(fig)

# ===== Fig 6: TPOT vs batch (3 cases ~ identical: decode is common, reuse can't touch it) =====
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for case, tk, fk, c in [("baseline", "cold_ttft", "cold_full", "#888"),
                        ("kv_reuse", "kv_warm", "kv_full", "#d62728"),
                        ("vt_reuse", "vt_ttft", "vt_full", "#2ca02c")]:
    ys = [(B[b][fk] - B[b][tk]) / DECODE for b in bf]
    ax.plot(bf, ys, "o-", label=case, color=c, lw=1.6)
ax.set_xlabel("batch size"); ax.set_ylabel("TPOT (ms / output token)")
ax.set_xticks(bf)
ax.set_title(f"{TITLE} TPOT ({FRAME}f) — 3 cases overlap (decode common)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / f"fig6_tpot.png", dpi=150); plt.close(fig)

# ===== Fig 7: throughput vs batch, BY n_frame (does the trend change with frames?) =====
FF = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # frame -> batch -> metric
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["model"] != MODEL or _stem(r["video_id"]) != VIDEO:
            continue
        nf = int(r["frames"]); bb = int(r["batch"] or 1)
        for (vv, mm), kk in [(("cold", "full"), "cold_full"),
                             (("kv_reuse", "full_warm"), "kv_full"),
                             (("vt_reuse", "full_inject"), "vt_full")]:
            if (r["variant"], r["metric"]) == (vv, mm):
                FF[nf][bb][kk].append(float(r["value_ms"]))
frames_all = [f for f in sorted(FF) if f <= FRAME]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (case, key) in zip(axes, [("baseline", "cold_full"), ("kv_reuse", "kv_full"), ("vt_reuse", "vt_full")]):
    for nf in frames_all:
        bs = [b for b in sorted(FF[nf]) if b <= 8 and key in FF[nf][b]]
        ys = [DECODE * 1000.0 / statistics.median(FF[nf][b][key]) for b in bs]
        ax.plot(bs, ys, "o-", label=f"{nf}f", lw=1.6)
    ax.set_title(case); ax.set_xlabel("batch size"); ax.set_xticks([1, 4, 8]); ax.grid(alpha=0.3)
axes[0].set_ylabel("throughput (tok/s)"); axes[0].legend(title="n_frame", fontsize=7)
fig.suptitle(f"{TITLE} throughput vs batch, by n_frame (retrieval hidden)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / f"fig7_throughput_by_frame.png", dpi=150); plt.close(fig)
print("\n=== Fig7 throughput by n_frame (baseline, tok/s) ===")
print(f"  {'n_frame':>8}" + "".join(f"{'b'+str(b):>8}" for b in [1, 4, 8]))
for nf in frames_all:
    row = f"  {nf:>8}"
    for b in [1, 4, 8]:
        row += f"{(DECODE*1000.0/statistics.median(FF[nf][b]['cold_full'])) if b in FF[nf] else 0:>8.0f}"
    print(row)

# ===== Fig 3: Break-even N* (batch=1) x tier x H2D on/off =====
prices = load_prices(); tiers = load_storage_tiers()
gpu_rate = prices["compute"]["gpu_h100_usd_per_hour"] / 3600.0
retention = prices["defaults"]["retention_time_days"]
import math
import numpy as np
tier_names = [t for t in tiers if t == "s3_same_region"] or list(tiers)   # Fig3/5: S3 only
R = retention / 30.0
BL = [b for b in (1, 4, 8) if b in B]                 # batch axis for Fig3/4/5
Ns = np.linspace(0, 60, 200)
Np = np.linspace(0.5, 60, 200)

# ALL of Fig3/4/5 EXCLUDE H2D / retrieval stall (resource_price = 0): reuse cost is
# compute front-end + storage rent only (retrieval assumed pipelined/hidden).
def rec_of(b):
    d = B[b]
    return {"cold_ttft": d["cold_ttft"], "kv_warm": d["kv_warm"], "tok_inject": d["vt_ttft"],
            "h2d_tok": 0.0, "h2d_kv": 0.0, "token_bytes": d["token_bytes"], "kv_bytes": d["kv_bytes"]}
def be0(var, b, tier):
    n, _ = break_even(var, rec_of(b), tier, gpu_rate, retention, include_egress=True, resource_price=0.0)
    return n
def tco(var, tier, d, N, retr=False):                  # retr=False: H2D/retrieval excluded
    g = gpu_rate
    if var == "baseline":
        return N * R * (d["cold_full"] / 1000 * g)
    if var == "kv_reuse":
        F = d["cold_ttft"] / 1000 * g
        rq = d["kv_full"] / 1000 * g
        if retr: rq += tier.network_cost_usd(d["kv_bytes"], g) + d["h2d_kv"] / 1000 * g
        return F + N * R * rq + tier.storage_cost_usd(d["kv_bytes"], retention)
    F = (d["cold_ttft"] - d["vt_ttft"]) / 1000 * g
    rq = d["vt_full"] / 1000 * g
    if retr: rq += tier.network_cost_usd(d["token_bytes"], g) + d["h2d_tok"] / 1000 * g
    return F + N * R * rq + tier.storage_cost_usd(d["token_bytes"], retention)

# ===== Fig 3: break-even N* vs batch, per tier (H2D excluded) =====
fig, axes = plt.subplots(1, len(tier_names), figsize=(4.0*len(tier_names), 3.6), sharey=True)
if len(tier_names) == 1: axes = [axes]
for ax, t in zip(axes, tier_names):
    tier = tiers[t]
    for var, c in [("vt_reuse", "#2ca02c"), ("kv_reuse", "#d62728")]:
        ys = [be0(var, b, tier) for b in BL]
        ys = [None if (y is None or math.isinf(y)) else y for y in ys]
        ax.plot(BL, ys, "o-", color=c, label=var, lw=1.8)
        for b, y in zip(BL, ys):
            if y is not None: ax.annotate(f"{y:.1f}", (b, y), fontsize=7, color=c)
    ax.set_title(f"{t}\n(bw={tier.bandwidth_gbps}GB/s, ${tier.usd_per_gb_month}/GB-mo)", fontsize=9)
    ax.set_xlabel("batch size"); ax.set_xticks(BL); ax.grid(alpha=0.3)
axes[0].set_ylabel("break-even N* (queries/month)"); axes[0].legend(fontsize=8)
fig.suptitle(f"{TITLE} break-even vs batch ({FRAME}f, H2D excl., retention={retention:g}d)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / f"fig3_breakeven.png", dpi=150); plt.close(fig)

# (Fig 4 TCO-vs-N removed per request; tco() kept for Fig 5 saving %.)

# ===== Fig 5: TCO saving vs baseline (%), grid = batch x tier (H2D excluded) =====
fig, axes = plt.subplots(len(BL), len(tier_names), figsize=(4.0*len(tier_names), 3.0*len(BL)),
                         sharex=True, sharey=True, squeeze=False)
for i, b in enumerate(BL):
    d = B[b]
    for j, t in enumerate(tier_names):
        ax = axes[i][j]; tier = tiers[t]
        base = tco("baseline", tier, d, Np)
        for var, c in [("vt_reuse", "#2ca02c"), ("kv_reuse", "#d62728")]:
            pct = (base - tco(var, tier, d, Np)) / base * 100.0
            ax.plot(Np, pct, color=c, label=var, lw=1.6)
        ax.axhline(0, color="k", ls=":", lw=0.8)
        if i == 0: ax.set_title(t, fontsize=9)
        if j == 0: ax.set_ylabel(f"batch={b}\nsaving (%)", fontsize=8)
        if i == len(BL)-1: ax.set_xlabel("N (queries/month)")
        ax.grid(alpha=0.3); ax.set_ylim(-20, 60)
axes[0][0].legend(fontsize=7)
fig.suptitle(f"{TITLE} TCO saving vs baseline ({FRAME}f, H2D excl.)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / f"fig5_saving.png", dpi=150); plt.close(fig)

# single-batch saving curves at the serving batch (8 if available, else max measured)
BB = 8 if 8 in B else max(B)
d = B[BB]

# ===== Fig 5_1 (local) / 5_2 (s3): TCO saving vs N, one figure per tier, L=retr EXCL | R=retr INCL =====
for fnum, tname, ylim in [("5_1", "local_nvme", (-20, 70)), ("5_2", "s3_same_region", (-100, 70))]:
    if tname not in tiers:
        continue
    tier = tiers[tname]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, (retr, plab) in zip(axes, [(False, "retrieval EXCLUDED"), (True, "retrieval INCLUDED")]):
        base = tco("baseline", tier, d, Np, retr=retr)
        for var, c in [("vt_reuse", "#2ca02c"), ("kv_reuse", "#d62728")]:
            pct = (base - tco(var, tier, d, Np, retr=retr)) / base * 100.0
            ax.plot(Np, pct, color=c, label=var, lw=1.8)
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("N (queries/month)"); ax.set_title(plab, fontsize=9)
        ax.set_ylim(*ylim); ax.grid(alpha=0.3)
    axes[0].set_ylabel("TCO saving vs baseline (%)"); axes[0].legend(fontsize=8)
    fig.suptitle(f"{TITLE} TCO saving — {tname} (batch={BB}, {FRAME}f)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"fig{fnum}_saving_{tname}_b{BB}.png", dpi=150); plt.close(fig)

# ===== Fig 8: TTFT latency breakdown (batch=1) — compute / storage->DRAM / H2D, per tier =====
# Both tiers shown (retrieval is tier-dependent). H2D is tiny -> annotated with an arrow.
d = B[1]
TIER_ORD = [t for t in ("local_nvme", "s3_same_region") if t in tiers] + \
           [t for t in tiers if t not in ("local_nvme", "s3_same_region")]
short = lambda t: t.replace("_same_region", "").replace("local_", "")
# baseline first (recompute: pure compute = encode+prefill, NO storage/retrieval), then kv/vt x tier.
comps = [("baseline\nrecompute", d["cold_ttft"], 0.0, 0.0)]  # (label, compute, storage->DRAM, H2D)
for var, comp_key, byte_key, h2d_key in [("kv_reuse", "kv_warm", "kv_bytes", "h2d_kv"),
                                         ("vt_reuse", "vt_ttft", "token_bytes", "h2d_tok")]:
    for t in TIER_ORD:
        sd = tiers[t].retrieval_time_s(int(d[byte_key])) * 1000.0
        comps.append((f"{'kv' if var=='kv_reuse' else 'vt'}\n{short(t)}", d[comp_key], sd, d[h2d_key]))
ymax = max(c[1] + c[2] + c[3] for c in comps)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
cc = {"compute": "#4c72b0", "sd": "#dd8452", "h2d": "#c44e52"}
for i, (lab, comp, sd, h2d) in enumerate(comps):
    ax.bar(i, comp, color=cc["compute"], label="compute (encode/prefill)" if i == 0 else None)
    ax.bar(i, sd, bottom=comp, color=cc["sd"], label="storage→DRAM" if i == 0 else None)
    ax.bar(i, h2d, bottom=comp + sd, color=cc["h2d"], label="H2D (DRAM→GPU)" if i == 0 else None)
    total = comp + sd + h2d
    ax.text(i, total + ymax * 0.015, f"{total:.0f}ms\n{(sd + h2d) / total * 100:.0f}% retr",
            ha="center", va="bottom", fontsize=8, fontweight="bold")
    if h2d > 0.5:   # baseline has no H2D
        ax.annotate(f"H2D {h2d:.0f}ms", xy=(i, total), xytext=(i + 0.30, total + ymax * 0.12),
                    fontsize=7, color=cc["h2d"], ha="left",
                    arrowprops=dict(arrowstyle="->", color=cc["h2d"], lw=0.9))
ax.axvline(0.5, color="k", ls=":", lw=0.7, alpha=0.5)   # separate baseline from reuse variants
ax.set_xticks(range(len(comps))); ax.set_xticklabels([c[0] for c in comps])
ax.set_ylabel("TTFT breakdown (ms)"); ax.set_ylim(0, ymax * 1.32)
ax.set_title(f"{TITLE} TTFT latency breakdown (batch=1, {FRAME}f)")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig8_breakdown.png", dpi=150); plt.close(fig)
print("\n=== Fig8 TTFT breakdown (batch=1, ms) ===")
print(f"  {'var/tier':<12}{'compute':>9}{'sto→DRAM':>10}{'H2D':>6}{'total':>8}{'retr%':>7}")
for lab, comp, sd, h2d in comps:
    tot = comp + sd + h2d
    print(f"  {lab.replace(chr(10), '/'):<12}{comp:>9.0f}{sd:>10.0f}{h2d:>6.0f}{tot:>8.0f}{(sd + h2d) / tot * 100:>6.0f}%")

# ---- console summary ----
print("\n=== Fig1 TTFT vs n_vis (batch=1, ms, H2D excl.) ===")
print(f"  {'n_vis':>7}{'frame':>6}{'baseline':>10}{'kv_reuse':>10}{'vt_reuse':>10}")
for nf in fs:
    print(f"  {FBn[nf]:>7}{nf:>6}{statistics.median(FB[nf]['cold_ttft']):>10.0f}"
          f"{statistics.median(FB[nf]['kv_warm']):>10.0f}{statistics.median(FB[nf]['vt_ttft']):>10.0f}")
print("\n=== Fig2 throughput (tok/s) ===")
print(f"  {'batch':>6}{'baseline':>10}{'kv_reuse':>10}{'vt_reuse':>10}")
for b in batches:
    print(f"  {b:>6}{tput(B[b]['cold_full']):>10.0f}{tput(B[b]['kv_full']):>10.0f}{tput(B[b]['vt_full']):>10.0f}")
print("\n=== Fig3 break-even N* (H2D excl.) — vt / kv ===")
print(f"  {'tier':<18}" + "".join(f"{'b'+str(b)+' vt':>9}{'b'+str(b)+' kv':>9}" for b in BL))
for t in tier_names:
    row = f"  {t:<18}"
    for b in BL:
        for var in ("vt_reuse", "kv_reuse"):
            n = be0(var, b, tiers[t]); row += f"{('never' if (n is None or math.isinf(n)) else f'{n:.1f}'):>9}"
    print(row)
print(f"\n[fig] saved -> {OUT}/fig{{1..5}}_internvl8b.png")
