"""Figure: OUR kv_reuse vs LMCache kv_reuse (head-to-head), LLaVA-OV-7B, b1, H100.

Reads:
  results/lmcache/reuse_lmcache.csv      (vLLM 0.18: cold, kv_ours[gpu_resident], kv_lmcache[dram])
  results/nextqa/reuse_real.csv          (vLLM 0.22: our kv_reuse ttft_warm + measured h2d_kv)
  results/lmcache/lmcache_retrieve_dram.csv  (LMCache REAL DRAM KV-load latency, from logs)
Writes results/lmcache/fig_lmcache_compare.png
"""
from __future__ import annotations
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO = "5396384503"
FRAMES = [16, 32, 64, 128]
NVIS = {16: 3136, 32: 6272, 64: 12544, 128: 25088}


def med_lmcache(path, variant, metric="ttft"):
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["variant"] == variant and r["metric"] == metric:
                d[int(r["frames"])].append(float(r["value_ms"]))
    return {fr: statistics.median(v) for fr, v in d.items()}


def med_reuse_real(path, variant, metric):
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r["model"] == "llava-ov-7b" and int(r.get("batch") or 1) == 1
                    and r["video_id"] == VIDEO and r["variant"] == variant and r["metric"] == metric):
                d[int(r["frames"])].append(float(r["value_ms"]))
    return {fr: statistics.median(v) for fr, v in d.items()}


def main():
    lmc = "results/lmcache/reuse_lmcache.csv"
    rr = "results/nextqa/reuse_real.csv"
    cold = med_lmcache(lmc, "cold")
    ours = med_lmcache(lmc, "kv_ours")
    lmcache = med_lmcache(lmc, "kv_lmcache")
    our_h2d = med_reuse_real(rr, "kv_reuse", "h2d_kv")

    # LMCache real DRAM retrieve (from logs); map by ascending n_kv ~ ascending frames
    retr = {}
    with open("results/lmcache/lmcache_retrieve_dram.csv") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["n_kv_tokens"]))
    for fr, row in zip(FRAMES, rows):
        retr[fr] = float(row["retrieve_ms_median"])

    x = [NVIS[fr] for fr in FRAMES]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # -- LEFT: warm TTFT (front cost), cold for reference --
    axL.plot(x, [cold[fr] for fr in FRAMES], "o-", color="#7f7f7f", label="cold (recompute)")
    axL.plot(x, [ours[fr] for fr in FRAMES], "s-", color="#2ca02c",
             label="our kv_reuse (GPU-resident warm)")
    axL.plot(x, [lmcache[fr] for fr in FRAMES], "^-", color="#d62728",
             label="LMCache kv_reuse (DRAM, real load)")
    axL.set_title("(a) Front cost: TTFT vs n_vis  (LLaVA-OV-7B, b1, H100)")
    axL.set_xlabel("n_vis (vision tokens)"); axL.set_ylabel("TTFT (ms)")
    axL.set_yscale("log"); axL.legend(fontsize=9); axL.grid(alpha=0.3)

    # -- RIGHT: retrieval validation --
    axR.plot(x, [our_h2d[fr] for fr in FRAMES], "s-", color="#1f77b4",
             label="our model: measured h2d_kv (vLLM 0.22)")
    axR.plot(x, [retr[fr] for fr in FRAMES], "^--", color="#ff7f0e",
             label="LMCache: REAL DRAM KV load (vLLM 0.18)")
    axR.set_title("(b) Retrieval cost — our computed term vs LMCache real load")
    axR.set_xlabel("n_vis (vision tokens)"); axR.set_ylabel("KV DRAM→GPU retrieval (ms)")
    axR.legend(fontsize=9); axR.grid(alpha=0.3)
    for fr in FRAMES:
        axR.annotate(f"{NVIS[fr]*56//1024/1024:.2f}GB" if False else "", (NVIS[fr], retr[fr]))

    fig.tight_layout()
    out = "results/lmcache/fig_lmcache_compare.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    # print the comparison table to stdout for the report
    print(f"\n{'fr':>4}{'n_vis':>7}{'cold':>9}{'ours_warm':>11}{'lmc_warm':>10}{'our_h2d':>9}{'lmc_retr':>10}")
    for fr in FRAMES:
        print(f"{fr:>4}{NVIS[fr]:>7}{cold[fr]:>9.1f}{ours[fr]:>11.1f}{lmcache[fr]:>10.1f}"
              f"{our_h2d[fr]:>9.1f}{retr[fr]:>10.1f}")


if __name__ == "__main__":
    main()
