"""Stage-B analysis entry point: turn measured primitives + cost config into the
paper's figures (CLAUDE.md Section 7). NO GPU, NO LLM — re-runnable instantly.

Pipeline (see CLAUDE.md Sections 5/7):
  Stage A (GPU, once): measure -> results/stage_timing.csv  (T_encode/prefill/decode, n_vision_tokens)
  Stage B (this file):  primitives CSV + config (models.yaml -> bytes; prices.yaml +
                        storage_tiers.yaml -> costs) -> break-even tables + figures.
Edit prices.yaml / storage_tiers.yaml and re-run THIS — no re-measurement needed.

Always writes tidy figure-data CSVs (the numbers behind every figure) to
results/figures/. If matplotlib is installed, also renders PNGs; if not, the CSVs
are ready for any plotting tool.

Figures produced:
  1. break_even   — break-even N vs n_vision_tokens, one line per storage tier, per
                    model & reuse type. THE core figure (a FAMILY of curves).
  2. cost_vs_N    — total $/query-stream vs N for baseline / kv_reuse / vt_reuse,
                    crossover = break-even (one panel per model,tier).
  3. cost_share   — at a fixed N, gpu/storage/network $ breakdown per variant.

Usage:
  python -m analyze.plots --primitives results/stage_timing.csv
  python -m analyze.plots --primitives results/stage_timing.csv --retention-days 7
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyze.price_model import (break_even_qpm, cost, merge_preprocess,  # noqa: E402
                                 load_primitives_any, synthetic_primitives)
from config import load_prices, load_storage_tiers  # noqa: E402

# query RATE grid (queries per MONTH) — N is now a per-month rate, matching the
# per-month storage cost (see price_model). retention is the other axis.
QPM_GRID = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 1000]
REUSE = ("kv_reuse", "vt_reuse")


def _fig_dir() -> Path:
    d = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results"))) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_tables(prims, tiers, prices, retention, fixed_N):
    """Return (break_even_rows, cost_vs_N_rows, cost_share_rows)."""
    be, cvn, cs = [], [], []
    for tier in tiers.values():
        for p in prims:
            row = {"model": p.model, "video_id": p.video_id, "batch": p.batch,
                   "n_vision_tokens": p.n_vision_tokens, "duration_s": p.duration_s,
                   "tier": tier.name}
            for v in REUSE:
                q = break_even_qpm(v, p, retention, tier=tier, prices=prices)
                qn = break_even_qpm(v, p, retention, tier=tier, prices=prices, include_egress=False)
                row[f"{v}_breakeven_qpm"] = "inf" if math.isinf(q) else round(q, 3)
                row[f"{v}_breakeven_qpm_noegress"] = "inf" if math.isinf(qn) else round(qn, 3)
            be.append(row)

            for qpm in QPM_GRID:
                for v in ("baseline",) + REUSE:
                    c = cost(v, p, qpm, retention, tier=tier, prices=prices)
                    cvn.append({"model": p.model, "video_id": p.video_id, "batch": p.batch,
                                "n_vision_tokens": p.n_vision_tokens, "tier": tier.name,
                                "variant": v, "queries_per_month": qpm,
                                "total_usd": round(c.total_usd, 8), "gpu_usd": round(c.gpu_usd, 8),
                                "storage_usd": round(c.storage_usd, 8),
                                "network_usd": round(c.network_usd, 8)})

            for v in ("baseline",) + REUSE:
                c = cost(v, p, fixed_N, retention, tier=tier, prices=prices)
                sh = c.shares()
                cs.append({"model": p.model, "video_id": p.video_id, "batch": p.batch,
                           "n_vision_tokens": p.n_vision_tokens, "tier": tier.name,
                           "variant": v, "queries_per_month": fixed_N,
                           "preprocess_usd": round(c.preprocess_usd, 8),
                           "encode_usd": round(c.encode_usd, 8), "prefill_usd": round(c.prefill_usd, 8),
                           "decode_usd": round(c.decode_usd, 8), "storage_usd": round(c.storage_usd, 8),
                           "network_usd": round(c.network_usd, 8), "total_usd": round(c.total_usd, 8),
                           "decode_pct": round(sh["decode"] * 100, 2),
                           "encode_pct": round(sh["encode"] * 100, 2),
                           "prefill_pct": round(sh["prefill"] * 100, 2)})
    return be, cvn, cs


def render_pngs(be, cvn, tiers, fig_dir, retention):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plots] matplotlib not installed — wrote figure-data CSVs only "
              "(install matplotlib to render PNGs).")
        return

    models = sorted({r["model"] for r in be})
    tier_names = list(tiers.keys())
    batches = sorted({r["batch"] for r in be})   # each batch regime = its own figure set

    n = 0
    for bsz in batches:
        beb = [r for r in be if r["batch"] == bsz]
        cvnb = [r for r in cvn if r["batch"] == bsz]

        # Fig 1: break-even rate vs n_vision_tokens, line per tier, facet per model.
        # Rendered both WITH egress and WITHOUT (egress is a volatile price knob).
        for v in REUSE:
            for tag, col in (("", f"{v}_breakeven_qpm"), ("_noegress", f"{v}_breakeven_qpm_noegress")):
                fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
                for ax, m in zip(axes[0], models):
                    for t in tier_names:
                        pts = sorted([(r["n_vision_tokens"], r[col])
                                      for r in beb if r["model"] == m and r["tier"] == t],
                                     key=lambda z: z[0])
                        xs = [x for x, y in pts if y != "inf"]
                        ys = [y for x, y in pts if y != "inf"]
                        ax.plot(xs, ys, marker="o", label=t)
                    egl = "egress excl." if tag else "egress incl."
                    ax.set_title(f"{m}\n{v} break-even ({egl}, batch={bsz})")
                    ax.set_xlabel("n_vision_tokens")
                    ax.set_ylabel("break-even rate (queries/month)")
                    ax.set_xscale("log"); ax.set_yscale("log")
                    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
                fig.tight_layout(); fig.savefig(fig_dir / f"fig_breakeven_{v}{tag}_b{bsz}.png", dpi=130)
                plt.close(fig); n += 1

        # Fig 2: cost vs N (largest video per model, one panel per tier)
        for m in models:
            nvs = [r["n_vision_tokens"] for r in cvnb if r["model"] == m]
            if not nvs:
                continue
            nv = max(nvs)
            fig, axes = plt.subplots(1, len(tier_names), figsize=(5 * len(tier_names), 4), squeeze=False)
            for ax, t in zip(axes[0], tier_names):
                for v in ("baseline",) + REUSE:
                    pts = sorted([(r["queries_per_month"], r["total_usd"]) for r in cvnb
                                  if r["model"] == m and r["tier"] == t and r["variant"] == v
                                  and r["n_vision_tokens"] == nv], key=lambda z: z[0])
                    ax.plot([x for x, _ in pts], [y for _, y in pts], marker=".", label=v)
                ax.set_title(f"{m} @ {nv} tok (batch={bsz})\ntier={t}"); ax.set_xlabel("queries / month")
                ax.set_ylabel("$ over retention"); ax.set_xscale("log"); ax.set_yscale("log")
                ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(fig_dir / f"fig_cost_vs_N_{m}_b{bsz}.png", dpi=130)
            plt.close(fig); n += 1
    print(f"[plots] rendered {n} PNGs -> {fig_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primitives", help="measured stage_timing.csv; omit for synthetic demo")
    ap.add_argument("--preprocess-csv", help="preprocess_timing.csv to attach C_preprocess (CPU)")
    ap.add_argument("--retention-days", type=float, default=None)
    ap.add_argument("--fixed-N", type=float, default=10.0,
                    help="query rate (queries/month) for the cost-share breakdown")
    a = ap.parse_args()

    prices = load_prices()
    tiers = load_storage_tiers()
    retention = a.retention_days or prices["defaults"]["retention_time_days"]
    prims = load_primitives_any(a.primitives) if a.primitives else synthetic_primitives()
    if a.preprocess_csv:
        prims = merge_preprocess(prims, a.preprocess_csv)
    if not a.primitives:
        print("*** SYNTHETIC primitives — pass --primitives results/stage_timing.csv for real ***")

    be, cvn, cs = build_tables(prims, tiers, prices, retention, a.fixed_N)
    fig_dir = _fig_dir()
    _write_csv(fig_dir / "break_even.csv", be)
    _write_csv(fig_dir / "cost_vs_N.csv", cvn)
    _write_csv(fig_dir / "cost_share.csv", cs)
    print(f"[plots] wrote figure-data CSVs ({len(be)} break-even rows over "
          f"{len(tiers)} tiers x {len(prims)} primitives) -> {fig_dir}")
    render_pngs(be, cvn, tiers, fig_dir, retention)


if __name__ == "__main__":
    main()
