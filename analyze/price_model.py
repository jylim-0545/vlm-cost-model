"""The break-even price model (CLAUDE.md Section 7).

Per query, per variant (cost in USD):
  baseline:    N * (T_encode + T_prefill + T_decode)
  KV reuse:    once(T_encode + T_prefill) + N * (T_decode + network_cost(bytes_kv))
               + C_store(bytes_kv) * retention
  token reuse: once(T_encode) + N * (T_prefill + T_decode + network_cost(bytes_vision))
               + C_store(bytes_vision) * retention

Network/retrieval cost is COMPUTED from I/O volume, never measured:
  retrieval_time = latency_fixed + read_bytes / bandwidth
  network_cost   = read_bytes * egress_price + retrieval_time * resource_price
where read_bytes (bytes_kv vs bytes_vision) is computed from config, and
bandwidth/egress_price/latency_fixed/resource_price are per-tier CONFIG params
(config/storage_tiers.yaml). We SWEEP tiers -> a FAMILY of break-even curves, not
one number. read_bytes is the lever: KV is 8-18x the vision-token bytes.

T_encode/prefill/decode are MEASURED (Layer 1); a labelled SYNTHETIC generator
lets the surface logic run end-to-end until real CSVs are joined.

Usage:
  python -m analyze.price_model                       # synthetic demo, all tiers
  python -m analyze.price_model --primitives <csv> --tier object_storage
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import StorageTier, load_models, load_prices, load_storage_tiers  # noqa: E402

VARIANTS = ("baseline", "kv_reuse", "vt_reuse")


@dataclass(frozen=True)
class Primitives:
    """Per (model, video, BATCH regime) Layer-1 measurement. Seconds are PER-REQUEST
    GPU-time medians at this batch. batch is a first-class axis: batching is NOT
    linear (decode/token drops sharply with batch), so each batch is a SEPARATE
    measured run — never extrapolate one batch's stages from another's."""
    model: str
    video_id: str
    n_vision_tokens: int
    t_encode_s: float
    t_prefill_s: float
    t_decode_s: float
    duration_s: float | None = None
    batch: int = 1
    t_preprocess_s: float = 0.0   # CPU: video decode + frame sample/resize (before encoder)


@dataclass
class CostBreakdown:
    """Per-variant cost split for cost-share analysis. preprocess is CPU; encode/
    prefill/decode are GPU; storage/network are the stored-state tier costs."""
    variant: str
    preprocess_usd: float
    encode_usd: float
    prefill_usd: float
    decode_usd: float
    storage_usd: float
    network_usd: float

    @property
    def gpu_usd(self) -> float:
        return self.encode_usd + self.prefill_usd + self.decode_usd

    @property
    def total_usd(self) -> float:
        return (self.preprocess_usd + self.encode_usd + self.prefill_usd
                + self.decode_usd + self.storage_usd + self.network_usd)

    def shares(self) -> dict[str, float]:
        t = self.total_usd or 1.0
        return {k: getattr(self, f"{k}_usd") / t for k in
                ("preprocess", "encode", "prefill", "decode", "storage", "network")}


def gpu_rate_per_s(prices: dict) -> float:
    return prices["compute"]["gpu_h100_usd_per_hour"] / 3600.0


def cpu_rate_per_s(prices: dict) -> float:
    return prices["compute"]["cpu_usd_per_vcpu_hour"] / 3600.0


def _read_bytes(variant: str, p: Primitives) -> int:
    cfg = load_models()
    if variant == "kv_reuse":
        return cfg.kv_bytes(p.model, p.n_vision_tokens)
    if variant == "vt_reuse":
        return cfg.vision_bytes(p.model, p.n_vision_tokens)
    return 0


def cost(variant: str, p: Primitives, n_per_month: float, retention_days: float, *,
         tier: StorageTier, prices: dict, include_egress: bool = True) -> CostBreakdown:
    """TOTAL $ over the retention window for a video queried `n_per_month` times/month
    and kept `retention_days` (R = retention_days/30 months). Storage rent is per-month
    (× R); the one-time store cost (encode[+prefill]) is paid ONCE, not per month."""
    g = gpu_rate_per_s(prices)
    cg = cpu_rate_per_s(prices)
    R = retention_days / 30.0
    total_q = n_per_month * R          # total accesses over the retention window
    pp, enc, pre, dec = p.t_preprocess_s, p.t_encode_s, p.t_prefill_s, p.t_decode_s

    if variant == "baseline":           # everything recomputed every query
        return CostBreakdown("baseline", total_q * pp * cg, total_q * enc * g,
                             total_q * pre * g, total_q * dec * g, 0.0, 0.0)

    rb = _read_bytes(variant, p)
    network = total_q * tier.network_cost_usd(rb, g, include_egress)
    storage = tier.storage_cost_usd(rb, retention_days)   # = $/GB-month x R
    if variant == "kv_reuse":           # preprocess+encode+prefill ONCE; decode every query
        return CostBreakdown("kv_reuse", pp * cg, enc * g, pre * g,
                             total_q * dec * g, storage, network)
    if variant == "vt_reuse":        # preprocess+encode ONCE; prefill+decode every query
        return CostBreakdown("vt_reuse", pp * cg, enc * g, total_q * pre * g,
                             total_q * dec * g, storage, network)
    raise ValueError(f"unknown variant {variant!r}")


def break_even_qpm(variant: str, p: Primitives, retention_days: float, *,
                   tier: StorageTier, prices: dict, include_egress: bool = True) -> float:
    """Smallest query RATE (queries per MONTH) at which `variant` beats baseline,
    given the state is retained `retention_days` (R months). inf if it never does.

    Over R months with N queries/month: baseline = N*R*b ; reuse = F + N*R*r + S_m*R.
    Break-even: N* (per month) = (F + S_m*R) / (R*(b-r)) = (F + storage_total)/(R*(b-r)).
    R->inf  => N* -> S_m/(b-r) (steady-state: monthly storage rent vs per-query saving).
    """
    if variant == "baseline":
        return 0.0
    g = gpu_rate_per_s(prices)
    pp_cost = p.t_preprocess_s * cpu_rate_per_s(prices)   # CPU preprocess $/query
    R = retention_days / 30.0
    b = pp_cost + (p.t_encode_s + p.t_prefill_s + p.t_decode_s) * g
    rb = _read_bytes(variant, p)
    if variant == "kv_reuse":
        fixed = pp_cost + (p.t_encode_s + p.t_prefill_s) * g   # one-time store cost ($)
        r = p.t_decode_s * g + tier.network_cost_usd(rb, g, include_egress)
    else:  # vt_reuse
        fixed = pp_cost + p.t_encode_s * g
        r = (p.t_prefill_s + p.t_decode_s) * g + tier.network_cost_usd(rb, g, include_egress)
    store = tier.storage_cost_usd(rb, retention_days)     # = S_m * R (total over retention)
    denom = b - r
    if denom <= 0:
        return math.inf      # reuse per-access >= baseline -> storage/recompute never pays off
    return (fixed + store) / (denom * R)


# ---- primitives I/O --------------------------------------------------------

def load_primitives_csv(path: str) -> list[Primitives]:
    """Wide schema from measure/stage_timing.py (transformers): one row per primitive
    with t_encode_s/t_prefill_s/t_decode_s columns."""
    rows: list[Primitives] = []
    with open(path) as f:
        for d in csv.DictReader(f):
            rows.append(Primitives(
                model=d["model"], video_id=d["video_id"],
                n_vision_tokens=int(d["n_vision_tokens"]),
                t_encode_s=float(d["t_encode_s"]), t_prefill_s=float(d["t_prefill_s"]),
                t_decode_s=float(d["t_decode_s"]),
                duration_s=float(d["duration_s"]) if d.get("duration_s") else None,
                batch=int(d.get("batch") or 1),
                t_preprocess_s=float(d.get("t_preprocess_s") or 0.0)))
    return rows


def load_primitives_from_vllm_csv(path: str) -> list[Primitives]:
    """Long schema from measure/stage_timing_vllm.py: rows tagged by `stage`
    (run --text-baseline so encode/prefill/decode are all present). Pivot to
    Primitives, taking the MEDIAN over run_idx per (model, video_id)."""
    import statistics
    from collections import defaultdict
    vals: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    meta: dict[tuple, dict] = {}
    with open(path) as f:
        for d in csv.DictReader(f):
            key = (d["model"], d["video_id"])
            vals[key][d["stage"]].append(float(d["value_s"]))
            meta[key] = d
    out: list[Primitives] = []
    for (model, vid), stages in vals.items():
        if not {"encode", "prefill", "decode"} <= stages.keys():
            continue   # need --text-baseline split; skip ttft-only rows
        m = meta[(model, vid)]
        out.append(Primitives(
            model=model, video_id=vid, n_vision_tokens=int(m["n_vision_tokens"]),
            t_encode_s=statistics.median(stages["encode"]),
            t_prefill_s=statistics.median(stages["prefill"]),
            t_decode_s=statistics.median(stages["decode"]),
            duration_s=float(m["duration_s"]) if m.get("duration_s") else None,
            batch=int(m.get("batch") or 1),
            t_preprocess_s=statistics.median(stages["preprocess"]) if "preprocess" in stages else 0.0))
    return out


def merge_preprocess(prims: list[Primitives], path: str) -> list[Primitives]:
    """Attach C_preprocess (CPU) from measure/preprocess_timing.py, keyed by
    (model, video_id). Primitives without a match keep t_preprocess_s=0."""
    m: dict[tuple, float] = {}
    with open(path) as f:
        for d in csv.DictReader(f):
            m[(d["model"], d["video_id"])] = float(d["t_preprocess_s"])
    return [replace(p, t_preprocess_s=m[(p.model, p.video_id)])
            if (p.model, p.video_id) in m else p for p in prims]


def load_primitives_any(path: str) -> list[Primitives]:
    """Auto-detect engine CSV format: long (vLLM, has `stage`) vs wide (transformers)."""
    with open(path) as f:
        header = f.readline()
    return (load_primitives_from_vllm_csv(path) if "stage" in header.split(",")
            else load_primitives_csv(path))


def synthetic_primitives() -> list[Primitives]:
    """PLACEHOLDER until Layer-1 CSVs are joined. Rough H100 proportionalities."""
    cfg = load_models()
    out: list[Primitives] = []
    for key in cfg.models:
        for n in (1024, 8192, 32768):
            out.append(Primitives(model=key, video_id=f"synthetic_n{n}", n_vision_tokens=n,
                                  t_encode_s=n / 20000.0, t_prefill_s=n / 8000.0,
                                  t_decode_s=256 / 120.0))
    return out


# ---- demo / CLI ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primitives", help="CSV from stage_timing.py; omit for synthetic demo")
    ap.add_argument("--retention-days", type=float, default=None)
    ap.add_argument("--tier", help="single tier; omit to sweep ALL tiers (the family)")
    a = ap.parse_args()

    prices = load_prices()
    tiers = load_storage_tiers()
    retention = a.retention_days or prices["defaults"]["retention_time_days"]
    sweep = [tiers[a.tier]] if a.tier else list(tiers.values())
    prims = load_primitives_any(a.primitives) if a.primitives else synthetic_primitives()
    if not a.primitives:
        print("*** SYNTHETIC primitives (placeholder timings) — for logic check only ***")

    print(f"\nBreak-even query RATE (queries/month, retention={retention}d) — family over tiers:")
    for tier in sweep:
        print(f"\n  ── tier={tier.name}  (bw={tier.bandwidth_gbps}GB/s, "
              f"egress=${tier.egress_price_usd_per_gb}/GB, store=${tier.usd_per_gb_month}/GB-mo) ──")
        print(f"  {'model':<16} {'n_tok':>7} | {'kv /mo':>8} {'kv(noEg)':>9} | "
              f"{'tok /mo':>8} {'tok(noEg)':>10}   (noEg = egress excluded)")
        for p in prims:
            def be(v, eg):
                x = break_even_qpm(v, p, retention, tier=tier, prices=prices, include_egress=eg)
                return "never" if math.isinf(x) else f"{x:.2f}"
            print(f"  {p.model:<16} {p.n_vision_tokens:>7} | "
                  f"{be('kv_reuse', True):>8} {be('kv_reuse', False):>9} | "
                  f"{be('vt_reuse', True):>8} {be('vt_reuse', False):>10}")


if __name__ == "__main__":
    main()
