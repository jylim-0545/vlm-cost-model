"""Synthetic query arrival + popularity (CLAUDE.md Section 6).

NO QA dataset has timestamps, so the workload (which video, how many times N, over
what retention window) is SYNTHESIZED here and kept strictly separate from the
cost-primitive measurement. Pure CPU, no GPU, no dataset.

Model:
  - Popularity: Zipf over `num_videos` (rank-1 hottest). p(rank k) ∝ 1/k^s.
  - Arrivals:   homogeneous Poisson, mean `rate_per_hour` over `duration_hours`.
                Each arrival draws a video by popularity.
Output per video: N accesses + first/last access time → feeds the price model's
(N, retention_time) axes. Reproducible: fixed seed, numpy default_rng.

Usage:
  python -m workload.arrival
  python -m workload.arrival --num-videos 200 --zipf-s 1.2 --rate-per-hour 80 --hours 24 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class ArrivalConfig:
    num_videos: int = 100
    zipf_s: float = 1.1          # popularity skew (>1 = heavy head)
    rate_per_hour: float = 50.0  # mean arrivals/hour (Poisson lambda)
    duration_hours: float = 24.0
    seed: int = 0


@dataclass
class VideoAccess:
    video_id: str
    rank: int                 # 1 = most popular
    popularity: float         # normalized prob
    n_accesses: int           # THE N in the price model
    first_access_h: float
    last_access_h: float
    span_h: float             # last - first; lower bound on useful retention


def zipf_probs(num_videos: int, s: float) -> np.ndarray:
    """Finite Zipf over ranks 1..num_videos (numpy's zipf is unbounded)."""
    ranks = np.arange(1, num_videos + 1, dtype=float)
    w = 1.0 / np.power(ranks, s)
    return w / w.sum()


def synthesize(cfg: ArrivalConfig) -> tuple[list[VideoAccess], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    probs = zipf_probs(cfg.num_videos, cfg.zipf_s)

    # total arrivals over the window ~ Poisson(rate * duration); times ~ Uniform (sorted)
    n_arrivals = int(rng.poisson(cfg.rate_per_hour * cfg.duration_hours))
    times = np.sort(rng.uniform(0.0, cfg.duration_hours, size=n_arrivals))
    picks = rng.choice(cfg.num_videos, size=n_arrivals, p=probs)  # 0-based rank index

    out: list[VideoAccess] = []
    for idx in range(cfg.num_videos):
        mask = picks == idx
        n = int(mask.sum())
        if n == 0:
            first = last = span = 0.0
        else:
            t = times[mask]
            first, last = float(t[0]), float(t[-1])
            span = last - first
        out.append(VideoAccess(
            video_id=f"vid_{idx:05d}",
            rank=idx + 1,
            popularity=float(probs[idx]),
            n_accesses=n,
            first_access_h=round(first, 4),
            last_access_h=round(last, 4),
            span_h=round(span, 4),
        ))
    return out, times


def _out_dir() -> Path:
    d = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_csv(rows: list[VideoAccess], cfg: ArrivalConfig) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _out_dir() / f"arrival_{ts}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def print_summary(rows: list[VideoAccess], cfg: ArrivalConfig) -> None:
    accessed = [r for r in rows if r.n_accesses > 0]
    total = sum(r.n_accesses for r in rows)
    print(f"\nArrival synthesis (seed={cfg.seed}, zipf_s={cfg.zipf_s}, "
          f"rate={cfg.rate_per_hour}/h x {cfg.duration_hours}h):")
    print(f"  total arrivals: {total}  |  videos touched: {len(accessed)}/{cfg.num_videos}")
    print(f"  {'rank':>4} {'video':>10} {'N':>6} {'pop':>8} {'span_h':>8}")
    for r in rows[:8]:
        print(f"  {r.rank:>4} {r.video_id:>10} {r.n_accesses:>6} "
              f"{r.popularity:>8.4f} {r.span_h:>8.2f}")
    if len(rows) > 8:
        print(f"  ... ({cfg.num_videos - 8} more)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-videos", type=int, default=100)
    ap.add_argument("--zipf-s", type=float, default=1.1)
    ap.add_argument("--rate-per-hour", type=float, default=50.0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = ArrivalConfig(num_videos=a.num_videos, zipf_s=a.zipf_s,
                        rate_per_hour=a.rate_per_hour, duration_hours=a.hours, seed=a.seed)
    rows, _ = synthesize(cfg)
    print_summary(rows, cfg)
    path = write_csv(rows, cfg)
    print(f"\nwrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
