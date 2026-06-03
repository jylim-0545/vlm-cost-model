"""Layer-"computed" primitive: stored-state byte sizes per model (CLAUDE.md Sec 5/10).

Pure computation — NO GPU, NO model load, NO dataset. This is the first sanity
check of the whole premise: how big is a stored vision-token blob vs a stored KV
cache, and what does it cost to keep on disk? Run this before any GPU work.

Outputs:
  - stdout table (human sanity check)
  - OUTPUT_DIR/byte_sizes_<timestamp>.csv  (append-only, tidy one row per model x n)

Usage:
  python -m measure.byte_sizes
  python -m measure.byte_sizes --n 256 1024 4096 16384 65536 --storage object_standard
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# allow `python measure/byte_sizes.py` as well as `-m measure.byte_sizes`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import KB, load_models, load_storage_tiers  # noqa: E402

# Default vision-token counts to sweep. Spans short clip (~256) to long video
# (tens of thousands of tokens), which is exactly the axis the paper cares about.
DEFAULT_NS = [256, 1024, 4096, 8192, 16384, 32768, 65536]


def _out_dir() -> Path:
    d = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_rows(ns: list[int], storage_tier: str) -> list[dict]:
    cfg = load_models()
    tier = load_storage_tiers()[storage_tier]

    rows: list[dict] = []
    for key, m in cfg.models.items():
        vbpt = m.vision_token_bytes(cfg.dtype_bytes)   # vision bytes / token
        kbpt = m.kv_token_bytes(cfg.dtype_bytes)        # kv bytes / token
        for n in ns:
            vbytes = vbpt * n
            kbytes = kbpt * n
            rows.append({
                "model": key,
                "llm_backbone": m.llm_backbone,
                "dtype": cfg.dtype_name,
                "n_vision_tokens": n,
                "vision_bytes_per_token": vbpt,
                "kv_bytes_per_token": kbpt,
                "kv_to_vision_ratio": round(kbpt / vbpt, 2),
                "vision_bytes_total": vbytes,
                "kv_bytes_total": kbytes,
                "vision_mb": round(vbytes / (KB * KB), 3),
                "kv_mb": round(kbytes / (KB * KB), 3),
                "storage_tier": storage_tier,
                "vision_usd_per_month": round(tier.storage_cost_usd(vbytes, 30), 6),
                "kv_usd_per_month": round(tier.storage_cost_usd(kbytes, 30), 6),
            })
    return rows


def print_table(rows: list[dict]) -> None:
    cfg = load_models()
    # per-token summary first (matches the CLAUDE.md Section 3 table)
    print(f"\nPer-token byte sizes (dtype={cfg.dtype_name}):")
    print(f"  {'model':<16} {'vision/tok':>11} {'kv/tok':>10} {'kv:vision':>10}")
    seen = set()
    for r in rows:
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        print(f"  {r['model']:<16} {r['vision_bytes_per_token']/KB:>8.1f} KB "
              f"{r['kv_bytes_per_token']/KB:>7.1f} KB {r['kv_to_vision_ratio']:>9.1f}x")

    print(f"\nTotal stored size + storage $/month (tier={rows[0]['storage_tier']}):")
    hdr = (f"  {'model':<16} {'n_tok':>7} {'vision':>10} {'kv':>10} "
           f"{'v $/mo':>10} {'kv $/mo':>10}")
    print(hdr)
    for r in rows:
        print(f"  {r['model']:<16} {r['n_vision_tokens']:>7} "
              f"{r['vision_mb']:>8.2f}MB {r['kv_mb']:>8.2f}MB "
              f"{r['vision_usd_per_month']:>10.5f} {r['kv_usd_per_month']:>10.5f}")


def write_csv(rows: list[dict]) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _out_dir() / f"byte_sizes_{ts}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=DEFAULT_NS,
                    help="vision-token counts to sweep")
    ap.add_argument("--storage", default="object_same_region",
                    choices=["local_nvme", "cloud_ssd", "object_same_region", "object_internet"],
                    help="storage tier for the $/month column")
    args = ap.parse_args()

    rows = compute_rows(args.n, args.storage)
    print_table(rows)
    path = write_csv(rows)
    print(f"\nwrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
