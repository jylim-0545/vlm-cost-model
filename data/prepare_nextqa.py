"""Prepare a NExT-QA video sample for cost-primitive measurement.

NExT-QA ships as `videos.zip` (1570 `NExTVideo/<id>.mp4`, short clips ~tens of sec).
For cost primitives we only need a SAMPLE spanning the length/resolution spread
(CLAUDE.md Section 6), and the files MUST sit on LOCAL NVMe (LOCAL_SCRATCH), never
NFS — else decode/preprocess timing is polluted by NFS bandwidth (Section 2).

So this util:
  1. reads the zip's central directory (uncompressed sizes) WITHOUT extracting,
  2. stratified-samples by size — a cheap proxy for length x resolution x bitrate —
     so the sample spans the spread (seeded, reproducible),
  3. extracts ONLY the sampled clips to LOCAL_SCRATCH (flattened to <id>.mp4),
  4. probes real metadata with decord (the decoder the VLM processors use),
  5. writes a tidy CSV that drives real-video stage_timing/throughput and supplies
     each measurement row's duration/resolution/fps metadata.

Needs neither ffmpeg (decord) nor pyarrow (we don't read the QA parquet — content is
irrelevant for cost primitives, Section 6).

Usage:
  python -m data.prepare_nextqa --n 16
  python -m data.prepare_nextqa --n 40 --seed 0 --zip /mnt/nas/VLM/datasets/nextqa/videos.zip
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import zipfile
from pathlib import Path

DEFAULT_ZIP = "/mnt/nas/VLM/datasets/nextqa/videos.zip"


def _local_dir() -> Path:
    d = Path(os.path.expanduser(os.environ.get("LOCAL_SCRATCH", "~/VLM/scratch"))) / "nextqa_videos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out_path() -> Path:
    d = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results")))
    d.mkdir(parents=True, exist_ok=True)
    return d / "nextqa_sample.csv"


def stratified_by_size(entries: list[tuple[str, int]], n: int, seed: int) -> list[tuple[str, int]]:
    """Split size-sorted entries into n equal buckets, pick one at random per bucket.
    Guarantees coverage across the size spread rather than clustering near the mode."""
    rng = random.Random(seed)
    ordered = sorted(entries, key=lambda x: x[1])
    L = len(ordered)
    if n >= L:
        return ordered
    out = []
    for b in range(n):
        lo, hi = b * L // n, (b + 1) * L // n
        out.append(ordered[rng.randrange(lo, max(hi, lo + 1))])
    return out


def load_nextqa_qa() -> dict[str, str]:
    """video_id (str) -> REAL query from NExT-QA parquet. MC (question + 5 options)
    preferred; OE (open-ended question) as fallback. Needs pyarrow."""
    import pandas as pd
    base = "/mnt/nas/VLM/datasets/nextqa"
    qa: dict[str, str] = {}
    # open-ended first (fallback), then overwrite with MC (richer, has options)
    for f in ["OE/validation-00000-of-00001.parquet", "OE/test-00000-of-00001.parquet"]:
        try:
            for _, r in pd.read_parquet(f"{base}/{f}").iterrows():
                qa.setdefault(str(r["video"]), str(r["question"]).strip())
        except Exception:
            pass
    try:
        mc = pd.read_parquet(f"{base}/MC/test-00000-of-00001.parquet")
        for _, r in mc.iterrows():
            opts = " | ".join(f"{chr(65 + i)}. {r[f'a{i}']}" for i in range(5) if f"a{i}" in r)
            qa[str(r["video"])] = f"{str(r['question']).strip()} | {opts} | Answer with the option's letter."
    except Exception:
        pass
    return qa


def probe(path: Path) -> dict:
    import decord
    vr = decord.VideoReader(str(path))
    n = len(vr)
    fps = float(vr.get_avg_fps())
    h, w, _ = vr[0].shape
    return dict(nframes=n, fps=round(fps, 3),
                duration_s=round(n / fps, 3) if fps else None,
                width=int(w), height=int(h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--n", type=int, default=16, help="number of videos to sample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    zf = zipfile.ZipFile(a.zip)
    mp4 = [(i.filename, i.file_size) for i in zf.infolist() if i.filename.endswith(".mp4")]
    print(f"[prep] {len(mp4)} mp4 in zip; sampling {a.n} stratified by size (seed={a.seed})")
    sample = stratified_by_size(mp4, a.n, a.seed)

    local = _local_dir()
    qa = load_nextqa_qa()                          # video_id -> REAL query (pyarrow)
    rows = []
    for name, zbytes in sample:
        vid = Path(name).stem
        dst = local / f"{vid}.mp4"
        if not dst.exists():
            with zf.open(name) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
        meta = probe(dst)
        rows.append({"video_id": vid, "path": str(dst), "zip_bytes": zbytes, **meta,
                     "question": qa.get(vid, "Describe in detail.")})
        print(f"[prep] {vid}: {meta['duration_s']}s {meta['width']}x{meta['height']} "
              f"{meta['fps']}fps {meta['nframes']}f | Q: {qa.get(vid, '(none)')[:55]}")

    rows.sort(key=lambda r: (r["duration_s"] or 0))
    out_csv = _out_path()
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    durs = [r["duration_s"] for r in rows if r["duration_s"]]
    ress = sorted({f"{r['width']}x{r['height']}" for r in rows})
    print(f"[prep] duration span: {min(durs):.1f}-{max(durs):.1f}s | resolutions: {ress}")
    print(f"[prep] extracted {len(rows)} -> {local}")
    print(f"[prep] wrote metadata -> {out_csv}")


if __name__ == "__main__":
    main()
