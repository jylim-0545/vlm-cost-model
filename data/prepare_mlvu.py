"""Prepare an MLVU (long-video) sample with REAL dataset queries for the n_vis sweep.

MLVU lives as mp4 under datasets/MLVU/MLVU/video/<task>/, with QA in
datasets/MLVU/MLVU/json/<task>.json (each entry: video, duration, question,
candidates, answer, question_type). Unlike NExT-QA the QA is readable plain JSON
(no pyarrow), so we attach the REAL question (+ MC options) per video.

Goal (path A): sweep n_vision_tokens up to the model context limit. MLVU is HD
(720p–1080p) and long, so few frames already give large n_vis. We pick a few HD
videos; the frame sweep in stage_timing_vllm then spans n_vis to the ceiling.

Copies sampled videos to LOCAL_SCRATCH (NVMe) and writes results/mlvu_sample.csv
(video_id, path, duration_s, width, height, fps, nframes, question, question_type).

Usage:
  python -m data.prepare_mlvu --n 3 --min-width 1280
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
from pathlib import Path

MLVU_ROOT = "/mnt/nas/VLM/datasets/MLVU/MLVU"


def _local_dir() -> Path:
    d = Path(os.path.expanduser(os.environ.get("LOCAL_SCRATCH", "~/VLM/scratch"))) / "mlvu_videos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_qa() -> dict[str, dict]:
    """video filename -> QA entry (first question seen for that video)."""
    qa: dict[str, dict] = {}
    for jf in glob.glob(f"{MLVU_ROOT}/json/*.json"):
        for e in json.load(open(jf)):
            qa.setdefault(e["video"], e)
    return qa


def format_query(entry: dict) -> str:
    q = entry["question"].strip()
    cands = entry.get("candidates")
    if cands:                                   # multiple-choice: present options (real task prompt)
        opts = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(cands))
        return f"{q}\n{opts}\nAnswer with the option's letter."
    return q


def probe(path: str) -> dict:
    import decord
    vr = decord.VideoReader(path)
    n = len(vr); fps = float(vr.get_avg_fps()); h, w, _ = vr[0].shape
    return dict(nframes=n, fps=round(fps, 3), duration_s=round(n / fps, 3) if fps else None,
                width=int(w), height=int(h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="number of videos to sample")
    ap.add_argument("--min-width", type=int, default=1280, help="prefer HD (more tokens/frame)")
    ap.add_argument("--pool", type=int, default=80, help="HD candidates to probe before stratified pick")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    qa = load_qa()
    paths = [f for f in glob.glob(f"{MLVU_ROOT}/video/*/*.mp4") if os.path.basename(f) in qa]
    rng.shuffle(paths)

    # probe a pool of HD candidates, then DURATION-stratified pick (span short->very long)
    pool = []
    for f in paths:
        if len(pool) >= max(a.pool, a.n * 6):
            break
        m = probe(f)
        if m["width"] >= a.min_width:
            pool.append((m["duration_s"], f, m))
    pool.sort(key=lambda x: x[0])
    L = len(pool)
    picks = [pool[min(b * L // a.n + (L // a.n) // 2, L - 1)] for b in range(a.n)] if L > a.n else pool

    local = _local_dir()
    rows = []
    for dur, f, meta in picks:
        name = os.path.basename(f)
        dst = local / name
        if not dst.exists():
            shutil.copyfile(f, dst)
        e = qa[name]
        rows.append({"video_id": Path(name).stem, "path": str(dst), **meta,
                     "question": format_query(e).replace("\n", " | "),
                     "question_type": e.get("question_type", "")})
        print(f"[mlvu] {name}: {meta['duration_s']}s ({meta['duration_s']/60:.1f}min) "
              f"{meta['width']}x{meta['height']} | Q: {e['question'][:60]}")

    out = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results"))) / "mlvu_sample.csv"
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    durs = [r["duration_s"] for r in rows]
    print(f"\n[mlvu] {len(rows)} videos, duration {min(durs):.0f}-{max(durs):.0f}s, "
          f"res {sorted({(r['width'], r['height']) for r in rows})}")
    print(f"[mlvu] wrote -> {out}")


if __name__ == "__main__":
    main()
