"""C_preprocess — CPU cost of turning a video into vision-encoder input
(CLAUDE.md Sections 5/7). This is the decode/resize done BEFORE the vision tower:
decord video decode + frame sampling + the model's resize/normalize to pixel_values.

It is paid EVERY query by baseline (re-decode each time) but ONCE by reuse (the
stored tokens/KV skip it), so it adds to the per-query reuse SAVING and lowers
break-even — and it grows for long videos (decode is seconds for MLVU). Priced at
cpu_usd_per_vcpu_hour.

CPU-ONLY: no GPU, no model weights (just the processor / a resize). Run it AFTER
GPU jobs finish so CPU contention doesn't pollute the timing. Video must be on
LOCAL disk (LOCAL_SCRATCH) so decode time isn't NFS-bound (Section 2).

Output: results/preprocess_timing.csv (model, video_id, n_frames, t_preprocess_s,
duration_s). Join into the price model with `analyze/plots.py --preprocess-csv`.

Usage:
  python -m measure.preprocess_timing --model qwen2.5-vl-7b --videos-csv results/nextqa_sample.csv --frames 16
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos-csv", required=True)
    ap.add_argument("--fps", type=float, default=2.0, help="Qwen-native fps frame sampling")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()

    spec = load_models().models[a.model]
    is_internvl = a.model.startswith("internvl")

    import decord
    import numpy as np
    decord.bridge.set_bridge("native")

    # model-specific resize/normalize to pixel_values (CPU). Qwen has a processor;
    # InternVL has none, so emulate its tile prep (resize each frame to 448).
    if is_internvl:
        import cv2
        size = 448

        def to_pixels(frames):
            return np.stack([cv2.resize(f, (size, size)) for f in frames]).astype("float32") / 255.0

        text = None
    else:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        text = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Describe."}]}],
            add_generation_prompt=True, tokenize=False)

        def to_pixels(frames):
            return proc(text=[text], videos=[frames], return_tensors="pt")  # CPU tensors

    def preprocess_once(path):
        t = time.perf_counter()
        vr = decord.VideoReader(path)
        from measure.frames import qwen_frame_indices
        idx = qwen_frame_indices(len(vr), vr.get_avg_fps(), fps=a.fps)
        frames = vr.get_batch(idx).asnumpy()
        _ = to_pixels(frames)
        return time.perf_counter() - t, len(idx)

    out_path = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results"))) / "preprocess_timing.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    f = open(out_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["model", "video_id", "n_frames", "t_preprocess_s",
                                      "t_preprocess_iqr_s", "duration_s", "timestamp"])
    if new:
        w.writeheader()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    with open(os.path.expanduser(a.videos_csv)) as vf:
        for d in csv.DictReader(vf):
            samples = []
            nfr = 0
            for i in range(a.warmup + a.runs):
                dt, nfr = preprocess_once(d["path"])
                if i >= a.warmup:
                    samples.append(dt)
            med = statistics.median(samples)
            iqr = (statistics.quantiles(samples, n=4)[2] - statistics.quantiles(samples, n=4)[0]
                   if len(samples) >= 2 else 0.0)
            vid = f"{Path(d['path']).stem}_{nfr}f"   # match stage_timing_vllm video_id
            w.writerow({"model": spec.key, "video_id": vid, "n_frames": nfr,
                        "t_preprocess_s": round(med, 6), "t_preprocess_iqr_s": round(iqr, 6),
                        "duration_s": d.get("duration_s") or "", "timestamp": ts})
            print(f"[prep] {vid}: t_preprocess={med*1e3:.1f}ms (dur={d.get('duration_s')}s)")
    f.close()
    print(f"[prep] wrote -> {out_path}")


if __name__ == "__main__":
    main()
