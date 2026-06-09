"""Pre-projector vs post-projector vision-token reuse — 3-way TTFT, InternVL (transformers).

Compares, per frame count (= #448px tiles, 256 vision tok/tile), the TTFT of:
  cold         = full recompute            = ViT(encoder) + projector(pixel_shuffle+mlp1) + prefill
  post_reuse   = reuse projector OUTPUT     = prefill only         (skip ViT + projector)  [= our vt_reuse / LMCache EC]
  pre_reuse    = reuse ENCODER (ViT) OUTPUT = projector + prefill  (skip ViT, RE-RUN projector)

vLLM cannot express pre_reuse (no "skip ViT, run projector" input path; embeds-inject skips the WHOLE
tower), so all three are measured in ONE engine — transformers — via stage subtraction (same-engine,
so valid; do NOT compare these absolute ms to vLLM numbers, per CLAUDE.md dual-engine rule):
  t_full  = full_forward (ViT+proj+prefill) ;  t_tower = extract_feature (ViT+proj) ;  t_vit = ViT alone
  cold = t_full ;  post = t_full - t_tower ;  pre = t_full - t_vit
The pre-vs-post gap = the projector (pixel_shuffle+mlp1) compute — the whole point of the question.

InternVL only (its InternViT is SHARED across 4/8/14B, so pre-projector features are cross-model
reusable — the real motivation for pre-projector reuse). batch=1 (adapter limit). H100.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models                       # noqa: E402
from measure.stage_timing import InternVLAdapter, _cuda_time, DEV   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="internvl3.5-8b")
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--csv", default="results/lmcache/preproj_internvl.csv")
    a = ap.parse_args()

    import torch
    name = torch.cuda.get_device_name(0)
    assert "H100" in name, f"H100 only; visible device is {name!r} (set CUDA_VISIBLE_DEVICES=1)"
    print(f"[gpu] {name}")

    spec = load_models().models[a.model]
    ad = InternVLAdapter(spec)
    ad.load()
    print(f"[model] {a.model} loaded (transformers); InternViT hidden="
          f"{ad.model.config.vision_config.hidden_size}, LLM hidden={ad.model.config.llm_config.hidden_size}")

    def med(fn) -> float:
        for _ in range(a.warmup):
            _cuda_time(fn)
        xs = [_cuda_time(fn)[0] for _ in range(a.runs)]
        return statistics.median(xs) * 1e3      # -> ms

    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(a.csv).exists()
    f = open(a.csv, "a", newline="")
    W = csv.DictWriter(f, fieldnames=["model", "frames", "n_vis", "t_vit_ms", "t_proj_ms",
                                      "t_prefill_ms", "cold_ttft_ms", "post_reuse_ttft_ms",
                                      "pre_reuse_ttft_ms", "engine"])
    if new:
        W.writeheader()

    print(f"\n{'fr':>4}{'n_vis':>7} | {'ViT':>8}{'proj':>8}{'prefill':>9} | "
          f"{'COLD':>9}{'post':>9}{'pre':>9} | pre−post")
    with torch.inference_mode():
        for nf in a.frames:
            n_vis, _ = ad.build_inputs(nf, ad.image_size, 1)   # P = nf tiles, 256 tok/tile
            pv = ad._pixel_values
            t_full = med(lambda: ad.full_forward())                       # ViT+proj+prefill
            t_tower = med(lambda: ad.encode())                            # extract_feature = ViT+proj
            t_vit = med(lambda: ad.model.vision_model(pixel_values=pv))   # ViT alone
            cold = t_full
            post = t_full - t_tower                                       # skip ViT+proj
            pre = t_full - t_vit                                          # skip ViT only
            t_proj = t_tower - t_vit
            t_prefill = t_full - t_tower
            W.writerow({"model": a.model, "frames": nf, "n_vis": n_vis,
                        "t_vit_ms": round(t_vit, 2), "t_proj_ms": round(t_proj, 2),
                        "t_prefill_ms": round(t_prefill, 2), "cold_ttft_ms": round(cold, 2),
                        "post_reuse_ttft_ms": round(post, 2), "pre_reuse_ttft_ms": round(pre, 2),
                        "engine": "transformers"})
            f.flush()
            print(f"{nf:>4}{n_vis:>7} | {t_vit:>8.1f}{t_proj:>8.1f}{t_prefill:>9.1f} | "
                  f"{cold:>9.1f}{post:>9.1f}{pre:>9.1f} | {pre - post:>+7.1f}")
    f.close()
    print(f"\n[done] {a.csv}  (transformers engine; relative 3-way only, do not mix with vLLM ms)")


if __name__ == "__main__":
    main()
