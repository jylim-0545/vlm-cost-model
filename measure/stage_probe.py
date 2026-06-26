"""Stage probe — measure the REAL per-stage time inside cold (ViT vs projector vs prefill) to check
whether the preproj 'pre' monkeypatch adds overhead that deflates the inferred encode (cold - vt_pre).

Wraps vision_model.forward + mlp1.forward (InternVL) / vision_tower + projector with CUDA events while
running the ORIGINAL cold path. Gives true ViT_ms and projector_ms (GPU time). Compare:
  measured ViT_ms   vs   (cold_ttft - vt_pre)   [the inferred encode]
If they match -> monkeypatch clean. If ViT_ms >> (cold-vt_pre) -> vt_pre is inflated (overhead).
Same engine as preproj (eager, in-process, mm_cache=0). InternVL only for now. H100.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models                                  # noqa: E402

STAGE = {"vit": [], "proj": []}
GEN = {"t0": None, "pre": []}    # pre = wall from generate() start to extract_feature entry (= mm preprocessing+sched)


def _wrap(mod, key):
    import torch
    if getattr(mod, "_probed", False):
        return
    of = mod.forward

    def w(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = of(*a, **k); e.record(); torch.cuda.synchronize()
        STAGE[key].append(s.elapsed_time(e))
        return r
    mod.forward = w
    mod._probed = True


def patch_probe_internvl():
    import vllm.model_executor.models.internvl as iv
    cls = next(c for c in vars(iv).values() if isinstance(c, type) and "extract_feature" in c.__dict__)
    orig = cls.extract_feature

    def patched(self, pixel_values):
        if GEN["t0"] is not None:                      # first extract_feature this generate
            GEN["pre"].append((time.perf_counter() - GEN["t0"]) * 1e3); GEN["t0"] = None
        _wrap(self.vision_model, "vit")
        _wrap(self.mlp1, "proj")
        return orig(self, pixel_values)
    cls.extract_feature = patched
    print(f"[probe] internvl: timing {cls.__name__}.vision_model + mlp1", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="internvl3.5-8b")
    ap.add_argument("--videos-csv", default="final_videos.csv")
    ap.add_argument("--video-id", default="5396384503")
    ap.add_argument("--frames", type=int, nargs="+", default=[128])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=40960)
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from measure.stage_timing_vllm import build_video_request_internvl
    from transformers import AutoTokenizer

    assert "H100" in torch.cuda.get_device_name(0)
    spec = load_models().models[a.model]
    assert spec.key.startswith("internvl"), "this probe is InternVL-only"
    patch_probe_internvl()
    tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
    llm = LLM(model=spec.repo_id, trust_remote_code=True, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.85, enforce_eager=True, enable_prefix_caching=False,
              mm_processor_cache_gb=0, max_num_seqs=max(a.batches), max_num_batched_tokens=32768,
              limit_mm_per_prompt={"video": 1})
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    row = next(r for r in __import__("csv").DictReader(open(a.videos_csv)) if r["video_id"] == a.video_id)
    print(f"\n[{a.model}] stage probe on {a.video_id}")
    print(f"{'f':>4}{'B':>3}{'ttft/req':>14}{'pre(batch)':>12}{'ViT/req':>10}{'proj/req':>10}{'batch_wall':>14}")
    for nf in a.frames:
        req = build_video_request_internvl(tok, row["path"], nf)[0]
        for B in a.batches:
            reqs = [req] * B
            for _ in range(a.warmup):
                llm.generate(reqs, sp1)
            walls, vit_runs, proj_runs, pre_runs = [], [], [], []
            for _ in range(a.runs):
                STAGE["vit"].clear(); STAGE["proj"].clear(); GEN["pre"].clear()
                torch.cuda.synchronize(); t0 = time.perf_counter(); GEN["t0"] = t0
                llm.generate(reqs, sp1)
                torch.cuda.synchronize(); walls.append((time.perf_counter() - t0) * 1e3 / B)
                vit_runs.append(sum(STAGE["vit"]) / B)
                proj_runs.append(sum(STAGE["proj"]) / B)
                if GEN["pre"]:
                    pre_runs.append(GEN["pre"][0])     # wall to first vision step (not /B: it's the batch's preprocessing)
            wall = statistics.median(walls); vit = statistics.median(vit_runs)
            proj = statistics.median(proj_runs); pre = statistics.median(pre_runs) if pre_runs else float("nan")
            print(f"{nf:>4}{B:>3}{wall:>14.1f}{pre:>12.1f}{vit:>10.1f}{proj:>10.1f}{(wall*B-pre):>14.1f}")
    print("\n  pre = generate-start -> vision (mm preprocessing, REAL vt_reuse skips this too).")
    print("  prefill≈ wall*B - pre - vit - proj.  REAL vt front ≈ prefill (+h2d); encode_saving = pre + vit + proj.")


if __name__ == "__main__":
    main()
