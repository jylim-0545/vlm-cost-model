"""Layer 2 — reuse cost: cold vs warm via vLLM prefix caching (CLAUDE.md Section 5/7).

Measures the empirical saving of REUSING stored inference state instead of
recomputing. vLLM V1 caches both the multimodal encoder output (mm cache) and the
prefill KV (prefix cache), so a 2nd identical (video+query) request skips BOTH
encode and prefill — exactly the "KV reuse" price model (skip encode+prefill, pay
decode only). cold == the no-reuse baseline (recompute every query).

  ttft_cold = encode + prefill        (reset_prefix_cache + reset_mm_cache each run)
  ttft_warm = ~0 (cache hit)          (run once to populate, then repeat)
  reuse_saving = median(cold) - median(warm)   # what KV reuse saves per query

enforce_eager + batch=1 isolated for clean latencies; median over >=5 runs,
warmups discarded. CUDA_VISIBLE_DEVICES=1. Long-format CSV (row per model,video,
stage,run_idx) -> OUTPUT_DIR/reuse_timing.csv.

NOTE: vLLM bundles token-reuse and KV-reuse (prefix cache stores KV). Splitting
token-only reuse (skip encode, still prefill) needs transformers injection
(inputs_embeds) — Section 5 says optional, not needed for the first break-even.

Usage:
  python -m measure.reuse_timing --model qwen2.5-vl-7b --num-images 1 2 4
  python -m measure.reuse_timing --model qwen2.5-vl-7b --videos-csv results/nextqa_sample.csv --fps 2.0
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models  # noqa: E402
from measure.stage_timing_vllm import build_image_request, build_video_request, mm_token_ids  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-images", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--videos-csv")
    ap.add_argument("--fps", type=float, default=2.0, help="Qwen-native fps frame sampling")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-model-len", type=int, default=49152)
    a = ap.parse_args()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1", "must run on the H100 (GPU 1)"
    spec = load_models().models[a.model]

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    mm_ids = mm_token_ids(spec.repo_id, spec.trust_remote_code)

    requests = []
    if a.videos_csv:
        with open(os.path.expanduser(a.videos_csv)) as f:
            for row in csv.DictReader(f):
                requests.append(build_video_request(processor, row["path"], a.fps)
                                + (float(row.get("duration_s") or 0),))
    else:
        for k in a.num_images:
            requests.append(build_image_request(processor, k, a.resolution) + (None,))

    print(f"[reuse] loading {spec.repo_id} (prefix_caching=ON) ...")
    llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code,
              max_model_len=a.max_model_len, gpu_memory_utilization=0.85,
              enforce_eager=True, enable_prefix_caching=True,
              limit_mm_per_prompt={"image": 8, "video": 1})
    sp = SamplingParams(temperature=0.0, min_tokens=1, max_tokens=1, ignore_eos=True, detokenize=False)

    def gen(req):
        t = time.perf_counter()
        o = llm.generate(req, sp, use_tqdm=False)[0]
        return time.perf_counter() - t, o

    # engine warmup (kernel compile) on a throwaway request
    warm_req, _ = build_image_request(processor, 1, a.resolution)
    for _ in range(a.warmup):
        gen(warm_req)
        llm.reset_prefix_cache(); llm.reset_mm_cache()

    out_path = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results"))) / "reuse_timing.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    fcsv = open(out_path, "a", newline="")
    fields = ["model", "video_id", "stage", "run_idx", "value_s", "n_vision_tokens",
              "prompt_tokens", "duration_s", "timestamp"]
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    if new:
        writer.writeheader()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    for req, vid, dur in requests:
        # COLD: clear both caches each run -> always recompute encode+prefill
        cold = []
        last = None
        for _ in range(a.runs):
            llm.reset_prefix_cache(); llm.reset_mm_cache()
            dt, last = gen(req)
            cold.append(dt)
        n_prompt = len(last.prompt_token_ids)
        n_vis = sum(1 for t in last.prompt_token_ids if t in mm_ids) if mm_ids else -1

        # WARM: populate once, then repeat -> prefix+mm cache hits (encode+prefill skipped)
        gen(req)
        warm = [gen(req)[0] for _ in range(a.runs)]

        for stage, vals in (("ttft_cold", cold), ("ttft_warm", warm)):
            for i, v in enumerate(vals):
                writer.writerow({"model": spec.key, "video_id": vid, "stage": stage,
                                 "run_idx": i, "value_s": round(v, 6), "n_vision_tokens": n_vis,
                                 "prompt_tokens": n_prompt, "duration_s": dur, "timestamp": ts})
        mc, mw = statistics.median(cold), statistics.median(warm)
        print(f"[reuse] {vid}: n_vis={n_vis} prompt={n_prompt}  "
              f"cold={mc*1e3:.1f}ms warm={mw*1e3:.1f}ms  "
              f"saving={ (mc-mw)*1e3:.1f}ms ({(1-mw/mc)*100:.0f}% of cold)")

    fcsv.close()
    print(f"[reuse] wrote rows -> {out_path}")


if __name__ == "__main__":
    main()
