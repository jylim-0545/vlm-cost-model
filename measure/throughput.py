"""Layer 2 — throughput / batching via vLLM (CLAUDE.md Section 5).

Batch-size sweep to bound the price model: per-output-token cost at batch=1
(latency-optimal) vs batch=max (throughput-optimal). vLLM continuous-batches B
distinct requests submitted together. Prefix caching OFF + mm cache OFF so each
request does real work (no cross-request reuse inflating throughput).

Uses a REAL video input (first row of --videos-csv at --frames) so the per-token
cost reflects the actual serving regime, and reuses the per-model video builders
from stage_timing_vllm (Qwen2.5 plain / Qwen3 metadata / InternVL <video>). Forced
fixed decode. Stores RAW per (model, batch, iter): wall_s, out_tokens, tok/s,
ms/out-token, req/s — so any throughput metric is derivable post-hoc.

Usage:
  python -m measure.throughput --model qwen2.5-vl-7b --videos-csv results/nextqa_sample.csv \
      --frames 16 --batches 1 4 8 16
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models  # noqa: E402
from measure.stage_timing_vllm import (build_video_request,  # noqa: E402
                                       build_video_request_internvl)


def gpu_used_gib(idx: int = 1) -> float:
    try:
        out = subprocess.run(["nvidia-smi", f"--id={idx}", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return round(int(out.stdout.strip()) / 1024, 3)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos-csv", default="results/nextqa_sample.csv")
    ap.add_argument("--frames", type=int, default=16, help="frames for the representative video input")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=40960)
    a = ap.parse_args()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1", "must run on the H100 (GPU 1)"
    spec = load_models().models[a.model]
    is_internvl = spec.key.startswith("internvl")
    max_b = max(a.batches)

    # first video in the CSV as the representative input
    with open(os.path.expanduser(a.videos_csv)) as f:
        row = next(csv.DictReader(f))
    q = (row.get("question") or "Describe in detail.").strip()
    path = row["path"]

    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams
    if is_internvl:
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        base_req, vid = build_video_request_internvl(tok, path, a.frames, query=q)
    else:
        processor = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        base_req, vid = build_video_request(processor, path, n_frames=a.frames,
                                            with_metadata=spec.key.startswith("qwen3"), query=q)

    print(f"[tput] loading {spec.repo_id} (max_num_seqs={max_b}, prefix+mm cache OFF) input={vid} ...")
    llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code,
              max_model_len=a.max_model_len, gpu_memory_utilization=0.9, max_num_seqs=max_b,
              enforce_eager=False, enable_prefix_caching=False, mm_processor_cache_gb=0,
              limit_mm_per_prompt={"image": max(8, a.frames), "video": 1})
    sp = SamplingParams(temperature=0.0, min_tokens=a.decode_tokens, max_tokens=a.decode_tokens,
                        ignore_eos=True, detokenize=False)

    for _ in range(a.warmup):                       # compile/cudagraph capture at max batch
        llm.generate([base_req] * max_b, sp, use_tqdm=False)

    out_path = Path(os.path.expanduser(os.environ.get("OUTPUT_DIR", "~/VLM/results"))) / "throughput.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    fcsv = open(out_path, "a", newline="")
    fields = ["model", "video_id", "batch", "iter", "wall_s", "out_tokens", "out_tok_per_s",
              "ms_per_out_token", "req_per_s", "decode_tokens", "frames", "peak_vram_gib", "timestamp"]
    w = csv.DictWriter(fcsv, fieldnames=fields)
    if new:
        w.writeheader()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    print(f"  {'batch':>5}{'tok/s':>10}{'ms/tok':>9}{'req/s':>8}")
    for B in sorted(a.batches):
        reqs = [base_req] * B
        try:
            sl = []
            for it in range(a.iters):
                t = time.perf_counter()
                outs = llm.generate(reqs, sp, use_tqdm=False)
                dt = time.perf_counter() - t
                ntok = sum(len(o.outputs[0].token_ids) for o in outs)
                sl.append(ntok / dt)
                w.writerow({"model": spec.key, "video_id": vid, "batch": B, "iter": it,
                            "wall_s": round(dt, 5), "out_tokens": ntok,
                            "out_tok_per_s": round(ntok / dt, 2), "ms_per_out_token": round(dt / ntok * 1e3, 4),
                            "req_per_s": round(B / dt, 4), "decode_tokens": a.decode_tokens,
                            "frames": a.frames, "peak_vram_gib": gpu_used_gib(1), "timestamp": ts})
            m = statistics.median(sl)
            print(f"  {B:>5}{m:>10.1f}{1/m*1e3:>9.3f}{B/(a.decode_tokens/m):>8.2f}")
        except Exception as e:
            print(f"  {B:>5}  FAILED ({type(e).__name__}: {str(e)[:50]}) — stop")
            break
    fcsv.close()
    print(f"[tput] wrote -> {out_path}")


if __name__ == "__main__":
    main()
