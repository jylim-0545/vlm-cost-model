"""Qwen2.5 GPU-stage table: ViT / merger / prefill / TPOT per (frame,batch), one model load.
mm_processor_cache ON so repeated generates skip CPU preprocess (we only need GPU stages here; decode+
preprocess are done on GPU via NVDEC in the figure). prefill = ttft - pre - vis (pre≈sched, tiny w/ cache).
Per-request (batch wall / B). Prints a table; figures built separately."""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1"); os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import argparse, csv, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models

STAGE = {"vis": [], "merger": [], "npatch": []}; GEN = {"t0": None, "pre": []}

def patch():
    import torch, vllm.model_executor.models.qwen2_5_vl as qv
    cls = next(c for c in vars(qv).values() if isinstance(c, type) and c.__name__ == "Qwen2_5_VisionTransformer")
    orig = cls.forward
    def patched(self, *a, **k):
        if GEN["t0"] is not None:
            GEN["pre"].append((time.perf_counter() - GEN["t0"]) * 1e3); GEN["t0"] = None
        if not getattr(self.merger, "_p", False):
            mo = self.merger.forward
            def mw(*aa, **kk):
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); r = mo(*aa, **kk); e.record(); torch.cuda.synchronize(); STAGE["merger"].append(s.elapsed_time(e)); return r
            self.merger.forward = mw; self.merger._p = True
        try: STAGE["npatch"].append(int(a[0].shape[0]))   # input patches to vision tower this call
        except Exception: pass
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = orig(self, *a, **k); e.record(); torch.cuda.synchronize(); STAGE["vis"].append(s.elapsed_time(e)); return r
    cls.forward = patched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-vl-7b")
    ap.add_argument("--video-id", default="5396384503"); ap.add_argument("--videos-csv", default="final_videos_pin.csv")
    ap.add_argument("--frames", type=int, nargs="+", default=[128, 32])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--decode-tokens", type=int, default=32); ap.add_argument("--runs", type=int, default=2); ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--mm-cache", type=int, default=0); ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--vmax", type=int, default=768); ap.add_argument("--vmin", type=int, default=128)
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from measure.stage_timing_vllm import build_video_request
    spec = load_models().models[a.model]; patch()
    mmk = {"max_pixels": a.vmax*28*28, "min_pixels": a.vmin*28*28}
    proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code, **mmk)
    llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.85, enforce_eager=True, enable_prefix_caching=False,
              mm_processor_cache_gb=a.mm_cache, mm_processor_kwargs=mmk, max_num_seqs=max(a.batches),
              max_num_batched_tokens=32768, limit_mm_per_prompt={"video": 1})
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    spD = SamplingParams(temperature=0.0, max_tokens=a.decode_tokens, detokenize=False)
    row = next(r for r in csv.DictReader(open(a.videos_csv)) if r["video_id"] == a.video_id)
    def run(reqs, sp, stage):
        B = len(reqs)
        for _ in range(a.warmup): llm.generate(reqs, sp)
        walls, vis, mrg, pre, ncalls, npat = [], [], [], [], [], []
        out = None
        for _ in range(a.runs):
            STAGE["vis"].clear(); STAGE["merger"].clear(); STAGE["npatch"].clear(); GEN["pre"].clear()
            torch.cuda.synchronize(); t0 = time.perf_counter(); GEN["t0"] = t0
            out = llm.generate(reqs, sp); torch.cuda.synchronize(); walls.append((time.perf_counter()-t0)*1e3/B)
            if stage:
                vis.append(sum(STAGE["vis"])/B); mrg.append(sum(STAGE["merger"])/B); pre.append(GEN["pre"][0]/B if GEN["pre"] else 0)
                ncalls.append(len(STAGE["vis"])); npat.append(sum(STAGE["npatch"]))   # total patches encoded this generate
        return (statistics.median(walls), statistics.median(vis) if vis else 0, statistics.median(mrg) if mrg else 0,
                statistics.median(pre) if pre else 0, (statistics.median(ncalls) if ncalls else 0),
                (statistics.median(npat) if npat else 0), out)
    print(f"\n{'f':>4}{'b':>3}{'n_vis':>7}{'ViT':>8}{'merger':>8}{'prefill':>9}{'ttft':>8}{'TPOT':>8}{'vcalls':>7}{'patches':>9}{'pat/B':>8}")
    rowsout = []
    for nf in a.frames:
        req = build_video_request(proc, row["path"], n_frames=nf, with_metadata=False)[0]
        for B in a.batches:
            reqs = [req]*B
            ttft, vis, mrg, pre, ncall, npat, out = run(reqs, sp1, True)
            full, *_ = run(reqs, spD, False)
            n_vis = sum(1 for t in out[0].prompt_token_ids if t == getattr(proc, "video_token_id", 151656))
            vit = vis - mrg; prefill = ttft - pre - vis; tpot = (full - ttft)/a.decode_tokens
            print(f"{nf:>4}{B:>3}{n_vis:>7}{vit:>8.1f}{mrg:>8.1f}{prefill:>9.1f}{ttft:>8.1f}{tpot:>8.2f}{ncall:>7.0f}{npat:>9.0f}{npat/B:>8.0f}")
            rowsout.append((nf, B, n_vis, round(vit,1), round(mrg,1), round(prefill,1), round(ttft,1), round(tpot,2), int(ncall), int(npat)))
    print("\nROWS:", rowsout)

if __name__ == "__main__":
    main()
