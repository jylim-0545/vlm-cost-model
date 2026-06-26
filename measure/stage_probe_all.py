"""Unified GPU-stage probe (4 models): pre (generate->vision) / vis (encode=vision tower) / prefill /
TPOT, per (frame,batch) on one pinned video. mm_cache=0 (no dedup; verified via patch count == B-scaled).
prefill = ttft - pre - vis. encode(vis) = whole vision tower (ViT+projector/merger/deepstack). Used to
correct vt_reuse (= prefill only) and cold (= GPU dec+prep + vis + prefill). H2D handled byte-based in TCO.
In-process, eager. Prints a table + ROWS."""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1"); os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import argparse, csv, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models

STAGE = {"vis": [], "npatch": []}; GEN = {"t0": None, "pre": []}

HOOK = {  # family -> (module path, class name, method, pixel-arg index)
    "internvl": ("vllm.model_executor.models.internvl", "InternVLChatModel", "extract_feature", 0),
    "llava": ("vllm.model_executor.models.llava_onevision", "LlavaOnevisionForConditionalGeneration", "_video_pixels_to_features", 1),
    "qwen2.5": ("vllm.model_executor.models.qwen2_5_vl", "Qwen2_5_VisionTransformer", "forward", 0),
    "qwen3": ("vllm.model_executor.models.qwen3_vl", "Qwen3_VisionTransformer", "forward", 0),
}


def install_hook(fam):
    import importlib, torch
    modpath, clsname, method, argi = HOOK[fam]
    cls = getattr(importlib.import_module(modpath), clsname)
    orig = getattr(cls, method)
    def patched(self, *a, **k):
        if GEN["t0"] is not None:
            GEN["pre"].append((time.perf_counter() - GEN["t0"]) * 1e3); GEN["t0"] = None
        try: STAGE["npatch"].append(int(a[argi].shape[0]))
        except Exception: pass
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = orig(self, *a, **k); e.record(); torch.cuda.synchronize()
        STAGE["vis"].append(s.elapsed_time(e)); return r
    setattr(cls, method, patched)
    print(f"[probe] hooked {clsname}.{method}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="internvl3.5-8b")
    ap.add_argument("--video-id", default="5396384503"); ap.add_argument("--videos-csv", default="final_videos_pin.csv")
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--decode-tokens", type=int, default=32); ap.add_argument("--runs", type=int, default=2); ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=40960); ap.add_argument("--mnbt", type=int, default=32768)
    ap.add_argument("--vmax", type=int, default=768); ap.add_argument("--vmin", type=int, default=128)
    ap.add_argument("--q3-longest-edge", type=int, default=768*28*28*256)
    ap.add_argument("--csv", default="results/final/stage_probe_all.csv")
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from measure.stage_timing_vllm import build_video_request, build_video_request_internvl
    assert "H100" in torch.cuda.get_device_name(0)
    spec = load_models().models[a.model]
    fam = ("internvl" if spec.key.startswith("internvl") else "llava" if spec.key.startswith("llava")
           else "qwen2.5" if spec.key.startswith("qwen2.5") else "qwen3" if spec.key.startswith("qwen3") else None)
    assert fam, spec.key
    install_hook(fam)
    needs_meta = fam == "qwen3"
    mmk = None
    if fam == "qwen2.5": mmk = {"max_pixels": a.vmax*28*28, "min_pixels": a.vmin*28*28}
    elif fam == "qwen3": mmk = {"size": {"longest_edge": a.q3_longest_edge, "shortest_edge": 4096}}
    mml = 32768 if fam == "llava" else a.max_model_len
    llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code, max_model_len=mml,
              gpu_memory_utilization=0.85, enforce_eager=True, enable_prefix_caching=False,
              mm_processor_cache_gb=0, mm_processor_kwargs=mmk, max_num_seqs=max(a.batches),
              max_num_batched_tokens=a.mnbt, limit_mm_per_prompt={"video": 1})
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    spD = SamplingParams(temperature=0.0, max_tokens=a.decode_tokens, detokenize=False)
    if fam == "internvl":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        vtid = tok.convert_tokens_to_ids("<|video_pad|>")
        make = lambda p, nf: build_video_request_internvl(tok, p, nf)[0]
    else:
        from transformers import AutoProcessor, AutoConfig
        vpx = {"max_pixels": a.vmax*28*28, "min_pixels": a.vmin*28*28} if fam == "qwen2.5" else {}
        proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code, **vpx)
        if needs_meta and hasattr(proc, "video_processor"): proc.video_processor.size.longest_edge = a.q3_longest_edge
        cfg = AutoConfig.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        vtid = getattr(cfg, "video_token_id", None) or getattr(cfg, "video_token_index", None)
        make = lambda p, nf: build_video_request(proc, p, n_frames=nf, with_metadata=needs_meta)[0]
    row = next(r for r in csv.DictReader(open(a.videos_csv)) if r["video_id"] == a.video_id)

    def run(reqs, sp, stage):
        B = len(reqs)
        for _ in range(a.warmup): llm.generate(reqs, sp)
        walls, vis, pre, npat = [], [], [], []
        out = None
        for _ in range(a.runs):
            STAGE["vis"].clear(); STAGE["npatch"].clear(); GEN["pre"].clear()
            torch.cuda.synchronize(); t0 = time.perf_counter(); GEN["t0"] = t0
            out = llm.generate(reqs, sp); torch.cuda.synchronize(); walls.append((time.perf_counter()-t0)*1e3/B)
            if stage:
                vis.append(sum(STAGE["vis"])/B); pre.append(GEN["pre"][0]/B if GEN["pre"] else 0); npat.append(sum(STAGE["npatch"]))
        return (statistics.median(walls), statistics.median(vis) if vis else 0,
                statistics.median(pre) if pre else 0, statistics.median(npat) if npat else 0, out)

    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(a.csv).exists(); f = open(a.csv, "a", newline=""); W = csv.writer(f)
    if new: W.writerow(["model", "video_id", "frames", "batch", "n_vis", "pre_ms", "encode_ms", "prefill_ms", "ttft_ms", "tpot_ms", "patches_perB"])
    print(f"\n[{a.model}] {'f':>4}{'b':>3}{'n_vis':>7}{'pre':>7}{'encode':>8}{'prefill':>9}{'ttft':>8}{'TPOT':>7}{'pat/B':>9}")
    for nf in a.frames:
        try: req = make(row["path"], nf)
        except Exception as e: print(f"  f{nf} build FAIL {type(e).__name__}:{str(e)[:60]}"); continue
        for B in a.batches:
            reqs = [req]*B
            try:
                ttft, vis, pre, npat, out = run(reqs, sp1, True)
                full, *_ = run(reqs, spD, False)
            except Exception as e:
                print(f"  f{nf} b{B} SKIP {type(e).__name__}:{str(e)[:70]}"); continue
            nvis = sum(1 for t in out[0].prompt_token_ids if t == vtid)
            prefill = ttft - pre - vis; tpot = (full - ttft)/a.decode_tokens; patB = npat/B
            print(f"     {nf:>4}{B:>3}{nvis:>7}{pre:>7.1f}{vis:>8.1f}{prefill:>9.1f}{ttft:>8.1f}{tpot:>7.2f}{patB:>9.0f}")
            W.writerow([a.model, a.video_id, nf, B, nvis, round(pre,1), round(vis,1), round(prefill,1), round(ttft,1), round(tpot,2), round(patB)]); f.flush()
    f.close(); print(f"[done] {a.csv}")


if __name__ == "__main__":
    main()
