"""Feasibility sweep: max feasible n_frame per (model, batch) on H100, MEASURED.
One subprocess per (model, batch) so an OOM (EngineDead) is isolated. Frames ascend near
the analytical frontier; last OK before OOM/context-overflow = max. Appends results/feasibility.csv.
Usage: python -m measure.feasibility <model> <batch>   (CUDA_VISIBLE_DEVICES=1)
"""
import os, sys, csv, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/home/jylim/VLM")
from config import load_models
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams
from measure.stage_timing_vllm import build_video_request, build_video_request_internvl

MODEL, BATCH = sys.argv[1], int(sys.argv[2])
# Capped 1280x720 video (real ~360 tok/frame). n_vis ceiling is resolution-INDEPENDENT
# (verified: encode/prefill/KV all ~prop to n_vis), so one video gives max_n_vis; per-resolution
# max_frames = max_n_vis / tok_per_frame(res). See memory: qwen-encode-nvis-sufficient.
VID = "results/qwen_res_encode/vid_1280x720.csv"
GPU_GB, UTIL = 80.0, 0.92
PARAMS = {"qwen2.5-vl-7b": 8.3, "qwen3-vl-8b": 8.8, "internvl3.5-4b": 4.7,
          "internvl3.5-8b": 8.4, "internvl3.5-14b": 15.1, "llava-ov-7b": 8.0}
CTX = {"qwen2.5-vl-7b": 128000, "qwen3-vl-8b": 262144, "internvl3.5-4b": 40960,
       "internvl3.5-8b": 40960, "internvl3.5-14b": 40960, "llava-ov-7b": 32768}
TPF = {"qwen2.5-vl-7b": 360, "qwen3-vl-8b": 360, "internvl3.5-4b": 256,
       "internvl3.5-8b": 256, "internvl3.5-14b": 256, "llava-ov-7b": 196}

cfg = load_models(); spec = cfg.models[MODEL]
is_internvl = MODEL.startswith("internvl"); is_llava = MODEL.startswith("llava"); needs_meta = MODEL.startswith("qwen3")
kv_kb = 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * 2 / 1024
# CAP the serving context at reuse_real's 40960 (Qwen's native 128k/262k would force an absurd
# max_model_len -> KV/batched-token alloc fails at engine init, AND we never serve at that length).
# Frontier is thus "max feasible n_vis within a 40960-token serving context" (our measurement config).
MML_CAP = 40960
ctx, tpf = min(CTX[MODEL], MML_CAP), TPF[MODEL]
kvcap = GPU_GB * UTIL - PARAMS[MODEL] * 2
C = min((ctx - 512) // tpf, int(kvcap * 1e6 / (BATCH * tpf * kv_kb)))   # analytical frontier (UNDER-est:
# ignores prefill activation, so real OOM is often <C; span up to 1.5C to cross the true boundary)
cands = sorted({16, 32, 64, 128} | {int(C * r) for r in
                (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5)})
cands = [f for f in cands if f * tpf <= ctx and f >= 8]
mml = min(ctx, max(cands) * tpf + 800)
print(f"[feas] {MODEL} b{BATCH}: analytical C={C}, cands={cands}, max_model_len={mml}", flush=True)

vpx = {} if (is_internvl or is_llava) else {"max_pixels": 768 * 28 * 28, "min_pixels": 128 * 28 * 28}
mm_kw = None if (is_internvl or is_llava) else {"max_pixels": 768 * 28 * 28, "min_pixels": 128 * 28 * 28}
if is_internvl:
    tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
    mk = lambda nf: build_video_request_internvl(tok, PATH, nf, query=Q)[0]
else:
    proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code, **vpx)
    mk = lambda nf: build_video_request(proc, PATH, n_frames=nf, with_metadata=needs_meta, query=Q)[0]
# real video-placeholder token id (count actual n_vis from the prompt, NOT nf*tpf estimate)
_acfg = AutoConfig.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
vtid = tok.convert_tokens_to_ids("<|video_pad|>") if is_internvl else (
    getattr(_acfg, "video_token_id", None) or getattr(_acfg, "video_token_index", None))
row = next(csv.DictReader(open(VID))); PATH = row["path"]; Q = (row.get("question") or "Describe.").strip()

llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code, max_model_len=mml,
          gpu_memory_utilization=UTIL, enforce_eager=True, max_num_seqs=BATCH,
          max_num_batched_tokens=mml, mm_processor_kwargs=mm_kw, enable_prefix_caching=False,
          mm_processor_cache_gb=0, limit_mm_per_prompt={"video": 1, "image": 1})
sp1 = SamplingParams(temperature=0.0, min_tokens=1, max_tokens=1, ignore_eos=True, detokenize=False)
out = open("results/feasibility.csv", "a", newline=""); W = csv.writer(out)
if out.tell() == 0: W.writerow(["model", "batch", "frame", "n_vis", "status", "detail"])

last_ok = 0; last_nv = 0
for nf in cands:
    try:
        req = mk(nf); reqs = [req] * BATCH
        t = time.perf_counter(); o = llm.generate(reqs, sp1, use_tqdm=False); dt = (time.perf_counter() - t) * 1e3
        nv = sum(1 for x in o[0].prompt_token_ids if x == vtid)   # REAL n_vis (all models)
        W.writerow([MODEL, BATCH, nf, nv, "OK", f"{dt:.0f}ms"]); out.flush()
        last_ok = nf; last_nv = nv; print(f"[feas]   {nf}f OK n_vis={nv} ({dt:.0f}ms)", flush=True)
    except Exception as e:
        W.writerow([MODEL, BATCH, nf, nf * tpf, "FAIL", f"{type(e).__name__}:{str(e)[:50]}"]); out.flush()
        print(f"[feas]   {nf}f FAIL {type(e).__name__}: {str(e)[:60]}", flush=True)
        break
print(f"[feas] MAX {MODEL} b{BATCH} = {last_ok}f (n_vis={last_nv})", flush=True)
