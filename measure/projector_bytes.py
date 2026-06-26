"""Measure PRE-projector (encoder output) vs POST-projector byte sizes per model — the tensor vt_reuse
stores. Hooks the vision-tower method and, inside, wraps the projector module to capture its INPUT
(=pre-projector = ViT/encoder output, what vt stores for InternVL/LLaVA/Qwen2.5) and OUTPUT (=post-
projector). Qwen3 = post-projector (EC, DeepStack) so report output. bytes = numel × 2 (bf16).
Reports bytes per post-vision-token so storage scales with n_vis. b1, in-process, eager."""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1"); os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models

CAP = {"pre_numel": None, "post_numel": None}
HOOK = {  # family -> (module, class, vision-tower method, projector attr name)
    "internvl": ("vllm.model_executor.models.internvl", "InternVLChatModel", "extract_feature", "mlp1"),
    "llava": ("vllm.model_executor.models.llava_onevision", "LlavaOnevisionForConditionalGeneration", "_video_pixels_to_features", "multi_modal_projector"),
    "qwen2.5": ("vllm.model_executor.models.qwen2_5_vl", "Qwen2_5_VisionTransformer", "forward", "merger"),
    "qwen3": ("vllm.model_executor.models.qwen3_vl", "Qwen3_VisionTransformer", "forward", "merger"),
}


def install(fam):
    import importlib
    modpath, clsname, method, projattr = HOOK[fam]
    cls = getattr(importlib.import_module(modpath), clsname)
    orig = getattr(cls, method)

    def wrapped(self, *a, **k):
        proj = getattr(self, projattr)
        if not getattr(proj, "_pb", False):
            pof = proj.forward
            def pw(*aa, **kk):
                try: CAP["pre_numel"] = int(aa[0].numel())
                except Exception: pass
                r = pof(*aa, **kk)
                try: CAP["post_numel"] = int((r[0] if isinstance(r, (tuple, list)) else r).numel())
                except Exception: pass
                return r
            proj.forward = pw; proj._pb = True
        return orig(self, *a, **k)
    setattr(cls, method, wrapped)
    print(f"[pb] hooked {clsname}.{method} + {projattr}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="internvl3.5-8b")
    ap.add_argument("--video-id", default="5396384503"); ap.add_argument("--videos-csv", default="final_videos_pin.csv")
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 128])
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--vmax", type=int, default=768); ap.add_argument("--vmin", type=int, default=128)
    ap.add_argument("--q3-longest-edge", type=int, default=768*28*28*256)
    ap.add_argument("--csv", default="results/final/projector_bytes.csv")
    a = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams
    from measure.stage_timing_vllm import build_video_request, build_video_request_internvl
    spec = load_models().models[a.model]
    fam = ("internvl" if spec.key.startswith("internvl") else "llava" if spec.key.startswith("llava")
           else "qwen2.5" if spec.key.startswith("qwen2.5") else "qwen3" if spec.key.startswith("qwen3") else None)
    install(fam)
    needs_meta = fam == "qwen3"
    mmk = ({"max_pixels": a.vmax*28*28, "min_pixels": a.vmin*28*28} if fam == "qwen2.5"
           else {"size": {"longest_edge": a.q3_longest_edge, "shortest_edge": 4096}} if fam == "qwen3" else None)
    mml = 32768 if fam == "llava" else a.max_model_len
    llm = LLM(model=spec.repo_id, trust_remote_code=spec.trust_remote_code, max_model_len=mml,
              gpu_memory_utilization=0.85, enforce_eager=True, enable_prefix_caching=False,
              mm_processor_cache_gb=0, mm_processor_kwargs=mmk, max_num_seqs=1,
              max_num_batched_tokens=32768, limit_mm_per_prompt={"video": 1})
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    if fam == "internvl":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        vtid = tok.convert_tokens_to_ids("<|video_pad|>"); make = lambda p, nf: build_video_request_internvl(tok, p, nf)[0]
    else:
        from transformers import AutoProcessor, AutoConfig
        vpx = {"max_pixels": a.vmax*28*28, "min_pixels": a.vmin*28*28} if fam == "qwen2.5" else {}
        proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code, **vpx)
        if needs_meta and hasattr(proc, "video_processor"): proc.video_processor.size.longest_edge = a.q3_longest_edge
        cfg = AutoConfig.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        vtid = getattr(cfg, "video_token_id", None) or getattr(cfg, "video_token_index", None)
        make = lambda p, nf: build_video_request(proc, p, n_frames=nf, with_metadata=needs_meta)[0]
    row = next(r for r in csv.DictReader(open(a.videos_csv)) if r["video_id"] == a.video_id)
    new = not Path(a.csv).exists(); f = open(a.csv, "a", newline=""); W = csv.writer(f)
    if new: W.writerow(["model", "family", "frames", "n_vis", "pre_numel", "post_numel", "pre_bytes", "post_bytes", "pre_B_per_token", "post_B_per_token"])
    print(f"\n[{a.model}] {'f':>4}{'n_vis':>7}{'pre_numel':>12}{'post_numel':>12}{'pre_MB':>8}{'post_MB':>8}{'preB/tok':>10}{'postB/tok':>10}")
    for nf in a.frames:
        CAP["pre_numel"] = CAP["post_numel"] = None
        out = llm.generate([make(row["path"], nf)], sp1)
        n = sum(1 for t in out[0].prompt_token_ids if t == vtid)
        pre, post = CAP["pre_numel"], CAP["post_numel"]
        pb, qb = (pre or 0)*2, (post or 0)*2
        ppt, qpt = (pb/n if n else 0), (qb/n if n else 0)
        print(f"{nf:>4}{n:>7}{pre or 0:>12}{post or 0:>12}{pb/1e6:>8.1f}{qb/1e6:>8.1f}{ppt:>10.0f}{qpt:>10.0f}")
        W.writerow([a.model, fam, nf, n, pre, post, pb, qb, round(ppt, 1), round(qpt, 1)]); f.flush()
    f.close(); print(f"[done] {a.csv}")


if __name__ == "__main__":
    main()
