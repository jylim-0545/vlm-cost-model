"""REAL vLLM TTFT — cold vs pre-projector (encoder) reuse vs post-projector reuse.

Per-frame TTFT of three vision-reuse variants, ALL measured by real vLLM generate (not stage
subtraction). Same video request, same engine; only the vision step differs, switched per request
by env VLM_REUSE_MODE via a per-model monkeypatch on the encoder->projector path:
  cold = original                       -> encoder(ViT) + projector + prefill   (full recompute)
  pre  = skip ViT, random ViT feats     -> projector + prefill   (reuse ENCODER output)
  post = random post-projector embeds   -> prefill only          (reuse PROJECTOR output)

Works because the engine runs IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the main-process
monkeypatch reaches the model, and mm_processor_cache_gb=0 so vision REALLY re-runs each generate.
No vLLM source edit. H100, batch=1, prefix caching OFF (prefill always runs). Qwen3-VL excluded
(DeepStack taps mid-ViT layers -> pre-projector ill-defined).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", "/mnt/nas/VLM/hf")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLM_REUSE_MODE", "cold")

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models                                  # noqa: E402

_FIRED = {"n": 0}


def _diag(mode, self, shape):
    if mode != "cold" and _FIRED["n"] < 2:
        _FIRED["n"] += 1
        print(f"[patch] FIRED mode={mode} class={type(self).__name__} in={shape}", file=sys.stderr, flush=True)


def patch_internvl():
    import torch, vllm.model_executor.models.internvl as iv
    cls = next(c for c in vars(iv).values() if isinstance(c, type) and "extract_feature" in c.__dict__)
    orig = cls.extract_feature

    def patched(self, pixel_values):
        mode = os.environ.get("VLM_REUSE_MODE", "cold")
        _diag(mode, self, tuple(pixel_values.shape))
        if mode == "cold":
            return orig(self, pixel_values)
        nt = pixel_values.shape[0]
        p = next(self.mlp1.parameters()); dev, dt = p.device, p.dtype
        H = [m for m in self.mlp1.modules() if isinstance(m, torch.nn.Linear)][-1].out_features
        vc = self.vision_model.config
        npatch = (vc.image_size // vc.patch_size) ** 2
        if mode == "post":
            ntok = int(npatch * (self.downsample_ratio ** 2))
            return torch.randn(nt, ntok, H, device=dev, dtype=dt)
        v = torch.randn(nt, npatch + 1, vc.hidden_size, device=dev, dtype=dt)[:, 1:, :]
        h = w = int(v.shape[1] ** 0.5)
        v = v.reshape(v.shape[0], h, w, -1)
        v = self.pixel_shuffle(v, scale_factor=self.downsample_ratio)
        v = v.reshape(v.shape[0], -1, v.shape[-1])
        return self.mlp1(v)
    cls.extract_feature = patched
    print(f"[patch] internvl: {cls.__name__}.extract_feature", file=sys.stderr, flush=True)


def patch_llava():
    import torch, math, vllm.model_executor.models.llava_onevision as lo
    cls = next(c for c in vars(lo).values()
               if isinstance(c, type) and "_video_pixels_to_features" in c.__dict__)
    orig = cls._video_pixels_to_features

    def patched(self, vision_tower, pixel_values):
        mode = os.environ.get("VLM_REUSE_MODE", "cold")
        _diag(mode, self, tuple(pixel_values.shape))
        if mode == "cold":
            return orig(self, vision_tower, pixel_values)
        nf = pixel_values.shape[0]
        p = next(self.multi_modal_projector.parameters()); dev, dt = p.device, p.dtype
        vc = self.config.vision_config
        H = self.config.text_config.hidden_size
        grid = vc.image_size // vc.patch_size
        if mode == "post":
            pooled = math.ceil(grid / 2) ** 2                 # apply_pooling stride 2
            return torch.randn(nf, pooled, H, device=dev, dtype=dt)
        feats = torch.randn(nf, grid * grid, vc.hidden_size, device=dev, dtype=dt)  # ViT output
        feats = self.multi_modal_projector(feats)
        feats = self.apply_pooling(feats)
        return feats
    cls._video_pixels_to_features = patched
    print(f"[patch] llava: {cls.__name__}._video_pixels_to_features", file=sys.stderr, flush=True)


def patch_qwen25():
    import torch, vllm.model_executor.models.qwen2_5_vl as qv
    cls = next(c for c in vars(qv).values()
               if isinstance(c, type) and c.__name__ == "Qwen2_5_VisionTransformer")
    orig = cls.forward

    def patched(self, x, grid_thw, *, encoder_metadata=None):
        mode = os.environ.get("VLM_REUSE_MODE", "cold")
        _diag(mode, self, tuple(x.shape))
        if mode == "cold":
            return orig(self, x, grid_thw, encoder_metadata=encoder_metadata)
        seq_len = x.shape[0]
        smu = self.spatial_merge_unit
        dev, dt = self.device, self.dtype
        if mode == "post":                                             # skip blocks + merger
            dmodel = self.merger.mlp[-1].output_size                   # d_model (3584; RowParallelLinear)
            return torch.randn(seq_len // smu, dmodel, device=dev, dtype=dt)
        ctx = self.merger.ln_q.weight.shape[0]                          # context_dim (1280; RMSNorm)
        meta = encoder_metadata or self.prepare_encoder_metadata(grid_thw)  # reverse_indices
        hs = torch.randn(seq_len, 1, ctx, device=dev, dtype=dt)        # "cached" block (encoder) output
        hs = self.merger(hs)                                          # run merger (projector)
        return hs[meta["reverse_indices"], :]
    cls.forward = patched
    print(f"[patch] qwen2.5: {cls.__name__}.forward", file=sys.stderr, flush=True)


PATCHERS = {"internvl": patch_internvl, "llava": patch_llava, "qwen2.5": patch_qwen25}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="internvl3.5-8b")
    ap.add_argument("--video", default=os.path.expanduser("~/VLM/scratch/nextqa_videos/5396384503.mp4"))
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768)
    ap.add_argument("--csv", default="results/lmcache/preproj_vllm.csv")
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from measure.stage_timing_vllm import build_video_request, build_video_request_internvl

    name = torch.cuda.get_device_name(0)
    assert "H100" in name, f"H100 only; got {name!r}"
    print(f"[gpu] {name}")
    spec = load_models().models[a.model]
    fam = ("internvl" if spec.key.startswith("internvl") else
           "llava" if spec.key.startswith("llava") else
           "qwen2.5" if spec.key.startswith("qwen2.5") else None)
    assert fam in PATCHERS, f"no pre-projector patcher for {spec.key} (Qwen3 excluded -> use EC)"
    PATCHERS[fam]()

    common = dict(model=spec.repo_id, trust_remote_code=spec.trust_remote_code,
                  max_model_len=a.max_model_len, gpu_memory_utilization=0.85, enforce_eager=True,
                  enable_prefix_caching=False, mm_processor_cache_gb=0,
                  max_num_batched_tokens=a.max_num_batched_tokens, limit_mm_per_prompt={"video": 1})
    if fam == "internvl":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        make_req = lambda nf: build_video_request_internvl(tok, a.video, nf)
        tok_per_frame = 256
    else:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        make_req = lambda nf: build_video_request(proc, a.video, n_frames=nf, with_metadata=False)
        tok_per_frame = 196 if fam == "llava" else 0   # qwen2.5 n_vis is dynamic -> 0 (TTFT is the check)
    llm = LLM(**common)
    sp = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)

    def med_ttft(req) -> float:
        for _ in range(a.warmup):
            llm.generate([req], sp)
        xs = []
        for _ in range(a.runs):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            llm.generate([req], sp)
            torch.cuda.synchronize(); xs.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(xs)

    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(a.csv).exists()
    f = open(a.csv, "a", newline="")
    W = csv.DictWriter(f, fieldnames=["model", "video_id", "frames", "n_vis", "cold_ttft_ms",
                                      "pre_reuse_ttft_ms", "post_reuse_ttft_ms", "engine"])
    if new:
        W.writeheader()
    vid = Path(a.video).stem
    print(f"\n[{a.model}] {'fr':>4}{'n_vis':>7} | {'COLD':>9}{'pre':>9}{'post':>9} | pre−post  cold−pre(ViT)")
    for nf in a.frames:
        req, _ = make_req(nf)
        n_vis = nf * tok_per_frame
        res = {}
        for mode in ("cold", "pre", "post"):
            os.environ["VLM_REUSE_MODE"] = mode
            res[mode] = med_ttft(req)
        os.environ["VLM_REUSE_MODE"] = "cold"
        W.writerow({"model": a.model, "video_id": vid, "frames": nf, "n_vis": n_vis,
                    "cold_ttft_ms": round(res["cold"], 2), "pre_reuse_ttft_ms": round(res["pre"], 2),
                    "post_reuse_ttft_ms": round(res["post"], 2), "engine": "vllm-inproc"})
        f.flush()
        print(f"[{a.model}] {nf:>4}{n_vis:>7} | {res['cold']:>9.1f}{res['pre']:>9.1f}{res['post']:>9.1f} | "
              f"{res['pre']-res['post']:>+7.1f}  {res['cold']-res['pre']:>+10.1f}")
    f.close()
    print(f"\n[done] {a.csv}")


if __name__ == "__main__":
    main()
