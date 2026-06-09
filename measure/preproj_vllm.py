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
_PIN = {"buf": None}


def _h2d(shape, dtype, dev):
    """Simulate loading the cached vision tensor from DRAM -> GPU: a pinned-CPU buffer (grown once
    per process to the max shape; realloc absorbed by warmup) transferred to GPU each call. The
    blocking .to(dev) IS the REAL DRAM->GPU H2D, included in the timed generate — so vt_reuse TTFT
    accounts for retrieval just like LMCache kv/EC (which also load DRAM->GPU). bytes = pre: encoder
    output (pre-projector); post: post-projector tokens."""
    import torch
    n = 1
    for s in shape:
        n *= int(s)
    b = _PIN["buf"]
    if b is None or b.numel() < n or b.dtype != dtype:
        _PIN["buf"] = torch.randn(n, dtype=dtype, pin_memory=True); b = _PIN["buf"]
    return b[:n].view(*shape).to(dev)   # blocking H2D (serial, like LMCache load)


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
            return _h2d((nt, ntok, H), dt, dev)                         # load post-projector tokens
        v = _h2d((nt, npatch + 1, vc.hidden_size), dt, dev)[:, 1:, :]   # load ViT (encoder) output
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
            return _h2d((nf, pooled, H), dt, dev)             # load post-projector tokens
        feats = _h2d((nf, grid * grid, vc.hidden_size), dt, dev)  # load ViT (encoder) output
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
            return _h2d((seq_len // smu, dmodel), dt, dev)             # load post-projector tokens
        ctx = self.merger.ln_q.weight.shape[0]                          # context_dim (1280; RMSNorm)
        meta = encoder_metadata or self.prepare_encoder_metadata(grid_thw)  # reverse_indices
        hs = _h2d((seq_len, 1, ctx), dt, dev)                          # load block (encoder) output
        hs = self.merger(hs)                                          # run merger (projector)
        return hs[meta["reverse_indices"], :]
    cls.forward = patched
    print(f"[patch] qwen2.5: {cls.__name__}.forward", file=sys.stderr, flush=True)


PATCHERS = {"internvl": patch_internvl, "llava": patch_llava, "qwen2.5": patch_qwen25}


def load_videos(path):
    import csv as _csv
    with open(path) as f:
        return list(_csv.DictReader(f))


def video_token_id(spec, fam):
    """The placeholder token id counted for n_vis (per family)."""
    from transformers import AutoTokenizer, AutoConfig
    if fam == "internvl":
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        return tok.convert_tokens_to_ids("<|video_pad|>")
    cfg = AutoConfig.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
    return getattr(cfg, "video_token_id", None) or getattr(cfg, "video_token_index", None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="internvl3.5-8b")
    ap.add_argument("--videos-csv", default="final_videos.csv")
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--decode-tokens", type=int, default=256, help="full-latency decode length (cold only)")
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768)
    ap.add_argument("--video-max-patches", type=int, default=768, help="Qwen per-frame max_pixels = N*28*28")
    ap.add_argument("--video-min-patches", type=int, default=128)
    ap.add_argument("--qwen3-longest-edge", type=int, default=768 * 28 * 28 * 256)
    ap.add_argument("--csv", default="results/final/preproj_vllm.csv")
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
           "qwen2.5" if spec.key.startswith("qwen2.5") else
           "qwen3" if spec.key.startswith("qwen3") else None)
    assert fam, f"unknown family for {spec.key}"
    # Qwen3 has no pre-projector patcher (DeepStack taps mid-ViT) -> measure COLD ONLY here
    # (full recompute, no patcher needed); its vt_reuse is EC via reuse_lmcache --mode ec.
    cold_only = fam not in PATCHERS
    if not cold_only:
        PATCHERS[fam]()
    vtid = video_token_id(spec, fam)
    needs_meta = fam == "qwen3"

    mm_kwargs = None
    if fam == "qwen2.5":
        mm_kwargs = {"max_pixels": a.video_max_patches * 28 * 28, "min_pixels": a.video_min_patches * 28 * 28}
    elif fam == "qwen3":
        mm_kwargs = {"size": {"longest_edge": a.qwen3_longest_edge, "shortest_edge": 4096}}
    common = dict(model=spec.repo_id, trust_remote_code=spec.trust_remote_code,
                  max_model_len=a.max_model_len, gpu_memory_utilization=0.85, enforce_eager=True,
                  enable_prefix_caching=False, mm_processor_cache_gb=0, mm_processor_kwargs=mm_kwargs,
                  max_num_seqs=max(a.batches), max_num_batched_tokens=a.max_num_batched_tokens,
                  limit_mm_per_prompt={"video": 1})
    if fam == "internvl":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        make_req = lambda path, nf: build_video_request_internvl(tok, path, nf)[0]
    else:
        from transformers import AutoProcessor
        vpx = {"max_pixels": a.video_max_patches * 28 * 28,
               "min_pixels": a.video_min_patches * 28 * 28} if fam == "qwen2.5" else {}
        proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code, **vpx)
        if needs_meta and hasattr(proc, "video_processor"):
            proc.video_processor.size.longest_edge = a.qwen3_longest_edge
        make_req = lambda path, nf: build_video_request(proc, path, n_frames=nf, with_metadata=needs_meta)[0]
    llm = LLM(**common)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    spD = SamplingParams(temperature=0.0, max_tokens=a.decode_tokens, detokenize=False)

    def med(reqs, sp):
        """median per-request wall (ms) over runs; reqs = list of B requests submitted together."""
        B = len(reqs)
        for _ in range(a.warmup):
            llm.generate(reqs, sp)
        xs = []
        for _ in range(a.runs):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = llm.generate(reqs, sp)
            torch.cuda.synchronize(); xs.append((time.perf_counter() - t0) * 1e3 / B)
        return statistics.median(xs), out

    videos = load_videos(a.videos_csv)
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(a.csv).exists()
    f = open(a.csv, "a", newline="")
    W = csv.DictWriter(f, fieldnames=["model", "dataset", "video_id", "res_label", "frames", "batch",
                                      "n_vis", "n_prompt_tokens", "variant", "metric", "value_ms", "engine"])
    if new:
        W.writeheader()

    def emit(v, nf, B, n_vis, npt, variant, metric, val):
        W.writerow({"model": a.model, "dataset": v["dataset"], "video_id": v["video_id"],
                    "res_label": v["res_label"], "frames": nf, "batch": B, "n_vis": n_vis,
                    "n_prompt_tokens": npt, "variant": variant, "metric": metric,
                    "value_ms": round(val, 3), "engine": "vllm-inproc"}); f.flush()

    print(f"\n[{a.model}] cold/pre/post TTFT (ms), per-request | video×frame×batch")
    for v in videos:
        for nf in a.frames:
            try:
                req1 = make_req(v["path"], nf)
            except Exception as e:
                print(f"  {v['video_id']} f{nf}: build FAIL {type(e).__name__}"); continue
            for B in a.batches:
                reqs = [req1] * B
                try:
                    os.environ["VLM_REUSE_MODE"] = "cold"
                    cold_t, out = med(reqs, sp1)
                    npt = len(out[0].prompt_token_ids)
                    n_vis = sum(1 for t in out[0].prompt_token_ids if t == vtid) if vtid else 0
                    cold_full, _ = med(reqs, spD)
                    pre_t = post_t = None
                    if not cold_only:
                        os.environ["VLM_REUSE_MODE"] = "pre"; pre_t, _ = med(reqs, sp1)
                        os.environ["VLM_REUSE_MODE"] = "post"; post_t, _ = med(reqs, sp1)
                    os.environ["VLM_REUSE_MODE"] = "cold"
                except Exception as e:
                    print(f"  {v['video_id']} f{nf} b{B}: SKIP {type(e).__name__}: {str(e)[:80]}")
                    os.environ["VLM_REUSE_MODE"] = "cold"; continue
                emit(v, nf, B, n_vis, npt, "cold", "ttft", cold_t)
                emit(v, nf, B, n_vis, npt, "cold", "full", cold_full)
                if not cold_only:
                    emit(v, nf, B, n_vis, npt, "vt_pre", "ttft", pre_t)
                    emit(v, nf, B, n_vis, npt, "vt_post", "ttft", post_t)
                ps = f"pre {pre_t:>8.1f} post {post_t:>8.1f}" if not cold_only else "(cold-only)"
                print(f"  {v['video_id']:>14} f{nf:>3} b{B:>2} n_vis={n_vis:>6} | "
                      f"cold {cold_t:>8.1f} {ps} | full {cold_full:>8.1f}")
    f.close()
    print(f"\n[done] {a.csv}")


if __name__ == "__main__":
    main()
