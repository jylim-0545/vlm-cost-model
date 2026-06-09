"""LMCache reuse measurement — REAL kv_reuse (KV offload+reload) and EC vt_reuse (Qwen3).

Per (model, video, frames, batch) on a storage TIER:
  - kv (--mode lmcache): LMCache KV connector. enable_prefix_caching=False forces a REAL tier
    LOAD on the warm request (else GPU-resident hit). warm = kv_reuse (skip encode+prefill).
  - ec (--mode ec): LMCache Encoder Cache (post-projector vt_reuse) — used for Qwen3 (DeepStack
    makes pre-projector ill-defined). warm = vt_reuse (skip the vision tower). NOTE: do NOT set
    mm_processor_cache_gb=0 for EC — that yields positional mm_hash and never hits.

cold (full recompute) is NOT measured here — it is the canonical `reuse_real.py` cold pass. We
record a within-engine `cold_ref` (the store/miss pass) only for sanity.

Runs in `vlmcost` (vLLM 0.22 + lmcache 0.4.6; c_ops loads). H100 ONLY (LMCache c_ops has no
Blackwell kernel). EC connector lives in measure/lmcache_ec_connector.py -> need PYTHONPATH=repo.
Writes results/final/ (isolated). ONE tier per process (LMCACHE_* read at engine init).
"""
from __future__ import annotations
import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_models                                    # noqa: E402


def configure_tier(tier: str, disk_path: str) -> dict:
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    if tier == "dram":
        cfg = {"LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": "40.0"}
    elif tier == "disk":
        cfg = {"LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": "0.5",
               "LMCACHE_USE_GDS": "False",   # GDS/cufile hangs on this FS; plain disk I/O
               "LMCACHE_LOCAL_DISK": f"file://{disk_path}", "LMCACHE_MAX_LOCAL_DISK_SIZE": "120.0"}
        Path(disk_path).mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"unknown tier {tier!r}")
    os.environ.update(cfg)
    return cfg


def load_videos(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llava-ov-7b", help="model KEY (config/models.yaml)")
    ap.add_argument("--videos-csv", default="final_videos.csv")
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--mode", choices=["lmcache", "ec"], default="lmcache")
    ap.add_argument("--tier", choices=["dram", "disk"], required=True)
    ap.add_argument("--disk-path", default=os.path.expanduser("~/VLM/scratch/lmcache_disk"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--max-num-batched-tokens", type=int, default=32768)
    ap.add_argument("--video-max-patches", type=int, default=768)
    ap.add_argument("--video-min-patches", type=int, default=128)
    ap.add_argument("--qwen3-longest-edge", type=int, default=768 * 28 * 28 * 256)
    ap.add_argument("--csv", default="results/final/reuse_lmcache.csv")
    a = ap.parse_args()

    spec = load_models().models[a.model]
    fam = ("internvl" if spec.key.startswith("internvl") else
           "llava" if spec.key.startswith("llava") else
           "qwen2.5" if spec.key.startswith("qwen2.5") else
           "qwen3" if spec.key.startswith("qwen3") else None)
    assert fam, f"unknown family for {spec.key}"
    # EC mode keeps content-hash mm cache ON (positional hash never hits); KV mode forces tier load.
    cfg = configure_tier(a.tier, a.disk_path)

    import torch
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig, ECTransferConfig
    from transformers import AutoTokenizer, AutoProcessor, AutoConfig
    from measure.stage_timing_vllm import build_video_request, build_video_request_internvl

    name = torch.cuda.get_device_name(0)
    assert "H100" in name, f"LMCache is H100-only (c_ops no sm_120 kernel); got {name!r}"
    print(f"[gpu] {name}  [mode] {a.mode} [tier] {a.tier} [cfg] {cfg}")

    # ---- per-family request builder + n_vis token id + engine mm kwargs ----
    needs_meta = fam == "qwen3"
    is_qwen = fam in ("qwen2.5", "qwen3")
    mm_kwargs = None
    if fam == "qwen2.5":
        mm_kwargs = {"max_pixels": a.video_max_patches * 28 * 28, "min_pixels": a.video_min_patches * 28 * 28}
    elif fam == "qwen3":
        mm_kwargs = {"size": {"longest_edge": a.qwen3_longest_edge, "shortest_edge": 4096}}

    if fam == "internvl":
        tok = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True)
        vtid = tok.convert_tokens_to_ids("<|video_pad|>")
        make_req = lambda path, nf: build_video_request_internvl(tok, path, nf)[0]
    else:
        vpx = {"max_pixels": a.video_max_patches * 28 * 28,
               "min_pixels": a.video_min_patches * 28 * 28} if fam == "qwen2.5" else {}
        proc = AutoProcessor.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code, **vpx)
        if needs_meta and hasattr(proc, "video_processor"):
            proc.video_processor.size.longest_edge = a.qwen3_longest_edge
        cfg_ = AutoConfig.from_pretrained(spec.repo_id, trust_remote_code=spec.trust_remote_code)
        vtid = getattr(cfg_, "video_token_id", None) or getattr(cfg_, "video_token_index", None)
        make_req = lambda path, nf: build_video_request(proc, path, n_frames=nf, with_metadata=needs_meta)[0]

    mml = 32768 if fam == "llava" else a.max_model_len
    common = dict(model=spec.repo_id, trust_remote_code=spec.trust_remote_code, max_model_len=mml,
                  gpu_memory_utilization=0.85, enforce_eager=True, mm_processor_kwargs=mm_kwargs,
                  max_num_seqs=max(a.batches), max_num_batched_tokens=a.max_num_batched_tokens,
                  limit_mm_per_prompt={"video": 1})
    if a.mode == "lmcache":
        ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both",
                               kv_load_failure_policy="recompute")
        llm = LLM(enable_prefix_caching=False, kv_transfer_config=ktc, **common)
        warm_variant = "kv_reuse"
    else:  # ec
        etc = ECTransferConfig(ec_connector="LMCacheECConnector", ec_role="ec_both",
                               ec_connector_module_path="measure.lmcache_ec_connector")
        llm = LLM(enable_prefix_caching=False, ec_transfer_config=etc, **common)
        warm_variant = "vt_reuse"
    sp = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    import vllm as _vllm
    lmver = __import__("lmcache").__version__

    def med(reqs):
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
                                      "n_vis", "n_prompt_tokens", "tier", "variant", "metric",
                                      "value_ms", "engine", "lmcache_ver"])
    if new:
        W.writeheader()

    def emit(v, nf, B, n_vis, npt, variant, val):
        W.writerow({"model": a.model, "dataset": v["dataset"], "video_id": v["video_id"],
                    "res_label": v["res_label"], "frames": nf, "batch": B, "n_vis": n_vis,
                    "n_prompt_tokens": npt, "tier": a.tier, "variant": variant, "metric": "ttft",
                    "value_ms": round(val, 3), "engine": f"vllm-{_vllm.__version__}",
                    "lmcache_ver": lmver}); f.flush()

    print(f"\n[{a.model}/{a.mode}/{a.tier}] warm {warm_variant} TTFT (ms), per-request")
    for v in videos:
        for nf in a.frames:
            try:
                req = make_req(v["path"], nf)
            except Exception as e:
                print(f"  {v['video_id']} f{nf}: build FAIL {type(e).__name__}"); continue
            try:
                _, out = med([req])                        # store/warm pass (populates tier + n_vis)
            except Exception as e:
                print(f"  {v['video_id']} f{nf}: store FAIL {type(e).__name__}: {str(e)[:70]}"); continue
            npt = len(out[0].prompt_token_ids)
            n_vis = sum(1 for t in out[0].prompt_token_ids if t == vtid) if vtid else 0
            for B in a.batches:                            # warm: all B hit/load from tier (real reuse)
                try:
                    warm, _ = med([req] * B)
                except Exception as e:
                    print(f"  {v['video_id']} f{nf} b{B}: SKIP {type(e).__name__}: {str(e)[:60]}"); continue
                emit(v, nf, B, n_vis, npt, warm_variant, warm)
                print(f"  {v['video_id']:>14} f{nf:>3} b{B:>2} n_vis={n_vis:>6} | "
                      f"{warm_variant} {warm:>7.1f} ms/req")
    f.close()
    print(f"\n[done] {a.csv}")


if __name__ == "__main__":
    main()
